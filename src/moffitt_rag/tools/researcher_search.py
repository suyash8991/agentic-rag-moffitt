"""
ResearcherSearchTool for the Moffitt Agentic RAG system.

This module implements a tool for searching researchers by their expertise,
interests, or background using semantic and keyword search with proper metadata filtering.
"""

import re
import json
import traceback
from typing import List, Dict, Any, Optional, Type, Union

from langchain.tools import BaseTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from ..db.hybrid_search import hybrid_search
from ..config.config import get_settings
from ..utils.logging import get_logger, log_tool_event
from .tool_utils import (
    coerce_to_dict,
    extract_name_from_url,
    extract_name_from_text,
    format_researcher_result,
    join_results
)

# Get logger for this module
logger = get_logger(__name__)


class ResearcherSearchInput(BaseModel):
    """Input model for the ResearcherSearchTool."""
    researcher_name: Optional[str] = Field(
        None,
        description="The full name of the researcher to search for. Use for specific person-related queries."
    )
    topic: Optional[str] = Field(
        None,
        description="The research topic, interest, or area of expertise to search for. Use for general subject matter queries."
    )


class ResearcherSearchTool(BaseTool):
    """
    Tool for searching researchers. Use this to find researchers by their name, expertise, interests, or background.
    The agent should provide either a researcher_name or a topic.
    """

    name: str = "ResearcherSearch"
    description: str = "Search for researchers by name or by research topic. Provide either 'researcher_name' for name searches or 'topic' for subject matter searches."
    args_schema: Type[BaseModel] = ResearcherSearchInput

    # Class-level tracking of queries and attempts (for rate limiting)
    _query_attempts = {}
    _max_attempts_per_query = 3

    def _log_tool_start(self, tool_input: Any) -> None:
        """Log the start of the tool execution."""
        logger.info("Starting ResearcherSearchTool execution")
        log_tool_event("researcher_search_start", {
            "tool_input_type": type(tool_input).__name__,
            "tool_input_length": len(str(tool_input)) if tool_input else 0
        })

    def _log_tool_complete(self, query: str, results: List[Document], formatted_results: str) -> None:
        """Log the completion of the tool execution."""
        logger.info(f"Formatted results (length: {len(formatted_results)})")
        log_tool_event("researcher_search_complete", {
            "query": query[:100],
            "result_count": len(results),
            "formatted_length": len(formatted_results)
        })

    def _log_tool_error(self, error: Exception, query: Optional[str] = None) -> None:
        """Log an error during tool execution."""
        logger.error(f"Error in researcher search: {error}")
        log_tool_event("researcher_search_error", {
            "error": str(error),
            "error_type": type(error).__name__,
            "query": query[:100] if query else "unknown"
        })

    def _parse_input(self, tool_input: Union[str, Dict], tool_call_id: Optional[str] = None) -> Union[str, Dict]:
        """
        Parse the input for LangChain compatibility.

        Args:
            tool_input: The input to parse
            tool_call_id: The tool call ID (used by LangChain)

        Returns:
            Union[str, Dict]: The parsed input in the format expected by LangChain
        """
        # This is the original LangChain method signature, so we need to
        # return the input in the format expected by LangChain
        return tool_input

    def _extract_search_params(self, tool_input: Union[str, Dict]) -> tuple:
        """
        Extract researcher_name and topic from the input.

        Args:
            tool_input: The input to parse

        Returns:
            tuple: (researcher_name, topic)
        """
        researcher_name = None
        topic = None

        if isinstance(tool_input, dict):
            researcher_name = tool_input.get("researcher_name")
            topic = tool_input.get("topic")
        elif isinstance(tool_input, str):
            try:
                data = coerce_to_dict(tool_input)
                researcher_name = data.get("researcher_name")
                topic = data.get("topic")
                if not researcher_name and not topic:
                    topic = tool_input
            except Exception:
                # If parsing fails, assume it's a topic search
                topic = tool_input

        return researcher_name, topic

    def _format_results(self, results: List[Document], query: str) -> str:
        """
        Format search results with researcher names and full content.

        Args:
            results: The search results
            query: The original query

        Returns:
            str: The formatted results
        """
        if not results:
            return f"No researchers found matching the query: {query}"

        formatted_results = []
        for doc in results:
            profile_url = doc.metadata.get("profile_url", "")

            # Get researcher name with fallbacks
            display_name = (
                doc.metadata.get("researcher_name", "").strip() or
                extract_name_from_url(profile_url) or
                extract_name_from_text(doc.page_content) or
                "Unknown Researcher"
            )

            program = doc.metadata.get("program", "Unknown Program")
            chunk_type = doc.metadata.get("chunk_type", "Unknown Type")
            full_content = doc.page_content

            # Format this result
            formatted_result = format_researcher_result(
                researcher_name=display_name,
                program=program,
                chunk_type=chunk_type,
                content=full_content,
                profile_url=profile_url
            )

            formatted_results.append(formatted_result)

        # Join all results with separators
        return join_results(formatted_results)

    def _check_rate_limit(self, query: str) -> bool:
        """
        Check if the query has exceeded the rate limit.

        Args:
            query: The search query

        Returns:
            bool: True if the query is allowed, False if it should be rate-limited
        """
        # Normalize query for comparison
        normalized_query = ' '.join(query.lower().split())

        # Get current count for this query
        current_count = self._query_attempts.get(normalized_query, 0)

        # Check if we've exceeded the limit
        if current_count >= self._max_attempts_per_query:
            logger.warning(f"Rate limit exceeded for query: {query} ({current_count} attempts)")
            return False

        # Increment the count
        self._query_attempts[normalized_query] = current_count + 1
        logger.info(f"Query attempt {current_count + 1}/{self._max_attempts_per_query} for: {query}")

        return True

    def _run(self, tool_input: Union[str, Dict]) -> str:
        """
        Run the tool with the given input.

        Args:
            tool_input: The input to the tool

        Returns:
            str: The formatted search results
        """
        self._log_tool_start(tool_input)

        try:
            # Extract search parameters
            researcher_name, topic = self._extract_search_params(tool_input)

            # Validate input
            if not researcher_name and not topic:
                logger.warning("No researcher_name or topic provided")
                return "Error: You must provide either a researcher_name or a topic to the ResearcherSearchTool."

            # Determine search type and parameters
            if researcher_name:
                logger.info(f"Performing a name search for: '{researcher_name}'")
                log_tool_event("name_search_detected", {"researcher_name": researcher_name[:50]})
                query = researcher_name
                alpha = 0.3  # Prioritize keyword matching for names
            else:  # topic must be present
                logger.info(f"Performing a topic search for: '{topic}'")
                log_tool_event("topic_search_detected", {"topic": topic[:50]})
                query = topic
                alpha = 0.7  # Prioritize semantic matching for topics

            # Check rate limit
            if not self._check_rate_limit(query):
                return (
                    f"You've made multiple attempts searching for '{query}' without success. "
                    f"Consider refining your search or trying a different approach."
                )

            # Log search parameters
            logger.info(f"Searching with query: '{query}' and alpha: {alpha}")
            log_tool_event("search_parameters", {
                "query": query[:100],
                "alpha": alpha,
                "search_type": "name" if researcher_name else "topic"
            })

            # Perform the search
            results = hybrid_search(query=query, k=5, alpha=alpha)

            # Log search completion
            logger.info(f"Search completed, found {len(results) if results else 0} results")
            log_tool_event("search_completed", {
                "query": query[:100],
                "result_count": len(results) if results else 0,
                "alpha": alpha
            })

            # Handle no results case
            if not results:
                logger.info(f"No researchers found matching: {query}")
                return f"No researchers found matching: {query}"

            # Format results
            formatted_results = self._format_results(results, query)

            # Log successful completion
            self._log_tool_complete(query, results, formatted_results)

            return formatted_results

        except Exception as e:
            # Get query value if it exists
            query = locals().get('query', None)

            # Log the error
            self._log_tool_error(e, query)

            # Add traceback for better debugging
            logger.error(f"Traceback: {traceback.format_exc()}")

            return f"Error searching for researchers: {str(e)}"

    async def _arun(self, tool_input: Union[str, Dict]) -> str:
        """
        Run the tool asynchronously.

        Args:
            tool_input: The input to the tool

        Returns:
            str: The formatted search results
        """
        return self._run(tool_input)