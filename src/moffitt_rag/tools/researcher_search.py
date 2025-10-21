"""
ResearcherSearchTool for the Moffitt Agentic RAG system.

This module implements a tool for searching researchers by their expertise,
interests, or background using semantic and keyword search with proper metadata filtering.
"""

import re
import json
import ast

from typing import List, Dict, Any, Optional, Type, Union

from langchain.tools import BaseTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from ..db.hybrid_search import hybrid_search
from ..config.config import get_settings
from ..db.vector_store import get_or_create_vector_db, similarity_search_with_score
from ..utils.logging import get_logger, log_tool_event

# Get logger for this module
logger = get_logger(__name__)


def _coerce_to_payload(maybe_str: str) -> Dict[str, Any]:
    s = (maybe_str or "").strip()

    # 1) direct JSON
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2) first {...} block
    m = re.search(r'\{.*?\}', s, flags=re.DOTALL)
    if m:
        block = m.group(0)
        for parser in (json.loads, ast.literal_eval):
            try:
                data = parser(block)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    # 3) simple "key: value" lines fallback
    kv = {}
    for line in s.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().strip('"\'')
            v = v.strip().strip('"\'')
            if k:
                kv[k] = v
    return kv

def extract_name_from_url(url: str) -> str:
    """
    Extract researcher name from profile URL.
    """
    if not url:
        return ""
    path_parts = url.rstrip('/').split('/')
    if len(path_parts) > 0:
        name_slug = path_parts[-1]
        name = name_slug.replace('-', ' ').title()
        return name
    return ""


def extract_name_from_text(text: str) -> str:
    """
    Extract researcher name from text content.
    """
    name_match = re.search(r'Name:\s*([^,\n]+)', text)
    if name_match:
        name = name_match.group(1).strip()
        if name and name != ',':
            return name
    degrees_match = re.search(r'^([^,]+),\s*([A-Z][A-Za-z]*\.?(?:\s+[A-Z][A-Za-z]*\.?)*)', text)
    if degrees_match:
        return degrees_match.group(1).strip()
    return ""


class ResearcherSearchInput(BaseModel):
    """Input model for the ResearcherSearchTool."""
    researcher_name: Optional[str] = Field(None, description="The full name of the researcher to search for. Use for specific person-related queries.")
    topic: Optional[str] = Field(None, description="The research topic, interest, or area of expertise to search for. Use for general subject matter queries.")


class ResearcherSearchTool(BaseTool):
    """
    Tool for searching researchers. Use this to find researchers by their name, expertise, interests, or background.
    The agent should provide either a researcher_name or a topic.
    """

    name: str = "ResearcherSearch"
    description: str = "Search for researchers by name or by research topic. Provide either 'researcher_name' for name searches or 'topic' for subject matter searches."
    args_schema: Type[BaseModel] = ResearcherSearchInput

    def _format_results(self, results: List[Document], query: str) -> str:
        """
        Format search results with researcher names and full content.
        """
        if not results:
            return f"No researchers found matching the query: {query}"

        # Remove deduplication logic - use all results directly

        formatted_results = []
        for doc in results:
            profile_url = doc.metadata.get("profile_url", "")
            display_name = doc.metadata.get("researcher_name", "").strip() or extract_name_from_url(profile_url) or extract_name_from_text(doc.page_content) or "Unknown Researcher"
            program = doc.metadata.get("program", "Unknown Program")
            chunk_type = doc.metadata.get("chunk_type", "Unknown Type")

            # Use the full content instead of a snippet
            full_content = doc.page_content

            formatted_results.append(
                f"Researcher: {display_name}\n"
                f"Program: {program}\n"
                f"Chunk Type: {chunk_type}\n"
                f"Content: {full_content}\n"
                f"Profile: {profile_url}\n"
            )

        return "\n\n---\n\n".join(formatted_results)

    def _run(self, tool_input: Union[str, Dict]) -> str:
        """
        Run the tool with a dictionary or JSON string input.
        """
        logger.info("Starting ResearcherSearchTool execution")
        
        # Log structured event for tool start
        log_tool_event("researcher_search_start", {
            "tool_input_type": type(tool_input).__name__,
            "tool_input_length": len(str(tool_input)) if tool_input else 0
        })
        
        researcher_name: Optional[str] = None
        topic: Optional[str] = None

        if isinstance(tool_input, dict):
            researcher_name = tool_input.get("researcher_name")
            topic = tool_input.get("topic")
        elif isinstance(tool_input, str):
            try:
                data = _coerce_to_payload(tool_input)
                researcher_name = data.get("researcher_name")
                topic = data.get("topic")
                if not researcher_name and not topic:
                    topic = tool_input
            except (json.JSONDecodeError, TypeError):
                # If it's not a JSON string, assume it's a topic search for backward compatibility
                topic = tool_input

        if not researcher_name and not topic:
            logger.warning("No researcher_name or topic provided")
            log_tool_event("researcher_search_error", {"error": "missing_input"})
            return "Error: You must provide either a researcher_name or a topic to the ResearcherSearchTool."

        if researcher_name:
            logger.info(f"Performing a name search for: '{researcher_name}'")
            log_tool_event("name_search_detected", {"researcher_name": researcher_name[:50]})
            query = researcher_name
            alpha = 0.3 # Prioritize keyword matching for names
        else: # topic must be present
            logger.info(f"Performing a topic search for: '{topic}'")
            log_tool_event("topic_search_detected", {"topic": topic[:50]})
            query = topic
            alpha = 0.7 # Prioritize semantic matching for topics

        logger.info(f"Searching with query: '{query}' and alpha: {alpha}")
        
        # Log search parameters
        log_tool_event("search_parameters", {
            "query": query[:100],
            "alpha": alpha,
            "search_type": "name" if researcher_name else "topic"
        })

        try:
            results = hybrid_search(
                query=query,
                k=5,
                alpha=alpha
            )
            
            logger.info(f"Search completed, found {len(results) if results else 0} results")
            
            # Log search results
            log_tool_event("search_completed", {
                "query": query[:100],
                "result_count": len(results) if results else 0,
                "alpha": alpha
            })

            if not results:
                logger.info(f"No researchers found matching: {query}")
                return f"No researchers found matching: {query}"

            formatted_results = self._format_results(results, query)
            logger.info(f"Formatted results (length: {len(formatted_results)})")
            
            # Log successful completion
            log_tool_event("researcher_search_complete", {
                "query": query[:100],
                "result_count": len(results),
                "formatted_length": len(formatted_results)
            })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in researcher search: {e}")
            log_tool_event("researcher_search_error", {
                "error": str(e),
                "query": query[:100] if 'query' in locals() else "unknown"
            })
            return f"Error searching for researchers: {str(e)}"

    async def _arun(self, tool_input: Union[str, Dict]) -> str:
        """
        Run the tool asynchronously.
        """
        return self._run(tool_input)