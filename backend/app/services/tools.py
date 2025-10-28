"""
Tools for the Moffitt Agentic RAG system.

This module provides tool implementations for the agent, including
ResearcherSearchTool, DepartmentFilterTool, and ProgramFilterTool.
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Type, Union

from langchain.tools import BaseTool
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .vector_db import similarity_search
from .researcher import get_researchers_by_department, get_researchers_by_program
from .hybrid_search import hybrid_search
from .name_normalization import NameNormalizationService
from ..utils.logging import get_logger, log_tool_event, log_search_event, log_error_event

# Setup logging
logger = get_logger(__name__)

# Initialize name normalization service (loads alias map on import)
_name_normalizer = NameNormalizationService()


def _parse_tool_input(tool_input: Union[str, Dict]) -> Dict[str, Any]:
    """
    Parse tool input handling both dict and JSON string formats.

    This handles cases where LangChain passes the entire input as a JSON string
    instead of properly parsed parameters.

    Args:
        tool_input: Either a dict or a string (potentially JSON)

    Returns:
        Dict with parsed parameters
    """
    # If already a dict, return as-is
    if isinstance(tool_input, dict):
        return tool_input

    # If string, try to parse as JSON
    if isinstance(tool_input, str):
        tool_input = tool_input.strip()

        # Try direct JSON parsing
        try:
            data = json.loads(tool_input)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting first {...} block
        match = re.search(r'\{.*?\}', tool_input, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: treat entire string as topic
        return {"topic": tool_input}

    return {}


class ResearcherSearchInput(BaseModel):
    """Input for the ResearcherSearch tool."""
    researcher_name: Optional[str] = Field(
        None,
        description="The name of the researcher to search for"
    )
    topic: Optional[str] = Field(
        None,
        description="The research topic to search for"
    )


class ResearcherSearchTool(BaseTool):
    """
    Tool for searching for researchers by name or by research topic.
    """
    name: str = "ResearcherSearch"
    description: str = """
    Find researchers by name or by research topic.
    - To search by name, provide the person's full name to the 'researcher_name' argument
    - To search by topic, provide the subject matter to the 'topic' argument
    - You must provide EITHER 'researcher_name' OR 'topic'
    """
    args_schema: Type[BaseModel] = ResearcherSearchInput

    def _run(self, researcher_name: Optional[str] = None, topic: Optional[str] = None) -> str:
        """
        Run the researcher search tool.

        Args:
            researcher_name: The name of the researcher to search for.
            topic: The research topic to search for.

        Returns:
            str: The search results.
        """
        # Log raw inputs for debugging
        log_tool_event("tool_input_raw", {
            "researcher_name_raw": str(researcher_name)[:100] if researcher_name else None,
            "researcher_name_type": type(researcher_name).__name__ if researcher_name else None,
            "topic_raw": str(topic)[:100] if topic else None,
            "topic_type": type(topic).__name__ if topic else None
        })

        # Handle case where LangChain passes the entire input as a JSON string
        # This happens when the agent sends something like: {"researcher_name": "Conor Lynch"}
        if researcher_name and isinstance(researcher_name, str):
            # Check if it looks like JSON
            if researcher_name.strip().startswith('{'):
                parsed = _parse_tool_input(researcher_name)
                researcher_name = parsed.get("researcher_name")
                topic = parsed.get("topic")

                # Log parsed values
                log_tool_event("input_parsed_from_json", {
                    "parsed_researcher_name": researcher_name[:50] if researcher_name else None,
                    "parsed_topic": topic[:50] if topic else None
                })

        # Log tool start with final values
        log_tool_event("researcher_search_start", {
            "researcher_name": researcher_name[:50] if researcher_name else None,
            "topic": topic[:50] if topic else None
        })

        if not researcher_name and not topic:
            log_tool_event("researcher_search_error", {"error": "missing_input"})
            return "Error: You must provide either 'researcher_name' or 'topic'."

        if researcher_name and topic:
            log_tool_event("researcher_search_error", {"error": "both_inputs_provided"})
            return "Error: Please provide EITHER 'researcher_name' OR 'topic', not both."

        # Determine the search query and alpha parameter
        if researcher_name:
            alpha = 0.3  # Favor keyword matching for name searches
            search_type = "name"
            log_tool_event("name_search_detected", {"researcher_name": researcher_name[:50]})

            # Normalize the provided name (alias first, then fuzzy if available)
            norm_result = _name_normalizer.normalize(researcher_name)
            if norm_result.canonical:
                log_tool_event("name_normalized", {
                    "input": researcher_name[:100],
                    "canonical": norm_result.canonical,
                    "method": norm_result.method,
                    "score": norm_result.score
                })
                query = norm_result.canonical
                metadata_filter = {"researcher_name": norm_result.canonical}
            else:
                log_tool_event("name_normalization_failed", {
                    "input": researcher_name[:100],
                    "score": norm_result.score
                })
                query = researcher_name
                metadata_filter = None
        else:
            query = topic
            alpha = 0.7  # Favor semantic matching for topic searches
            search_type = "topic"
            log_tool_event("topic_search_detected", {"topic": topic[:50]})

        # Log search parameters
        log_tool_event("search_parameters", {
            "query": query[:100],
            "alpha": alpha,
            "search_type": search_type,
            "k": 5
        })

        # Perform hybrid search
        try:
            results = hybrid_search(
                query=query,
                k=5,
                alpha=alpha,
                filter=metadata_filter if researcher_name else None,
                search_type=search_type
            )

            # Log search completion
            log_search_event(
                query=query,
                search_type=search_type,
                alpha=alpha,
                result_count=len(results) if results else 0
            )

            if not results:
                log_tool_event("no_results_found", {"query": query[:100]})
                if researcher_name:
                    return f"No information found for researcher: {researcher_name}"
                else:
                    return f"No information found for topic: {topic}"

            # Format the results
            formatted_results = []
            for i, doc in enumerate(results):
                try:
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        # Use full content instead of truncating
                        content = doc.page_content if hasattr(doc, 'page_content') else "No content available"
                        formatted_results.append(f"Result {i+1}:\n{content}")
                        continue

                    name = doc.metadata.get("researcher_name", "Unknown Researcher")
                    program = doc.metadata.get("program", "Unknown Program")
                    department = doc.metadata.get("department", "Unknown Department")
                    profile_url = doc.metadata.get("profile_url", "")
                    chunk_type = doc.metadata.get("chunk_type", "Unknown Type")

                    # Use full content instead of truncating at 1000 chars
                    content = doc.page_content if hasattr(doc, 'page_content') and doc.page_content else "No content available"

                    formatted_results.append(
                        f"Researcher: {name}\n"
                        f"Program: {program}\n"
                        f"Department: {department}\n"
                        f"Chunk Type: {chunk_type}\n"
                        f"Content: {content}\n"
                        f"Profile: {profile_url}\n"
                    )
                except Exception as e:
                    logger.error(f"Error formatting result {i+1}: {e}")
                    log_error_event(e, {"result_index": i+1, "query": query[:100]})
                    formatted_results.append(f"Error processing result {i+1}")

            # Log successful completion with document details
            log_tool_event("researcher_search_complete", {
                "query": query[:100],
                "result_count": len(results),
                "formatted_length": len("\n\n---\n\n".join(formatted_results)),
                "returned_documents": [
                    {
                        "researcher_name": doc.metadata.get("researcher_name", "Unknown") if hasattr(doc, 'metadata') and doc.metadata else "Unknown",
                        "program": doc.metadata.get("program", "Unknown") if hasattr(doc, 'metadata') and doc.metadata else "Unknown",
                        "department": doc.metadata.get("department", "Unknown") if hasattr(doc, 'metadata') and doc.metadata else "Unknown",
                        "chunk_type": doc.metadata.get("chunk_type", "Unknown") if hasattr(doc, 'metadata') and doc.metadata else "Unknown",
                        "profile_url": doc.metadata.get("profile_url", "") if hasattr(doc, 'metadata') and doc.metadata else "",
                        "content_length": len(doc.page_content) if hasattr(doc, 'page_content') else 0,
                        "content_preview": doc.page_content[:300] if hasattr(doc, 'page_content') else ""
                    }
                    for doc in results
                ]
            })

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error in ResearcherSearch: {e}")
            log_error_event(e, {"query": query[:100], "search_type": search_type})
            return f"Error searching for {'researcher' if researcher_name else 'topic'}: {str(e)}"

    def _arun(self, researcher_name: Optional[str] = None, topic: Optional[str] = None) -> str:
        """
        Run the researcher search tool asynchronously.

        Args:
            researcher_name: The name of the researcher to search for.
            topic: The research topic to search for.

        Returns:
            str: The search results.
        """
        # For now, we'll just call the synchronous version
        # In the future, this can be updated to use async functions
        return self._run(researcher_name=researcher_name, topic=topic)


class DepartmentFilterTool(BaseTool):
    """
    Tool for filtering researchers by academic department.
    """
    name: str = "DepartmentFilter"
    description: str = "Find researchers in a specific academic department"

    def _run(self, department: str) -> str:
        """
        Run the department filter tool.

        Args:
            department: The department name to filter by.

        Returns:
            str: The filtered results.
        """
        logger.info(f"Running DepartmentFilter with department={department}")

        if not department:
            return "Error: You must provide a department name."

        # Get researchers by department
        try:
            researchers = get_researchers_by_department(department)

            if not researchers:
                return f"No researchers found in department: {department}"

            # Format the results
            formatted_results = [
                f"Department: {department}\n"
                f"Researchers: {len(researchers)}\n"
                f"\nResearchers in this department:"
            ]

            for researcher in researchers:
                formatted_results.append(
                    f"- {researcher.name}: {researcher.program}"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error in DepartmentFilter: {e}")
            return f"Error filtering by department: {str(e)}"

    def _arun(self, department: str) -> str:
        """
        Run the department filter tool asynchronously.

        Args:
            department: The department name to filter by.

        Returns:
            str: The filtered results.
        """
        # For now, we'll just call the synchronous version
        return self._run(department=department)


class ProgramFilterTool(BaseTool):
    """
    Tool for filtering researchers by research program.
    """
    name: str = "ProgramFilter"
    description: str = "Find researchers in a specific research program"

    def _run(self, program: str) -> str:
        """
        Run the program filter tool.

        Args:
            program: The program name to filter by.

        Returns:
            str: The filtered results.
        """
        logger.info(f"Running ProgramFilter with program={program}")

        if not program:
            return "Error: You must provide a program name."

        # Get researchers by program
        try:
            researchers = get_researchers_by_program(program)

            if not researchers:
                return f"No researchers found in program: {program}"

            # Format the results
            formatted_results = [
                f"Program: {program}\n"
                f"Researchers: {len(researchers)}\n"
                f"\nResearchers in this program:"
            ]

            for researcher in researchers:
                formatted_results.append(
                    f"- {researcher.name}: {researcher.department}"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error in ProgramFilter: {e}")
            return f"Error filtering by program: {str(e)}"

    def _arun(self, program: str) -> str:
        """
        Run the program filter tool asynchronously.

        Args:
            program: The program name to filter by.

        Returns:
            str: The filtered results.
        """
        # For now, we'll just call the synchronous version
        return self._run(program=program)


def get_tools() -> List[BaseTool]:
    """
    Get all tools for the agent.

    Returns:
        List[BaseTool]: The list of tools.
    """
    return [
        ResearcherSearchTool(),
        DepartmentFilterTool(),
        ProgramFilterTool()
    ]
