"""
Tools for the Moffitt Agentic RAG system.

This module provides tool implementations for the agent, including
ResearcherSearchTool, DepartmentFilterTool, and ProgramFilterTool.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from langchain.tools import BaseTool
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .vector_db import similarity_search
from .researcher import get_researchers_by_department, get_researchers_by_program

# Setup logging
logger = logging.getLogger(__name__)


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
        logger.info(f"Running ResearcherSearch with name={researcher_name}, topic={topic}")

        if not researcher_name and not topic:
            return "Error: You must provide either 'researcher_name' or 'topic'."

        if researcher_name and topic:
            return "Error: Please provide EITHER 'researcher_name' OR 'topic', not both."

        # Determine the search query
        query = researcher_name if researcher_name else topic

        # Perform the search
        try:
            results = similarity_search(query, k=5)

            if not results:
                if researcher_name:
                    return f"No information found for researcher: {researcher_name}"
                else:
                    return f"No information found for topic: {topic}"

            # Format the results
            formatted_results = []
            for i, doc in enumerate(results):
                try:
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        content = doc.page_content[:1000] if hasattr(doc, 'page_content') else "No content available"
                        formatted_results.append(f"Result {i+1}:\n{content}")
                        continue

                    researcher_name = doc.metadata.get("researcher_name", "Unknown Researcher")
                    program = doc.metadata.get("program", "Unknown Program")
                    department = doc.metadata.get("department", "Unknown Department")
                    profile_url = doc.metadata.get("profile_url", "")

                    content = doc.page_content[:1000] if hasattr(doc, 'page_content') and doc.page_content else "No content available"

                    formatted_results.append(
                        f"Researcher: {researcher_name}\n"
                        f"Program: {program}\n"
                        f"Department: {department}\n"
                        f"Content: {content}...\n"
                        f"Profile: {profile_url}\n"
                        f"[INFORMATION SUFFICIENCY NOTE: This contains basic information about the researcher.]"
                    )
                except Exception as e:
                    logger.error(f"Error formatting result {i+1}: {e}")
                    formatted_results.append(f"Error processing result {i+1}")

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error in ResearcherSearch: {e}")
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