"""
DepartmentFilterTool for the Moffitt Agentic RAG system.

This module implements a tool for filtering researchers by their department.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from langchain.tools import BaseTool

from ..db.vector_store import get_or_create_vector_db
from ..data.loader import load_all_researcher_profiles
from ..config.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepartmentFilterTool(BaseTool):
    """
    Tool for filtering researchers by their department.

    This tool allows filtering researchers by their academic department,
    and also provides a list of available departments.
    """

    name: str = "DepartmentFilter"
    description: str = "Filter researchers by their department"

    def _run(self, department: str) -> str:
        """
        Run the tool with the given department.

        Args:
            department (str): The department to filter by,
                              or 'list' to show available departments

        Returns:
            str: The filtered researchers formatted as a string
        """
        # First check if we're asked to list available departments
        if department.lower() in ["list", "show", "available"]:
            return self._list_departments()

        logger.info(f"Filtering researchers by department: {department}")

        # Get the database
        db = get_or_create_vector_db()

        # Use similarity_search with a metadata filter
        try:
            # First try exact match
            results = db.similarity_search(
                query="",  # Empty query because we're just filtering
                k=20,
                filter={"department": {"$eq": department}}
            )

            # If no results, try a more flexible match
            if not results:
                # Get all results and filter manually for better flexibility
                results = db.similarity_search(
                    query=f"Department: {department}",
                    k=20
                )

                # Filter to keep only those with matching department
                filtered_results = []
                seen_ids = set()
                for doc in results:
                    researcher_id = doc.metadata.get("researcher_id")
                    dept = doc.metadata.get("department", "")

                    # Check if this department contains our query term
                    if (department.lower() in dept.lower() and
                        researcher_id not in seen_ids):
                        filtered_results.append(doc)
                        seen_ids.add(researcher_id)

                results = filtered_results
        except Exception as e:
            logger.error(f"Error filtering by department: {e}")
            return f"Error filtering by department: {e}"

        # Format results
        if not results:
            return f"No researchers found in department: {department}. Try 'list' to see available departments."

        # Deduplicate results by researcher_id
        seen_ids = set()
        unique_results = []
        for doc in results:
            researcher_id = doc.metadata.get("researcher_id")
            if researcher_id and researcher_id not in seen_ids:
                unique_results.append(doc)
                seen_ids.add(researcher_id)

        formatted_results = [
            f"Researchers in {department} department:\n"
        ]

        for doc in unique_results:
            name = doc.metadata.get("researcher_name", "Unknown")
            program = doc.metadata.get("program", "Unknown Program")
            formatted_results.append(f"- {name} ({program})")

        return "\n".join(formatted_results)

    def _list_departments(self) -> str:
        """
        List all available departments.

        Returns:
            str: The list of departments formatted as a string
        """
        logger.info("Listing all available departments")

        # First try to read the summary.json file
        summary_path = os.path.join(get_settings().processed_data_dir, "summary.json")
        try:
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                    departments = summary.get("departments", {})
                    if departments:
                        return self._format_departments(departments)
        except Exception as e:
            logger.warning(f"Error reading summary.json: {e}")

        # If summary file doesn't exist or doesn't have departments,
        # generate them from the profiles
        try:
            profiles = load_all_researcher_profiles()
            departments = {}
            for profile in profiles:
                if profile.department:
                    departments[profile.department] = departments.get(profile.department, 0) + 1

            return self._format_departments(departments)
        except Exception as e:
            logger.error(f"Error generating department list: {e}")
            return f"Error generating department list: {e}"

    def _format_departments(self, departments: Dict[str, int]) -> str:
        """
        Format a dictionary of departments and counts as a string.

        Args:
            departments (Dict[str, int]): The departments and counts

        Returns:
            str: The formatted departments
        """
        if not departments:
            return "No departments found."

        result = ["Available Departments:"]
        for dept, count in sorted(departments.items()):
            result.append(f"- {dept} ({count} researchers)")

        return "\n".join(result)

    async def _arun(self, department: str) -> str:
        """
        Run the tool asynchronously with the given department.

        Args:
            department (str): The department to filter by,
                              or 'list' to show available departments

        Returns:
            str: The filtered researchers formatted as a string
        """
        # For now, just call the synchronous version
        return self._run(department)