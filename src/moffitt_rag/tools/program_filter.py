"""
ProgramFilterTool for the Moffitt Agentic RAG system.

This module implements a tool for filtering researchers by their research program.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Set

from langchain.tools import BaseTool

from ..db.vector_store import get_or_create_vector_db
from ..data.loader import load_all_researcher_profiles
from ..config.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgramFilterTool(BaseTool):
    """
    Tool for filtering researchers by their research program.

    This tool allows filtering researchers by their research program,
    and also provides a list of available programs.
    """

    name: str = "ProgramFilter"
    description: str = "Filter researchers by their research program"

    def _run(self, program: str) -> str:
        """
        Run the tool with the given program.

        Args:
            program (str): The program to filter by,
                           or 'list' to show available programs

        Returns:
            str: The filtered researchers formatted as a string
        """
        # First check if we're asked to list available programs
        if program.lower() in ["list", "show", "available"]:
            return self._list_programs()

        logger.info(f"Filtering researchers by program: {program}")

        # Get the database
        db = get_or_create_vector_db()

        # Use similarity_search with a metadata filter
        try:
            # First try exact match on program field
            results = db.similarity_search(
                query="",  # Empty query because we're just filtering
                k=20,
                filter={"program": {"$eq": program}}
            )

            # If few results, try a more flexible match
            if len(results) < 5:
                # Try semantic search with the program name
                semantic_results = db.similarity_search(
                    query=f"Program: {program}",
                    k=20
                )

                # Combine results
                all_results = results + semantic_results

                # Deduplicate by researcher_id
                seen_ids = set()
                unique_results = []
                for doc in all_results:
                    researcher_id = doc.metadata.get("researcher_id")
                    if researcher_id and researcher_id not in seen_ids:
                        unique_results.append(doc)
                        seen_ids.add(researcher_id)

                results = unique_results
        except Exception as e:
            logger.error(f"Error filtering by program: {e}")
            return f"Error filtering by program: {e}"

        # Format results
        if not results:
            return f"No researchers found in program: {program}. Try 'list' to see available programs."

        formatted_results = [
            f"Researchers in {program} program:\n"
        ]

        for doc in results[:10]:  # Limit to top 10
            name = doc.metadata.get("name", "Unknown")
            dept = doc.metadata.get("department", "Unknown Department")
            formatted_results.append(f"- {name} ({dept})")

        return "\n".join(formatted_results)

    def _list_programs(self) -> str:
        """
        List all available programs.

        Returns:
            str: The list of programs formatted as a string
        """
        logger.info("Listing all available programs")

        # First try to read the summary.json file
        summary_path = os.path.join(get_settings().processed_data_dir, "summary.json")
        try:
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                    programs = summary.get("programs", {})
                    if programs:
                        return self._format_programs(programs)
        except Exception as e:
            logger.warning(f"Error reading summary.json: {e}")

        # If summary file doesn't exist or doesn't have programs,
        # generate them from the profiles
        try:
            profiles = load_all_researcher_profiles()
            programs = {}
            for profile in profiles:
                if profile.primary_program:
                    programs[profile.primary_program] = programs.get(profile.primary_program, 0) + 1
                if profile.research_program:
                    programs[profile.research_program] = programs.get(profile.research_program, 0) + 1

            return self._format_programs(programs)
        except Exception as e:
            logger.error(f"Error generating program list: {e}")
            return f"Error generating program list: {e}"

    def _format_programs(self, programs: Dict[str, int]) -> str:
        """
        Format a dictionary of programs and counts as a string.

        Args:
            programs (Dict[str, int]): The programs and counts

        Returns:
            str: The formatted programs
        """
        if not programs:
            return "No programs found."

        result = ["Available Research Programs:"]
        for program, count in sorted(programs.items()):
            result.append(f"- {program} ({count} researchers)")

        return "\n".join(result)

    async def _arun(self, program: str) -> str:
        """
        Run the tool asynchronously with the given program.

        Args:
            program (str): The program to filter by,
                           or 'list' to show available programs

        Returns:
            str: The filtered researchers formatted as a string
        """
        # For now, just call the synchronous version
        return self._run(program)