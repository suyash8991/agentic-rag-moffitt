"""
Researcher services.

This module provides functions for working with researcher data,
including loading, filtering, and transforming researcher profiles.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..core.config import settings
from ..models.researcher import (
    ResearcherProfileSummary,
    ResearcherProfileDetail,
    ResearcherList
)

# Setup logging
logger = logging.getLogger(__name__)


def list_researchers(
    skip: int = 0,
    limit: int = 10,
    department: Optional[str] = None,
    program: Optional[str] = None,
) -> ResearcherList:
    """
    List researchers with pagination and optional filtering.

    Args:
        skip: Number of researchers to skip (for pagination)
        limit: Maximum number of researchers to return
        department: Optional department filter
        program: Optional program filter

    Returns:
        ResearcherList: List of researchers with pagination info
    """
    # Get the list of researcher JSON files
    try:
        researchers = []
        data_dir = settings.PROCESSED_DATA_DIR

        # List all JSON files in the directory
        json_files = [
            f for f in os.listdir(data_dir)
            if f.endswith(".json") and f != "summary.json"
        ]

        # Apply filtering if needed
        filtered_researchers = []
        total_count = 0

        for filename in json_files:
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Apply filters if provided
                    if department and data.get("department") != department:
                        continue

                    if program and data.get("primary_program") != program:
                        continue

                    # Increment total count
                    total_count += 1

                    # Skip if needed (for pagination)
                    if total_count <= skip:
                        continue

                    # Add to results if within limit
                    if len(filtered_researchers) < limit:
                        # Create a ResearcherProfileSummary
                        researcher = ResearcherProfileSummary(
                            researcher_id=data.get("researcher_id", ""),
                            researcher_name=data.get("researcher_name", ""),
                            primary_program=data.get("primary_program"),
                            department=data.get("department"),
                            profile_url=data.get("profile_url", ""),
                            degrees=data.get("degrees", []),
                            title=data.get("title"),
                            overview=data.get("overview"),
                            research_interests=data.get("research_interests", [])
                        )
                        filtered_researchers.append(researcher)
            except Exception as e:
                logger.error(f"Error loading researcher from {file_path}: {e}")

        return ResearcherList(
            items=filtered_researchers,
            total=total_count,
            skip=skip,
            limit=limit
        )

    except Exception as e:
        logger.error(f"Error listing researchers: {e}")
        # Return empty list on error
        return ResearcherList(
            items=[],
            total=0,
            skip=skip,
            limit=limit
        )


def get_researcher_by_id(researcher_id: str) -> Optional[ResearcherProfileDetail]:
    """
    Get detailed information for a specific researcher.

    Args:
        researcher_id: ID of the researcher

    Returns:
        Optional[ResearcherProfileDetail]: Researcher profile or None if not found
    """
    try:
        # Look for the researcher's JSON file
        data_dir = settings.PROCESSED_DATA_DIR

        # Try direct filename match first (faster)
        direct_path = os.path.join(data_dir, f"{researcher_id}.json")
        if os.path.exists(direct_path):
            with open(direct_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return ResearcherProfileDetail.model_validate(data)

        # If not found by direct match, search all files
        for filename in os.listdir(data_dir):
            if filename.endswith(".json") and filename != "summary.json":
                file_path = os.path.join(data_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("researcher_id") == researcher_id:
                        return ResearcherProfileDetail.model_validate(data)

        # If we get here, researcher was not found
        return None

    except Exception as e:
        logger.error(f"Error getting researcher {researcher_id}: {e}")
        return None


def list_departments() -> List[str]:
    """
    List all available departments.

    Returns:
        List[str]: List of department names
    """
    try:
        # First try to read the summary.json file
        summary_path = os.path.join(settings.PROCESSED_DATA_DIR, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                departments = summary.get("departments", {})
                return sorted(departments.keys())

        # If summary file doesn't exist, extract from individual files
        departments = set()
        data_dir = settings.PROCESSED_DATA_DIR

        for filename in os.listdir(data_dir):
            if filename.endswith(".json") and filename != "summary.json":
                file_path = os.path.join(data_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "department" in data and data["department"]:
                        departments.add(data["department"])

        return sorted(list(departments))

    except Exception as e:
        logger.error(f"Error listing departments: {e}")
        return []


def list_programs() -> List[str]:
    """
    List all available research programs.

    Returns:
        List[str]: List of program names
    """
    try:
        # First try to read the summary.json file
        summary_path = os.path.join(settings.PROCESSED_DATA_DIR, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                programs = summary.get("programs", {})
                return sorted(programs.keys())

        # If summary file doesn't exist, extract from individual files
        programs = set()
        data_dir = settings.PROCESSED_DATA_DIR

        for filename in os.listdir(data_dir):
            if filename.endswith(".json") and filename != "summary.json":
                file_path = os.path.join(data_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "primary_program" in data and data["primary_program"]:
                        programs.add(data["primary_program"])

        return sorted(list(programs))

    except Exception as e:
        logger.error(f"Error listing programs: {e}")
        return []


def get_researchers_by_department(department: str) -> List[ResearcherProfileSummary]:
    """
    Get all researchers in a specific department.

    Args:
        department: The department name to filter by.

    Returns:
        List[ResearcherProfileSummary]: List of researchers in the department
    """
    try:
        # Use list_researchers function with department filter
        result = list_researchers(skip=0, limit=100, department=department)
        return result.items
    except Exception as e:
        logger.error(f"Error getting researchers by department {department}: {e}")
        return []


def get_researchers_by_program(program: str) -> List[ResearcherProfileSummary]:
    """
    Get all researchers in a specific program.

    Args:
        program: The program name to filter by.

    Returns:
        List[ResearcherProfileSummary]: List of researchers in the program
    """
    try:
        # Use list_researchers function with program filter
        result = list_researchers(skip=0, limit=100, program=program)
        return result.items
    except Exception as e:
        logger.error(f"Error getting researchers by program {program}: {e}")
        return []