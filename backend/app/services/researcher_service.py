"""
Researcher business logic service.

This module contains business logic for researcher operations,
separated from data access concerns. It uses the repository pattern
for data access via dependency injection.
"""

import logging
from typing import List, Optional

from ..repositories.researcher_repository import IResearcherRepository
from ..models.researcher import (
    ResearcherProfileSummary,
    ResearcherProfileDetail,
    ResearcherList
)

logger = logging.getLogger(__name__)


class ResearcherService:
    """
    Service for researcher business logic.

    This service handles all business logic related to researchers,
    delegating data access to the injected repository.
    """

    def __init__(self, repository: IResearcherRepository):
        """
        Initialize the researcher service.

        Args:
            repository: The repository implementation to use for data access
        """
        self.repository = repository
        logger.debug(f"Initialized ResearcherService with repository: {type(repository).__name__}")

    def list_researchers(
        self,
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
        try:
            # Build filters dictionary
            filters = {}
            if department:
                filters["department"] = department
            if program:
                filters["primary_program"] = program

            # Get all researchers matching filters from repository
            all_researchers = self.repository.list_all(filters=filters if filters else None)
            total_count = len(all_researchers)

            # Apply pagination (business logic)
            paginated_researchers = all_researchers[skip : skip + limit]

            # Transform to summary models (business logic)
            items = [
                self._to_summary_model(data)
                for data in paginated_researchers
            ]

            return ResearcherList(
                items=items,
                total=total_count,
                skip=skip,
                limit=limit
            )

        except Exception as e:
            logger.error(f"Error listing researchers: {e}")
            return ResearcherList(
                items=[],
                total=0,
                skip=skip,
                limit=limit
            )

    def get_researcher_by_id(self, researcher_id: str) -> Optional[ResearcherProfileDetail]:
        """
        Get detailed information for a specific researcher.

        Args:
            researcher_id: ID of the researcher

        Returns:
            Optional[ResearcherProfileDetail]: Researcher profile or None if not found
        """
        try:
            data = self.repository.get_by_id(researcher_id)
            if data:
                return ResearcherProfileDetail.model_validate(data)
            return None

        except Exception as e:
            logger.error(f"Error getting researcher {researcher_id}: {e}")
            return None

    def list_departments(self) -> List[str]:
        """
        List all available departments.

        Returns:
            List[str]: List of department names
        """
        try:
            return self.repository.get_unique_values("department")

        except Exception as e:
            logger.error(f"Error listing departments: {e}")
            return []

    def list_programs(self) -> List[str]:
        """
        List all available research programs.

        Returns:
            List[str]: List of program names
        """
        try:
            return self.repository.get_unique_values("primary_program")

        except Exception as e:
            logger.error(f"Error listing programs: {e}")
            return []

    def get_researchers_by_department(self, department: str) -> List[ResearcherProfileSummary]:
        """
        Get all researchers in a specific department.

        Args:
            department: The department name to filter by.

        Returns:
            List[ResearcherProfileSummary]: List of researchers in the department
        """
        try:
            # Use list_researchers with department filter
            result = self.list_researchers(skip=0, limit=1000, department=department)
            return result.items

        except Exception as e:
            logger.error(f"Error getting researchers by department {department}: {e}")
            return []

    def get_researchers_by_program(self, program: str) -> List[ResearcherProfileSummary]:
        """
        Get all researchers in a specific program.

        Args:
            program: The program name to filter by.

        Returns:
            List[ResearcherProfileSummary]: List of researchers in the program
        """
        try:
            # Use list_researchers with program filter
            result = self.list_researchers(skip=0, limit=1000, program=program)
            return result.items

        except Exception as e:
            logger.error(f"Error getting researchers by program {program}: {e}")
            return []

    def _to_summary_model(self, data: dict) -> ResearcherProfileSummary:
        """
        Transform raw data dictionary to ResearcherProfileSummary model.

        Args:
            data: Raw researcher data dictionary

        Returns:
            ResearcherProfileSummary: Transformed summary model
        """
        return ResearcherProfileSummary(
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
