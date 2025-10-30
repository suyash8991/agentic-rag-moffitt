"""
Dependencies for API endpoints.

This module contains shared dependencies that can be used across
multiple API endpoints.
"""

from typing import Annotated
from pathlib import Path

from fastapi import Depends, Query
from ..core.security import get_api_key
from ..core.config import settings
from ..repositories.researcher_repository import FileSystemResearcherRepository
from ..services.researcher_service import ResearcherService


# Common dependencies
ApiKey = Annotated[str, Depends(get_api_key)]


# Common parameters
def common_parameters(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of items to return"),
):
    """
    Common query parameters for pagination.

    Args:
        skip: Number of items to skip
        limit: Maximum number of items to return

    Returns:
        Dict with pagination parameters
    """
    return {"skip": skip, "limit": limit}


# Service dependencies
def get_researcher_service() -> ResearcherService:
    """
    Dependency provider for ResearcherService.

    Creates and returns a ResearcherService instance with the appropriate
    repository implementation injected. This enables dependency injection
    at the endpoint level.

    Returns:
        ResearcherService: Service instance with repository injected
    """
    # Create repository (could be swapped with different implementations)
    repository = FileSystemResearcherRepository(data_dir=settings.PROCESSED_DATA_DIR)

    # Create and return service with repository injected
    return ResearcherService(repository=repository)


# Typed dependency for ResearcherService
ResearcherServiceDep = Annotated[ResearcherService, Depends(get_researcher_service)]