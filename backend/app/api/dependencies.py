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
from ..services.query_status_service import (
    QueryStatusService,
    InMemoryQueryStatusRepository
)
from ..services.embedding_service import EmbeddingService


# Common dependencies
ApiKey = Annotated[str, Depends(get_api_key)]


# Singleton instances for services that should be shared across requests
_query_status_service_instance = None
_embedding_service_instance = None


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


def get_query_status_service() -> QueryStatusService:
    """
    Dependency provider for QueryStatusService.

    Returns a singleton instance that is shared across all requests.
    This ensures query status tracking persists across different API calls.

    Returns:
        QueryStatusService: Singleton service instance
    """
    global _query_status_service_instance

    if _query_status_service_instance is None:
        # Create repository and service on first use
        repository = InMemoryQueryStatusRepository()
        _query_status_service_instance = QueryStatusService(repository)

    return _query_status_service_instance


def get_embedding_service() -> EmbeddingService:
    """
    Dependency provider for EmbeddingService.

    Returns a singleton instance that is shared across all requests.
    This ensures the embedding model is loaded once and cached.

    Returns:
        EmbeddingService: Singleton service instance
    """
    global _embedding_service_instance

    if _embedding_service_instance is None:
        # Create service on first use
        _embedding_service_instance = EmbeddingService()

    return _embedding_service_instance


# Typed dependencies for new services
QueryStatusServiceDep = Annotated[QueryStatusService, Depends(get_query_status_service)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]