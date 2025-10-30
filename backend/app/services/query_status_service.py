"""
Query status management service.

This module provides a service for managing query statuses, replacing
the global _query_statuses dictionary with proper dependency injection.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from ..models.query import QueryStatus

logger = logging.getLogger(__name__)


class IQueryStatusRepository(ABC):
    """Abstract interface for query status storage."""

    @abstractmethod
    def create(self, query_id: str, data: Dict[str, Any]) -> None:
        """Create a new query status entry."""
        pass

    @abstractmethod
    def get(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get query status by ID."""
        pass

    @abstractmethod
    def update(self, query_id: str, data: Dict[str, Any]) -> None:
        """Update query status."""
        pass

    @abstractmethod
    def exists(self, query_id: str) -> bool:
        """Check if query status exists."""
        pass


class InMemoryQueryStatusRepository(IQueryStatusRepository):
    """In-memory implementation of query status repository."""

    def __init__(self):
        """Initialize the in-memory storage."""
        self._storage: Dict[str, Dict[str, Any]] = {}
        logger.debug("Initialized InMemoryQueryStatusRepository")

    def create(self, query_id: str, data: Dict[str, Any]) -> None:
        """Create a new query status entry."""
        self._storage[query_id] = data

    def get(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get query status by ID."""
        return self._storage.get(query_id)

    def update(self, query_id: str, data: Dict[str, Any]) -> None:
        """Update query status."""
        if query_id in self._storage:
            self._storage[query_id].update(data)

    def exists(self, query_id: str) -> bool:
        """Check if query status exists."""
        return query_id in self._storage


class QueryStatusService:
    """Service for managing query statuses."""

    def __init__(self, repository: IQueryStatusRepository):
        """
        Initialize the query status service.

        Args:
            repository: Repository for storing query statuses
        """
        self.repository = repository
        logger.debug(f"Initialized QueryStatusService with repository: {type(repository).__name__}")

    def create_status(
        self,
        query_id: str,
        query: str,
        query_type: str,
        streaming: bool,
        max_results: int
    ) -> None:
        """
        Create a new query status entry.

        Args:
            query_id: Unique query identifier
            query: The query text
            query_type: Type of query
            streaming: Whether streaming is enabled
            max_results: Maximum number of results
        """
        data = {
            "query_id": query_id,
            "status": "processing",
            "query": query,
            "query_type": query_type,
            "start_time": datetime.now().isoformat(),
            "progress": 0.0,
            "streaming": streaming,
            "max_results": max_results,
        }
        self.repository.create(query_id, data)

    def update_progress(self, query_id: str, progress: float) -> None:
        """
        Update query progress.

        Args:
            query_id: Query identifier
            progress: Progress value (0.0 to 1.0)
        """
        self.repository.update(query_id, {"progress": progress})

    def mark_completed(self, query_id: str, answer: str) -> None:
        """
        Mark query as completed with answer.

        Args:
            query_id: Query identifier
            answer: The final answer
        """
        self.repository.update(query_id, {
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "answer": answer
        })

    def mark_error(self, query_id: str, error: str) -> None:
        """
        Mark query as failed with error.

        Args:
            query_id: Query identifier
            error: Error message
        """
        self.repository.update(query_id, {
            "status": "error",
            "error": error,
            "end_time": datetime.now().isoformat()
        })

    def get_status(self, query_id: str) -> Optional[QueryStatus]:
        """
        Get the status of a query.

        Args:
            query_id: The ID of the query

        Returns:
            Optional[QueryStatus]: The query status, or None if not found
        """
        data = self.repository.get(query_id)
        if not data:
            return None

        return QueryStatus(
            query_id=query_id,
            status=data["status"],
            progress=data["progress"],
            completed=data["status"] == "completed",
            error=data.get("error"),
        )
