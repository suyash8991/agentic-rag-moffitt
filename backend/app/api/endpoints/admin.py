"""
Admin API endpoints.

This module provides endpoints for administrative tasks like
rebuilding the vector database and retrieving system statistics.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ...core.security import get_api_key
from ...services.vector_db import rebuild_vector_database, get_database_stats

router = APIRouter()


class RebuildRequest(BaseModel):
    """Request model for database rebuild."""
    force: bool = False


class RebuildResponse(BaseModel):
    """Response model for database rebuild."""
    task_id: str
    status: str


class DatabaseStats(BaseModel):
    """Database statistics model."""
    total_researchers: int
    total_chunks: int
    last_updated: Optional[str] = None
    status: str


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_database(
    rebuild_request: RebuildRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
):
    """
    Rebuild the vector database.

    This operation can be time-consuming, so it runs in the background.

    Args:
        rebuild_request: Rebuild options
        background_tasks: FastAPI background tasks
        api_key: API key for authentication

    Returns:
        RebuildResponse: Task ID and status
    """
    try:
        task_id = rebuild_vector_database(force=rebuild_request.force, background_tasks=background_tasks)
        return {"task_id": task_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=DatabaseStats)
async def get_stats(
    api_key: str = Depends(get_api_key),
):
    """
    Get vector database statistics.

    Args:
        api_key: API key for authentication

    Returns:
        DatabaseStats: Database statistics
    """
    try:
        stats = get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))