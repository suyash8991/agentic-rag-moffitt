"""
Dependencies for API endpoints.

This module contains shared dependencies that can be used across
multiple API endpoints.
"""

from typing import Annotated

from fastapi import Depends, Path, Query
from ..core.security import get_api_key


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