"""
Repository layer for data access.

This module provides abstract interfaces and concrete implementations
for data access, following the Repository Pattern to separate data
access logic from business logic.
"""

from .researcher_repository import (
    IResearcherRepository,
    FileSystemResearcherRepository,
)

__all__ = [
    "IResearcherRepository",
    "FileSystemResearcherRepository",
]