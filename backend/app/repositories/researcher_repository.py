"""
Researcher repository interfaces and implementations.

This module provides data access for researcher profiles, following
the Repository Pattern to separate data access from business logic.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class IResearcherRepository(ABC):
    """
    Abstract interface for researcher data access.

    This interface defines the contract for accessing researcher data,
    allowing different implementations (file system, database, cache, etc.)
    without changing business logic.
    """

    @abstractmethod
    def list_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List all researchers with optional filtering.

        Args:
            filters: Optional dictionary of filter criteria
                     (e.g., {"department": "X", "program": "Y"})

        Returns:
            List[Dict[str, Any]]: List of researcher data dictionaries
        """
        pass

    @abstractmethod
    def get_by_id(self, researcher_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single researcher by ID.

        Args:
            researcher_id: The researcher's unique identifier

        Returns:
            Optional[Dict[str, Any]]: Researcher data or None if not found
        """
        pass

    @abstractmethod
    def get_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary information (departments, programs, statistics).

        Returns:
            Optional[Dict[str, Any]]: Summary data or None if not available
        """
        pass

    @abstractmethod
    def get_unique_values(self, field: str) -> List[str]:
        """
        Get unique values for a specific field across all researchers.

        Args:
            field: The field name (e.g., "department", "primary_program")

        Returns:
            List[str]: Sorted list of unique values
        """
        pass


class FileSystemResearcherRepository(IResearcherRepository):
    """
    File system implementation of researcher repository.

    This implementation reads researcher data from JSON files in a directory.
    All file I/O logic is isolated here, making it easy to swap implementations
    or add caching layers.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize the file system repository.

        Args:
            data_dir: Path to the directory containing researcher JSON files
        """
        self.data_dir = data_dir
        logger.debug(f"Initialized FileSystemResearcherRepository with data_dir: {data_dir}")

    def list_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List all researchers from JSON files with optional filtering.

        Args:
            filters: Optional dictionary of filter criteria

        Returns:
            List[Dict[str, Any]]: List of researcher data dictionaries
        """
        try:
            researchers = []

            # List all JSON files in the directory
            json_files = [
                f for f in os.listdir(self.data_dir)
                if f.endswith(".json") and f != "summary.json"
            ]

            for filename in json_files:
                file_path = os.path.join(self.data_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        # Apply filters if provided
                        if filters:
                            if not self._matches_filters(data, filters):
                                continue

                        researchers.append(data)

                except Exception as e:
                    logger.error(f"Error loading researcher from {file_path}: {e}")

            return researchers

        except Exception as e:
            logger.error(f"Error listing researchers: {e}")
            return []

    def get_by_id(self, researcher_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single researcher by ID from JSON files.

        Args:
            researcher_id: The researcher's unique identifier

        Returns:
            Optional[Dict[str, Any]]: Researcher data or None if not found
        """
        try:
            # Try direct filename match first (faster)
            direct_path = os.path.join(self.data_dir, f"{researcher_id}.json")
            if os.path.exists(direct_path):
                with open(direct_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            # If not found by direct match, search all files
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".json") and filename != "summary.json":
                    file_path = os.path.join(self.data_dir, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("researcher_id") == researcher_id:
                            return data

            return None

        except Exception as e:
            logger.error(f"Error getting researcher {researcher_id}: {e}")
            return None

    def get_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary information from summary.json file.

        Returns:
            Optional[Dict[str, Any]]: Summary data or None if not available
        """
        try:
            summary_path = os.path.join(self.data_dir, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None

        except Exception as e:
            logger.error(f"Error reading summary file: {e}")
            return None

    def get_unique_values(self, field: str) -> List[str]:
        """
        Get unique values for a field by reading all researcher files.

        Args:
            field: The field name to extract unique values from

        Returns:
            List[str]: Sorted list of unique values
        """
        try:
            # First try to get from summary file
            summary = self.get_summary()
            if summary and field == "department":
                departments = summary.get("departments", {})
                if departments:
                    return sorted(departments.keys())
            elif summary and field == "primary_program":
                programs = summary.get("programs", {})
                if programs:
                    return sorted(programs.keys())

            # If summary doesn't have it, extract from individual files
            values = set()
            researchers = self.list_all()

            for researcher in researchers:
                value = researcher.get(field)
                if value:
                    values.add(value)

            return sorted(list(values))

        except Exception as e:
            logger.error(f"Error getting unique values for field '{field}': {e}")
            return []

    def _matches_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if researcher data matches all provided filters.

        Args:
            data: The researcher data dictionary
            filters: Dictionary of filter criteria

        Returns:
            bool: True if data matches all filters, False otherwise
        """
        for key, value in filters.items():
            # Handle None/null filters (skip filtering)
            if value is None:
                continue

            # Handle nested fields (e.g., "metadata.field")
            if "." in key:
                current = data
                for part in key.split("."):
                    if not isinstance(current, dict) or part not in current:
                        return False
                    current = current[part]
                if current != value:
                    return False
            else:
                # Simple field comparison
                if data.get(key) != value:
                    return False

        return True