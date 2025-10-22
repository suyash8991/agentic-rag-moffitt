"""
Base classes and interfaces for tools in the Moffitt Agentic RAG system.

This module defines interfaces and base classes for various tool types,
supporting better SOLID principles adherence in the tools implementation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Generic type for tool input
T = TypeVar('T', bound=BaseModel)
# Generic type for tool output
U = TypeVar('U')


class ToolCapability(ABC):
    """Base interface for tool capabilities."""
    pass


class SearchCapable(ToolCapability):
    """Interface for tools that can perform searches."""

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Any]:
        """
        Perform a search using the tool.

        Args:
            query (str): The search query
            **kwargs: Additional search parameters

        Returns:
            List[Any]: The search results
        """
        pass


class FilterCapable(ToolCapability):
    """Interface for tools that can filter results."""

    @abstractmethod
    def filter(self, filter_value: str, **kwargs) -> List[Any]:
        """
        Filter results using the tool.

        Args:
            filter_value (str): The value to filter by
            **kwargs: Additional filter parameters

        Returns:
            List[Any]: The filtered results
        """
        pass


class FormattingCapable(ToolCapability):
    """Interface for tools that can format results."""

    @abstractmethod
    def format_results(self, results: List[Any], **kwargs) -> str:
        """
        Format results into a string.

        Args:
            results (List[Any]): The results to format
            **kwargs: Additional formatting parameters

        Returns:
            str: The formatted results
        """
        pass


class RateLimitCapable(ToolCapability):
    """Interface for tools with rate limiting."""

    @abstractmethod
    def check_rate_limit(self, key: str) -> bool:
        """
        Check if the rate limit for a key has been exceeded.

        Args:
            key (str): The key to check

        Returns:
            bool: True if the rate limit has not been exceeded, False otherwise
        """
        pass

    @abstractmethod
    def increment_rate_limit(self, key: str) -> None:
        """
        Increment the rate limit counter for a key.

        Args:
            key (str): The key to increment
        """
        pass


class MoffittBaseTool(BaseTool, ABC):
    """
    Base class for all tools in the Moffitt Agentic RAG system.

    This class provides common functionality for all tools, including
    logging and error handling.
    """

    def _log_tool_start(self, tool_input: Any) -> None:
        """
        Log the start of a tool execution.

        Args:
            tool_input (Any): The input to the tool
        """
        from ..utils.logging import log_tool_event

        logger.info(f"Starting {self.name} execution")
        log_tool_event(f"{self.name.lower()}_start", {
            "tool_input_type": type(tool_input).__name__,
            "tool_input_length": len(str(tool_input)) if tool_input else 0
        })

    def _log_tool_complete(self, tool_input: Any, result: Any) -> None:
        """
        Log the completion of a tool execution.

        Args:
            tool_input (Any): The input to the tool
            result (Any): The result of the tool execution
        """
        from ..utils.logging import log_tool_event

        logger.info(f"Completed {self.name} execution")
        log_tool_event(f"{self.name.lower()}_complete", {
            "tool_input_type": type(tool_input).__name__,
            "result_length": len(str(result)) if result else 0
        })

    def _log_tool_error(self, tool_input: Any, error: Exception) -> None:
        """
        Log an error during tool execution.

        Args:
            tool_input (Any): The input to the tool
            error (Exception): The error that occurred
        """
        from ..utils.logging import log_tool_event

        logger.error(f"Error in {self.name} execution: {error}")
        log_tool_event(f"{self.name.lower()}_error", {
            "tool_input_type": type(tool_input).__name__,
            "error": str(error),
            "error_type": type(error).__name__
        })


class SearchTool(MoffittBaseTool, SearchCapable, FormattingCapable):
    """
    Base class for search tools.

    This class provides common functionality for tools that perform searches
    and format the results.
    """

    # Class-level tracking of queries and attempts (for rate limiting)
    _query_attempts = {}
    _max_attempts_per_query = 3

    def check_rate_limit(self, query: str) -> bool:
        """
        Check if the query has exceeded the rate limit.

        Args:
            query (str): The search query

        Returns:
            bool: True if the query is allowed, False if it should be rate-limited
        """
        # Normalize query for comparison (lowercase and remove extra spaces)
        normalized_query = ' '.join(query.lower().split())

        # Get current count for this query
        current_count = self._query_attempts.get(normalized_query, 0)

        # Check if we've exceeded the limit
        if current_count >= self._max_attempts_per_query:
            logger.warning(f"Rate limit exceeded for query: {query} ({current_count} attempts)")
            return False

        # Increment the count
        self._query_attempts[normalized_query] = current_count + 1
        logger.info(f"Query attempt {current_count + 1}/{self._max_attempts_per_query} for: {query}")

        return True

    def increment_rate_limit(self, query: str) -> None:
        """
        Increment the rate limit counter for a query.

        Args:
            query (str): The query to increment
        """
        # Normalize query for comparison (lowercase and remove extra spaces)
        normalized_query = ' '.join(query.lower().split())

        # Get current count for this query
        current_count = self._query_attempts.get(normalized_query, 0)

        # Increment the count
        self._query_attempts[normalized_query] = current_count + 1


class FilterTool(MoffittBaseTool, FilterCapable, FormattingCapable):
    """
    Base class for filter tools.

    This class provides common functionality for tools that filter results
    and format the output.
    """

    def _run(self, filter_value: str) -> str:
        """
        Run the filter tool with the given filter value.

        Args:
            filter_value (str): The value to filter by

        Returns:
            str: The formatted filtered results
        """
        self._log_tool_start(filter_value)

        try:
            # Check if the request is for a list of available values
            if filter_value.lower() == "list":
                result = self.get_available_values()
                self._log_tool_complete(filter_value, result)
                return result

            # Perform the filtering
            filtered_results = self.filter(filter_value)

            # Format the results
            formatted_results = self.format_results(filtered_results, filter_value=filter_value)

            self._log_tool_complete(filter_value, formatted_results)
            return formatted_results

        except Exception as e:
            self._log_tool_error(filter_value, e)
            return f"Error in {self.name}: {str(e)}"

    @abstractmethod
    def get_available_values(self) -> str:
        """
        Get a list of available values for this filter.

        Returns:
            str: A formatted string listing the available values
        """
        pass


class TypedSearchTool(MoffittBaseTool, Generic[T, U]):
    """
    Base class for search tools with strongly typed inputs and outputs.

    This class provides common functionality for tools that perform searches
    with strongly typed inputs and outputs.
    """

    args_schema: Type[T]

    def _run(self, tool_input: Union[str, Dict, T]) -> str:
        """
        Run the tool with the given input.

        Args:
            tool_input (Union[str, Dict, T]): The input to the tool

        Returns:
            str: The formatted results
        """
        self._log_tool_start(tool_input)

        try:
            # Parse the input
            parsed_input = self._parse_input(tool_input)

            # Perform the search
            search_results = self._search(parsed_input)

            # Format the results
            formatted_results = self._format_results(search_results)

            self._log_tool_complete(tool_input, formatted_results)
            return formatted_results

        except Exception as e:
            self._log_tool_error(tool_input, e)
            return f"Error in {self.name}: {str(e)}"

    @abstractmethod
    def _parse_input(self, tool_input: Union[str, Dict, T]) -> T:
        """
        Parse the input to the tool.

        Args:
            tool_input (Union[str, Dict, T]): The input to the tool

        Returns:
            T: The parsed input
        """
        pass

    @abstractmethod
    def _search(self, parsed_input: T) -> List[U]:
        """
        Perform the search with the parsed input.

        Args:
            parsed_input (T): The parsed input

        Returns:
            List[U]: The search results
        """
        pass

    @abstractmethod
    def _format_results(self, results: List[U]) -> str:
        """
        Format the search results.

        Args:
            results (List[U]): The search results

        Returns:
            str: The formatted results
        """
        pass