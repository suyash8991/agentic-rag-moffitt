"""
Logging utilities for the Moffitt Agentic RAG system.

This module provides enhanced logging functionality including structured
event logging for better observability and analytics.
"""

import logging
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


# Configure log directory
LOG_DIR = Path(__file__).parent.parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, 'event'):
            log_data['event'] = record.event
        if hasattr(record, 'data'):
            log_data['data'] = record.data

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_logger(name: str, log_file: Optional[str] = "app.log") -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: The name of the logger (typically __name__)
        log_file: Optional log file name. If None, logs to stdout only.

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if handlers haven't been added yet
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Console handler with standard formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler with structured JSON formatting
        if log_file:
            file_path = LOG_DIR / log_file
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(StructuredFormatter())
            logger.addHandler(file_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


def log_tool_event(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    level: str = "INFO",
    logger_name: str = "moffitt_rag.tools"
) -> None:
    """
    Log a structured tool event.

    Args:
        event: Event name/type (e.g., "researcher_search_start")
        data: Optional dictionary of event data
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Name of the logger to use

    Example:
        log_tool_event("researcher_search_start", {
            "query": "cancer evolution",
            "alpha": 0.7
        })
    """
    logger = get_logger(logger_name)

    # Get the logging level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create a log record with extra attributes
    message = f"Tool Event: {event}"

    # Add structured data to the log record
    extra = {
        'event': event,
        'data': data or {}
    }

    # Log with the structured data
    logger.log(log_level, message, extra=extra)


def log_search_event(
    query: str,
    search_type: str,
    alpha: float,
    result_count: int,
    logger_name: str = "moffitt_rag.search"
) -> None:
    """
    Log a search event with standardized fields.

    Args:
        query: The search query
        search_type: Type of search ("name" or "topic")
        alpha: Alpha parameter used
        result_count: Number of results returned
        logger_name: Name of the logger to use
    """
    log_tool_event(
        event="search_executed",
        data={
            "query": query[:100],  # Truncate long queries
            "search_type": search_type,
            "alpha": alpha,
            "result_count": result_count
        },
        logger_name=logger_name
    )


def log_error_event(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    logger_name: str = "moffitt_rag.errors"
) -> None:
    """
    Log an error event with context.

    Args:
        error: The exception that occurred
        context: Optional context data about the error
        logger_name: Name of the logger to use
    """
    log_tool_event(
        event="error_occurred",
        data={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        },
        level="ERROR",
        logger_name=logger_name
    )
