"""
Logging module for the Moffitt Agentic RAG system.

This module provides a central configuration for logging across the system,
including file and console handlers with proper formatting and log rotation.
"""

import os
import sys
import logging
import logging.handlers
import datetime
import traceback
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, Any, Callable, Type, Union, List, Tuple

# Define log directory relative to project root
# By default, logs will be stored in a 'logs' directory at the project root
DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent.parent / "logs"

# Define log levels for different environments
LOG_LEVELS = {
    "dev": logging.DEBUG,
    "test": logging.INFO,
    "prod": logging.WARNING
}

# Define log formats
SIMPLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DETAILED_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
QUERY_FORMAT = "%(asctime)s [%(levelname)s] QUERY %(message)s"
ERROR_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s\n%(exc_info)s"

# Get environment from environment variable or default to "dev"
ENV = os.environ.get("MOFFITT_ENV", "dev").lower()

# Create the logs directory if it doesn't exist
def ensure_log_directory(log_dir: Path = DEFAULT_LOG_DIR) -> Path:
    """
    Ensure the log directory exists.

    Args:
        log_dir (Path, optional): Directory to store log files.
            Defaults to DEFAULT_LOG_DIR.

    Returns:
        Path: Path to the log directory
    """
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

# Initialize the logging configuration
def init_logging(
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    log_dir: Optional[Union[str, Path]] = None,
    enable_query_log: bool = True,
    enable_error_log: bool = True,
    module_specific_levels: Optional[Dict[str, int]] = None
) -> None:
    """
    Initialize the logging system.

    This function sets up the root logger with console and file handlers,
    and configures additional handlers for query and error logs.

    Args:
        console_level (Optional[int], optional): Logging level for console output.
            Defaults to the level based on environment.
        file_level (Optional[int], optional): Logging level for file output.
            Defaults to DEBUG.
        log_dir (Optional[Union[str, Path]], optional): Directory to store log files.
            Defaults to DEFAULT_LOG_DIR.
        enable_query_log (bool, optional): Whether to enable query logging.
            Defaults to True.
        enable_error_log (bool, optional): Whether to enable error logging.
            Defaults to True.
        module_specific_levels (Optional[Dict[str, int]], optional):
            Dictionary mapping module names to specific log levels.
            Defaults to None.
    """
    # Determine log levels
    default_level = LOG_LEVELS.get(ENV, logging.INFO)
    console_level = console_level if console_level is not None else default_level
    file_level = file_level if file_level is not None else logging.DEBUG

    # Ensure log directory exists
    log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    log_dir = ensure_log_directory(log_dir)

    # Reset root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # Set root logger level to the lowest of all handlers to ensure all messages
    # are propagated to the appropriate handlers
    root_logger.setLevel(min(console_level, file_level, logging.DEBUG))

    # Create formatters
    simple_formatter = logging.Formatter(SIMPLE_FORMAT)
    detailed_formatter = logging.Formatter(DETAILED_FORMAT)
    query_formatter = logging.Formatter(QUERY_FORMAT)
    error_formatter = logging.Formatter(ERROR_FORMAT)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # Create main log file handler with daily rotation
    main_log_file = log_dir / "app.log"
    main_file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=main_log_file,
        when="midnight",
        interval=1,
        backupCount=30  # Keep 30 days of logs
    )
    main_file_handler.setLevel(file_level)
    main_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(main_file_handler)

    # Create query log file handler with size-based rotation
    if enable_query_log:
        query_log_file = log_dir / "queries.log"
        query_file_handler = logging.handlers.RotatingFileHandler(
            filename=query_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        query_file_handler.setLevel(logging.INFO)
        query_file_handler.setFormatter(query_formatter)
        query_logger = logging.getLogger("moffitt_rag.queries")
        query_logger.setLevel(logging.INFO)
        query_logger.addHandler(query_file_handler)
        query_logger.propagate = False  # Don't propagate to root logger to avoid duplication

    # Create error log file handler with daily rotation
    if enable_error_log:
        error_log_file = log_dir / "errors.log"
        error_file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=error_log_file,
            when="midnight",
            interval=1,
            backupCount=90  # Keep 90 days of error logs
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(error_formatter)

        # Create a filter that only allows ERROR and CRITICAL messages
        class ErrorFilter(logging.Filter):
            def filter(self, record):
                return record.levelno >= logging.ERROR

        error_file_handler.addFilter(ErrorFilter())
        root_logger.addHandler(error_file_handler)

    # Set up module-specific logging levels
    if module_specific_levels:
        for module_name, level in module_specific_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(level)

    # Log configuration details
    logging.info(f"Logging initialized: console={logging.getLevelName(console_level)}, "
                f"file={logging.getLevelName(file_level)}, env={ENV}, "
                f"log_dir={log_dir}")


# Utility function to get a logger for a module
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name (Optional[str], optional): Logger name.
            If None, uses the caller's module name. Defaults to None.

    Returns:
        logging.Logger: The logger instance
    """
    if name is None:
        # Get the caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        name = module.__name__ if module else "__main__"

    return logging.getLogger(name)


# Context manager for query logging
class QueryLogger:
    """
    Context manager for logging queries and their results.

    Example:
        with QueryLogger("Who is John Doe?") as qlog:
            result = agent.invoke({"input": "Who is John Doe?"})
            qlog.set_result(result)
            # Additional context can be added with qlog.add_context(key, value)

    This will log the query, execution time, and result to the query log.
    """

    def __init__(self, query: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the QueryLogger.

        Args:
            query (str): The query being executed
            context (Optional[Dict[str, Any]], optional): Additional context to log.
                Defaults to None.
        """
        self.query = query
        self.context = context or {}
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
        self.logger = logging.getLogger("moffitt_rag.queries")

    def __enter__(self):
        """Enter the context manager and log the query start."""
        self.start_time = datetime.datetime.now()
        self.logger.info(f"START QUERY: {self.query}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and log the query end."""
        self.end_time = datetime.datetime.now()
        execution_time = (self.end_time - self.start_time).total_seconds()

        if exc_type is not None:
            # If an exception occurred, log it
            self.error = str(exc_val)
            self.logger.error(
                f"ERROR QUERY: {self.query} | Time: {execution_time:.2f}s | "
                f"Error: {self.error}"
            )
            # Log traceback to error log
            error_logger = logging.getLogger("moffitt_rag.errors")
            error_logger.error(
                f"Error executing query '{self.query}': {self.error}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
        else:
            # Log the successful completion
            result_str = str(self.result)[:500] + "..." if self.result and len(str(self.result)) > 500 else str(self.result)
            self.logger.info(
                f"FINISH QUERY: {self.query} | Time: {execution_time:.2f}s | "
                f"Result: {result_str}"
            )

        # Log additional context if present
        if self.context:
            self.logger.info(f"QUERY CONTEXT: {self.query} | {self.context}")

        # Don't suppress exceptions
        return False

    def set_result(self, result: Any) -> None:
        """
        Set the result of the query.

        Args:
            result (Any): The result to log
        """
        self.result = result

    def add_context(self, key: str, value: Any) -> None:
        """
        Add additional context to the log.

        Args:
            key (str): Context key
            value (Any): Context value
        """
        self.context[key] = value


# Decorator to log function calls
def log_function_call(level: int = logging.DEBUG):
    """
    Decorator to log function calls.

    Args:
        level (int, optional): Logging level. Defaults to logging.DEBUG.

    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.log(
                level,
                f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}"
            )
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} returned: {result}")
                return result
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {e}",
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


# Exception logging helper
def log_exception(
    exception: Exception,
    logger: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
    message: Optional[str] = None,
    include_traceback: bool = True
) -> None:
    """
    Log an exception.

    Args:
        exception (Exception): The exception to log
        logger (Optional[logging.Logger], optional):
            Logger to use. Defaults to root logger.
        level (int, optional): Logging level. Defaults to logging.ERROR.
        message (Optional[str], optional):
            Additional message to include. Defaults to None.
        include_traceback (bool, optional):
            Whether to include traceback. Defaults to True.
    """
    if logger is None:
        logger = logging.getLogger()

    if message is None:
        message = f"Exception: {type(exception).__name__}: {str(exception)}"

    if include_traceback:
        logger.log(
            level,
            message,
            exc_info=exception
        )
    else:
        logger.log(level, message)


# Configure global exception hook to log unhandled exceptions
def configure_exception_logging():
    """Configure global exception hook to log unhandled exceptions."""
    def log_uncaught_exception(exc_type, exc_value, exc_traceback):
        """Log uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Call the default excepthook for KeyboardInterrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the exception
        error_logger = logging.getLogger("moffitt_rag.errors")
        error_logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

        # Call the default exception handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    # Set the exception hook
    sys.excepthook = log_uncaught_exception