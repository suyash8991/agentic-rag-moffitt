"""
Logging module for the Moffitt Agentic RAG system.

This module provides a central configuration for logging across the system,
including file and console handlers with proper formatting and log rotation.
It also supports structured logging with JSON format for better analysis.
"""

import os
import sys
import json
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

# Global flag to track if logging has been initialized
_LOGGING_INITIALIZED = False

# Dictionary to track initialized structured loggers to prevent duplicates
_INITIALIZED_STRUCTURED_LOGGERS = {}

# Define log levels for different environments
LOG_LEVELS = {
    "dev": logging.INFO,  # Changed from DEBUG to INFO to reduce verbosity
    "test": logging.INFO,
    "prod": logging.WARNING
}

# Define log formats
SIMPLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DETAILED_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
QUERY_FORMAT = "%(asctime)s [%(levelname)s] QUERY %(message)s"
ERROR_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s\n%(exc_info)s"

# Define structured log types
STRUCTURED_LOG_TYPES = [
    "agent",     # Agent-related logs
    "query",     # User queries
    "tool",      # Tool usage
    "vector_db", # Vector database operations
    "llm",       # LLM-related logs
    "api",       # API-related logs
    "ui"         # UI-related logs
]

# Get environment from environment variable or default to "dev"
ENV = os.environ.get("MOFFITT_ENV", "dev").lower()

# JSON formatter for structured logging
class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.

    This formatter is used to generate structured logs in JSON format
    for better analysis and filtering.
    """

    def __init__(self, time_format=None):
        """
        Initialize the formatter with the specified time format.

        Args:
            time_format: Format string for the timestamp. If None, uses ISO format.
        """
        self.time_format = time_format

    def format(self, record):
        """
        Format the record as JSON.

        Args:
            record: The log record to format.

        Returns:
            str: JSON-formatted log entry
        """
        log_data = {
            "timestamp": self.formatTime(record, self.time_format),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "lineno": record.lineno,
            "process": record.process,
            "thread": record.thread
        }

        # Add exception info if available
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "data") and isinstance(record.data, dict):
            log_data.update(record.data)

        # Convert to JSON and return
        return json.dumps(log_data)

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
    enable_structured_logs: bool = True,
    module_specific_levels: Optional[Dict[str, int]] = None,
    force_reinit: bool = False
) -> None:
    """
    Initialize the logging system.

    Args:
        console_level: Logging level for console output. Defaults to the level based on environment.
        file_level: Logging level for file output. Defaults to INFO.
        log_dir: Directory to store log files. Defaults to DEFAULT_LOG_DIR.
        enable_query_log: Whether to enable query logging. Defaults to True.
        enable_error_log: Whether to enable error logging. Defaults to True.
        enable_structured_logs: Whether to enable structured JSON logging. Defaults to True.
        module_specific_levels: Dictionary mapping module names to specific log levels.
        force_reinit: Force reinitialization even if logging was already initialized. Defaults to False.

    Returns:
        None
    """
    # Declare globals at the beginning of the function
    global _LOGGING_INITIALIZED
    global _INITIALIZED_STRUCTURED_LOGGERS

    # Skip initialization if already done, unless forced
    if _LOGGING_INITIALIZED and not force_reinit:
        return

    # If we're reinitializing, clear the structured loggers tracking dictionary
    if force_reinit:
        _INITIALIZED_STRUCTURED_LOGGERS = {}
    # Determine log levels
    default_level = LOG_LEVELS.get(ENV, logging.INFO)
    console_level = console_level if console_level is not None else default_level
    file_level = file_level if file_level is not None else logging.INFO  # Changed from DEBUG to INFO

    # Ensure log directory exists
    log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    log_dir = ensure_log_directory(log_dir)

    # Clean up existing loggers to prevent duplicate handlers
    def clean_logger(logger):
        for handler in list(logger.handlers):  # Use a copy of the list to avoid modification during iteration
            logger.removeHandler(handler)
            # Close the handler to ensure proper resource cleanup
            try:
                handler.close()
            except:
                pass  # Ignore errors during cleanup

    # Clean up root logger
    root_logger = logging.getLogger()
    clean_logger(root_logger)

    # Clean up query logger if it exists
    if enable_query_log:
        query_logger = logging.getLogger("moffitt_rag.queries")
        clean_logger(query_logger)

    # Clean up all structured loggers if using structured logging
    if enable_structured_logs:
        for log_type in STRUCTURED_LOG_TYPES:
            structured_logger = logging.getLogger(f"moffitt_rag.structured.{log_type}")
            clean_logger(structured_logger)

    # Set root logger level to INFO to ensure we don't log DEBUG messages
    root_logger.setLevel(min(console_level, file_level))

    # Create formatters
    simple_formatter = logging.Formatter(SIMPLE_FORMAT)
    detailed_formatter = logging.Formatter(DETAILED_FORMAT)
    query_formatter = logging.Formatter(QUERY_FORMAT)
    error_formatter = logging.Formatter(ERROR_FORMAT)

    # Create console handler with proper encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)
    # Set the stream encoding to UTF-8 if available
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    root_logger.addHandler(console_handler)

    # Create main log file handler with daily rotation
    main_log_file = log_dir / "app.log"
    main_file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=main_log_file,
        when="midnight",
        interval=1,
        backupCount=30,  # Keep 30 days of logs
        encoding='utf-8'  # Use UTF-8 encoding for international characters
    )
    main_file_handler.setLevel(logging.INFO)  # Force INFO level regardless of file_level setting
    main_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(main_file_handler)

    # Create query log file handler with size-based rotation
    if enable_query_log:
        query_log_file = log_dir / "queries.log"
        query_file_handler = logging.handlers.RotatingFileHandler(
            filename=query_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding='utf-8'  # Use UTF-8 encoding for international characters
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
            backupCount=90,  # Keep 90 days of error logs
            encoding='utf-8'  # Use UTF-8 encoding for international characters
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(error_formatter)

        # Create a filter that only allows ERROR and CRITICAL messages
        class ErrorFilter(logging.Filter):
            def filter(self, record):
                return record.levelno >= logging.ERROR

        error_file_handler.addFilter(ErrorFilter())
        root_logger.addHandler(error_file_handler)

    # Create structured log handlers for each component
    if enable_structured_logs:
        # Create a structured logs subdirectory
        structured_log_dir = log_dir / "structured"
        if not structured_log_dir.exists():
            structured_log_dir.mkdir(parents=True, exist_ok=True)

        # Create JSON formatter
        json_formatter = JsonFormatter()

        # Create a handler for each component type
        for log_type in STRUCTURED_LOG_TYPES:
            # Skip if this structured logger was already initialized in this session
            logger_name = f"moffitt_rag.structured.{log_type}"
            if logger_name in _INITIALIZED_STRUCTURED_LOGGERS:
                continue

            # Create the log file with daily rotation
            log_file = structured_log_dir / f"{log_type}.json"
            handler = logging.handlers.TimedRotatingFileHandler(
                filename=log_file,
                when="midnight",
                interval=1,
                backupCount=30,  # Keep 30 days of logs
                encoding='utf-8'  # Use UTF-8 encoding for international characters
            )
            handler.setLevel(logging.INFO)  # Use INFO level for structured logs
            handler.setFormatter(json_formatter)

            # Create a logger for this component
            component_logger = logging.getLogger(logger_name)
            component_logger.setLevel(logging.INFO)
            component_logger.addHandler(handler)
            component_logger.propagate = False  # Don't propagate to root logger

            # Track that we've initialized this logger
            _INITIALIZED_STRUCTURED_LOGGERS[logger_name] = True

        logging.info(f"Structured logging enabled with {len(STRUCTURED_LOG_TYPES)} component loggers")

    # Set up module-specific logging levels
    if module_specific_levels:
        for module_name, level in module_specific_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(level)

    # Log configuration details
    logging.info(f"Logging initialized: console={logging.getLevelName(console_level)}, "
                f"file={logging.getLevelName(file_level)}, env={ENV}, "
                f"log_dir={log_dir}, structured_logs={enable_structured_logs}")

    # Set the global initialization flag
    _LOGGING_INITIALIZED = True


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


# Structured logging helper functions
def log_structured(
    component: str,
    event: str,
    level: int = logging.INFO,
    data: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None
) -> None:
    """
    Log a structured event to the appropriate component logger.

    Args:
        component (str): The component type (must be one of STRUCTURED_LOG_TYPES)
        event (str): The event name/type
        level (int, optional): Log level. Defaults to logging.INFO.
        data (Optional[Dict[str, Any]], optional): Additional data to log.
            Defaults to None.
        message (Optional[str], optional): Optional human-readable message.
            Defaults to None.
    """
    if component not in STRUCTURED_LOG_TYPES:
        # Fall back to a general structured logger if component not recognized
        component = "agent"

    # Get the appropriate logger
    logger = logging.getLogger(f"moffitt_rag.structured.{component}")

    # Prepare the log record
    if data is None:
        data = {}

    # Add the event type to the data
    data["event"] = event

    # Use a default message if none provided
    if message is None:
        message = f"{component.upper()} {event}"

    # Create a log record with the data
    record = logging.LogRecord(
        name=logger.name,
        level=level,
        pathname=__file__,
        lineno=0,  # We don't have a specific line number for this
        msg=message,
        args=(),
        exc_info=None
    )

    # Add the data to the record
    setattr(record, "data", data)

    # Log the record
    logger.handle(record)


def log_agent_event(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
    message: Optional[str] = None
) -> None:
    """Log an agent-related event."""
    log_structured("agent", event, level, data, message)


def log_query_event(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
    message: Optional[str] = None
) -> None:
    """Log a query-related event."""
    log_structured("query", event, level, data, message)


def log_tool_event(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
    message: Optional[str] = None
) -> None:
    """Log a tool usage event."""
    log_structured("tool", event, level, data, message)


def log_vector_db_event(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
    message: Optional[str] = None
) -> None:
    """Log a vector database event."""
    log_structured("vector_db", event, level, data, message)


def log_llm_event(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
    message: Optional[str] = None
) -> None:
    """Log an LLM-related event."""
    log_structured("llm", event, level, data, message)


def log_api_event(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
    message: Optional[str] = None
) -> None:
    """Log an API-related event."""
    log_structured("api", event, level, data, message)


def log_ui_event(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
    message: Optional[str] = None
) -> None:
    """Log a UI-related event."""
    log_structured("ui", event, level, data, message)


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

        # Log to structured error log as well
        log_structured("agent", "uncaught_exception", logging.CRITICAL, {
            "exception_type": str(exc_type.__name__),
            "exception_value": str(exc_value),
            "traceback": traceback.format_exception(exc_type, exc_value, exc_traceback)
        })

        # Call the default exception handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    # Set the exception hook
    sys.excepthook = log_uncaught_exception