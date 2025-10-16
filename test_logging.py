#!/usr/bin/env python
"""
Test script for the logging system.

Run this script to verify that the logging system is working correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import our logging module
from src.moffitt_rag.utils import logging as mlog

def main():
    """Test the logging system."""
    # Initialize logging
    print("Initializing logging system...")
    mlog.init_logging()

    # Get a logger
    logger = mlog.get_logger("test_logging")

    # Log some messages
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")

    # Test exception logging
    try:
        raise ValueError("This is a test exception")
    except Exception as e:
        mlog.log_exception(e, logger, message="Caught test exception")

    # Test query logging
    try:
        print("Testing query logging...")
        with mlog.QueryLogger("Who is John Doe?", context={"source": "test"}) as qlog:
            # Simulate some processing
            import time
            time.sleep(1)
            result = {"name": "John Doe", "info": "Test researcher"}
            qlog.set_result(result)
            qlog.add_context("response_time", "1s")
    except Exception as e:
        print(f"Error testing query logging: {e}")

    # Test failed query
    try:
        print("Testing failed query logging...")
        with mlog.QueryLogger("Who is Jane Doe?") as qlog:
            # Simulate a failure
            raise RuntimeError("Test failure")
    except Exception:
        pass  # The QueryLogger should log the exception

    # Test function call logging
    @mlog.log_function_call()
    def test_function(arg1, arg2, kwarg1=None):
        """Test function to demonstrate function call logging."""
        return f"{arg1} + {arg2} + {kwarg1}"

    print("Testing function call logging...")
    test_function("a", "b", kwarg1="c")

    # Configure exception logging
    mlog.configure_exception_logging()

    print("\nTest completed. Check the following log files:")
    log_dir = project_root / "logs"
    print(f"  - {log_dir / 'app.log'}")
    print(f"  - {log_dir / 'queries.log'}")
    print(f"  - {log_dir / 'errors.log'}")

if __name__ == "__main__":
    main()