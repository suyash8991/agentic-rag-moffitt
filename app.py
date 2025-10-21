"""
Simple entry point for running the Moffitt Researcher Assistant Streamlit app.

This script sets up the necessary Python path for imports to work correctly.
"""

import os
import sys
from pathlib import Path

def main():
    # Get the project root directory
    root_dir = Path(__file__).parent.absolute()

    # Add the src directory to the Python path
    src_dir = root_dir / "src"
    sys.path.insert(0, str(src_dir))

    # Initialize logging system
    from moffitt_rag.utils.logging import init_logging, configure_exception_logging
    
    # Initialize logging with appropriate settings for Streamlit
    init_logging(
        console_level=None,  # Use environment-based level
        file_level=None,     # Use DEBUG for files
        enable_query_log=True,
        enable_error_log=True,
        enable_structured_logs=True
    )
    
    # Configure global exception logging
    configure_exception_logging()

    # Import and run the Streamlit app
    from moffitt_rag.streamlit.app import main as app_main
    app_main()

if __name__ == "__main__":
    main()