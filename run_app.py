"""
Runner script for Moffitt Cancer Center Researcher Assistant.

This script configures the Python path and runs the Streamlit application.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit application with the correct import paths."""
    # Get the current directory (project root)
    project_root = Path(__file__).parent.absolute()

    # Add src directory to Python path
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

    # Initialize logging system
    from utils.logging import init_logging, configure_exception_logging, get_logger
    
    # Initialize logging
    init_logging(
        console_level=None,  # Use environment-based level
        file_level=None,     # Use DEBUG for files
        enable_query_log=True,
        enable_error_log=True,
        enable_structured_logs=True
    )
    
    # Configure global exception logging
    configure_exception_logging()
    
    # Get logger for this script
    logger = get_logger(__name__)

    # Path to the app file
    app_path = project_root / "src" / "web_app" / "app.py"

    # Check if the app file exists
    if not app_path.exists():
        logger.error(f"App file not found at {app_path}")
        return 1

    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{str(src_path)}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(src_path)

    # Run the Streamlit app with the correct Python path
    logger.info(f"Starting Streamlit app from: {app_path}")
    logger.info(f"Python path includes: {src_path}")

    result = subprocess.run(["streamlit", "run", str(app_path)], env=env)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())