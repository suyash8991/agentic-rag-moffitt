"""
Run script for the Moffitt Agentic RAG Streamlit application.

This script provides a simple way to launch the Streamlit application
from the command line.

Usage:
    python run_streamlit.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()

    # Make sure the current directory is in the Python path
    sys.path.insert(0, str(current_dir))

    # Streamlit app path
    app_path = current_dir / "src" / "moffitt_rag" / "streamlit" / "app.py"

    # Check if the app file exists
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        return 1

    # Launch the Streamlit app
    print(f"Launching Streamlit app from {app_path}")
    result = subprocess.run(["streamlit", "run", str(app_path)])

    return result.returncode

if __name__ == "__main__":
    sys.exit(main())