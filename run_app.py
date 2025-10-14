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

    # Path to the app file
    app_path = project_root / "src" / "moffitt_rag" / "streamlit" / "app.py"

    # Check if the app file exists
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        return 1

    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{str(src_path)}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(src_path)

    # Run the Streamlit app with the correct Python path
    print(f"Starting Streamlit app from: {app_path}")
    print(f"Python path includes: {src_path}")

    result = subprocess.run(["streamlit", "run", str(app_path)], env=env)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())