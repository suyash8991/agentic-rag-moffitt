"""
Data directory initialization script.

This script creates all required data directories and .gitkeep files
to ensure the directory structure is preserved in version control.
"""

import os
from pathlib import Path


def create_directory_structure():
    """Create required data directories."""

    # Define directory structure
    base_dir = Path(__file__).parent.parent
    directories = [
        base_dir / "data" / "vector_db",
        base_dir / "data" / "processed",
        base_dir / "data" / "markdown",
        base_dir / "data" / "raw_html",
    ]

    print("Creating data directory structure...")

    for directory in directories:
        # Create directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")

        # Create .gitkeep file to preserve empty directories in git
        gitkeep_file = directory / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
            print(f"  └─ Added .gitkeep")

    print("\n✅ Data directory structure initialized successfully!")
    print("\nDirectory structure:")
    print("data/")
    for directory in directories:
        rel_path = directory.relative_to(base_dir / "data")
        print(f"  └─ {rel_path}/")


if __name__ == "__main__":
    create_directory_structure()
