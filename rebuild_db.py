"""
CLI script to rebuild the vector database from researcher profiles.

This script provides a command-line interface for rebuilding the vector database.
The core rebuild logic is in backend/app/services/vector_db_builder.py and is
shared with the backend API.

Usage:
    python rebuild_db.py [--no-backup] [--force]

Options:
    --no-backup: Skip backing up the existing database
    --force: Force rebuild even if database exists
"""

import sys
import logging
import argparse
from pathlib import Path

# Add backend to path so we can import its modules
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from app.core.config import settings
from app.services.vector_db_builder import rebuild_vector_database

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the rebuild CLI script."""
    parser = argparse.ArgumentParser(
        description="Rebuild the Moffitt RAG vector database from researcher profiles"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up the existing database"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if database exists"
    )

    args = parser.parse_args()

    # Get paths - resolve them relative to the script location (project root)
    project_root = Path(__file__).parent

    # Use absolute paths from project root (CLI override for backend paths)
    processed_dir = project_root / "data" / "processed"
    vector_db_dir = project_root / "data" / "vector_db"
    collection_name = settings.COLLECTION_NAME

    logger.info("=" * 80)
    logger.info("Moffitt RAG Vector Database Rebuild (CLI)")
    logger.info("=" * 80)
    logger.info(f"Processed data directory: {processed_dir}")
    logger.info(f"Vector database directory: {vector_db_dir}")
    logger.info(f"Collection name: {collection_name}")
    logger.info(f"Backup enabled: {not args.no_backup}")
    logger.info(f"Force rebuild: {args.force}")
    logger.info("=" * 80)

    # Validate paths
    if not processed_dir.exists():
        logger.error(f"Processed data directory not found: {processed_dir}")
        sys.exit(1)

    # Run rebuild using shared module
    success = rebuild_vector_database(
        processed_dir=processed_dir,
        vector_db_dir=vector_db_dir,
        collection_name=collection_name,
        backup=not args.no_backup,
        force=args.force
    )

    if success:
        logger.info("✓ Vector database rebuild completed successfully!")
        sys.exit(0)
    else:
        logger.error("✗ Vector database rebuild failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
