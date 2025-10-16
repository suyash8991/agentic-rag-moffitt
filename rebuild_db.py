"""
Script to rebuild the vector database with updated schema.

This script rebuilds the vector database from scratch to incorporate
the new researcher_name field and other schema changes.
"""

import os
import logging
import shutil
import sys

from src.moffitt_rag.config.config import get_settings
from src.moffitt_rag.db.vector_store import create_vector_db
from src.moffitt_rag.data.loader import load_all_chunks

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

def rebuild_vector_database(backup=True):
    """
    Rebuild the vector database from scratch.

    Args:
        backup (bool, optional): Whether to back up the existing database. Defaults to True.
    """
    vector_db_dir = settings.vector_db_dir

    # Check if the database exists
    if os.path.exists(vector_db_dir):
        if backup:
            # Create a backup
            timestamp = os.path.getmtime(vector_db_dir)
            backup_dir = f"{vector_db_dir}_backup_{int(timestamp)}"

            logger.info(f"Creating backup of existing database at {backup_dir}")
            try:
                shutil.copytree(vector_db_dir, backup_dir)
                logger.info(f"Backup created successfully at {backup_dir}")
            except Exception as e:
                logger.error(f"Failed to create backup: {e}")
                response = input("Continue without backup? (y/n): ").lower()
                if response != 'y':
                    logger.info("Aborting database rebuild")
                    return False

        # Remove existing database
        logger.info(f"Removing existing database at {vector_db_dir}")
        try:
            shutil.rmtree(vector_db_dir)
            logger.info("Existing database removed")
        except Exception as e:
            logger.error(f"Failed to remove existing database: {e}")
            logger.info("Aborting database rebuild")
            return False

    # Load all chunks
    logger.info("Loading all chunks for database rebuild")
    try:
        chunks = load_all_chunks()
        logger.info(f"Loaded {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        logger.info("Aborting database rebuild")
        return False

    # Create new vector database
    logger.info("Creating new vector database")
    try:
        db = create_vector_db(chunks=chunks)
        logger.info(f"Vector database rebuilt successfully with {len(chunks)} chunks")
        return True
    except Exception as e:
        logger.error(f"Failed to create vector database: {e}")
        logger.info("Database rebuild failed")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--no-backup":
        backup = False
    else:
        backup = True

    logger.info("Starting vector database rebuild")
    success = rebuild_vector_database(backup=backup)

    if success:
        logger.info("Vector database rebuild completed successfully")
    else:
        logger.error("Vector database rebuild failed")
        sys.exit(1)