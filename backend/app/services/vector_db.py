"""
Vector database services.

This module provides functions for interacting with the vector database,
including creating, loading, and searching the database.
"""

import os
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
from fastapi import BackgroundTasks

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from ..core.config import settings
from ..models.researcher import ResearcherProfileSummary
from .vector_db_builder import rebuild_vector_database as _rebuild_db_core

# Setup logging
logger = logging.getLogger(__name__)

# In-memory cache for database statistics
_db_stats = {
    "total_researchers": 0,
    "total_chunks": 0,
    "last_updated": None,
    "status": "not_initialized"
}

# Active rebuild tasks
_active_tasks: Dict[str, Dict[str, Any]] = {}


def get_embedding_function():
    """
    Get the embedding function for the vector database.

    Returns:
        HuggingFaceEmbeddings: The embedding function
    """
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME, model_kwargs={"trust_remote_code": "true"})


def load_vector_db():
    """
    Load the vector database.

    Returns:
        Optional[Chroma]: The vector database, or None if it doesn't exist
    """
    # Check if the vector database exists
    if not os.path.exists(settings.VECTOR_DB_DIR):
        logger.warning(f"Vector database directory {settings.VECTOR_DB_DIR} does not exist")
        return None

    # Get the embedding function
    embedding_function = get_embedding_function()

    # Load the vector database
    logger.info(f"Loading vector database from {settings.VECTOR_DB_DIR}...")

    try:
        db = Chroma(
            persist_directory=settings.VECTOR_DB_DIR,
            embedding_function=embedding_function,
            collection_name=settings.COLLECTION_NAME,
        )
        count = db._collection.count()
        logger.info(f"Vector database loaded with {count} chunks")

        # Update stats
        _db_stats["total_chunks"] = count
        _db_stats["status"] = "loaded"
        _db_stats["last_updated"] = datetime.now().isoformat()

        return db
    except Exception as e:
        logger.error(f"Failed to load vector database: {e}")
        _db_stats["status"] = "error"
        return None


def get_or_create_vector_db():
    """
    Get the vector database, creating it if it doesn't exist.

    Returns:
        Chroma: The vector database
    """
    db = load_vector_db()

    if db is None:
        logger.warning("Vector database not found or failed to load")
        _db_stats["status"] = "not_found"

    return db


def similarity_search(query: str, k: int = 4,
                      filter: Optional[Dict[str, Any]] = None,
                      db: Optional[Chroma] = None):
    """
    Perform a similarity search on the vector database.

    Args:
        query (str): The query to search for
        k (int, optional): The number of results to return. Defaults to 4.
        filter (Optional[Dict[str, Any]], optional):
            Metadata filter. Defaults to None.
        db (Optional[Chroma], optional):
            The vector database. If None, the database will be loaded.
            Defaults to None.

    Returns:
        List[Document]: The search results
    """
    # If db is None, load the database
    if db is None:
        db = get_or_create_vector_db()
        if db is None:
            logger.error("Failed to load or create vector database")
            # Return a list with a fallback Document
            from langchain_core.documents import Document
            return [Document(page_content="No vector database available. Please contact system administrator.", metadata={"researcher_name": "System", "program": "N/A", "department": "N/A", "profile_url": ""})]

    # Perform the search
    try:
        results = db.similarity_search(query, k=k, filter=filter)

        # If no results were found, return a helpful message
        if not results:
            logger.warning(f"No results found for query: {query}")
            from langchain_core.documents import Document
            return [Document(page_content=f"No results found for query: {query}. Try another search term.", metadata={"researcher_name": "System", "program": "N/A", "department": "N/A", "profile_url": ""})]

        return results
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        # Return a list with an error Document
        from langchain_core.documents import Document
        return [Document(page_content=f"Error searching the database: {str(e)}", metadata={"researcher_name": "System", "program": "N/A", "department": "N/A", "profile_url": ""})]


def rebuild_vector_database(force: bool = False, background_tasks: BackgroundTasks = None):
    """
    Rebuild the vector database.

    This function will rebuild the vector database by loading all researcher profiles,
    creating chunks, and storing them in the database. It can run in the background
    if a BackgroundTasks instance is provided.

    Args:
        force (bool, optional): Force rebuild even if the database exists. Defaults to False.
        background_tasks (BackgroundTasks, optional): FastAPI BackgroundTasks. Defaults to None.

    Returns:
        str: Task ID for tracking the rebuild process
    """
    # Generate a task ID
    task_id = str(uuid.uuid4())

    # Create task info
    task_info = {
        "task_id": task_id,
        "status": "pending",
        "start_time": datetime.now().isoformat(),
        "progress": 0.0,
        "force": force
    }

    # Store the task info
    _active_tasks[task_id] = task_info

    # Define the rebuild function
    def _rebuild_db():
        try:
            # Update task status
            _active_tasks[task_id]["status"] = "running"

            # Check if the vector database directory exists
            if not os.path.exists(settings.VECTOR_DB_DIR):
                os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)

            # If the database exists and force is False, don't rebuild
            if os.path.exists(os.path.join(settings.VECTOR_DB_DIR, "chroma.sqlite3")) and not force:
                logger.info("Vector database already exists and force=False, skipping rebuild")
                _active_tasks[task_id]["status"] = "skipped"
                return

            # Update task progress
            _active_tasks[task_id]["progress"] = 0.1

            # Progress callback to update task progress
            def update_progress(progress: float):
                _active_tasks[task_id]["progress"] = progress
                logger.debug(f"Rebuild progress: {progress * 100:.1f}%")

            # Run the actual rebuild using shared module
            logger.info("Starting vector database rebuild...")
            processed_dir = Path(settings.PROCESSED_DATA_DIR)
            vector_db_dir = Path(settings.VECTOR_DB_DIR)

            success = _rebuild_db_core(
                processed_dir=processed_dir,
                vector_db_dir=vector_db_dir,
                collection_name=settings.COLLECTION_NAME,
                backup=True,  # Always backup in backend
                force=force,
                progress_callback=update_progress
            )

            if not success:
                raise Exception("Rebuild failed - check logs for details")

            # Update stats from the actual database
            db = load_vector_db()
            if db:
                collection = db._collection
                chunk_count = collection.count()

                # Get unique researchers from metadata
                # Note: This is a rough count, could be improved with actual metadata query
                unique_researchers = set()
                try:
                    # Get all metadata to count unique researchers
                    results = collection.get(include=["metadatas"])
                    for metadata in results.get("metadatas", []):
                        if metadata and "researcher_name" in metadata:
                            unique_researchers.add(metadata["researcher_name"])
                except Exception as e:
                    logger.warning(f"Could not count unique researchers: {e}")

                _db_stats["total_researchers"] = len(unique_researchers) if unique_researchers else 0
                _db_stats["total_chunks"] = chunk_count
            else:
                # Fallback to basic estimates
                _db_stats["total_researchers"] = 0
                _db_stats["total_chunks"] = 0
            _db_stats["last_updated"] = datetime.now().isoformat()
            _db_stats["status"] = "ready"

            # Update task status
            _active_tasks[task_id]["status"] = "completed"
            _active_tasks[task_id]["progress"] = 1.0
            _active_tasks[task_id]["end_time"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Error rebuilding vector database: {e}")
            # Update task status
            _active_tasks[task_id]["status"] = "error"
            _active_tasks[task_id]["error"] = str(e)
            _db_stats["status"] = "error"

    # If background_tasks is provided, run in the background
    if background_tasks is not None:
        background_tasks.add_task(_rebuild_db)
    else:
        # Otherwise, run synchronously
        _rebuild_db()

    return task_id


def get_database_stats():
    """
    Get vector database statistics.

    Returns:
        Dict[str, Any]: Database statistics
    """
    # If the database is not loaded yet, try to load it
    if _db_stats["status"] == "not_initialized":
        db = get_or_create_vector_db()
        if db is not None:
            _db_stats["total_chunks"] = db._collection.count()
            _db_stats["status"] = "loaded"
            _db_stats["last_updated"] = datetime.now().isoformat()

    return _db_stats