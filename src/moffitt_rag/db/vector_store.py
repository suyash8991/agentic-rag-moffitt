"""
Vector database functionality using Chroma.

This module provides functions for creating, managing, and
querying the vector database for researcher profiles.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from ..config.config import get_settings
from ..data.models import ResearcherChunk
from ..data.loader import load_all_chunks

# Get settings
settings = get_settings()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embedding_function():
    """
    Get the embedding function for the vector database.

    Returns:
        HuggingFaceEmbeddings: The embedding function
    """
    logger.info(f"Loading embedding model: {settings.embedding_model_name}")
    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)


def create_vector_db(chunks: Optional[List[ResearcherChunk]] = None):
    """
    Create a vector database from researcher profiles.

    Args:
        chunks (Optional[List[ResearcherChunk]], optional):
            The chunks to add to the database. If None, all chunks will be loaded.
            Defaults to None.

    Returns:
        Chroma: The vector database
    """
    # Create the directory for the vector database if it doesn't exist
    os.makedirs(settings.vector_db_dir, exist_ok=True)

    # Get the embedding function
    embedding_function = get_embedding_function()

    # If chunks is None, load all chunks
    if chunks is None:
        chunks = load_all_chunks()

    # Create texts and metadatas
    texts = []
    metadatas = []
    ids = []

    for chunk in chunks:
        texts.append(chunk.text)
        metadatas.append({
            "researcher_id": chunk.researcher_id,
            "name": chunk.name,
            "program": chunk.program,
            "department": chunk.department,
            "research_interests": chunk.research_interests,
            "chunk_type": chunk.chunk_type,
            "profile_url": chunk.profile_url,
        })
        ids.append(chunk.chunk_id)

    # Create the vector database
    logger.info(f"Creating vector database with {len(chunks)} chunks...")
    db = Chroma.from_texts(
        texts=texts,
        embedding=embedding_function,
        metadatas=metadatas,
        ids=ids,
        persist_directory=settings.vector_db_dir,
        collection_name=settings.collection_name,
    )

    logger.info(f"Vector database created with {len(chunks)} chunks")
    return db


def load_vector_db():
    """
    Load the vector database.

    Returns:
        Optional[Chroma]: The vector database, or None if it doesn't exist
    """
    # Check if the vector database exists
    if not os.path.exists(settings.vector_db_dir):
        logger.warning(f"Vector database directory {settings.vector_db_dir} does not exist")
        return None

    # Get the embedding function
    embedding_function = get_embedding_function()

    # Load the vector database
    logger.info(f"Loading vector database from {settings.vector_db_dir}...")

    try:
        db = Chroma(
            persist_directory=settings.vector_db_dir,
            embedding_function=embedding_function,
            collection_name=settings.collection_name,
        )
        logger.info(f"Vector database loaded with {db._collection.count()} chunks")
        return db
    except Exception as e:
        logger.error(f"Failed to load vector database: {e}")
        return None


def get_or_create_vector_db():
    """
    Get the vector database, creating it if it doesn't exist.

    Returns:
        Chroma: The vector database
    """
    db = load_vector_db()

    if db is None:
        logger.info("Vector database not found, creating a new one...")
        db = create_vector_db()

    return db


def create_retriever(db: Optional[Chroma] = None,
                     search_type: str = "similarity",
                     search_kwargs: Optional[Dict[str, Any]] = None) -> VectorStoreRetriever:
    """
    Create a retriever for the vector database.

    Args:
        db (Optional[Chroma], optional):
            The vector database. If None, the database will be loaded.
            Defaults to None.
        search_type (str, optional):
            The search type. One of "similarity", "mmr", or "similarity_score_threshold".
            Defaults to "similarity".
        search_kwargs (Optional[Dict[str, Any]], optional):
            Additional keyword arguments for the search. Defaults to None.

    Returns:
        VectorStoreRetriever: The retriever
    """
    # If db is None, load the database
    if db is None:
        db = get_or_create_vector_db()

    # Set default search kwargs if None
    if search_kwargs is None:
        search_kwargs = {"k": 4}

    # Create the retriever
    retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    return retriever


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

    # Perform the search
    return db.similarity_search(query, k=k, filter=filter)


def similarity_search_with_score(query: str, k: int = 4,
                                filter: Optional[Dict[str, Any]] = None,
                                db: Optional[Chroma] = None):
    """
    Perform a similarity search with scores on the vector database.

    Args:
        query (str): The query to search for
        k (int, optional): The number of results to return. Defaults to 4.
        filter (Optional[Dict[str, Any]], optional):
            Metadata filter. Defaults to None.
        db (Optional[Chroma], optional):
            The vector database. If None, the database will be loaded.
            Defaults to None.

    Returns:
        List[Tuple[Document, float]]: The search results with scores
    """
    # If db is None, load the database
    if db is None:
        db = get_or_create_vector_db()

    # Perform the search
    return db.similarity_search_with_score(query, k=k, filter=filter)


def max_marginal_relevance_search(query: str, k: int = 4,
                                 fetch_k: int = 20,
                                 lambda_mult: float = 0.5,
                                 filter: Optional[Dict[str, Any]] = None,
                                 db: Optional[Chroma] = None):
    """
    Perform a max marginal relevance search on the vector database.

    Args:
        query (str): The query to search for
        k (int, optional): The number of results to return. Defaults to 4.
        fetch_k (int, optional):
            The number of results to fetch before reranking. Defaults to 20.
        lambda_mult (float, optional):
            The diversity parameter. Defaults to 0.5.
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

    # Perform the search
    return db.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
    )