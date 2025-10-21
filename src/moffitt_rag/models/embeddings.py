"""
Embedding model functionality for the Moffitt RAG system.

This module provides functions for creating and using embedding models.
"""

import os
from typing import List, Optional, Dict, Any

from langchain_community.embeddings import HuggingFaceEmbeddings

from ..config.config import get_settings
from ..utils.logging import get_logger, log_vector_db_event

# Get settings
settings = get_settings()

# Get logger for this module
logger = get_logger(__name__)


def get_embedding_model(model_name: Optional[str] = None):
    """
    Get an embedding model.

    Args:
        model_name (Optional[str], optional):
            The name of the embedding model to use.
            If None, the model from settings will be used.
            Defaults to None.

    Returns:
        HuggingFaceEmbeddings: The embedding model
    """
    if model_name is None:
        model_name = settings.embedding_model_name

    logger.info(f"Loading embedding model: {model_name}")

    # Log structured event for embedding model loading
    log_vector_db_event("embedding_model_loading", {
        "model_name": model_name
    })

    try:
        # Create the embedding model
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)

        # Log successful loading
        log_vector_db_event("embedding_model_loaded", {
            "model_name": model_name
        })

        return embedding_model
    except Exception as e:
        # Log error loading model
        log_vector_db_event("embedding_model_error", {
            "model_name": model_name,
            "error": str(e)
        })
        raise


def generate_embeddings(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts (List[str]): The texts to generate embeddings for
        model_name (Optional[str], optional):
            The name of the embedding model to use.
            If None, the model from settings will be used.
            Defaults to None.

    Returns:
        List[List[float]]: The embeddings
    """
    # Get the embedding model
    embedding_model = get_embedding_model(model_name)

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} texts")

    # Log structured event for embedding generation start
    log_vector_db_event("embedding_generation_start", {
        "text_count": len(texts),
        "model_name": model_name or settings.embedding_model_name
    })

    try:
        start_time = __import__('time').time()
        embeddings = embedding_model.embed_documents(texts)
        elapsed_time = __import__('time').time() - start_time

        # Log successful embedding generation
        log_vector_db_event("embedding_generation_complete", {
            "text_count": len(texts),
            "embedding_count": len(embeddings),
            "elapsed_time_ms": round(elapsed_time * 1000),
            "model_name": model_name or settings.embedding_model_name
        })

        return embeddings
    except Exception as e:
        # Log error generating embeddings
        log_vector_db_event("embedding_generation_error", {
            "text_count": len(texts),
            "model_name": model_name or settings.embedding_model_name,
            "error": str(e)
        })
        raise


def embed_query(query: str, model_name: Optional[str] = None) -> List[float]:
    """
    Generate an embedding for a query.

    Args:
        query (str): The query to generate an embedding for
        model_name (Optional[str], optional):
            The name of the embedding model to use.
            If None, the model from settings will be used.
            Defaults to None.

    Returns:
        List[float]: The embedding
    """
    # Get the embedding model
    embedding_model = get_embedding_model(model_name)

    # Log structured event for query embedding start
    log_vector_db_event("query_embedding_start", {
        "query_length": len(query),
        "model_name": model_name or settings.embedding_model_name
    })

    try:
        start_time = __import__('time').time()

        # Generate embedding
        embedding = embedding_model.embed_query(query)

        elapsed_time = __import__('time').time() - start_time

        # Log successful query embedding
        log_vector_db_event("query_embedding_complete", {
            "query_length": len(query),
            "embedding_size": len(embedding),
            "elapsed_time_ms": round(elapsed_time * 1000),
            "model_name": model_name or settings.embedding_model_name
        })

        return embedding
    except Exception as e:
        # Log error generating query embedding
        log_vector_db_event("query_embedding_error", {
            "query_length": len(query),
            "model_name": model_name or settings.embedding_model_name,
            "error": str(e)
        })
        raise


def embedding_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate the cosine similarity between two embeddings.

    Args:
        embedding1 (List[float]): The first embedding
        embedding2 (List[float]): The second embedding

    Returns:
        float: The cosine similarity (between -1 and 1, higher is more similar)
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Convert to numpy arrays
    vec1 = np.array(embedding1).reshape(1, -1)
    vec2 = np.array(embedding2).reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(vec1, vec2)[0][0]

    return similarity