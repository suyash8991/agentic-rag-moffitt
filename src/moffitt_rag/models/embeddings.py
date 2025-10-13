"""
Embedding model functionality for the Moffitt RAG system.

This module provides functions for creating and using embedding models.
"""

import os
import logging
from typing import List, Optional, Dict, Any

from langchain_community.embeddings import HuggingFaceEmbeddings

from ..config.config import get_settings

# Get settings
settings = get_settings()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    # Create the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    return embedding_model


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
    embeddings = embedding_model.embed_documents(texts)

    return embeddings


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

    # Generate embedding
    embedding = embedding_model.embed_query(query)

    return embedding


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