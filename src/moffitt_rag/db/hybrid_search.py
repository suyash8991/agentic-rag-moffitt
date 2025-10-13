"""
Hybrid search functionality for the Moffitt RAG system.

This module provides functions for performing hybrid searches that combine
vector similarity with keyword-based retrieval.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .vector_store import get_or_create_vector_db, similarity_search_with_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def keyword_search(query: str, texts: List[str], metadata: List[Dict[str, Any]],
                  k: int = 4) -> List[Tuple[Document, float]]:
    """
    Perform a simple keyword-based search.

    Args:
        query (str): The query to search for
        texts (List[str]): The texts to search in
        metadata (List[Dict[str, Any]]): The metadata for each text
        k (int, optional): The number of results to return. Defaults to 4.

    Returns:
        List[Tuple[Document, float]]: The search results with scores
    """
    # Create a list of documents
    docs = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadata)]

    # Split the query into keywords
    keywords = re.findall(r'\w+', query.lower())

    # Calculate scores for each document
    scored_docs = []
    for doc in docs:
        text = doc.page_content.lower()

        # Count the number of keywords that appear in the text
        score = sum(1 for keyword in keywords if keyword in text)

        # Normalize the score by the number of keywords
        if keywords:
            score = score / len(keywords)
        else:
            score = 0.0

        scored_docs.append((doc, score))

    # Sort by score (descending) and take the top k
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:k]


def hybrid_search(query: str, k: int = 4, alpha: float = 0.5,
                 filter: Optional[Dict[str, Any]] = None):
    """
    Perform a hybrid search combining vector similarity and keyword matching.

    Args:
        query (str): The query to search for
        k (int, optional): The number of results to return. Defaults to 4.
        alpha (float, optional): Weight for semantic search (0-1). Defaults to 0.5.
            Higher values give more weight to semantic search, lower to keyword search.
        filter (Optional[Dict[str, Any]], optional): Metadata filter. Defaults to None.

    Returns:
        List[Document]: The search results
    """
    # Get the vector database
    db = get_or_create_vector_db()

    # Get all chunks from the database to perform keyword search
    # This is a simplification - in a real-world scenario with a large database,
    # you would need a more efficient approach
    results = db.get()
    texts = results['documents']
    ids = results['ids']
    metadatas = results['metadatas']

    # Perform semantic search
    logger.info(f"Performing semantic search for query: {query}")
    semantic_results = similarity_search_with_score(query, k=k*2, filter=filter, db=db)

    # Convert to dictionary for easier lookup
    semantic_scores = {doc.metadata.get('chunk_id', i): score
                      for i, (doc, score) in enumerate(semantic_results)}

    # Perform keyword search
    logger.info(f"Performing keyword search for query: {query}")
    keyword_results = keyword_search(query, texts, metadatas, k=k*2)

    # Convert to dictionary for easier lookup
    keyword_scores = {doc.metadata.get('chunk_id', i): score
                     for i, (doc, score) in enumerate(keyword_results)}

    # Combine the results
    combined_scores = {}

    # Add all documents from both results
    for doc, score in semantic_results:
        doc_id = doc.metadata.get('chunk_id')
        combined_scores[doc_id] = {
            'doc': doc,
            'semantic_score': score,
            'keyword_score': keyword_scores.get(doc_id, 0.0)
        }

    for doc, score in keyword_results:
        doc_id = doc.metadata.get('chunk_id')
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                'doc': doc,
                'semantic_score': semantic_scores.get(doc_id, 0.0),
                'keyword_score': score
            }

    # Calculate combined scores
    for doc_id, scores in combined_scores.items():
        # Normalize semantic score (higher is better)
        semantic_score = scores['semantic_score']
        # Note: for Chroma, lower distances are better, so we invert
        if semantic_score > 0:
            semantic_score = 1.0 / semantic_score

        # Combined score is weighted average
        combined_scores[doc_id]['combined_score'] = (
            alpha * semantic_score + (1 - alpha) * scores['keyword_score']
        )

    # Sort by combined score (descending)
    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )

    # Return the top k documents
    return [item['doc'] for item in sorted_results[:k]]


class HybridRetriever(BaseRetriever):
    """
    A retriever that combines semantic search with keyword search.

    This retriever performs both semantic and keyword searches and
    combines the results using a weighted average.
    """

    def __init__(self, k: int = 4, alpha: float = 0.5,
                filter: Optional[Dict[str, Any]] = None):
        """
        Initialize the hybrid retriever.

        Args:
            k (int, optional): The number of results to return. Defaults to 4.
            alpha (float, optional): Weight for semantic search (0-1). Defaults to 0.5.
                Higher values give more weight to semantic search, lower to keyword search.
            filter (Optional[Dict[str, Any]], optional): Metadata filter. Defaults to None.
        """
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.filter = filter

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for the query.

        Args:
            query (str): The query to search for

        Returns:
            List[Document]: The search results
        """
        return hybrid_search(query, k=self.k, alpha=self.alpha, filter=self.filter)