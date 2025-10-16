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

        # Check metadata for researcher_name and other fields
        researcher_name = doc.metadata.get("researcher_name", "").lower()
        name = doc.metadata.get("name", "").lower()

        # Combine query into a single string for exact matching
        query_lower = query.lower()

        # Base score - count keywords in text
        base_score = sum(1 for keyword in keywords if keyword in text)

        # Name match bonus
        name_match_bonus = 0

        # Split query into words for better matching
        query_words = query_lower.split()
        is_name_query = len(query_words) <= 4  # Likely a name if 1-4 words

        # If this appears to be a name query, check for name matches
        if is_name_query:
            # Calculate name match score
            name_fields = [researcher_name, name]
            for name_field in name_fields:
                if not name_field:
                    continue

                # Exact match
                if query_lower == name_field:
                    name_match_bonus = 5.0  # High bonus for exact match
                    break

                # Contains match (substring)
                if query_lower in name_field:
                    name_match_bonus = 4.0  # Good bonus for contains match
                    break

                # Check if all query words are in the name field
                if all(word in name_field for word in query_words):
                    name_match_bonus = 3.0  # Moderate bonus for all words match
                    break

                # Check if at least one query word is in the name field
                if any(word in name_field for word in query_words):
                    name_match_bonus = 2.0  # Small bonus for partial match
                    break

        # Normalize base score by keyword count
        if keywords:
            normalized_score = base_score / len(keywords)
        else:
            normalized_score = 0.0

        # Combine scores (name match can push score above 1.0)
        total_score = normalized_score + name_match_bonus

        scored_docs.append((doc, total_score))

    # Sort by score (descending) and take the top k
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:k]


def hybrid_search(query: str, k: int = 4, alpha: float = 0.5,
                 filter: Optional[Dict[str, Any]] = None,
                 min_score_threshold: float = 0.2):
    """
    Perform a hybrid search combining vector similarity and keyword matching.

    Args:
        query (str): The query to search for
        k (int, optional): The number of results to return. Defaults to 4.
        alpha (float, optional): Weight for semantic search (0-1). Defaults to 0.5.
            Higher values give more weight to semantic search, lower to keyword search.
        filter (Optional[Dict[str, Any]], optional): Metadata filter. Defaults to None.
        min_score_threshold (float, optional): Minimum combined score to consider a result good.

    Returns:
        List[Document]: The search results
    """
    # Get the vector database
    db = get_or_create_vector_db()

    # Determine if this is likely a name search
    is_name_search = len(query.split()) <= 4 and any(word[0].isupper() for word in query.split())

    # For name searches, try exact metadata filtering first
    if is_name_search and not filter:
        logger.info(f"Detected possible name search: '{query}'")

        # Extract potential name
        words = query.split()
        name_parts = []

        for word in words:
            if word[0].isupper():
                name_parts.append(word)

        potential_name = " ".join(name_parts) if name_parts else query
        potential_name = potential_name

        # Create exact match filter
        name_filter = {
            "$or": [
                {"researcher_name": {"$eq": potential_name}},
                {"name": {"$eq": potential_name}}
            ]
        }

        # Try exact name match first
        logger.info(f"Trying exact name metadata filtering")
        exact_results = similarity_search_with_score(
            query="",  # Empty query since we're just filtering
            filter=name_filter,
            k=k,
            db=db
        )

        if exact_results:
            logger.info(f"Found {len(exact_results)} results via exact name matching")
            return [doc for doc, _ in exact_results]
        else:
            # If exact match fails, we need to do a manual partial match
            # Get all documents and filter manually
            logger.info(f"No exact matches, trying manual partial matching")
            all_results = db.get()
            texts = all_results['documents']
            metadatas = all_results['metadatas']

            # Manual partial match filtering
            matching_docs = []
            for i, metadata in enumerate(metadatas):
                researcher_name = metadata.get("researcher_name", "").lower()
                name = metadata.get("name", "").lower()

                # Check if the query is contained in researcher_name or name
                if (potential_name in researcher_name) or (potential_name in name):
                    doc = Document(page_content=texts[i], metadata=metadata)
                    matching_docs.append(doc)

            if matching_docs:
                logger.info(f"Found {len(matching_docs)} partial matches manually")
                return matching_docs[:k]

    # Get all chunks from the database to perform keyword search
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

    # Get the top results
    top_results = sorted_results[:k]

    # Check if we have good matches
    if top_results and top_results[0]['combined_score'] < min_score_threshold:
        # Poor match quality, try adjusting alpha
        if alpha > 0.3:
            logger.info(f"Poor match quality (score: {top_results[0]['combined_score']}). Retrying with lower alpha...")
            return hybrid_search(query, k=k, alpha=0.1, filter=filter, min_score_threshold=min_score_threshold)

    # Return the top k documents
    return [item['doc'] for item in top_results[:k]]


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