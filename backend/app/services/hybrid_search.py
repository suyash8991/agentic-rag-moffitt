"""
Hybrid search implementation combining vector similarity and keyword matching.

This module provides hybrid search functionality that combines semantic vector search
with keyword-based text search for improved retrieval accuracy.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from langchain_core.documents import Document

from .vector_db import get_or_create_vector_db
from ..utils.logging import get_logger, log_tool_event

# Setup logging
logger = get_logger(__name__)


def keyword_search(
    query: str,
    documents: List[Document],
    k: int = 5
) -> List[Tuple[Document, float]]:
    """
    Perform keyword-based search on documents.

    Args:
        query: The search query
        documents: List of documents to search
        k: Number of results to return

    Returns:
        List of (document, score) tuples sorted by relevance
    """
    # Normalize query
    query_lower = query.lower()
    query_terms = set(re.findall(r'\w+', query_lower))

    if not query_terms:
        return []

    # Score each document
    scored_docs = []
    for doc in documents:
        # Combine page content and metadata for searching
        searchable_text = doc.page_content.lower()

        # Add metadata fields to searchable text
        if hasattr(doc, 'metadata') and doc.metadata:
            for key, value in doc.metadata.items():
                if isinstance(value, str):
                    searchable_text += f" {value.lower()}"

        # Count term matches
        text_terms = re.findall(r'\w+', searchable_text)
        term_freq = defaultdict(int)
        for term in text_terms:
            if term in query_terms:
                term_freq[term] += 1

        # Calculate score based on:
        # 1. Number of matching terms
        # 2. Term frequency
        # 3. Term proximity (bonus for exact phrase matches)
        matching_terms = len(term_freq)
        total_frequency = sum(term_freq.values())

        # Check for exact phrase match
        phrase_match = 1.5 if query_lower in searchable_text else 1.0

        # Calculate final score
        if matching_terms > 0:
            score = (matching_terms / len(query_terms)) * total_frequency * phrase_match
            scored_docs.append((doc, score))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return top k results
    return scored_docs[:k]


def hybrid_search(
    query: str,
    k: int = 5,
    alpha: float = 0.5,
    filter: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Perform hybrid search combining vector similarity and keyword matching.

    Args:
        query: The search query
        k: Number of results to return
        alpha: Balance between semantic (1.0) and keyword (0.0) search
               - alpha=0.0: Pure keyword search
               - alpha=0.5: Equal weight to both
               - alpha=1.0: Pure semantic search
               - alpha=0.7: Recommended for topic searches (favor semantic)
               - alpha=0.3: Recommended for name searches (favor keyword)
        filter: Optional metadata filter for vector search

    Returns:
        List of documents ranked by combined score
    """
    # Log hybrid search start
    log_tool_event("hybrid_search_start", {
        "query": query[:100],
        "k": k,
        "alpha": alpha,
        "has_filter": filter is not None
    })

    # Get vector database
    db = get_or_create_vector_db()
    if db is None:
        logger.error("Failed to load vector database")
        log_tool_event("hybrid_search_error", {"error": "database_unavailable"})
        return []

    # Perform semantic search (vector-based) with scores
    try:
        # Use ChromaDB's built-in similarity_search_with_score method
        semantic_results = db.similarity_search_with_score(
            query=query,
            k=k * 2,  # Get more candidates for hybrid scoring
            filter=filter
        )
        log_tool_event("semantic_search_complete", {
            "result_count": len(semantic_results),
            "k_requested": k * 2
        })
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        log_tool_event("semantic_search_error", {"error": str(e)})
        semantic_results = []

    # Get all documents for keyword search
    try:
        # Get more documents from the database for keyword search
        all_docs = db.similarity_search(query, k=k * 3, filter=filter)
        log_tool_event("documents_retrieved", {
            "document_count": len(all_docs),
            "k_requested": k * 3
        })
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        log_tool_event("document_retrieval_error", {"error": str(e)})
        all_docs = []

    # Perform keyword search
    keyword_results = keyword_search(query, all_docs, k=k * 2)
    log_tool_event("keyword_search_complete", {
        "result_count": len(keyword_results),
        "k_requested": k * 2
    })

    # Normalize scores to [0, 1] range
    def normalize_scores(scored_docs: List[Tuple[Any, float]]) -> Dict[str, float]:
        """Normalize scores to 0-1 range."""
        if not scored_docs:
            return {}

        scores = [score for _, score in scored_docs]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0

        normalized = {}
        for doc, score in scored_docs:
            # Use document content as key (not ideal but works for deduplication)
            doc_key = doc.page_content[:100]  # Use first 100 chars as key
            normalized_score = (score - min_score) / score_range if score_range > 0 else 0.0
            normalized[doc_key] = normalized_score

        return normalized

    # Normalize semantic scores (these are already distances, need to invert)
    semantic_scores = {}
    for doc, distance in semantic_results:
        doc_key = doc.page_content[:100]
        # Convert distance to similarity (smaller distance = higher similarity)
        # Assuming distance is in range [0, 2] for cosine distance
        similarity = 1.0 - (distance / 2.0)
        semantic_scores[doc_key] = max(0.0, min(1.0, similarity))

    # Normalize keyword scores
    keyword_scores = normalize_scores(keyword_results)

    # Combine scores with alpha weighting
    combined_scores = {}
    all_doc_keys = set(semantic_scores.keys()) | set(keyword_scores.keys())

    # Create a mapping from doc_key to actual document
    doc_map = {}
    for doc, _ in semantic_results:
        doc_key = doc.page_content[:100]
        if doc_key not in doc_map:
            doc_map[doc_key] = doc
    for doc, _ in keyword_results:
        doc_key = doc.page_content[:100]
        if doc_key not in doc_map:
            doc_map[doc_key] = doc

    for doc_key in all_doc_keys:
        semantic_score = semantic_scores.get(doc_key, 0.0)
        keyword_score = keyword_scores.get(doc_key, 0.0)

        # Combined score: alpha * semantic + (1 - alpha) * keyword
        combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
        combined_scores[doc_key] = combined_score

    log_tool_event("score_combination_complete", {
        "unique_documents": len(combined_scores),
        "semantic_count": len(semantic_scores),
        "keyword_count": len(keyword_scores)
    })

    # Sort by combined score descending
    sorted_doc_keys = sorted(
        combined_scores.keys(),
        key=lambda x: combined_scores[x],
        reverse=True
    )

    # Return top k documents
    result_docs = []
    for doc_key in sorted_doc_keys[:k]:
        if doc_key in doc_map:
            result_docs.append(doc_map[doc_key])

    # Log hybrid search completion
    log_tool_event("hybrid_search_complete", {
        "query": query[:100],
        "result_count": len(result_docs),
        "alpha": alpha,
        "k": k
    })

    return result_docs
