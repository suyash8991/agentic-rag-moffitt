"""
LangSmith utilities for the Moffitt Agentic RAG system.

This module provides helper functions for LangSmith integration,
including metadata enrichment and custom callbacks.
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from langsmith import Client
from app.core.config import settings


def is_langsmith_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    return settings.LANGCHAIN_TRACING_V2


def get_langsmith_client() -> Optional[Client]:
    """Get LangSmith client if tracing is enabled."""
    if not is_langsmith_enabled():
        return None

    return Client(
        api_key=settings.LANGCHAIN_API_KEY,
        api_url=settings.LANGCHAIN_ENDPOINT
    )


def create_run_metadata(
    query_id: str,
    query: str,
    query_type: str,
    user_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create standardized metadata for LangSmith runs.

    Args:
        query_id: Unique query identifier
        query: User query text
        query_type: Type of query (general, researcher, etc.)
        user_id: Optional user identifier
        **kwargs: Additional metadata fields

    Returns:
        Dictionary of metadata
    """
    metadata = {
        "query_id": query_id,
        "query_type": query_type,
        "query_preview": query[:100],
        "query_length": len(query),
        "timestamp": datetime.now().isoformat(),
        "system": "moffitt-agentic-rag",
        "version": "1.0.0",
    }

    if user_id:
        metadata["user_id"] = user_id

    # Add any additional metadata
    metadata.update(kwargs)

    return metadata


def create_run_tags(
    query_type: str,
    query_id: str,
    additional_tags: Optional[List[str]] = None
) -> List[str]:
    """
    Create standardized tags for LangSmith runs.

    Args:
        query_type: Type of query
        query_id: Query identifier
        additional_tags: Optional additional tags

    Returns:
        List of tags
    """
    tags = [
        "moffitt-rag",
        f"query-type:{query_type}",
        f"qid:{query_id[:8]}",
    ]

    if additional_tags:
        tags.extend(additional_tags)

    return tags


def add_researcher_results_to_metadata(
    metadata: Dict[str, Any],
    results: List[Any]
) -> Dict[str, Any]:
    """
    Add researcher search results to run metadata.

    Args:
        metadata: Existing metadata dictionary
        results: List of search results

    Returns:
        Updated metadata dictionary
    """
    metadata["result_count"] = len(results)

    if results:
        # Add first few researcher names for quick reference
        researcher_names = []
        for result in results[:5]:
            if hasattr(result, 'metadata') and 'researcher_name' in result.metadata:
                researcher_names.append(result.metadata['researcher_name'])

        if researcher_names:
            metadata["researchers_found"] = researcher_names

    return metadata
