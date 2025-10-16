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

        # If this appears to be a name query, check for name matches with enhanced scoring
        if is_name_query:
            # Calculate name match score with more sophisticated logic
            name_fields = [researcher_name, name]

            # Extract a potential researcher name from the query
            potential_name = query_lower
            # Remove common prefixes like "who is", "about", etc.
            name_prefixes = ["who is ", "find ", "about ", "researcher named ", "dr. ", "doctor ", "professor "]
            for prefix in name_prefixes:
                if potential_name.startswith(prefix):
                    potential_name = potential_name.replace(prefix, "", 1)
                    break

            # Clean up potential name (remove trailing punctuation)
            potential_name = potential_name.rstrip('.,?!:;')

            # Get last name for partial matching (if query has multiple words)
            name_parts = potential_name.split()
            last_name = name_parts[-1] if len(name_parts) > 1 else ""

            # Only use last name matching if it's substantial (at least 4 chars)
            use_last_name = last_name and len(last_name) > 3

            for name_field in name_fields:
                if not name_field:
                    continue

                # Exact full name match (highest priority)
                if potential_name == name_field:
                    name_match_bonus = 10.0  # Very high bonus for exact match
                    break

                # Full name is contained in the name field
                if potential_name in name_field:
                    name_match_bonus = 8.0  # High bonus for contains match
                    break

                # Name field is contained in the query
                # (handles cases where query has extra words but contains the full name)
                if name_field and name_field in potential_name:
                    name_match_bonus = 7.0
                    break

                # All name parts are in the name field (in any order)
                # This helps with different name formats (First Last vs Last, First)
                if all(part in name_field for part in name_parts if len(part) > 2):
                    name_match_bonus = 6.0
                    break

                # Last name exact match (important for disambiguation)
                if use_last_name and last_name in name_field.split():
                    name_match_bonus = 5.0
                    break

                # Last name contained in name field
                if use_last_name and last_name in name_field:
                    name_match_bonus = 4.0
                    break

                # Check if all query words are in the name field (in any order)
                if all(word in name_field for word in query_words):
                    name_match_bonus = 3.0
                    break

                # Check if at least one substantial query word is in the name field
                # (skip very short words that could lead to false matches)
                substantial_matches = [word for word in query_words if len(word) > 3 and word in name_field]
                if substantial_matches:
                    name_match_bonus = 2.0 * (len(substantial_matches) / len(query_words))
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

    # Improved name search detection
    # Check for multiple indicators of a name search
    indicators = []

    # Indicator 1: Short query with capitalized words
    has_short_query = len(query.split()) <= 4
    has_capitalized = any(word[0].isupper() for word in query.split() if len(word) > 1)
    if has_short_query and has_capitalized:
        indicators.append("capitalized_words")

    # Indicator 2: Contains phrases like "who is", "about", etc.
    name_search_phrases = ["who is", "find", "about", "researcher named", "dr.", "doctor", "professor"]
    if any(phrase in query.lower() for phrase in name_search_phrases):
        indicators.append("name_phrases")

    # Indicator 3: Query follows pattern like "who is X" or "about X"
    name_patterns = [
        r"who\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",  # who is First Last
        r"about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",     # about First Last
        r"find\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",      # find First Last
    ]
    if any(re.search(pattern, query) for pattern in name_patterns):
        indicators.append("name_pattern_match")

    # Determine if this is a name search based on indicators
    is_name_search = len(indicators) > 0

    # For name searches, try exact metadata filtering first with enhanced detection
    if is_name_search and not filter:
        logger.info(f"Detected name search: '{query}' (indicators: {', '.join(indicators)})")

        # Extract potential name with improved method
        potential_name = ""

        # Try to extract from patterns first (most reliable)
        for pattern in name_patterns:
            match = re.search(pattern, query)
            if match:
                potential_name = match.group(1)
                logger.info(f"Extracted name from pattern: '{potential_name}'")
                break

        # If pattern extraction failed, try word-based extraction
        if not potential_name:
            words = query.split()
            name_parts = []

            # Extract consecutive capitalized words
            for word in words:
                if len(word) > 1 and word[0].isupper():
                    # Remove any trailing punctuation
                    clean_word = word.rstrip('.,?!:;')
                    name_parts.append(clean_word)
                elif name_parts:  # Stop at first non-capitalized word after finding capitalized ones
                    break

            if name_parts:
                potential_name = " ".join(name_parts)
                logger.info(f"Extracted name from capitalization: '{potential_name}'")
            else:
                # Last resort - use the whole query for short queries
                if len(words) <= 3:
                    # Clean the query for use as a name
                    cleaned_query = re.sub(r'^(who\s+is\s+|about\s+|find\s+)', '', query, flags=re.IGNORECASE)
                    cleaned_query = cleaned_query.strip().rstrip('.,?!:;')
                    potential_name = cleaned_query
                    logger.info(f"Using cleaned query as name: '{potential_name}'")

        if potential_name:
            # Try various name matching strategies in decreasing order of precision

            # Strategy 1: Exact match on researcher_name or name fields
            exact_filter = {
                "$or": [
                    {"researcher_name": {"$eq": potential_name}},
                    {"name": {"$eq": potential_name}}
                ]
            }

            # Try exact name match first
            logger.info(f"Strategy 1: Trying exact metadata match for: '{potential_name}'")
            exact_results = similarity_search_with_score(
                query="",  # Empty query since we're just filtering
                filter=exact_filter,
                k=k,
                db=db
            )

            if exact_results:
                logger.info(f"Found {len(exact_results)} results via exact name matching")
                # Prioritize core chunks first
                core_results = []
                other_results = []
                for doc, score in exact_results:
                    if doc.metadata.get("chunk_type") == "core":
                        core_results.append((doc, score))
                    else:
                        other_results.append((doc, score))

                # Return core chunks first, then others, up to k total
                combined_results = [doc for doc, _ in (core_results + other_results)][:k]
                return combined_results

            # Strategy 2: Try exact match specifically on core chunks (most relevant)
            core_filter = {
                "$and": [
                    {"$or": [
                        {"researcher_name": {"$eq": potential_name}},
                        {"name": {"$eq": potential_name}}
                    ]},
                    {"chunk_type": {"$eq": "core"}}  # Only search core chunks
                ]
            }

            logger.info(f"Strategy 2: Trying exact match on core chunks for: '{potential_name}'")
            core_results = similarity_search_with_score(
                query="",  # Empty query since we're just filtering
                filter=core_filter,
                k=k,
                db=db
            )

            if core_results:
                logger.info(f"Found {len(core_results)} core chunk results via exact name matching")
                return [doc for doc, _ in core_results]

            # Strategy 3: Manual partial matching with strict chunk type prioritization
            logger.info(f"Strategy 3: Trying manual partial matching for: '{potential_name}'")
            all_results = db.get()
            texts = all_results['documents']
            metadatas = all_results['metadatas']
            ids = all_results['ids']

            # Create separate lists for different chunk types
            core_matches = []
            interests_matches = []
            other_matches = []

            # Query variants for improved matching
            name_lower = potential_name.lower()
            name_parts = name_lower.split()
            last_name = name_parts[-1] if name_parts else ""

            for i, metadata in enumerate(metadatas):
                researcher_name = metadata.get("researcher_name", "").lower()
                name = metadata.get("name", "").lower()
                chunk_type = metadata.get("chunk_type", "")

                # Check for full name match first (highest priority)
                full_name_match = (name_lower in researcher_name) or (name_lower in name)

                # Check for last name match (medium priority)
                last_name_match = False
                if last_name and len(last_name) > 3:  # Only use last name if it's substantial
                    last_name_match = (last_name in researcher_name) or (last_name in name)

                # If we have any kind of match
                if full_name_match or last_name_match:
                    doc = Document(page_content=texts[i], metadata=metadata)
                    match_score = 1.0 if full_name_match else 0.7  # Higher score for full matches

                    # Sort by chunk type priority
                    if chunk_type == "core":
                        core_matches.append((doc, match_score))
                    elif chunk_type == "interests":
                        interests_matches.append((doc, match_score * 0.9))  # Slightly lower priority
                    else:
                        other_matches.append((doc, match_score * 0.8))  # Lowest priority

            # Combine matches in priority order
            all_matches = core_matches + interests_matches + other_matches

            if all_matches:
                logger.info(f"Found {len(all_matches)} partial matches manually " +
                           f"({len(core_matches)} core, {len(interests_matches)} interests, " +
                           f"{len(other_matches)} other)")

                # Sort by score and return top k
                all_matches.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in all_matches[:k]]

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