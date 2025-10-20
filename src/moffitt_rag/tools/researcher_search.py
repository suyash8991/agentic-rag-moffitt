"""
ResearcherSearchTool for the Moffitt Agentic RAG system.

This module implements a tool for searching researchers by their expertise,
interests, or background using semantic and keyword search with proper metadata filtering.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain.tools import BaseTool
from langchain_core.documents import Document

from ..db.hybrid_search import hybrid_search
from ..config.config import get_settings
from ..db.vector_store import get_or_create_vector_db, similarity_search_with_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_name_from_url(url: str) -> str:
    """
    Extract researcher name from profile URL.

    Args:
        url (str): The profile URL

    Returns:
        str: The extracted name
    """
    if not url:
        return ""

    # Extract the last part of the URL path which often contains the name
    path_parts = url.rstrip('/').split('/')
    if len(path_parts) > 0:
        name_slug = path_parts[-1]
        # Convert from slug format to name format
        name = name_slug.replace('-', ' ').title()
        return name
    return ""


def extract_name_from_text(text: str) -> str:
    """
    Extract researcher name from text content.

    Args:
        text (str): The text content that might contain a name

    Returns:
        str: The extracted name
    """
    # Try to find name in a "Name: " pattern
    name_match = re.search(r'Name:\s*([^,\n]+)', text)
    if name_match:
        name = name_match.group(1).strip()
        if name and name != ',':  # Check if name is non-empty and not just a comma
            return name

    # Try to find a name at the beginning followed by degrees
    degrees_match = re.search(r'^([^,]+),\s*([A-Z][A-Za-z]*\.?(?:\s+[A-Z][A-Za-z]*\.?)*)', text)
    if degrees_match:
        return degrees_match.group(1).strip()

    return ""


def extract_relevant_snippet(text: str, query: str, max_length: int = 200) -> str:
    """
    Extract the most relevant snippet from a text based on a query.

    Args:
        text (str): The text to extract from
        query (str): The query to search for
        max_length (int, optional): Maximum length of the snippet. Defaults to 200.

    Returns:
        str: The most relevant snippet
    """
    # If text is short enough, return the whole thing
    if len(text) <= max_length:
        return text

    # Split the query into terms
    query_terms = set(re.findall(r'\w+', query.lower()))

    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Score each sentence based on query term matches
    scored_sentences = []
    for sentence in sentences:
        # Count the number of query terms in the sentence
        sentence_terms = set(re.findall(r'\w+', sentence.lower()))
        score = len(query_terms.intersection(sentence_terms))
        scored_sentences.append((sentence, score))

    # Sort sentences by score (descending)
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    # Take the top sentences until we reach the max length
    snippet = ""
    for sentence, _ in scored_sentences:
        if len(snippet) + len(sentence) + 1 <= max_length:
            snippet += sentence + " "
        else:
            break

    # If no good matches, just take the beginning of the text
    if not snippet:
        snippet = text[:max_length].rsplit(" ", 1)[0] + "..."

    return snippet.strip()


class ResearcherSearchTool(BaseTool):
    """
    Tool for searching researchers by their expertise, interests, or background.

    This tool uses metadata filtering for name searches and hybrid search
    for topic searches to find the most relevant researchers.

    For optimal results when searching for a specific researcher by name,
    set is_name_search=True to ensure more accurate results.
    """

    name: str = "ResearcherSearch"
    description: str = "Search for researchers by their expertise, interests, or background. Set is_name_search=True for name-based searches for better accuracy."

    # Class-level tracking of queries and attempts
    _query_attempts = {}
    _max_attempts_per_query = 3  # Maximum attempts for the same query

    def _check_rate_limit(self, query: str) -> bool:
        """
        Check if the query has exceeded the rate limit.

        Args:
            query (str): The search query

        Returns:
            bool: True if the query is allowed, False if it should be rate-limited
        """
        # Normalize query for comparison (lowercase, remove extra spaces, and remove punctuation)
        import re
        normalized_query = ' '.join(re.sub(r'[^\w\s]', '', query.lower()).split())

        # Check for semantic similarity with previous queries to prevent minor variations
        # causing redundant searches
        similar_queries = []
        for prev_query in self._query_attempts.keys():
            # Simple similarity check - share at least 70% of words
            prev_words = set(prev_query.split())
            curr_words = set(normalized_query.split())
            if prev_words and curr_words:  # Avoid division by zero
                # Calculate Jaccard similarity (intersection over union)
                intersection = len(prev_words.intersection(curr_words))
                union = len(prev_words.union(curr_words))
                similarity = intersection / union if union > 0 else 0

                if similarity > 0.7:  # 70% similar
                    similar_queries.append(prev_query)

        # If we have similar queries, use the one with the highest count
        if similar_queries:
            max_count = 0
            for similar_query in similar_queries:
                if self._query_attempts[similar_query] > max_count:
                    max_count = self._query_attempts[similar_query]
                    normalized_query = similar_query  # Use the existing query for counting

        # Get current count for this query
        current_count = self._query_attempts.get(normalized_query, 0)

        # Check if we've exceeded the limit
        if current_count >= self._max_attempts_per_query:
            logger.warning(f"Rate limit exceeded for query: {query} ({current_count} attempts)")
            return False

        # Increment the count
        self._query_attempts[normalized_query] = current_count + 1
        logger.info(f"Query attempt {current_count + 1}/{self._max_attempts_per_query} for: {query}")

        return True

    def _is_name_search(self, query: str) -> bool:
        """
        Determine if the query is likely a name search.

        Args:
            query (str): The search query

        Returns:
            bool: True if the query is likely a name search, False otherwise
        """
        query_lower = query.lower()

        # Check for phrases that indicate name searches
        name_search_indicators = ["who is", "find", "about", "researcher named", "dr.", "doctor", "professor"]
        is_name_search_by_phrase = any(indicator in query_lower for indicator in name_search_indicators)

        # Check for capitalized words which likely indicate proper names
        words = query.split()
        capitalized_words = [word for word in words if len(word) > 1 and word[0].isupper() and word.lower() not in ["what", "who", "where", "when", "why", "how"]]
        has_capitalized_words = len(capitalized_words) >= 1
        is_short_query = len(words) <= 4
        is_name_search_by_capitalization = has_capitalized_words and is_short_query

        return is_name_search_by_phrase or is_name_search_by_capitalization

    def _extract_potential_name(self, query: str) -> str:
        """
        Extract potential researcher name from a query.

        Args:
            query (str): The search query

        Returns:
            str: The potential researcher name
        """
        # Method 1: Look for specific patterns
        patterns = [
            r"who is ([A-Z][a-z]+ [A-Z][a-z]+)",  # "who is John Doe"
            r"about ([A-Z][a-z]+ [A-Z][a-z]+)",   # "about John Doe"
            r"find ([A-Z][a-z]+ [A-Z][a-z]+)",    # "find John Doe"
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)

        # Method 2: Extract capitalized words (likely names)
        words = query.split()
        capitalized_sequence = []

        for word in words:
            if len(word) > 1 and word[0].isupper() and word.lower() not in ["what", "who", "where", "when", "why", "how"]:
                capitalized_sequence.append(word)
            elif capitalized_sequence:  # We've reached the end of a capitalized sequence
                break

        if len(capitalized_sequence) >= 1:
            return " ".join(capitalized_sequence)

        # Method 3: If query is short and simple, use it as is
        if len(words) <= 2:
            return query

        # No clear name found
        return ""

    def _format_results(self, results: List[Document], query: str) -> str:
        """
        Format search results with researcher names and relevant snippets.

        Args:
            results (List[Document]): The search results
            query (str): The original query

        Returns:
            str: The formatted results as a string
        """
        if not results:
            return f"No researchers found matching the query: {query}"

        # Keep track of unique researcher IDs to avoid duplicates
        seen_researcher_ids = set()
        unique_results = []

        for doc in results:
            researcher_id = doc.metadata.get("researcher_id", "unknown")
            # Only include each researcher once (first chunk)
            if researcher_id not in seen_researcher_ids:
                seen_researcher_ids.add(researcher_id)
                unique_results.append(doc)

        # Format each unique result
        formatted_results = []
        for doc in unique_results:
            researcher_id = doc.metadata["researcher_id"]
            profile_url = doc.metadata.get("profile_url", "")

            # Get researcher name from metadata, fall back to extraction if empty
            display_name = doc.metadata.get("researcher_name", "").strip()

            if not display_name:
                # Try to extract name from URL
                url_name = extract_name_from_url(profile_url)

                # Try to extract name from text if URL extraction failed
                text_name = extract_name_from_text(doc.page_content)

                # Use the best name we could find
                display_name = text_name or url_name or "Unknown Researcher"
                logger.info(f"Extracted name '{display_name}' for researcher_id {researcher_id} from {'text' if text_name else 'URL'}")

            program = doc.metadata.get("program", "Unknown Program")

            # Extract the most relevant snippet from the content
            snippet = extract_relevant_snippet(doc.page_content, query)

            formatted_results.append(
                f"Researcher: {display_name}\n"
                f"Program: {program}\n"
                f"Relevance: {snippet}\n"
                f"Profile: {profile_url}\n"
            )

        return "\n".join(formatted_results)

    def _check_information_sufficiency(self, result_text: str, query: str) -> bool:
        """
        Check if the search results provide sufficient information to answer the query.

        For simple name queries like "Who is X?", we consider the result sufficient if it contains
        basic information about the researcher (name and program).

        Args:
            result_text (str): The formatted search result
            query (str): The original query

        Returns:
            bool: True if information is sufficient to answer the query
        """
        # Check if this is a simple "who is X" query
        if re.search(r'^who\s+is\s+', query.lower()):
            # For simple identity queries, check if we have the basic info
            if "Researcher:" in result_text and "Program:" in result_text:
                # Check if the query contains a name that's in the result
                query_lower = query.lower()
                # Extract potential name from query (after "who is")
                match = re.search(r'who\s+is\s+([^?]+)', query_lower)
                if match:
                    potential_name = match.group(1).strip()
                    # Check if this name appears in the result
                    if potential_name in result_text.lower():
                        logger.info("Information deemed sufficient for simple name query")

                        # Make this more obvious to the agent by adding a note
                        enhanced_result = result_text + "\n\n[INFORMATION SUFFICIENCY NOTE: This result contains the basic information about the researcher. No further searches are needed for a simple identity query.]"
                        return True, enhanced_result

        # Default case - information may not be sufficient
        return False, result_text

    def _run(self, query: str, is_name_search: bool = None) -> str:
        """
        Run the tool with the given query.

        Args:
            query (str): The search query
            is_name_search (bool, optional): Explicitly specify if this is a name search.
                If None, will be auto-detected. Defaults to None.

        Returns:
            str: The search results formatted as a string
        """
        logger.info(f"Searching for researchers matching: {query}")

        # Check if we've exceeded the rate limit for this query
        if not self._check_rate_limit(query):
            # Return rate limit message
            return (
                f"You've made multiple attempts searching for '{query}' without success. "
                f"Based on the available data, this researcher may not be in the database. "
                f"Please try a different search approach or look for other researchers "
                f"with similar expertise."
            )

        # Determine if this is likely a name search (use provided value or auto-detect)
        if is_name_search is None:
            is_name_search = self._is_name_search(query)
            logger.info(f"Query auto-detected as {'name search' if is_name_search else 'topic search'}")
        else:
            logger.info(f"Query explicitly marked as {'name search' if is_name_search else 'topic search'}")

        # If this is a name search, use metadata filtering with higher priority
        if is_name_search:
            # Extract potential researcher name using detection methods
            potential_name = self._extract_potential_name(query)

            if potential_name:
                logger.info(f"Detected potential researcher name: {potential_name}")

                # Get the database
                db = get_or_create_vector_db()

                # Direct name search
                direct_filter = {
                    "researcher_name": {"$eq": potential_name}
                }

                # Search with direct filter first
                logger.info(f"Trying direct metadata match for: {potential_name}")
                direct_results = similarity_search_with_score(
                    query="",  # Empty query since we're just filtering
                    filter=direct_filter,
                    k=5,
                    db=db
                )

                if direct_results:
                    # Direct match found, this is the highest priority
                    logger.info(f"Found {len(direct_results)} direct matches for {potential_name}")
                    formatted_result = self._format_results([doc for doc, _ in direct_results], query)

                    # Check if this result provides sufficient information for a simple query
                    is_sufficient, enhanced_result = self._check_information_sufficiency(formatted_result, query)
                    return enhanced_result

                # If direct match fails, try exact chunk types for the core information
                # Prioritize the core chunk type which contains the main researcher information
                exact_filter = {
                    "$and": [
                        {"researcher_name": {"$eq": potential_name}},
                        {"chunk_type": {"$eq": "core"}}  # Prioritize core information chunks
                    ]
                }

                # Search with exact filter first
                logger.info(f"Trying exact metadata match for core chunks: {potential_name}")
                exact_results = similarity_search_with_score(
                    query="",  # Empty query since we're just filtering
                    filter=exact_filter,
                    k=5,
                    db=db
                )

                # If exact match fails, try partial match using $in operator
                # Since ChromaDB doesn't support $contains, we'll need to do partial matching in code
                if not exact_results:
                    logger.info(f"No exact core matches, trying to search all chunks for: {potential_name}")
                    # Get all documents
                    all_results = db.get()
                    texts = all_results['documents']
                    metadatas = all_results['metadatas']

                    # Manual partial match filtering with chunk type prioritization
                    core_matches = []
                    other_matches = []

                    for i, metadata in enumerate(metadatas):
                        researcher_name = metadata.get("researcher_name", "").lower()
                        chunk_type = metadata.get("chunk_type", "")

                        # Check if the name is contained in researcher_name field
                        if potential_name.lower() in researcher_name:
                            doc = Document(page_content=texts[i], metadata=metadata)
                            # Prioritize core chunks (basic researcher info)
                            if chunk_type == "core":
                                core_matches.append((doc, 1.0))  # Higher score for core chunks
                            else:
                                other_matches.append((doc, 0.8))  # Lower score for other chunks

                    # Prioritize core matches, then add other matches
                    all_matches = core_matches + other_matches

                    if all_matches:
                        logger.info(f"Found {len(all_matches)} partial matches manually ({len(core_matches)} core)")
                        formatted_result = self._format_results([doc for doc, _ in all_matches], query)

                        # Check if this result provides sufficient information for a simple query
                        is_sufficient, enhanced_result = self._check_information_sufficiency(formatted_result, query)
                        return enhanced_result
                else:
                    # If we found exact matches, use them
                    logger.info(f"Found {len(exact_results)} exact core matches")
                    formatted_result = self._format_results([doc for doc, _ in exact_results], query)

                    # Check if this result provides sufficient information for a simple query
                    is_sufficient, enhanced_result = self._check_information_sufficiency(formatted_result, query)
                    return enhanced_result

        # For topic searches or if metadata filtering failed, use hybrid search
        # For name searches, give even more weight to keyword matching (lower alpha)
        alpha = 0.2 if is_name_search else 0.7  # Reduced alpha for name searches to favor keyword matching
        logger.info(f"Using hybrid search with alpha={alpha}")

        results = hybrid_search(
            query=query,
            k=5,
            alpha=alpha  # Adjusted based on query type
        )

        if not results:
            return f"No researchers found matching the query: {query}"

        formatted_result = self._format_results(results, query)

        # Check if this result provides sufficient information for a simple query
        is_sufficient, enhanced_result = self._check_information_sufficiency(formatted_result, query)
        return enhanced_result

    async def _arun(self, query: str, is_name_search: bool = None) -> str:
        """
        Run the tool asynchronously with the given query.

        Args:
            query (str): The search query
            is_name_search (bool, optional): Explicitly specify if this is a name search.
                If None, will be auto-detected. Defaults to None.

        Returns:
            str: The search results formatted as a string
        """
        # For now, just call the synchronous version
        return self._run(query, is_name_search)