"""
ResearcherSearchTool for the Moffitt Agentic RAG system.

This module implements a tool for searching researchers by their expertise,
interests, or background using semantic and keyword search.
"""

import re
import logging
from typing import List, Dict, Any, Optional

from langchain.tools import BaseTool

from ..db.hybrid_search import hybrid_search
from ..config.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    This tool uses hybrid search (combining vector search with keyword search)
    to find the most relevant researchers for a given query.
    """

    name = "ResearcherSearch"
    description = "Search for researchers by their expertise, interests, or background"

    def _run(self, query: str) -> str:
        """
        Run the tool with the given query.

        Args:
            query (str): The search query

        Returns:
            str: The search results formatted as a string
        """
        logger.info(f"Searching for researchers matching: {query}")

        # Use the hybrid search functionality to find relevant researchers
        results = hybrid_search(
            query=query,
            k=5,
            alpha=0.7  # Balanced weight between semantic and keyword
        )

        # Format results with researcher names and relevant snippets
        if not results:
            return f"No researchers found matching the query: {query}"

        formatted_results = []
        for doc in results:
            researcher_id = doc.metadata["researcher_id"]
            name = doc.metadata["name"]
            program = doc.metadata.get("program", "Unknown Program")

            # Extract the most relevant snippet from the content
            snippet = extract_relevant_snippet(doc.page_content, query)

            formatted_results.append(
                f"Researcher: {name}\n"
                f"Program: {program}\n"
                f"Relevance: {snippet}\n"
                f"Profile: {doc.metadata['profile_url']}\n"
            )

        return "\n".join(formatted_results)

    async def _arun(self, query: str) -> str:
        """
        Run the tool asynchronously with the given query.

        Args:
            query (str): The search query

        Returns:
            str: The search results formatted as a string
        """
        # For now, just call the synchronous version
        return self._run(query)