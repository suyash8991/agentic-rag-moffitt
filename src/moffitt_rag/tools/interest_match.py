"""
InterestMatchTool for the Moffitt Agentic RAG system.

This module implements a tool for finding researchers with similar research interests.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain.tools import BaseTool

from ..db.hybrid_search import hybrid_search
from ..data.loader import load_all_researcher_profiles
from ..models.embeddings import embed_query, embedding_similarity
from ..config.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterestMatchTool(BaseTool):
    """
    Tool for finding researchers with similar research interests.

    This tool can find researchers similar to a named researcher,
    or researchers with interests matching a specific query.
    """

    name = "InterestMatch"
    description = "Find researchers with similar research interests"

    def _run(self, query: str) -> str:
        """
        Run the tool with the given query.

        Args:
            query (str): The query, which can be:
                         - "similar to [researcher name]" to find similar researchers
                         - Any research topic to find researchers with matching interests

        Returns:
            str: The matching researchers formatted as a string
        """
        # First check if it's asking for researchers similar to a named researcher
        researcher_name = None
        if "similar to" in query.lower():
            # Extract name after "similar to"
            match = re.search(r"similar to\s+([A-Za-z\s\.-]+)", query.lower())
            if match:
                researcher_name = match.group(1).strip()

        if researcher_name:
            # Find researchers similar to the named researcher
            return self._find_similar_researchers(researcher_name)
        else:
            # Find researchers with interests matching the query
            return self._find_matching_interests(query)

    def _find_similar_researchers(self, researcher_name: str) -> str:
        """
        Find researchers similar to a named researcher.

        Args:
            researcher_name (str): The name of the researcher

        Returns:
            str: The similar researchers formatted as a string
        """
        logger.info(f"Finding researchers similar to: {researcher_name}")

        # Load all profiles
        profiles = load_all_researcher_profiles()

        # Find the named researcher
        target_profile = None
        for profile in profiles:
            if researcher_name.lower() in profile.name.lower():
                target_profile = profile
                break

        if not target_profile:
            return f"Could not find a researcher named '{researcher_name}'. Please check the name and try again."

        # Create a query from the target researcher's interests and overview
        query_text = ""
        if target_profile.research_interests:
            query_text += " ".join(target_profile.research_interests)
        if target_profile.overview:
            query_text += " " + target_profile.overview

        # Use the hybrid search to find similar researchers
        results = hybrid_search(query_text, k=6, alpha=0.8)

        # Filter out the target researcher from results
        results = [doc for doc in results if doc.metadata.get("researcher_id") != target_profile.researcher_id]

        # Format results
        if not results:
            return f"No similar researchers found for {target_profile.name}."

        formatted_results = [
            f"Researchers with interests similar to {target_profile.name}:\n"
        ]

        for doc in results:
            name = doc.metadata.get("name", "Unknown")
            program = doc.metadata.get("program", "Unknown Program")

            # Find relevant text from the content
            relevant_snippet = self._extract_relevant_snippet(doc.page_content, query_text, max_length=150)

            formatted_results.append(f"- {name} ({program})")
            if relevant_snippet:
                formatted_results.append(f"  Research focus: {relevant_snippet}")

        return "\n".join(formatted_results)

    def _find_matching_interests(self, query: str) -> str:
        """
        Find researchers with interests matching a query.

        Args:
            query (str): The research interest query

        Returns:
            str: The matching researchers formatted as a string
        """
        logger.info(f"Finding researchers with interests matching: {query}")

        # Use hybrid search to find researchers with matching interests
        results = hybrid_search(query, k=5, alpha=0.7)

        # Format results
        if not results:
            return f"No researchers found with interests matching '{query}'."

        formatted_results = [
            f"Researchers with interests in '{query}':\n"
        ]

        for doc in results:
            name = doc.metadata.get("name", "Unknown")
            program = doc.metadata.get("program", "Unknown Program")
            interests = doc.metadata.get("research_interests", [])

            # Find most relevant interests
            relevant_interests = self._find_relevant_interests(interests, query)

            formatted_results.append(f"- {name} ({program})")
            if relevant_interests:
                interest_str = ", ".join(relevant_interests[:3])
                formatted_results.append(f"  Relevant interests: {interest_str}")
            else:
                # Extract relevant snippet from content if no explicit interests match
                relevant_snippet = self._extract_relevant_snippet(doc.page_content, query, max_length=100)
                if relevant_snippet:
                    formatted_results.append(f"  Research focus: {relevant_snippet}")

        return "\n".join(formatted_results)

    def _find_relevant_interests(self, interests: List[str], query: str) -> List[str]:
        """
        Find the most relevant interests from a list based on a query.

        Args:
            interests (List[str]): The list of interests
            query (str): The query to match against

        Returns:
            List[str]: The most relevant interests
        """
        if not interests:
            return []

        # Split query into terms
        query_terms = query.lower().split()

        # Score each interest based on term overlap
        scored_interests = []
        for interest in interests:
            # Count the number of query terms that appear in the interest
            score = sum(1 for term in query_terms if term.lower() in interest.lower())
            scored_interests.append((interest, score))

        # Sort by score (descending) and return those with any match
        sorted_interests = [i for i, s in sorted(scored_interests, key=lambda x: x[1], reverse=True) if s > 0]
        return sorted_interests

    def _extract_relevant_snippet(self, text: str, query: str, max_length: int = 200) -> str:
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

    async def _arun(self, query: str) -> str:
        """
        Run the tool asynchronously with the given query.

        Args:
            query (str): The query to search for

        Returns:
            str: The search results formatted as a string
        """
        # For now, just call the synchronous version
        return self._run(query)