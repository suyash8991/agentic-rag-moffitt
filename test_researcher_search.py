"""
Test script to debug the researcher search functionality.

This script helps diagnose issues with the researcher search system, particularly
focusing on finding known researchers that should be in the database.
"""

import logging
import sys
import argparse
import re
from langchain_core.documents import Document
from typing import Dict, Any, List, Optional, Tuple

# Configure logging to show more detailed info
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the necessary modules
from src.moffitt_rag.db.vector_store import get_or_create_vector_db, similarity_search_with_score
from src.moffitt_rag.db.hybrid_search import hybrid_search, keyword_search
from src.moffitt_rag.tools.researcher_search import (
    ResearcherSearchTool,
    extract_name_from_url,
    extract_name_from_text
)

# List of known researchers that are expected to be found easily
KNOWN_RESEARCHERS = [
    "Theresa Boyle",
    "Ahmad Tarhini",
    "Noemi Andor",
    "Eric Padron",
    "John Cleveland",
    "Aleksandra Karolak"
]


def check_chroma_db_for_researcher(name: str):
    """
    Check if a researcher exists in the Chroma DB.
    """
    print(f"\n=== Checking Chroma DB for researcher: {name} ===")

    # Get the vector database
    db = get_or_create_vector_db()

    # Get all chunks from the database
    results = db.get()
    texts = results['documents']
    ids = results['ids']
    metadatas = results['metadatas']

    print(f"Total chunks in DB: {len(texts)}")

    # Check if the name exists in any metadata
    name_lower = name.lower()
    found_in_metadata = []
    found_in_text = []

    for i, metadata in enumerate(metadatas):
        if name_lower in metadata.get("name", "").lower():
            found_in_metadata.append((ids[i], metadata))

    # Check if the name exists in any text
    for i, text in enumerate(texts):
        if name_lower in text.lower():
            found_in_text.append((ids[i], metadatas[i]))

    print(f"Found in metadata: {len(found_in_metadata)} chunks")
    for i, (id, metadata) in enumerate(found_in_metadata[:5]):  # Show first 5 results
        print(f"  {i+1}. ID: {id}")
        print(f"     Name: {metadata.get('name')}")
        print(f"     Program: {metadata.get('program')}")
        print(f"     URL: {metadata.get('profile_url')}")

    print(f"\nFound in text: {len(found_in_text)} chunks")
    for i, (id, metadata) in enumerate(found_in_text[:5]):  # Show first 5 results
        print(f"  {i+1}. ID: {id}")
        print(f"     Name: {metadata.get('name')}")
        print(f"     Program: {metadata.get('program')}")
        print(f"     URL: {metadata.get('profile_url')}")

    return found_in_metadata, found_in_text


def test_keyword_search(query: str, texts: List[str], metadatas: List[Dict[str, Any]], k: int = 5):
    """
    Test the keyword_search function with the given query.
    """
    print(f"\n=== Testing keyword_search for: {query} ===")

    results = keyword_search(query, texts, metadatas, k)

    print(f"Found {len(results)} results")
    for i, (doc, score) in enumerate(results):
        print(f"  {i+1}. Score: {score:.4f}")
        print(f"     Name: {doc.metadata.get('name')}")
        print(f"     Program: {doc.metadata.get('program')}")
        print(f"     URL: {doc.metadata.get('profile_url')}")
        print(f"     Text preview: {doc.page_content[:100]}...")

    return results


def test_hybrid_search(query: str, k: int = 5, alpha: float = 0.7):
    """
    Test the hybrid_search function with the given query and parameters.
    """
    print(f"\n=== Testing hybrid_search for: {query} (k={k}, alpha={alpha}) ===")

    results = hybrid_search(query=query, k=k, alpha=alpha)

    print(f"Found {len(results)} results")
    for i, doc in enumerate(results):
        print(f"  {i+1}. Name: {doc.metadata.get('name')}")
        print(f"     Program: {doc.metadata.get('program')}")
        print(f"     URL: {doc.metadata.get('profile_url')}")
        print(f"     Text preview: {doc.page_content[:100]}...")

    return results


def test_with_different_alphas(query: str, alphas: List[float] = [0.3, 0.5, 0.7, 0.9]):
    """
    Test the hybrid_search function with different alpha values.
    """
    print(f"\n=== Testing hybrid_search with different alphas for: {query} ===")

    for alpha in alphas:
        print(f"\nAlpha = {alpha}")
        results = hybrid_search(query=query, k=3, alpha=alpha)

        for i, doc in enumerate(results):
            print(f"  {i+1}. Name: {doc.metadata.get('name')}")
            print(f"     Program: {doc.metadata.get('program')}")
            print(f"     URL: {doc.metadata.get('profile_url')}")


def test_direct_vector_similarity(name: str, k: int = 5):
    """
    Test direct vector similarity search for the given researcher name.
    This function helps diagnose what ChromaDB is actually returning before any hybrid search logic.
    """
    print(f"\n=== Testing Direct Vector Similarity Search for: {name} ===")

    # Get the vector database
    db = get_or_create_vector_db()

    # Perform vector search with scores
    results = similarity_search_with_score(name, k=k, db=db)

    print(f"Found {len(results)} results")
    for i, (doc, score) in enumerate(results):
        # Score in Chroma is a distance - lower is better
        print(f"  {i+1}. Distance Score: {score:.6f} (lower is better)")
        print(f"     Name: {doc.metadata.get('name', '')}")
        print(f"     Researcher Name: {doc.metadata.get('researcher_name', '')}")
        print(f"     Program: {doc.metadata.get('program', '')}")
        print(f"     URL: {doc.metadata.get('profile_url', '')}")
        print(f"     Text preview: {doc.page_content[:100]}...")

        # Check if the text contains the researcher name
        name_in_text = name.lower() in doc.page_content.lower()
        print(f"     Contains name in text: {'Yes' if name_in_text else 'No'}")

    return results


def test_researcher_search_tool(name: str, is_name_search: bool = None):
    """
    Test the ResearcherSearchTool directly for the given researcher name.
    This helps diagnose what the agent is actually using.

    Args:
        name (str): The researcher name to search for
        is_name_search (bool, optional): Explicitly specify if this is a name search.
            If None, will be auto-detected. Defaults to None.
    """
    param_text = "with auto-detection" if is_name_search is None else f"with is_name_search={is_name_search}"
    print(f"\n=== Testing ResearcherSearchTool for: {name} {param_text} ===")

    # Initialize the tool
    tool = ResearcherSearchTool()

    # Run the tool with the name
    result = tool._run(name, is_name_search=is_name_search)

    # Print the result
    print("Tool Output:")
    print(result)

    # Check if the result contains the researcher name
    name_in_result = name.lower() in result.lower()
    print(f"Contains researcher name: {'Yes' if name_in_result else 'No'}")

    return result


def test_name_detection(query: str):
    """
    Test the name detection logic used by the ResearcherSearchTool.
    """
    print(f"\n=== Testing Name Detection Logic for: {query} ===")

    query_lower = query.lower()

    # Method 1: Check for phrases that indicate name searches
    name_search_indicators = ["who is", "find", "about", "researcher named", "dr.", "doctor", "professor"]
    is_name_search_by_phrase = any(indicator in query_lower for indicator in name_search_indicators)

    # Method 2: Check for capitalized words which likely indicate proper names
    words = query.split()
    capitalized_words = [word for word in words if len(word) > 1 and word[0].isupper() and word.lower() not in ["what", "who", "where", "when", "why", "how"]]
    has_capitalized_words = len(capitalized_words) >= 1
    is_short_query = len(words) <= 3
    is_name_search_by_capitalization = has_capitalized_words and is_short_query

    # Method 3: Check for specific known researchers by name
    important_researchers = [
        "theresa boyle", "ahmad tarhini", "noemi andor", "eric padron",
        "john cleveland", "aleksandra karolak"
    ]
    is_known_researcher = any(researcher in query_lower for researcher in important_researchers)

    # Combine all detection methods
    is_name_search = is_name_search_by_phrase or is_name_search_by_capitalization or is_known_researcher

    # Print the detection results
    print(f"Query: '{query}'")
    print(f"Detected as name search: {'Yes' if is_name_search else 'No'}")
    print("Reasons:")
    print(f"  - Phrase indicators: {'Yes' if is_name_search_by_phrase else 'No'}")
    print(f"  - Capitalized words: {'Yes' if is_name_search_by_capitalization else 'No'}")
    print(f"  - Known researcher: {'Yes' if is_known_researcher else 'No'}")
    if capitalized_words:
        print(f"Capitalized words found: {capitalized_words}")

    # Calculate the alpha value that would be used
    alpha = 0.3 if is_name_search else 0.7
    print(f"Alpha value that would be used: {alpha}")

    return is_name_search, alpha


def test_name_extraction(text: str):
    """
    Test the name extraction functions for a given text.
    """
    print(f"\n=== Testing Name Extraction ===")

    # Test URL extraction
    if text.startswith("http"):
        extracted_name = extract_name_from_url(text)
        print(f"URL: {text}")
        print(f"Extracted name from URL: '{extracted_name}'")

    # Test text extraction
    extracted_name = extract_name_from_text(text)
    print(f"Text sample: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
    print(f"Extracted name from text: '{extracted_name}'")

    return extracted_name


def test_all_known_researchers():
    """
    Run a comprehensive test on all known researchers.
    """
    print("\n===== Testing All Known Researchers =====")

    db = get_or_create_vector_db()
    results = db.get()
    texts = results['documents']
    metadatas = results['metadatas']

    # Check presence of each known researcher in the database
    for name in KNOWN_RESEARCHERS:
        print(f"\n----- Testing '{name}' -----")

        # Check direct presence in metadata
        print("\nChecking metadata:")
        name_lower = name.lower()
        found_in_metadata = False

        for metadata in metadatas:
            researcher_name = metadata.get("researcher_name", "").lower()
            meta_name = metadata.get("name", "").lower()

            if name_lower in researcher_name or name_lower in meta_name:
                found_in_metadata = True
                print(f"  FOUND in metadata: {metadata.get('researcher_name', '')} / {metadata.get('name', '')}")
                break

        if not found_in_metadata:
            print(f"  NOT FOUND in metadata")

        # Check presence in text content
        print("\nChecking text content:")
        found_in_text = False
        found_count = 0

        for text in texts:
            if name_lower in text.lower():
                found_in_text = True
                found_count += 1

        if found_in_text:
            print(f"  FOUND in {found_count} text chunks")
        else:
            print(f"  NOT FOUND in any text chunks")

        # Test keyword search
        print("\nTesting keyword search:")
        keyword_results = keyword_search(name, texts, metadatas, k=3)
        if keyword_results:
            top_score = keyword_results[0][1]
            print(f"  Found with keyword search, top score: {top_score:.4f}")
        else:
            print("  No results from keyword search")

        # Test vector search
        print("\nTesting vector search:")
        vector_results = similarity_search_with_score(name, k=3, db=db)
        if vector_results:
            top_score = vector_results[0][1]
            print(f"  Found with vector search, top distance: {top_score:.6f} (lower is better)")
        else:
            print("  No results from vector search")

        # Test hybrid search with different alphas
        for alpha in [0.1, 0.3, 0.5, 0.7]:
            print(f"\nTesting hybrid search (alpha={alpha}):")
            hybrid_results = hybrid_search(name, k=3, alpha=alpha)
            if hybrid_results:
                print(f"  Found with hybrid search (alpha={alpha})")
                print(f"  Top result: {hybrid_results[0].metadata.get('researcher_name', hybrid_results[0].metadata.get('name', 'Unknown'))}")
            else:
                print(f"  No results from hybrid search (alpha={alpha})")

        # Test ResearcherSearchTool with auto-detection
        print("\nTesting ResearcherSearchTool with auto-detection:")
        tool = ResearcherSearchTool()
        result = tool._run(name)
        if name.lower() in result.lower():
            print(f"  FOUND with ResearcherSearchTool (auto-detection)")
        else:
            print(f"  NOT FOUND with ResearcherSearchTool (auto-detection)")

        # Test ResearcherSearchTool with explicit is_name_search=True
        print("\nTesting ResearcherSearchTool with explicit is_name_search=True:")
        result_explicit = tool._run(name, is_name_search=True)
        if name.lower() in result_explicit.lower():
            print(f"  FOUND with ResearcherSearchTool (is_name_search=True)")
        else:
            print(f"  NOT FOUND with ResearcherSearchTool (is_name_search=True)")


def main():
    """
    Main function to run the tests.
    """
    parser = argparse.ArgumentParser(description="Test the researcher search functionality.")
    parser.add_argument("--name", type=str, default="Theresa Boyle", help="Name of the researcher to search for")
    parser.add_argument("--all", action="store_true", help="Test all known researchers")
    parser.add_argument("--check-db", action="store_true", help="Check the database for the researcher")
    parser.add_argument("--keyword", action="store_true", help="Test keyword search")
    parser.add_argument("--hybrid", action="store_true", help="Test hybrid search")
    parser.add_argument("--alphas", action="store_true", help="Test different alpha values")
    parser.add_argument("--vector", action="store_true", help="Test direct vector similarity")
    parser.add_argument("--tool", action="store_true", help="Test the ResearcherSearchTool")
    parser.add_argument("--explicit-name", action="store_true", help="Test with explicit is_name_search=True")
    parser.add_argument("--detection", action="store_true", help="Test name detection")
    parser.add_argument("--extraction", action="store_true", help="Test name extraction")

    args = parser.parse_args()

    # If no specific test is selected, run all tests
    run_all = not (args.check_db or args.keyword or args.hybrid or args.alphas or args.vector
                  or args.tool or args.detection or args.extraction)

    name = args.name

    if args.all:
        # Test all known researchers
        test_all_known_researchers()
        return

    # Get the vector database and results if needed
    db = get_or_create_vector_db()
    results = db.get()
    texts = results['documents']
    metadatas = results['metadatas']

    if run_all or args.check_db:
        # Check if the researcher exists in the Chroma DB
        found_in_metadata, found_in_text = check_chroma_db_for_researcher(name)

    if run_all or args.keyword:
        # Test the keyword_search function
        keyword_results = test_keyword_search(name, texts, metadatas)

    if run_all or args.hybrid:
        # Test the hybrid_search function with default alpha (0.7)
        hybrid_results = test_hybrid_search(name)

    if run_all or args.alphas:
        # Test with different alpha values
        test_with_different_alphas(name)

    if run_all or args.vector:
        # Test direct vector similarity search
        vector_results = test_direct_vector_similarity(name)

    if run_all or args.tool:
        # Test the ResearcherSearchTool with auto-detection
        tool_result = test_researcher_search_tool(name)

        # Test with explicit is_name_search if requested
        if args.explicit_name:
            tool_explicit_result = test_researcher_search_tool(name, is_name_search=True)

    if run_all or args.detection:
        # Test name detection logic
        is_name, alpha = test_name_detection(name)

    if run_all or args.extraction:
        # Test name extraction from text or URL
        if name.startswith('http'):
            extraction_result = test_name_extraction(name)
        else:
            # Get a sample text that might contain the name
            sample_text = ""
            for text in texts:
                if name.lower() in text.lower():
                    sample_text = text
                    break

            if sample_text:
                extraction_result = test_name_extraction(sample_text)
            else:
                print(f"\n=== Cannot test name extraction - no text found containing '{name}' ===")

    # Print a summary if running all tests
    if run_all:
        print("\n===== Summary =====")
        print(f"Researcher: {name}")
        print(f"Found in metadata: {'Yes' if found_in_metadata else 'No'}")
        print(f"Found in text: {'Yes' if found_in_text else 'No'}")
        print(f"Found with keyword search: {'Yes' if 'keyword_results' in locals() and keyword_results else 'No'}")
        print(f"Found with hybrid search: {'Yes' if 'hybrid_results' in locals() and hybrid_results else 'No'}")
        print(f"Found with direct vector search: {'Yes' if 'vector_results' in locals() and vector_results else 'No'}")
        print(f"Found with ResearcherSearchTool: {'Yes' if 'tool_result' in locals() and name.lower() in tool_result.lower() else 'No'}")

    if len(sys.argv) == 1:
        # If no args provided, print help
        parser.print_help()


if __name__ == "__main__":
    main()