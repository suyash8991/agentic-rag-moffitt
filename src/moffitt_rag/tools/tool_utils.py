"""
Utility functions for tools in the Moffitt RAG system.

This module provides utility functions for parsing inputs and formatting outputs
for the tools in the system.
"""

import re
import json
import ast
from typing import Dict, Any, Optional

from ..utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


def coerce_to_dict(input_str: str) -> Dict[str, Any]:
    """
    Convert a string to a dictionary.

    This function tries multiple approaches to parse a string into a dictionary:
    1. Direct JSON parsing
    2. Finding and parsing the first JSON block
    3. Parsing "key: value" lines

    Args:
        input_str (str): The string to convert

    Returns:
        Dict[str, Any]: The parsed dictionary
    """
    s = (input_str or "").strip()

    # 1) direct JSON
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2) first {...} block
    m = re.search(r'\{.*?\}', s, flags=re.DOTALL)
    if m:
        block = m.group(0)
        for parser in (json.loads, ast.literal_eval):
            try:
                data = parser(block)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    # 3) simple "key: value" lines fallback
    kv = {}
    for line in s.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().strip('"\'')
            v = v.strip().strip('"\'')
            if k:
                kv[k] = v
    return kv


def extract_name_from_url(url: str) -> str:
    """
    Extract a researcher's name from a profile URL.

    Args:
        url (str): The profile URL

    Returns:
        str: The extracted name or empty string if extraction fails
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
    Extract a researcher's name from text content.

    Args:
        text (str): The text content

    Returns:
        str: The extracted name or empty string if extraction fails
    """
    # Try to find name in a "Name: " pattern
    name_match = re.search(r'Name:\s*([^,\n]+)', text)
    if name_match:
        name = name_match.group(1).strip()
        if name and name != ',':
            return name

    # Try to find a name at the beginning followed by degrees
    degrees_match = re.search(r'^([^,]+),\s*([A-Z][A-Za-z]*\.?(?:\s+[A-Z][A-Za-z]*\.?)*)', text)
    if degrees_match:
        return degrees_match.group(1).strip()

    return ""


def format_researcher_result(
    researcher_name: str,
    program: str,
    chunk_type: str,
    content: str,
    profile_url: str
) -> str:
    """
    Format a researcher result.

    Args:
        researcher_name (str): The researcher's name
        program (str): The researcher's program
        chunk_type (str): The chunk type (core, interests, etc.)
        content (str): The content
        profile_url (str): The profile URL

    Returns:
        str: The formatted result
    """
    return (
        f"Researcher: {researcher_name}\n"
        f"Program: {program}\n"
        f"Chunk Type: {chunk_type}\n"
        f"Content: {content}\n"
        f"Profile: {profile_url}\n"
    )


def join_results(results: list, separator: str = "\n\n---\n\n") -> str:
    """
    Join multiple results with a separator.

    Args:
        results (list): The results to join
        separator (str, optional): The separator to use.
            Defaults to "\n\n---\n\n".

    Returns:
        str: The joined results
    """
    return separator.join(results)