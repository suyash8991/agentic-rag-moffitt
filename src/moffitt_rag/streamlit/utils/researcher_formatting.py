"""
Researcher formatting utilities for the Streamlit application.

This module provides functions for formatting researcher information
in a structured and visually appealing way.
"""

import re
import streamlit as st
from typing import Dict, Any, List, Optional, Union

def extract_researcher_info(text: str) -> Dict[str, Any]:
    """
    Extract researcher information from search result text.

    This function parses the text to extract structured information about researchers.

    Args:
        text (str): The search result text to parse

    Returns:
        Dict[str, Any]: Structured researcher information
    """
    # Default researcher info structure
    researcher = {
        "name": "",
        "title": "",
        "program": "",
        "department": "",
        "interests": [],
        "profile_url": "",
        "collaborators": [],
        "additional_info": ""
    }

    # Extract name (usually at the beginning of the result)
    name_match = re.search(r'^(Dr\.\s*)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)', text, re.MULTILINE)
    if name_match:
        researcher["name"] = name_match.group(0).strip()

    # Extract program
    program_match = re.search(r'(?:Program|program):\s*([^,\n]+)', text)
    if program_match:
        researcher["program"] = program_match.group(1).strip()

    # Extract department
    dept_match = re.search(r'(?:Department|department):\s*([^,\n]+)', text)
    if dept_match:
        researcher["department"] = dept_match.group(1).strip()

    # Extract title
    title_match = re.search(r'(?:Title|title|Position|position):\s*([^,\n]+)', text)
    if title_match:
        researcher["title"] = title_match.group(1).strip()

    # Extract interests
    interests_match = re.search(r'(?:Research Interests|Interests|research interests|interests):\s*([^\n]+)', text)
    if interests_match:
        interests_text = interests_match.group(1).strip()
        interests = [i.strip() for i in interests_text.split(',')]
        researcher["interests"] = interests

    # Extract profile URL
    url_match = re.search(r'https?://[^\s\)]+', text)
    if url_match:
        researcher["profile_url"] = url_match.group(0)

    # Extract collaborators
    collab_match = re.search(r'(?:Collaborator|Collaborators|collaborator|collaborators):\s*([^\n]+)', text)
    if collab_match:
        collaborators_text = collab_match.group(1).strip()
        collaborators = [c.strip() for c in collaborators_text.split(',')]
        researcher["collaborators"] = collaborators

    # Any additional information not extracted
    researcher["additional_info"] = text

    return researcher

def format_researcher_card(researcher: Dict[str, Any]) -> str:
    """
    Format researcher information as an HTML card.

    Args:
        researcher (Dict[str, Any]): Structured researcher information

    Returns:
        str: HTML-formatted researcher card
    """
    # Create the card HTML
    html = f'<div class="researcher-card">'

    # Name
    if researcher.get("name"):
        html += f'<div class="researcher-name">{researcher["name"]}</div>'

    # Title
    if researcher.get("title"):
        html += f'<div class="researcher-title">{researcher["title"]}</div>'

    # Program & Department
    if researcher.get("program") or researcher.get("department"):
        html += '<div class="researcher-affiliations">'
        if researcher.get("program"):
            html += f'<span class="researcher-program">{researcher["program"]}</span>'
        if researcher.get("department"):
            html += f'<span class="researcher-department">{researcher["department"]}</span>'
        html += '</div>'

    # Research Interests
    if researcher.get("interests") and len(researcher["interests"]) > 0:
        html += '<div class="researcher-interests"><strong>Research Interests:</strong> '
        html += ", ".join(researcher["interests"])
        html += '</div>'

    # Collaborators
    if researcher.get("collaborators") and len(researcher["collaborators"]) > 0:
        html += '<div class="researcher-interests"><strong>Collaborators:</strong> '
        html += ", ".join(researcher["collaborators"])
        html += '</div>'

    # Profile Link
    if researcher.get("profile_url"):
        html += f'<a href="{researcher["profile_url"]}" target="_blank" class="researcher-link">View Researcher Profile →</a>'

    html += '</div>'

    return html
