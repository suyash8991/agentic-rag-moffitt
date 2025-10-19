"""
Test file to verify the updated data model changes.
"""

import json
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.moffitt_rag.data.models import ResearcherProfile, ResearcherChunk


def test_researcher_profile_model():
    """Test that the ResearcherProfile model works with updated structure."""

    # Sample data with researcher_name and department but no name field
    sample_data = {
        "profile_url": "https://www.moffitt.org/research-science/researchers/test-researcher",
        "last_updated": "2025-10-19T00:30:20.667765",
        "researcher_id": "12345",
        "degrees": ["PhD"],
        "title": "Associate Professor",
        "primary_program": "Biostatistics and Bioinformatics",
        "research_program": "Immuno-Oncology Program",
        "researcher_name": "Test Researcher",
        "department": "Biostatistics and Bioinformatics",
        "overview": "Research overview text",
        "research_interests": ["Interest 1", "Interest 2"],
        "content_hash": "hash123456"
    }

    # Create a ResearcherProfile object
    profile = ResearcherProfile(**sample_data)

    # Test that researcher_name is used correctly
    assert profile.researcher_name == "Test Researcher"
    assert profile.full_name == "Test Researcher, PhD"

    # Test department is set correctly
    assert profile.department == "Biostatistics and Bioinformatics"

    # Test to_document() method
    doc = profile.to_document()
    assert "researcher_name" in doc["metadata"]
    assert "department" in doc["metadata"]
    assert doc["metadata"]["researcher_name"] == "Test Researcher"
    # Check that 'name' is not in metadata
    assert "name" not in doc["metadata"]

    print("ResearcherProfile tests passed!")


def test_researcher_chunk_model():
    """Test that the ResearcherChunk model works with updated structure."""

    # Sample chunk data with researcher_name but no name field
    sample_chunk_data = {
        "chunk_id": "12345_abcd_core",
        "text": "Sample text content",
        "researcher_id": "12345",
        "researcher_name": "Test Researcher",
        "program": "Biostatistics and Bioinformatics",
        "department": "Biostatistics and Bioinformatics",
        "research_interests": ["Interest 1", "Interest 2"],
        "chunk_type": "core",
        "profile_url": "https://www.moffitt.org/research-science/researchers/test-researcher"
    }

    # Create a ResearcherChunk object
    chunk = ResearcherChunk(**sample_chunk_data)

    # Test that researcher_name is set correctly
    assert chunk.researcher_name == "Test Researcher"

    # Test that department is set correctly
    assert chunk.department == "Biostatistics and Bioinformatics"

    print("ResearcherChunk tests passed!")


if __name__ == "__main__":
    test_researcher_profile_model()
    test_researcher_chunk_model()
    print("All tests passed!")