#!/usr/bin/env python
"""
Test script to verify that education fields are properly included in researcher chunks.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add the src directory to the path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from moffitt_rag.data.models import ResearcherProfile, Education
from moffitt_rag.data.loader import create_researcher_chunks

# Create a test researcher profile with education data
test_profile = ResearcherProfile(
    researcher_id="test123",
    profile_url="https://example.com/researcher/test123",
    content_hash="abcdef1234567890",
    last_updated=datetime.now(),
    researcher_name="Jane Doe",
    degrees=["MD", "PhD"],
    title="Associate Professor",
    primary_program="Cancer Biology",
    department="Oncology",
    overview="Researcher specializing in cancer genetics.",
    research_interests=["Cancer Genetics", "Epigenetics", "Drug Development"],
    education=[
        Education(
            type="PhD",
            institution="University of California, San Francisco",
            specialty="Cancer Biology"
        ),
        Education(
            type="MD",
            institution="Harvard Medical School",
            specialty="Oncology"
        )
    ]
)

# Create chunks from the profile
chunks = create_researcher_chunks(test_profile)

# Print the core chunk to verify education field is included
print("=== Core Chunk ===")
core_chunk = next(chunk for chunk in chunks if chunk.chunk_type == "core")
print(core_chunk.text)

# Print the text representation to verify education field is included
print("\n=== Text Representation ===")
print(test_profile.to_text())