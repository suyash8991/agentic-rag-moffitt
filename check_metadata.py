"""
Script to check if researcher_name is present in vector database metadata.
"""

import logging
import json
import sys
import os
import subprocess
from collections import Counter
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if running in the correct environment
try:
    from src.moffitt_rag.db.vector_store import get_or_create_vector_db
except ImportError as e:
    print(f"ImportError: {e}")
    print("\nYou need to run this script in the proper Python environment.")
    print("Please run with:")
    print("(moffitt-agentic-rag) PS C:\\Coding\\Projects\\moffitt-agentic-rag> python check_metadata.py")
    sys.exit(1)

def check_metadata_fields():
    """Check what fields are present in metadata."""
    # Get the vector database
    db = get_or_create_vector_db()

    # Get all chunks from the database
    results = db.get()
    metadatas = results['metadatas']

    # Check what fields are present in the first metadata
    if metadatas:
        print("\nFields in first metadata entry:")
        for field, value in metadatas[0].items():
            print(f"  {field}: {value[:50]}..." if isinstance(value, str) and len(value) > 50 else f"  {field}: {value}")

    # Count metadata entries with researcher_name
    researcher_name_count = sum(1 for m in metadatas if "researcher_name" in m)
    print(f"\nEntries with researcher_name field: {researcher_name_count} out of {len(metadatas)}")

    # Check for non-empty researcher_name values
    non_empty_count = sum(1 for m in metadatas if m.get("researcher_name", "").strip())
    print(f"Entries with non-empty researcher_name values: {non_empty_count}")

    # Show sample entries with researcher_name
    print("\nSample entries with researcher_name:")
    samples = [m for m in metadatas if "researcher_name" in m][:5]
    for i, sample in enumerate(samples):
        print(f"\n  Sample {i+1}:")
        print(f"    researcher_id: {sample.get('researcher_id', '')}")
        print(f"    name: '{sample.get('name', '')}'")
        print(f"    researcher_name: '{sample.get('researcher_name', '')}'")
        print(f"    program: {sample.get('program', '')}")
        print(f"    profile_url: {sample.get('profile_url', '')}")

if __name__ == "__main__":
    check_metadata_fields()