"""
Check Chroma DB metadata for researcher_name field.
"""

import logging
import json
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the necessary modules
try:
    from src.moffitt_rag.db.vector_store import get_or_create_vector_db
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this script in the correct environment.")
    exit(1)

def check_db_metadata_fields():
    """
    Check what fields are present in the Chroma DB metadata.
    """
    print("\n=== Checking Chroma DB Metadata Fields ===")

    # Get the vector database
    try:
        db = get_or_create_vector_db()
    except Exception as e:
        print(f"Error getting vector database: {e}")
        return

    # Get all metadata from the database
    try:
        results = db.get()
        metadatas = results['metadatas']
        ids = results['ids']
    except Exception as e:
        print(f"Error retrieving data from vector database: {e}")
        return

    # Check what fields are present
    field_counts = {}
    for metadata in metadatas:
        for field, value in metadata.items():
            if field not in field_counts:
                field_counts[field] = 0
            field_counts[field] += 1

    print(f"\nFound {len(metadatas)} total chunks in database")
    print("\nMetadata fields present:")
    for field, count in field_counts.items():
        print(f"  {field}: {count} chunks")

    # Check specifically for researcher_name
    researcher_name_present = sum(1 for m in metadatas if "researcher_name" in m)
    print(f"\nChunks with researcher_name field: {researcher_name_present} out of {len(metadatas)}")

    # Check for non-empty researcher_name values
    non_empty_count = sum(1 for m in metadatas if m.get("researcher_name", "").strip())
    print(f"Chunks with non-empty researcher_name values: {non_empty_count}")

    # Show sample entries
    print("\nSample metadata entries:")
    for i, metadata in enumerate(metadatas[:3]):
        print(f"\nSample {i+1}:")
        for field, value in metadata.items():
            print(f"  {field}: {value}")

    # Check for "Theresa Boyle" in metadata
    theresa_in_name = [
        id for id, m in zip(ids, metadatas)
        if "theresa boyle" in m.get("name", "").lower()
    ]

    theresa_in_researcher_name = [
        id for id, m in zip(ids, metadatas)
        if "theresa boyle" in m.get("researcher_name", "").lower()
    ]

    print(f"\nChunks with 'Theresa Boyle' in name field: {len(theresa_in_name)}")
    print(f"Chunks with 'Theresa Boyle' in researcher_name field: {len(theresa_in_researcher_name)}")

    # Check profile URLs containing "theresa-boyle"
    theresa_in_url = [
        id for id, m in zip(ids, metadatas)
        if "theresa-boyle" in m.get("profile_url", "").lower()
    ]

    print(f"Chunks with 'theresa-boyle' in profile_url: {len(theresa_in_url)}")
    if theresa_in_url:
        print("\nExample chunks with 'theresa-boyle' in URL:")
        for i, id in enumerate(theresa_in_url[:3]):
            idx = ids.index(id)
            metadata = metadatas[idx]
            print(f"\n  Chunk {i+1}:")
            print(f"    ID: {id}")
            print(f"    name: '{metadata.get('name', '')}'")
            print(f"    researcher_name: '{metadata.get('researcher_name', '')}'")
            print(f"    profile_url: {metadata.get('profile_url', '')}")

if __name__ == "__main__":
    check_db_metadata_fields()