"""
List all researcher names in the Chroma DB.
"""

import logging
import sys
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the necessary modules
from src.moffitt_rag.db.vector_store import get_or_create_vector_db


def list_all_researchers():
    """
    List all researcher names in the Chroma DB.
    """
    print("\n=== Listing all researchers in the Chroma DB ===")

    # Get the vector database
    db = get_or_create_vector_db()

    # Get all chunks from the database
    results = db.get()
    metadatas = results['metadatas']

    # Extract all researcher names from both name and researcher_name fields
    name_field_names = []
    researcher_name_field_names = []
    combined_names = []

    for metadata in metadatas:
        # Check name field
        name = metadata.get("name", "").strip()
        if name:
            name_field_names.append(name)
            combined_names.append(name)

        # Check researcher_name field
        researcher_name = metadata.get("researcher_name", "").strip()
        if researcher_name:
            researcher_name_field_names.append(researcher_name)
            if name != researcher_name:  # Only add if different from name
                combined_names.append(researcher_name)

    # Count unique names from each field
    name_field_counts = Counter(name_field_names)
    researcher_name_field_counts = Counter(researcher_name_field_names)
    combined_counts = Counter(combined_names)

    print(f"\nFound {len(name_field_counts)} unique names in 'name' field")
    print(f"Found {len(researcher_name_field_counts)} unique names in 'researcher_name' field")
    print(f"Found {len(combined_counts)} combined unique researcher names in {len(metadatas)} chunks")

    # Print fields from first metadata entry to check structure
    if metadatas:
        print("\nFields in first metadata entry:")
        for field, value in metadatas[0].items():
            print(f"  {field}: {value}")

    # Count metadata entries with researcher_name field
    has_field = sum(1 for m in metadatas if "researcher_name" in m)
    print(f"\nMetadata entries with researcher_name field: {has_field} out of {len(metadatas)}")

    # Use combined names for the rest of the script
    name_counts = combined_counts

    # Sort by frequency (descending)
    sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)

    # Print the top 50 names
    print("\nTop 50 most frequent names:")
    for i, (name, count) in enumerate(sorted_names[:50]):
        print(f"  {i+1}. {name} ({count} chunks)")

    # Check if specific researchers exist
    target_names = ["Theresa Boyle", "Noemi Andor", "Eric Padron"]
    print("\nChecking for specific researchers:")
    for name in target_names:
        found = False
        for researcher_name in name_counts:
            if name.lower() in researcher_name.lower():
                print(f"  {name} found as '{researcher_name}' ({name_counts[researcher_name]} chunks)")
                found = True
        if not found:
            print(f"  {name} not found in any researcher name")

    # Search for partial matches
    print("\nPartial name matches:")
    for name in target_names:
        first_name = name.split()[0].lower() if " " in name else name.lower()
        last_name = name.split()[-1].lower() if " " in name else ""

        first_matches = []
        last_matches = []

        for researcher_name in name_counts:
            if first_name and first_name in researcher_name.lower():
                first_matches.append((researcher_name, name_counts[researcher_name]))
            if last_name and last_name in researcher_name.lower():
                last_matches.append((researcher_name, name_counts[researcher_name]))

        print(f"\n  Matches for {name}:")
        if first_matches:
            print(f"    First name matches ({first_name}):")
            for match, count in sorted(first_matches, key=lambda x: x[1], reverse=True)[:5]:
                print(f"      {match} ({count} chunks)")
        else:
            print(f"    No first name matches found for '{first_name}'")

        if last_name:
            if last_matches:
                print(f"    Last name matches ({last_name}):")
                for match, count in sorted(last_matches, key=lambda x: x[1], reverse=True)[:5]:
                    print(f"      {match} ({count} chunks)")
            else:
                print(f"    No last name matches found for '{last_name}'")


def list_unique_programs():
    """
    List all unique program values in the Chroma DB.
    """
    print("\n=== Listing all programs in the Chroma DB ===")

    # Get the vector database
    db = get_or_create_vector_db()

    # Get all chunks from the database
    results = db.get()
    metadatas = results['metadatas']

    # Extract all program values
    programs = []
    for metadata in metadatas:
        program = metadata.get("program", "").strip()
        if program:
            programs.append(program)

    # Count unique programs
    program_counts = Counter(programs)

    print(f"\nFound {len(program_counts)} unique programs in {len(metadatas)} chunks")

    # Print all programs and their counts
    print("\nPrograms:")
    for program, count in sorted(program_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {program}: {count} chunks")


def list_empty_names():
    """
    List chunks with empty name fields.
    """
    print("\n=== Listing chunks with empty names ===")

    # Get the vector database
    db = get_or_create_vector_db()

    # Get all chunks from the database
    results = db.get()
    ids = results['ids']
    metadatas = results['metadatas']
    texts = results['documents']

    # Find chunks with empty names in both fields
    empty_name_chunks = []
    empty_researcher_name_chunks = []
    empty_both_names_chunks = []

    for i, metadata in enumerate(metadatas):
        name = metadata.get("name", "").strip()
        researcher_name = metadata.get("researcher_name", "").strip()

        if not name:
            empty_name_chunks.append((ids[i], metadata, texts[i]))

        if not researcher_name:
            empty_researcher_name_chunks.append((ids[i], metadata, texts[i]))

        if not name and not researcher_name:
            empty_both_names_chunks.append((ids[i], metadata, texts[i]))

    print(f"\nFound {len(empty_name_chunks)} chunks with empty name fields")
    print(f"Found {len(empty_researcher_name_chunks)} chunks with empty researcher_name fields")
    print(f"Found {len(empty_both_names_chunks)} chunks with both name fields empty")

    if empty_both_names_chunks:
        print("\nSample chunks with both name fields empty:")
        for i, (id, metadata, text) in enumerate(empty_both_names_chunks[:5]):
            print(f"\n  {i+1}. ID: {id}")
            print(f"     name: '{metadata.get('name', '')}'")
            print(f"     researcher_name: '{metadata.get('researcher_name', '')}'")
            print(f"     Program: {metadata.get('program', '')}")
            print(f"     URL: {metadata.get('profile_url', '')}")
            print(f"     Text preview: {text[:100]}...")

    if empty_name_chunks and not empty_both_names_chunks:
        print("\nSample chunks with empty name field but non-empty researcher_name:")
        for i, (id, metadata, text) in enumerate(empty_name_chunks[:5]):
            if metadata.get("researcher_name", "").strip():
                print(f"\n  {i+1}. ID: {id}")
                print(f"     name: '{metadata.get('name', '')}'")
                print(f"     researcher_name: '{metadata.get('researcher_name', '')}'")
                print(f"     Program: {metadata.get('program', '')}")
                print(f"     URL: {metadata.get('profile_url', '')}")
                print(f"     Text preview: {text[:100]}...")


if __name__ == "__main__":
    list_all_researchers()
    list_unique_programs()
    list_empty_names()