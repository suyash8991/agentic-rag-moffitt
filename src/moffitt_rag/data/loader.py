"""
Data loading functionality for researcher profiles.

This module provides functions for loading and processing researcher profiles
from the JSON files in the processed directory.
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Generator, Tuple
from pathlib import Path

from tqdm import tqdm
from ..config.config import get_settings
from .models import ResearcherProfile, ResearcherChunk

# Get settings
settings = get_settings()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_researcher_profile(file_path: str) -> Optional[ResearcherProfile]:
    """
    Load a researcher profile from a JSON file.

    Args:
        file_path (str): The path to the JSON file

    Returns:
        Optional[ResearcherProfile]: The researcher profile, or None if loading failed
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create a ResearcherProfile from the data
        profile = ResearcherProfile.model_validate(data)
        return profile
    except Exception as e:
        logger.error(f"Failed to load researcher profile from {file_path}: {e}")
        return None


def load_all_researcher_profiles() -> List[ResearcherProfile]:
    """
    Load all researcher profiles from the processed directory.

    Returns:
        List[ResearcherProfile]: The list of researcher profiles
    """
    profiles = []

    # Get a list of all JSON files in the processed directory
    json_files = [
        os.path.join(settings.processed_data_dir, f)
        for f in os.listdir(settings.processed_data_dir)
        if f.endswith(".json") and f != "summary.json"
    ]

    # Load each profile
    logger.info(f"Loading {len(json_files)} researcher profiles...")
    for file_path in tqdm(json_files):
        profile = load_researcher_profile(file_path)
        if profile:
            profiles.append(profile)

    logger.info(f"Loaded {len(profiles)} researcher profiles")
    return profiles


def create_researcher_chunks(profile: ResearcherProfile, chunk_size: int = 1024) -> List[ResearcherChunk]:
    """
    Create chunks from a researcher profile for embedding.

    Args:
        profile (ResearcherProfile): The researcher profile
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 1024.

    Returns:
        List[ResearcherChunk]: The list of chunks
    """
    chunks = []

    # Create a unique identifier based on multiple fields to ensure uniqueness
    # Use a combination of name, profile_url, and any available program/department
    unique_data = f"{profile.name}_{profile.profile_url}_{profile.primary_program or ''}_{profile.department or ''}"

    # Make sure we have at least some non-empty string for the hash
    if not unique_data.strip('_'):
        # If all fields were empty, use the current timestamp for uniqueness
        import time
        unique_data = f"profile_{time.time()}"

    unique_hash = hashlib.md5(unique_data.encode()).hexdigest()[:8]

    # Create a unique prefix for chunk IDs
    if profile.researcher_id == "unknown":
        # For unknown researcher_id, use a completely different prefix pattern
        prefix = f"researcher_{unique_hash}_"
    else:
        # For known researcher_id, include a short hash to ensure uniqueness
        prefix = f"{profile.researcher_id}_{unique_hash[:4]}_"

    # Add core information chunk
    core_info = "\n".join([
        f"Name: {profile.full_name}",
        f"Title: {profile.title}" if profile.title else "",
        f"Program: {profile.primary_program}" if profile.primary_program else "",
        f"Department: {profile.department}" if profile.department else "",
        f"Overview: {profile.overview}" if profile.overview else ""
    ])

    core_chunk = ResearcherChunk(
        chunk_id=f"{prefix}core",
        text=core_info,
        researcher_id=profile.researcher_id,
        name=profile.name,
        researcher_name=profile.researcher_name,  # Added researcher_name
        program=profile.primary_program,
        department=profile.department,
        research_interests=profile.research_interests,
        chunk_type="core",
        profile_url=profile.profile_url
    )
    chunks.append(core_chunk)

    # Add research interests chunk
    if profile.research_interests:
        interests_text = "Research Interests: " + "\n".join(profile.research_interests)
        interests_chunk = ResearcherChunk(
            chunk_id=f"{prefix}interests",
            text=interests_text,
            researcher_id=profile.researcher_id,
            name=profile.name,
            researcher_name=profile.researcher_name,  # Added researcher_name
            program=profile.primary_program,
            department=profile.department,
            research_interests=profile.research_interests,
            chunk_type="interests",
            profile_url=profile.profile_url
        )
        chunks.append(interests_chunk)

    # Add publication chunks
    if profile.publications:
        # Group publications into chunks of appropriate size
        pub_chunks = []
        current_chunk = []
        current_size = 0

        for pub in profile.publications:
            pub_text = "\n".join([
                f"Title: {pub.title}",
                f"Authors: {pub.authors}" if pub.authors else "",
                f"Journal: {pub.journal}" if pub.journal else "",
                f"Year: {pub.year}" if pub.year else ""
            ])

            # If adding this publication would exceed the chunk size,
            # create a new chunk
            if current_size + len(pub_text) > chunk_size and current_chunk:
                pub_chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(pub_text)
            current_size += len(pub_text)

        # Add any remaining publications
        if current_chunk:
            pub_chunks.append("\n\n".join(current_chunk))

        # Create chunks for each group of publications
        for i, chunk_text in enumerate(pub_chunks):
            pub_chunk = ResearcherChunk(
                chunk_id=f"{prefix}pubs_{i}",
                text=f"Publications:\n{chunk_text}",
                researcher_id=profile.researcher_id,
                name=profile.name,
                researcher_name=profile.researcher_name,  # Added researcher_name
                program=profile.primary_program,
                department=profile.department,
                research_interests=profile.research_interests,
                chunk_type="publications",
                profile_url=profile.profile_url
            )
            chunks.append(pub_chunk)

    # Add grant chunks
    if profile.grants:
        # Group grants into chunks of appropriate size
        grant_chunks = []
        current_chunk = []
        current_size = 0

        for grant in profile.grants:
            grant_text = grant.description

            # If adding this grant would exceed the chunk size,
            # create a new chunk
            if current_size + len(grant_text) > chunk_size and current_chunk:
                grant_chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(grant_text)
            current_size += len(grant_text)

        # Add any remaining grants
        if current_chunk:
            grant_chunks.append("\n\n".join(current_chunk))

        # Create chunks for each group of grants
        for i, chunk_text in enumerate(grant_chunks):
            grant_chunk = ResearcherChunk(
                chunk_id=f"{prefix}grants_{i}",
                text=f"Grants:\n{chunk_text}",
                researcher_id=profile.researcher_id,
                name=profile.name,
                researcher_name=profile.researcher_name,  # Added researcher_name
                program=profile.primary_program,
                department=profile.department,
                research_interests=profile.research_interests,
                chunk_type="grants",
                profile_url=profile.profile_url
            )
            chunks.append(grant_chunk)

    logger.debug(f"Created {len(chunks)} chunks for researcher {profile.name}")
    return chunks


def load_all_chunks() -> List[ResearcherChunk]:
    """
    Load all researcher profiles and create chunks for embedding.

    Returns:
        List[ResearcherChunk]: The list of chunks
    """
    profiles = load_all_researcher_profiles()
    all_chunks = []

    logger.info(f"Creating chunks for {len(profiles)} researcher profiles...")
    for profile in tqdm(profiles):
        chunks = create_researcher_chunks(profile)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} chunks total")
    return all_chunks


def find_duplicates(items: List[str]) -> List[str]:
    """
    Find duplicate items in a list.

    Args:
        items (List[str]): The list of items to check for duplicates

    Returns:
        List[str]: The list of duplicate items
    """
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)


def deduplicate_chunks(chunks: List[ResearcherChunk]) -> List[ResearcherChunk]:
    """
    Make chunk IDs unique by adding a counter suffix when duplicates are found.

    Args:
        chunks (List[ResearcherChunk]): The list of chunks to deduplicate

    Returns:
        List[ResearcherChunk]: The list of chunks with unique IDs
    """
    id_count = {}
    deduplicated_chunks = []

    for chunk in chunks:
        # If this ID has been seen before, add a suffix
        if chunk.chunk_id in id_count:
            id_count[chunk.chunk_id] += 1
            new_id = f"{chunk.chunk_id}_{id_count[chunk.chunk_id]}"
            logger.debug(f"Renamed duplicate chunk ID {chunk.chunk_id} to {new_id}")

            # Create a new chunk with the updated ID
            new_chunk = ResearcherChunk(
                chunk_id=new_id,
                text=chunk.text,
                researcher_id=chunk.researcher_id,
                name=chunk.name,
                researcher_name=chunk.researcher_name,  # Added researcher_name
                program=chunk.program,
                department=chunk.department,
                research_interests=chunk.research_interests,
                chunk_type=chunk.chunk_type,
                profile_url=chunk.profile_url
            )
            deduplicated_chunks.append(new_chunk)
        else:
            id_count[chunk.chunk_id] = 1
            deduplicated_chunks.append(chunk)

    return deduplicated_chunks


def get_researcher_stats() -> Dict[str, Any]:
    """
    Get statistics about the researcher profiles.

    Returns:
        Dict[str, Any]: The statistics
    """
    # Try to load the summary.json file if it exists
    summary_path = os.path.join(settings.processed_data_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # If the summary file doesn't exist, generate statistics
    profiles = load_all_researcher_profiles()

    # Count the number of profiles
    count = len(profiles)

    # Count the number of profiles by program
    programs = {}
    for profile in profiles:
        if profile.primary_program:
            programs[profile.primary_program] = programs.get(profile.primary_program, 0) + 1

    # Count the number of profiles by department
    departments = {}
    for profile in profiles:
        if profile.department:
            departments[profile.department] = departments.get(profile.department, 0) + 1

    # Count the most common research interests
    interests = {}
    for profile in profiles:
        for interest in profile.research_interests:
            interests[interest] = interests.get(interest, 0) + 1

    # Sort the interests by frequency
    top_interests = dict(sorted(interests.items(), key=lambda x: x[1], reverse=True)[:20])

    return {
        "count": count,
        "programs": programs,
        "departments": departments,
        "top_research_interests": top_interests
    }