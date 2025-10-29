"""
Script to rebuild the vector database from researcher profiles.

This script loads researcher profiles from the processed data directory,
creates chunks using the documented strategy, and builds a ChromaDB vector database.

Usage:
    python rebuild_db.py [--no-backup] [--force]

Options:
    --no-backup: Skip backing up the existing database
    --force: Force rebuild even if database exists
"""

import os
import sys
import json
import hashlib
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add backend to path so we can import its modules
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from app.core.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResearcherChunk:
    """Represents a chunk of researcher information."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        researcher_id: str,
        researcher_name: str,
        program: Optional[str] = None,
        department: Optional[str] = None,
        research_interests: Optional[List[str]] = None,
        chunk_type: str = "core",
        profile_url: Optional[str] = None
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.researcher_id = researcher_id
        self.researcher_name = researcher_name
        self.program = program
        self.department = department
        self.research_interests = research_interests or []
        self.chunk_type = chunk_type
        self.profile_url = profile_url


def load_researcher_profile(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a researcher profile from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing researcher data, or None if loading failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load profile from {file_path}: {e}")
        return None


def load_all_researcher_profiles(processed_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all researcher profiles from the processed directory.

    Args:
        processed_dir: Path to the processed data directory

    Returns:
        List of researcher profile dictionaries
    """
    profiles = []

    # Get all JSON files except summary.json
    json_files = [
        f for f in processed_dir.glob("*.json")
        if f.name != "summary.json" and f.name != ".gitkeep"
    ]

    logger.info(f"Loading {len(json_files)} researcher profiles...")

    for file_path in json_files:
        profile = load_researcher_profile(file_path)
        if profile:
            profiles.append(profile)

    logger.info(f"Successfully loaded {len(profiles)} researcher profiles")
    return profiles


def create_researcher_chunks(profile: Dict[str, Any], chunk_size: int = 1024) -> List[ResearcherChunk]:
    """
    Create chunks from a researcher profile for embedding.

    Implements the chunking strategy:
    - core: Basic information (name, title, program, department, overview, education)
    - interests: Research interests
    - publications: Publications (may span multiple chunks)
    - grants: Grant information (may span multiple chunks)

    Args:
        profile: The researcher profile dictionary
        chunk_size: Maximum size of each chunk (default: 1024)

    Returns:
        List of ResearcherChunk objects
    """
    chunks = []

    # Extract key fields
    researcher_id = profile.get("researcher_id", "unknown")
    researcher_name = profile.get("researcher_name", "Unknown")
    primary_program = profile.get("primary_program")
    department = profile.get("department")
    profile_url = profile.get("profile_url", "")
    research_interests = profile.get("research_interests", [])

    # Create unique hash for chunk IDs
    unique_data = f"{researcher_name}_{profile_url}_{primary_program or ''}_{department or ''}"
    if not unique_data.strip('_'):
        unique_data = f"profile_{datetime.now().timestamp()}"

    unique_hash = hashlib.md5(unique_data.encode()).hexdigest()[:8]

    # Create prefix for chunk IDs
    if researcher_id == "unknown":
        prefix = f"researcher_{unique_hash}_"
    else:
        prefix = f"{researcher_id}_{unique_hash[:4]}_"

    # 1. CORE INFORMATION CHUNK
    core_parts = [
        f"Name: {researcher_name}",
    ]

    # Add degrees if available
    degrees = profile.get("degrees", [])
    if degrees:
        core_parts[0] += f" ({', '.join(degrees)})"

    # Add title
    title = profile.get("title", "")
    if title:
        core_parts.append(f"Title: {title}")

    # Add program and department
    if primary_program:
        core_parts.append(f"Program: {primary_program}")

    if department:
        core_parts.append(f"Department: {department}")

    # Add overview
    overview = profile.get("overview", "")
    if overview:
        core_parts.append(f"Overview: {overview}")

    # Add education information
    education = profile.get("education", [])
    if education:
        edu_lines = ["Education:"]
        for edu in education:
            edu_type = edu.get("type", "")
            institution = edu.get("institution", "")
            specialty = edu.get("specialty", "")
            degree = edu.get("degree", "")

            parts = [f"  - {edu_type} at {institution}"]
            if degree:
                parts.append(f", Degree: {degree}")
            if specialty:
                parts.append(f", Specialty: {specialty}")

            edu_lines.append("".join(parts))

        core_parts.extend(edu_lines)

    core_text = "\n".join(core_parts)

    core_chunk = ResearcherChunk(
        chunk_id=f"{prefix}core",
        text=core_text,
        researcher_id=researcher_id,
        researcher_name=researcher_name,
        program=primary_program,
        department=department,
        research_interests=research_interests,
        chunk_type="core",
        profile_url=profile_url
    )
    chunks.append(core_chunk)

    # 2. RESEARCH INTERESTS CHUNK
    if research_interests:
        interests_text = "Research Interests:\n" + "\n".join([f"- {interest}" for interest in research_interests])

        interests_chunk = ResearcherChunk(
            chunk_id=f"{prefix}interests",
            text=interests_text,
            researcher_id=researcher_id,
            researcher_name=researcher_name,
            program=primary_program,
            department=department,
            research_interests=research_interests,
            chunk_type="interests",
            profile_url=profile_url
        )
        chunks.append(interests_chunk)

    # 3. PUBLICATIONS CHUNKS
    publications = profile.get("publications", [])
    if publications:
        pub_chunks = []
        current_chunk = []
        current_size = 0

        for pub in publications:
            pub_lines = [f"Title: {pub.get('title', 'Untitled')}"]

            if pub.get('authors'):
                pub_lines.append(f"Authors: {pub['authors']}")

            if pub.get('journal'):
                pub_lines.append(f"Journal: {pub['journal']}")

            if pub.get('year'):
                pub_lines.append(f"Year: {pub['year']}")

            if pub.get('pubmed_id'):
                pub_lines.append(f"PubMed ID: {pub['pubmed_id']}")

            pub_text = "\n".join(pub_lines)

            # Check if adding this publication exceeds chunk size
            if current_size + len(pub_text) > chunk_size and current_chunk:
                pub_chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(pub_text)
            current_size += len(pub_text)

        # Add remaining publications
        if current_chunk:
            pub_chunks.append("\n\n".join(current_chunk))

        # Create chunks
        for i, chunk_text in enumerate(pub_chunks):
            pub_chunk = ResearcherChunk(
                chunk_id=f"{prefix}pubs_{i}",
                text=f"Publications:\n\n{chunk_text}",
                researcher_id=researcher_id,
                researcher_name=researcher_name,
                program=primary_program,
                department=department,
                research_interests=research_interests,
                chunk_type="publications",
                profile_url=profile_url
            )
            chunks.append(pub_chunk)

    # 4. GRANTS CHUNKS
    grants = profile.get("grants", [])
    if grants:
        grant_chunks = []
        current_chunk = []
        current_size = 0

        for grant in grants:
            grant_text = grant.get("description", "")

            # Check if adding this grant exceeds chunk size
            if current_size + len(grant_text) > chunk_size and current_chunk:
                grant_chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(grant_text)
            current_size += len(grant_text)

        # Add remaining grants
        if current_chunk:
            grant_chunks.append("\n\n".join(current_chunk))

        # Create chunks
        for i, chunk_text in enumerate(grant_chunks):
            grant_chunk = ResearcherChunk(
                chunk_id=f"{prefix}grants_{i}",
                text=f"Grants:\n\n{chunk_text}",
                researcher_id=researcher_id,
                researcher_name=researcher_name,
                program=primary_program,
                department=department,
                research_interests=research_interests,
                chunk_type="grants",
                profile_url=profile_url
            )
            chunks.append(grant_chunk)

    logger.debug(f"Created {len(chunks)} chunks for researcher {researcher_name}")
    return chunks


def create_vector_db(chunks: List[ResearcherChunk], vector_db_dir: Path, collection_name: str) -> Chroma:
    """
    Create a ChromaDB vector database from researcher chunks.

    Args:
        chunks: List of ResearcherChunk objects
        vector_db_dir: Directory to store the database
        collection_name: Name of the collection

    Returns:
        Chroma database instance
    """
    logger.info(f"Creating vector database with {len(chunks)} chunks...")

    # Get embedding function
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
    embedding_function = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"trust_remote_code": True}
    )

    # Prepare data for ChromaDB
    texts = []
    metadatas = []
    ids = []

    for chunk in chunks:
        texts.append(chunk.text)

        # Convert research interests list to string
        research_interests_str = "; ".join(chunk.research_interests) if chunk.research_interests else ""

        metadatas.append({
            "researcher_id": chunk.researcher_id or "unknown",
            "researcher_name": chunk.researcher_name or "Unknown",
            "program": chunk.program or "",
            "department": chunk.department or "",
            "research_interests": research_interests_str,
            "chunk_type": chunk.chunk_type,
            "profile_url": chunk.profile_url or "",
        })

        ids.append(chunk.chunk_id)

    # Check for duplicate IDs
    if len(ids) != len(set(ids)):
        from collections import Counter
        id_counts = Counter(ids)
        duplicates = {id: count for id, count in id_counts.items() if count > 1}
        logger.error(f"Found {len(duplicates)} duplicate IDs: {list(duplicates.keys())[:5]}...")
        raise ValueError("Duplicate chunk IDs found. Each chunk must have a unique ID.")

    # Create the vector database
    logger.info("Creating ChromaDB collection...")
    db = Chroma.from_texts(
        texts=texts,
        embedding=embedding_function,
        metadatas=metadatas,
        ids=ids,
        persist_directory=str(vector_db_dir),
        collection_name=collection_name,
    )

    logger.info(f"Vector database created successfully with {len(chunks)} chunks")
    return db


def backup_existing_database(vector_db_dir: Path) -> Optional[Path]:
    """
    Create a backup of the existing vector database.

    Args:
        vector_db_dir: Path to the vector database directory

    Returns:
        Path to backup directory, or None if backup failed
    """
    if not vector_db_dir.exists():
        return None

    try:
        timestamp = int(datetime.now().timestamp())
        backup_dir = vector_db_dir.parent / f"{vector_db_dir.name}_backup_{timestamp}"

        logger.info(f"Creating backup at {backup_dir}...")
        shutil.copytree(vector_db_dir, backup_dir)
        logger.info(f"Backup created successfully")

        return backup_dir
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None


def rebuild_vector_database(
    processed_dir: Path,
    vector_db_dir: Path,
    collection_name: str,
    backup: bool = True,
    force: bool = False
) -> bool:
    """
    Rebuild the vector database from scratch.

    Args:
        processed_dir: Path to processed data directory
        vector_db_dir: Path to vector database directory
        collection_name: Name of the ChromaDB collection
        backup: Whether to backup existing database
        force: Force rebuild even if database exists

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if database exists
        chroma_db_file = vector_db_dir / "chroma.sqlite3"

        if chroma_db_file.exists() and not force:
            logger.warning("Vector database already exists. Use --force to rebuild.")
            return False

        # Backup existing database if requested
        if backup and vector_db_dir.exists():
            backup_path = backup_existing_database(vector_db_dir)
            if backup_path:
                logger.info(f"Backup created at: {backup_path}")

        # Remove existing database
        if vector_db_dir.exists():
            logger.info(f"Removing existing database at {vector_db_dir}...")
            shutil.rmtree(vector_db_dir)

        # Create directory
        vector_db_dir.mkdir(parents=True, exist_ok=True)

        # Load all researcher profiles
        logger.info(f"Loading researcher profiles from {processed_dir}...")
        profiles = load_all_researcher_profiles(processed_dir)

        if not profiles:
            logger.error("No researcher profiles found!")
            return False

        # Create chunks from all profiles
        logger.info("Creating chunks from researcher profiles...")
        all_chunks = []

        for profile in profiles:
            chunks = create_researcher_chunks(profile)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(profiles)} researchers")

        # Create vector database
        db = create_vector_db(all_chunks, vector_db_dir, collection_name)

        # Verify the database
        count = db._collection.count()
        logger.info(f"Database verification: {count} chunks stored")

        if count != len(all_chunks):
            logger.warning(f"Chunk count mismatch: expected {len(all_chunks)}, got {count}")

        logger.info("Vector database rebuild completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to rebuild vector database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main entry point for the rebuild script."""
    parser = argparse.ArgumentParser(
        description="Rebuild the Moffitt RAG vector database from researcher profiles"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up the existing database"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if database exists"
    )

    args = parser.parse_args()

    # Get paths - resolve them relative to the script location (project root)
    project_root = Path(__file__).parent

    # Use absolute paths from project root
    processed_dir = project_root / "data" / "processed"
    vector_db_dir = project_root / "data" / "vector_db"
    collection_name = settings.COLLECTION_NAME

    logger.info("=" * 80)
    logger.info("Moffitt RAG Vector Database Rebuild")
    logger.info("=" * 80)
    logger.info(f"Processed data directory: {processed_dir}")
    logger.info(f"Vector database directory: {vector_db_dir}")
    logger.info(f"Collection name: {collection_name}")
    logger.info(f"Backup enabled: {not args.no_backup}")
    logger.info(f"Force rebuild: {args.force}")
    logger.info("=" * 80)

    # Validate paths
    if not processed_dir.exists():
        logger.error(f"Processed data directory not found: {processed_dir}")
        sys.exit(1)

    # Run rebuild
    success = rebuild_vector_database(
        processed_dir=processed_dir,
        vector_db_dir=vector_db_dir,
        collection_name=collection_name,
        backup=not args.no_backup,
        force=args.force
    )

    if success:
        logger.info("✓ Vector database rebuild completed successfully!")
        sys.exit(0)
    else:
        logger.error("✗ Vector database rebuild failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
