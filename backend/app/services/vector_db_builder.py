"""
Shared module for vector database rebuild operations.

This module contains the core logic for rebuilding the vector database
from researcher profiles. It's used by both the CLI script (rebuild_db.py)
and the backend API (vector_db.py).
"""

import json
import hashlib
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.core.config import settings

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


class ResearcherChunkBuilder:
    """Builds chunks from researcher profiles with specialized methods for each chunk type."""

    def __init__(self, profile: Dict[str, Any], chunk_size: int = 1024):
        """
        Initialize the chunk builder.

        Args:
            profile: Researcher profile dictionary
            chunk_size: Maximum size for variable-length chunks
        """
        self.profile = profile
        self.chunk_size = chunk_size

        # Extract common fields
        self.researcher_id = profile.get("researcher_id", "unknown")
        self.researcher_name = profile.get("researcher_name", "Unknown")
        self.primary_program = profile.get("primary_program")
        self.department = profile.get("department")
        self.profile_url = profile.get("profile_url", "")
        self.research_interests = profile.get("research_interests", [])

        # Generate unique prefix for chunk IDs
        self.prefix = self._generate_chunk_prefix()

    def _generate_chunk_prefix(self) -> str:
        """Generate unique prefix for chunk IDs."""
        unique_data = f"{self.researcher_name}_{self.profile_url}_{self.primary_program or ''}_{self.department or ''}"
        if not unique_data.strip('_'):
            unique_data = f"profile_{datetime.now().timestamp()}"

        unique_hash = hashlib.md5(unique_data.encode()).hexdigest()[:8]

        if self.researcher_id == "unknown":
            return f"researcher_{unique_hash}_"
        return f"{self.researcher_id}_{unique_hash[:4]}_"

    def build_all_chunks(self) -> List[ResearcherChunk]:
        """
        Build all chunks for the researcher profile.

        Returns:
            List[ResearcherChunk]: All created chunks
        """
        chunks = []
        chunks.append(self._build_core_chunk())

        interests_chunk = self._build_interests_chunk()
        if interests_chunk:
            chunks.append(interests_chunk)

        chunks.extend(self._build_publication_chunks())
        chunks.extend(self._build_grant_chunks())

        logger.debug(f"Created {len(chunks)} chunks for researcher {self.researcher_name}")
        return chunks

    def _build_core_chunk(self) -> ResearcherChunk:
        """Build core information chunk."""
        core_parts = [f"Name: {self.researcher_name}"]

        # Add degrees
        degrees = self.profile.get("degrees", [])
        if degrees:
            core_parts[0] += f" ({', '.join(degrees)})"

        # Add title
        title = self.profile.get("title", "")
        if title:
            core_parts.append(f"Title: {title}")

        # Add program and department
        if self.primary_program:
            core_parts.append(f"Program: {self.primary_program}")
        if self.department:
            core_parts.append(f"Department: {self.department}")

        # Add overview
        overview = self.profile.get("overview", "")
        if overview:
            core_parts.append(f"Overview: {overview}")

        # Add education
        education = self.profile.get("education", [])
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

        return ResearcherChunk(
            chunk_id=f"{self.prefix}core",
            text=core_text,
            researcher_id=self.researcher_id,
            researcher_name=self.researcher_name,
            program=self.primary_program,
            department=self.department,
            research_interests=self.research_interests,
            chunk_type="core",
            profile_url=self.profile_url
        )

    def _build_interests_chunk(self) -> Optional[ResearcherChunk]:
        """Build research interests chunk."""
        if not self.research_interests:
            return None

        interests_text = "Research Interests:\n" + "\n".join([f"- {interest}" for interest in self.research_interests])

        return ResearcherChunk(
            chunk_id=f"{self.prefix}interests",
            text=interests_text,
            researcher_id=self.researcher_id,
            researcher_name=self.researcher_name,
            program=self.primary_program,
            department=self.department,
            research_interests=self.research_interests,
            chunk_type="interests",
            profile_url=self.profile_url
        )

    def _build_publication_chunks(self) -> List[ResearcherChunk]:
        """Build publication chunks (may span multiple chunks if large)."""
        publications = self.profile.get("publications", [])
        if not publications:
            return []

        # Format all publications
        formatted_pubs = []
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

            formatted_pubs.append("\n".join(pub_lines))

        # Split into size-appropriate chunks
        pub_chunk_texts = self._split_into_chunks(formatted_pubs)

        # Create ResearcherChunk objects
        chunks = []
        for i, chunk_text in enumerate(pub_chunk_texts):
            chunk = ResearcherChunk(
                chunk_id=f"{self.prefix}pubs_{i}",
                text=f"Publications:\n\n{chunk_text}",
                researcher_id=self.researcher_id,
                researcher_name=self.researcher_name,
                program=self.primary_program,
                department=self.department,
                research_interests=self.research_interests,
                chunk_type="publications",
                profile_url=self.profile_url
            )
            chunks.append(chunk)

        return chunks

    def _build_grant_chunks(self) -> List[ResearcherChunk]:
        """Build grant chunks (may span multiple chunks if large)."""
        grants = self.profile.get("grants", [])
        if not grants:
            return []

        # Extract grant descriptions
        grant_descriptions = [grant.get("description", "") for grant in grants if grant.get("description")]

        # Split into size-appropriate chunks
        grant_chunk_texts = self._split_into_chunks(grant_descriptions)

        # Create ResearcherChunk objects
        chunks = []
        for i, chunk_text in enumerate(grant_chunk_texts):
            chunk = ResearcherChunk(
                chunk_id=f"{self.prefix}grants_{i}",
                text=f"Grants:\n\n{chunk_text}",
                researcher_id=self.researcher_id,
                researcher_name=self.researcher_name,
                program=self.primary_program,
                department=self.department,
                research_interests=self.research_interests,
                chunk_type="grants",
                profile_url=self.profile_url
            )
            chunks.append(chunk)

        return chunks

    def _split_into_chunks(self, items: List[str]) -> List[str]:
        """
        Split a list of text items into size-appropriate chunks.

        Args:
            items: List of text items to chunk

        Returns:
            List[str]: List of combined chunk texts
        """
        result_chunks = []
        current_chunk = []
        current_size = 0

        for item in items:
            # Check if adding this item exceeds chunk size
            if current_size + len(item) > self.chunk_size and current_chunk:
                result_chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(item)
            current_size += len(item)

        # Add remaining items
        if current_chunk:
            result_chunks.append("\n\n".join(current_chunk))

        return result_chunks


def create_researcher_chunks(profile: Dict[str, Any], chunk_size: int = 1024) -> List[ResearcherChunk]:
    """
    Create chunks from a researcher profile for embedding.

    This function uses ResearcherChunkBuilder to create structured chunks
    following the chunking strategy:
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
    builder = ResearcherChunkBuilder(profile, chunk_size)
    return builder.build_all_chunks()


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
    force: bool = False,
    progress_callback: Optional[callable] = None
) -> bool:
    """
    Rebuild the vector database from scratch.

    Args:
        processed_dir: Path to processed data directory
        vector_db_dir: Path to vector database directory
        collection_name: Name of the ChromaDB collection
        backup: Whether to backup existing database
        force: Force rebuild even if database exists
        progress_callback: Optional callback function for progress updates (0.0 to 1.0)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if database exists
        chroma_db_file = vector_db_dir / "chroma.sqlite3"

        if chroma_db_file.exists() and not force:
            logger.warning("Vector database already exists. Use force=True to rebuild.")
            return False

        # Progress: 0% - Starting
        if progress_callback:
            progress_callback(0.0)

        # Backup existing database if requested
        if backup and vector_db_dir.exists():
            backup_path = backup_existing_database(vector_db_dir)
            if backup_path:
                logger.info(f"Backup created at: {backup_path}")

        # Progress: 10% - Backup complete
        if progress_callback:
            progress_callback(0.1)

        # Remove existing database
        if vector_db_dir.exists():
            logger.info(f"Removing existing database at {vector_db_dir}...")
            shutil.rmtree(vector_db_dir)

        # Create directory
        vector_db_dir.mkdir(parents=True, exist_ok=True)

        # Progress: 20% - Directory prepared
        if progress_callback:
            progress_callback(0.2)

        # Load all researcher profiles
        logger.info(f"Loading researcher profiles from {processed_dir}...")
        profiles = load_all_researcher_profiles(processed_dir)

        if not profiles:
            logger.error("No researcher profiles found!")
            return False

        # Progress: 40% - Profiles loaded
        if progress_callback:
            progress_callback(0.4)

        # Create chunks from all profiles
        logger.info("Creating chunks from researcher profiles...")
        all_chunks = []

        for i, profile in enumerate(profiles):
            chunks = create_researcher_chunks(profile)
            all_chunks.extend(chunks)

            # Update progress during chunking (40% to 60%)
            if progress_callback:
                chunk_progress = 0.4 + (0.2 * (i + 1) / len(profiles))
                progress_callback(chunk_progress)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(profiles)} researchers")

        # Progress: 60% - Chunks created
        if progress_callback:
            progress_callback(0.6)

        # Create vector database
        db = create_vector_db(all_chunks, vector_db_dir, collection_name)

        # Progress: 90% - Database created
        if progress_callback:
            progress_callback(0.9)

        # Verify the database
        count = db._collection.count()
        logger.info(f"Database verification: {count} chunks stored")

        if count != len(all_chunks):
            logger.warning(f"Chunk count mismatch: expected {len(all_chunks)}, got {count}")

        # Progress: 100% - Complete
        if progress_callback:
            progress_callback(1.0)

        logger.info("Vector database rebuild completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to rebuild vector database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
