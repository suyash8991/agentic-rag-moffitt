"""
Data models for researcher profiles.

This module defines Pydantic models for researcher profiles and related data.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Publication(BaseModel):
    """A research publication."""

    title: str
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None
    pubmed_id: Optional[str] = None
    pmc_id: Optional[str] = None


class Grant(BaseModel):
    """A research grant."""

    description: str
    source: Optional[str] = None
    period: Optional[str] = None


class Education(BaseModel):
    """Educational background."""

    type: str
    institution: str
    specialty: Optional[str] = None


class Contact(BaseModel):
    """Contact information."""

    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    contact_url: Optional[str] = None


class ResearcherProfile(BaseModel):
    """
    A researcher profile from Moffitt Cancer Center.

    This model represents the structured data for a researcher,
    including their biographical information, research interests,
    publications, and grants.
    """

    # Required fields
    researcher_id: str
    profile_url: str
    content_hash: str
    last_updated: datetime

    # Biographical information
    name: str = ""
    researcher_name: str = ""  # Added field to match JSON structure
    degrees: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    primary_program: Optional[str] = None
    research_program: Optional[str] = None
    department: Optional[str] = None

    # Research information
    overview: Optional[str] = None
    research_interests: List[str] = Field(default_factory=list)
    associations: List[str] = Field(default_factory=list)

    # Publications and grants
    publications: List[Publication] = Field(default_factory=list)
    grants: List[Grant] = Field(default_factory=list)

    # Education and contact
    education: List[Education] = Field(default_factory=list)
    contact: Optional[Contact] = None

    # Additional fields
    photo_url: Optional[str] = None

    @property
    def full_name(self) -> str:
        """
        Get the full name with degrees.

        Returns:
            str: The full name with degrees
        """
        if not self.degrees:
            return self.name

        degrees_str = ", ".join(self.degrees)
        return f"{self.name}, {degrees_str}"

    @property
    def publication_count(self) -> int:
        """
        Get the number of publications.

        Returns:
            int: The number of publications
        """
        return len(self.publications)

    @property
    def grant_count(self) -> int:
        """
        Get the number of grants.

        Returns:
            int: The number of grants
        """
        return len(self.grants)

    def to_document(self) -> Dict[str, Any]:
        """
        Convert the profile to a document for embedding.

        Returns:
            Dict[str, Any]: The document representation
        """
        return {
            "researcher_id": self.researcher_id,
            "text": self.to_text(),
            "metadata": {
                "name": self.name,
                "researcher_name": self.researcher_name,  # Added researcher_name to metadata
                "program": self.primary_program,
                "department": self.department,
                "research_interests": self.research_interests,
                "publication_count": self.publication_count,
                "grant_count": self.grant_count,
                "profile_url": self.profile_url
            }
        }

    def to_text(self) -> str:
        """
        Convert the profile to a text representation for embedding.

        Returns:
            str: The text representation
        """
        text_parts = []

        # Basic information
        text_parts.append(f"Name: {self.full_name}")

        if self.title:
            text_parts.append(f"Title: {self.title}")

        if self.primary_program:
            text_parts.append(f"Program: {self.primary_program}")

        if self.department:
            text_parts.append(f"Department: {self.department}")

        # Research information
        if self.overview:
            text_parts.append(f"Overview: {self.overview}")

        if self.research_interests:
            text_parts.append(f"Research Interests: {', '.join(self.research_interests)}")

        if self.associations:
            text_parts.append(f"Associations: {', '.join(self.associations)}")

        # Publications
        if self.publications:
            text_parts.append("Publications:")
            for i, pub in enumerate(self.publications[:10], 1):  # Limit to 10 publications
                text_parts.append(f"  {i}. {pub.title}")
                if pub.authors:
                    text_parts.append(f"     Authors: {pub.authors}")
                if pub.journal:
                    year_str = f" ({pub.year})" if pub.year else ""
                    text_parts.append(f"     Journal: {pub.journal}{year_str}")

        # Grants
        if self.grants:
            text_parts.append("Grants:")
            for i, grant in enumerate(self.grants[:5], 1):  # Limit to 5 grants
                text_parts.append(f"  {i}. {grant.description}")

        return "\n".join(text_parts)


class ResearcherChunk(BaseModel):
    """
    A chunk of a researcher profile for retrieval.

    This model represents a chunk of text from a researcher profile
    along with metadata for retrieval.
    """

    chunk_id: str
    text: str
    researcher_id: str
    name: str
    researcher_name: str = ""  # Added field to match JSON structure
    program: Optional[str] = None
    department: Optional[str] = None
    research_interests: List[str] = Field(default_factory=list)
    chunk_type: str  # 'core', 'interests', 'publications', 'grants'
    profile_url: str