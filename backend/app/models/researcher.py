"""
Models for researcher data.

This module contains Pydantic models for researcher profiles
and related data structures.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Publication(BaseModel):
    """Publication model."""

    title: str
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None
    pubmed_id: Optional[str] = None
    pmc_id: Optional[str] = None


class Grant(BaseModel):
    """Grant model."""

    description: str
    source: Optional[str] = None
    period: Optional[str] = None


class Education(BaseModel):
    """Education model."""

    type: str
    institution: str
    specialty: Optional[str] = None


class Contact(BaseModel):
    """Contact information model."""

    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    contact_url: Optional[str] = None


class ResearcherProfileBase(BaseModel):
    """Base model for researcher profiles."""

    researcher_id: str
    researcher_name: str = ""
    primary_program: Optional[str] = None
    department: Optional[str] = None
    profile_url: str


class ResearcherProfileSummary(ResearcherProfileBase):
    """Summary model for researcher profiles."""

    degrees: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    overview: Optional[str] = None
    research_interests: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic config for the model."""

        from_attributes = True


class ResearcherProfileDetail(ResearcherProfileSummary):
    """Detailed model for researcher profiles."""

    research_program: Optional[str] = None
    associations: List[str] = Field(default_factory=list)
    publications: List[Publication] = Field(default_factory=list)
    grants: List[Grant] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    contact: Optional[Contact] = None
    photo_url: Optional[str] = None
    last_updated: Optional[datetime] = None

    class Config:
        """Pydantic config for the model."""

        from_attributes = True


class ResearcherList(BaseModel):
    """List of researchers with pagination."""

    items: List[ResearcherProfileSummary]
    total: int
    skip: int
    limit: int