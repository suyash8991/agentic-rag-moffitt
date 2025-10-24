"""
Models for query handling.

This module contains Pydantic models for handling queries
and responses in the RAG system.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Type of query being performed."""

    GENERAL = "general"
    RESEARCHER = "researcher"
    DEPARTMENT = "department"
    PROGRAM = "program"
    COLLABORATION = "collaboration"


class ToolType(str, Enum):
    """Type of tool to use for the query."""

    RESEARCHER_SEARCH = "researcher_search"
    DEPARTMENT_FILTER = "department_filter"
    PROGRAM_FILTER = "program_filter"
    INTEREST_MATCH = "interest_match"
    COLLABORATION = "collaboration"


class QueryRequest(BaseModel):
    """Query request model."""

    query: str
    query_type: Optional[QueryType] = QueryType.GENERAL
    streaming: bool = True
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results to return")


class ToolCall(BaseModel):
    """Tool call model."""

    tool_type: ToolType
    input: Dict[str, Any]
    output: Optional[str] = None


class QueryStep(BaseModel):
    """Step in the query processing."""

    thought: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class QueryResponse(BaseModel):
    """Query response model."""

    query_id: str
    query: str
    answer: str
    steps: List[QueryStep] = Field(default_factory=list)
    completed: bool = False
    error: Optional[str] = None


class QueryStatus(BaseModel):
    """Status of a query."""

    query_id: str
    status: str
    progress: float = 0.0
    completed: bool = False
    error: Optional[str] = None


class StreamingMessage(BaseModel):
    """Message for streaming responses."""

    message_type: str  # "thought", "tool_call", "tool_result", "answer", "error", "complete"
    content: str
    data: Optional[Dict[str, Any]] = None