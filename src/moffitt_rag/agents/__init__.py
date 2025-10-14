"""
Agentic orchestration and reasoning components.

This package contains components for creating and using agents
to interact with researcher data.
"""

from .agent import create_researcher_agent
from .reflection import reflect_on_answer, create_reflective_agent_executor
from .examples import (
    basic_agent_example,
    run_example_queries,
    researcher_search_example,
    department_filter_example,
    researcher_similarity_example,
    collaboration_example
)

__all__ = [
    "create_researcher_agent",
    "reflect_on_answer",
    "create_reflective_agent_executor",
    "basic_agent_example",
    "run_example_queries",
    "researcher_search_example",
    "department_filter_example",
    "researcher_similarity_example",
    "collaboration_example"
]