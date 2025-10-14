"""
Example usage of the researcher agent.

This module provides examples of how to use the researcher agent
to answer queries about Moffitt Cancer Center researchers.
"""

import logging
from typing import Dict, Any, List, Optional

from .agent import create_researcher_agent
from ..models.llm import LLMProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def basic_agent_example(query: str) -> str:
    """
    Run a basic example using the researcher agent.

    Args:
        query (str): The query to ask the agent

    Returns:
        str: The agent's response
    """
    # Create the agent
    agent = create_researcher_agent(
        # You can specify a provider if needed
        # llm_provider=LLMProvider.GROQ,
        # model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        enable_reflection=True
    )

    # Run the agent
    response = agent.invoke({"input": query})

    # Extract the output
    if isinstance(response, dict) and "output" in response:
        return response["output"]
    else:
        return str(response)


def run_example_queries() -> Dict[str, str]:
    """
    Run example queries to demonstrate the agent's capabilities.

    Returns:
        Dict[str, str]: Dictionary of queries and responses
    """
    example_queries = [
        "Who studies cancer evolution at Moffitt?",
        "Tell me about researchers in the Immunology department",
        "Find researchers similar to Robert Gatenby",
        "What potential collaborations exist between Biostatistics and Cancer Epidemiology?",
        "Which researchers at Moffitt study immunotherapy resistance mechanisms?"
    ]

    results = {}
    for query in example_queries:
        logger.info(f"Running example query: {query}")
        response = basic_agent_example(query)
        results[query] = response

    return results


# Specific examples for different agent capabilities
def researcher_search_example(topic: str) -> str:
    """
    Example of using the agent to search for researchers by topic.

    Args:
        topic (str): The research topic to search for

    Returns:
        str: The agent's response
    """
    query = f"Who studies {topic} at Moffitt Cancer Center?"
    return basic_agent_example(query)


def department_filter_example(department: str) -> str:
    """
    Example of using the agent to filter researchers by department.

    Args:
        department (str): The department to filter by

    Returns:
        str: The agent's response
    """
    query = f"Tell me about researchers in the {department} department"
    return basic_agent_example(query)


def researcher_similarity_example(researcher_name: str) -> str:
    """
    Example of using the agent to find similar researchers.

    Args:
        researcher_name (str): The name of the researcher

    Returns:
        str: The agent's response
    """
    query = f"Find researchers similar to {researcher_name}"
    return basic_agent_example(query)


def collaboration_example(dept1: str, dept2: str) -> str:
    """
    Example of using the agent to find potential collaborations.

    Args:
        dept1 (str): The first department
        dept2 (str): The second department

    Returns:
        str: The agent's response
    """
    query = f"What potential collaborations exist between {dept1} and {dept2}?"
    return basic_agent_example(query)