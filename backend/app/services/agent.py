"""
Agent service for the Moffitt Agentic RAG system.

This module provides functions for handling agent operations,
including query processing and agent coordination.
"""

import uuid
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..models.query import QueryRequest, QueryResponse, QueryStatus, QueryStep, ToolCall
from .llm import generate_text, generate_structured_output
from .vector_db import similarity_search

# Setup logging
logger = logging.getLogger(__name__)

# In-memory store for query statuses
# In production, this should be replaced with a database
_query_statuses: Dict[str, Dict[str, Any]] = {}


async def process_query(
    query_id: str,
    query: str,
    query_type: str = "general",
    streaming: bool = False,
    max_results: int = 5,
) -> QueryResponse:
    """
    Process a query using the Agentic RAG system.

    Args:
        query_id: Unique ID for this query
        query: The query text
        query_type: The type of query (general, researcher, etc.)
        streaming: Whether to stream the response
        max_results: Maximum number of results to return

    Returns:
        QueryResponse: The query response
    """
    try:
        # Create a new query status
        _query_statuses[query_id] = {
            "query_id": query_id,
            "status": "processing",
            "query": query,
            "query_type": query_type,
            "start_time": datetime.now().isoformat(),
            "progress": 0.0,
            "streaming": streaming,
            "max_results": max_results,
        }

        # For now, this is a simplified version that performs a direct search
        # In a more complete implementation, this would use an agent to determine
        # the best approach and tools to use

        # Step 1: Determine what information to look for
        steps = []
        steps.append(
            QueryStep(
                thought="Analyzing the query to determine what information to search for."
            )
        )
        _query_statuses[query_id]["progress"] = 0.2

        # Step 2: Perform a search
        steps.append(
            QueryStep(
                thought="Searching for relevant researchers based on the query.",
                tool_calls=[
                    ToolCall(
                        tool_type="researcher_search",
                        input={"query": query},
                        output=None  # Will be filled in with the search results
                    )
                ]
            )
        )

        # Perform the actual search
        search_results = similarity_search(query, k=max_results)

        # Format the search results
        formatted_results = []
        for doc in search_results:
            try:
                # Check if metadata is None before trying to access it
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    logger.warning(f"Document without metadata found in search results")
                    content = doc.page_content[:300] if hasattr(doc, 'page_content') and doc.page_content else "No content available"
                    formatted_results.append(f"Content: {content}...")
                    continue

                researcher_name = doc.metadata.get("researcher_name", "Unknown Researcher")
                program = doc.metadata.get("program", "Unknown Program")
                department = doc.metadata.get("department", "Unknown Department")
                profile_url = doc.metadata.get("profile_url", "")
                content = doc.page_content[:300] if hasattr(doc, 'page_content') and doc.page_content else "No content available"
                formatted_results.append(f"Researcher: {researcher_name}\nProgram: {program}\nDepartment: {department}\nContent: {content}...\nProfile: {profile_url}")
            except Exception as doc_error:
                logger.error(f"Error processing search result document: {doc_error}")
                formatted_results.append("Error processing this search result.")

        # Update the tool call with the search results
        if steps and len(steps) > 0 and hasattr(steps[-1], 'tool_calls') and steps[-1].tool_calls and len(steps[-1].tool_calls) > 0:
            steps[-1].tool_calls[0].output = "\n\n---\n\n".join(formatted_results)
        else:
            # Handle the case where tool_calls is None or empty
            logger.warning("No tool calls found in steps, cannot update output")
            # Add a step with tool calls if needed
            if not steps or len(steps) == 0:
                steps.append(
                    QueryStep(
                        thought="Searching for relevant researchers based on the query.",
                        tool_calls=[
                            ToolCall(
                                tool_type="researcher_search",
                                input={"query": query},
                                output="\n\n---\n\n".join(formatted_results)
                            )
                        ]
                    )
                )
            elif not hasattr(steps[-1], 'tool_calls') or not steps[-1].tool_calls or len(steps[-1].tool_calls) == 0:
                steps[-1].tool_calls = [
                    ToolCall(
                        tool_type="researcher_search",
                        input={"query": query},
                        output="\n\n---\n\n".join(formatted_results)
                    )
                ]

        _query_statuses[query_id]["progress"] = 0.6

        # Step 3: Generate a response
        steps.append(
            QueryStep(
                thought="Generating a response based on the search results."
            )
        )

        # Prepare the prompt for response generation
        search_results_text = ""

        # Safely access the search results if they exist
        # Look for the step with tool calls for researcher_search
        search_step_index = -1
        for i, step in enumerate(steps):
            if (hasattr(step, 'tool_calls') and step.tool_calls and
                len(step.tool_calls) > 0 and
                step.tool_calls[0].tool_type == "researcher_search" and
                step.tool_calls[0].output):
                search_step_index = i
                break

        if search_step_index >= 0:
            search_results_text = steps[search_step_index].tool_calls[0].output
        elif formatted_results:
            search_results_text = "\n\n---\n\n".join(formatted_results)
        else:
            search_results_text = "No search results found for this query."

        prompt = f"""
        I need to answer a query about researchers at Moffitt Cancer Center.

        Query: {query}

        Here are the search results:

        {search_results_text}

        Based on these results, provide a comprehensive answer to the query. Focus on the most relevant information and cite specific researchers and their work when applicable.
        """

        # Generate the response
        answer = await generate_text(
            prompt=prompt,
            system_prompt="You are a helpful assistant providing information about researchers at Moffitt Cancer Center. Your responses should be informative, professional, and based on the search results provided.",
            temperature=0.7,
        )
        _query_statuses[query_id]["progress"] = 1.0

        # Create the final response
        response = QueryResponse(
            query_id=query_id,
            query=query,
            answer=answer,
            steps=steps,
            completed=True,
            error=None,
        )

        # Update the query status
        _query_statuses[query_id]["status"] = "completed"
        _query_statuses[query_id]["end_time"] = datetime.now().isoformat()
        _query_statuses[query_id]["answer"] = answer

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")

        # Update the query status
        if query_id in _query_statuses:
            _query_statuses[query_id]["status"] = "error"
            _query_statuses[query_id]["error"] = str(e)
            _query_statuses[query_id]["end_time"] = datetime.now().isoformat()

        # Return an error response
        return QueryResponse(
            query_id=query_id,
            query=query,
            answer=f"Error processing query: {str(e)}",
            steps=steps if 'steps' in locals() else [],
            completed=False,
            error=str(e),
        )


def query_status(query_id: str) -> Optional[QueryStatus]:
    """
    Get the status of a query.

    Args:
        query_id: The ID of the query

    Returns:
        Optional[QueryStatus]: The query status, or None if not found
    """
    if query_id not in _query_statuses:
        return None

    status = _query_statuses[query_id]
    return QueryStatus(
        query_id=query_id,
        status=status["status"],
        progress=status["progress"],
        completed=status["status"] == "completed",
        error=status.get("error"),
    )