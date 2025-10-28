"""
Agent service for the Moffitt Agentic RAG system.

This module provides functions for handling agent operations,
including query processing and agent coordination.
"""

import uuid
import logging
import asyncio
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from ..models.query import QueryRequest, QueryResponse, QueryStatus, QueryStep, ToolCall
from .llm import generate_text, generate_structured_output, get_llm_model
from .vector_db import similarity_search
from .tools import get_tools
from .limited_call import create_limited_call_agent_executor
from ..core.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    AGENT_PROMPT_TEMPLATE,
)
from ..utils.logging import get_logger, log_tool_event

# Setup logging
logger = get_logger(__name__)

# In-memory store for query statuses
# In production, this should be replaced with a database
_query_statuses: Dict[str, Dict[str, Any]] = {}


def create_researcher_agent(
    temperature: float = 0.2,
    max_llm_calls: int = 6
) -> Any:
    """
    Create a researcher agent for the Moffitt RAG system.

    Args:
        temperature: Temperature setting for the LLM
        max_llm_calls: Maximum number of LLM calls allowed per query

    Returns:
        Any: The agent executor
    """
    try:
        # Get the language model
        logger.info("Initializing language model...")
        llm = get_llm_model(temperature=temperature)
        logger.info("Language model initialized successfully")

        # Create the tools
        logger.info("Initializing agent tools...")
        tools = get_tools()
        logger.info(f"Successfully created {len(tools)} tools")

        # Create the prompt
        logger.info("Creating agent prompt template")
        # Extract tool names as a list for the tool_names variable
        tool_names = [tool.name for tool in tools]

        prompt = PromptTemplate.from_template(
            template=AGENT_PROMPT_TEMPLATE,
            partial_variables={
                "system_message": DEFAULT_SYSTEM_PROMPT,
                "tool_names": ", ".join(tool_names)
            }
        )
        logger.info(f"Agent prompt template created successfully with tools: {tool_names}")

        # Create the agent
        logger.info("Creating the agent with LangChain's create_react_agent")
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        logger.info("Agent created successfully")

        # Create the agent executor
        logger.info("Creating agent executor")
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        logger.info("Agent executor created successfully")

        # Add call limiting
        logger.info(f"Adding call limiting with max_calls={max_llm_calls}")
        agent_executor = create_limited_call_agent_executor(agent_executor, max_llm_calls)
        logger.info("Call limiting added successfully")

        logger.info("Researcher agent created successfully")
        return agent_executor

    except Exception as e:
        logger.error(f"Error creating researcher agent: {e}")
        raise


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
        # Log the original user query
        log_tool_event("user_query_received", {
            "query_id": query_id,
            "user_query": query,
            "query_type": query_type,
            "streaming": streaming,
            "max_results": max_results
        })

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

        # Create or get the agent
        logger.info("Creating researcher agent...")
        agent = create_researcher_agent(temperature=0.7, max_llm_calls=6)
        logger.info("Agent created successfully")

        # Initialize steps for tracking
        steps = []
        steps.append(
            QueryStep(
                thought="Initializing agent to process the query."
            )
        )
        _query_statuses[query_id]["progress"] = 0.2

        # If streaming is enabled, reject the request
        if streaming:
            raise ValueError(
                "Streaming is enabled. Use WebSocket endpoint for streaming responses."
            )

        # Process the query with the agent
        logger.info(f"Invoking agent with query: {query[:50]}...")
        _query_statuses[query_id]["progress"] = 0.5

        # Format the query as a dictionary with "input" key as expected by LangChain
        result = agent.invoke({"input": query})
        logger.info("Agent invocation successful")
        logger.debug(f"Response: {str(result)[:100]}...")

        _query_statuses[query_id]["progress"] = 0.9

        # Extract the answer from the result
        answer = result.get("output", str(result))

        # Log the agent's interpretation and actions
        log_tool_event("agent_processing_complete", {
            "query_id": query_id,
            "original_user_query": query,
            "agent_interpretation": str(result)[:500],  # First 500 chars
            "has_intermediate_steps": "intermediate_steps" in result
        })

        # Create a representation of the steps
        # For now, we'll create a simplified version since LangChain doesn't expose the internal steps directly
        if "Thought:" in answer and "Action:" in answer:
            # Extract thoughts and actions from the raw answer
            thoughts_and_actions = re.findall(r'Thought: (.*?)(?=\nAction:|$)', answer, re.DOTALL)
            actions = re.findall(r'Action: (.*?)(?=\nAction Input:|$)', answer, re.DOTALL)
            action_inputs = re.findall(r'Action Input: (.*?)(?=\nObservation:|$)', answer, re.DOTALL)
            observations = re.findall(r'Observation: (.*?)(?=\nThought:|$)', answer, re.DOTALL)

            # Create steps from the extracted information
            for i in range(len(thoughts_and_actions)):
                thought = thoughts_and_actions[i].strip() if i < len(thoughts_and_actions) else ""

                step = QueryStep(thought=thought)

                # Add tool calls if action exists
                if i < len(actions) and i < len(action_inputs):
                    action = actions[i].strip()
                    action_input = action_inputs[i].strip()

                    # Try to parse the action input as JSON
                    try:
                        import json
                        input_dict = json.loads(action_input)
                    except Exception:
                        input_dict = {"input": action_input}

                    observation = observations[i].strip() if i < len(observations) else ""

                    step.tool_calls = [
                        ToolCall(
                            tool_type=action,
                            input=input_dict,
                            output=observation
                        )
                    ]

                steps.append(step)

        # Ensure we have at least one step
        if not steps:
            steps.append(
                QueryStep(
                    thought="Processing query using agent."
                )
            )

        # Create the final step for generating the answer
        steps.append(
            QueryStep(
                thought="Generating a response based on the agent's analysis."
            )
        )

        # Extract the final answer if it follows the expected format
        if "Final Answer:" in answer:
            # Extract everything after "Final Answer:"
            answer = answer.split("Final Answer:", 1)[1].strip()

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
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Update the query status
        if query_id in _query_statuses:
            _query_statuses[query_id]["status"] = "error"
            _query_statuses[query_id]["error"] = str(e)
            _query_statuses[query_id]["end_time"] = datetime.now().isoformat()

        # Initialize steps if not defined
        if 'steps' not in locals() or not steps:
            steps = [
                QueryStep(
                    thought="Initializing agent to process the query."
                )
            ]

        # Create a user-friendly error message
        error_message = str(e)
        user_message = "I apologize, but I encountered an error while processing your query."

        # Check for specific error types and provide better messages
        if "langchain" in str(e).lower() or "openai" in str(e).lower() or "groq" in str(e).lower():
            user_message += " There seems to be an issue with the language model provider."
        elif "search" in str(e).lower() or "vector" in str(e).lower():
            user_message += " There was a problem searching the researcher database."
        elif "parse" in str(e).lower() or "json" in str(e).lower():
            user_message += " There was an issue with the response format."
        elif "timeout" in str(e).lower() or "time" in str(e).lower():
            user_message += " The operation took too long to complete."

        # Add a generic suggestion
        user_message += " Please try again with a more specific query, or contact support if the issue persists."

        # No formatting needed, use the error message directly

        # Return an error response
        return QueryResponse(
            query_id=query_id,
            query=query,
            answer=user_message,
            steps=steps,
            completed=False,
            error=error_message,
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