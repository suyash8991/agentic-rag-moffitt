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
from ..utils.langsmith import create_run_metadata, create_run_tags

# Setup logging
logger = get_logger(__name__)


class AgentResponseParser:
    """Parses agent responses and extracts structured information."""

    @staticmethod
    def parse_steps_from_answer(answer: str) -> List[QueryStep]:
        """
        Parse reasoning steps from agent answer.

        Args:
            answer: Raw agent answer text

        Returns:
            List[QueryStep]: Parsed steps
        """
        steps = []

        # Check if answer contains reasoning steps
        if "Thought:" in answer and "Action:" in answer:
            thoughts_and_actions = re.findall(r'Thought: (.*?)(?=\nAction:|$)', answer, re.DOTALL)
            actions = re.findall(r'Action: (.*?)(?=\nAction Input:|$)', answer, re.DOTALL)
            action_inputs = re.findall(r'Action Input: (.*?)(?=\nObservation:|$)', answer, re.DOTALL)
            observations = re.findall(r'Observation: (.*?)(?=\nThought:|$)', answer, re.DOTALL)

            # Create steps from extracted information
            for i in range(len(thoughts_and_actions)):
                thought = thoughts_and_actions[i].strip() if i < len(thoughts_and_actions) else ""
                step = QueryStep(thought=thought)

                # Add tool calls if action exists
                if i < len(actions) and i < len(action_inputs):
                    action = actions[i].strip()
                    action_input = action_inputs[i].strip()

                    # Parse action input as JSON
                    input_dict = AgentResponseParser._parse_action_input(action_input)
                    observation = observations[i].strip() if i < len(observations) else ""

                    step.tool_calls = [
                        ToolCall(
                            tool_type=action,
                            input=input_dict,
                            output=observation
                        )
                    ]

                steps.append(step)

        return steps

    @staticmethod
    def _parse_action_input(action_input: str) -> Dict[str, Any]:
        """
        Parse action input string to dictionary.

        Args:
            action_input: Action input string (potentially JSON)

        Returns:
            Dict[str, Any]: Parsed input dictionary
        """
        try:
            import json
            return json.loads(action_input)
        except Exception:
            return {"input": action_input}

    @staticmethod
    def extract_final_answer(answer: str) -> str:
        """
        Extract the final answer from agent output.

        Args:
            answer: Raw agent answer

        Returns:
            str: Extracted final answer
        """
        if "Final Answer:" in answer:
            return answer.split("Final Answer:", 1)[1].strip()
        return answer


class ErrorMessageFormatter:
    """Formats user-friendly error messages."""

    @staticmethod
    def format_error_message(error: Exception) -> str:
        """
        Create user-friendly error message from exception.

        Args:
            error: The exception that occurred

        Returns:
            str: User-friendly error message
        """
        base_message = "I apologize, but I encountered an error while processing your query."
        error_str = str(error).lower()

        # Check for specific error types
        if "langchain" in error_str or "openai" in error_str or "groq" in error_str:
            return f"{base_message} There seems to be an issue with the language model provider. Please try again with a more specific query, or contact support if the issue persists."
        elif "search" in error_str or "vector" in error_str:
            return f"{base_message} There was a problem searching the researcher database. Please try again with a more specific query, or contact support if the issue persists."
        elif "parse" in error_str or "json" in error_str:
            return f"{base_message} There was an issue with the response format. Please try again with a more specific query, or contact support if the issue persists."
        elif "timeout" in error_str or "time" in error_str:
            return f"{base_message} The operation took too long to complete. Please try again with a more specific query, or contact support if the issue persists."

        return f"{base_message} Please try again with a more specific query, or contact support if the issue persists."


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


        prompt = PromptTemplate.from_template(
            template=AGENT_PROMPT_TEMPLATE,
            partial_variables={
                "system_message": DEFAULT_SYSTEM_PROMPT,
            }
        )

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
    query_status_service: "QueryStatusService",
    query_type: str = "general",
    streaming: bool = False,
    max_results: int = 5,
) -> QueryResponse:
    """
    Process a query using the Agentic RAG system.

    This method orchestrates the query processing workflow, delegating
    responsibilities to specialized services for status management,
    response parsing, and error handling.

    Args:
        query_id: Unique ID for this query
        query: The query text
        query_status_service: Service for managing query status (injected)
        query_type: The type of query (general, researcher, etc.)
        streaming: Whether to stream the response
        max_results: Maximum number of results to return

    Returns:
        QueryResponse: The query response
    """
    # Initialize helper classes
    response_parser = AgentResponseParser()
    error_formatter = ErrorMessageFormatter()

    # Initialize steps list for tracking
    steps = [QueryStep(thought="Initializing agent to process the query.")]

    try:
        # Log query receipt
        log_tool_event("user_query_received", {
            "query_id": query_id,
            "user_query": query,
            "query_type": query_type,
            "streaming": streaming,
            "max_results": max_results
        })

        # Create query status
        query_status_service.create_status(query_id, query, query_type, streaming, max_results)
        query_status_service.update_progress(query_id, 0.2)

        # Validate streaming configuration
        if streaming:
            raise ValueError("Streaming is enabled. Use WebSocket endpoint for streaming responses.")

        # Create agent
        logger.info("Creating researcher agent...")
        agent = create_researcher_agent(temperature=0.7, max_llm_calls=6)
        logger.info("Agent created successfully")

        # Invoke agent with query
        logger.info(f"Invoking agent with query: {query[:50]}...")
        query_status_service.update_progress(query_id, 0.5)

        result = _invoke_agent_with_tracing(agent, query, query_id, query_type, max_results, streaming)

        logger.info("Agent invocation successful")
        logger.debug(f"Response: {str(result)[:100]}...")
        query_status_service.update_progress(query_id, 0.9)

        # Extract and process answer
        raw_answer = result.get("output", str(result))

        # Log agent processing completion
        log_tool_event("agent_processing_complete", {
            "query_id": query_id,
            "original_user_query": query,
            "agent_interpretation": str(result)[:500],
            "has_intermediate_steps": "intermediate_steps" in result
        })

        # Parse reasoning steps from answer
        parsed_steps = response_parser.parse_steps_from_answer(raw_answer)
        if parsed_steps:
            steps.extend(parsed_steps)
        else:
            steps.append(QueryStep(thought="Processing query using agent."))

        # Add final step
        steps.append(QueryStep(thought="Generating a response based on the agent's analysis."))

        # Extract final answer
        final_answer = response_parser.extract_final_answer(raw_answer)

        # Mark as completed
        query_status_service.update_progress(query_id, 1.0)
        query_status_service.mark_completed(query_id, final_answer)

        # Return successful response
        return QueryResponse(
            query_id=query_id,
            query=query,
            answer=final_answer,
            steps=steps,
            completed=True,
            error=None,
        )

    except Exception as e:
        # Log error
        logger.error(f"Error processing query: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Update status
        query_status_service.mark_error(query_id, str(e))

        # Format user-friendly error message
        user_message = error_formatter.format_error_message(e)

        # Return error response
        return QueryResponse(
            query_id=query_id,
            query=query,
            answer=user_message,
            steps=steps,
            completed=False,
            error=str(e),
        )


def _invoke_agent_with_tracing(
    agent: Any,
    query: str,
    query_id: str,
    query_type: str,
    max_results: int,
    streaming: bool
) -> Dict[str, Any]:
    """
    Invoke agent with LangSmith tracing metadata.

    Args:
        agent: The agent executor
        query: Query text
        query_id: Query identifier
        query_type: Type of query
        max_results: Maximum results
        streaming: Streaming flag

    Returns:
        Dict[str, Any]: Agent invocation result
    """
    metadata = create_run_metadata(
        query_id=query_id,
        query=query,
        query_type=query_type,
        max_results=max_results,
        streaming=streaming,
    )
    tags = create_run_tags(query_type=query_type, query_id=query_id)

    return agent.invoke(
        {"input": query},
        config={
            "metadata": metadata,
            "tags": tags,
            "run_name": f"Query: {query[:50]}..."
        }
    )




def query_status(query_id: str, query_status_service: "QueryStatusService") -> Optional[QueryStatus]:
    """
    Get the status of a query.

    Args:
        query_id: The ID of the query
        query_status_service: Service for managing query status (injected)

    Returns:
        Optional[QueryStatus]: The query status, or None if not found
    """
    return query_status_service.get_status(query_id)