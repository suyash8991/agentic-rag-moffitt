"""
Agent implementation for the Moffitt RAG system.

This module provides functions for creating and using a researcher agent
that can answer queries about Moffitt Cancer Center researchers.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ..tools import (
    ResearcherSearchTool,
    DepartmentFilterTool,
    ProgramFilterTool,
    InterestMatchTool,
    CollaborationTool
)
from ..models.llm import get_llm_model, LLMProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default system prompt for the researcher agent
DEFAULT_SYSTEM_PROMPT = """
You are an intelligent research assistant for Moffitt Cancer Center.

Your purpose is to help users find information about researchers at Moffitt,
their expertise, research interests, and potential collaborations.

Use the available tools to search for researchers, filter by department or program,
find similar research interests, or discover potential collaborations.

Always include citations to the source information in your responses.
"""

# Define the agent prompt template
AGENT_PROMPT_TEMPLATE = """
{system_message}

You have access to the following tools:
{tools}

Use the tools to answer the user's query. Follow these guidelines:
1. For general researcher searches, use ResearcherSearch
2. To find researchers in a specific department, use DepartmentFilter
3. To find researchers in a specific program, use ProgramFilter
4. To find researchers with similar interests, use InterestMatch
5. To discover potential collaborations, use Collaboration

Remember to provide source information and always be helpful and accurate.

User Query: {input}

{agent_scratchpad}
"""


def create_researcher_agent(
    llm_provider: Optional[LLMProvider] = None,
    model_name: Optional[str] = None,
    system_message: Optional[str] = None,
    temperature: float = 0.2,
    enable_reflection: bool = True
) -> AgentExecutor:
    """
    Create a researcher agent for the Moffitt RAG system.

    Args:
        llm_provider (Optional[LLMProvider], optional):
            The LLM provider to use. Defaults to None.
        model_name (Optional[str], optional):
            The model name to use. Defaults to None.
        system_message (Optional[str], optional):
            The system message to use. Defaults to None.
        temperature (float, optional):
            The temperature to use. Defaults to 0.2.
        enable_reflection (bool, optional):
            Whether to enable reflection. Defaults to True.

    Returns:
        AgentExecutor: The agent executor
    """
    # Get the language model
    llm = get_llm_model(
        provider=llm_provider,
        model_name=model_name,
        temperature=temperature
    )

    # Create the tools
    tools = [
        ResearcherSearchTool(),
        DepartmentFilterTool(),
        ProgramFilterTool(),
        InterestMatchTool(),
        CollaborationTool()
    ]

    # Use the default system message if none is provided
    if system_message is None:
        system_message = DEFAULT_SYSTEM_PROMPT

    # Create the prompt
    prompt = PromptTemplate.from_template(
        template=AGENT_PROMPT_TEMPLATE,
        partial_variables={"system_message": system_message}
    )

    # Create the agent
    logger.info("Creating researcher agent")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # Add reflection if enabled
    if enable_reflection:
        from .reflection import create_reflective_agent_executor
        agent_executor = create_reflective_agent_executor(agent_executor)
        logger.info("Enabled reflection for researcher agent")

    return agent_executor