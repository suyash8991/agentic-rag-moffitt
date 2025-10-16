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

IMPORTANT TOOL USAGE INSTRUCTIONS:
1. You have specialized tools to search for researchers. Use these tools by following the exact format:
   - First state your thought process
   - Then specify the tool name EXACTLY as provided (e.g., "ResearcherSearch")
   - Finally provide only the necessary input for the tool

2. When using tools, follow this exact pattern:
   Thought: [Your reasoning about what tool to use and why]
   Action: [EXACT TOOL NAME]
   Action Input: [Only the search query or filter term]

3. Available tools and when to use them:
   - ResearcherSearch: Find researchers by expertise, interests, or name
   - DepartmentFilter: Find researchers in a specific academic department
   - ProgramFilterTool: Find researchers in a specific research program
   - InterestMatchTool: Find researchers with similar research interests
   - CollaborationTool: Discover potential collaborations between research areas

4. Wait for each tool's response before using another tool.

Always include citations to the source information in your responses.
"""

# Define the agent prompt template
AGENT_PROMPT_TEMPLATE = """
{system_message}

You have access to the following tools:

{tools}

When using tools, you MUST follow this exact format:

Thought: I need to find information about X.
Action: [TOOL NAME]
Action Input: [INPUT FOR THE TOOL]

The available tool names are: {tool_names}

IMPORTANT:
- Only use the exact tool name (e.g., "ResearcherSearch") for the Action line
- Do NOT use phrases like "Utilize ResearcherSearch to find..." - just use the tool name
- Provide only the query in the Action Input line

Follow these guidelines for tool selection:
1. For general researcher searches, use ResearcherSearch
2. To find researchers in a specific department, use DepartmentFilter
3. To find researchers in a specific program, use ProgramFilter
4. To find researchers with similar interests, use InterestMatch
5. To discover potential collaborations, use Collaboration

Example of CORRECT format:
Thought: I need to find information about Dr. Smith and his research.
Action: ResearcherSearch
Action Input: Dr. Smith

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
    try:
        # Log the start of agent creation
        logger.info(f"Creating researcher agent: provider={llm_provider}, model={model_name}, reflection={enable_reflection}")

        # Get the language model
        logger.info("Initializing language model...")
        try:
            # Check if using Euron and handle it specially
            if llm_provider == LLMProvider.EURON:
                logger.info("Using Euron provider with streaming disabled")
                llm = get_llm_model(
                    provider=llm_provider,
                    model_name=model_name,
                    temperature=temperature,
                    stream=False  # Explicitly disable streaming for Euron
                )
            else:
                llm = get_llm_model(
                    provider=llm_provider,
                    model_name=model_name,
                    temperature=temperature
                )
            logger.info("Language model initialized successfully")
        except Exception as e:
            import traceback
            logger.error(f"Failed to initialize language model: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to initialize language model: {e}") from e

        # Create the tools
        logger.info("Initializing agent tools...")
        try:
            tools = [
                ResearcherSearchTool(),
                DepartmentFilterTool(),
                ProgramFilterTool(),
                InterestMatchTool(),
                CollaborationTool()
            ]
            logger.info(f"Successfully created {len(tools)} tools")
        except Exception as e:
            import traceback
            logger.error(f"Failed to initialize tools: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to initialize agent tools: {e}") from e

        # Use the default system message if none is provided
        if system_message is None:
            logger.info("Using default system prompt")
            system_message = DEFAULT_SYSTEM_PROMPT
        else:
            logger.info("Using custom system prompt")

        # Create the prompt
        logger.info("Creating agent prompt template")
        try:
            # Extract tool names as a list for the tool_names variable
            tool_names = [tool.name for tool in tools]

            prompt = PromptTemplate.from_template(
                template=AGENT_PROMPT_TEMPLATE,
                partial_variables={
                    "system_message": system_message,
                    "tool_names": ", ".join(tool_names)
                }
            )
            logger.info(f"Agent prompt template created successfully with tools: {tool_names}")
        except Exception as e:
            import traceback
            logger.error(f"Failed to create prompt template: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to create agent prompt: {e}") from e

        # Create the agent
        logger.info("Creating the agent with LangChain's create_react_agent")
        try:
            agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
            logger.info("Agent created successfully")
        except Exception as e:
            import traceback
            logger.error(f"Failed to create agent: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to create agent: {e}") from e

        # Create the agent executor
        logger.info("Creating agent executor")
        try:
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True
            )
            logger.info("Agent executor created successfully")
        except Exception as e:
            import traceback
            logger.error(f"Failed to create agent executor: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to create agent executor: {e}") from e

        # Add reflection if enabled
        if enable_reflection:
            try:
                logger.info("Adding reflection capability to agent")
                from .reflection import create_reflective_agent_executor
                agent_executor = create_reflective_agent_executor(agent_executor)
                logger.info("Reflection capability added successfully")
            except Exception as e:
                import traceback
                logger.error(f"Failed to add reflection: {type(e).__name__}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.warning("Continuing without reflection due to error")

        logger.info("Researcher agent created successfully")
        return agent_executor

    except Exception as e:
        import traceback
        logger.error(f"Unhandled error in create_researcher_agent: {type(e).__name__}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise