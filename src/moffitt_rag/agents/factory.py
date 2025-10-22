"""
Agent factory implementation for the Moffitt Agentic RAG system.

This module implements the AgentFactory interface to create agents with
various capabilities while adhering to SOLID principles.
"""

import traceback
from typing import Any, Dict, List, Optional, Union, Type

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from ..models.llm import get_llm_model, LLMProvider
from ..utils.logging import get_logger, log_agent_event
from .interfaces import AgentFactory

# Get logger for this module
logger = get_logger(__name__)


class ResearcherAgentFactory(AgentFactory):
    """
    Factory for creating researcher agents.

    This factory creates agents specialized for researcher search and
    information retrieval, implementing the AgentFactory interface.
    """

    # Default system prompt for the researcher agent
    DEFAULT_SYSTEM_PROMPT = """
    You are an intelligent research assistant for Moffitt Cancer Center.

    Your purpose is to help users find information about researchers at Moffitt,
    their expertise, research interests, and potential collaborations.

    IMPORTANT TOOL USAGE INSTRUCTIONS:
    1. You have specialized tools to search for researchers. Use these tools by following the exact format:
       - First state your thought process
       - Then specify the tool name EXACTLY as provided (e.g., "ResearcherSearch")
       - Finally provide only the necessary input for the tool in a structured format.

    2. When using tools, follow this exact pattern:
       Thought: [Your reasoning about what tool to use and why]
       Action: [EXACT TOOL NAME]
       Action Input: [JSON object with the correct arguments for the tool]

    3. Available tools and when to use them:
       - For queries about a specific research topic (e.g., "who works in immunology?"), you should first try to see if there is a relevant research program or department. Use the ProgramFilter or DepartmentFilter tools for this.
       - If you don't find a relevant program or department, or if the query is more general, use the ResearcherSearch tool.
       - ResearcherSearch: Find researchers by name or by research topic.
         - To search by name, provide the person's full name to the 'researcher_name' argument (e.g., Action Input: {"researcher_name": "Conor Lynch"}).
         - To search by topic, provide the subject matter to the 'topic' argument (e.g., Action Input: {"topic": "cancer genomics"}).
         - You must provide EITHER 'researcher_name' OR 'topic'.
       - DepartmentFilter: Find researchers in a specific academic department
       - ProgramFilter: Find researchers in a specific research program

       IMPORTANT TOOL RESTRICTIONS:
       - These are the ONLY available tools - you CANNOT use any other tools
       - You CANNOT visit websites directly - there is no 'Visit' tool

    4. Wait for each tool's response before using another tool.

    5. CRITICAL - Avoid redundant searches:
       - When a tool returns information labeled with [INFORMATION SUFFICIENCY NOTE], the returned information is sufficient to answer simple queries like "Who is X?". DO NOT make additional search queries for the same researcher.
       - For simple name queries like "Who is Theresa Boyle?", a single successful ResearcherSearch call is usually sufficient.
       - Do not search for the same researcher multiple times with slight variations.
       - Focus on synthesizing the information you already have instead of repeatedly searching.

    6. Know when to stop searching:
       - If the first search provides the researcher's name, program, and basic information, this is often enough for simple identity questions.
       - Only continue searching if you need specific additional information that wasn't in the first result.
       - If you get results about the wrong researcher, be specific in your follow-up search to avoid the same issue.
       - No more than 2-3 searches should be needed for most queries.

    Always include citations to the source information in your responses, and focus on providing concise, accurate information from the searches you've already performed.
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

    When you have enough information and are ready to provide a final answer, you MUST use the following format:

    Thought: I now have enough information to answer the question.
    Final Answer: [YOUR DETAILED ANSWER]
    Do no p
    The available tool names are: {tool_names}

    IMPORTANT GUIDELINES:
    - Only use the exact tool name (e.g., "ResearcherSearch") for the Action line.
    - Do NOT use phrases like "Utilize ResearcherSearch to find..." - just use the tool name.
    - Provide only the query in the Action Input line.
    - CHECK THE SEARCH RESULTS CAREFULLY before making additional queries.
    - PAY ATTENTION to information sufficiency notes in search results.
    - BE EFFICIENT and avoid redundant or nearly identical searches.
    - For simple queries like "Who is X?", ONE successful search is usually enough.
    - When you have sufficient information, ALWAYS use the "Final Answer:" format to provide your response.
    - DO NOT try to answer the question without using the "Final Answer:" format.
    - Synthesize the information you have before searching again.

    TOOL RESTRICTIONS:
    - ONLY use the tools listed above ({tool_names}) - no other tools are available.
    - You CANNOT directly visit websites - there is no "Visit" tool.

    Follow these guidelines for tool selection:
    1. For general researcher searches, use ResearcherSearch.
    2. To find researchers in a specific department, use DepartmentFilter.
    3. To find researchers in a specific program, use ProgramFilter.

    Example of CORRECT tool usage:
    Thought: I need to find information about Dr. Smith and his research.
    Action: ResearcherSearch
    Action Input: {{"researcher_name": "Dr. Smith"}}

    Example of EFFICIENT search behavior and CORRECT final answer format:
    Thought: I need information about Theresa Boyle.
    Action: ResearcherSearch
    Action Input: {{"researcher_name": "Theresa Boyle"}}
    [After receiving search results with basic information]
    Thought: The search provided sufficient information about Theresa Boyle, including her program (Pathology) and research focus. I will now synthesize this information to answer the query.
    Final Answer: Theresa Boyle is a researcher at Moffitt Cancer Center in the Pathology program. She is involved in RNA Panel Research funded by the Salah Foundation, working with collaborators E. Haura and F. Pellini. Her profile can be found at https://www.moffitt.org/research-science/researchers/theresa-boyle.

    Source: Moffitt Cancer Center Researcher Database

    Remember to provide source information and always be helpful and accurate.

    User Query: {input}

    {agent_scratchpad}
    """

    def __init__(self, tool_classes: Optional[List[Type[BaseTool]]] = None):
        """
        Initialize the ResearcherAgentFactory.

        Args:
            tool_classes (Optional[List[Type[BaseTool]]], optional):
                List of tool classes to use. If None, default tools will be used.
        """
        from ..tools import (
            ResearcherSearchTool,
            DepartmentFilterTool,
            ProgramFilterTool
        )

        self.tool_classes = tool_classes or [
            ResearcherSearchTool,
            DepartmentFilterTool,
            ProgramFilterTool
        ]

    def create_llm(
        self,
        provider: Optional[Union[str, LLMProvider]] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs
    ) -> BaseLanguageModel:
        """
        Create a language model for the agent.

        Args:
            provider (Optional[Union[str, LLMProvider]], optional):
                The LLM provider to use. Defaults to None.
            model_name (Optional[str], optional):
                The model name to use. Defaults to None.
            temperature (float, optional):
                The temperature to use. Defaults to 0.2.
            **kwargs: Additional arguments to pass to the language model.

        Returns:
            BaseLanguageModel: The language model

        Raises:
            ValueError: If the language model could not be created
        """
        logger.info(f"Creating language model: provider={provider}, model={model_name}")

        try:
            # Convert string provider to enum if needed
            if isinstance(provider, str):
                try:
                    provider = LLMProvider(provider.lower())
                except ValueError:
                    logger.warning(f"Invalid provider string: {provider}. Using default.")
                    provider = None

            # Special handling for Euron provider
            if provider == LLMProvider.EURON:
                logger.info("Using Euron provider with streaming disabled")
                llm = get_llm_model(
                    provider=provider,
                    model_name=model_name,
                    temperature=temperature,
                    stream=False,  # Explicitly disable streaming for Euron
                    **kwargs
                )
            else:
                llm = get_llm_model(
                    provider=provider,
                    model_name=model_name,
                    temperature=temperature,
                    **kwargs
                )

            logger.info("Language model created successfully")
            return llm

        except Exception as e:
            logger.error(f"Failed to create language model: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to create language model: {e}")

    def create_tools(self) -> List[BaseTool]:
        """
        Create tools for the agent.

        Returns:
            List[BaseTool]: The tools for the agent

        Raises:
            ValueError: If the tools could not be created
        """
        logger.info("Creating agent tools")

        try:
            tools = [tool_class() for tool_class in self.tool_classes]
            logger.info(f"Successfully created {len(tools)} tools")
            return tools

        except Exception as e:
            logger.error(f"Failed to create tools: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to create agent tools: {e}")

    def create_prompt(
        self,
        system_message: Optional[str] = None
    ) -> PromptTemplate:
        """
        Create the prompt template for the agent.

        Args:
            system_message (Optional[str], optional):
                The system message to use. If None, the default system message will be used.

        Returns:
            PromptTemplate: The prompt template

        Raises:
            ValueError: If the prompt template could not be created
        """
        logger.info("Creating agent prompt template")

        try:
            # Use the default system message if none is provided
            if system_message is None:
                logger.info("Using default system prompt")
                system_message = self.DEFAULT_SYSTEM_PROMPT
            else:
                logger.info("Using custom system prompt")

            # Create tools to extract their names
            tools = self.create_tools()
            tool_names = [tool.name for tool in tools]

            prompt = PromptTemplate.from_template(
                template=self.AGENT_PROMPT_TEMPLATE,
                partial_variables={
                    "system_message": system_message,
                    "tool_names": ", ".join(tool_names)
                }
            )

            logger.info(f"Agent prompt template created successfully with tools: {tool_names}")
            return prompt

        except Exception as e:
            logger.error(f"Failed to create prompt template: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to create agent prompt: {e}")

    def create_agent_executor(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        prompt: PromptTemplate,
        handle_parsing_errors: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> AgentExecutor:
        """
        Create an agent executor.

        Args:
            llm (BaseLanguageModel): The language model
            tools (List[BaseTool]): The tools
            prompt (PromptTemplate): The prompt template
            handle_parsing_errors (bool, optional):
                Whether to handle parsing errors. Defaults to True.
            verbose (bool, optional):
                Whether to enable verbose output. Defaults to True.
            **kwargs: Additional arguments to pass to the agent executor

        Returns:
            AgentExecutor: The agent executor

        Raises:
            ValueError: If the agent executor could not be created
        """
        logger.info("Creating agent executor")

        try:
            # Create the agent
            agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

            # Create the agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                handle_parsing_errors=handle_parsing_errors,
                verbose=verbose,
                **kwargs
            )

            logger.info("Agent executor created successfully")
            return agent_executor

        except Exception as e:
            logger.error(f"Failed to create agent executor: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to create agent executor: {e}")

    def create_agent(
        self,
        llm_provider: Optional[Union[str, LLMProvider]] = None,
        model_name: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.2,
        enable_reflection: bool = False,
        max_llm_calls: int = 6,
        **kwargs
    ) -> AgentExecutor:
        """
        Create a complete researcher agent.

        This method orchestrates the entire agent creation process,
        including LLM initialization, tool creation, prompt setup,
        and agent executor creation.

        Args:
            llm_provider (Optional[Union[str, LLMProvider]], optional):
                The LLM provider to use. Defaults to None.
            model_name (Optional[str], optional):
                The model name to use. Defaults to None.
            system_message (Optional[str], optional):
                The system message to use. Defaults to None.
            temperature (float, optional):
                The temperature to use. Defaults to 0.2.
            enable_reflection (bool, optional):
                Whether to enable reflection. Defaults to False.
            max_llm_calls (int, optional):
                Maximum number of LLM calls allowed. Defaults to 6.
            **kwargs: Additional arguments to pass to the agent.

        Returns:
            AgentExecutor: The complete agent executor
        """
        try:
            # Log the start of agent creation
            logger.info(f"Creating researcher agent: provider={llm_provider}, model={model_name}")

            # Structured logging for agent creation
            log_agent_event("agent_creation_start", {
                "provider": str(llm_provider) if llm_provider else "default",
                "model_name": model_name if model_name else "default",
                "enable_reflection": enable_reflection,
                "temperature": temperature,
                "max_llm_calls": max_llm_calls
            })

            # Initialize the language model
            llm = self.create_llm(
                provider=llm_provider,
                model_name=model_name,
                temperature=temperature
            )

            # Create the tools
            tools = self.create_tools()

            # Create the prompt template
            prompt = self.create_prompt(system_message=system_message)

            # Create the agent executor
            agent_executor = self.create_agent_executor(
                llm=llm,
                tools=tools,
                prompt=prompt,
                handle_parsing_errors=True,
                verbose=True
            )

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

            # Add call limiting
            try:
                logger.info(f"Adding call limiting with max_calls={max_llm_calls}")
                from .limited_call import create_limited_call_agent_executor
                agent_executor = create_limited_call_agent_executor(agent_executor, max_llm_calls)
                logger.info("Call limiting added successfully")
            except Exception as e:
                import traceback
                logger.error(f"Failed to add call limiting: {type(e).__name__}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.warning("Continuing without call limiting due to error")

            logger.info("Researcher agent created successfully")

            # Structured logging for agent creation completion
            log_agent_event("agent_creation_complete", {
                "provider": str(llm_provider) if llm_provider else "default",
                "model_name": model_name if model_name else "default",
                "enable_reflection": enable_reflection,
                "tool_count": len(tools)
            })

            return agent_executor

        except Exception as e:
            logger.error(f"Unhandled error in create_agent: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to create researcher agent: {e}")

# Create a singleton instance of the factory
default_factory = ResearcherAgentFactory()