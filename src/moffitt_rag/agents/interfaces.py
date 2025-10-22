"""
Agent interfaces and capability abstractions for the Moffitt Agentic RAG system.

This module defines interfaces for different agent capabilities and abstractions
to support the SOLID principles in the agent implementation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import BasePromptTemplate


class AgentCapability(ABC):
    """Base interface for all agent capabilities."""
    pass


class ReflectionCapable(AgentCapability):
    """Interface for agents that can reflect on their answers."""

    @abstractmethod
    def reflect_on_answer(self, question: str, answer: str) -> str:
        """
        Reflect on an answer to improve it.

        Args:
            question (str): The original question
            answer (str): The answer to reflect on

        Returns:
            str: The improved answer
        """
        pass


class ToolUsageCapable(AgentCapability):
    """Interface for agents that can use tools."""

    @abstractmethod
    def get_available_tools(self) -> List[BaseTool]:
        """
        Get the list of available tools.

        Returns:
            List[BaseTool]: The available tools
        """
        pass

    @abstractmethod
    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent.

        Args:
            tool (BaseTool): The tool to add
        """
        pass


class CallLimitCapable(AgentCapability):
    """Interface for agents with call limiting."""

    @abstractmethod
    def get_max_calls(self) -> int:
        """
        Get the maximum number of allowed calls.

        Returns:
            int: The maximum number of calls
        """
        pass

    @abstractmethod
    def get_remaining_calls(self) -> int:
        """
        Get the remaining number of calls.

        Returns:
            int: The remaining number of calls
        """
        pass

    @abstractmethod
    def set_max_calls(self, max_calls: int) -> None:
        """
        Set the maximum number of allowed calls.

        Args:
            max_calls (int): The maximum number of calls
        """
        pass


class AgentFactory(ABC):
    """Interface for agent factories."""

    @abstractmethod
    def create_llm(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs
    ) -> BaseLanguageModel:
        """
        Create a language model.

        Args:
            provider (Optional[str], optional): The provider to use.
            model_name (Optional[str], optional): The model name to use.
            temperature (float, optional): The temperature. Defaults to 0.2.
            **kwargs: Additional arguments to pass to the language model.

        Returns:
            BaseLanguageModel: The language model
        """
        pass

    @abstractmethod
    def create_tools(self) -> List[BaseTool]:
        """
        Create the tools for the agent.

        Returns:
            List[BaseTool]: The tools
        """
        pass

    @abstractmethod
    def create_prompt(
        self,
        system_message: Optional[str] = None
    ) -> BasePromptTemplate:
        """
        Create the prompt template for the agent.

        Args:
            system_message (Optional[str], optional): The system message.

        Returns:
            BasePromptTemplate: The prompt template
        """
        pass

    @abstractmethod
    def create_agent_executor(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        prompt: BasePromptTemplate,
        **kwargs
    ) -> Any:
        """
        Create an agent executor.

        Args:
            llm (BaseLanguageModel): The language model
            tools (List[BaseTool]): The tools
            prompt (BasePromptTemplate): The prompt template
            **kwargs: Additional arguments to pass to the agent executor

        Returns:
            Any: The agent executor
        """
        pass

    @abstractmethod
    def create_agent(
        self,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.2,
        enable_reflection: bool = False,
        max_llm_calls: int = 6,
        **kwargs
    ) -> Any:
        """
        Create an agent.

        Args:
            llm_provider (Optional[str], optional): The LLM provider to use.
            model_name (Optional[str], optional): The model name to use.
            system_message (Optional[str], optional): The system message.
            temperature (float, optional): The temperature.
            enable_reflection (bool, optional): Whether to enable reflection.
            max_llm_calls (int, optional): Maximum number of LLM calls allowed.
            **kwargs: Additional arguments to pass to the agent.

        Returns:
            Any: The agent
        """
        pass