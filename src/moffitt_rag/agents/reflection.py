"""
Reflection mechanisms for the agent's reasoning.

This module provides functions for the agent to reflect on its answers
and improve them before responding.
"""

import logging
from typing import Dict, Any, List, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ..models.llm import get_llm_model, LLMProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reflection prompt for improving answers
REFLECTION_PROMPT = """
You are a reflective research assistant for Moffitt Cancer Center.

You were given a question about researchers, and you've generated a preliminary answer.
Now it's time to reflect on your answer and improve it.

User's original question:
{question}

Your preliminary answer:
{answer}

Please reflect on your answer and improve it by considering:
1. Did you directly answer the user's question?
2. Did you provide specific details about researchers when relevant?
3. Is the information accurate and properly attributed?
4. Could you organize the information better?
5. Is there anything you should clarify or any assumptions you made?

Based on this reflection, provide an improved answer:
"""


def reflect_on_answer(
    question: str,
    answer: str,
    llm_provider: Optional[LLMProvider] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Reflect on an answer to improve it.

    Args:
        question (str): The original question
        answer (str): The preliminary answer
        llm_provider (Optional[LLMProvider], optional):
            The LLM provider to use. Defaults to None.
        model_name (Optional[str], optional):
            The model name to use. Defaults to None.

    Returns:
        str: The improved answer
    """
    logger.info("Reflecting on answer")

    # Get the language model
    llm = get_llm_model(
        provider=llm_provider,
        model_name=model_name,
        temperature=0.3  # Lower temperature for more focused reflection
    )

    # Format the reflection prompt
    reflection_input = REFLECTION_PROMPT.format(
        question=question,
        answer=answer
    )

    # Generate the improved answer
    improved_answer = llm.invoke(reflection_input)

    # If the LLM returned a message object, extract the content
    if hasattr(improved_answer, 'content'):
        improved_answer = improved_answer.content

    return improved_answer


class ReflectiveAgentExecutor:
    """
    A wrapper class around an agent executor that adds reflection capabilities.
    """

    def __init__(self, agent_executor):
        """
        Initialize the reflective agent executor.

        Args:
            agent_executor: The original agent executor
        """
        self.agent_executor = agent_executor
        logger.info("Created ReflectiveAgentExecutor wrapper")

    def invoke(self, inputs: Dict[str, Any], **kwargs):
        """
        Invoke the agent with reflection.

        Args:
            inputs: The inputs for the agent
            **kwargs: Additional arguments

        Returns:
            The improved result
        """
        logger.info("Invoking agent with reflection")

        # Call the original invoke method
        preliminary_result = self.agent_executor.invoke(inputs, **kwargs)

        # Extract the question and answer
        question = inputs if isinstance(inputs, str) else inputs.get("input", "")
        if isinstance(preliminary_result, dict):
            answer = preliminary_result.get("output", "")
        else:
            answer = str(preliminary_result)

        # Reflect on the answer
        try:
            logger.info("Reflecting on agent's answer")
            improved_answer = reflect_on_answer(
                question=question,
                answer=answer
            )
            logger.info("Reflection completed successfully")

            # Update the result
            if isinstance(preliminary_result, dict):
                preliminary_result["output"] = improved_answer
                return preliminary_result
            else:
                return improved_answer
        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            # If reflection fails, return the original result
            return preliminary_result

    def __getattr__(self, name):
        """
        Pass through any other attributes to the wrapped agent executor.

        Args:
            name: The attribute name

        Returns:
            The attribute from the wrapped agent executor
        """
        return getattr(self.agent_executor, name)


def create_reflective_agent_executor(agent_executor) -> Any:
    """
    Create a wrapper around an agent executor that adds reflection.

    Args:
        agent_executor: The original agent executor

    Returns:
        Any: The wrapped agent executor with reflection
    """
    return ReflectiveAgentExecutor(agent_executor)