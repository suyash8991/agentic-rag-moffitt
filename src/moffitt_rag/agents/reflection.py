"""
Reflection mechanisms for the agent's reasoning.

This module provides functions for the agent to reflect on its answers
and improve them before responding.
"""

import logging
import re
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

CRITICAL REFLECTION GUIDELINES:
For simple identity questions (e.g., "Who is X?"):
- BE CONCISE - such questions need direct, focused answers
- AVOID OVERTHINKING - if you found basic info about the researcher, that's often sufficient
- DO NOT APOLOGIZE for limitations if you provided accurate information
- DO NOT claim you need more information if you already have enough to answer the basic question
- PRIORITIZE providing a clean, clear answer with what you know

For all questions, please reflect on your answer and improve it by considering:
1. Did you directly answer the user's question? (Most important criteria!)
2. Did you provide specific details about researchers when relevant?
3. Is the information accurate and properly attributed?
4. Is the answer well-organized and concise?
5. Did you avoid redundant or contradictory statements?
6. Are you making good use of the information you have already obtained?

COMMON REFLECTION PITFALLS TO AVOID:
- Claiming you need more information when you have enough for a basic answer
- Apologizing for limitations when you've provided accurate information
- Making the answer longer or more complex than necessary
- Adding unfounded speculations beyond what the data supports
- Claiming you need to search for information you already found
- Overqualifying statements that are clearly supported by the data

Based on this reflection, provide an improved answer that directly addresses the user's question
in a clear, concise, and accurate manner:
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

    # Check if this is a simple identity query that might not need complex reflection
    is_simple_identity_query = False
    if re.search(r'^who\s+is\s+\w+', question.lower()):
        is_simple_identity_query = True
        logger.info("Detected simple identity query - will use more focused reflection")

    # Get the language model
    llm = get_llm_model(
        provider=llm_provider,
        model_name=model_name,
        # Use even lower temperature for simple identity queries to ensure concise, focused answers
        temperature=0.2 if is_simple_identity_query else 0.3
    )

    # Format the reflection prompt
    reflection_input = REFLECTION_PROMPT.format(
        question=question,
        answer=answer
    )

    # Add special note for simple identity queries to prevent overthinking
    if is_simple_identity_query:
        # Check if the answer already contains sufficient information
        has_researcher_info = "researcher:" in answer.lower() and "program:" in answer.lower()
        has_profile_info = "profile:" in answer.lower()

        if has_researcher_info and has_profile_info and len(answer) < 1500:
            logger.info("Answer appears to already contain essential information for identity query")
            reflection_input += "\n\nNote: This is a simple identity question with an answer that already contains basic researcher information. Focus on ensuring the answer is direct, concise, and properly formatted rather than seeking additional information."

    # Generate the improved answer
    improved_answer = llm.invoke(reflection_input)

    # If the LLM returned a message object, extract the content
    if hasattr(improved_answer, 'content'):
        improved_answer = improved_answer.content

    # Post-processing for simple identity queries to ensure conciseness
    if is_simple_identity_query:
        # If the improved answer is very long, it might be overthinking
        if len(improved_answer) > 2000:
            logger.warning("Reflected answer for simple identity query is too long - applying post-processing")
            # Extract the most relevant parts (first few paragraphs often contain the core answer)
            paragraphs = improved_answer.split("\n\n")
            # Keep first 2-3 substantive paragraphs
            substantive_paragraphs = [p for p in paragraphs if len(p) > 50][:3]
            improved_answer = "\n\n".join(substantive_paragraphs)

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