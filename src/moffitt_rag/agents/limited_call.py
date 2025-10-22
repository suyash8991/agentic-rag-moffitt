"""
Call-limited agent executor for the Moffitt RAG system.

This module provides a wrapper for LangChain agent executors that limits
the number of LLM calls that can be made during a single query to prevent
excessive API usage and improve efficiency.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from langchain.agents import AgentExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LimitedCallAgentExecutor:
    """
    A wrapper around LangChain's AgentExecutor that limits the number of LLM calls.

    This class enforces a maximum number of LLM calls per query, automatically
    terminating execution and returning the best result so far when the limit
    is reached.
    """

    def __init__(self, agent_executor: AgentExecutor, max_calls: int = 6):
        """
        Initialize the LimitedCallAgentExecutor.

        Args:
            agent_executor (AgentExecutor): The original agent executor to wrap
            max_calls (int, optional): Maximum LLM calls per query. Defaults to 3.
        """
        self.agent_executor = agent_executor
        self.max_calls = max_calls
        self.calls_counter = 0
        self.intermediate_results = []
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created LimitedCallAgentExecutor with max_calls={max_calls}")

    def invoke(self, inputs: Union[Dict[str, Any], str], **kwargs) -> Dict[str, Any]:
        """
        Invoke the agent with a call limit.

        Args:
            inputs (Union[Dict[str, Any], str]): The inputs for the agent
            **kwargs: Additional arguments for the agent executor

        Returns:
            Dict[str, Any]: The agent's response
        """
        # Reset counter for new queries
        self.calls_counter = 0
        self.intermediate_results = []

        # Standardize input to dictionary format
        if isinstance(inputs, str):
            input_dict = {"input": inputs}
        else:
            input_dict = inputs

        # Extract the query for logging
        query = input_dict.get("input", "No query provided")
        self.logger.info(f"Starting execution with call limit {self.max_calls} for query: {query}")

        # Run the agent with call limiting
        result = self._run_with_limits(input_dict, **kwargs)

        # Log the result
        self.logger.info(f"Completed execution after {self.calls_counter} calls")

        return result

    def _run_with_limits(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent with call limits.

        Args:
            inputs (Dict[str, Any]): The inputs for the agent
            **kwargs: Additional arguments for the agent executor

        Returns:
            Dict[str, Any]: The agent's response
        """
        # Original LangChain AgentExecutor behavior uses callbacks to track internal state
        # Since we can't directly intercept the LLM calls, we'll have to patch the _take_next_step
        # method to count calls
        original_take_next_step = self.agent_executor._take_next_step

        # Define a patched method that increments our counter
        def patched_take_next_step(agent_executor, *args, **kwargs):
            """Patched method to intercept and count LLM calls."""
            # Increment call counter
            self.calls_counter += 1
            self.logger.info(f"LLM call {self.calls_counter}/{self.max_calls}")

            # If we've reached the limit, raise an exception to stop execution
            if self.calls_counter > self.max_calls:
                raise ValueError(f"Reached maximum LLM calls ({self.max_calls})")

            # Call the original method and return its result
            return original_take_next_step(*args, **kwargs)

        # Replace the method temporarily
        self.agent_executor._take_next_step = patched_take_next_step.__get__(
            self.agent_executor, type(self.agent_executor)
        )

        try:
            # Try to execute the agent with the patched method
            result = self.agent_executor.invoke(inputs, **kwargs)
            self.intermediate_results.append(result)
            return result
        except ValueError as e:
            # Check if this is our limit exception
            if "Reached maximum LLM calls" in str(e):
                self.logger.warning(f"Reached maximum LLM calls ({self.max_calls}). Returning best result so far.")
                return self._get_best_result()
            else:
                # Re-raise other ValueErrors
                raise
        except Exception as e:
            # Log any other exceptions
            self.logger.error(f"Error during agent execution: {e}")
            raise
        finally:
            # Always restore the original method
            self.agent_executor._take_next_step = original_take_next_step

    def _get_best_result(self) -> Dict[str, Any]:
        """
        Select the best result from intermediate results.

        Returns:
            Dict[str, Any]: The best result
        """
        # If we have no results, return a default message
        if not self.intermediate_results:
            return {
                "output": (
                    "I apologize, but I wasn't able to generate a complete response "
                    "within the allowed number of steps. Please try asking a more "
                    "specific question or breaking your query into smaller parts."
                )
            }

        # For simple cases, return the last result
        if len(self.intermediate_results) == 1:
            return self.intermediate_results[0]

        # Find the most comprehensive result
        # We'll use a heuristic based on length and completion
        best_result = None
        best_score = -1

        for result in self.intermediate_results:
            output = result.get("output", "")

            # Calculate a score based on various heuristics
            score = len(output)  # Base score is length

            # Bonus for answers that seem complete
            if any(phrase in output.lower() for phrase in [
                "in summary", "to summarize", "in conclusion", "to conclude"
            ]):
                score += 500

            # Bonus for answers with citations
            if "source:" in output.lower() or "citation:" in output.lower():
                score += 300

            # Bonus for answers with URLs
            if re.search(r'https?://\S+', output):
                score += 200

            if score > best_score:
                best_score = score
                best_result = result

        # Add a note about early termination
        if best_result and isinstance(best_result, dict) and "output" in best_result:
            best_result["output"] += (
                "\n\n(Note: This response was generated with a limited number of API "
                "calls for efficiency.)"
            )

        return best_result

    def _is_final_answer(self, result: Dict[str, Any]) -> bool:
        """
        Determine if the result contains a final answer.

        Args:
            result (Dict[str, Any]): The agent's result

        Returns:
            bool: True if this is a final answer, False otherwise
        """
        if not isinstance(result, dict) or "output" not in result:
            return False

        output = result["output"]

        # Check for common concluding phrases
        concluding_phrases = [
            "to summarize", "in summary", "in conclusion", "to conclude",
            "based on the information", "no further searches are needed"
        ]

        if any(phrase in output.lower() for phrase in concluding_phrases):
            return True

        # Check if the output explicitly mentions sources/citations
        if "source:" in output.lower() or "citation:" in output.lower():
            return True

        return False

    def __getattr__(self, name):
        """
        Pass through any attributes not found in this class to the wrapped executor.

        Args:
            name: Attribute name

        Returns:
            Any: The attribute from the wrapped agent executor
        """
        return getattr(self.agent_executor, name)


def create_limited_call_agent_executor(agent_executor: AgentExecutor, max_calls: int = 6) -> LimitedCallAgentExecutor:
    """
    Create a wrapper around an agent executor that limits LLM calls.

    Args:
        agent_executor (AgentExecutor): The original agent executor
        max_calls (int, optional): Maximum LLM calls per query. Defaults to 3.

    Returns:
        LimitedCallAgentExecutor: The wrapped agent executor with call limiting
    """
    return LimitedCallAgentExecutor(agent_executor, max_calls)