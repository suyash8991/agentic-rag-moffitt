"""
Limited call agent executor.

This module provides a wrapper for LangChain agent executors that limits
the number of LLM calls and extracts the final answer.
"""

import re
import logging
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor

# Setup logging
logger = logging.getLogger(__name__)


class LimitedCallAgentExecutor:
    """
    Wrapper for LangChain agent executor that limits the number of LLM calls.
    """

    def __init__(
        self,
        agent_executor: AgentExecutor,
        max_calls: int = 6
    ):
        """
        Initialize the limited call agent executor.

        Args:
            agent_executor: The LangChain agent executor to wrap
            max_calls: Maximum number of LLM calls allowed
        """
        self.agent_executor = agent_executor
        self.max_calls = max_calls
        self.calls_counter = 0
        self.intermediate_results = []

    def invoke(self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke the agent with a limited number of LLM calls.

        Args:
            input_data: The input data for the agent
            config: Optional configuration dict (for LangSmith metadata, tags, etc.)

        Returns:
            Dict[str, Any]: The agent's response
        """
        try:
            # Reset the calls counter
            self.calls_counter = 0
            self.intermediate_results = []

            # Create a wrapper for _take_next_step
            original_take_next_step = self.agent_executor._take_next_step

            # Define the wrapped function
            def limited_take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=None,
            ):
                # Increment the calls counter
                self.calls_counter += 1
                logger.info(f"LLM call {self.calls_counter}/{self.max_calls}")

                # Check if we've reached the maximum number of calls
                if self.calls_counter > self.max_calls:
                    logger.warning(f"Maximum number of LLM calls reached ({self.max_calls})")
                    return {"output": self._get_best_result().get("output", "")}

                # Take the next step
                result = original_take_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager,
                )

                # Store the intermediate result
                if isinstance(result, dict) and "output" in result:
                    self.intermediate_results.append(result)

                    # Check if this result looks like a final answer
                    if self._is_final_answer(result):
                        logger.info("Final answer reached, stopping")
                        return result

                return result

            # Replace the _take_next_step method
            self.agent_executor._take_next_step = limited_take_next_step

            # Invoke the agent with config if provided
            if config:
                response = self.agent_executor.invoke(input_data, config=config)
            else:
                response = self.agent_executor.invoke(input_data)

            # Restore the original _take_next_step method
            self.agent_executor._take_next_step = original_take_next_step

            # Extract the final answer
            if "output" in response:
                output = response["output"]
                if "Final Answer:" in output:
                    final_answer = output.split("Final Answer:", 1)[1].strip()
                    response["output"] = final_answer

            return response

        except Exception as e:
            logger.error(f"Error in LimitedCallAgentExecutor: {e}")
            # Restore the original _take_next_step method
            self.agent_executor._take_next_step = original_take_next_step
            return {"output": f"Error: {str(e)}"}

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

        # Check for "Final Answer:" directly
        if "Final Answer:" in output:
            return True

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


def create_limited_call_agent_executor(
    agent_executor: AgentExecutor,
    max_calls: int = 6
) -> LimitedCallAgentExecutor:
    """
    Create a limited call agent executor.

    Args:
        agent_executor: The LangChain agent executor to wrap
        max_calls: Maximum number of LLM calls allowed

    Returns:
        LimitedCallAgentExecutor: The limited call agent executor
    """
    return LimitedCallAgentExecutor(
        agent_executor=agent_executor,
        max_calls=max_calls
    )