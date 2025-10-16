"""
Euron.ai API integration for the Moffitt RAG system.

This module provides a custom ChatEuron class that integrates with the Euron.ai API
for language model capabilities.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, Iterator, Mapping, ClassVar

import httpx
from pydantic import BaseModel, Field, model_validator

from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage
)
from langchain_core.outputs import ChatGeneration, ChatResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Euron API URL
EURON_API_URL = "https://api.euron.one/api/v1/euri/chat/completions"

class ChatEuron(BaseChatModel):
    """Euron chat model integration."""

    api_key: Optional[str] = None
    """Euron API key."""

    model: str = "gpt-4.1-nano"
    """Model name to use."""

    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    temperature: float = 0.7
    """Sampling temperature."""

    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Additional kwargs for the model."""

    streaming: bool = False
    """Whether to stream the results."""

    max_retries: int = 3
    """Maximum number of retries."""

    api_base_url: str = EURON_API_URL
    """Base URL for Euron API."""

    timeout: Optional[int] = 120
    """Timeout for API requests in seconds."""

    client: Optional[httpx.Client] = None
    """HTTPX client."""

    @model_validator(mode="after")
    def validate_environment(self) -> "ChatEuron":
        """Validate environment."""
        if self.api_key is None:
            self.api_key = os.getenv("EURON_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "Euron API key not found. Please set the EURON_API_KEY environment "
                    "variable or pass api_key to the constructor."
                )
        return self

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "euron-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            **self.model_kwargs,
        }

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        """Format messages for the API."""
        message_dicts = []
        for message in messages:
            if isinstance(message, ChatMessage):
                message_dicts.append({
                    "role": message.role,
                    "content": message.content
                })
            elif isinstance(message, HumanMessage):
                message_dicts.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                message_dicts.append({
                    "role": "assistant",
                    "content": message.content
                })
            elif isinstance(message, SystemMessage):
                message_dicts.append({
                    "role": "system",
                    "content": message.content
                })
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        return message_dicts

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the model."""
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop"] = stop

        message_dicts = self._create_message_dicts(messages)

        try:
            # Initialize the client if needed
            if self.client is None:
                self.client = httpx.Client(timeout=self.timeout)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "messages": message_dicts,
                "model": params["model"],
                "temperature": params.get("temperature"),
                "max_tokens": params.get("max_tokens"),
                "top_p": params.get("top_p")
            }

            # Filter out None values
            data = {k: v for k, v in data.items() if v is not None}

            logger.info(f"Making request to Euron API with model {params['model']}")
            response = self.client.post(
                self.api_base_url,
                headers=headers,
                json=data,
            )

            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            response_data = response.json()

            if "choices" not in response_data:
                raise ValueError(f"Unexpected response format from Euron API: {response_data}")

            # Extract the message content from the response
            content = response_data["choices"][0]["message"]["content"]

            # Create a ChatGeneration object
            generation = ChatGeneration(
                message=AIMessage(content=content),
                generation_info={"finish_reason": response_data["choices"][0].get("finish_reason", None)}
            )

            # Return the result
            return ChatResult(generations=[generation])

        except Exception as e:
            logger.error(f"Error making request to Euron API: {e}")
            raise

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate a response from the model."""
        # Since this is a custom integration, we'll implement a simple async version
        # that actually just calls the sync version. This can be optimized later.
        import asyncio
        return await asyncio.to_thread(
            self._generate, messages, stop, run_manager, **kwargs
        )

    def _stream(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Iterator[ChatGeneration]:
        """
        Simulate streaming for compatibility with LangChain agents.

        Euron API doesn't natively support streaming, so we generate the full response
        and yield it as a single chunk.
        """
        logger.info("Streaming requested for Euron API, using non-streaming fallback")

        # Get the full response using the regular generate method
        chat_result = self._generate(messages, stop=stop, **kwargs)

        # If there's a generation, yield it as a single chunk
        if chat_result.generations:
            yield chat_result.generations[0]