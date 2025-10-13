"""
Language model functionality for the Moffitt RAG system.

This module provides functions for creating and using language models
from various providers (OpenAI, Groq) for the agent's reasoning.
"""

import os
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Literal

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ..config.config import get_settings

# Get settings
settings = get_settings()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """
    Enum for LLM providers.
    """
    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"


def get_llm_model(
    provider: Optional[LLMProvider] = None,
    model_name: Optional[str] = None,
    stream: bool = False,
    temperature: float = 0.7,
    **kwargs
) -> BaseChatModel:
    """
    Get a language model from the specified provider.

    Args:
        provider (Optional[LLMProvider], optional):
            The provider to use. If None, will use the provider from settings.
            Defaults to None.
        model_name (Optional[str], optional):
            The name of the language model to use.
            If None, the model from settings will be used.
            Defaults to None.
        stream (bool, optional):
            Whether to stream the response. Defaults to False.
        temperature (float, optional):
            The temperature to use for generation. Defaults to 0.7.
        **kwargs: Additional parameters to pass to the LLM constructor.

    Returns:
        BaseChatModel: The language model
    """
    # Set defaults from settings
    if provider is None:
        provider_str = os.getenv("LLM_PROVIDER", "groq")
        provider = LLMProvider(provider_str.lower())

    if model_name is None:
        if provider == LLMProvider.OPENAI:
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        elif provider == LLMProvider.GROQ:
            model_name = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        else:  # OLLAMA
            model_name = settings.llm_model_name

    logger.info(f"Loading LLM model: {provider} / {model_name}")

    # Set up callback manager if streaming
    callback_manager = None
    if stream:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Create the language model based on provider
    if provider == LLMProvider.OPENAI:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=stream,
            openai_api_key=openai_api_key,
            **kwargs
        )

    elif provider == LLMProvider.GROQ:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            streaming=stream,
            groq_api_key=groq_api_key,
            **kwargs
        )

    else:  # OLLAMA
        # Note: Ollama doesn't return a BaseChatModel but we're keeping the interface similar
        llm = Ollama(
            model=model_name,
            temperature=temperature,
            callback_manager=callback_manager,
            base_url=settings.ollama_base_url,
            **kwargs
        )

    return llm


def generate_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Generate text using a language model.

    Args:
        prompt (str): The prompt to generate text from
        system_prompt (Optional[str], optional):
            The system prompt to use. Defaults to None.
        provider (Optional[LLMProvider], optional):
            The provider to use. If None, will use the provider from settings.
            Defaults to None.
        model_name (Optional[str], optional):
            The name of the language model to use.
            If None, the model from settings will be used.
            Defaults to None.
        temperature (float, optional):
            The temperature to use for generation. Defaults to 0.7.
        max_tokens (Optional[int], optional):
            The maximum number of tokens to generate. Defaults to None.

    Returns:
        str: The generated text
    """
    # Get the language model
    kwargs = {}
    if max_tokens:
        if provider == LLMProvider.OLLAMA:
            kwargs["num_predict"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

    llm = get_llm_model(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        **kwargs
    )

    # Generate text
    logger.info(f"Generating text with prompt: {prompt[:50]}...")

    # Handle different model types
    if isinstance(llm, (ChatOpenAI, ChatGroq)):
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = llm.invoke(messages).content
    else:  # Ollama
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = llm.invoke(full_prompt)

    return response


def generate_structured_output(
    prompt: str,
    output_schema: Dict[str, Any],
    system_prompt: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Generate structured output using a language model.

    This function is used to generate structured output according to a schema,
    for example, extracting specific fields from text.

    Args:
        prompt (str): The prompt to generate text from
        output_schema (Dict[str, Any]): The schema for the output
        system_prompt (Optional[str], optional):
            The system prompt to use. Defaults to None.
        provider (Optional[LLMProvider], optional):
            The provider to use. If None, will use the provider from settings.
            Defaults to None.
        model_name (Optional[str], optional):
            The name of the language model to use.
            If None, the model from settings will be used.
            Defaults to None.
        temperature (float, optional):
            The temperature to use for generation. Defaults to 0.2.

    Returns:
        Dict[str, Any]: The generated structured output
    """
    import json

    # Format the prompt to request structured output
    structured_prompt = f"""
    {prompt}

    Provide your response as a valid JSON object with the following schema:
    {json.dumps(output_schema, indent=2)}

    Your response should be ONLY the JSON object, with no additional text.
    """

    # If no system prompt provided, use a default one for structured output
    if system_prompt is None:
        system_prompt = """You are a helpful assistant that provides responses in valid JSON format according to the specified schema.
        Do not include any text outside of the JSON object."""

    # Generate text
    response = generate_text(
        prompt=structured_prompt,
        system_prompt=system_prompt,
        provider=provider,
        model_name=model_name,
        temperature=temperature
    )

    try:
        # Try to extract JSON from the response
        # First, try to parse the entire response
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                return result

            # Try one more approach - find the first { and last }
            first_brace = response.find('{')
            last_brace = response.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json_str = response[first_brace:last_brace+1]
                result = json.loads(json_str)
                return result

            raise ValueError("Could not extract JSON from response")

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse response as JSON: {e}")
        logger.debug(f"Response: {response}")
        return {"error": "Failed to parse response as JSON", "raw_response": response}