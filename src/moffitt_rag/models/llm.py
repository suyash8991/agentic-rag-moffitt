"""
Language model functionality for the Moffitt RAG system.

This module provides functions for creating and using language models
from various providers (OpenAI, Groq) for the agent's reasoning.
"""

import os
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Literal

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Import our custom Euron integration
from .euron_chat import ChatEuron

from ..config.config import get_settings
from ..utils.logging import get_logger, log_llm_event

# Get settings
settings = get_settings()

# Get logger for this module
logger = get_logger(__name__)


class LLMProvider(str, Enum):
    """
    Enum for LLM providers.
    """
    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"
    EURON = "euron"


def get_llm_model(
    provider: Optional[LLMProvider] = None,
    model_name: Optional[str] = None,
    stream: bool = False,
    temperature: float = 0.7,
    fallback_on_rate_limit: bool = True,
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
        fallback_on_rate_limit (bool, optional):
            Whether to try alternative providers if rate limiting occurs.
            Defaults to True.
        **kwargs: Additional parameters to pass to the LLM constructor.

    Returns:
        BaseChatModel: The language model
    """
    logger.info(f"Getting LLM model: provider={provider}, model={model_name}, stream={stream}, temperature={temperature}")
    
    # Log structured event for LLM model request
    log_llm_event("llm_model_request", {
        "provider": str(provider) if provider else "default",
        "model_name": model_name if model_name else "default",
        "stream": stream,
        "temperature": temperature,
        "fallback_on_rate_limit": fallback_on_rate_limit
    })
    
    # Available providers to try in order if rate limiting occurs
    # This will be used if fallback_on_rate_limit is True
    provider_fallbacks = [
        (LLMProvider.OPENAI, os.getenv("OPENAI_MODEL", "gpt-4o")),
        (LLMProvider.GROQ, os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")),
        (LLMProvider.EURON, os.getenv("EURON_MODEL", "gpt-4.1-nano")),
        (LLMProvider.OLLAMA, settings.llm_model_name),
    ]

    # Determine which providers to try and in what order
    providers_to_try = []

    # Always try the requested provider first
    if provider is not None:
        # If a specific provider was requested, try it first
        providers_to_try.append((provider, model_name))

        if fallback_on_rate_limit:
            # Then add other providers as fallbacks
            for fallback_provider, fallback_model in provider_fallbacks:
                if fallback_provider != provider:
                    providers_to_try.append((fallback_provider, fallback_model))
    else:
        # No specific provider requested, try the default from environment
        provider_str = os.getenv("LLM_PROVIDER", "groq").lower()
        try:
            default_provider = LLMProvider(provider_str)
        except ValueError:
            logger.warning(f"Invalid LLM provider in environment: '{provider_str}', defaulting to Groq")
            default_provider = LLMProvider.GROQ

        # Start with the default provider
        if default_provider == LLMProvider.OPENAI:
            default_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        elif default_provider == LLMProvider.GROQ:
            default_model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        elif default_provider == LLMProvider.EURON:
            default_model = os.getenv("EURON_MODEL", "gpt-4.1-nano")
        else:  # OLLAMA
            default_model = settings.llm_model_name

        providers_to_try.append((default_provider, default_model))

        if fallback_on_rate_limit:
            # Add other providers as fallbacks
            for fallback_provider, fallback_model in provider_fallbacks:
                if fallback_provider != default_provider:
                    providers_to_try.append((fallback_provider, fallback_model))

    # Try providers in order until one succeeds
    last_exception = None
    for provider_to_try, model_to_try in providers_to_try:
        try:
            # If model_name was explicitly provided, use it instead of the fallback
            if model_name is not None and provider_to_try == provider:
                model_to_try = model_name

            logger.info(f"Attempting to initialize LLM: Provider={provider_to_try.value}, Model={model_to_try}")

            # Set up callback manager if streaming
            callback_manager = None
            if stream:
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                logger.debug("Configured streaming callback handler")

            # Create the language model based on provider
            if provider_to_try == LLMProvider.OPENAI:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    logger.warning("OPENAI_API_KEY environment variable not set. Skipping OpenAI fallback.")
                    continue

                logger.info(f"Initializing OpenAI ChatModel with model={model_to_try}")
                llm = ChatOpenAI(
                    model=model_to_try,
                    temperature=temperature,
                    streaming=stream,
                    openai_api_key=openai_api_key,
                    **kwargs
                )

            elif provider_to_try == LLMProvider.GROQ:
                groq_api_key = os.getenv("GROQ_API_KEY")
                # Check if the API key is a placeholder value or missing
                if not groq_api_key or groq_api_key == "your_groq_api_key_here":
                    logger.warning("GROQ_API_KEY environment variable not set or contains placeholder. Skipping Groq fallback.")
                    continue

                logger.info(f"Initializing Groq ChatModel with model={model_to_try}")
                # Note: Groq client doesn't support custom retry parameters directly
                # We'll use the default retry behavior
                llm = ChatGroq(
                    model=model_to_try,
                    temperature=temperature,
                    streaming=stream,
                    groq_api_key=groq_api_key,
                    max_retries=5,  # Maximum 5 retries for rate limits
                    **kwargs
                )

            elif provider_to_try == LLMProvider.EURON:
                euron_api_key = os.getenv("EURON_API_KEY")
                # Check if the API key is missing
                if not euron_api_key:
                    logger.warning("EURON_API_KEY environment variable not set. Skipping Euron fallback.")
                    continue

                logger.info(f"Initializing Euron ChatModel with model={model_to_try}")
                llm = ChatEuron(
                    model=model_to_try,
                    temperature=temperature,
                    api_key=euron_api_key,
                    max_retries=3,  # Maximum 3 retries for rate limits
                    timeout=120,    # 2-minute timeout
                    **kwargs
                )

            else:  # OLLAMA
                # Note: Ollama doesn't return a BaseChatModel but we're keeping the interface similar
                logger.info(f"Initializing Ollama LLM with model={model_to_try}, base_url={settings.ollama_base_url}")
                # Check if Ollama is accessible
                import requests
                try:
                    # Just check if the server is accessible
                    response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
                    if response.status_code != 200:
                        logger.warning(f"Ollama server returned status code {response.status_code}, it may not be functioning correctly. Skipping Ollama fallback.")
                        continue
                except requests.exceptions.ConnectionError:
                    logger.warning(f"Could not connect to Ollama server at {settings.ollama_base_url}. Skipping Ollama fallback.")
                    continue

                llm = Ollama(
                    model=model_to_try,
                    temperature=temperature,
                    callback_manager=callback_manager,
                    base_url=settings.ollama_base_url,
                    **kwargs
                )

            logger.info(f"Successfully initialized LLM: {provider_to_try.value}/{model_to_try}")
            
            # Log successful completion
            log_llm_event("llm_model_success", {
                "provider": provider_to_try.value,
                "model_name": model_to_try,
                "stream": stream,
                "temperature": temperature
            })
            
            return llm

        except Exception as e:
            import traceback
            logger.warning(f"Error initializing LLM with provider {provider_to_try.value}: {type(e).__name__}: {str(e)}")
            logger.debug(f"Traceback for {provider_to_try.value} error: {traceback.format_exc()}")
            
            # Log structured error event
            log_llm_event("llm_model_error", {
                "provider": provider_to_try.value,
                "model_name": model_to_try,
                "error": str(e),
                "error_type": type(e).__name__,
                "is_rate_limit": "429" in str(e) or "rate limit" in str(e).lower() or "too many requests" in str(e).lower()
            })
            
            last_exception = e

            # Check if this is a rate limit error and we should try the next provider
            if "429" in str(e) or "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                logger.warning(f"Rate limit error detected with provider {provider_to_try.value}. Trying next provider if available.")
                continue

    # If we get here, all providers failed
    import traceback
    logger.error("All LLM providers failed to initialize")
    
    # Log final failure event
    log_llm_event("llm_model_failure", {
        "error": str(last_exception) if last_exception else "unknown",
        "error_type": type(last_exception).__name__ if last_exception else "unknown",
        "providers_tried": len(providers_to_try)
    })
    
    if last_exception:
        logger.error(f"Last error: {type(last_exception).__name__}: {str(last_exception)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"All LLM providers failed to initialize. Last error: {str(last_exception)}") from last_exception
    else:
        raise ValueError("All LLM providers failed to initialize for unknown reasons")


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
    if isinstance(llm, (ChatOpenAI, ChatGroq, ChatEuron)):
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