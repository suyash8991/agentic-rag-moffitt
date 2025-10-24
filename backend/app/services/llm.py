"""
LLM service for the Moffitt Agentic RAG system.

This module provides functions for working with language models,
including text generation and structured output generation.
"""

import os
import logging
from enum import Enum
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from ..core.config import settings

# Setup logging
logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """LLM provider enum."""
    OPENAI = "openai"
    GROQ = "groq"


def get_llm_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> BaseChatModel:
    """
    Get a language model from the specified provider.

    Args:
        provider: The provider to use (defaults to settings.LLM_PROVIDER)
        model_name: The model name to use (defaults to settings.LLM_MODEL_NAME)
        temperature: Temperature setting for generation

    Returns:
        BaseChatModel: Language model instance
    """
    # Use settings if not specified
    if provider is None:
        provider = settings.LLM_PROVIDER.lower()

    # Select provider-specific model if not explicitly specified
    if model_name is None:
        if provider == "openai":
            model_name = settings.OPENAI_MODEL
        elif provider == "groq":
            model_name = settings.GROQ_MODEL
        else:
            model_name = settings.LLM_MODEL_NAME

    logger.info(f"Getting LLM model: provider={provider}, model={model_name}, temperature={temperature}")

    # Try to create the model
    try:
        if provider == "openai":
            # OpenAI
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=openai_api_key,
            )

        elif provider == "groq":
            # Groq
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key or groq_api_key == "your_groq_api_key_here":
                raise ValueError("GROQ_API_KEY environment variable not set or contains placeholder")

            model = ChatGroq(
                model=model_name,
                temperature=temperature,
                groq_api_key=groq_api_key,
            )

        else:
            # Fallback to Groq if provider not supported
            logger.warning(f"Provider {provider} not supported, falling back to Groq")

            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key or groq_api_key == "your_groq_api_key_here":
                raise ValueError("GROQ_API_KEY environment variable not set or contains placeholder")

            model = ChatGroq(
                model=model_name or "llama2-70b-4096",
                temperature=temperature,
                groq_api_key=groq_api_key,
            )

        return model

    except Exception as e:
        logger.error(f"Error creating LLM model: {e}")
        raise


async def generate_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> str:
    """
    Generate text using a language model.

    Args:
        prompt: The prompt to generate text from
        system_prompt: Optional system prompt
        provider: Optional provider override
        model_name: Optional model name override
        temperature: Temperature for generation

    Returns:
        str: Generated text
    """
    try:
        # Get the LLM model
        llm = get_llm_model(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
        )

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Generate text
        response = await llm.ainvoke(messages)
        return response.content

    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise


async def generate_structured_output(
    prompt: str,
    output_schema: Dict[str, Any],
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Generate structured output using a language model.

    Args:
        prompt: The prompt to generate from
        output_schema: Schema for the output
        system_prompt: Optional system prompt
        provider: Optional provider override
        model_name: Optional model name override
        temperature: Temperature (lower for structured output)

    Returns:
        Dict[str, Any]: Structured output
    """
    import json
    import re

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
    response_text = await generate_text(
        prompt=structured_prompt,
        system_prompt=system_prompt,
        provider=provider,
        model_name=model_name,
        temperature=temperature,
    )

    try:
        # Try to parse the response as JSON
        # First, try to parse the entire response
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                return result

            # Try one more approach - find the first { and last }
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json_str = response_text[first_brace:last_brace+1]
                result = json.loads(json_str)
                return result

            raise ValueError("Could not extract JSON from response")

    except Exception as e:
        logger.error(f"Failed to parse response as JSON: {e}")
        return {
            "error": "Failed to parse response as JSON",
            "raw_response": response_text[:1000]  # Truncate long responses
        }