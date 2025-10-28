"""
Configuration settings for the application.

This module provides settings based on environment variables.
"""

import os
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    """Application settings."""

    # API settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Moffitt Agentic RAG"
    API_KEY: str = os.getenv("API_KEY", "dev_api_key")
    API_KEY_NAME: str = "X-API-Key"
    API_HOST: str = os.getenv("API_HOST", "localhost")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}"
    ]

    # Data settings
    VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", str(Path("../data/vector_db").resolve()))
    PROCESSED_DATA_DIR: str = os.getenv("PROCESSED_DATA_DIR", str(Path("../data/processed").resolve()))
    COLLECTION_NAME: str = "moffitt_researchers"

    # Embedding settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    # LLM settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "llama3")

    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Groq settings
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Euron settings
    EURON_API_KEY: Optional[str] = os.getenv("EURON_API_KEY")
    EURON_MODEL: str = os.getenv("EURON_MODEL", "gpt-4.1-nano")

    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True


# Initialize settings
settings = Settings()