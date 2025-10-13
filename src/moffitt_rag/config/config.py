"""
Configuration settings for the Moffitt Agentic RAG system.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MARKDOWN_DATA_DIR = os.path.join(DATA_DIR, "markdown")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_html")
SCHEMAS_DIR = os.path.join(ROOT_DIR, "schemas")

# Database settings
VECTOR_DB_DIR = os.path.join(ROOT_DIR, "vector_db")
DEFAULT_COLLECTION_NAME = "moffitt_researchers"

# Model settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# API settings
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Chunking settings
MAX_CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128


class Settings(BaseModel):
    """Settings for the Moffitt Agentic RAG System."""

    # Data paths
    data_dir: Path = Field(default=DATA_DIR)
    processed_data_dir: Path = Field(default=PROCESSED_DATA_DIR)
    markdown_data_dir: Path = Field(default=MARKDOWN_DATA_DIR)
    raw_data_dir: Path = Field(default=RAW_DATA_DIR)
    schemas_dir: Path = Field(default=SCHEMAS_DIR)

    # Database settings
    vector_db_dir: Path = Field(default=VECTOR_DB_DIR)
    collection_name: str = Field(default=DEFAULT_COLLECTION_NAME)

    # Model settings
    embedding_model_name: str = Field(default=EMBEDDING_MODEL_NAME)
    llm_model_name: str = Field(default=LLM_MODEL_NAME)
    ollama_base_url: str = Field(default=OLLAMA_BASE_URL)

    # API settings
    api_host: str = Field(default=API_HOST)
    api_port: int = Field(default=API_PORT)

    # Chunking settings
    max_chunk_size: int = Field(default=MAX_CHUNK_SIZE)
    chunk_overlap: int = Field(default=CHUNK_OVERLAP)


# Create a global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Return the settings instance.

    Returns:
        Settings: The settings instance
    """
    return settings