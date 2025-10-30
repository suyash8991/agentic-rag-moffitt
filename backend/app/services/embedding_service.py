"""
Embedding service for managing embedding model lifecycle.

This module provides a service for managing the embedding model,
replacing the global _cached_embedding_function with proper dependency injection.
"""

import time
import logging
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings

from ..core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for managing embedding model lifecycle.

    This service handles loading, caching, and providing access to the
    embedding model. It replaces the global _cached_embedding_function.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the embedding model to use (defaults to settings)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._embedding_function: Optional[HuggingFaceEmbeddings] = None
        logger.debug(f"Initialized EmbeddingService with model: {self.model_name}")

    def get_embedding_function(self) -> HuggingFaceEmbeddings:
        """
        Get the cached embedding function, loading it if necessary.

        Returns:
            HuggingFaceEmbeddings: The embedding function
        """
        if self._embedding_function is None:
            self._load_embedding_function()

        logger.debug("Using cached embedding model (no reload needed)")
        return self._embedding_function

    def _load_embedding_function(self) -> None:
        """Load the embedding function and cache it."""
        start_time = time.time()
        logger.info(f"⏱️ [PERF] Loading embedding model (first time): {self.model_name}")

        self._embedding_function = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"trust_remote_code": True}
        )

        load_time = time.time() - start_time
        logger.info(f"⏱️ [PERF] Embedding model loaded in {load_time:.3f}s (cached for future use)")

        if load_time > 2.0:
            logger.warning(f"⚠️ [PERF] Slow embedding model load: {load_time:.3f}s > 2.0s threshold")

    def is_loaded(self) -> bool:
        """
        Check if the embedding function is loaded.

        Returns:
            bool: True if loaded, False otherwise
        """
        return self._embedding_function is not None

    def reload(self) -> None:
        """Force reload of the embedding function."""
        logger.info("Reloading embedding model...")
        self._embedding_function = None
        self._load_embedding_function()
