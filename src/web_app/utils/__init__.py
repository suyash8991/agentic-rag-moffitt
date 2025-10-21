"""
Utility functions for the Moffitt Agentic RAG web application.

This package provides utility functions and helpers for the
Streamlit-based web interface.
"""

from .styling import (
    set_page_config,
    apply_styles,
    format_title,
    format_subtitle,
    format_user_message,
    format_assistant_message
)

__all__ = [
    "set_page_config",
    "apply_styles",
    "format_title",
    "format_subtitle",
    "format_user_message",
    "format_assistant_message"
]