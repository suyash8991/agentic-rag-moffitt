"""
State management for the Moffitt Agentic RAG web application.

This package provides functions and utilities for managing application
state in the Streamlit-based web interface.
"""

from .session import (
    init_session_state,
    get_current_page,
    set_current_page,
    add_user_message,
    add_assistant_message,
    clear_conversation_history,
    get_conversation_history
)

__all__ = [
    "init_session_state",
    "get_current_page",
    "set_current_page",
    "add_user_message",
    "add_assistant_message",
    "clear_conversation_history",
    "get_conversation_history"
]