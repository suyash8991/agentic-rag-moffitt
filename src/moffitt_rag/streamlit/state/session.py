"""
Session state management for the Streamlit application.

This module provides functions for initializing and managing the
Streamlit session state variables.
"""

import streamlit as st
import logging

# Configure logging
logger = logging.getLogger(__name__)

def init_session_state():
    """
    Initialize all session state variables.

    This function ensures all required session state variables
    are initialized with default values if they don't exist.
    """
    # Only log initialization once
    initialized = False

    if 'current_state' not in st.session_state:
        st.session_state.current_state = "chat"
        initialized = True

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
        initialized = True

    if initialized:
        logger.info("Session state initialized")


def get_current_page():
    """
    Get the current page from session state.

    Returns:
        str: The current page name
    """
    return st.session_state.get("current_state", "chat")


def set_current_page(page_name):
    """
    Update the current page in the session state.

    Args:
        page_name (str): The name of the page to display
    """
    if page_name in ["chat", "explore", "settings"]:
        st.session_state.current_state = page_name
        logger.info(f"Current page set to: {page_name}")
    else:
        logger.warning(f"Invalid page name: {page_name}")


def add_user_message(content):
    """
    Add a user message to the conversation history.

    Args:
        content (str): The message content
    """
    if not 'conversation_history' in st.session_state:
        st.session_state.conversation_history = []

    st.session_state.conversation_history.append({
        "role": "user",
        "content": content
    })


def add_assistant_message(content):
    """
    Add an assistant message to the conversation history.

    Args:
        content (str): The message content
    """
    if not 'conversation_history' in st.session_state:
        st.session_state.conversation_history = []

    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": content
    })


def clear_conversation_history():
    """
    Clear the conversation history in the session state.
    """
    st.session_state.conversation_history = []
    logger.info("Conversation history cleared")


def get_conversation_history():
    """
    Get the conversation history from session state.

    Returns:
        list: The conversation history
    """
    if not 'conversation_history' in st.session_state:
        st.session_state.conversation_history = []

    return st.session_state.conversation_history