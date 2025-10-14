"""
Moffitt Cancer Center Researcher Assistant

A Streamlit application for the Moffitt Agentic RAG system, providing
access to researcher information using natural language queries.
"""

import os
import streamlit as st
import logging
from typing import List, Dict, Any, Optional

# Import session state management
from moffitt_rag.streamlit.state.session import (
    init_session_state,
    get_current_page,
    set_current_page,
    add_user_message,
    add_assistant_message,
    clear_conversation_history,
    get_conversation_history
)

# Import styling utilities
from moffitt_rag.streamlit.utils.styling import (
    set_page_config,
    apply_styles,
    format_title,
    format_subtitle,
    format_user_message,
    format_assistant_message
)

# Import components
from moffitt_rag.streamlit.components.sidebar import render_sidebar
from moffitt_rag.streamlit.components.chat import render_chat_interface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define app states
APP_STATES = ["chat", "explore", "settings"]

def render_header():
    """Render the application header"""
    st.markdown(format_title("Moffitt Cancer Center Researcher Assistant"), unsafe_allow_html=True)
    st.markdown(format_subtitle("Ask questions about researchers, their expertise, and potential collaborations"), unsafe_allow_html=True)

# Sidebar rendering is now handled by the imported component

# Chat interface is now handled by the imported component

def render_researcher_explorer():
    """Render the researcher exploration interface (placeholder)"""
    st.header("Explore Researchers")
    st.info("Researcher explorer will be implemented in the next version")

def render_settings():
    """Render the settings interface (placeholder)"""
    st.header("Settings")
    st.info("Settings interface will be implemented in the next version")

    if st.button("Clear Conversation History"):
        clear_conversation_history()
        st.success("Conversation history cleared")
        st.rerun()

def main():
    """Main application entry point"""
    # Set page configuration
    set_page_config()

    # Apply custom styling
    apply_styles()

    # Initialize session state
    init_session_state()

    # Render the sidebar
    render_sidebar()

    # Render the header
    render_header()

    # Render the appropriate interface based on current state
    current_page = get_current_page()
    if current_page == "chat":
        render_chat_interface()
    elif current_page == "explore":
        render_researcher_explorer()
    elif current_page == "settings":
        render_settings()

if __name__ == "__main__":
    main()