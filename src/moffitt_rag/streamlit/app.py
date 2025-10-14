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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define app states
APP_STATES = ["chat", "explore", "settings"]

def render_header():
    """Render the application header"""
    st.title("Moffitt Cancer Center Researcher Assistant")
    st.write("Ask questions about researchers, their expertise, and potential collaborations")

def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.header("Navigation")

        if st.button("💬 Chat", use_container_width=True):
            set_current_page("chat")
            st.rerun()

        if st.button("🔍 Explore Researchers", use_container_width=True):
            set_current_page("explore")
            st.rerun()

        if st.button("⚙️ Settings", use_container_width=True):
            set_current_page("settings")
            st.rerun()

        # About section
        st.write("---")
        st.subheader("About")
        st.write(
            "This application provides an interface to explore "
            "researcher information at Moffitt Cancer Center."
        )
        st.caption("Moffitt Agentic RAG System v0.1.0")

def render_chat_interface():
    """Render the main chat interface (placeholder)"""
    st.header("Chat")
    st.info("Chat interface will be implemented in the next version")

    # Simple message input
    user_query = st.text_input("Ask a question (placeholder):")
    if user_query:
        add_user_message(user_query)
        add_assistant_message("This is a placeholder response. The chat functionality will be implemented in the next version.")
        st.rerun()

    # Display conversation history
    conversation_history = get_conversation_history()
    if conversation_history:
        st.subheader("Conversation History")
        for message in conversation_history:
            role = "You: " if message["role"] == "user" else "Assistant: "
            st.text(f"{role}{message['content']}")

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
    st.set_page_config(
        page_title="Moffitt Researcher Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

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