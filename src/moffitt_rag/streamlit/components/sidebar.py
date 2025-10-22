"""
Sidebar component for the Streamlit application.

This module provides the sidebar navigation and status display.
"""

import streamlit as st
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from moffitt_rag.streamlit.state.session import (
    get_current_page,
    set_current_page
)

def render_navigation():
    """
    Render the navigation buttons in the sidebar.

    This function displays buttons for navigating between different pages.
    """
    # Determine current page for highlighting the active button
    current_page = get_current_page()

    # Chat button
    if st.button(
        "💬 Chat",
        use_container_width=True,
        type="primary" if current_page == "chat" else "secondary",
        key="nav_chat"
    ):
        set_current_page("chat")
        st.rerun()

    # Settings button
    if st.button(
        "⚙️ Settings",
        use_container_width=True,
        type="primary" if current_page == "settings" else "secondary",
        key="nav_settings"
    ):
        set_current_page("settings")
        st.rerun()


def render_about_section():
    """
    Render the about section in the sidebar.

    This function displays information about the application.
    """
    st.write("---")
    st.caption("Moffitt Agentic RAG System v0.1.0")


def render_logo():
    """
    Render the Moffitt Cancer Center logo in the sidebar.
    """
    # Placeholder for actual logo - could be a local file in production
    logo_url = "https://moffitt.org/media/14349/moffitt-logo-white.png"
    st.image(logo_url, width=200)


def render_sidebar():
    """
    Render the optimized sidebar with navigation and about section.
    No logo to avoid broken images.
    """
    with st.sidebar:
        # Skip logo to avoid broken image
        # Add title directly
        st.title("Moffitt Assistant")

        # Render navigation immediately
        render_navigation()

        # Add about section
        render_about_section()