"""
Sidebar component for the Streamlit application.

This module provides the sidebar navigation and status display.
"""

import streamlit as st
from moffitt_rag.streamlit.state.session import (
    get_current_page,
    set_current_page
)

def render_navigation():
    """
    Render the navigation buttons in the sidebar.

    This function displays buttons for navigating between different pages.
    """
    st.header("Navigation")

    # Determine current page for highlighting the active button
    current_page = get_current_page()

    # Chat button
    if st.button(
        "💬 Chat",
        use_container_width=True,
        type="primary" if current_page == "chat" else "secondary"
    ):
        set_current_page("chat")
        st.rerun()

    # Explorer button
    if st.button(
        "🔍 Explore Researchers",
        use_container_width=True,
        type="primary" if current_page == "explore" else "secondary"
    ):
        set_current_page("explore")
        st.rerun()

    # Settings button
    if st.button(
        "⚙️ Settings",
        use_container_width=True,
        type="primary" if current_page == "settings" else "secondary"
    ):
        set_current_page("settings")
        st.rerun()


def render_about_section():
    """
    Render the about section in the sidebar.

    This function displays information about the application.
    """
    st.write("---")
    st.subheader("About")
    st.write(
        "This application provides an interface to explore "
        "researcher information at Moffitt Cancer Center."
    )
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
    Render the complete sidebar with navigation, logo, and about section.
    """
    with st.sidebar:
        render_logo()
        st.write("---")
        render_navigation()
        render_about_section()