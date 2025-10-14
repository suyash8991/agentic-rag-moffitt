"""
Styling utilities for the Streamlit application.

This module provides functions for styling the Streamlit interface,
including CSS styles and page configuration.
"""

import streamlit as st

def set_page_config():
    """
    Set the page configuration for the Streamlit app.

    This configures the page title, icon, layout, and sidebar state.
    """
    st.set_page_config(
        page_title="Moffitt Researcher Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_css():
    """
    Get the CSS styles for the application.

    Returns:
        str: CSS styles as a string
    """
    return """
    <style>
        /* Main title styling */
        .main-title {
            font-size: 2.5rem;
            color: #00538b; /* Moffitt blue */
            margin-bottom: 1rem;
        }

        /* Subtitle styling */
        .subtitle {
            font-size: 1.2rem;
            color: #444;
            margin-bottom: 2rem;
        }

        /* User message styling */
        .user-message {
            background-color: #141715;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #00538b; /* Moffitt blue */
        }

        /* Assistant message styling */
        .assistant-message {
            background-color: #202229;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #7c3aed; /* Purple */
        }

        /* Sources section styling */
        .sources-section {
            background-color: #f8fafc;
            padding: 0.5rem;
            border-radius: 0.5rem;
            font-size: 0.8rem;
            border: 1px solid #e2e8f0;
        }

        /* Section dividers */
        .divider {
            margin-top: 1rem;
            margin-bottom: 1rem;
            border-top: 1px solid #e2e8f0;
        }
    </style>
    """

def apply_styles():
    """
    Apply CSS styles to the Streamlit app.

    This function injects the CSS styles into the app.
    """
    st.markdown(load_css(), unsafe_allow_html=True)

def format_title(title_text):
    """
    Format the main title with custom styling.

    Args:
        title_text (str): The title text

    Returns:
        str: Formatted HTML for the title
    """
    return f"<div class='main-title'>{title_text}</div>"

def format_subtitle(subtitle_text):
    """
    Format a subtitle with custom styling.

    Args:
        subtitle_text (str): The subtitle text

    Returns:
        str: Formatted HTML for the subtitle
    """
    return f"<div class='subtitle'>{subtitle_text}</div>"

def format_user_message(message_text):
    """
    Format a user message with custom styling.

    Args:
        message_text (str): The message text

    Returns:
        str: Formatted HTML for the user message
    """
    return f"<div class='user-message'>{message_text}</div>"

def format_assistant_message(message_text):
    """
    Format an assistant message with custom styling.

    Args:
        message_text (str): The message text

    Returns:
        str: Formatted HTML for the assistant message
    """
    return f"<div class='assistant-message'>{message_text}</div>"