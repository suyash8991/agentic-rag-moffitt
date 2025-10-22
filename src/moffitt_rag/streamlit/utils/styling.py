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
        /* CSS Variables for dark theme */
        :root {
            --moffitt-blue: #1c82ca;         /* Brighter blue for dark theme */
            --moffitt-blue-light: #40a0e7;   /* Even brighter blue for hover states */
            --moffitt-blue-pale: #1e3a5a;    /* Darker blue background */
            --accent-color: #9470ff;         /* Brighter purple for dark theme */
            --accent-color-light: #b192ff;   /* Even brighter purple for hover states */
            --text-dark: #ffffff;            /* White for main text in dark theme */
            --text-medium: #ffffff;          /* Light gray for secondary text */
            --text-light: #cccccc;           /* Medium gray for tertiary text */
            --bg-light: #2d2d2d;             /* Darker background */
            --bg-medium: #222222;            /* Even darker background */
            --bg-dark: #1a1a1a;              /* Darkest background */
            --border-light: #444444;         /* Darker border color */
            --success-color: #1da977;        /* Slightly brighter green */
            --error-color: #e55757;          /* Brighter red */
            --warning-color: #ffb938;        /* Brighter orange/yellow */
            --info-color: #5491f5;           /* Brighter blue for info */
            --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.4);  /* Stronger shadow for dark theme */
            --shadow-md: 0 4px 6px 0 rgba(0, 0, 0, 0.5);  /* Even stronger shadow */
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
        }

        /* Global styling improvements - ChatGPT-like appearance - Dark theme */
        .stApp {
            font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            color: var(--text-dark);
            background-color: #13151a; /* Darker background for entire app */
        }

        /* Force dark text on dark background */
        .stMarkdown, .stText {
            color: #ffffff !important;
        }

        /* Make the app more streamlined and modern with tighter spacing */
        .main .block-container {
            padding-top: 0;  /* Remove default padding */
            padding-bottom: 0;  /* Remove default padding */
            max-width: 900px; /* More like ChatGPT width */
            margin: 0 auto;
        }

        /* Remove all margins from streamlit elements */
        .stMarkdown, .stText {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }

        /* Remove spaces in chat interface */
        div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }

        /* Button improvements */
        .stButton button {
            border-radius: var(--radius-sm);
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
        }

        .stButton button:hover {
            box-shadow: var(--shadow-md);
        }

        /* Improve header styling */
        .stHeadingContainer {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }

        /* Main title styling - more compact like Claude/ChatGPT */
        .main-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--moffitt-blue);
            margin-bottom: 0.5rem;
            letter-spacing: -0.01em;
        }

        /* Subtitle styling - more subtle */
        .subtitle {
            font-size: 0.9rem;
            font-weight: 400;
            color: var(--text-light);
            margin-bottom: 1rem;
        }

        /* Improved chat message styling - Dark theme */
        .user-message {
            background-color: #2a3555; /* Dark blue for user messages */
            padding: 1rem;
            border-radius: var(--radius-md);
            margin-bottom: 1rem;
            border-left: 4px solid var(--moffitt-blue);
            box-shadow: var(--shadow-sm);
            color: #ffffff; /* White text */
        }

        /* Assistant message styling - Dark theme */
        .assistant-message {
            background-color: #2a2b31; /* Dark gray for assistant messages */
            padding: 1rem;
            border-radius: var(--radius-md);
            margin-bottom: 1rem;
            border-left: 4px solid var(--accent-color);
            box-shadow: var(--shadow-sm);
            color: #ffffff; /* White text */
        }

        /* Sources section styling */
        .sources-section {
            background-color: var(--bg-medium);
            padding: 0.75rem;
            border-radius: var(--radius-sm);
            font-size: 0.9rem;
            border: 1px solid var(--border-light);
            margin-top: 0.5rem;
        }

        /* Section dividers */
        .divider {
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
            border-top: 1px solid var(--border-light);
        }

        /* Custom button styling */
        .moffitt-button {
            background-color: var(--moffitt-blue);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: var(--radius-sm);
            font-weight: 500;
            border: none;
            cursor: pointer;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
        }

        .moffitt-button:hover {
            background-color: var(--moffitt-blue-light);
            box-shadow: var(--shadow-md);
        }

        /* Input field styling */
        .stTextInput > div > div > input {
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-light);
            padding: 0.75rem;
        }

        .stTextInput > div > div > input:focus {
            border-color: var(--moffitt-blue);
            box-shadow: 0 0 0 1px var(--moffitt-blue-light);
        }

        /* Card styling for info displays */
        .info-card {
            background-color: var(--bg-light);
            padding: 1rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-light);
            box-shadow: var(--shadow-sm);
            margin-bottom: 1rem;
        }

        /* Navigation container styling */
        .navigation-container {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }

        /* Footer styling */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            margin: 0;
            padding: 1rem;
            border-top: 1px solid var(--border-light);
            font-size: 0.8rem;
            color: var(--text-light);
            text-align: center;
        }

        /* ChatGPT-style examples - No extra space */
        .chatgpt-examples {
            margin: 0.5rem auto;
            max-width: 650px;
        }

        .chatgpt-examples-header {
            text-align: center;
            margin-bottom: 1rem;
            font-size: 1.1rem;
            font-weight: 500;
            color: var(--text-medium);
        }

        /* Style the example buttons like ChatGPT - Dark theme */
        .chatgpt-examples .stButton button {
            background-color: #2c2d30 !important; /* Dark gray background */
            border: 1px solid var(--border-light);
            color: #ffffff !important; /* White text for dark theme */
            font-weight: 400;
            font-size: 0.9rem;
            padding: 0.75rem;
            height: auto;
            line-height: 1.5;
            margin-bottom: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
            text-align: left;
            white-space: normal;
        }

        .chatgpt-examples .stButton button:hover {
            border-color: var(--moffitt-blue-light);
            background-color: #3a3b40 !important; /* Slightly lighter on hover */
        }

        /* ChatGPT-style input container - Dark theme - No extra spacing */
        .chatgpt-input-container {
            background-color: #1e2023;  /* Darker background for input area */
            padding: 0.5rem 1rem;
            border-top: 1px solid #393b40;  /* Slightly lighter border */
            border-radius: 0;
            margin: 0;
            position: sticky;
            bottom: 0;
            z-index: 100;
        }

        /* Style the input like ChatGPT - Dark theme */
        .chatgpt-input-container .stTextInput > div {
            border: 1px solid #454952;  /* Medium gray border */
            border-radius: 0.75rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
            background-color: #2d2d33 !important;  /* Dark input background */
        }

        .chatgpt-input-container .stTextInput > div > div > input {
            border: none;
            padding: 0.75rem 1rem;
            font-size: 0.95rem;
            color: #ffffff !important;  /* Force white text */
            background-color: #2d2d33 !important;  /* Match the parent */
        }

        /* Researcher info styling - Dark theme */
        .researcher-card {
            background-color: #2a2a2a; /* Dark gray background */
            border: 1px solid #3d3d3d; /* Darker border */
            border-radius: var(--radius-md);
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
        }

        .researcher-card:hover {
            box-shadow: var(--shadow-md);
            border-color: var(--moffitt-blue-light);
            background-color: #323232; /* Slightly lighter on hover */
        }

        .researcher-name {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--moffitt-blue-light); /* Brighter blue for visibility */
            margin-bottom: 0.5rem;
        }

        .researcher-title {
            font-size: 0.95rem;
            color: #e6e6e6; /* Light gray for visibility */
            font-style: italic;
            margin-bottom: 0.75rem;
        }

        .researcher-program {
            display: inline-block;
            background-color: #1e3a5a; /* Dark blue background */
            color: #89c4ff; /* Light blue text */
            padding: 0.25rem 0.5rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .researcher-department {
            display: inline-block;
            background-color: #2d2d35; /* Dark background */
            color: #cccccc; /* Light text */
            padding: 0.25rem 0.5rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .researcher-interests {
            margin-top: 0.75rem;
            font-size: 0.9rem;
            color: #d9d9d9; /* Light gray for readability */
        }

        .researcher-link {
            display: inline-block;
            margin-top: 0.75rem;
            color: #5ba3e0; /* Brighter blue link */
            font-size: 0.9rem;
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: border-color 0.2s ease;
        }

        .researcher-link:hover {
            border-color: #5ba3e0;
            color: #7ab6ea; /* Even brighter on hover */
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