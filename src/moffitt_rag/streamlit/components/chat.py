"""
Chat interface component for the Streamlit application.

This module provides the chat interface for interacting with
the researcher assistant.
"""

import streamlit as st
from moffitt_rag.streamlit.state.session import (
    add_user_message,
    add_assistant_message,
    get_conversation_history
)
from moffitt_rag.streamlit.utils.styling import (
    format_user_message,
    format_assistant_message
)

def render_message_history():
    """
    Render the conversation message history.

    This function displays all messages in the conversation history
    with appropriate styling for user and assistant messages.
    """
    conversation_history = get_conversation_history()

    if not conversation_history:
        return

    for message in conversation_history:
        if message["role"] == "user":
            st.markdown(format_user_message(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(format_assistant_message(message["content"]), unsafe_allow_html=True)


def render_message_input():
    """
    Render the message input field and handle submission.

    This function displays a text input field for the user to enter
    messages and processes the input when submitted.
    """
    user_query = st.text_input("Ask a question about Moffitt researchers:", key="user_query")

    if user_query:
        # Add the user message to the conversation history
        add_user_message(user_query)

        # Temporarily add a placeholder response
        # In a real implementation, this would call the agent
        add_assistant_message(
            "This is a placeholder response. The actual agent integration "
            "will be implemented in a future version."
        )

        # Clear the input field and refresh the UI
        st.session_state.user_query = ""
        st.rerun()


def render_example_queries():
    """
    Render example query buttons that users can click.

    This function displays a set of example queries that the user
    can click to quickly try out the system.
    """
    with st.expander("Example questions you can ask"):
        example_queries = [
            "Who studies cancer evolution at Moffitt?",
            "Tell me about researchers in the Immunology department",
            "Find researchers similar to Robert Gatenby",
            "What potential collaborations exist between Biostatistics and Cancer Epidemiology?",
            "Which researchers at Moffitt study immunotherapy resistance mechanisms?"
        ]

        # Create a button for each example query
        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            # Alternate between columns for a two-column layout
            with cols[i % 2]:
                if st.button(example, key=f"example_{hash(example)}"):
                    # Process the example query as if the user had typed it
                    add_user_message(example)
                    add_assistant_message(
                        "This is a placeholder response for the example query. "
                        "The actual agent integration will be implemented in a future version."
                    )
                    st.rerun()


def render_chat_interface():
    """
    Render the complete chat interface.

    This function renders the chat header, message history,
    example queries, and message input field.
    """
    st.header("Chat with Moffitt Researcher Assistant")

    # Show info message while the agent integration is in progress
    st.info("Chat functionality is under development. Currently showing placeholder responses.")

    # Render the conversation history
    render_message_history()

    # Add a visual separator
    st.markdown("---")

    # Render example queries
    render_example_queries()

    # Render the message input field
    render_message_input()