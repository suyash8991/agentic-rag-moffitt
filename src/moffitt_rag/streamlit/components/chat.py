"""
Chat interface component for the Streamlit application.

This module provides the chat interface for interacting with
the researcher assistant.
"""

import streamlit as st
import sys
import os
from typing import Optional, Dict, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from moffitt_rag.streamlit.state.session import (
    add_user_message,
    add_assistant_message,
    get_conversation_history
)
from moffitt_rag.streamlit.utils.styling import (
    format_user_message,
    format_assistant_message
)

def invoke_agent(query: str) -> Optional[Dict[str, Any]]:
    """
    Helper function to invoke the researcher agent.

    Args:
        query (str): The user query to process

    Returns:
        Optional[Dict[str, Any]]: The agent response, or None if an error occurred
    """
    # Use direct print statements to ensure visibility regardless of logging settings
    print("\n====================== AGENT INVOCATION START ======================")
    print(f"Processing query: {query}")

    # We'll store errors here and set in session state at the end
    current_error = None

    try:
        # Import the agent module
        from moffitt_rag.agents.agent import create_researcher_agent
        import traceback
        import logging
        import inspect

        print("Successfully imported agent module")

        logger = logging.getLogger(__name__)
        logger.info(f"Starting agent invocation for query: {query[:50]}...")

        # Check if vector_db is available in session state
        if 'vector_db' in st.session_state and st.session_state.vector_db is not None:
            print(f"Vector database is available with {st.session_state.vector_db._collection.count()} chunks")

            try:
                # Use st.cache_resource to avoid recreating the agent on each rerun
                @st.cache_resource
                def get_agent():
                    try:
                        print("Creating researcher agent...")
                        print("Checking tools first...")

                        # Check each tool individually before creating agent
                        from moffitt_rag.tools import (
                            ResearcherSearchTool,
                            DepartmentFilterTool,
                            ProgramFilterTool,
                            InterestMatchTool,
                            CollaborationTool
                        )

                        print("Imported tools modules successfully")

                        # Test each tool creation
                        tool_errors = []
                        try:
                            researcher_tool = ResearcherSearchTool()
                            print(f"ResearcherSearchTool created successfully")
                            # Check if the tool has dependencies
                            sig = inspect.signature(researcher_tool._run)
                            print(f"ResearcherSearchTool._run signature: {sig}")
                        except Exception as e:
                            error = f"ResearcherSearchTool creation error: {type(e).__name__}: {str(e)}"
                            print(error)
                            tool_errors.append(error)

                        try:
                            department_tool = DepartmentFilterTool()
                            print(f"DepartmentFilterTool created successfully")
                        except Exception as e:
                            error = f"DepartmentFilterTool creation error: {type(e).__name__}: {str(e)}"
                            print(error)
                            tool_errors.append(error)

                        try:
                            program_tool = ProgramFilterTool()
                            print(f"ProgramFilterTool created successfully")
                        except Exception as e:
                            error = f"ProgramFilterTool creation error: {type(e).__name__}: {str(e)}"
                            print(error)
                            tool_errors.append(error)

                        try:
                            interest_tool = InterestMatchTool()
                            print(f"InterestMatchTool created successfully")
                        except Exception as e:
                            error = f"InterestMatchTool creation error: {type(e).__name__}: {str(e)}"
                            print(error)
                            tool_errors.append(error)

                        try:
                            collab_tool = CollaborationTool()
                            print(f"CollaborationTool created successfully")
                        except Exception as e:
                            error = f"CollaborationTool creation error: {type(e).__name__}: {str(e)}"
                            print(error)
                            tool_errors.append(error)

                        if tool_errors:
                            print(f"WARNING: {len(tool_errors)} tools had creation errors")
                            for error in tool_errors:
                                print(f"  - {error}")

                        print("Now creating the actual researcher agent...")
                        agent = create_researcher_agent()
                        print("Agent created successfully")

                        # Test if agent has the expected attributes
                        if hasattr(agent, 'invoke'):
                            print("Agent has invoke method")
                        else:
                            print("WARNING: Agent does not have invoke method")

                        return agent
                    except Exception as e:
                        error_msg = f"Agent creation error: {type(e).__name__}: {str(e)}"
                        print(error_msg)
                        print(f"Traceback: {traceback.format_exc()}")
                        st.error(error_msg)
                        # Set directly in session state instead of using nonlocal
                        st.session_state.last_error = error_msg
                        raise

                # Create or get the cached agent
                print("Getting cached agent or creating new one...")
                agent = get_agent()
                print("Agent successfully initialized")

                # Show a spinner while the agent is thinking
                with st.spinner("Thinking..."):
                    try:
                        # Invoke the agent with the user query
                        print(f"Invoking agent with query: {query[:50]}...")
                        # Format the query as a dictionary with "input" key as expected by LangChain
                        result = agent.invoke({"input": query})
                        print("Agent invocation successful")
                        print(f"Response: {str(result)[:100]}...")
                        return result
                    except Exception as e:
                        error_msg = f"Agent invocation error: {type(e).__name__}: {str(e)}"
                        print(error_msg)
                        print(f"Traceback: {traceback.format_exc()}")
                        st.error(error_msg)
                        current_error = error_msg
                        return None
            except Exception as e:
                error_msg = f"Agent initialization error: {type(e).__name__}: {str(e)}"
                print(error_msg)
                print(f"Traceback: {traceback.format_exc()}")
                st.error(error_msg)
                current_error = error_msg
                return None
        else:
            # Vector database is not available
            db_status = "Vector database is not available or not initialized"
            print(db_status)
            st.warning(db_status)
            current_error = db_status
            return None
    except Exception as e:
        # Fall back to a placeholder response if the agent can't be loaded
        error_msg = f"Module import error: {type(e).__name__}: {str(e)}"
        import traceback
        trace = traceback.format_exc()
        print(error_msg)
        print(f"Traceback: {trace}")
        st.error(error_msg)
        current_error = f"{error_msg}\n{trace}"
        return None
    finally:
        print("====================== AGENT INVOCATION END ======================\n")
        # Store the current error for display in the UI
        if current_error:
            st.session_state.last_error = current_error

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
    # Add debug expander that shows detailed error info when available
    if 'last_error' in st.session_state and st.session_state.last_error:
        with st.expander("Debug Information", expanded=False):
            st.error(f"Last error: {st.session_state.last_error}")

    # Use a form for proper input handling
    with st.form(key="message_form", clear_on_submit=True):
        user_query = st.text_input("Ask a question about Moffitt researchers:")
        submit_button = st.form_submit_button("Send")

        if submit_button and user_query:
            # Add the user message to the conversation history
            add_user_message(user_query)

            # Clear previous error if any
            if 'last_error' in st.session_state:
                st.session_state.last_error = None

            # Invoke the agent and get the response
            response = invoke_agent(user_query)

            if response:
                # Add the agent response to the conversation history
                add_assistant_message(response.get('output', str(response)))
            else:
                # Fall back to a more specific error message if available
                if 'last_error' in st.session_state and st.session_state.last_error:
                    # Extract a concise version of the error for the message
                    error_summary = st.session_state.last_error.split('\n')[0]
                    add_assistant_message(
                        f"I'm sorry, but I couldn't process your query due to the following error:\n\n"
                        f"**{error_summary}**\n\n"
                        f"See the debug information for more details."
                    )
                else:
                    # Generic fallback if no specific error is captured
                    add_assistant_message(
                        "I'm sorry, but I couldn't process your query at this time. "
                        "Please check the vector database status in the settings page "
                        "or try again with a different question."
                    )

            # Set a flag to trigger rerun after form submission
            st.session_state.form_submitted = True

    # If a form was just submitted, rerun the app to update the UI
    if st.session_state.get("form_submitted", False):
        st.session_state.form_submitted = False
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
                    # Add the user message to the conversation history
                    add_user_message(example)

                    # Clear previous error if any
                    if 'last_error' in st.session_state:
                        st.session_state.last_error = None

                    # Invoke the agent and get the response
                    response = invoke_agent(example)

                    if response:
                        # Add the agent response to the conversation history
                        add_assistant_message(response.get('output', str(response)))
                    else:
                        # Fall back to a more specific error message if available
                        if 'last_error' in st.session_state and st.session_state.last_error:
                            # Extract a concise version of the error for the message
                            error_summary = st.session_state.last_error.split('\n')[0]
                            add_assistant_message(
                                f"I'm sorry, but I couldn't process your query due to the following error:\n\n"
                                f"**{error_summary}**\n\n"
                                f"See the debug information for more details."
                            )
                        else:
                            # Generic fallback if no specific error is captured
                            add_assistant_message(
                                "I'm sorry, but I couldn't process your query at this time. "
                                "Please check the vector database status in the settings page "
                                "or try again with a different question."
                            )

                    # Rerun to update the UI
                    st.rerun()


def render_chat_interface():
    """
    Render the complete chat interface.

    This function renders the chat header, message history,
    example queries, and message input field.
    """
    st.header("Chat with Moffitt Researcher Assistant")

    # Check if the vector database is available
    if 'vector_db' in st.session_state and st.session_state.vector_db is not None:
        st.info("You can ask questions about Moffitt researchers, their expertise, and potential collaborations.")

    # Render the conversation history
    render_message_history()

    # Add a visual separator
    st.markdown("---")

    # Render example queries
    render_example_queries()

    # Render the message input field
    render_message_input()