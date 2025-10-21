"""
Moffitt Cancer Center Researcher Assistant

A Streamlit application for the Moffitt Agentic RAG system, providing
access to researcher information using natural language queries.
"""

import os
import sys
import streamlit as st
from typing import List, Dict, Any, Optional
from pathlib import Path
import datetime

# Import centralized logging
from ..utils.logging import get_logger, log_ui_event, init_logging, configure_exception_logging

# Get logger for this module
logger = get_logger(__name__)
# Handle imports in a way that works both when imported as a module and when run directly
try:
    # First try relative imports (when imported as a module)
    from .state.session import (
        init_session_state,
        get_current_page,
        set_current_page,
        add_user_message,
        add_assistant_message,
        clear_conversation_history,
        get_conversation_history
    )
    from .utils.styling import (
        set_page_config,
        apply_styles,
        format_title,
        format_subtitle,
        format_user_message,
        format_assistant_message
    )
    from .components.sidebar import render_sidebar
    from .components.chat import render_chat_interface
except ImportError:
    # If that fails, try absolute imports (when run directly)
    # Add project root to path if needed
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent  # src directory
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Now try absolute imports
    from moffitt_rag.streamlit.state.session import (
        init_session_state,
        get_current_page,
        set_current_page,
        add_user_message,
        add_assistant_message,
        clear_conversation_history,
        get_conversation_history
    )
    from moffitt_rag.streamlit.utils.styling import (
        set_page_config,
        apply_styles,
        format_title,
        format_subtitle,
        format_user_message,
        format_assistant_message
    )
    from moffitt_rag.streamlit.components.sidebar import render_sidebar
    from moffitt_rag.streamlit.components.chat import render_chat_interface

# Import our custom logging configuration
from moffitt_rag.utils.logging import (
    init_logging,
    configure_exception_logging,
    get_logger,
    log_ui_event
)

# Initialize structured logging system
init_logging()

# Configure exception logging
configure_exception_logging()

# Get logger for this module
logger = get_logger(__name__)

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
    """Render the settings interface with vector database management options and diagnostics"""
    st.header("Settings")

    # Vector Database Management Section
    st.subheader("Vector Database")

    # Display current vector database status
    if 'vector_db' in st.session_state and st.session_state.vector_db is not None:
        try:
            # Try to get the count of documents in the vector database
            count = st.session_state.vector_db._collection.count()
            st.success(f"Vector database is loaded with {count} chunks")
        except Exception as e:
            st.warning("Vector database is initialized but may be empty or inaccessible")
    else:
        st.error("Vector database is not loaded")

    # Add button to reload the vector database
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Reload Vector Database"):
            try:
                # Import vector database functions
                from moffitt_rag.db.vector_store import get_or_create_vector_db

                with st.spinner("Reloading vector database..."):
                    # Reload the vector database
                    vector_db = get_or_create_vector_db()
                    st.session_state.vector_db = vector_db
                    st.success("Vector database reloaded successfully")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to reload vector database: {e}")

    with col2:
        if st.button("Rebuild Vector Database"):
            try:
                # Import vector database functions
                from moffitt_rag.db.vector_store import create_vector_db
                from moffitt_rag.data.loader import load_all_chunks

                with st.spinner("Loading data and rebuilding vector database. This may take a while..."):
                    # Load all chunks
                    chunks = load_all_chunks()
                    # Create a new vector database with the chunks
                    vector_db = create_vector_db(chunks)
                    # Update the session state
                    st.session_state.vector_db = vector_db
                    st.success(f"Vector database rebuilt successfully with {len(chunks)} chunks")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to rebuild vector database: {e}")

    # Diagnostics Section
    st.subheader("Diagnostics")

    with st.expander("System Diagnostics", expanded=False):
        # Environment information
        st.write("### Environment")
        import platform
        st.write(f"Python version: {platform.python_version()}")
        st.write(f"Operating system: {platform.system()} {platform.version()}")

        # Display .env configuration (without showing actual API keys)
        st.write("### Environment Variables")
        import os
        env_vars = {
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "Not set"),
            "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "Not set"),
            "OPENAI_API_KEY": "Present" if os.getenv("OPENAI_API_KEY") else "Not set",
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "Not set"),
            "GROQ_API_KEY": "Present" if os.getenv("GROQ_API_KEY") else "Not set",
            "GROQ_MODEL": os.getenv("GROQ_MODEL", "Not set"),
            "LLM_MODEL_NAME": os.getenv("LLM_MODEL_NAME", "Not set"),
            "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "Not set"),
        }

        for key, value in env_vars.items():
            if key.endswith("API_KEY"):
                st.write(f"{key}: {value}")
            else:
                st.write(f"{key}: {value}")

        # Check if the API key for the selected provider is present
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        if provider == "groq" and not os.getenv("GROQ_API_KEY"):
            st.error("❌ GROQ_API_KEY is missing. Please add it to your .env file.")
        elif provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OPENAI_API_KEY is missing. Please add it to your .env file.")
        else:
            st.success("✅ API key for selected provider is present.")

        # Check for placeholder API keys
        if provider == "groq" and os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
            st.error("❌ GROQ_API_KEY contains the placeholder value. Please update it with your actual API key.")

    # Test individual tools
    if st.button("Test Agent Tools"):
        with st.spinner("Testing tools..."):
            results_container = st.container()

            with results_container:
                st.write("### Tool Test Results")

                # Test vector DB directly
                st.write("Testing vector database...")
                try:
                    from moffitt_rag.db.vector_store import similarity_search
                    test_results = similarity_search("cancer", k=2)
                    st.success(f"✅ Vector DB search successful with {len(test_results)} results")
                except Exception as e:
                    import traceback
                    st.error(f"❌ Vector DB search failed: {type(e).__name__}: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

                # Test each tool individually
                st.write("### Testing individual tools:")
                try:
                    from moffitt_rag.tools import (
                        ResearcherSearchTool,
                        DepartmentFilterTool,
                        ProgramFilterTool,
                        InterestMatchTool,
                        CollaborationTool
                    )

                    tools = [
                        ("ResearcherSearchTool", ResearcherSearchTool, "cancer"),
                        ("DepartmentFilterTool", DepartmentFilterTool, "Oncology"),
                        ("ProgramFilterTool", ProgramFilterTool, "Cancer Biology"),
                        ("InterestMatchTool", InterestMatchTool, "immunotherapy"),
                        ("CollaborationTool", CollaborationTool, "immunology bioinformatics")
                    ]

                    for name, tool_class, test_query in tools:
                        st.write(f"Testing {name}...")
                        try:
                            tool = tool_class()
                            result = tool._run(test_query)
                            st.success(f"✅ {name}: Success")
                            with st.expander(f"{name} Sample Result"):
                                st.text(result[:500] + ("..." if len(result) > 500 else ""))
                        except Exception as e:
                            import traceback
                            st.error(f"❌ {name}: Failed with {type(e).__name__}: {str(e)}")
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())

                except Exception as e:
                    import traceback
                    st.error(f"Failed to import tools: {type(e).__name__}: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

    # Test LLM connection
    if st.button("Test LLM Connection"):
        with st.spinner("Testing LLM connection..."):
            try:
                from moffitt_rag.models.llm import get_llm_model, generate_text

                # Try to generate a simple response
                result = generate_text(
                    "Say hello and identify yourself in one sentence.",
                    system_prompt="You are a helpful assistant."
                )

                st.success("✅ LLM connection successful!")
                st.write("### LLM Response:")
                st.write(result)
            except Exception as e:
                import traceback
                st.error(f"❌ LLM connection failed: {type(e).__name__}: {str(e)}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

    # Conversation History Management
    st.subheader("Conversation History")

    if st.button("Clear Conversation History"):
        clear_conversation_history()
        st.success("Conversation history cleared")
        st.rerun()

    # Logging Settings
    st.subheader("Logging Settings")

    with st.expander("Configure Logging", expanded=False):
        # Create columns for log levels
        col1, col2 = st.columns(2)

        with col1:
            # Console log level
            console_level = st.selectbox(
                "Console Log Level",
                options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=1,  # Default to INFO
                key="console_log_level"
            )

            # Enable query logging
            enable_query_log = st.checkbox(
                "Enable Query Logging",
                value=True,
                key="enable_query_log"
            )

            # Enable structured logging
            enable_structured_logs = st.checkbox(
                "Enable Structured Logging",
                value=True,
                key="enable_structured_logs"
            )

        with col2:
            # File log level
            file_level = st.selectbox(
                "File Log Level",
                options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=0,  # Default to DEBUG
                key="file_log_level"
            )

            # Enable error logging
            enable_error_log = st.checkbox(
                "Enable Error Logging",
                value=True,
                key="enable_error_log"
            )

        # Apply button for logging settings
        if st.button("Apply Logging Settings"):
            # Import logging utils
            from moffitt_rag.utils.logging import init_logging
            import logging

            # Convert string levels to logging constants
            log_levels = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL
            }

            # Re-initialize logging with new settings
            init_logging(
                console_level=log_levels[console_level],
                file_level=log_levels[file_level],
                enable_query_log=enable_query_log,
                enable_error_log=enable_error_log,
                enable_structured_logs=enable_structured_logs
            )

            # Log the change
            log_ui_event("logging_settings_changed", {
                "console_level": console_level,
                "file_level": file_level,
                "enable_query_log": enable_query_log,
                "enable_error_log": enable_error_log,
                "enable_structured_logs": enable_structured_logs
            })

            st.success(f"Logging settings updated: Console={console_level}, File={file_level}")

    # Log file viewer
    with st.expander("View Recent Logs", expanded=False):
        log_type = st.selectbox(
            "Log Type",
            options=["Application Log", "Error Log", "Query Log", "Agent Log", "Vector DB Log", "LLM Log"],
            index=0
        )

        # Map selection to file
        log_file_map = {
            "Application Log": "app.log",
            "Error Log": "errors.log",
            "Query Log": "queries.log",
            "Agent Log": "structured/agent.json",
            "Vector DB Log": "structured/vector_db.json",
            "LLM Log": "structured/llm.json"
        }

        if st.button("Load Recent Logs"):
            selected_file = log_file_map.get(log_type)
            if selected_file:
                try:
                    # Find logs directory
                    from pathlib import Path
                    from moffitt_rag.utils.logging import DEFAULT_LOG_DIR

                    log_path = DEFAULT_LOG_DIR / selected_file

                    if log_path.exists():
                        # Read the last 100 lines of the log file
                        with open(log_path, "r") as file:
                            lines = file.readlines()
                            # Show the most recent logs (last 100 lines)
                            recent_logs = "".join(lines[-100:])
                            st.text_area("Recent Logs", recent_logs, height=300)
                    else:
                        st.warning(f"Log file not found: {selected_file}")
                except Exception as e:
                    st.error(f"Error loading log file: {e}")

def main():
    """Main application entry point"""
    # Initialize logging system if not already initialized
    try:
        init_logging(
            console_level=None,  # Use environment-based level
            file_level=None,     # Use DEBUG for files
            enable_query_log=True,
            enable_error_log=True,
            enable_structured_logs=True
        )
        configure_exception_logging()
        logger.info("Logging system initialized successfully")
    except Exception as e:
        # If logging initialization fails, continue without it
        print(f"Warning: Could not initialize logging system: {e}")
    
    # Log application start
    logger.info("Starting Moffitt Agentic RAG application")
    log_ui_event("application_start", {
        "timestamp": datetime.datetime.now().isoformat(),
        "environment": os.environ.get("MOFFITT_ENV", "dev")
    })

    # Set page configuration
    set_page_config()

    # Apply custom styling
    apply_styles()

    # Initialize session state
    init_session_state()

    # Initialize vector database
    try:
        # Import the vector database module - use absolute imports
        # to avoid any issues when running the app in different modes
        from moffitt_rag.db.vector_store import get_or_create_vector_db

        # Use st.cache_resource to avoid reloading on each rerun
        @st.cache_resource
        def load_vector_db():
            st.info("Initializing vector database. This may take a moment on first run...")
            return get_or_create_vector_db()

        # Initialize vector DB
        vector_db = load_vector_db()

        # Store in session state for use in components
        st.session_state.vector_db = vector_db

    except Exception as e:
        st.warning(f"Could not initialize vector database. Some functionality may be limited. Error: {e}")
        st.session_state.vector_db = None

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