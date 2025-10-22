"""
Utility functions for the Streamlit application.

This package contains utility functions for styling, data loading, researcher formatting,
and agent integration.
"""

from .styling import (
    set_page_config,
    apply_styles,
    format_title,
    format_subtitle,
    format_user_message,
    format_assistant_message
)

from .researcher_formatting import (
    extract_researcher_info,
    format_researcher_card,
    display_researcher_results
)