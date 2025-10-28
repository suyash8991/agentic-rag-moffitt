"""
Environment validation script for Moffitt Agentic RAG System.

This script validates that all required environment variables and dependencies
are properly configured before starting the application.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print_info("Checking Python version...")
    if sys.version_info < (3, 10):
        print_error(f"Python 3.10+ required. Current: {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print_success(f"Python {sys.version_info.major}.{sys.version_info.minor} ✓")
    return True


def check_env_file() -> bool:
    """Check if .env file exists."""
    print_info("Checking .env file...")
    env_file = Path(".env")
    if not env_file.exists():
        print_error(".env file not found")
        print_info("Please copy .env.example to .env and configure it")
        return False
    print_success(".env file exists ✓")
    return True


def check_required_env_vars() -> Tuple[bool, List[str]]:
    """Check if all required environment variables are set."""
    print_info("Checking environment variables...")

    from dotenv import load_dotenv
    load_dotenv()

    required_vars = [
        ("LLM_PROVIDER", "LLM provider (openai or groq)"),
    ]

    optional_vars = [
        ("GROQ_API_KEY", "Groq API key (required if using Groq)"),
        ("OPENAI_API_KEY", "OpenAI API key (required if using OpenAI)"),
    ]

    errors = []
    warnings = []

    # Check required vars
    for var, description in required_vars:
        value = os.getenv(var)
        if not value:
            errors.append(f"{var} not set ({description})")
            print_error(f"{var} not set")
        else:
            print_success(f"{var} = {value}")

    # Check LLM provider-specific keys
    llm_provider = os.getenv("LLM_PROVIDER", "").lower()

    if llm_provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key or groq_key == "your_groq_api_key_here":
            errors.append("GROQ_API_KEY not properly configured")
            print_error("GROQ_API_KEY not properly configured")
        else:
            print_success(f"GROQ_API_KEY is set (length: {len(groq_key)})")

    elif llm_provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your_openai_api_key_here":
            errors.append("OPENAI_API_KEY not properly configured")
            print_error("OPENAI_API_KEY not properly configured")
        else:
            print_success(f"OPENAI_API_KEY is set (length: {len(openai_key)})")

    else:
        warnings.append(f"Unknown LLM provider: {llm_provider}")
        print_warning(f"Unknown LLM provider: {llm_provider}")

    return len(errors) == 0, errors + warnings


def check_data_directories() -> bool:
    """Check if required data directories exist."""
    print_info("Checking data directories...")

    from dotenv import load_dotenv
    load_dotenv()

    # Get paths from environment or use defaults
    vector_db_dir = os.getenv("VECTOR_DB_DIR", "../data/vector_db")
    processed_data_dir = os.getenv("PROCESSED_DATA_DIR", "../data/processed")

    all_exist = True

    # Check if directories exist
    for dir_path, name in [
        (vector_db_dir, "Vector DB directory"),
        (processed_data_dir, "Processed data directory"),
    ]:
        path = Path(dir_path)
        if not path.exists():
            print_warning(f"{name} does not exist: {dir_path}")
            print_info(f"Creating directory: {dir_path}")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_success(f"Created {name} ✓")
            except Exception as e:
                print_error(f"Failed to create {name}: {e}")
                all_exist = False
        else:
            print_success(f"{name} exists ✓")

    return all_exist


def check_dependencies() -> bool:
    """Check if required Python packages are installed."""
    print_info("Checking Python dependencies...")

    required_packages = [
        "fastapi",
        "uvicorn",
        "langchain",
        "chromadb",
        "sentence_transformers",
        "pydantic",
        "python_dotenv",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print_success(f"{package} installed ✓")
        except ImportError:
            print_error(f"{package} not installed")
            missing_packages.append(package)

    if missing_packages:
        print_error(f"Missing packages: {', '.join(missing_packages)}")
        print_info("Run: pip install -r requirements.txt")
        return False

    return True


def main():
    """Main validation function."""
    print_header("Moffitt Agentic RAG - Environment Validation")

    checks = [
        ("Python Version", check_python_version),
        (".env File", check_env_file),
        ("Environment Variables", lambda: check_required_env_vars()[0]),
        ("Data Directories", check_data_directories),
        ("Python Dependencies", check_dependencies),
    ]

    results = []

    for check_name, check_func in checks:
        print(f"\n{Colors.BOLD}Checking: {check_name}{Colors.ENDC}")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_error(f"Error during {check_name} check: {e}")
            results.append((check_name, False))

    # Print summary
    print_header("Validation Summary")

    all_passed = all(result for _, result in results)

    for check_name, result in results:
        if result:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")

    print("\n" + "="*60 + "\n")

    if all_passed:
        print_success("All checks passed! The application is ready to run.")
        print_info("Start the application with: uvicorn main:app --reload")
        return 0
    else:
        print_error("Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
