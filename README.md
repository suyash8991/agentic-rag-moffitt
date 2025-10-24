# Moffitt Researcher Agent (Streamlit Implementation)

**IMPORTANT NOTE: This is an archived version of the Moffitt Researcher Agent that uses Streamlit. The current active implementation uses FastAPI backend with React frontend. This branch is maintained for historical and reference purposes only.**

## Overview

This repository contains the Streamlit-based implementation of the Moffitt Cancer Center Researcher Agent. It provides a conversational AI assistant that can answer questions about Moffitt Cancer Center researchers, their departments, research interests, and publications.

## Architecture

This implementation is built with the following components:

- **Streamlit**: For the web interface and application structure
- **LangChain**: For creating the conversational agent and managing interactions
- **Vector Database (ChromaDB)**: For storing researcher embeddings and enabling semantic search
- **LLM Providers**: Compatible with various providers (Groq, OpenAI, etc.)

## Running the Application

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Set up your environment variables:
```
cp .env.example .env
# Edit .env with your API keys
```

3. Run the application:
```
python run_app.py
```

## Features

- Conversational interface for researcher information
- Vector-based semantic search
- Multiple LLM provider support
- Administrative tools for database management

## Moving to the New Implementation

The project has been migrated to a FastAPI/React architecture. To access the new implementation, check out the `main` branch of this repository.

For questions about this archive or the new implementation, please contact the project maintainers.