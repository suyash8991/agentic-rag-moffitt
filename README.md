# Moffitt Agentic RAG System

This is a production implementation of the Moffitt Agentic RAG (Retrieval-Augmented Generation) system, using FastAPI for the backend and React for the frontend.

## Project Structure

- `backend/` - FastAPI backend
- `frontend/` - React frontend
- `data/` - Data directory for researcher profiles and vector database

## Setup

1. Create a `.env` file based on `.env.docker.example`:
   ```bash
   cp .env.docker.example .env
   ```

2. Update the `.env` file with your API keys:
   - Get a Groq API key from [groq.com](https://console.groq.com/keys)
   - Get an OpenAI API key from [openai.com](https://platform.openai.com/account/api-keys) (optional)

## Running with Docker

Start both the backend and frontend:
```bash
docker-compose up
```

The backend API will be available at http://localhost:8000
The frontend will be available at http://localhost:3000 (when implemented)

API documentation is available at http://localhost:8000/docs

## Running Locally

For local development, you can run the backend and frontend separately.

### Backend

See the [backend README](backend/README.md) for setup and running instructions.

### Frontend

Frontend implementation is in progress.

## Development Notes

- The backend is built with FastAPI and uses Pydantic for data validation
- The frontend will be built with React and TypeScript
- The RAG system uses ChromaDB for vector storage and LangChain for the agent framework
- LLM integration supports Groq and OpenAI providers