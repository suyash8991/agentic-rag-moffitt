# Moffitt Agentic RAG System

This is a production implementation of the Moffitt Agentic RAG (Retrieval-Augmented Generation) system, using FastAPI for the backend and React for the frontend.

## Architecture Note

This branch contains the FastAPI/React implementation of the Moffitt Researcher Agent. If you're looking for the previous Streamlit implementation, it's available in the `archive/streamlit-implementation` branch.

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

## Architectural Migration

This project has undergone an architectural transition from a Streamlit monolithic application to a FastAPI backend with a React frontend. This change offers several benefits:

1. **Better Separation of Concerns**: Distinct backend and frontend layers
2. **Improved Performance**: Optimized API with lightweight frontend
3. **Enhanced User Experience**: Modern React-based UI with responsive design
4. **Better Scalability**: Independent scaling of frontend and backend components
5. **Superior Developer Experience**: Specialized frontend and backend technologies

The original Streamlit implementation is preserved in the `archive/streamlit-implementation` branch for reference and historical purposes.