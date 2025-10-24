# Moffitt Agentic RAG Backend

This is the FastAPI backend for the Moffitt Agentic RAG system.

## Setup

1. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

2. Update the `.env` file with your API keys:
   - Get a Groq API key from [groq.com](https://console.groq.com/keys)
   - Get an OpenAI API key from [openai.com](https://platform.openai.com/account/api-keys) (optional)

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally

Start the server:
```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

API documentation is available at http://localhost:8000/docs

## Running with Docker

From the project root:
```bash
docker-compose up
```

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/researchers` - List researchers
- `GET /api/researchers/{id}` - Get researcher details
- `GET /api/departments` - List departments
- `GET /api/programs` - List programs
- `POST /api/query` - Process a query
- `GET /api/query/{id}` - Get query status
- `GET /api/query/{id}/stream` - Stream query response using SSE
- `WebSocket /api/ws/query` - Stream query response using WebSocket
- `POST /api/admin/rebuild` - Rebuild vector database
- `GET /api/admin/stats` - Get database statistics