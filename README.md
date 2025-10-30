# Moffitt Agentic RAG System

This is a production implementation of the Moffitt Agentic RAG (Retrieval-Augmented Generation) system, using FastAPI for the backend and React for the frontend.

## Project Summary

The Moffitt Agentic RAG System provides an intelligent interface for querying and retrieving information about researchers at the Moffitt Cancer Center. The system enables natural language queries such as:

- "Who in BioEngineering studies cancer evolution?"
- "Show researchers collaborating between Biostatistics and Cancer Epidemiology."
- "Find experts in immunotherapy at Moffitt."

## Architecture Overview

The project follows a modern architecture with a FastAPI backend and React frontend, implemented as a full-stack monorepo:

```
User Query
   │
   ▼
┌─────────────────────────┐
│  Agent (LangChain ReAct)│
│  - Analyzes query       │
│  - Selects tools        │
│  - Orchestrates search  │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Specialized Tools      │
│  - ResearcherSearch     │
│  - DepartmentFilter     │
│  - ProgramFilter        │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Vector Database        │
│  (ChromaDB)             │
└─────────────┬───────────┘
              │
              ▼
        Response to User
```

### Architecture Note

This branch contains the FastAPI/React implementation of the Moffitt Researcher Agent. If you're looking for the previous Streamlit implementation, it's available in the `archive/streamlit-implementation` branch.

### Key Components

1. **Backend (FastAPI)**
   - RESTful API endpoints for query processing
   - Agent orchestration using LangChain
   - Vector database integration with ChromaDB
   - Multi-provider LLM support (OpenAI, Groq)

2. **Frontend (React + TypeScript)**
   - Chat interface for natural language interaction
   - Responsive design with Tailwind CSS
   - API client integration with backend

3. **Agent System**
   - LangChain ReAct framework for reasoning
   - Specialized tools for different search capabilities
   - Reflection mechanism for improved responses

4. **Vector Database**
   - Efficient chunking strategy for researcher profiles
   - Hybrid search combining vector similarity and keyword matching

## Project Structure

- `backend/` - FastAPI backend
- `frontend/` - React frontend
- `data/` - Data directory for researcher profiles and vector database
- `docs/` - Documentation for the system

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
The frontend will be available at http://localhost:3000

API documentation is available at http://localhost:8000/docs

## Running Locally

For local development, you can run the backend and frontend separately.

### Backend

See the [backend README](backend/README.md) for setup and running instructions. For deeper details:
- Technical reference: `backend/docs/TECHNICAL_REFERENCE.md`
- Name normalization design: `backend/docs/NAME_NORMALIZATION_DESIGN.md`

### Frontend

See the [frontend README](frontend/README.md) for setup and running instructions.

## Rebuilding the Vector Database

If you need to rebuild the vector database from the processed researcher profiles:

```bash
# From project root (CLI)
python rebuild_db.py --force

# Or via backend API (if exposed)
POST /api/admin/vector-db/rebuild?force=true
```

**Options:**
- `--force`: Force rebuild even if database exists
- `--no-backup`: Skip backing up the existing database

The rebuild process:
1. Loads all researcher profiles from `data/processed/`
2. Creates chunks following the documented strategy (core, interests, publications, grants)
3. Generates embeddings using the configured model
4. Stores in `data/vector_db/` as a ChromaDB collection

**Architecture Note**: Both the CLI script and backend API share the same core rebuild logic from `backend/app/services/vector_db_builder.py`, ensuring consistency across interfaces.

**Note**: Rebuilding takes several minutes depending on the number of profiles and your hardware.

## Project Phase Overview

### Phase 1 — Data Processing

- Crawl and parse researcher profiles from Moffitt Cancer Center
- Convert to structured data format (JSON)
- Create clean markdown representations
- Validate with schema ensuring required fields

### Phase 2 — Agentic AI Implementation

- Create vector database using ChromaDB
- Implement LLM integration with multiple providers
- Build specialized tools for different query types
- Develop agent orchestration using LangChain
- Create frontend interface for user interaction

## Architectural Benefits

This project has undergone an architectural transition from a Streamlit monolithic application to a FastAPI backend with a React frontend. This change offers several benefits:

1. **Better Separation of Concerns**: Distinct backend and frontend layers
2. **Improved Performance**: Optimized API with lightweight frontend
3. **Enhanced User Experience**: Modern React-based UI with responsive design
4. **Better Scalability**: Independent scaling of frontend and backend components
5. **Superior Developer Experience**: Specialized frontend and backend technologies

## Development Resources

For detailed development documentation, see:
- [DEVELOPMENT.md](docs/DEVELOPMENT.md) - Comprehensive code documentation
- [TOOLS.md](docs/TOOLS.md) - Specialized tools documentation

