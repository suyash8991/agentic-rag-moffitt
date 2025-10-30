# Moffitt Agentic RAG System: Development Guide

This document provides comprehensive documentation for developers working on the Moffitt Agentic RAG system, explaining the codebase structure, key components, and their interactions.

## Project Structure

```
moffitt-agentic-rag/
├── backend/                          # FastAPI Backend Application
│   ├── app/                          # Main application code
│   │   ├── api/                      # API endpoints
│   │   │   ├── endpoints/
│   │   │   │   ├── admin.py          # Admin operations
│   │   │   │   ├── query.py          # Query processing
│   │   │   │   └── researchers.py    # Researcher data endpoints
│   │   │   └── dependencies.py       # Shared dependencies
│   │   ├── core/                     # Core application config
│   │   │   ├── config.py             # Environment settings
│   │   │   ├── prompts.py            # LLM prompts
│   │   │   └── security.py           # API key authentication
│   │   ├── models/                   # Data models (Pydantic)
│   │   │   ├── query.py              # Query request/response models
│   │   │   └── researcher.py         # Researcher profile models
│   │   ├── services/                 # Business logic
│   │   │   ├── agent.py              # Agent orchestration
│   │   │   ├── llm.py                # LLM provider integration
│   │   │   ├── researcher.py         # Researcher data service
│   │   │   ├── tools.py              # Tool implementations
│   │   │   ├── vector_db.py          # Vector database service
│   │   │   ├── vector_db_builder.py  # Shared rebuild logic
│   │   │   └── limited_call.py       # Rate limiting
│   │   └── utils/                    # Utility functions
│   ├── main.py                       # FastAPI application entry
│   └── requirements.txt              # Python dependencies
├── frontend/                         # React Frontend Application
│   ├── src/
│   │   ├── components/               # React components
│   │   ├── services/                 # API client services
│   │   └── assets/                   # Static assets
├── data/                             # Data Storage
│   ├── markdown/                     # Researcher data in markdown
│   ├── processed/                    # Processed data files
│   └── vector_db/                    # ChromaDB vector storage
├── rebuild_db.py                     # CLI script for vector DB rebuild
└── docs/                             # Project Documentation
```

## End-to-End Query Flow

The agent workflow involves multiple components working together in a sequential pipeline:

1. **User Interface** → **Agent** → **Tools** → **Vector Database** → **Response Generation**

```
User Query
   │
   ▼
┌─────────────────────────┐
│  Agent (agent.py)       │
│  - Analyzes query       │
│  - Selects tools        │
│  - Orchestrates search  │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Tools (tools/*.py)     │
│  - ResearcherSearch     │
│  - DepartmentFilter     │
│  - ProgramFilter        │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Hybrid Search          │
│  (hybrid_search.py)     │
│  - Vector search        │
│  - Keyword search       │
│  - Combined scoring     │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Vector Database        │
│  (vector_store.py)      │
│  - ChromaDB storage     │
│  - Embedding generation │
│  - Document retrieval   │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Response Generation    │
│  - LLM synthesis        │
│  - Reflection           │
│  - Source citation      │
└─────────────┬───────────┘
              │
              ▼
        Final Response
```

## Key Components Documentation

### 1. Configuration (`backend/app/core/`)

#### `config.py`
**Purpose**: Centralized configuration management using Pydantic's settings.

**Key Settings**:
- API configuration (keys, endpoints)
- Vector database paths
- LLM provider settings (OpenAI, Groq)
- CORS configuration

**Usage**:
```python
from app.core.config import settings

data_dir = settings.VECTOR_DB_DIR
llm_provider = settings.LLM_PROVIDER
```

### 2. Agent System (`backend/app/services/agent.py`)

**Purpose**: Orchestrates query processing using LangChain's ReAct framework.

**Key Functions**:
- `create_researcher_agent()`: Creates an agent with configured tools and LLM
- `process_query()`: Processes user queries through the agent

**Agent Creation Process**:
1. Initialize language model (OpenAI/Groq)
2. Create specialized tools
3. Set up prompt templates
4. Create ReAct agent
5. Apply call limiting and reflection (optional)

**Usage**:
```python
from app.services.agent import create_researcher_agent

agent = create_researcher_agent(temperature=0.7, max_llm_calls=6)
response = agent.invoke({"input": "Who studies cancer evolution?"})
```

### 3. LLM Integration (`backend/app/services/llm.py`)

**Purpose**: Interface to multiple language model providers.

**Supported Providers**:
- OpenAI (GPT models)
- Groq (Llama models)
- Fallback mechanisms

**Key Functions**:
- `get_llm_model()`: Creates LLM instance with specified provider
- `generate_text()`: Generates text from prompt
- `generate_structured_output()`: Generates JSON-structured responses

### 4. Vector Database (`backend/app/services/vector_db.py`)

**Purpose**: Manages ChromaDB for storing and retrieving researcher profile embeddings.

**Key Functions**:
- `get_embedding_function()`: Gets embedding model
- `load_vector_db()`: Loads existing database
- `get_or_create_vector_db()`: Gets or creates database
- `similarity_search()`: Performs semantic search

**Database Schema**:
Each document contains:
- Text content (researcher information)
- Metadata (researcher_id, name, program, department, etc.)
- Embedding vector

### 5. Chunking Strategy (`backend/app/services/vector_db_builder.py`)

**Purpose**: Optimizes retrieval by dividing researcher profiles into logical chunks.

**Chunk Types**:
- `core`: Basic researcher information (name, title, program, department)
- `interests`: Research interests
- `publications`: Publications (may span multiple chunks)
- `grants`: Grant information

**Chunk ID Structure**:
```
{researcher_id}_{hash_prefix}_{chunk_type}[_{index}]
```

For example:
- `24764_8a5b_core` - Core information
- `24764_8a5b_interests` - Research interests
- `24764_8a5b_pubs_0` - First publications chunk

**Benefits**:
- Improved retrieval relevance
- Optimized token usage
- Better context management

### 6. Vector Database Rebuild (`backend/app/services/vector_db_builder.py`)

**Purpose**: Provides shared rebuild logic for both CLI and API interfaces.

**Architecture**: The rebuild functionality is centralized in a shared module to ensure consistency and avoid code duplication:

```
┌─────────────────────────┐
│  CLI Script             │
│  (rebuild_db.py)        │
│  - Arg parsing          │
│  - Path overrides       │
│  - User feedback        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Shared Module          │◄─────────────┐
│  (vector_db_builder.py) │              │
│  - Profile loading      │              │
│  - Chunking logic       │              │
│  - Embedding generation │              │
│  - Database creation    │              │
│  - Progress callbacks   │              │
└───────────┬─────────────┘              │
            │                            │
            ▼                            │
┌─────────────────────────┐              │
│  Backend API            │              │
│  (vector_db.py)         │──────────────┘
│  - Task management      │
│  - Progress tracking    │
│  - Stats updates        │
└─────────────────────────┘
```

**Key Functions**:
- `load_researcher_profile()`: Loads single JSON profile
- `load_all_researcher_profiles()`: Loads all profiles from directory
- `create_researcher_chunks()`: Implements chunking strategy
- `create_vector_db()`: Creates ChromaDB with embeddings
- `backup_existing_database()`: Backs up existing database
- `rebuild_vector_database()`: Main orchestration function with progress callbacks

**Usage Examples**:

```python
# From CLI script
from app.services.vector_db_builder import rebuild_vector_database

success = rebuild_vector_database(
    processed_dir=Path("data/processed"),
    vector_db_dir=Path("data/vector_db"),
    collection_name="researchers",
    backup=True,
    force=True
)

# From backend API with progress tracking
def update_progress(progress: float):
    task_info["progress"] = progress

success = rebuild_vector_database(
    processed_dir=Path(settings.PROCESSED_DATA_DIR),
    vector_db_dir=Path(settings.VECTOR_DB_DIR),
    collection_name=settings.COLLECTION_NAME,
    backup=True,
    force=force,
    progress_callback=update_progress
)
```

**Benefits**:
- Single source of truth for rebuild logic
- Consistent behavior across CLI and API
- Easier maintenance and testing
- Progress tracking support for frontend

### 7. API Endpoints (`backend/app/api/endpoints/`)

#### `query.py`
**Purpose**: Handles query requests and responses.

**Endpoints**:
- `POST /api/query`: Process a query
- `GET /api/query/{query_id}`: Get query status
- `GET /api/query/{query_id}/stream`: Stream query response (SSE)
- `WebSocket /api/ws/query`: WebSocket endpoint for streaming

#### `researchers.py`
**Purpose**: Provides researcher information.

**Endpoints**:
- `GET /api/researchers`: List researchers
- `GET /api/researchers/{id}`: Get researcher details
- `GET /api/departments`: List departments
- `GET /api/programs`: List programs

#### `admin.py`
**Purpose**: Administrative operations including database rebuild.

**Endpoints**:
- `POST /api/admin/vector-db/rebuild`: Rebuild vector database
- `GET /api/admin/vector-db/stats`: Get database statistics
- `GET /api/admin/vector-db/tasks/{task_id}`: Get rebuild task status

### 8. Frontend Components (`frontend/src/components/`)

#### Chat Interface
**Purpose**: User interaction with the agent.

**Key Components**:
- `ChatContainer`: Manages chat state and messages
- `ChatMessage`: Renders individual messages
- `ChatInput`: Handles user input

#### API Integration
**Purpose**: Communication with backend.

**Key Functions**:
- `sendQuery()`: Sends query to backend
- `fetchHealth()`: Checks API health
- `fetchSettings()`: Gets system settings

## Working with the Codebase

### Creating a New Tool

1. Create a new file in `backend/app/services/tools/`
2. Implement a class extending `BaseTool` from LangChain
3. Override `_run()` and `_arun()` methods
4. Add the tool to the list in `get_tools()`

Example:
```python
from langchain.tools import BaseTool

class MyNewTool(BaseTool):
    name: str = "MyNewTool"
    description: str = "Description of what the tool does"

    def _run(self, query: str) -> str:
        # Implementation
        return result

    def _arun(self, query: str) -> str:
        # Async implementation (optional)
        return self._run(query)

# In tools.py, add to get_tools()
def get_tools():
    return [
        ResearcherSearchTool(),
        DepartmentFilterTool(),
        ProgramFilterTool(),
        MyNewTool()  # Add your new tool
    ]
```

### Modifying the Agent

To customize the agent's behavior:

1. Modify the system prompt in `backend/app/core/prompts.py`
2. Adjust the agent creation in `backend/app/services/agent.py`
3. Update the query processing in `backend/app/api/endpoints/query.py`

### Troubleshooting

**Vector Database Issues**:
- Check that `data/vector_db` directory exists
- Verify embeddings model is accessible
- Use `load_vector_db()` with debug=True

**LLM Provider Issues**:
- Check API keys in environment variables
- Verify network connectivity to provider APIs
- Set fallback providers in config

**Agent Execution Issues**:
- Enable verbose mode in `AgentExecutor`
- Check logs for parsing errors
- Verify tool formats match prompt instructions

## Example: Agent Invocation

```python
from app.services.agent import create_researcher_agent

# Create agent
agent = create_researcher_agent(temperature=0.3, max_llm_calls=5)

# Process query
result = agent.invoke({
    "input": "Who studies cancer evolution at Moffitt?"
})

# Extract answer
answer = result.get("output", str(result))
print(answer)
```

This will:
1. Create a researcher agent
2. Invoke the agent with a query about cancer evolution
3. Extract and print the response

## API Testing

The API can be tested using the FastAPI Swagger documentation at http://localhost:8000/docs, which provides an interactive interface for all endpoints.

For programmatic testing:

```python
import requests

API_URL = "http://localhost:8000/api"
API_KEY = "your_api_key"

# Send a query
response = requests.post(
    f"{API_URL}/query",
    headers={"X-API-Key": API_KEY},
    json={"query": "Who studies cancer evolution?", "streaming": False}
)

# Print the response
print(response.json())
```