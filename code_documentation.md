# Moffitt Agentic RAG System: Code Documentation

This document provides an overview of the codebase structure, explains the purpose of each module, and documents key functions to make navigation easier.

## Project Structure

```
moffitt-agentic-rag/
├── data/                     # Data directory containing researcher profiles
│   ├── processed/            # JSON files of researcher profiles
│   ├── markdown/             # Markdown files of researcher profiles
│   └── raw_html/             # Original HTML content
│
├── schemas/                  # JSON schemas for data validation
│
├── src/                      # Source code
│   └── moffitt_rag/          # Main package
│       ├── config/           # Configuration management
│       ├── data/             # Data processing modules
│       ├── db/               # Vector database functionality
│       ├── models/           # Embedding and language models
│       ├── agents/           # Agent orchestration
│       ├── tools/            # Agent tools
│       ├── api/              # FastAPI backend
│       └── frontend/         # Streamlit frontend
│
├── requirements.txt          # Project dependencies
└── setup.py                  # Package installation script
```

## Module Documentation

### Configuration (`src/moffitt_rag/config/`)

#### `config.py`
**Purpose**: Centralized configuration management for the entire system.

**Key Components**:
- `Settings` class: Pydantic model for all system settings
- Environment variable integration via `dotenv`
- Path configuration for data directories
- Model settings for embeddings and LLMs

**Key Functions**:
- `get_settings()`: Returns the global settings instance

**Usage**:
```python
from moffitt_rag.config.config import get_settings

settings = get_settings()
data_dir = settings.processed_data_dir
```

### Data Processing (`src/moffitt_rag/data/`)

#### `models.py`
**Purpose**: Defines structured data models for researcher profiles and related information.

**Key Components**:
- `ResearcherProfile`: Main model for researcher data
- Supporting models: `Publication`, `Grant`, `Education`, `Contact`
- `ResearcherChunk`: Model for text chunks used in vector database

**Key Methods**:
- `ResearcherProfile.to_document()`: Converts profile to format for embedding
- `ResearcherProfile.to_text()`: Creates text representation of profile
- Property getters for derived attributes (`full_name`, `publication_count`, etc.)

**Usage**:
```python
from moffitt_rag.data.models import ResearcherProfile

profile = ResearcherProfile.model_validate(data)
text = profile.to_text()
```

#### `loader.py`
**Purpose**: Loads researcher profiles from JSON files and processes them for embedding.

**Key Functions**:
- `load_researcher_profile(file_path)`: Loads a single profile from JSON
- `load_all_researcher_profiles()`: Loads all profiles from the processed directory
- `create_researcher_chunks(profile)`: Divides profiles into chunks for optimal retrieval
- `load_all_chunks()`: Loads all profiles and creates chunks for them
- `get_researcher_stats()`: Generates statistics about the dataset

**Usage**:
```python
from moffitt_rag.data.loader import load_all_chunks

chunks = load_all_chunks()
```

### Vector Database (`src/moffitt_rag/db/`)

#### `vector_store.py`
**Purpose**: Manages the Chroma vector database for storing and retrieving researcher profile embeddings.

**Key Components**:
- Database creation and management functions
- Retriever creation for LangChain integration
- Various search functions with different retrieval strategies

**Key Functions**:
- `create_vector_db(chunks)`: Creates a new vector database from researcher chunks
- `load_vector_db()`: Loads an existing database from disk
- `get_or_create_vector_db()`: Gets existing database or creates a new one
- `create_retriever(db, search_type, search_kwargs)`: Creates a LangChain retriever
- `similarity_search(query, k, filter)`: Performs semantic search
- `max_marginal_relevance_search(query, k, fetch_k, lambda_mult)`: Performs search with diversity

**Usage**:
```python
from moffitt_rag.db.vector_store import get_or_create_vector_db, similarity_search

# Get or create the database
db = get_or_create_vector_db()

# Perform a search
results = similarity_search("cancer evolution research", k=5)
```

#### `hybrid_search.py`
**Purpose**: Implements hybrid search that combines vector similarity with keyword matching.

**Key Components**:
- Keyword-based search function
- Weighted hybrid search combining semantic and keyword approaches
- LangChain-compatible hybrid retriever class

**Key Functions**:
- `keyword_search(query, texts, metadata)`: Performs keyword matching
- `hybrid_search(query, k, alpha)`: Combines semantic and keyword search with weighted scoring
- `HybridRetriever`: LangChain retriever implementing hybrid search

**Usage**:
```python
from moffitt_rag.db.hybrid_search import hybrid_search, HybridRetriever

# Perform a hybrid search
results = hybrid_search("cancer evolution research", k=5, alpha=0.7)

# Create a hybrid retriever for LangChain
retriever = HybridRetriever(k=5, alpha=0.7)
```

### Embedding and Language Models (`src/moffitt_rag/models/`)

#### `embeddings.py`
**Purpose**: Provides functionality for generating and managing embeddings using SentenceTransformers.

**Key Components**:
- Embedding model creation with configurable model name
- Functions for generating embeddings for texts and queries
- Utility for calculating embedding similarity

**Key Functions**:
- `get_embedding_model(model_name)`: Creates a HuggingFace embedding model
- `generate_embeddings(texts)`: Generates embeddings for a list of texts
- `embed_query(query)`: Generates an embedding for a single query
- `embedding_similarity(embedding1, embedding2)`: Calculates cosine similarity

**Usage**:
```python
from moffitt_rag.models.embeddings import generate_embeddings, embed_query

# Generate embeddings for multiple texts
texts = ["Researcher focuses on cancer genomics", "Expert in immunotherapy"]
embeddings = generate_embeddings(texts)

# Generate embedding for a query
query_embedding = embed_query("cancer research")
```

#### `llm.py`
**Purpose**: Provides an interface for using language models from multiple providers (OpenAI, Groq, Ollama).

**Key Components**:
- Multi-provider LLM support (OpenAI, Groq, Ollama)
- Environment-based configuration
- Text generation with system prompts
- Structured JSON output generation

**Key Functions**:
- `get_llm_model(provider, model_name)`: Creates an LLM from the specified provider
- `generate_text(prompt, system_prompt)`: Generates text from a prompt
- `generate_structured_output(prompt, output_schema)`: Generates structured JSON output

**Usage**:
```python
from moffitt_rag.models.llm import generate_text, LLMProvider

# Generate text using Groq
response = generate_text(
    "Explain the benefits of vector databases for RAG systems",
    provider=LLMProvider.GROQ
)

# Generate text using OpenAI
response = generate_text(
    "Summarize this researcher's focus",
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o"
)
```

### Agent Orchestration (`src/moffitt_rag/agents/`)

*(Not yet implemented)*

This module will handle:
- Agent configuration and setup
- Tool selection and orchestration
- Response generation with citations

### Agent Tools (`src/moffitt_rag/tools/`)

#### `researcher_search.py`
**Purpose**: Implements a tool for searching researchers by expertise, interests, or background.

**Key Components**:
- Semantic search using hybrid search functionality
- Relevant snippet extraction from search results
- Formatted results with researcher details

**Key Functions**:
- `extract_relevant_snippet(text, query)`: Extracts relevant snippets from text
- `ResearcherSearchTool._run(query)`: Performs the search and formats results

**Usage**:
```python
from moffitt_rag.tools import ResearcherSearchTool

# Create the tool
tool = ResearcherSearchTool()

# Use the tool
results = tool.run("cancer immunotherapy resistance mechanisms")
```

#### `department_filter.py`
**Purpose**: Filters researchers by their academic department.

**Key Components**:
- Department-based filtering using metadata
- Department listing functionality
- Flexible matching for partial department names

**Key Functions**:
- `DepartmentFilterTool._run(department)`: Filters researchers by department
- `DepartmentFilterTool._list_departments()`: Lists all available departments

**Usage**:
```python
from moffitt_rag.tools import DepartmentFilterTool

# Create the tool
tool = DepartmentFilterTool()

# List all departments
departments = tool.run("list")

# Filter by department
immunology_researchers = tool.run("Immunology")
```

#### `program_filter.py`
**Purpose**: Filters researchers by their research program.

**Key Components**:
- Program-based filtering using metadata
- Program listing functionality
- Combines exact matches with semantic search

**Key Functions**:
- `ProgramFilterTool._run(program)`: Filters researchers by program
- `ProgramFilterTool._list_programs()`: Lists all available programs

**Usage**:
```python
from moffitt_rag.tools import ProgramFilterTool

# Create the tool
tool = ProgramFilterTool()

# List all programs
programs = tool.run("list")

# Filter by program
bioengineering_researchers = tool.run("BioEngineering")
```

#### `interest_match.py`
**Purpose**: Finds researchers with similar research interests.

**Key Components**:
- Similar researcher discovery
- Interest-based matching
- Relevant interest and snippet extraction

**Key Functions**:
- `InterestMatchTool._find_similar_researchers(researcher_name)`: Finds similar researchers
- `InterestMatchTool._find_matching_interests(query)`: Finds researchers by interests
- `InterestMatchTool._find_relevant_interests(interests, query)`: Finds relevant interests

**Usage**:
```python
from moffitt_rag.tools import InterestMatchTool

# Create the tool
tool = InterestMatchTool()

# Find researchers similar to a named researcher
similar_researchers = tool.run("similar to John Cleveland")

# Find researchers with specific interests
cancer_evolution_researchers = tool.run("cancer evolution")
```

#### `collaboration.py`
**Purpose**: Discovers potential collaborations between researchers or departments.

**Key Components**:
- Interdepartmental collaboration discovery
- Researcher-specific collaboration suggestions
- Common research interest detection

**Key Functions**:
- `CollaborationTool._find_interdepartmental_collaborations(query)`: Finds collaborations between departments
- `CollaborationTool._find_researcher_collaborations(query)`: Finds collaborators for a researcher
- `CollaborationTool._find_collaborations_between_groups(group1, group2)`: Finds collaborations between groups
- `CollaborationTool._find_common_interests(r1, r2)`: Finds common interests between researchers

**Usage**:
```python
from moffitt_rag.tools import CollaborationTool

# Create the tool
tool = CollaborationTool()

# Find interdepartmental collaborations
dept_collaborations = tool.run("between Biostatistics and Cancer Epidemiology")

# Find collaborators for a researcher
researcher_collaborations = tool.run("for John Cleveland")
```

### API (`src/moffitt_rag/api/`)

*(Not yet implemented)*

This module will provide:
- FastAPI routes for querying the system
- Request/response models
- Error handling and logging

### Frontend (`src/moffitt_rag/frontend/`)

*(Not yet implemented)*

This module will include:
- Streamlit chat interface
- Visualization components
- User session management