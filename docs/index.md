# Moffitt Agentic RAG System Documentation

## Overview

The Moffitt Agentic RAG (Retrieval-Augmented Generation) system is designed to provide intelligent access to researcher information from Moffitt Cancer Center. This documentation provides detailed information about how the system works, its components, and the implementation details.

## Table of Contents

1. [Input Data and ChromaDB Storage](input_data.md)
   - Data structure overview
   - Data loading process
   - ChromaDB storage implementation
   - Database structure details

2. [ChromaDB Chunking Process](chromadb_chunking.md)
   - Chunking strategy
   - Chunk ID structure
   - Code implementation
   - Deduplication process

3. [Agent Tools Logic](tools_logic.md)
   - ResearcherSearchTool
   - DepartmentFilterTool
   - ProgramFilterTool
   - InterestMatchTool
   - CollaborationTool

4. [Detailed Explanation of researcher_search.py](researcher_search.md)
   - Helper functions
   - ResearcherSearchTool class
   - Rate limiting logic
   - Name detection algorithms
   - Result formatting

5. [Agent Workflow and File Interactions](agent_workflow.md)
   - End-to-end query flow
   - Agent initialization
   - Query processing
   - Tool execution
   - Database interaction
   - Response generation

## System Architecture

The Moffitt Agentic RAG system uses a modular architecture:

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
│  - InterestMatch        │
│  - Collaboration        │
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

## Example Usage

Here's a simple example of how to use the system:

```python
from src.moffitt_rag.agents.agent import create_researcher_agent
from src.moffitt_rag.models.llm import LLMProvider

# Create the agent
agent = create_researcher_agent(
    llm_provider=LLMProvider.GROQ,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3,
    enable_reflection=True
)

# Ask a question
response = agent.invoke({
    "input": "Who studies cancer evolution at Moffitt?"
})

# Print the answer
print(response["output"])
```

## Key Features

- **Intelligent Search**: Combines vector similarity with keyword matching for optimal results
- **Agentic Decision Making**: Uses LLM-based agent to select the most appropriate search strategy
- **Specialized Tools**: Purpose-built tools for different search types (researcher, department, program)
- **Self-Reflection**: Ability to review and improve answers before responding
- **Multi-LLM Support**: Works with multiple language model providers (OpenAI, Groq, Euron, Ollama)
- **Chunking Optimization**: Intelligently chunks researcher profiles for efficient retrieval

## Project Structure

```
moffitt-agentic-rag/
├── data/                     # Data directory containing researcher profiles
│   ├── processed/            # JSON files of researcher profiles
│   ├── markdown/             # Markdown files of researcher profiles
│   └── raw_html/             # Original HTML content
│
├── docs/                     # Documentation files
│
├── src/                      # Source code
│   └── moffitt_rag/          # Main package
│       ├── config/           # Configuration management
│       ├── data/             # Data processing modules
│       ├── db/               # Vector database functionality
│       ├── models/           # Embedding and language models
│       ├── agents/           # Agent orchestration
│       ├── tools/            # Agent tools
│       ├── api/              # FastAPI backend (planned)
│       └── frontend/         # Streamlit frontend (in progress)
│
├── requirements.txt          # Project dependencies
└── setup.py                  # Package installation script
```

## Getting Started

To set up and run the system:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/moffitt-agentic-rag.git
   cd moffitt-agentic-rag
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the Streamlit App**:
   ```bash
   python run_app.py
   # or
   streamlit run app.py
   ```

## License

This project is proprietary and confidential. All rights reserved.

## Contributors

- Lead Engineer: Tejas Kumar Leelavathi
- Contact: [tejas.leelavathi@example.com](mailto:tejas.leelavathi@example.com)