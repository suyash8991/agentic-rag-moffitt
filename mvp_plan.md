# Agentic RAG System for Moffitt Cancer Center: Design Overview

As a senior AI engineer assessing this project, I'll outline an efficient approach to build a high-quality agentic RAG system with speed of implementation in mind.

## Data Assessment

Current assets:
- 127 structured researcher profiles
- Data includes: names, titles, research interests, publications, grants
- Multiple formats (HTML, Markdown, JSON) available
- Clean, structured data ready for embedding

## Technical Stack Selection

I recommend the following stack for rapid development with high performance:

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Vector Database** | **Chroma** | Open-source, Python-native, easy setup, supports hybrid search |
| **Embeddings** | **SentenceTransformers** (all-MiniLM-L6-v2) | Balanced speed/quality, 384-dim embeddings, fast inference |
| **Orchestration** | **LangChain** | Pre-built agents, tools framework, rapid development |
| **LLM Interface** | **Ollama** (Llama 3 8B or Mistral) | Local deployment, no token costs, privacy control |
| **Backend** | **FastAPI** | Async support, auto-docs, type checking |
| **Frontend** | **Streamlit** | Rapid prototyping, minimal code, built-in chat UI |
| **Deployment** | **Docker** + local | Self-contained, easy to migrate later |

## Rapid Implementation Plan (6-week timeline)

### Week 1: Embedding & Vector Storage (2-3 days)
1. Install dependencies: `pip install chromadb sentence-transformers langchain ollama`
2. Create embedding function using SentenceTransformers
3. Process researcher JSON files into embeddings
4. Store in Chroma with metadata (program, department, research areas)
5. Implement semantic search functionality

### Week 2: Basic RAG Implementation (2-3 days)
1. Connect embeddings to LangChain RAG pipeline
2. Set up retrieval with hybrid search (embedding + keyword)
3. Implement basic query processor with Llama 3 8B
4. Create simple prompt templates
5. Build response generator with source citations

### Week 3: Agent Capabilities (3-4 days)
1. Define specialized tools for the agent:
   - ResearcherSearchTool (semantic search)
   - DepartmentFilterTool (filter by department)
   - ProgramFilterTool (filter by program)
   - InterestMatchTool (find similar research interests)
   - CollaborationTool (find potential collaborators)
2. Implement agent using LangChain Agent framework
3. Create orchestrator for tool selection
4. Build response synthesizer

### Week 4: Advanced Features (3-4 days)
1. Implement cross-researcher reasoning
2. Add publication analysis capabilities
3. Develop grant/funding insights
4. Create collaboration network mapping
5. Implement follow-up question handling

### Week 5: Frontend & Integration (2-3 days)
1. Build Streamlit interface with chat functionality
2. Create visualization components (researcher networks, topic clusters)
3. Implement session management
4. Add feedback mechanism
5. Design explanation interface for agent reasoning

### Week 6: Testing, Refinement & Documentation (2-3 days)
1. Conduct user acceptance testing
2. Fine-tune prompts and retrieval mechanisms
3. Optimize performance bottlenecks
4. Create comprehensive documentation
5. Prepare for deployment

## Key Implementation Details

### Vector Database Design
```python
# Example Chroma schema
researcher_schema = {
    "document": "researcher biography/text",
    "metadata": {
        "researcher_id": "unique identifier",
        "name": "researcher name",
        "program": "primary program",
        "department": "department",
        "research_areas": ["list", "of", "research", "interests"],
        "degrees": ["PhD", "MD", "etc"],
        "publication_count": 42,
        "grant_count": 7,
        "source": "profile_url"
    }
}
```

### Hybrid Search Implementation
```python
def hybrid_search(query, k=5):
    """Combines semantic and keyword search for optimal retrieval"""
    semantic_results = chroma.similarity_search(query, k=k)
    keyword_results = chroma.keyword_search(query, k=k)

    # Merge and deduplicate results
    combined_results = merge_results(semantic_results, keyword_results)
    return combined_results[:k]
```

### Agent Architecture
```
User Query → Orchestrator → [Tool Selection]
                               ↓
                           [Retrieval]
                               ↓
                           [Reasoning]
                               ↓
                           [Response Generation]
                               ↓
                           [Citation Collection]
                               ↓
                           User Response
```

### Chunking Strategy
- Core researcher info (name, title, program, interests) as single chunk
- Publications chunked individually (Title + abstract + authors)
- Grants as individual chunks
- Biography broken into semantic paragraphs
- Cross-references maintained via metadata

## Distinctive Features

1. **Contextual Awareness**: Agent maintains awareness of researcher connections and institutional hierarchy

2. **Multi-hop Reasoning**: Can connect researchers by shared interests even without direct collaboration

3. **Transparent Citations**: Every claim links to source data with researcher profile reference

4. **Visualized Insights**: Network graphs show collaborations, topic clusters, and interdisciplinary connections

5. **Clarification Mechanism**: Agent asks clarifying questions for ambiguous queries

## Potential Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Distinguishing between similarly named researchers | Implement entity disambiguation using metadata |
| Handling sparse data fields | Develop fallback strategies for missing information |
| Tracking citation sources | Create source-tracking mechanism in retrieval pipeline |
| Performance bottlenecks with many queries | Implement caching for common queries |
| Balancing recall vs. precision | Dynamic k parameter based on query complexity |

## Rapid Startup Code (Key Components)

### 1. Embedding & Database Setup
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os, json

# Fast, balanced embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Process researcher data
researchers = []
for filename in os.listdir("data/processed"):
    if filename.endswith(".json"):
        with open(f"data/processed/{filename}", "r") as f:
            researcher = json.load(f)

            # Create document from researcher data
            text = f"Name: {researcher['name']}\n"
            text += f"Title: {researcher.get('title', '')}\n"
            text += f"Program: {researcher.get('primary_program', '')}\n"
            text += f"Research interests: {', '.join(researcher.get('research_interests', []))}\n"
            text += f"Overview: {researcher.get('overview', '')}\n"

            # Add to collection with metadata
            researchers.append({
                "text": text,
                "metadata": {
                    "researcher_id": researcher.get("researcher_id", ""),
                    "name": researcher.get("name", ""),
                    "program": researcher.get("primary_program", ""),
                    "department": researcher.get("department", ""),
                    "research_interests": researcher.get("research_interests", []),
                    "source": researcher.get("profile_url", "")
                }
            })

# Create vector store
db = Chroma.from_texts(
    texts=[r["text"] for r in researchers],
    embedding=embeddings,
    metadatas=[r["metadata"] for r in researchers],
    persist_directory="chroma_db"
)
```

### 2. Agent Definition
```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Setup LLM
llm = Ollama(model="llama3")

# Define tools
tools = [
    Tool(
        name="ResearcherSearch",
        func=lambda query: db.similarity_search(query, k=5),
        description="Search for researchers by their expertise, interests, or background"
    ),
    Tool(
        name="ProgramFilter",
        func=lambda program: db.similarity_search_with_relevance_scores(
            "", k=10, filter={"program": program}
        ),
        description="Filter researchers by their program"
    ),
    # Additional tools...
]

# Create agent prompt
agent_prompt = PromptTemplate.from_template("""
You are an AI assistant for Moffitt Cancer Center.
Your job is to help users find information about researchers, their expertise,
and potential collaborations.

{format_instructions}

User Query: {input}
{agent_scratchpad}
""")

# Create agent
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### 3. FastAPI + Streamlit Integration
```python
import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/query")
async def process_query(query: Query):
    try:
        response = agent_executor.run(query.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit app (separate file)
st.title("Moffitt Researcher Intelligence")
query = st.chat_input("Ask about Moffitt researchers...")
if query:
    response = requests.post("http://localhost:8000/query", json={"text": query}).json()
    st.write(response["response"])
```

## Development Velocity Recommendations

1. **Daily Builds**: Implement continuous integration with daily milestones
2. **Staged Deployment**: Roll out features incrementally (search → basic RAG → agent → UI)
3. **Parallel Development**: Work on embedding/DB and agent logic simultaneously
4. **Feature Flags**: Implement toggles for experimental features
5. **Rapid Prototyping**: Use notebooks for quick experimentation before integration

By following this approach, you can have a functional prototype within 2-3 weeks and a production-ready system within 6 weeks, leveraging existing open-source components while building a customized solution for the Moffitt Cancer Center's specific needs.