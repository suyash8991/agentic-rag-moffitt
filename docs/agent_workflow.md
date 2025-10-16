# Agent Workflow and File Interactions

This document explains the end-to-end workflow of the Moffitt Agentic RAG system, from user query to response generation, and details how the various files and components interact.

## End-to-End Query Flow

The agent workflow involves multiple components working together in a sequential pipeline:

1. **User Interface** → **Agent** → **Tools** → **Vector Database** → **Response Generation**

Let's examine each step in detail:

## Agent Initialization

The process begins in `src/moffitt_rag/agents/agent.py` with the `create_researcher_agent` function:

```python
def create_researcher_agent(
    llm_provider: Optional[LLMProvider] = None,
    model_name: Optional[str] = None,
    system_message: Optional[str] = None,
    temperature: float = 0.2,
    enable_reflection: bool = True
) -> AgentExecutor:
```

This function sets up the agent with the following steps:

1. **Language Model Initialization**:
   ```python
   llm = get_llm_model(
       provider=llm_provider,
       model_name=model_name,
       temperature=temperature
   )
   ```
   - The language model is loaded using the specified provider (OpenAI, Groq, Euron, or Ollama)
   - For Euron provider, streaming is disabled
   - Error handling includes detailed logging and traceback information

2. **Tool Creation**:
   ```python
   tools = [
       ResearcherSearchTool(),
       DepartmentFilterTool(),
       ProgramFilterTool(),
       InterestMatchTool(),
       CollaborationTool()
   ]
   ```
   - Each tool is instantiated
   - Tools are configured for the specific task of researcher information retrieval

3. **Prompt Template Setup**:
   ```python
   prompt = PromptTemplate.from_template(
       template=AGENT_PROMPT_TEMPLATE,
       partial_variables={
           "system_message": system_message or DEFAULT_SYSTEM_PROMPT,
           "tool_names": ", ".join([tool.name for tool in tools])
       }
   )
   ```
   - Uses a predefined template with placeholders for system message, tools, and input
   - System message can be customized or uses the default

4. **ReAct Agent Creation**:
   ```python
   agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
   ```
   - Creates a ReAct agent using LangChain's framework
   - ReAct agents follow a Thought-Action-Observation pattern

5. **Agent Executor Creation**:
   ```python
   agent_executor = AgentExecutor(
       agent=agent,
       tools=tools,
       verbose=True,
       handle_parsing_errors=True
   )
   ```
   - Creates an executor that manages the agent's interactions
   - Handles parsing errors gracefully

6. **Reflection Capability Addition** (optional):
   ```python
   if enable_reflection:
       from .reflection import create_reflective_agent_executor
       agent_executor = create_reflective_agent_executor(agent_executor)
   ```
   - If enabled, wraps the agent executor in a reflective wrapper
   - This adds self-reflection capabilities to improve answers

## Query Processing

When a user submits a query, the agent is invoked:

```python
response = agent.invoke({"input": "Who studies cancer evolution at Moffitt?"})
```

The agent processes the query through these steps:

1. **Query Analysis**:
   - The agent receives the query and begins processing
   - The LLM analyzes the query to determine intent and required tools
   - This analysis is guided by the system prompt instructions

2. **ReAct Reasoning**:
   - The agent follows the ReAct pattern: Thought → Action → Observation
   - Example of this pattern:
     ```
     Thought: I need to find researchers who study cancer evolution at Moffitt.
     Action: ResearcherSearch
     Action Input: cancer evolution
     Observation: [Tool Results]
     Thought: Based on these results, I can see several researchers work on cancer evolution...
     ```

3. **Tool Selection**:
   - The agent selects the appropriate tool based on the query type
   - Selection is guided by the instructions in the system prompt:
     ```
     1. For general researcher searches, use ResearcherSearch
     2. To find researchers in a specific department, use DepartmentFilter
     3. To find researchers in a specific program, use ProgramFilter
     4. To find researchers with similar interests, use InterestMatch
     5. To discover potential collaborations, use Collaboration
     ```

4. **Tool Invocation**:
   - The agent invokes the selected tool with the appropriate input
   - The tool executes its search or filtering logic
   - Results are returned to the agent for further processing

## Tool Execution

The selected tool (e.g., `ResearcherSearchTool`) is executed with the following process:

1. **Query Analysis**:
   - The tool analyzes the query to determine the search type
   - For `ResearcherSearch`, it determines if it's a name search or topic search
   - For name searches, it sets alpha=0.3 (favoring keyword matching)
   - For topic searches, it sets alpha=0.7 (favoring semantic matching)

2. **Search Execution**:
   - The tool executes the appropriate search method
   - For known researchers, it may perform a direct metadata lookup
   - Otherwise, it uses hybrid search with the determined alpha value
   - Example:
     ```python
     results = hybrid_search(query=query, k=5, alpha=alpha)
     ```

3. **Result Formatting**:
   - The tool formats the results as a string
   - Includes researcher names, programs, relevant snippets, and profile URLs
   - Returns the formatted results to the agent

## Database Interaction

The hybrid search interacts with the vector database through these steps:

1. **Vector Database Access**:
   ```python
   db = get_or_create_vector_db()
   ```
   - Gets an existing database or creates a new one if needed
   - Connects to the embedding model (SentenceTransformers)

2. **Semantic Search Execution**:
   ```python
   semantic_results = similarity_search_with_score(query, k=k*2, filter=filter, db=db)
   ```
   - Performs a vector similarity search based on embeddings
   - Returns documents with similarity scores

3. **Keyword Search Execution**:
   ```python
   keyword_results = keyword_search(query, texts, metadatas, k=k*2)
   ```
   - Performs a keyword-based search
   - Uses special name matching logic for researcher names
   - Returns documents with keyword match scores

4. **Result Combination**:
   ```python
   # Calculate combined scores
   for doc_id, scores in combined_scores.items():
       combined_scores[doc_id]['combined_score'] = (
           alpha * semantic_score + (1 - alpha) * scores['keyword_score']
       )
   ```
   - Combines semantic and keyword scores using the alpha parameter
   - Higher alpha values give more weight to semantic search
   - Lower alpha values give more weight to keyword search

5. **Result Selection**:
   ```python
   # Sort by combined score (descending)
   sorted_results = sorted(
       combined_scores.values(),
       key=lambda x: x['combined_score'],
       reverse=True
   )

   # Get the top results
   top_results = sorted_results[:k]
   ```
   - Sorts results by combined score
   - Returns the top k results

## Response Generation and Reflection

Once the tools have provided results, the agent generates a response:

1. **Result Processing**:
   - The agent receives results from the tools
   - It continues its reasoning process using the ReAct framework
   - Additional tools may be invoked if needed for clarification

2. **Response Synthesis**:
   - The agent uses the LLM to synthesize a coherent response
   - It incorporates information from tool results
   - It provides citations and sources for the information

3. **Reflection Process** (if enabled):
   ```python
   def invoke(self, inputs: Dict[str, Any], **kwargs):
       # Call the original invoke method
       preliminary_result = self.agent_executor.invoke(inputs, **kwargs)

       # Extract the question and answer
       question = inputs if isinstance(inputs, str) else inputs.get("input", "")
       answer = preliminary_result.get("output", "")

       # Reflect on the answer
       improved_answer = reflect_on_answer(
           question=question,
           answer=answer
       )

       # Update the result
       preliminary_result["output"] = improved_answer
       return preliminary_result
   ```
   - Takes the preliminary answer
   - Passes it to the reflection function
   - Reflection evaluates the answer based on criteria:
     1. Did it directly answer the user's question?
     2. Did it provide specific details about researchers?
     3. Is the information accurate and properly attributed?
     4. Could the organization be improved?
     5. Is clarification needed?
   - Reflection generates an improved answer
   - The improved answer is returned to the user

## File Interactions

The system involves several key file interactions:

### Data Loading Files

- `src/moffitt_rag/data/loader.py`:
  - Functions: `load_researcher_profile`, `load_all_researcher_profiles`, `create_researcher_chunks`
  - Loads JSON files from `data/processed/` directory
  - Processes profiles into chunks for the vector database

- `src/moffitt_rag/data/models.py`:
  - Classes: `ResearcherProfile`, `ResearcherChunk`, `Publication`, `Grant`, `Education`, `Contact`
  - Defines Pydantic models for structured data
  - Includes methods for text conversion and document creation

### Vector Database Files

- `src/moffitt_rag/db/vector_store.py`:
  - Functions: `create_vector_db`, `load_vector_db`, `get_or_create_vector_db`, `similarity_search`
  - Manages interactions with ChromaDB
  - Handles database creation, loading, and querying
  - Database files stored in `vector_db/` directory

- `src/moffitt_rag/db/hybrid_search.py`:
  - Functions: `keyword_search`, `hybrid_search`
  - Classes: `HybridRetriever`
  - Implements hybrid search algorithms
  - Combines vector similarity with keyword matching

### Tool Files

- `src/moffitt_rag/tools/researcher_search.py`:
  - Class: `ResearcherSearchTool`
  - Functions: `extract_name_from_url`, `extract_name_from_text`, `extract_relevant_snippet`
  - Implements the main researcher search functionality

- `src/moffitt_rag/tools/department_filter.py`, `program_filter.py`, `interest_match.py`, `collaboration.py`:
  - Implement specialized search and filtering tools
  - Each handles a specific aspect of researcher information retrieval

### Agent Files

- `src/moffitt_rag/agents/agent.py`:
  - Function: `create_researcher_agent`
  - Constants: `DEFAULT_SYSTEM_PROMPT`, `AGENT_PROMPT_TEMPLATE`
  - Creates and configures the agent
  - Sets up the ReAct framework

- `src/moffitt_rag/agents/reflection.py`:
  - Function: `reflect_on_answer`, `create_reflective_agent_executor`
  - Class: `ReflectiveAgentExecutor`
  - Adds reflection capabilities to the agent
  - Improves answers before returning to the user

### LLM Integration Files

- `src/moffitt_rag/models/llm.py`:
  - Functions: `get_llm_model`, `generate_text`, `generate_structured_output`
  - Enum: `LLMProvider`
  - Provides interface to multiple LLM providers
  - Handles API calls, error recovery, and fallbacks

## Configuration Files

- `src/moffitt_rag/config/config.py`:
  - Class: `Settings`
  - Function: `get_settings`
  - Manages configuration using Pydantic models
  - Loads settings from environment variables

## Complete Workflow Diagram

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

## Example Agent Invocation

Here's a concrete example of how to invoke the agent:

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

# Invoke the agent with a query
response = agent.invoke({
    "input": "Who studies cancer evolution at Moffitt?"
})

# Print the response
print(response["output"])
```

This will:
1. Create a researcher agent using the Groq provider and the Llama 4 Scout model
2. Invoke the agent with a query about cancer evolution
3. Print the response, which will include information about researchers at Moffitt studying cancer evolution

## Data Flow Summary

To summarize the data flow through the system:

1. **JSON Files** → **ResearcherProfile Objects** → **ResearcherChunk Objects** → **ChromaDB**

2. **User Query** → **Agent** → **Tool Selection** → **Hybrid Search** → **Document Retrieval** → **Result Formatting** → **Response Generation** → **Reflection** → **Final Response**

This workflow enables the system to provide intelligent, accurate responses to queries about Moffitt Cancer Center researchers, leveraging the power of vector search and large language models in a coherent agent-based framework.