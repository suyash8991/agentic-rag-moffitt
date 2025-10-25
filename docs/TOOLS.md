# Agent Tools Documentation

The Moffitt Agentic RAG system uses specialized tools to perform different search and filtering operations on researcher data. This document explains each tool's purpose, logic, and usage.

## Tool Selection Flow

The agent selects tools based on the user query, following these guidelines:

```
User Query
   │
   ▼
┌─────────────────────────┐
│  Query Analysis         │
│  - Intent detection     │
│  - Query type matching  │
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Tool Selection         │
│  ┌─────────────────────┐
│  │ ResearcherSearch    │◀── General researcher queries
│  │                     │    "Who studies cancer?"
│  └─────────────────────┘
│  ┌─────────────────────┐
│  │ DepartmentFilter    │◀── Department-specific queries
│  │                     │    "Researchers in Immunology"
│  └─────────────────────┘
│  ┌─────────────────────┐
│  │ ProgramFilter       │◀── Program-specific queries
│  │                     │    "BioEngineering researchers"
│  └─────────────────────┘
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Tool Execution         │
│  - Parameter formatting │
│  - Search execution     │
│  - Result formatting    │
└─────────────────────────┘
```

## ResearcherSearch Tool

**Purpose**: Searches for researchers by expertise, interests, or name.

**Implementation**: `backend/app/services/tools.py` - `ResearcherSearchTool` class

### Logic
1. **Query Analysis**:
   - Detects if the query is likely a name search (e.g., "who is Dr. Smith")
   - Recognizes research topic searches (e.g., "cancer genomics researchers")

2. **Search Parameter Adjustment**:
   - For name searches: `alpha=0.3` (favors keyword matching)
   - For topic searches: `alpha=0.7` (favors semantic matching)

3. **Search Process**:
   - Uses hybrid search combining vector similarity and keyword matching
   - Applies metadata filters when appropriate

4. **Result Formatting**:
   - Returns researcher name, program, department
   - Includes relevant content snippets
   - Provides profile URL

### Example Usage

```python
from backend.app.services.tools import ResearcherSearchTool

# Create tool instance
tool = ResearcherSearchTool()

# Search by researcher name
result = tool.run("Theresa Boyle")
print(result)
# Output:
# Researcher: Theresa Boyle
# Program: Pathology
# Department: Tumor Microenvironment and Metastasis
# Content: Focuses on interpreting molecular results...
# Profile: https://www.moffitt.org/research-science/researchers/theresa-boyle

# Search by research topic
result = tool.run("cancer genomics")
print(result)
# Output: [List of researchers working on cancer genomics]
```

### Input Schema

```python
class ResearcherSearchInput(BaseModel):
    """Input for the ResearcherSearch tool."""
    researcher_name: Optional[str] = Field(
        None,
        description="The name of the researcher to search for"
    )
    topic: Optional[str] = Field(
        None,
        description="The research topic to search for"
    )
```

## DepartmentFilter Tool

**Purpose**: Filters researchers by their academic department.

**Implementation**: `backend/app/services/tools.py` - `DepartmentFilterTool` class

### Logic
1. **Department Listing**:
   - Special handling for "list" query: returns all available departments

2. **Department Filtering**:
   - Uses metadata filtering in ChromaDB
   - Applies fuzzy matching for department names
   - Groups results by researcher to avoid duplicates

3. **Result Formatting**:
   - Returns department name
   - Lists researchers in the department
   - Includes program information for each researcher

### Example Usage

```python
from backend.app.services.tools import DepartmentFilterTool

# Create tool instance
tool = DepartmentFilterTool()

# List all departments
departments = tool.run("list")
print(departments)
# Output: [List of all departments at Moffitt]

# Filter researchers by department
immunology_researchers = tool.run("Immunology")
print(immunology_researchers)
# Output:
# Department: Immunology
# Researchers: 12
#
# Researchers in this department:
# - John Doe: Cancer Biology
# - Jane Smith: Immuno-Oncology
# [...]
```

## ProgramFilter Tool

**Purpose**: Filters researchers by their research program.

**Implementation**: `backend/app/services/tools.py` - `ProgramFilterTool` class

### Logic
1. **Program Listing**:
   - Special handling for "list" query: returns all available programs

2. **Program Filtering**:
   - Uses metadata filtering in ChromaDB
   - Handles partial program name matches
   - Groups results by researcher to avoid duplicates

3. **Result Formatting**:
   - Returns program name
   - Lists researchers in the program
   - Includes department information for each researcher

### Example Usage

```python
from backend.app.services.tools import ProgramFilterTool

# Create tool instance
tool = ProgramFilterTool()

# List all programs
programs = tool.run("list")
print(programs)
# Output: [List of all programs at Moffitt]

# Filter researchers by program
bioengineering_researchers = tool.run("BioEngineering")
print(bioengineering_researchers)
# Output:
# Program: BioEngineering
# Researchers: 8
#
# Researchers in this program:
# - Alice Johnson: Molecular Oncology
# - Bob Williams: Cancer Imaging
# [...]
```

## Hybrid Search Implementation

All tools utilize the hybrid search functionality, which combines vector-based semantic search with keyword-based text search:

```
Query
 │
 ├────────────────────┐
 │                    │
 ▼                    ▼
┌─────────────┐    ┌─────────────┐
│ Vector      │    │ Keyword     │
│ Search      │    │ Search      │
│             │    │             │
│ - Semantic  │    │ - Text      │
│   similarity│    │   matching  │
└──────┬──────┘    └──────┬──────┘
       │                  │
       │                  │
       ▼                  ▼
┌──────────────────────────────┐
│ Score Combination            │
│                              │
│ combined_score =             │
│   alpha * semantic_score +   │
│   (1-alpha) * keyword_score  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Result Ranking               │
│ - Sort by combined score     │
│ - Return top K results       │
└──────────────────────────────┘
```

### Hybrid Search Parameters

- `query`: The search query text
- `k`: Number of results to return
- `alpha`: Balance between semantic (1.0) and keyword (0.0) search
- `filter`: Optional metadata filters

### Code Example

```python
def hybrid_search(query: str, k: int = 4, alpha: float = 0.5,
                 filter: Optional[Dict[str, Any]] = None):
    """
    Perform a hybrid search combining vector similarity and keyword matching.

    Args:
        query: The search query
        k: Number of results to return
        alpha: Balance between semantic (1.0) and keyword (0.0) search
        filter: Optional metadata filter

    Returns:
        List of documents with combined ranking
    """
    # Get vector database
    db = get_or_create_vector_db()

    # Semantic search (vector-based)
    semantic_results = similarity_search_with_score(
        query=query,
        k=k*2,
        filter=filter,
        db=db
    )

    # Keyword search (text-based)
    keyword_results = keyword_search(
        query=query,
        texts=[doc.page_content for doc in db.get()],
        metadatas=[doc.metadata for doc in db.get()],
        k=k*2
    )

    # Combine results with weighted scoring
    combined_scores = {}

    for doc_id, semantic_score in semantic_results:
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                'document': doc_id,
                'semantic_score': semantic_score,
                'keyword_score': 0,
                'combined_score': 0
            }

    for doc_id, keyword_score in keyword_results:
        if doc_id in combined_scores:
            combined_scores[doc_id]['keyword_score'] = keyword_score
        else:
            combined_scores[doc_id] = {
                'document': doc_id,
                'semantic_score': 0,
                'keyword_score': keyword_score,
                'combined_score': 0
            }

    # Calculate combined scores
    for doc_id, scores in combined_scores.items():
        combined_scores[doc_id]['combined_score'] = (
            alpha * scores['semantic_score'] +
            (1 - alpha) * scores['keyword_score']
        )

    # Sort by combined score (descending)
    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )

    # Return top k results
    return [result['document'] for result in sorted_results[:k]]
```

## Tool Implementation Best Practices

When extending the tool system with new tools, follow these best practices:

1. **Input Validation**:
   - Use Pydantic models for input validation
   - Provide clear error messages for invalid inputs

2. **Error Handling**:
   - Catch and handle exceptions gracefully
   - Return helpful error messages to the agent

3. **Result Formatting**:
   - Format results consistently across tools
   - Include metadata and context for better agent understanding

4. **Async Support**:
   - Implement both sync (`_run`) and async (`_arun`) methods
   - Use proper async patterns for I/O-bound operations

5. **Rate Limiting**:
   - Implement rate limiting for external API calls
   - Add backoff and retry mechanisms for transient failures

6. **Documentation**:
   - Provide clear descriptions for the tool and its parameters
   - Document example inputs and outputs

## Example: Creating a New Tool

Here's how to create a new tool for the system:

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, List

# Input model for the tool
class PublicationSearchInput(BaseModel):
    """Input for the PublicationSearch tool."""
    topic: str = Field(
        description="The publication topic to search for"
    )
    year: Optional[int] = Field(
        None,
        description="Filter publications by year"
    )

# Tool implementation
class PublicationSearchTool(BaseTool):
    """
    Tool for searching publications by topic and year.
    """
    name: str = "PublicationSearch"
    description: str = "Search for publications by topic and year"
    args_schema: Type[BaseModel] = PublicationSearchInput

    def _run(self, topic: str, year: Optional[int] = None) -> str:
        """
        Run the publication search tool.

        Args:
            topic: The publication topic to search for
            year: Optional year filter

        Returns:
            str: Formatted search results
        """
        try:
            # Perform search using hybrid search
            filter = {"chunk_type": "publications"}
            if year:
                filter["year"] = year

            results = hybrid_search(
                query=topic,
                filter=filter,
                alpha=0.7,  # Favor semantic search for topics
                k=5
            )

            # Format results
            if not results:
                return f"No publications found on topic: {topic}"

            formatted_results = []
            for i, doc in enumerate(results):
                # Format publication information
                formatted_results.append(
                    f"Publication {i+1}:\n"
                    f"Title: {doc.metadata.get('title', 'Unknown')}\n"
                    f"Authors: {doc.metadata.get('authors', 'Unknown')}\n"
                    f"Year: {doc.metadata.get('year', 'Unknown')}\n"
                    f"Journal: {doc.metadata.get('journal', 'Unknown')}\n"
                    f"Researcher: {doc.metadata.get('researcher_name', 'Unknown')}\n"
                )

            return "\n\n".join(formatted_results)

        except Exception as e:
            return f"Error searching for publications: {str(e)}"

    async def _arun(self, topic: str, year: Optional[int] = None) -> str:
        """
        Run the publication search tool asynchronously.
        """
        # For now, just call the sync version
        return self._run(topic=topic, year=year)
```

To add this tool to the system, update the `get_tools()` function in `tools.py`:

```python
def get_tools() -> List[BaseTool]:
    """
    Get all tools for the agent.
    """
    return [
        ResearcherSearchTool(),
        DepartmentFilterTool(),
        ProgramFilterTool(),
        PublicationSearchTool()  # Add the new tool
    ]
```