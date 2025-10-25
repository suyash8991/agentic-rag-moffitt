# Agent Tools Logic

The Moffitt Agentic RAG system uses several specialized tools that implement different search capabilities. These tools are defined in the `src/moffitt_rag/tools/` directory.

## ResearcherSearchTool

**Purpose**: Searches for researchers by expertise, interests, or name.

**Implementation**: `src/moffitt_rag/tools/researcher_search.py`

**Logic**:
1. **Name Detection**: Determines if the query is likely a name search:
   - Checks for phrases like "who is", "find", "about"
   - Identifies capitalized words (proper names)
   - Matches against a list of known researchers

2. **Search Parameter Adjustment**:
   - For name searches, uses `alpha=0.3` (favors keyword matching)
   - For topic searches, uses `alpha=0.7` (favors semantic matching)

3. **Search Process**:
   - For known researchers: Performs direct metadata lookup
   - Otherwise: Uses hybrid search (combining vector and keyword search)
   - Fallback mechanisms for low-quality matches

4. **Result Formatting**:
   - Includes full content from each chunk (up to 1024 characters per chunk)
   - Returns all chunks without deduplication (up to 5 chunks by default)
   - Formats results with researcher name, program, chunk type, content, and profile link
   - Uses clear separators between chunks to improve readability

**Key Features**:
- Rate limiting to prevent excessive search attempts
- Special handling for known researchers
- Name extraction from URLs and text content
- Complete content inclusion instead of limited snippets
- Chunk type identification for better context

**Class Definition**:
```python
class ResearcherSearchTool(BaseTool):
    """
    Tool for searching researchers by their expertise, interests, or background.
    """

    name: str = "ResearcherSearch"
    description: str = "Search for researchers by their expertise, interests, or background"

    def _run(self, query: str) -> str:
        # Determine if this is a name search
        # Perform search with appropriate parameters
        # Format and return results
```

## DepartmentFilterTool

**Purpose**: Filters researchers by their academic department.

**Implementation**: `src/moffitt_rag/tools/department_filter.py`

**Logic**:
1. **Department Listing**: If query is "list", returns all available departments
2. **Department Filtering**:
   - Performs a metadata filter search in ChromaDB
   - Flexible matching for partial department names
3. **Result Aggregation**:
   - Groups results by researcher to avoid duplicates
   - Formats results with department context

**Key Features**:
- Department name normalization
- Fuzzy matching for department names
- Special handling for department listing

**Class Definition**:
```python
class DepartmentFilterTool(BaseTool):
    """
    Tool for filtering researchers by their department.
    """

    name: str = "DepartmentFilter"
    description: str = "Filter researchers by their department"

    def _run(self, department: str) -> str:
        # Check if request is to list departments
        # Filter researchers by department
        # Format and return results
```

## ProgramFilterTool

**Purpose**: Filters researchers by their research program.

**Implementation**: `src/moffitt_rag/tools/program_filter.py`

**Logic**:
1. **Program Listing**: If query is "list", returns all available programs
2. **Program Filtering**:
   - Performs a metadata filter search in ChromaDB
   - Handles both exact and partial program name matches
3. **Result Aggregation**:
   - Groups results by researcher to avoid duplicates
   - Formats results with program context

**Key Features**:
- Program name normalization
- Fuzzy matching for program names
- Special handling for program listing

**Class Definition**:
```python
class ProgramFilterTool(BaseTool):
    """
    Tool for filtering researchers by their research program.
    """

    name: str = "ProgramFilter"
    description: str = "Filter researchers by their research program"

    def _run(self, program: str) -> str:
        # Check if request is to list programs
        # Filter researchers by program
        # Format and return results
```

## InterestMatchTool

**Purpose**: Finds researchers with similar research interests.

**Implementation**: `src/moffitt_rag/tools/interest_match.py`

**Logic**:
1. **Interest Matching**:
   - For named researchers: Finds other researchers with similar interests
   - For topic queries: Finds researchers with matching interests
2. **Relevance Scoring**:
   - Ranks results by interest relevance
   - Extracts matching interests for context
3. **Result Formatting**:
   - Groups results by researcher
   - Includes relevant interests in output

**Key Features**:
- Semantic matching of research interests
- Two distinct operating modes (researcher-based vs topic-based)
- Context-aware result presentation

**Class Definition**:
```python
class InterestMatchTool(BaseTool):
    """
    Tool for finding researchers with similar research interests.
    """

    name: str = "InterestMatch"
    description: str = "Find researchers with similar research interests"

    def _run(self, query: str) -> str:
        # Determine query type (researcher name or topic)
        # Find matching interests or similar researchers
        # Format and return results
```

## CollaborationTool

**Purpose**: Discovers potential collaborations between researchers or departments.

**Implementation**: `src/moffitt_rag/tools/collaboration.py`

**Logic**:
1. **Collaboration Type Detection**:
   - Between departments (e.g., "between Biostatistics and Cancer Epidemiology")
   - For specific researchers (e.g., "for John Cleveland")
   - Between research areas (e.g., "between immunotherapy and genomics")
2. **Collaboration Scoring**:
   - Identifies common research interests
   - Considers department and program overlap
   - Evaluates publication and grant similarity
3. **Result Presentation**:
   - Organizes by collaboration strength
   - Explains potential collaboration areas

**Key Features**:
- Multi-faceted collaboration discovery
- Quantitative collaboration scoring
- Explanatory result formatting

**Class Definition**:
```python
class CollaborationTool(BaseTool):
    """
    Tool for discovering potential collaborations between researchers or departments.
    """

    name: str = "Collaboration"
    description: str = "Find potential collaborations between researchers or departments"

    def _run(self, query: str) -> str:
        # Parse the collaboration query type
        # Find relevant collaborations
        # Format and return results
```

## Tool Selection in the Agent

The agent selects tools based on the user query type. This selection is guided by instructions in the system prompt:

```
Follow these guidelines for tool selection:
1. For general researcher searches, use ResearcherSearch
2. To find researchers in a specific department, use DepartmentFilter
3. To find researchers in a specific program, use ProgramFilter
4. To find researchers with similar interests, use InterestMatch
5. To discover potential collaborations, use Collaboration
```

The agent analyzes the user query, determines the appropriate tool, and invokes it with relevant parameters. The tools then handle the specific search logic and return formatted results to the agent.

## Tool Interaction with Hybrid Search

Most tools utilize the hybrid search functionality implemented in `src/moffitt_rag/db/hybrid_search.py`, which combines vector-based and keyword-based search:

```python
def hybrid_search(query: str, k: int = 4, alpha: float = 0.5,
                 filter: Optional[Dict[str, Any]] = None,
                 min_score_threshold: float = 0.2):
    """
    Perform a hybrid search combining vector similarity and keyword matching.
    """
    # Perform semantic search
    semantic_results = similarity_search_with_score(query, k=k*2, filter=filter, db=db)

    # Perform keyword search
    keyword_results = keyword_search(query, texts, metadatas, k=k*2)

    # Combine results with weighted scoring
    # Return the top k documents
```

The tools can adjust the `alpha` parameter to control the balance between semantic similarity and keyword matching, allowing for optimized search results based on query type.

## Common Tool Features

All tools share several common features:

1. **Rate Limiting**: Prevents excessive search attempts
2. **Error Handling**: Graceful handling of search failures
3. **Result Formatting**: Consistent output format for agent consumption
4. **Metadata Filtering**: Ability to filter results by metadata fields
5. **Logging**: Detailed logging for debugging and monitoring

These shared features ensure consistent behavior and robust operation across all search tools.