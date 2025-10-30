# SOLID Principles & Best Practices Analysis

**Project**: Moffitt Agentic RAG System
**Date**: October 29, 2025
**Status**: Comprehensive Analysis Complete

---

## Executive Summary

This document provides a detailed analysis of the Moffitt Agentic RAG codebase against SOLID principles, design patterns, and best practices. The analysis identifies architectural issues, code smells, and optimization opportunities to improve maintainability, testability, and performance.

### Key Findings

**Critical Issues:**
- ✗ 7 major Single Responsibility Principle violations
- ✗ 4 Open/Closed Principle violations with hard-coded dependencies
- ✗ Multiple services with 200+ line functions (god methods)
- ✗ No dependency injection - tight coupling throughout
- ✗ Global state prevents isolated testing
- ✗ Direct file system dependencies - cannot mock

**Positive Aspects:**
- ✓ Performance instrumentation added recently
- ✓ Embedding function caching implemented
- ✓ LangSmith integration for observability
- ✓ Clear separation of API, services, and models

### Priority Action Items

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| **P0** | Extract repository layer from services | High | Medium |
| **P0** | Implement dependency injection | High | High |
| **P1** | Break down 200+ line functions | Medium | Medium |
| **P1** | Abstract LLM providers with strategy pattern | Medium | Low |
| **P2** | Remove global state | Medium | Medium |
| **P2** | Add caching layer for file operations | Low | Low |

---

## Table of Contents

1. [Single Responsibility Principle (SRP)](#1-single-responsibility-principle-srp)
2. [Open/Closed Principle (OCP)](#2-openclosed-principle-ocp)
3. [Liskov Substitution Principle (LSP)](#3-liskov-substitution-principle-lsp)
4. [Interface Segregation Principle (ISP)](#4-interface-segregation-principle-isp)
5. [Dependency Inversion Principle (DIP)](#5-dependency-inversion-principle-dip)
6. [Code Smells](#6-code-smells)
7. [Testability Issues](#7-testability-issues)
8. [Best Practice Violations](#8-best-practice-violations)
9. [Optimization Opportunities](#9-optimization-opportunities)
10. [Refactoring Roadmap](#10-refactoring-roadmap)

---

## 1. Single Responsibility Principle (SRP)

> "A class should have one, and only one, reason to change"

### Critical Violations

#### 1.1 `backend/app/services/researcher.py`

**Problem**: Multiple responsibilities mixed together

**Current Issues:**
- File I/O operations (lines 59-62, 131-142, 173-178, 209-213)
- Business logic (filtering, pagination)
- Data transformation (creating model instances)
- Directory scanning
- Summary parsing vs individual file parsing

**Example** (lines 25-111):
```python
def list_researchers(skip: int = 0, limit: int = 10, ...):
    # Direct file access
    data_dir = settings.PROCESSED_DATA_DIR
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    # Parsing
    for filename in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Filtering logic
    if department and data.get("department") != department:
        continue

    # Pagination
    researchers = researchers[skip : skip + limit]

    # Model creation
    return PaginatedResearcherResponse(...)
```

**Refactoring Plan:**

```python
# 1. Repository Layer (Data Access)
class IResearcherRepository(ABC):
    @abstractmethod
    def list(self, filters: Dict) -> List[Dict]: pass

    @abstractmethod
    def get_by_id(self, researcher_id: str) -> Optional[Dict]: pass

class FileResearcherRepository(IResearcherRepository):
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    def list(self, filters: Dict) -> List[Dict]:
        # Only file I/O here
        json_files = [f for f in self._data_dir.glob("*.json")]
        return [self._load_file(f) for f in json_files]

    def _load_file(self, path: Path) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

# 2. Service Layer (Business Logic)
class ResearcherService:
    def __init__(self, repository: IResearcherRepository):
        self._repository = repository

    def list_researchers(
        self,
        skip: int = 0,
        limit: int = 10,
        department: Optional[str] = None,
        program: Optional[str] = None
    ) -> PaginatedResearcherResponse:
        # Load data
        all_researchers = self._repository.list({})

        # Filter
        filtered = self._apply_filters(
            all_researchers, department, program
        )

        # Paginate
        paginated = self._paginate(filtered, skip, limit)

        # Transform to models
        summaries = [
            self._to_summary(data) for data in paginated
        ]

        return PaginatedResearcherResponse(
            researchers=summaries,
            total=len(filtered),
            skip=skip,
            limit=limit
        )

    def _apply_filters(self, data, department, program):
        # Filter logic only
        pass

    def _paginate(self, data, skip, limit):
        # Pagination logic only
        return data[skip:skip + limit]

    def _to_summary(self, data: Dict) -> ResearcherProfileSummary:
        # Transformation logic only
        return ResearcherProfileSummary(**data)

# 3. API Layer (already separate in researchers.py endpoint)
```

**Benefits:**
- Each class has one reason to change
- Can swap file storage for database without changing business logic
- Easy to test each layer independently
- Clear separation of concerns

---

#### 1.2 `backend/app/services/vector_db.py`

**Problem**: Single file handles 6 different concerns

**Current Responsibilities:**
1. Embedding model caching (lines 40-68)
2. Database loading (lines 71-112)
3. Search operations (lines 131-183)
4. Database rebuilding (lines 185-302)
5. Statistics tracking (lines 305-320)
6. Background task management (lines 216-300)

**Issues:**
- 320 lines doing everything
- Global state: `_cached_embedding_function`, `_db_stats`, `_active_tasks`
- `rebuild_vector_database()` is 117 lines
- Mixed concerns make testing difficult

**Refactoring Plan:**

```python
# 1. Embedding Cache Service
class EmbeddingCache:
    def __init__(self):
        self._cached_model: Optional[HuggingFaceEmbeddings] = None

    def get_or_create(self, model_name: str) -> HuggingFaceEmbeddings:
        if self._cached_model is None:
            self._cached_model = HuggingFaceEmbeddings(
                model_name=model_name
            )
        return self._cached_model

# 2. Vector Database Service (CRUD only)
class VectorDatabaseService:
    def __init__(
        self,
        embedding_cache: EmbeddingCache,
        config: Settings
    ):
        self._embedding_cache = embedding_cache
        self._config = config
        self._db: Optional[Chroma] = None

    def load(self) -> Optional[Chroma]:
        # Load database only
        pass

    def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        # Search only
        pass

# 3. Database Rebuild Service
class DatabaseRebuildService:
    def __init__(
        self,
        db_service: VectorDatabaseService,
        builder: VectorDatabaseBuilder,
        backup_service: DatabaseBackupService
    ):
        self._db = db_service
        self._builder = builder
        self._backup = backup_service

    async def rebuild(
        self,
        force: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> bool:
        # Orchestrate rebuild only
        if not force and self._db.exists():
            return False

        await self._backup.create()
        await self._builder.build(progress_callback)
        return True

# 4. Database Stats Service
class DatabaseStatsService:
    def __init__(self, db_service: VectorDatabaseService):
        self._db = db_service
        self._stats_cache: Dict[str, Any] = {}

    def get_stats(self) -> Dict[str, Any]:
        # Stats only
        pass

    def update_stats(self):
        # Update cache
        pass
```

**Benefits:**
- Each service has single, clear responsibility
- Can test each service independently
- Easy to understand and maintain
- Global state eliminated

---

#### 1.3 `backend/app/services/agent.py`

**Problem**: Massive `process_query()` function with multiple responsibilities

**Current Issues** (lines 105-330):
- Query processing (226 lines!)
- Status tracking
- Response parsing with complex regex
- Error handling
- User message generation
- LangSmith integration

**Example of the problem:**
```python
async def process_query(query_id, query, query_type, ...):
    # Status management
    _query_statuses[query_id] = {...}

    # Agent creation
    agent = create_researcher_agent(...)

    # Status updates
    _query_statuses[query_id]["progress"] = 0.2

    # LangSmith metadata
    metadata = create_run_metadata(...)

    # Agent invocation
    result = agent.invoke(...)

    # Complex regex parsing (lines 208-243)
    thoughts = re.findall(r'Thought: (.*?)(?=\nAction:|$)', ...)
    actions = re.findall(r'Action: (.*?)(?=\nAction Input:|$)', ...)

    # Error handling with user messages
    except Exception as e:
        error_msg = "I apologize for the technical difficulty..."
        return QueryResponse(answer=error_msg, ...)

    # More status updates
    _query_statuses[query_id]["status"] = "completed"
```

**Refactoring Plan:**

```python
# 1. Query State Manager
class QueryStateManager:
    def __init__(self):
        self._states: Dict[str, QueryState] = {}

    def create(self, query_id: str, query: str) -> QueryState:
        state = QueryState(
            query_id=query_id,
            query=query,
            status=QueryStatus.PENDING
        )
        self._states[query_id] = state
        return state

    def update_progress(self, query_id: str, progress: float):
        self._states[query_id].progress = progress

    def complete(self, query_id: str):
        self._states[query_id].status = QueryStatus.COMPLETED

# 2. Response Parser
class AgentResponseParser:
    def parse(self, raw_response: Dict) -> ParsedResponse:
        # Extract thoughts, actions, observations
        thoughts = self._extract_thoughts(raw_response)
        actions = self._extract_actions(raw_response)
        return ParsedResponse(thoughts=thoughts, actions=actions)

    def _extract_thoughts(self, response: Dict) -> List[str]:
        # Regex logic isolated here
        pass

# 3. Query Processor
class QueryProcessor:
    def __init__(
        self,
        agent_factory: AgentFactory,
        state_manager: QueryStateManager,
        response_parser: AgentResponseParser,
        error_formatter: ErrorMessageFormatter
    ):
        self._agent_factory = agent_factory
        self._state = state_manager
        self._parser = response_parser
        self._error_formatter = error_formatter

    async def process(
        self,
        query_id: str,
        query: str,
        query_type: str = "general"
    ) -> QueryResponse:
        # Create state
        state = self._state.create(query_id, query)

        try:
            # Create agent
            agent = self._agent_factory.create()

            # Update progress
            self._state.update_progress(query_id, 0.2)

            # Invoke
            raw_result = await agent.invoke(query)

            # Parse
            parsed = self._parser.parse(raw_result)

            # Complete
            self._state.complete(query_id)

            return QueryResponse(answer=parsed.answer)

        except Exception as e:
            error_msg = self._error_formatter.format(e)
            return QueryResponse(answer=error_msg, error=str(e))
```

**Benefits:**
- Each component has one responsibility
- 226-line function split into manageable pieces
- Easy to test each component
- Can swap implementations (e.g., different error formatters)

---

#### 1.4 `backend/app/services/vector_db_builder.py`

**Problem**: 517-line file doing everything for database building

**Current Responsibilities:**
1. File loading (lines 50-95)
2. Chunking strategy (lines 98-308) - 210 lines!
3. Database creation (lines 311-375)
4. Backup operations (lines 378-402)
5. Rebuild orchestration (lines 405-517)

**Refactoring Plan:**

```python
# 1. Profile Loader
class ResearcherProfileLoader:
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    def load_all(self) -> Iterator[Dict]:
        # Yield profiles one at a time (memory efficient)
        for file_path in self._data_dir.glob("*.json"):
            if file_path.name != "summary.json":
                yield self._load_file(file_path)

# 2. Chunking Strategy (Interface + Implementations)
class IChunkingStrategy(ABC):
    @abstractmethod
    def create_chunks(self, profile: Dict) -> List[ResearcherChunk]:
        pass

class StandardChunkingStrategy(IChunkingStrategy):
    def __init__(self, chunk_size: int = 1024):
        self._chunk_size = chunk_size

    def create_chunks(self, profile: Dict) -> List[ResearcherChunk]:
        chunks = []
        chunks.extend(self._create_core_chunk(profile))
        chunks.extend(self._create_interest_chunks(profile))
        chunks.extend(self._create_publication_chunks(profile))
        chunks.extend(self._create_grant_chunks(profile))
        return chunks

    def _create_core_chunk(self, profile: Dict) -> List[ResearcherChunk]:
        # Core chunking only
        pass

# 3. Vector Database Builder
class VectorDatabaseBuilder:
    def __init__(
        self,
        loader: ResearcherProfileLoader,
        chunking_strategy: IChunkingStrategy,
        embedding_function: HuggingFaceEmbeddings
    ):
        self._loader = loader
        self._chunking = chunking_strategy
        self._embedding = embedding_function

    def build(
        self,
        output_dir: Path,
        progress_callback: Optional[Callable] = None
    ):
        all_chunks = []

        for i, profile in enumerate(self._loader.load_all()):
            chunks = self._chunking.create_chunks(profile)
            all_chunks.extend(chunks)

            if progress_callback:
                progress_callback(i)

        # Create database
        db = Chroma.from_texts(
            texts=[c.text for c in all_chunks],
            embedding=self._embedding,
            persist_directory=str(output_dir)
        )

        return db

# 4. Backup Service
class DatabaseBackupService:
    def create_backup(self, source: Path) -> Optional[Path]:
        # Backup logic only
        pass

# 5. Rebuild Orchestrator
class DatabaseRebuildOrchestrator:
    def __init__(
        self,
        builder: VectorDatabaseBuilder,
        backup: DatabaseBackupService
    ):
        self._builder = builder
        self._backup = backup

    def rebuild(
        self,
        force: bool = False,
        backup: bool = True
    ) -> bool:
        # Orchestration only
        if backup:
            self._backup.create_backup(db_path)

        self._builder.build(db_path)
        return True
```

---

#### 1.5 `backend/app/services/tools.py`

**Problem**: `ResearcherSearchTool._run()` does too much (177 lines)

**Current Responsibilities:**
- Input parsing (lines 121-132)
- Name normalization (lines 154-171)
- Search orchestration (lines 186-198)
- Result formatting (lines 221-272)
- Error handling (lines 274-277)

**Refactoring Plan:**

```python
# 1. Tool Input Parser
class ToolInputParser:
    def parse_researcher_search(
        self,
        researcher_name: Optional[str],
        topic: Optional[str]
    ) -> SearchRequest:
        # Handle JSON input parsing
        if researcher_name and researcher_name.strip().startswith('{'):
            parsed = json.loads(researcher_name)
            researcher_name = parsed.get("researcher_name")
            topic = parsed.get("topic")

        return SearchRequest(
            researcher_name=researcher_name,
            topic=topic
        )

# 2. Search Result Formatter
class SearchResultFormatter:
    def format_documents(
        self,
        documents: List[Document]
    ) -> str:
        # Formatting logic only
        formatted = []
        for i, doc in enumerate(documents):
            formatted.append(self._format_single(i, doc))
        return "\n\n---\n\n".join(formatted)

    def _format_single(self, index: int, doc: Document) -> str:
        # Format single document
        pass

# 3. Simplified Tool
class ResearcherSearchTool(BaseTool):
    def __init__(
        self,
        input_parser: ToolInputParser,
        search_service: ISearchService,
        result_formatter: SearchResultFormatter
    ):
        super().__init__()
        self._parser = input_parser
        self._search = search_service
        self._formatter = result_formatter

    def _run(
        self,
        researcher_name: Optional[str] = None,
        topic: Optional[str] = None
    ) -> str:
        # Parse input
        request = self._parser.parse_researcher_search(
            researcher_name, topic
        )

        # Validate
        if not request.researcher_name and not request.topic:
            return "Error: Provide either researcher_name or topic"

        # Search
        results = self._search.search(request)

        # Format
        return self._formatter.format_documents(results)
```

---

### Summary: SRP Violations

| File | Lines | Issues | Priority |
|------|-------|--------|----------|
| `researcher.py` | 219 | File I/O + business logic mixed | P0 |
| `vector_db.py` | 320 | 6 different responsibilities | P0 |
| `agent.py` | 355 | 226-line function | P0 |
| `vector_db_builder.py` | 517 | 210-line chunking function | P1 |
| `tools.py` | 429 | 177-line tool function | P1 |

---

## 2. Open/Closed Principle (OCP)

> "Software entities should be open for extension, but closed for modification"

### Critical Violations

#### 2.1 Hard-coded LLM Provider Logic

**File**: `backend/app/services/llm.py` (lines 46-105)

**Problem**: Adding new LLM providers requires modifying existing code

```python
def get_llm_model(temperature: float = 0.7):
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY
        )

    elif provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set")
        return ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=temperature,
            groq_api_key=settings.GROQ_API_KEY
        )

    else:
        # Fallback logic
        pass
```

**Issue**: Each new provider requires:
1. Adding elif branch
2. Modifying existing function
3. Risk of breaking existing providers

**Solution**: Strategy Pattern

```python
# 1. Abstract Provider Interface
class ILLMProvider(ABC):
    @abstractmethod
    def create_model(
        self,
        model_name: str,
        temperature: float
    ) -> BaseLanguageModel:
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        pass

# 2. Concrete Providers
class OpenAIProvider(ILLMProvider):
    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model

    def validate_config(self) -> bool:
        return bool(self._api_key)

    def create_model(self, model_name: str, temperature: float):
        if not self.validate_config():
            raise ValueError("OpenAI API key not configured")

        return ChatOpenAI(
            model=model_name or self._model,
            temperature=temperature,
            openai_api_key=self._api_key
        )

class GroqProvider(ILLMProvider):
    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model

    def validate_config(self) -> bool:
        return bool(self._api_key)

    def create_model(self, model_name: str, temperature: float):
        if not self.validate_config():
            raise ValueError("Groq API key not configured")

        return ChatGroq(
            model=model_name or self._model,
            temperature=temperature,
            groq_api_key=self._api_key
        )

# 3. Provider Factory (with Registry Pattern)
class LLMProviderFactory:
    _providers: Dict[str, ILLMProvider] = {}

    @classmethod
    def register(cls, name: str, provider: ILLMProvider):
        """Register a new provider - OPEN FOR EXTENSION"""
        cls._providers[name] = provider

    @classmethod
    def get_provider(cls, name: str) -> ILLMProvider:
        """Get provider - CLOSED FOR MODIFICATION"""
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")
        return cls._providers[name]

    @classmethod
    def create_model(
        cls,
        provider_name: str,
        temperature: float = 0.7
    ) -> BaseLanguageModel:
        provider = cls.get_provider(provider_name)
        return provider.create_model(None, temperature)

# 4. Initialization (in config or startup)
def initialize_providers(config: Settings):
    LLMProviderFactory.register(
        "openai",
        OpenAIProvider(config.OPENAI_API_KEY, config.OPENAI_MODEL)
    )
    LLMProviderFactory.register(
        "groq",
        GroqProvider(config.GROQ_API_KEY, config.GROQ_MODEL)
    )

    # NEW PROVIDER - NO CODE MODIFICATION NEEDED
    LLMProviderFactory.register(
        "anthropic",
        AnthropicProvider(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
    )

# 5. Usage
def get_llm_model(temperature: float = 0.7):
    return LLMProviderFactory.create_model(
        settings.LLM_PROVIDER,
        temperature
    )
```

**Benefits:**
- ✓ Add new providers without modifying existing code
- ✓ Easy to test each provider independently
- ✓ Clear extension point via `register()`
- ✓ Follows OCP perfectly

---

#### 2.2 Hard-coded Chunking Strategy

**File**: `backend/app/services/vector_db_builder.py` (lines 98-308)

**Problem**: 210-line function with hard-coded chunking logic

**Current Code** (simplified):
```python
def create_researcher_chunks(profile: Dict, chunk_size: int = 1024):
    chunks = []

    # Hard-coded chunking logic
    # 1. Core chunk
    core_text = f"Name: {profile['name']}\n..."
    chunks.append(ResearcherChunk(text=core_text, type="core"))

    # 2. Interests chunk
    if profile.get("interests"):
        interests_text = "Interests:\n" + ...
        chunks.append(ResearcherChunk(text=interests_text, type="interests"))

    # 3. Publications chunks (complex logic)
    if profile.get("publications"):
        # ... 50 lines of publication chunking
        pass

    # 4. Grants chunks
    if profile.get("grants"):
        # ... 40 lines of grant chunking
        pass

    return chunks
```

**Issue**: Cannot easily:
- Change chunking strategy
- A/B test different approaches
- Support domain-specific chunking

**Solution**: Strategy Pattern

```python
# 1. Chunking Strategy Interface
class IChunkingStrategy(ABC):
    @abstractmethod
    def create_chunks(
        self,
        profile: Dict,
        config: ChunkingConfig
    ) -> List[ResearcherChunk]:
        pass

# 2. Configuration
@dataclass
class ChunkingConfig:
    chunk_size: int = 1024
    include_publications: bool = True
    include_grants: bool = True
    max_publications_per_chunk: int = 5

# 3. Concrete Strategies
class StandardChunkingStrategy(IChunkingStrategy):
    """Current implementation"""

    def create_chunks(self, profile, config):
        chunks = []
        chunks.extend(self._create_core(profile))
        chunks.extend(self._create_interests(profile))

        if config.include_publications:
            chunks.extend(self._create_publications(profile, config))

        if config.include_grants:
            chunks.extend(self._create_grants(profile, config))

        return chunks

    def _create_core(self, profile):
        # Core logic
        pass

class PublicationFocusedStrategy(IChunkingStrategy):
    """Emphasize publications for researchers with many papers"""

    def create_chunks(self, profile, config):
        chunks = []

        # Different implementation
        if len(profile.get("publications", [])) > 50:
            # Create more granular publication chunks
            chunks.extend(self._create_detailed_pubs(profile))
        else:
            # Standard approach
            chunks.extend(self._create_standard_pubs(profile))

        return chunks

class GrantFocusedStrategy(IChunkingStrategy):
    """Emphasize grants for PIs"""

    def create_chunks(self, profile, config):
        # Different implementation focusing on grants
        pass

# 4. Strategy Factory
class ChunkingStrategyFactory:
    _strategies = {
        "standard": StandardChunkingStrategy(),
        "publication_focused": PublicationFocusedStrategy(),
        "grant_focused": GrantFocusedStrategy()
    }

    @classmethod
    def register_strategy(cls, name: str, strategy: IChunkingStrategy):
        """OPEN FOR EXTENSION"""
        cls._strategies[name] = strategy

    @classmethod
    def get_strategy(cls, name: str) -> IChunkingStrategy:
        """CLOSED FOR MODIFICATION"""
        return cls._strategies.get(name, cls._strategies["standard"])

# 5. Usage in Builder
class VectorDatabaseBuilder:
    def __init__(
        self,
        chunking_strategy: str = "standard",
        config: Optional[ChunkingConfig] = None
    ):
        self._strategy = ChunkingStrategyFactory.get_strategy(
            chunking_strategy
        )
        self._config = config or ChunkingConfig()

    def build(self, profiles: List[Dict]):
        all_chunks = []
        for profile in profiles:
            chunks = self._strategy.create_chunks(profile, self._config)
            all_chunks.extend(chunks)
        return all_chunks
```

**Benefits:**
- ✓ Easy to add new strategies without modifying existing code
- ✓ Can A/B test different chunking approaches
- ✓ Domain-specific strategies for different researcher types
- ✓ Configuration-driven behavior

---

#### 2.3 Hard-coded Search Algorithm

**File**: `backend/app/services/hybrid_search.py` (lines 86-261)

**Problem**: Monolithic hybrid search with hard-coded scoring

**Solution**: Strategy + Composite Pattern

```python
# 1. Search Strategy Interface
class ISearchStrategy(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        k: int,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        pass

# 2. Concrete Strategies
class SemanticSearchStrategy(ISearchStrategy):
    def __init__(self, vector_db: IVectorStore):
        self._vector_db = vector_db

    def search(self, query, k, filter):
        return self._vector_db.similarity_search(query, k, filter)

class KeywordSearchStrategy(ISearchStrategy):
    def __init__(self, vector_db: IVectorStore):
        self._vector_db = vector_db

    def search(self, query, k, filter):
        # Keyword-based search
        return self._keyword_search(query, k, filter)

# 3. Composite Strategy (Hybrid)
class HybridSearchStrategy(ISearchStrategy):
    def __init__(
        self,
        semantic: ISearchStrategy,
        keyword: ISearchStrategy,
        alpha: float = 0.5
    ):
        self._semantic = semantic
        self._keyword = keyword
        self._alpha = alpha

    def search(self, query, k, filter):
        # Get results from both
        semantic_results = self._semantic.search(query, k*2, filter)
        keyword_results = self._keyword.search(query, k*2, filter)

        # Combine with alpha weighting
        combined = self._combine_results(
            semantic_results,
            keyword_results,
            self._alpha
        )

        return combined[:k]

    def _combine_results(self, semantic, keyword, alpha):
        # Combining logic
        pass

# 4. Usage
def create_search_strategy(strategy_type: str) -> ISearchStrategy:
    if strategy_type == "semantic":
        return SemanticSearchStrategy(vector_db)
    elif strategy_type == "keyword":
        return KeywordSearchStrategy(vector_db)
    elif strategy_type == "hybrid":
        return HybridSearchStrategy(
            SemanticSearchStrategy(vector_db),
            KeywordSearchStrategy(vector_db),
            alpha=0.5
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_type}")
```

---

#### 2.4 Hard-coded Tool List

**File**: `backend/app/services/tools.py` (lines 417-428)

**Problem**: Tools are hard-coded in a function

```python
def get_tools():
    """Get all the tools for the agent."""
    return [
        ResearcherSearchTool(),
        DepartmentFilterTool(),
        ProgramFilterTool(),
    ]
```

**Issue**: Cannot dynamically add tools

**Solution**: Registry Pattern

```python
# 1. Tool Registry
class ToolRegistry:
    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool_class: Type[BaseTool]) -> Type[BaseTool]:
        """Decorator to register tools - OPEN FOR EXTENSION"""
        tool = tool_class()
        cls._tools[tool.name] = tool
        return tool_class

    @classmethod
    def get_tools(cls, categories: Optional[List[str]] = None) -> List[BaseTool]:
        """Get registered tools - CLOSED FOR MODIFICATION"""
        if categories:
            return [
                t for t in cls._tools.values()
                if hasattr(t, 'category') and t.category in categories
            ]
        return list(cls._tools.values())

    @classmethod
    def get_tool(cls, name: str) -> Optional[BaseTool]:
        return cls._tools.get(name)

# 2. Register tools with decorator
@ToolRegistry.register
class ResearcherSearchTool(BaseTool):
    name = "ResearcherSearch"
    category = "search"
    # ... implementation

@ToolRegistry.register
class DepartmentFilterTool(BaseTool):
    name = "DepartmentFilter"
    category = "filter"
    # ... implementation

@ToolRegistry.register
class ProgramFilterTool(BaseTool):
    name = "ProgramFilter"
    category = "filter"
    # ... implementation

# 3. NEW TOOL - NO MODIFICATION NEEDED
@ToolRegistry.register
class PublicationSearchTool(BaseTool):
    name = "PublicationSearch"
    category = "search"
    # ... implementation

# 4. Usage
def get_tools(categories: Optional[List[str]] = None):
    return ToolRegistry.get_tools(categories)

# Get all tools
all_tools = get_tools()

# Get only search tools
search_tools = get_tools(categories=["search"])
```

**Benefits:**
- ✓ Add tools without modifying core code
- ✓ Can filter tools by category
- ✓ Discoverable via decorator
- ✓ Easy to enable/disable tools

---

### Summary: OCP Violations

| Violation | File | Solution | Priority |
|-----------|------|----------|----------|
| Hard-coded LLM providers | `llm.py` | Strategy + Registry | P1 |
| Hard-coded chunking | `vector_db_builder.py` | Strategy pattern | P1 |
| Hard-coded search | `hybrid_search.py` | Strategy + Composite | P2 |
| Hard-coded tool list | `tools.py` | Registry pattern | P2 |

---

## 3. Liskov Substitution Principle (LSP)

> "Objects should be replaceable with instances of their subtypes without altering correctness"

### Violations

#### 3.1 Fake Async Methods in Tools

**Files**: `backend/app/services/tools.py`

**Problem**: `_arun()` methods just call sync `_run()`, violating LSP

**Examples:**

**ResearcherSearchTool** (lines 279-292):
```python
def _arun(
    self,
    researcher_name: Optional[str] = None,
    topic: Optional[str] = None
) -> str:
    """
    Async version of the tool (currently just calls sync version).
    """
    # For now, we'll just call the synchronous version
    return self._run(researcher_name=researcher_name, topic=topic)
```

**Same Issue In:**
- `DepartmentFilterTool._arun()` (lines 342-353)
- `ProgramFilterTool._arun()` (lines 403-414)

**Why This Violates LSP:**
1. `BaseTool` expects `_arun()` to be truly async
2. Calling sync code in async method blocks event loop
3. Cannot substitute with other async tools safely
4. Misleading - appears async but isn't

**Expected Behavior:**
```python
# What LSP expects
async def _arun(self, ...):
    result = await some_async_operation()
    return result
```

**Solution: Proper Async Implementation**

```python
class ResearcherSearchTool(BaseTool):
    async def _arun(
        self,
        researcher_name: Optional[str] = None,
        topic: Optional[str] = None
    ) -> str:
        """Truly async implementation"""
        # Use async file I/O
        profiles = await self._load_profiles_async()

        # Use async search
        results = await self._search_async(
            researcher_name, topic
        )

        return self._format_results(results)

    async def _load_profiles_async(self):
        # Use aiofiles for async file reading
        import aiofiles
        async with aiofiles.open(path, 'r') as f:
            content = await f.read()
        return json.loads(content)

    async def _search_async(self, name, topic):
        # Use asyncio for concurrent operations
        tasks = [
            self._semantic_search(name or topic),
            self._keyword_search(name or topic)
        ]
        results = await asyncio.gather(*tasks)
        return self._merge_results(results)
```

**Alternative: If Sync is Necessary**

```python
class ResearcherSearchTool(BaseTool):
    async def _arun(
        self,
        researcher_name: Optional[str] = None,
        topic: Optional[str] = None
    ) -> str:
        """Run sync code in thread pool to not block event loop"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # Use default executor
            self._run,
            researcher_name,
            topic
        )
```

---

#### 3.2 LimitedCallAgentExecutor

**File**: `backend/app/services/limited_call.py` (lines 18-234)

**Problem**: Not a proper subclass, violates LSP

**Current Implementation:**
```python
class LimitedCallAgentExecutor:
    """Wrapper for LangChain agent executor"""

    def __init__(self, agent_executor: AgentExecutor, max_calls: int = 6):
        self.agent_executor = agent_executor  # Wraps instead of extends
        self.max_calls = max_calls

    def invoke(self, input_data: Dict, config: Optional[Dict] = None):
        # Custom invoke implementation
        return self.agent_executor.invoke(input_data, config=config)
```

**Issues:**
1. Not a subclass of `AgentExecutor` but acts like one
2. Cannot substitute for `AgentExecutor` everywhere
3. Only implements `invoke()`, missing other methods
4. Misleading name (ends with "Executor" but isn't one)

**Solutions:**

**Option 1: Proper Subclass**
```python
class LimitedCallAgentExecutor(AgentExecutor):
    """Properly extends AgentExecutor"""

    def __init__(
        self,
        agent,
        tools,
        max_calls: int = 6,
        **kwargs
    ):
        super().__init__(agent=agent, tools=tools, **kwargs)
        self._max_calls = max_calls
        self._call_count = 0

    def _take_next_step(self, *args, **kwargs):
        self._call_count += 1
        if self._call_count > self._max_calls:
            raise MaxCallsExceededError(
                f"Exceeded {self._max_calls} LLM calls"
            )
        return super()._take_next_step(*args, **kwargs)
```

**Option 2: Clear Wrapper/Decorator**
```python
class CallLimitedAgentWrapper:
    """
    Wrapper that limits LLM calls for any AgentExecutor.
    Uses Decorator pattern explicitly.
    """

    def __init__(self, executor: AgentExecutor, max_calls: int = 6):
        self._executor = executor
        self._max_calls = max_calls

    def invoke(self, input_data: Dict, config: Optional[Dict] = None):
        # Delegate to wrapped executor
        return self._executor.invoke(input_data, config=config)

    # Implement other AgentExecutor methods
    def __getattr__(self, name):
        """Delegate all other methods to wrapped executor"""
        return getattr(self._executor, name)
```

**Option 3: Use Callbacks (LangChain Way)**
```python
class CallLimitCallback(BaseCallbackHandler):
    """Use LangChain's callback system instead"""

    def __init__(self, max_calls: int = 6):
        self.max_calls = max_calls
        self.call_count = 0

    def on_llm_start(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count > self.max_calls:
            raise MaxCallsExceededError()

# Usage
callbacks = [CallLimitCallback(max_calls=6)]
executor.invoke({"input": query}, config={"callbacks": callbacks})
```

---

### Summary: LSP Violations

| Violation | File | Impact | Priority |
|-----------|------|--------|----------|
| Fake async methods | `tools.py` | Blocks event loop | P1 |
| Non-substitutable wrapper | `limited_call.py` | Confusing API | P2 |

---

## 4. Interface Segregation Principle (ISP)

> "Clients should not be forced to depend on interfaces they don't use"

### Violations

#### 4.1 Fat Model Classes

**File**: `backend/app/models/researcher.py`

**Problem**: Monolithic model with all data

```python
class ResearcherProfileDetail(ResearcherProfileSummary):
    """Full detail - 15+ fields"""
    publications: List[Publication] = []
    grants: List[Grant] = []
    education: List[Education] = []
    contact: Optional[Contact] = None
    awards: List[str] = []
    clinical_specialties: List[str] = []
    board_certifications: List[str] = []
    professional_affiliations: List[str] = []
    # ... more fields
```

**Issues:**
- Clients needing basic info must handle all fields
- Over-fetching data
- Cannot enforce minimal dependencies

**Solution: Role-Based Interfaces (Protocols)**

```python
# 1. Define focused protocols
class IResearcherBasicInfo(Protocol):
    """Minimal info for listings"""
    researcher_id: str
    researcher_name: str
    title: str
    program: str

class IResearcherPublications(Protocol):
    """Just publications"""
    publications: List[Publication]

class IResearcherGrants(Protocol):
    """Just grants"""
    grants: List[Grant]

class IResearcherContact(Protocol):
    """Contact information"""
    contact: Optional[Contact]

# 2. Clients depend on specific protocols
class ResearcherListView:
    """Only needs basic info"""
    def render(self, researcher: IResearcherBasicInfo):
        return f"{researcher.researcher_name} - {researcher.title}"

class PublicationView:
    """Only needs publications"""
    def render(self, researcher: IResearcherPublications):
        return [p.title for p in researcher.publications]

# 3. Full model implements all protocols
class ResearcherProfile(
    IResearcherBasicInfo,
    IResearcherPublications,
    IResearcherGrants,
    IResearcherContact
):
    """Complete implementation"""
    # All fields
    pass
```

---

#### 4.2 Fat Service Modules

**File**: `backend/app/services/vector_db.py`

**Problem**: Single module provides unrelated operations

**Current Interface:**
```python
# Clients importing for search also get:
from app.services.vector_db import (
    get_embedding_function,  # Might not need
    load_vector_db,          # Might not need
    similarity_search,       # This is what they want
    rebuild_vector_database, # Don't need
    get_database_stats,      # Don't need
)
```

**Solution: Split into Focused Modules**

```python
# 1. vector_db/search.py
def similarity_search(query, k, filter):
    """Search operations only"""
    pass

# 2. vector_db/management.py
def load_vector_db():
    """Database management"""
    pass

def get_database_stats():
    """Statistics"""
    pass

# 3. vector_db/rebuild.py
def rebuild_vector_database():
    """Rebuild operations"""
    pass

# 4. vector_db/embeddings.py
def get_embedding_function():
    """Embedding management"""
    pass

# 5. Usage - clients import only what they need
from app.services.vector_db.search import similarity_search
# Don't need to import rebuild, stats, etc.
```

---

### Summary: ISP Violations

| Violation | File | Solution | Priority |
|-----------|------|----------|----------|
| Fat model classes | `researcher.py` | Protocol-based | P2 |
| Fat service modules | `vector_db.py` | Split modules | P2 |

---

## 5. Dependency Inversion Principle (DIP)

> "Depend on abstractions, not concretions"

### Critical Violations

#### 5.1 Direct File System Dependencies

**Files**: `researcher.py`, `vector_db_builder.py`

**Problem**: Direct file system access throughout

**Example** (researcher.py lines 44-62):
```python
def list_researchers(skip, limit, department, program):
    data_dir = settings.PROCESSED_DATA_DIR
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    for filename in json_files:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
```

**Issues:**
- Cannot swap storage (database, S3, cache)
- Cannot mock for testing
- Hard-coded to file system

**Solution: Repository Pattern**

```python
# 1. Abstract Repository Interface
class IResearcherRepository(ABC):
    @abstractmethod
    def list(
        self,
        skip: int = 0,
        limit: int = 10,
        filters: Optional[Dict] = None
    ) -> Tuple[List[Dict], int]:
        """Returns (data, total_count)"""
        pass

    @abstractmethod
    def get_by_id(self, researcher_id: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def search(self, query: str) -> List[Dict]:
        pass

# 2. File System Implementation
class FileSystemResearcherRepository(IResearcherRepository):
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._cache: Optional[List[Dict]] = None

    def list(self, skip, limit, filters):
        # Load from files
        all_data = self._load_all()

        # Apply filters
        filtered = self._apply_filters(all_data, filters)

        # Paginate
        total = len(filtered)
        paginated = filtered[skip:skip + limit]

        return paginated, total

    def _load_all(self) -> List[Dict]:
        if self._cache is None:
            self._cache = []
            for file_path in self._data_dir.glob("*.json"):
                if file_path.name != "summary.json":
                    with open(file_path, 'r') as f:
                        self._cache.append(json.load(f))
        return self._cache

    def get_by_id(self, researcher_id: str) -> Optional[Dict]:
        file_path = self._data_dir / f"{researcher_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

# 3. Database Implementation (future)
class DatabaseResearcherRepository(IResearcherRepository):
    def __init__(self, db_connection):
        self._db = db_connection

    def list(self, skip, limit, filters):
        # SQL query
        query = "SELECT * FROM researchers"
        # Add filters, pagination
        return self._db.execute(query).fetchall()

    def get_by_id(self, researcher_id: str):
        return self._db.execute(
            "SELECT * FROM researchers WHERE id = ?",
            (researcher_id,)
        ).fetchone()

# 4. S3 Implementation (future)
class S3ResearcherRepository(IResearcherRepository):
    def __init__(self, bucket_name: str, s3_client):
        self._bucket = bucket_name
        self._s3 = s3_client

    def list(self, skip, limit, filters):
        # Load from S3
        objects = self._s3.list_objects_v2(Bucket=self._bucket)
        # ...

# 5. Service Layer (depends on abstraction)
class ResearcherService:
    def __init__(self, repository: IResearcherRepository):
        self._repository = repository  # Depends on interface!

    def list_researchers(self, skip, limit, filters):
        data, total = self._repository.list(skip, limit, filters)
        # Transform to models
        return self._transform(data, total)

# 6. Dependency Injection in FastAPI
def get_researcher_repository() -> IResearcherRepository:
    # Can swap implementations here
    if settings.STORAGE_TYPE == "filesystem":
        return FileSystemResearcherRepository(
            Path(settings.PROCESSED_DATA_DIR)
        )
    elif settings.STORAGE_TYPE == "database":
        return DatabaseResearcherRepository(get_db_connection())
    elif settings.STORAGE_TYPE == "s3":
        return S3ResearcherRepository(
            settings.S3_BUCKET,
            boto3.client('s3')
        )

def get_researcher_service(
    repo: IResearcherRepository = Depends(get_researcher_repository)
) -> ResearcherService:
    return ResearcherService(repo)

# 7. Endpoint (depends on abstraction)
@router.get("/researchers")
async def list_researchers(
    service: ResearcherService = Depends(get_researcher_service),
    skip: int = 0,
    limit: int = 10
):
    return service.list_researchers(skip, limit, {})
```

**Benefits:**
- ✓ Can swap storage without changing service
- ✓ Easy to test with mock repository
- ✓ Can add caching layer
- ✓ Clear abstraction boundary

---

#### 5.2 Direct LangChain Dependencies

**File**: `backend/app/services/agent.py`

**Problem**: Tight coupling to LangChain

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

def create_researcher_agent(temperature, max_llm_calls):
    llm = get_llm_model(temperature=temperature)
    tools = get_tools()
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor
```

**Issues:**
- Cannot swap to different agent framework
- Hard to test
- Tightly coupled to LangChain API

**Solution: Abstract Agent Interface**

```python
# 1. Agent Framework Interface
class IAgentFramework(ABC):
    @abstractmethod
    def create_agent(
        self,
        llm_config: Dict,
        tools: List,
        prompt: str
    ) -> 'IAgent':
        pass

class IAgent(ABC):
    @abstractmethod
    async def invoke(
        self,
        input: str,
        config: Optional[Dict] = None
    ) -> str:
        pass

# 2. LangChain Implementation
class LangChainFramework(IAgentFramework):
    def create_agent(self, llm_config, tools, prompt):
        from langchain.agents import create_react_agent, AgentExecutor

        llm = self._create_llm(llm_config)
        lc_tools = self._convert_tools(tools)
        agent = create_react_agent(
            llm=llm,
            tools=lc_tools,
            prompt=PromptTemplate.from_template(prompt)
        )
        executor = AgentExecutor(agent=agent, tools=lc_tools)
        return LangChainAgent(executor)

class LangChainAgent(IAgent):
    def __init__(self, executor):
        self._executor = executor

    async def invoke(self, input, config):
        result = self._executor.invoke({"input": input}, config=config)
        return result["output"]

# 3. Alternative Implementation (LlamaIndex, etc.)
class LlamaIndexFramework(IAgentFramework):
    def create_agent(self, llm_config, tools, prompt):
        # Different implementation
        pass

# 4. Service depends on abstraction
class AgentService:
    def __init__(
        self,
        framework: IAgentFramework,
        llm_config: Dict,
        tools: List,
        prompt: str
    ):
        self._agent = framework.create_agent(llm_config, tools, prompt)

    async def process_query(self, query: str) -> str:
        return await self._agent.invoke(query)

# 5. Dependency Injection
def get_agent_framework() -> IAgentFramework:
    framework_type = settings.AGENT_FRAMEWORK  # "langchain" or "llamaindex"

    if framework_type == "langchain":
        return LangChainFramework()
    elif framework_type == "llamaindex":
        return LlamaIndexFramework()
    else:
        raise ValueError(f"Unknown framework: {framework_type}")

def get_agent_service(
    framework: IAgentFramework = Depends(get_agent_framework)
) -> AgentService:
    return AgentService(
        framework=framework,
        llm_config=get_llm_config(),
        tools=get_tools(),
        prompt=get_prompt()
    )
```

---

#### 5.3 Direct ChromaDB Dependencies

**File**: `backend/app/services/vector_db.py`

**Problem**: Tight coupling to ChromaDB

```python
from langchain_chroma import Chroma

def load_vector_db():
    db = Chroma(
        persist_directory=settings.VECTOR_DB_DIR,
        embedding_function=embedding_function,
        collection_name=settings.COLLECTION_NAME,
    )
    return db
```

**Solution: Abstract Vector Store**

```python
# 1. Vector Store Interface
class IVectorStore(ABC):
    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        pass

    @abstractmethod
    def add_documents(
        self,
        documents: List[Document]
    ) -> List[str]:
        pass

    @abstractmethod
    def delete_collection(self):
        pass

# 2. ChromaDB Implementation
class ChromaVectorStore(IVectorStore):
    def __init__(
        self,
        persist_directory: Path,
        embedding_function,
        collection_name: str
    ):
        self._db = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
            collection_name=collection_name
        )

    def similarity_search(self, query, k, filter):
        return self._db.similarity_search(query, k=k, filter=filter)

    def add_documents(self, documents):
        return self._db.add_documents(documents)

# 3. Pinecone Implementation (alternative)
class PineconeVectorStore(IVectorStore):
    def __init__(self, index_name: str, embedding_function):
        import pinecone
        self._index = pinecone.Index(index_name)
        self._embedding = embedding_function

    def similarity_search(self, query, k, filter):
        # Pinecone API
        query_embedding = self._embedding.embed_query(query)
        results = self._index.query(
            vector=query_embedding,
            top_k=k,
            filter=filter
        )
        return self._convert_results(results)

# 4. Weaviate Implementation
class WeaviateVectorStore(IVectorStore):
    # Different implementation
    pass

# 5. Service depends on abstraction
class VectorSearchService:
    def __init__(self, vector_store: IVectorStore):
        self._store = vector_store

    def search(self, query: str, k: int = 5) -> List[Document]:
        return self._store.similarity_search(query, k)

# 6. Factory
def create_vector_store(
    store_type: str,
    config: Dict
) -> IVectorStore:
    if store_type == "chroma":
        return ChromaVectorStore(
            persist_directory=Path(config["persist_dir"]),
            embedding_function=config["embedding"],
            collection_name=config["collection"]
        )
    elif store_type == "pinecone":
        return PineconeVectorStore(
            index_name=config["index"],
            embedding_function=config["embedding"]
        )
    elif store_type == "weaviate":
        return WeaviateVectorStore(...)
```

---

#### 5.4 Direct Settings Singleton Dependency

**Problem**: Global `settings` imported everywhere

```python
from ..core.config import settings

def list_researchers():
    data_dir = settings.PROCESSED_DATA_DIR  # Direct dependency
```

**Issues:**
- Cannot test with different configurations
- Global mutable state
- Hidden dependency

**Solution: Inject Configuration**

```python
# 1. Pass config explicitly
class ResearcherService:
    def __init__(self, config: Settings):
        self._config = config

    def list_researchers(self):
        data_dir = self._config.PROCESSED_DATA_DIR

# 2. Or use dependency injection
def get_config() -> Settings:
    return settings

@router.get("/researchers")
async def list_researchers(
    config: Settings = Depends(get_config)
):
    data_dir = config.PROCESSED_DATA_DIR
```

---

#### 5.5 Global State Dependencies

**Problem**: Global variables throughout

**Examples:**
```python
# tools.py line 27
_name_normalizer = NameNormalizationService()

# vector_db.py lines 30-42
_db_stats = {...}
_active_tasks: Dict[str, Dict[str, Any]] = {}
_cached_embedding_function = None

# agent.py lines 33-35
_query_statuses: Dict[str, Dict[str, Any]] = {}
```

**Solution: Dependency Injection**

```python
# Before (global state)
_name_normalizer = NameNormalizationService()

class ResearcherSearchTool:
    def _run(self, researcher_name):
        normalized = _name_normalizer.normalize(researcher_name)

# After (dependency injection)
class ResearcherSearchTool:
    def __init__(self, normalizer: INameNormalizer):
        self._normalizer = normalizer

    def _run(self, researcher_name):
        normalized = self._normalizer.normalize(researcher_name)

# Injection in FastAPI
def get_name_normalizer() -> INameNormalizer:
    return NameNormalizationService()

def get_researcher_tool(
    normalizer: INameNormalizer = Depends(get_name_normalizer)
) -> ResearcherSearchTool:
    return ResearcherSearchTool(normalizer)
```

---

### Summary: DIP Violations

| Violation | Files | Priority | Effort |
|-----------|-------|----------|--------|
| Direct file system | `researcher.py`, `vector_db_builder.py` | P0 | High |
| Direct LangChain | `agent.py` | P1 | Medium |
| Direct ChromaDB | `vector_db.py` | P1 | Medium |
| Global settings | Multiple | P2 | Low |
| Global state | Multiple | P1 | Medium |

---

## 6. Code Smells

### 6.1 God Classes

**Definition**: Classes that know or do too much

| File | Lines | Responsibilities | Should Be |
|------|-------|------------------|-----------|
| `vector_db.py` | 320 | 6 concerns | 6 classes |
| `vector_db_builder.py` | 517 | 5 concerns | 5 classes |
| `agent.py` | 355 | 4 concerns | 4 classes |

---

### 6.2 Long Methods

**Definition**: Methods over 50-100 lines are hard to understand and test

| Method | Lines | File | Priority |
|--------|-------|------|----------|
| `process_query()` | 226 | `agent.py` | P0 |
| `create_researcher_chunks()` | 210 | `vector_db_builder.py` | P0 |
| `hybrid_search()` | 175 | `hybrid_search.py` | P1 |
| `ResearcherSearchTool._run()` | 177 | `tools.py` | P1 |
| `list_researchers()` | 86 | `researcher.py` | P2 |

**Refactoring Approach:**
1. Extract sub-methods
2. Apply Single Level of Abstraction (SLA) principle
3. Create helper classes

**Example**:
```python
# Before: 226-line method
async def process_query(query_id, query, ...):
    # 50 lines of setup
    # 80 lines of processing
    # 60 lines of parsing
    # 36 lines of error handling

# After: Extract sub-methods
async def process_query(query_id, query, ...):
    state = self._create_query_state(query_id, query)
    agent = self._create_agent()

    try:
        result = await self._invoke_agent(agent, query)
        parsed = self._parse_response(result)
        self._update_state_success(state, parsed)
        return self._create_response(parsed)
    except Exception as e:
        error_msg = self._handle_error(e)
        return self._create_error_response(error_msg)
```

---

### 6.3 Duplicated Code

**Pattern 1: File Reading**

Duplicated in:
- `researcher.py` (lines 59-62, 131-140, 173-178, 209-213)
- `vector_db_builder.py` (lines 60-66, 88-92)

```python
# Appears 6+ times
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
```

**Solution:**
```python
class JSONFileReader:
    @staticmethod
    def read(path: Path) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def read_all(directory: Path, pattern: str = "*.json") -> List[Dict]:
        return [
            JSONFileReader.read(f)
            for f in directory.glob(pattern)
            if f.name != "summary.json"
        ]
```

---

**Pattern 2: Error Handling**

Similar error handling throughout:
```python
except Exception as e:
    logger.error(f"Error loading researchers: {e}")
    return []
```

**Solution:**
```python
def handle_and_log_error(
    logger,
    error: Exception,
    context: str,
    default_return=None
):
    logger.error(f"{context}: {error}")
    if default_return is not None:
        return default_return
    raise
```

---

**Pattern 3: Metadata Extraction**

Duplicated metadata extraction:
```python
name = doc.metadata.get("researcher_name", "Unknown")
program = doc.metadata.get("program", "Unknown")
department = doc.metadata.get("department", "Unknown")
```

**Solution:**
```python
@dataclass
class DocumentMetadata:
    researcher_name: str = "Unknown"
    program: str = "Unknown"
    department: str = "Unknown"
    profile_url: str = ""

    @classmethod
    def from_dict(cls, metadata: Dict) -> 'DocumentMetadata':
        return cls(
            researcher_name=metadata.get("researcher_name", "Unknown"),
            program=metadata.get("program", "Unknown"),
            department=metadata.get("department", "Unknown"),
            profile_url=metadata.get("profile_url", "")
        )
```

---

### 6.4 Magic Numbers

**Definition**: Unexplained numbers in code

**Examples:**

```python
# tools.py
alpha = 0.3  # Why 0.3?
alpha = 0.7  # Why 0.7?
k = 5        # Why 5?
[:100]       # Why 100?

# agent.py
temperature = 0.7      # Why 0.7?
max_llm_calls = 6      # Why 6?

# vector_db_builder.py
chunk_size = 1024      # Why 1024?

# hybrid_search.py
k * 2  # Why multiply by 2?
k * 3  # Why multiply by 3?
```

**Solution: Configuration Constants**

```python
# config/search_config.py
@dataclass
class SearchConfig:
    """Search algorithm configuration"""

    # Alpha values for hybrid search weighting
    NAME_SEARCH_ALPHA: float = 0.3  # Favor keyword for names
    TOPIC_SEARCH_ALPHA: float = 0.7  # Favor semantic for topics

    # Result counts
    DEFAULT_RESULT_COUNT: int = 5
    CANDIDATE_MULTIPLIER: int = 2  # Get k*2 candidates
    DOCUMENT_MULTIPLIER: int = 3   # Retrieve k*3 documents

    # Display limits
    QUERY_PREVIEW_LENGTH: int = 100
    RESULT_TRUNCATION_LENGTH: int = 1000

@dataclass
class AgentConfig:
    """Agent configuration"""
    DEFAULT_TEMPERATURE: float = 0.7
    MAX_LLM_CALLS: int = 6
    QUERY_TIMEOUT_SECONDS: int = 300

@dataclass
class ChunkingConfig:
    """Chunking configuration"""
    DEFAULT_CHUNK_SIZE: int = 1024
    MAX_PUBLICATIONS_PER_CHUNK: int = 5
    MAX_GRANTS_PER_CHUNK: int = 3

# Usage
def hybrid_search(query, k, search_type):
    config = SearchConfig()

    if search_type == "name":
        alpha = config.NAME_SEARCH_ALPHA
    else:
        alpha = config.TOPIC_SEARCH_ALPHA

    candidates = k * config.CANDIDATE_MULTIPLIER
```

---

### 6.5 Primitive Obsession

**Definition**: Using primitives instead of small objects

**Examples:**

**String-based Status** (should be enum):
```python
# agent.py
_query_statuses[query_id] = {
    "status": "processing",  # String instead of enum
    "query_id": query_id,
    "progress": 0.0
}
```

**Dict for Complex Data** (should be class):
```python
# vector_db.py
_db_stats = {
    "total_researchers": 0,
    "total_chunks": 0,
    "last_updated": None,
    "status": "not_initialized"
}
```

**Solution: Proper Types**

```python
# 1. Status as Enum
class QueryStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class DatabaseStatus(Enum):
    NOT_INITIALIZED = "not_initialized"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"

# 2. Proper Classes
@dataclass
class QueryState:
    query_id: str
    status: QueryStatus
    query: str
    start_time: datetime
    progress: float = 0.0
    error: Optional[str] = None

@dataclass
class DatabaseStats:
    total_researchers: int = 0
    total_chunks: int = 0
    last_updated: Optional[datetime] = None
    status: DatabaseStatus = DatabaseStatus.NOT_INITIALIZED

# 3. Usage
# Before
_query_statuses[query_id] = {
    "status": "processing",
    "query": query
}

# After
_query_states[query_id] = QueryState(
    query_id=query_id,
    status=QueryStatus.PROCESSING,
    query=query,
    start_time=datetime.now()
)
```

---

### 6.6 Feature Envy

**Definition**: Method that seems more interested in a class other than the one it's in

**Example** (tools.py lines 224-251):
```python
class ResearcherSearchTool:
    def _run(self, ...):
        # Tool is obsessed with Document internals
        if not hasattr(doc, 'metadata'):
            content = doc.page_content if hasattr(doc, 'page_content') else "No content"

        name = doc.metadata.get("researcher_name", "Unknown")
        program = doc.metadata.get("program", "Unknown")
        department = doc.metadata.get("department", "Unknown")
        # ... more doc manipulation
```

**Solution: Move to Document or Create Formatter**

```python
# Option 1: Extend Document
class ResearcherDocument(Document):
    @property
    def researcher_name(self) -> str:
        return self.metadata.get("researcher_name", "Unknown")

    @property
    def program(self) -> str:
        return self.metadata.get("program", "Unknown")

    def format_for_display(self) -> str:
        return f"""
        Researcher: {self.researcher_name}
        Program: {self.program}
        Department: {self.department}
        Content: {self.page_content}
        """

# Option 2: Create Formatter (better - follows SRP)
class DocumentFormatter:
    def format(self, doc: Document) -> str:
        metadata = self._extract_metadata(doc)
        return self._format_template(metadata, doc.page_content)

    def _extract_metadata(self, doc: Document) -> DocumentMetadata:
        return DocumentMetadata.from_dict(doc.metadata)

# Usage
class ResearcherSearchTool:
    def __init__(self, formatter: DocumentFormatter):
        self._formatter = formatter

    def _run(self, ...):
        results = search(...)
        return "\n\n".join(
            self._formatter.format(doc) for doc in results
        )
```

---

### Summary: Code Smells

| Smell | Count | Priority | Effort |
|-------|-------|----------|--------|
| God classes | 3 | P0 | High |
| Long methods (>200 lines) | 4 | P0 | Medium |
| Duplicated code | 10+ | P1 | Low |
| Magic numbers | 15+ | P2 | Low |
| Primitive obsession | 5+ | P2 | Low |
| Feature envy | 2 | P2 | Low |

---

## 7. Testability Issues

### 7.1 Global State Prevents Testing

**Problem**: Cannot test functions independently

**Examples:**

```python
# tools.py line 27
_name_normalizer = NameNormalizationService()

# Cannot inject mock for testing
def test_researcher_search():
    # How to mock _name_normalizer?
    tool = ResearcherSearchTool()
    result = tool._run("John Doe")
```

**Solution:**

```python
# Dependency Injection
class ResearcherSearchTool:
    def __init__(self, normalizer: INameNormalizer):
        self._normalizer = normalizer

# Now testable
def test_researcher_search():
    mock_normalizer = Mock(spec=INameNormalizer)
    mock_normalizer.normalize.return_value = NormResult(
        canonical="John Doe",
        method="exact"
    )

    tool = ResearcherSearchTool(mock_normalizer)
    result = tool._run("john doe")

    assert "John Doe" in result
    mock_normalizer.normalize.assert_called_once()
```

---

### 7.2 Direct File System Access

**Problem**: Cannot test without real files

```python
def list_researchers():
    data_dir = settings.PROCESSED_DATA_DIR
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    # Need actual files to test
```

**Solution: Repository Pattern**

```python
# Test with mock repository
def test_list_researchers():
    mock_repo = Mock(spec=IResearcherRepository)
    mock_repo.list.return_value = ([mock_data], 1)

    service = ResearcherService(mock_repo)
    result = service.list_researchers()

    assert result.total == 1
```

---

### 7.3 Tight Coupling to External Services

**Problem**: Hard to test without LLM/Vector DB

```python
def process_query(query):
    # Calls real LLM
    llm = ChatGroq(...)
    agent = create_react_agent(llm=llm, ...)
    result = agent.invoke({"input": query})
```

**Solution: Mock via Interfaces**

```python
def test_process_query():
    mock_llm = Mock(spec=ILLMProvider)
    mock_llm.create_model.return_value = Mock()

    mock_framework = Mock(spec=IAgentFramework)
    mock_agent = Mock(spec=IAgent)
    mock_agent.invoke.return_value = "Test response"
    mock_framework.create_agent.return_value = mock_agent

    service = AgentService(mock_framework, llm_config={}, ...)
    result = await service.process_query("test query")

    assert result == "Test response"
```

---

### 7.4 Missing Dependency Injection

**Current State**: Services create their own dependencies

```python
def create_researcher_agent():
    llm = get_llm_model()  # Created internally
    tools = get_tools()     # Created internally
    # Hard to inject mocks
```

**Improved with DI:**

```python
class AgentFactory:
    def __init__(
        self,
        llm_factory: ILLMFactory,
        tool_registry: IToolRegistry
    ):
        self._llm_factory = llm_factory
        self._tool_registry = tool_registry

    def create(self, config: AgentConfig):
        llm = self._llm_factory.create(config.llm_config)
        tools = self._tool_registry.get_tools()
        return self._build_agent(llm, tools)

# Testing
def test_agent_creation():
    mock_llm_factory = Mock()
    mock_tool_registry = Mock()

    factory = AgentFactory(mock_llm_factory, mock_tool_registry)
    agent = factory.create(test_config)

    # Verify mocks were called correctly
```

---

### Summary: Testability Issues

| Issue | Impact | Solution | Priority |
|-------|--------|----------|----------|
| Global state | Cannot isolate tests | DI | P0 |
| File system coupling | Need real files | Repository pattern | P0 |
| External service coupling | Need real LLM/DB | Interface abstraction | P1 |
| Missing DI | Hard to mock | Add DI container | P1 |

---

## 8. Best Practice Violations

### 8.1 Logging

**Issues:**

1. **Inconsistent log levels:**
```python
logger.info(f"Loading embedding model: {model}")  # Should be DEBUG
logger.warning(f"Vector database directory does not exist")  # Should be ERROR
```

2. **Excessive logging in hot paths:**
```python
# tools.py - logs 10+ events per search
log_tool_event("user_query_received", ...)
log_tool_event("input_parsed_from_json", ...)
log_tool_event("name_search_detected", ...)
log_tool_event("name_normalized", ...)
# ... continues
```

3. **Missing structured logging:**
```python
# String interpolation instead of structured data
logger.info(f"Query: {query}, Type: {type}")

# Should be:
logger.info("Query received", extra={
    "query": query,
    "type": type,
    "query_id": query_id
})
```

**Solution:**

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# Usage
logger.info(
    "query_received",
    query_id=query_id,
    query_length=len(query),
    query_type=query_type
)

# Configure log levels properly
logger.debug("Loading embedding model")  # Not INFO
logger.error("Database directory missing")  # Not WARNING
```

---

### 8.2 Error Handling

**Issues:**

1. **Bare except clauses:**
```python
try:
    # code
except Exception as e:  # Too broad
    logger.error(f"Error: {e}")
```

2. **Swallowing exceptions:**
```python
try:
    data = load_file(path)
except Exception as e:
    logger.error(f"Failed: {e}")
    return []  # Silently returns empty
```

3. **Generic error messages:**
```python
return "Error: Something went wrong"  # Not helpful
```

**Solution:**

```python
# 1. Specific exceptions
class ResearcherNotFoundError(Exception):
    pass

class DatabaseConnectionError(Exception):
    pass

# 2. Proper error handling
try:
    data = load_researcher(researcher_id)
except FileNotFoundError:
    raise ResearcherNotFoundError(
        f"Researcher {researcher_id} not found"
    )
except PermissionError as e:
    logger.error("Permission denied", error=str(e))
    raise
except Exception as e:
    logger.exception("Unexpected error loading researcher")
    raise

# 3. Error context
class QueryProcessingError(Exception):
    def __init__(self, query_id: str, message: str, cause: Exception):
        self.query_id = query_id
        self.cause = cause
        super().__init__(f"Query {query_id}: {message}")
```

---

### 8.3 Async/Await

**Issues:**

1. **Fake async methods:**
```python
async def _arun(self, ...):
    return self._run(...)  # Not actually async!
```

2. **Blocking calls in async functions:**
```python
async def process_query(query):
    result = agent.invoke(query)  # Blocks event loop!
```

3. **No async file I/O:**
```python
async def load_data():
    with open(file) as f:  # Blocking I/O in async!
        return f.read()
```

**Solution:**

```python
# 1. True async
async def _arun(self, query):
    # Use asyncio.to_thread for CPU-bound work
    return await asyncio.to_thread(self._run, query)

# 2. Non-blocking calls
async def process_query(query):
    result = await asyncio.to_thread(agent.invoke, query)

# 3. Async file I/O
import aiofiles

async def load_data(path):
    async with aiofiles.open(path, 'r') as f:
        return await f.read()
```

---

### 8.4 Configuration

**Issues:**

1. **Direct environment variable access:**
```python
api_key = os.getenv("OPENAI_API_KEY")  # Scattered throughout
```

2. **Mutable global config:**
```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Modified at runtime
```

3. **Missing validation:**
```python
# No validation that required keys are present
api_key = settings.OPENAI_API_KEY  # Could be None!
```

**Solution:**

```python
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    # Required fields
    GROQ_API_KEY: str

    # Optional with defaults
    LLM_PROVIDER: str = "groq"

    @validator('GROQ_API_KEY')
    def validate_groq_key(cls, v, values):
        if values.get('LLM_PROVIDER') == 'groq' and not v:
            raise ValueError("GROQ_API_KEY required when using Groq")
        return v

    @validator('OPENAI_API_KEY')
    def validate_openai_key(cls, v, values):
        if values.get('LLM_PROVIDER') == 'openai' and not v:
            raise ValueError("OPENAI_API_KEY required when using OpenAI")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True

# Validate on startup
try:
    settings = Settings()
except ValidationError as e:
    logger.error("Configuration validation failed")
    sys.exit(1)
```

---

### 8.5 Security

**Issues:**

1. **API keys in logs:**
```python
logger.info(f"Using key: {settings.API_KEY}")  # Exposed in logs!
```

2. **No input validation:**
```python
@router.post("/query")
async def query(request: QueryRequest):
    # No sanitization of request.query
    result = process_query(request.query)
```

3. **Path traversal risk:**
```python
file_path = os.path.join(data_dir, filename)
# If filename is "../../etc/passwd", could escape directory
```

**Solution:**

```python
# 1. Mask sensitive data in logs
def mask_api_key(key: str) -> str:
    if len(key) > 8:
        return f"{key[:4]}...{key[-4:]}"
    return "****"

logger.info(f"Using key: {mask_api_key(settings.API_KEY)}")

# 2. Input validation
from pydantic import BaseModel, validator

class QueryRequest(BaseModel):
    query: str

    @validator('query')
    def validate_query(cls, v):
        if len(v) > 1000:
            raise ValueError("Query too long")
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

# 3. Path validation
from pathlib import Path

def safe_join(base_dir: Path, filename: str) -> Path:
    # Resolve and ensure still within base_dir
    full_path = (base_dir / filename).resolve()
    if not full_path.is_relative_to(base_dir):
        raise ValueError("Invalid path: attempted directory traversal")
    return full_path
```

---

### Summary: Best Practice Violations

| Category | Issues | Priority | Effort |
|----------|--------|----------|--------|
| Logging | Inconsistent levels, too verbose | P2 | Low |
| Error handling | Bare except, swallowing errors | P1 | Medium |
| Async/await | Fake async, blocking calls | P1 | Medium |
| Configuration | Mutable, unvalidated | P2 | Low |
| Security | Exposed keys, no validation | P1 | Medium |

---

## 9. Optimization Opportunities

### 9.1 Performance

#### 9.1.1 File System Operations

**Problem**: Scanning all files on every request

```python
def list_researchers():
    # Scans directory on every call
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    for filename in json_files:
        with open(file_path) as f:
            data = json.load(f)
```

**Solution: Caching**

```python
from functools import lru_cache
from datetime import datetime, timedelta

class ResearcherCache:
    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._ttl = timedelta(seconds=ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            if datetime.now() - self._timestamps[key] < self._ttl:
                return self._cache[key]
            else:
                # Expired
                del self._cache[key]
                del self._timestamps[key]
        return None

    def set(self, key: str, value: Any):
        self._cache[key] = value
        self._timestamps[key] = datetime.now()

    def invalidate(self, key: Optional[str] = None):
        if key:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
        else:
            self._cache.clear()
            self._timestamps.clear()

# Usage
cache = ResearcherCache(ttl_seconds=300)

def list_researchers():
    cached = cache.get("all_researchers")
    if cached:
        return cached

    # Load from files
    data = _load_from_files()
    cache.set("all_researchers", data)
    return data
```

---

#### 9.1.2 Repeated Searches

**Problem**: Agent makes similar searches repeatedly

**Solution: Query Result Caching**

```python
from hashlib import md5

class SearchCache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 600):
        self._cache: Dict[str, Tuple[List, datetime]] = {}
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)

    def _make_key(self, query: str, k: int, filter: Dict) -> str:
        # Hash query parameters
        key_data = f"{query}|{k}|{json.dumps(filter, sort_keys=True)}"
        return md5(key_data.encode()).hexdigest()

    def get(self, query: str, k: int, filter: Dict) -> Optional[List]:
        key = self._make_key(query, k, filter)
        if key in self._cache:
            results, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._ttl:
                return results
            else:
                del self._cache[key]
        return None

    def set(self, query: str, k: int, filter: Dict, results: List):
        # LRU eviction
        if len(self._cache) >= self._max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        key = self._make_key(query, k, filter)
        self._cache[key] = (results, datetime.now())

# Usage
search_cache = SearchCache()

def hybrid_search(query, k, filter):
    cached = search_cache.get(query, k, filter)
    if cached:
        logger.info("Cache hit")
        return cached

    # Perform actual search
    results = _perform_search(query, k, filter)
    search_cache.set(query, k, filter, results)
    return results
```

---

#### 9.1.3 Large Result Formatting

**Problem**: Formatting all results in memory

**Solution: Streaming/Generator**

```python
# Before: Load all in memory
def format_results(documents: List[Document]) -> str:
    formatted = []
    for doc in documents:
        formatted.append(self._format_single(doc))
    return "\n\n".join(formatted)

# After: Use generator
def format_results(documents: List[Document]) -> Iterator[str]:
    """Yield formatted results one at a time"""
    for i, doc in enumerate(documents):
        yield self._format_single(i, doc)
        if i < len(documents) - 1:
            yield "\n\n---\n\n"

# Usage
def _run(self, ...):
    results = search(...)
    # Stream results instead of loading all
    return "".join(format_results(results))
```

---

### 9.2 Memory

#### 9.2.1 Loading All Files

**Problem**: `load_all_researcher_profiles()` loads everything

**Solution: Streaming**

```python
# Before: Load all at once
def load_all_researcher_profiles(dir: Path) -> List[Dict]:
    profiles = []
    for file in dir.glob("*.json"):
        with open(file) as f:
            profiles.append(json.load(f))  # All in memory!
    return profiles

# After: Stream with generator
def load_researcher_profiles(dir: Path) -> Iterator[Dict]:
    """Yield profiles one at a time"""
    for file in dir.glob("*.json"):
        if file.name != "summary.json":
            with open(file, 'r') as f:
                yield json.load(f)

# Usage in builder
def build_database(profiles_dir: Path):
    all_chunks = []
    for profile in load_researcher_profiles(profiles_dir):
        chunks = create_chunks(profile)
        all_chunks.extend(chunks)

        # Process in batches to limit memory
        if len(all_chunks) >= 1000:
            db.add_documents(all_chunks)
            all_chunks = []  # Clear memory

    # Add remaining
    if all_chunks:
        db.add_documents(all_chunks)
```

---

#### 9.2.2 Unbounded In-Memory Caches

**Problem**: Caches grow without limit

```python
# agent.py - grows forever
_query_statuses: Dict[str, Dict[str, Any]] = {}
```

**Solution: LRU Cache**

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value

        # Evict oldest if over limit
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

# Usage
_query_statuses = LRUCache(max_size=1000)
```

---

### 9.3 Database

#### 9.3.1 Batch Operations

**Problem**: Adding documents one at a time

**Solution: Batching**

```python
# Before
for chunk in chunks:
    db.add_document(chunk)  # Individual API call each time

# After
BATCH_SIZE = 100

def add_chunks_batched(db, chunks: List):
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        db.add_documents(batch)  # Batch API call
        logger.info(f"Added batch {i//BATCH_SIZE + 1}")
```

---

#### 9.3.2 Connection Pooling

**Solution: Reuse connections**

```python
class ChromaDBPool:
    def __init__(self, max_connections: int = 5):
        self._pool: Queue = Queue(maxsize=max_connections)
        self._max = max_connections
        self._count = 0

    def get_connection(self):
        if not self._pool.empty():
            return self._pool.get()
        elif self._count < self._max:
            self._count += 1
            return self._create_connection()
        else:
            # Wait for available connection
            return self._pool.get()

    def return_connection(self, conn):
        self._pool.put(conn)

    def _create_connection(self):
        return Chroma(...)
```

---

### Summary: Optimization Opportunities

| Category | Opportunity | Impact | Effort |
|----------|-------------|--------|--------|
| **Performance** | File caching | High | Low |
| **Performance** | Search result caching | Medium | Low |
| **Performance** | Stream results | Low | Low |
| **Memory** | Stream file loading | Medium | Medium |
| **Memory** | LRU caches | Low | Low |
| **Database** | Batch operations | Medium | Low |
| **Database** | Connection pooling | Low | Medium |

---

## 10. Refactoring Roadmap

### Priority 0 (Must Fix - High Impact, Blocks Progress)

#### 1. Extract Repository Layer

**Impact**: Critical - Blocks testing and future storage changes
**Effort**: Medium (2-3 days)
**Files**: `researcher.py`, `vector_db_builder.py`

**Steps:**
1. Create `IResearcherRepository` interface
2. Implement `FileSystemResearcherRepository`
3. Update `researcher.py` to use repository
4. Write tests with mock repository
5. Update `vector_db_builder.py` similarly

**Success Criteria:**
- Can run tests without real files
- Can swap file storage for database
- Services depend on interface, not implementation

---

#### 2. Implement Dependency Injection

**Impact**: Critical - Enables testing, reduces coupling
**Effort**: High (3-5 days)
**Files**: All services

**Steps:**
1. Remove global variables (`_name_normalizer`, `_cached_embedding_function`, etc.)
2. Add constructor injection to all services
3. Create FastAPI dependency providers
4. Update endpoints to use Depends()
5. Write tests with mock dependencies

**Example Pattern:**
```python
# Before
_name_normalizer = NameNormalizationService()

class Tool:
    def run(self):
        return _name_normalizer.normalize(...)

# After
class Tool:
    def __init__(self, normalizer: INameNormalizer):
        self._normalizer = normalizer

def get_tool(normalizer: INameNormalizer = Depends(get_normalizer)):
    return Tool(normalizer)
```

**Success Criteria:**
- No global service instances
- All dependencies injected
- Can test with mocks

---

#### 3. Break Down God Methods

**Impact**: High - Improves maintainability
**Effort**: Medium (2-3 days)
**Files**: `agent.py`, `vector_db_builder.py`, `tools.py`, `hybrid_search.py`

**Target Methods:**
- `process_query()` (226 lines) → Extract 5-6 methods
- `create_researcher_chunks()` (210 lines) → Extract 4 methods
- `hybrid_search()` (175 lines) → Extract 3 methods
- `ResearcherSearchTool._run()` (177 lines) → Extract 4 methods

**Approach:**
1. Identify logical sections
2. Extract to private methods
3. Apply Single Level of Abstraction
4. Add tests for extracted methods

**Success Criteria:**
- No method over 50 lines
- Each method has single, clear purpose
- Methods are testable independently

---

### Priority 1 (Should Fix - Important for Quality)

#### 4. Abstract LLM Providers

**Impact**: Medium - Enables provider swapping
**Effort**: Low (1 day)
**Files**: `llm.py`

**Steps:**
1. Create `ILLMProvider` interface
2. Implement `OpenAIProvider`, `GroqProvider`
3. Create `LLMProviderFactory` with registry
4. Update `get_llm_model()` to use factory
5. Add tests for each provider

**Success Criteria:**
- Can add new providers without modifying existing code
- Each provider is independently testable
- Clean extension point

---

#### 5. Split Vector DB Module

**Impact**: Medium - Improves organization
**Effort**: Low (1 day)
**Files**: `vector_db.py`

**Steps:**
1. Split into: `search.py`, `management.py`, `rebuild.py`, `embeddings.py`
2. Update imports throughout codebase
3. Ensure clean module boundaries

**Success Criteria:**
- Focused modules with single responsibility
- Clear import paths
- No circular dependencies

---

#### 6. Implement Proper Async

**Impact**: Medium - Fixes event loop blocking
**Effort**: Medium (2 days)
**Files**: `tools.py`, `agent.py`

**Steps:**
1. Replace fake async methods with `asyncio.to_thread()`
2. Use `aiofiles` for file I/O
3. Make database calls truly async
4. Test with concurrent requests

**Success Criteria:**
- No blocking calls in async functions
- Event loop not blocked
- Better concurrency

---

#### 7. Remove Global State

**Impact**: Medium - Enables testing
**Effort**: Medium (2 days)
**Files**: `agent.py`, `vector_db.py`, `tools.py`

**Steps:**
1. Convert global dicts to classes
2. Inject state managers instead of using globals
3. Add proper cleanup/lifecycle management
4. Write tests

**Success Criteria:**
- No global mutable state
- State is managed explicitly
- Tests are isolated

---

### Priority 2 (Nice to Have - Quality of Life)

#### 8. Add Configuration Constants

**Impact**: Low - Improves readability
**Effort**: Low (0.5 days)
**Files**: Multiple

**Steps:**
1. Create `config/constants.py`
2. Extract all magic numbers
3. Document each constant
4. Replace throughout codebase

---

#### 9. Implement Caching

**Impact**: Low - Performance improvement
**Effort**: Low (1 day)
**Files**: `researcher.py`, `vector_db.py`

**Steps:**
1. Implement `ResearcherCache`
2. Implement `SearchCache`
3. Add cache warming on startup
4. Add cache invalidation endpoints

---

#### 10. Structured Logging

**Impact**: Low - Better observability
**Effort**: Low (0.5 days)
**Files**: All

**Steps:**
1. Configure `structlog`
2. Replace string interpolation with structured logs
3. Review and fix log levels
4. Add log sampling for hot paths

---

### Estimated Timeline

| Priority | Items | Total Effort | Parallel Work | Timeline |
|----------|-------|--------------|---------------|----------|
| **P0** | 3 items | 7-11 days | 2 developers | 2 weeks |
| **P1** | 4 items | 7-8 days | 2 developers | 1.5 weeks |
| **P2** | 3 items | 2-2.5 days | 1 developer | 0.5 weeks |
| **Total** | 10 items | 16-21.5 days | 2 developers | **4 weeks** |

---

### Recommended Approach

**Phase 1 (Week 1-2): Foundation - P0 Items**
- Extract repository layer
- Implement dependency injection
- Break down god methods
- **Outcome**: Testable, maintainable codebase

**Phase 2 (Week 3): Improvements - P1 Items**
- Abstract LLM providers
- Split vector DB module
- Fix async issues
- Remove global state
- **Outcome**: Better architecture, no blocking

**Phase 3 (Week 4): Polish - P2 Items**
- Add constants
- Implement caching
- Structured logging
- **Outcome**: Production-ready, performant

---

## Conclusion

This analysis identified **significant architectural issues** that should be addressed to improve code quality, testability, and maintainability:

### Critical Issues (P0):
1. ✗ **No repository layer** - Direct file system access everywhere
2. ✗ **No dependency injection** - Global state and tight coupling
3. ✗ **God methods** - Functions with 200+ lines

### Important Issues (P1):
4. ✗ **Hard-coded dependencies** - Cannot swap implementations
5. ✗ **Fake async methods** - Blocks event loop
6. ✗ **Global mutable state** - Cannot test in isolation

### Quality Issues (P2):
7. ✗ **Magic numbers** - Hard-coded constants throughout
8. ✗ **No caching** - Repeated file system operations
9. ✗ **Inconsistent logging** - Wrong levels, too verbose

### Recommended Next Steps

1. **Start with P0 items** - These block progress on testing and maintenance
2. **Use incremental refactoring** - Don't rewrite everything at once
3. **Write tests as you go** - Prove refactoring doesn't break functionality
4. **Get code reviews** - Ensure team alignment on patterns

### Success Metrics

After refactoring:
- ✓ Test coverage > 80%
- ✓ No function > 50 lines
- ✓ All services use dependency injection
- ✓ Can swap storage without code changes
- ✓ Can test without external dependencies

---

**Document Version**: 1.0
**Last Updated**: October 29, 2025
**Next Review**: After Phase 1 completion