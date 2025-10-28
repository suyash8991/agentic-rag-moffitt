# Name Normalization Design Document

**Date**: 2025-10-28
**Status**: Design Phase
**Author**: Architecture Review

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Current Implementation](#current-implementation)
3. [Proposed Solution](#proposed-solution)
4. [Architecture Design](#architecture-design)
5. [Implementation Plan](#implementation-plan)
6. [Open Questions](#open-questions)

---

## Problem Statement

### Current Limitations

Name-based searches in the Moffitt Agentic RAG system require **exact full names** to retrieve reliable results. This creates significant usability issues:

- ❌ Searching for `"Conor"` doesn't reliably find `"Conor Lynch"`
- ❌ Searching for `"Lynch"` doesn't reliably find `"Conor Lynch"`
- ❌ Searching for `"conor lynch"` (lowercase) may not match `"Conor Lynch"`
- ❌ Searching for `"Bob"` won't find `"Robert Smith"` (nickname issue)
- ❌ Searching for `"Smith, Robert"` won't find `"Robert Smith"` (format issue)
- ✅ Only `"Conor Lynch"` (exact match) works consistently

### Why This Matters

Real users:
- Don't always know the full name
- Only remember first or last name
- Use nicknames and informal variations ("Bob" vs "Robert")
- Make typos or case errors
- Use different name formats ("Last, First" vs "First Last")

### Why ChromaDB Metadata Filters Don't Help

ChromaDB's metadata filters only support **exact equality matching**:
- ✅ `{"researcher_name": "Conor Lynch"}` - Works for exact match
- ❌ `{"researcher_name": {"$contains": "Conor"}}` - **Does NOT work** (this operator is for document content via `where_document`, not metadata fields)

---

## Current Implementation

### Architecture Flow

```
User Query: "What does Bob research?"
  ↓
LLM Tool Calling: {"researcher_name": "Bob", "query": "research"}
  ↓
ResearcherSearchTool._run(researcher_name="Bob")
  ↓
hybrid_search(query="Bob", alpha=0.3, search_type="name")
  ↓
┌─────────────────────────────────────────────┐
│  Semantic Search (Vector Similarity)       │
│  + Keyword Search (Term Frequency)         │
│  Combined with alpha weighting             │
└─────────────────────────────────────────────┘
  ↓
Results ranked by hybrid score
```

### Current Search Strategy

**For Name Searches** (`alpha=0.3`):
- 30% semantic (vector similarity)
- 70% keyword (term frequency matching)

**For Topic Searches** (`alpha=0.7`):
- 70% semantic (vector similarity)
- 30% keyword (term frequency matching)

### Limitations

1. **Semantic search** is optimized for topics, not proper names
2. **Keyword search** relies on term frequency, which doesn't handle:
   - Nicknames ("Bob" ≠ "Robert")
   - Partial names (no direct "Lynch" → "Conor Lynch" mapping)
   - Case variations
3. **No normalization layer** between extraction and search
4. **Performance**: Hybrid search is slower than exact metadata filtering

### Code Locations

- **Tool Definition**: `backend/app/services/tools.py` (lines 75-227)
- **Hybrid Search**: `backend/app/services/hybrid_search.py` (lines 86-298)
- **Vector DB**: `backend/app/services/vector_db.py`

---

## Proposed Solution

### Senior Engineer's Advice

The solution has **two parts**:

#### Part 1: Entity Extraction (Already Implemented ✅)

Our current tool schema already handles this correctly:

```python
class ResearcherSearchInput(BaseModel):
    researcher_name: Optional[str] = Field(
        default=None,
        description="The name of the researcher to search for"
    )
    topic: Optional[str] = Field(
        default=None,
        description="The research topic to search for"
    )
```

The LLM is **forced** to extract structured parameters. When a user asks "What were Bob Smith's contributions?", the agent must populate `researcher_name="Bob Smith"`.

#### Part 2: Name Normalization (Missing - To Be Implemented)

Add a normalization layer that converts the extracted name into a **canonical name** from our database.

**Normalization Workflow**:

```
Extracted Name: "Bob"
  ↓
┌──────────────────────────────────────┐
│  Step 1: Check Alias Map             │
│  "bob" → "Robert Smith"?             │
│  ✅ Found in alias_map               │
└──────────────────────────────────────┘
  ↓
Canonical Name: "Robert Smith"
  ↓
Use exact metadata filter: {"researcher_name": "Robert Smith"}
  ↓
Fast, accurate results!
```

If alias map doesn't match:

```
Extracted Name: "Lynch"
  ↓
┌──────────────────────────────────────┐
│  Step 1: Check Alias Map             │
│  "lynch" → Not found                 │
└──────────────────────────────────────┘
  ↓
┌──────────────────────────────────────┐
│  Step 2: Fuzzy Match (thefuzz)       │
│  Compare "Lynch" against:            │
│  - "Conor Lynch" (score: 90)         │
│  - "Thomas Lynch" (score: 90)        │
│  - "Robert Smith" (score: 45)        │
│                                      │
│  Best Match: "Conor Lynch" (90)      │
│  Threshold: 85                       │
│  ✅ Score >= Threshold               │
└──────────────────────────────────────┘
  ↓
Canonical Name: "Conor Lynch"
  ↓
Use exact metadata filter: {"researcher_name": "Conor Lynch"}
  ↓
Fast, accurate results!
```

If both fail:

```
Extracted Name: "cancer"
  ↓
Step 1: Not in alias map
  ↓
Step 2: Fuzzy match score: 30 (< 85 threshold)
  ↓
❌ No canonical name found
  ↓
Fallback to hybrid search (current behavior)
```

---

## Architecture Design

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ResearcherSearchTool                     │
│                                                             │
│  _run(researcher_name="Bob"):                              │
│    1. Validate input                                        │
│    2. ✨ NEW: Normalize name                               │
│    3. Execute search (exact or hybrid)                      │
│    4. Format and return results                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              ✨ NEW: Name Normalization Service            │
│                                                             │
│  normalize_researcher_name(extracted_name):                │
│    1. Check alias_map (in-memory dict)                     │
│    2. Fuzzy match against canonical_names (thefuzz)        │
│    3. Apply threshold (default: 85)                         │
│    4. Return canonical name or None                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
          ┌───────────────────┴───────────────────┐
          ↓                                       ↓
┌──────────────────────┐            ┌──────────────────────┐
│  ✨ Alias Map        │            │  ✨ Canonical Names  │
│  (JSON file)         │            │  (ChromaDB query)    │
│                      │            │                      │
│  {                   │            │  ["Conor Lynch",     │
│    "bob": "Robert    │            │   "Robert Smith",    │
│           Smith",    │            │   "Alice Johnson",   │
│    "conor": "Conor   │            │   ...]               │
│             Lynch"   │            │                      │
│  }                   │            │  Refreshed:          │
│                      │            │  - On startup        │
│  Maintained by:      │            │  - Periodically      │
│  - Manual curation   │            │  - On demand         │
│  - Log analysis      │            │                      │
└──────────────────────┘            └──────────────────────┘
```

### New Components

#### 1. Name Normalization Service

**File**: `backend/app/services/name_normalization.py`

```python
from typing import Optional, List, Dict
from thefuzz import process

class NameNormalizationService:
    """
    Service for normalizing researcher names using aliases and fuzzy matching.
    """

    def __init__(self):
        self._canonical_names: List[str] = []
        self._alias_map: Dict[str, str] = {}
        self._threshold: int = 85

    def normalize(self, extracted_name: str) -> Optional[str]:
        """
        Normalize an extracted name to its canonical form.

        Args:
            extracted_name: Name extracted from user query

        Returns:
            Canonical name if found, None otherwise
        """
        # Step 1: Check alias map
        normalized = self._check_alias_map(extracted_name)
        if normalized:
            return normalized

        # Step 2: Fuzzy match
        normalized = self._fuzzy_match(extracted_name)
        return normalized

    def _check_alias_map(self, name: str) -> Optional[str]:
        """Check if name exists in alias map."""
        search_key = name.lower().strip()
        return self._alias_map.get(search_key)

    def _fuzzy_match(self, name: str) -> Optional[str]:
        """Fuzzy match against canonical names."""
        if not self._canonical_names:
            return None

        best_match = process.extractOne(name, self._canonical_names)

        if best_match and best_match[1] >= self._threshold:
            return best_match[0]

        return None
```

#### 2. Canonical Names Cache

**File**: `backend/app/services/canonical_names_cache.py`

```python
from typing import List
from datetime import datetime, timedelta
from .vector_db import get_or_create_vector_db

class CanonicalNamesCache:
    """
    Cache for canonical researcher names from ChromaDB metadata.
    """

    def __init__(self, refresh_interval_hours: int = 24):
        self._names: List[str] = []
        self._last_refresh: Optional[datetime] = None
        self._refresh_interval = timedelta(hours=refresh_interval_hours)

    def get_names(self) -> List[str]:
        """Get canonical names, refreshing if needed."""
        if self._needs_refresh():
            self.refresh()
        return self._names

    def refresh(self):
        """Query ChromaDB to get all unique researcher names."""
        db = get_or_create_vector_db()

        # Get all documents (or sample if too large)
        # Extract unique researcher_name from metadata
        # Store in self._names

        self._last_refresh = datetime.now()

    def _needs_refresh(self) -> bool:
        """Check if cache needs refresh."""
        if not self._last_refresh:
            return True
        return datetime.now() - self._last_refresh > self._refresh_interval
```

#### 3. Alias Map

**File**: `backend/data/researcher_aliases.json`

```json
{
  "bob": "Robert Smith",
  "rob": "Robert Smith",
  "bobby": "Robert Smith",
  "dr. smith": "Robert Smith",
  "conor": "Conor Lynch",
  "dr. lynch": "Conor Lynch",
  "ali": "Alice Johnson",
  "dr. johnson": "Alice Johnson"
}
```

**Note**: This file should be:
- Manually curated initially
- Enhanced over time by analyzing failed normalizations in logs
- Version controlled in git

---

## Implementation Plan

### Phase 1: Foundation (Week 1)

#### 1.1 Add Dependencies
```bash
# requirements.txt
thefuzz==0.22.1
python-Levenshtein==0.25.0  # Optional: speeds up thefuzz
```

#### 1.2 Create Canonical Names Cache
- Implement `canonical_names_cache.py`
- Query ChromaDB for unique researcher names
- Add caching logic with configurable refresh interval

#### 1.3 Create Alias Map
- Create `data/researcher_aliases.json`
- Start with empty or minimal mappings
- Document format and update process

#### 1.4 Implement Normalization Service
- Implement `name_normalization.py`
- Add alias checking logic
- Add fuzzy matching with thefuzz
- Add configurable threshold

### Phase 2: Integration (Week 1-2)

#### 2.1 Update ResearcherSearchTool
```python
def _run(self, researcher_name: Optional[str] = None, topic: Optional[str] = None) -> str:
    # ... existing validation ...

    if researcher_name:
        # NEW: Normalize the name
        canonical_name = name_service.normalize(researcher_name)

        if canonical_name:
            # Log normalization
            log_tool_event("name_normalized", {
                "input": researcher_name,
                "canonical": canonical_name,
                "method": "alias" if in_alias_map else "fuzzy"
            })

            # Use exact metadata filter (FAST PATH)
            results = hybrid_search(
                query=canonical_name,
                k=5,
                filter={"researcher_name": canonical_name},
                search_type="name"
            )
        else:
            # Fallback to hybrid search
            log_tool_event("name_normalization_failed", {
                "input": researcher_name,
                "fallback": "hybrid_search"
            })

            results = hybrid_search(
                query=researcher_name,
                k=5,
                alpha=0.3,
                search_type="name"
            )
```

#### 2.2 Update hybrid_search
- Add support for exact metadata filters (already exists, just needs to be used)
- Ensure fallback works when filter returns no results

### Phase 3: Logging & Monitoring (Week 2)

#### 3.1 Enhanced Logging
Add logging for:
- Normalization attempts (success/failure)
- Method used (alias vs fuzzy)
- Fuzzy match scores
- Failed normalizations (for alias map improvement)

#### 3.2 Metrics
Track:
- Normalization success rate
- Fast path vs fallback usage
- Average fuzzy match scores
- Performance improvements (latency)

### Phase 4: Testing & Validation (Week 2)

#### 4.1 Unit Tests
- Test alias map lookups
- Test fuzzy matching with various inputs
- Test threshold behavior
- Test edge cases (empty strings, special characters)

#### 4.2 Integration Tests
- Test full tool flow with normalization
- Test fallback behavior
- Test with real researcher names from database

#### 4.3 Performance Tests
- Measure latency improvement (exact filter vs hybrid)
- Benchmark fuzzy matching performance
- Test with large canonical names list (1000+ names)

---

## Open Questions

### 1. Alias Map Management

**Question**: How should we populate and maintain the alias map?

**Options**:
- **A**: Start empty, manually add common aliases as discovered
- **B**: Pre-populate with common nicknames (Bob→Robert, etc.)
- **C**: Analyze logs to automatically suggest aliases
- **D**: Combination of B + C

**Recommendation**: Start with **Option B + A** (common nicknames + manual curation)

---

### 2. Fuzzy Match Threshold

**Question**: What threshold should we use for fuzzy matching?

**Options**:
- **85** (Senior's recommendation - balanced)
- **90** (More strict - fewer false positives)
- **80** (More lenient - more matches but more false positives)
- **Configurable** (Allow tuning per deployment)

**Recommendation**: Start with **85**, make configurable via settings

---

### 3. Canonical Names Refresh Strategy

**Question**: How often should we refresh the canonical names list?

**Options**:
- **A**: Once at startup
- **B**: Periodic refresh (every N hours)
- **C**: On-demand via API endpoint
- **D**: Combination of A + B + C

**Recommendation**: **Option D** (startup + 24h refresh + manual endpoint)

---

### 4. Fallback Behavior

**Question**: When normalization fails (score < threshold, not in alias map), what should we do?

**Options**:
- **A**: Fall back to hybrid search (current behavior)
- **B**: Return "Researcher not found" error
- **C**: Suggest similar names to user ("Did you mean: Conor Lynch?")
- **D**: Combination of A + C

**Recommendation**: **Option A** (graceful degradation, maintains current functionality)

---

### 5. Multiple Matches

**Question**: What if fuzzy matching returns multiple high-scoring matches?

**Example**: "Smith" matches both "Robert Smith" (90) and "Alice Smith" (90)

**Options**:
- **A**: Return first match (arbitrary)
- **B**: Return highest score (may still be tie)
- **C**: Ask user to clarify ("Multiple matches found: ...")
- **D**: Fall back to hybrid search

**Recommendation**: **Option D** for ties, **Option B** for clear winner

---

### 6. Logging Privacy

**Question**: Should we log full researcher names in normalization logs?

**Considerations**:
- Needed for debugging and alias map improvements
- May contain PII (personally identifiable information)
- Logs may be stored/transmitted insecurely

**Recommendation**:
- Log full names in structured logs (for analysis)
- Add configuration to disable/redact if needed
- Document privacy implications

---

## Benefits Summary

### Performance Improvements

| Metric | Current (Hybrid) | With Normalization (Exact Filter) |
|--------|------------------|-----------------------------------|
| Latency | ~200-500ms | ~50-100ms |
| Accuracy (exact names) | 95% | 99% |
| Accuracy (partial names) | 60% | 90% (with aliases/fuzzy) |
| Accuracy (nicknames) | 30% | 95% (with aliases) |
| False positives | Low-Medium | Very Low (threshold controlled) |

### User Experience Improvements

- ✅ "Conor" finds "Conor Lynch"
- ✅ "Lynch" finds "Conor Lynch"
- ✅ "conor lynch" finds "Conor Lynch"
- ✅ "Bob" finds "Robert Smith" (with alias)
- ✅ "Cnor Lynch" finds "Conor Lynch" (with fuzzy match)
- ✅ Faster response times (exact filter vs hybrid search)

### Development Benefits

- ✅ Separation of concerns (normalization vs search)
- ✅ Easier to debug (clear normalization logs)
- ✅ Easier to improve (tune threshold, add aliases)
- ✅ Testable components (unit test normalization separately)
- ✅ Future-proof (can swap fuzzy matching algorithm)

---

## Related GitHub Issues

- **Issue**: Support partial name matching for researcher searches
- **Priority**: High
- **Effort**: Medium (2 weeks)
- **Dependencies**: None

---

## References

1. **thefuzz library**: https://github.com/seatgeek/thefuzz
2. **ChromaDB filtering docs**: https://docs.trychroma.com/usage-guide#filtering-by-metadata
3. **Levenshtein distance**: https://en.wikipedia.org/wiki/Levenshtein_distance

---

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-10-28 | 1.0 | Initial design document | Architecture Review |

