# LangSmith Integration Plan

**Project**: Moffitt Agentic RAG System
**Date**: October 28, 2025
**Status**: Ready for Implementation

---

## Table of Contents

1. [Overview](#overview)
2. [Why LangSmith?](#why-langsmith)
3. [Current State](#current-state)
4. [Implementation Phases](#implementation-phases)
5. [Configuration Changes](#configuration-changes)
6. [Code Changes](#code-changes)
7. [Testing & Verification](#testing--verification)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## Overview

### What is LangSmith?

LangSmith is Anthropic's observability and evaluation platform for LangChain applications. It provides:

- **Visual Trace Exploration**: See every step of agent execution
- **LLM Call Monitoring**: Track all model calls, tokens, and costs
- **Performance Metrics**: Latency, success rates, and error tracking
- **Debugging Tools**: Inspect inputs, outputs, and intermediate steps
- **Evaluation Framework**: Test and compare agent performance
- **Dataset Management**: Store test cases and evaluation datasets

### Why Add LangSmith to This Project?

The Moffitt Agentic RAG system is perfect for LangSmith because it:

1. **Uses LangChain Agents**: ReAct agents with multiple tool calls
2. **Has Complex Workflows**: Multi-step reasoning with vector search
3. **Needs Monitoring**: Track researcher queries and response quality
4. **Benefits from Debugging**: Understand why certain researchers are/aren't found
5. **Can Use Evaluation**: Already has `data/qna_seed.csv` for test cases

---

## Current State

### ✅ Already Installed
```
langsmith==0.4.34  # Already in requirements.txt (line 63)
```

### ✅ LangChain Stack Present
```
langchain==0.3.27
langchain-core==0.3.79
langchain-groq==0.3.8
langchain-openai==0.3.35
langchain-chroma==0.2.6
```

### ❌ Not Configured
- No LangSmith environment variables in `.env.example`
- No initialization code in application startup
- No tracing metadata added to agent calls
- LangSmith callbacks not enabled

### 📊 Available Data
- **119 researcher profiles** in vector database
- **qna_seed.csv** with 40+ Q&A pairs (perfect for evaluation datasets)
- **Structured logging** already in place (`backend/app/utils/logging.py`)

---

## Implementation Phases

### Phase 1: Basic Configuration (30 minutes)
**Goal**: Get LangSmith tracing working

**Tasks**:
1. Add environment variables
2. Update config files
3. Initialize LangSmith on app startup
4. Test basic tracing

### Phase 2: Enhanced Tracing (1 hour)
**Goal**: Add rich metadata to traces

**Tasks**:
1. Add project/run metadata
2. Tag runs by query type
3. Add custom metadata (researcher names, search results)
4. Enable automatic error capture

### Phase 3: Evaluation Setup (1 hour)
**Goal**: Create evaluation datasets

**Tasks**:
1. Convert qna_seed.csv to LangSmith dataset
2. Create evaluation function
3. Run baseline evaluation
4. Set up automated evaluation runs

### Phase 4: Production Features (Optional)
**Goal**: Advanced monitoring and feedback

**Tasks**:
1. Cost tracking per query
2. User feedback collection
3. Performance dashboards
4. Alert configuration

---

## Configuration Changes

### 1. Environment Variables

#### Add to `.env.example`:

```bash
# ============================================
# LangSmith Configuration (Optional)
# ============================================
# Get your API key at: https://smith.langchain.com/settings

# Enable/Disable LangSmith
LANGCHAIN_TRACING_V2=false

# LangSmith API Key
LANGCHAIN_API_KEY=your_langsmith_api_key_here

# LangSmith Project Name
LANGCHAIN_PROJECT=moffitt-agentic-rag-dev

# LangSmith Endpoint (default)
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

#### Add to `backend/.env.example`:

Same as above.

#### Add to `.env.docker.example`:

Same as above.

### 2. Backend Configuration

#### Update `backend/app/core/config.py`:

```python
# Add to Settings class (around line 60):

    # LangSmith settings
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "moffitt-agentic-rag")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
```

---

## Code Changes

### 1. Initialize LangSmith on Startup

#### Update `backend/main.py`:

```python
# Add near the top (after imports):
import os
from app.core.config import settings

# Add after app initialization, before route setup:

# Initialize LangSmith if enabled
if settings.LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY or ""
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT

    logger.info(f"✓ LangSmith tracing enabled for project: {settings.LANGCHAIN_PROJECT}")
else:
    logger.info("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")
```

### 2. Add Metadata to Agent Calls

#### Update `backend/app/services/agent.py`:

Add rich metadata to agent invocations:

```python
# In process_query function (around line 104):

async def process_query(
    query_id: str,
    query: str,
    query_type: str = "general",
    streaming: bool = False,
    max_results: int = 5,
) -> QueryResponse:
    """Process a query using the Agentic RAG system."""

    try:
        # ... existing code ...

        # Create agent
        agent = create_researcher_agent(temperature=0.7, max_llm_calls=6)

        # Add metadata for LangSmith tracing
        metadata = {
            "query_id": query_id,
            "query_type": query_type,
            "max_results": max_results,
            "streaming": streaming,
            "user_query": query[:100],  # First 100 chars
        }

        # Add tags for better filtering in LangSmith
        tags = [
            "moffitt-rag",
            f"query-type:{query_type}",
            f"query-id:{query_id[:8]}",  # Short ID for easy filtering
        ]

        # Invoke with metadata and tags
        result = agent.invoke(
            {"input": query},
            config={
                "metadata": metadata,
                "tags": tags,
                "run_name": f"Query: {query[:50]}..."  # Readable name in UI
            }
        )

        # ... rest of existing code ...

    except Exception as e:
        # ... existing error handling ...
```

### 3. Add LangSmith Utility Module (Optional)

Create `backend/app/utils/langsmith.py`:

```python
"""
LangSmith utilities for the Moffitt Agentic RAG system.

This module provides helper functions for LangSmith integration,
including metadata enrichment and custom callbacks.
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from langsmith import Client
from app.core.config import settings


def is_langsmith_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    return settings.LANGCHAIN_TRACING_V2


def get_langsmith_client() -> Optional[Client]:
    """Get LangSmith client if tracing is enabled."""
    if not is_langsmith_enabled():
        return None

    return Client(
        api_key=settings.LANGCHAIN_API_KEY,
        api_url=settings.LANGCHAIN_ENDPOINT
    )


def create_run_metadata(
    query_id: str,
    query: str,
    query_type: str,
    user_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create standardized metadata for LangSmith runs.

    Args:
        query_id: Unique query identifier
        query: User query text
        query_type: Type of query (general, researcher, etc.)
        user_id: Optional user identifier
        **kwargs: Additional metadata fields

    Returns:
        Dictionary of metadata
    """
    metadata = {
        "query_id": query_id,
        "query_type": query_type,
        "query_preview": query[:100],
        "query_length": len(query),
        "timestamp": datetime.now().isoformat(),
        "system": "moffitt-agentic-rag",
        "version": "1.0.0",
    }

    if user_id:
        metadata["user_id"] = user_id

    # Add any additional metadata
    metadata.update(kwargs)

    return metadata


def create_run_tags(
    query_type: str,
    query_id: str,
    additional_tags: Optional[List[str]] = None
) -> List[str]:
    """
    Create standardized tags for LangSmith runs.

    Args:
        query_type: Type of query
        query_id: Query identifier
        additional_tags: Optional additional tags

    Returns:
        List of tags
    """
    tags = [
        "moffitt-rag",
        f"query-type:{query_type}",
        f"qid:{query_id[:8]}",
    ]

    if additional_tags:
        tags.extend(additional_tags)

    return tags


def add_researcher_results_to_metadata(
    metadata: Dict[str, Any],
    results: List[Any]
) -> Dict[str, Any]:
    """
    Add researcher search results to run metadata.

    Args:
        metadata: Existing metadata dictionary
        results: List of search results

    Returns:
        Updated metadata dictionary
    """
    metadata["result_count"] = len(results)

    if results:
        # Add first few researcher names for quick reference
        researcher_names = []
        for result in results[:5]:
            if hasattr(result, 'metadata') and 'researcher_name' in result.metadata:
                researcher_names.append(result.metadata['researcher_name'])

        if researcher_names:
            metadata["researchers_found"] = researcher_names

    return metadata
```

---

## Testing & Verification

### Step 1: Get LangSmith API Key

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Sign up or log in
3. Go to Settings → API Keys
4. Create a new API key
5. Copy the key

### Step 2: Configure Environment

```bash
# Add to your .env file
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_xxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=moffitt-agentic-rag-dev
```

### Step 3: Start the Backend

```bash
cd backend
uvicorn main:app --reload
```

Look for the log message:
```
✓ LangSmith tracing enabled for project: moffitt-agentic-rag-dev
```

### Step 4: Run Test Query

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev_api_key" \
  -d '{
    "query": "Who studies cancer evolution?",
    "query_type": "researcher"
  }'
```

### Step 5: View in LangSmith

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Select your project: `moffitt-agentic-rag-dev`
3. You should see the trace with:
   - Agent execution steps
   - Tool calls (researcher search, etc.)
   - LLM invocations
   - Inputs and outputs
   - Latency metrics

### What You Should See

```
Trace: Query: Who studies cancer evolution?
├── Agent: create_researcher_agent
│   ├── LLM: llama-3.3-70b-versatile (Thought)
│   ├── Tool: researcher_search
│   │   └── Vector Search: similarity_search
│   ├── LLM: llama-3.3-70b-versatile (Action)
│   └── LLM: llama-3.3-70b-versatile (Final Answer)
```

---

## Advanced Features

### 1. Create Evaluation Dataset from qna_seed.csv

```python
# Script: backend/scripts/create_langsmith_dataset.py

import csv
from pathlib import Path
from langsmith import Client

def create_evaluation_dataset():
    """Create LangSmith dataset from qna_seed.csv"""

    client = Client()
    dataset_name = "moffitt-researcher-qa"

    # Create dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Q&A pairs for evaluating researcher search quality"
    )

    # Load QA pairs from CSV
    csv_path = Path(__file__).parent.parent.parent / "data" / "qna_seed.csv"

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            question = row['question']
            answer = row['answer']

            # Add to dataset
            client.create_example(
                dataset_id=dataset.id,
                inputs={"query": question},
                outputs={"answer": answer}
            )

    print(f"✓ Created dataset '{dataset_name}' with {len(list(reader))} examples")

if __name__ == "__main__":
    create_evaluation_dataset()
```

### 2. Run Evaluation

```python
# Script: backend/scripts/run_evaluation.py

from langsmith import Client
from langsmith.evaluation import evaluate
from app.services.agent import process_query

async def evaluate_query(inputs: dict) -> dict:
    """Evaluation function"""
    response = await process_query(
        query_id="eval",
        query=inputs["query"],
        query_type="researcher"
    )
    return {"answer": response.answer}

def run_evaluation():
    """Run evaluation on the dataset"""

    client = Client()

    # Run evaluation
    results = evaluate(
        evaluate_query,
        data="moffitt-researcher-qa",
        evaluators=[
            # Add custom evaluators here
        ],
        experiment_prefix="moffitt-rag-eval"
    )

    print(f"✓ Evaluation complete: {results}")

if __name__ == "__main__":
    run_evaluation()
```

### 3. Add User Feedback

```python
# In backend/app/api/endpoints/query.py

from langsmith import Client

@router.post("/query/{query_id}/feedback")
async def submit_feedback(
    query_id: str,
    feedback: dict,
    api_key: str = Depends(verify_api_key)
):
    """Submit user feedback for a query"""

    if settings.LANGCHAIN_TRACING_V2:
        client = Client()

        # Find the run by query_id
        runs = client.list_runs(
            filter=f'metadata.query_id = "{query_id}"',
            limit=1
        )

        for run in runs:
            # Add feedback
            client.create_feedback(
                run_id=run.id,
                key="user_rating",
                score=feedback.get("rating", 0),
                comment=feedback.get("comment", "")
            )

            return {"message": "Feedback submitted"}

    return {"message": "LangSmith not enabled"}
```

### 4. Cost Tracking

```python
# Get cost information from runs
def get_query_cost(query_id: str) -> float:
    """Get cost for a specific query"""

    client = Client()

    runs = client.list_runs(
        filter=f'metadata.query_id = "{query_id}"',
        limit=1
    )

    total_cost = 0.0
    for run in runs:
        if run.total_cost:
            total_cost += run.total_cost

    return total_cost
```

---

## Troubleshooting

### Issue: Traces Not Appearing

**Solution**:
1. Check environment variables are set correctly
2. Verify API key is valid
3. Check console logs for LangSmith errors
4. Ensure `LANGCHAIN_TRACING_V2=true`

### Issue: Missing Metadata

**Solution**:
- Metadata must be passed in `config` parameter to agent.invoke()
- Use the format shown in Code Changes section

### Issue: Slow Performance

**Solution**:
- LangSmith adds minimal overhead (~50-100ms)
- If concerned, disable in production: `LANGCHAIN_TRACING_V2=false`
- Or use sampling: only trace 10% of queries

### Issue: API Rate Limits

**Solution**:
- Free tier: 5000 traces/month
- Upgrade plan if needed
- Use sampling for high-volume production

---

## Benefits Summary

### For Development
- ✅ **Visual Debugging**: See exactly what the agent is doing
- ✅ **Error Tracking**: Understand failures quickly
- ✅ **Performance Monitoring**: Identify slow steps

### For Production
- ✅ **Quality Monitoring**: Track response quality over time
- ✅ **Cost Analysis**: Monitor LLM costs per query
- ✅ **User Feedback**: Collect and analyze user ratings

### For Evaluation
- ✅ **Dataset Management**: Store test cases in one place
- ✅ **Automated Testing**: Run evaluations on every change
- ✅ **A/B Testing**: Compare different prompts/models

---

## Implementation Checklist

### Phase 1: Basic Setup
- [ ] Get LangSmith API key from smith.langchain.com
- [ ] Add environment variables to `.env.example` files
- [ ] Update `backend/app/core/config.py` with LangSmith settings
- [ ] Add initialization code to `backend/main.py`
- [ ] Update `backend/app/services/agent.py` with metadata
- [ ] Test with a sample query
- [ ] Verify traces appear in LangSmith dashboard

### Phase 2: Enhanced Tracing
- [ ] Create `backend/app/utils/langsmith.py` utility module
- [ ] Add standardized metadata to all agent calls
- [ ] Add tags for query types
- [ ] Add researcher results to metadata
- [ ] Test different query types

### Phase 3: Evaluation
- [ ] Create `backend/scripts/create_langsmith_dataset.py`
- [ ] Convert qna_seed.csv to LangSmith dataset
- [ ] Create evaluation script
- [ ] Run baseline evaluation
- [ ] Document evaluation process

### Phase 4: Production (Optional)
- [ ] Add user feedback endpoint
- [ ] Create cost tracking dashboard
- [ ] Set up alerts for errors/performance
- [ ] Configure sampling for high volume

---

## References

### Documentation
- [LangSmith Docs](https://docs.smith.langchain.com/)
- [LangChain Tracing](https://python.langchain.com/docs/langsmith/walkthrough)
- [Evaluation Guide](https://docs.smith.langchain.com/evaluation)

### API Keys
- [LangSmith Dashboard](https://smith.langchain.com)
- [API Key Settings](https://smith.langchain.com/settings)

### Support
- [LangSmith Discord](https://discord.gg/langchain)
- [GitHub Issues](https://github.com/langchain-ai/langsmith-sdk)

---

## Estimated Timeline

| Phase | Time | Description |
|-------|------|-------------|
| **Phase 1** | 30 min | Basic configuration and first traces |
| **Phase 2** | 1 hour | Enhanced metadata and tags |
| **Phase 3** | 1 hour | Evaluation dataset and runs |
| **Phase 4** | 2 hours | Advanced features (optional) |
| **Total** | 2-4.5 hours | Depends on features needed |

---

## Next Steps

1. **Immediate**: Get LangSmith API key and add to .env
2. **Quick Win**: Complete Phase 1 (basic tracing) - 30 minutes
3. **High Value**: Complete Phase 2 (rich metadata) - 1 hour
4. **Long Term**: Set up evaluation framework - 1 hour

Start with Phase 1 to see immediate value, then expand based on needs!

---

*Document created: October 28, 2025*
*Ready for implementation*
