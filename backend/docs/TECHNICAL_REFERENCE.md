# Backend Technical Reference

This document provides a technical overview of the Moffitt Agentic RAG backend: code layout, key components, runtime configuration, and operational workflows. It complements `backend/README.md` (quick start and endpoints).

## Architecture Overview

- FastAPI application entrypoint at `backend/main.py` wires CORS, health/settings routes, and includes API routers under `app/api`.
- Retrieval stack built on Chroma (LangChain) with hybrid search combining vector similarity and keyword scoring (`app/services/hybrid_search.py`).
- Agentic orchestration via LangChain ReAct agent and custom tools (`app/services/agent.py`, `app/services/tools.py`).
- Name normalization layer (aliases + fuzzy) for robust researcher name lookup (`app/services/name_normalization.py`) using data in `backend/data/researcher_aliases.json`.
- Centralized structured logging utilities in `app/utils/logging.py`.

## Directory Layout

- `backend/main.py` — FastAPI app construction and router inclusion.
- `backend/README.md` — Quickstart, endpoints, Docker usage.
- `backend/NAME_NORMALIZATION_DESIGN.md` — Design doc for name normalization.
- `backend/TECHNICAL_REFERENCE.md` — This file.
- `backend/data/researcher_aliases.json` — Alias map generated from processed data.
- `backend/requirements.txt` — Backend deps (UTF‑16 encoded). See also `requirements.normalization.txt` for fuzzy matching packages.
- `backend/app/`
  - `api/` — FastAPI routers and dependencies
    - `endpoints/admin.py` — Admin endpoints (rebuild, stats).
    - `endpoints/query.py` — Query processing endpoints (agent).
    - `endpoints/researchers.py` — Researcher listing/details.
  - `core/`
    - `config.py` — Environment-driven settings (CORS, model/provider, data dirs).
    - `prompts.py` — System and agent prompt templates.
    - `security.py` — API key dependency for protected endpoints.
  - `models/`
    - `query.py` — Pydantic models: QueryRequest/Response/Status.
    - `researcher.py` — Researcher profile models.
  - `services/`
    - `agent.py` — Creates ReAct agent, processes queries, tracks status.
    - `tools.py` — LangChain tools: ResearcherSearch, DepartmentFilter, ProgramFilter.
    - `hybrid_search.py` — Hybrid scoring (semantic + keyword), metadata filter support.
    - `vector_db.py` — Chroma DB load/search, embeddings, rebuild scaffolding.
    - `llm.py` — LLM accessors for providers, text generation helpers.
    - `limited_call.py` — Guardrail wrapper to limit LLM call count per query.
    - `name_normalization.py` — Alias-first then fuzzy normalization service.
  - `utils/`
    - `logging.py` — JSON-structured logging and event helpers.

## Configuration

Settings are defined in `app/core/config.py` (loaded via `.env`). Key variables:

- `API_HOST`, `API_PORT` — Serve host/port.
- `API_KEY`, `API_KEY_NAME` — API key protection for selected routes.
- `CORS_ORIGINS` — Allowed origins for frontend.
- `VECTOR_DB_DIR` — Persisted Chroma directory.
- `PROCESSED_DATA_DIR` — Source of processed researcher JSON files.
- `COLLECTION_NAME` — Chroma collection name.
- `EMBEDDING_MODEL_NAME` — HuggingFace sentence embeddings model.
- LLM provider and models:
  - `LLM_PROVIDER` — `openai`, `groq`, `euron`, or local `ollama`.
  - `OPENAI_API_KEY`, `OPENAI_MODEL`
  - `GROQ_API_KEY`, `GROQ_MODEL`
  - `EURON_API_KEY`, `EURON_MODEL`
  - `OLLAMA_BASE_URL`

## Data and Vector Store

- Embeddings: `HuggingFaceEmbeddings` with `EMBEDDING_MODEL_NAME`.
- Vector store: `Chroma` persisted under `VECTOR_DB_DIR` (`app/services/vector_db.py`).
- Search APIs:
  - `similarity_search(query, k, filter)` — Simple vector retrieval.
  - `hybrid_search(query, k, alpha, filter, search_type)` — Combines vector results with keyword scoring; supports metadata filter for exact matches (e.g., `{"researcher_name": "Conor Lynch"}`).
- Rebuild: `rebuild_vector_database(...)` scaffolding exists (simulates progress). Admin endpoints may trigger this.

## Agent and Tools

- Agent: `create_researcher_agent()` builds a ReAct agent over tools with prompt from `core/prompts.py`. Execution limited by `limited_call.py`.
- Tooling (`services/tools.py`):
  - `ResearcherSearch` — Handles name or topic queries. For name queries:
    1) Normalize via `NameNormalizationService`.
    2) If canonical found, use exact metadata filter on canonical name (fast path).
    3) Else, fallback to hybrid search with `alpha=0.3`.
  - `DepartmentFilter`, `ProgramFilter` — Filter by metadata via helper services.

## Name Normalization

- Alias map at `backend/data/researcher_aliases.json` generated from filenames under `data/processed` (script lives at project `scripts/generate_aliases.py`).
- Service: `NameNormalizationService`
  - Loads alias map; keys are lowercase variants; values are Title Case canonical names.
  - Alias-first exact match; then fuzzy match against canonical names (if `thefuzz` installed).
  - Threshold default 85 (configurable via service constructor); returns method and score for logging.
- Dependencies: install `thefuzz` and optional `python-Levenshtein` (see `backend/requirements.normalization.txt`).

## Logging and Observability

- Use `app/utils/logging.py` helpers: `get_logger`, `log_tool_event`, `log_search_event`, `log_error_event`.
- Tools and hybrid search emit structured events for inputs, parameters, normalization outcomes, and results.
- Agent logs original user queries and interpretations for traceability.

## Running Locally

- Environment: copy `.env.example` to `.env` and set provider API keys.
- Install deps (note `requirements.txt` is UTF‑16):
  - `pip install -r backend/requirements.txt`
  - `pip install -r backend/requirements.normalization.txt` (for fuzzy matching)
- Start server:
  - `uvicorn main:app --reload` from `backend/`.
- Docs: `http://localhost:8000/docs`.

## Adding/Refreshing Aliases

- Source files: `data/processed/*.json` named `first-last.json` (may have multi-token last names).
- Generate alias map: `python scripts/generate_aliases.py` from repo root.
- Variants included:
  - Full name; `Dr.` prefixed; `PhD`/`Ph.D.` suffixed.
  - Last-name-only when unique; first-name-only when unique.
  - Keys are lowercase without extra punctuation; values are canonical names.

## Common Tasks

- Search by name (exact or partial): relies on normalization, then exact filter or hybrid fallback.
- Search by topic: set `alpha=0.7` in hybrid search via `ResearcherSearch`.
- Update embeddings/model: change `EMBEDDING_MODEL_NAME` in `.env`.
- Change LLM provider: update `LLM_PROVIDER` and provider-specific keys/models.

## Notes and Caveats

- `backend/requirements.txt` is UTF‑16 encoded; prefer adding new packages via a separate fragment or convert to UTF‑8 carefully.
- Chroma metadata filtering supports equality only. Partial metadata matches require the normalization + exact filter flow or hybrid fallback.
- Rebuild function in `vector_db.py` is a stub to simulate progress; plug in actual chunking/ingest when ready.

