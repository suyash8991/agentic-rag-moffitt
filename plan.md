# Moffitt Cancer Center Researcher Data Project Plan

## Project Overview

This project aims to build an Agentic Retrieval-Augmented Generation (RAG) system for Moffitt Cancer Center researcher data. The system will enable natural-language queries about researchers, their expertise, and collaborations.

## Phase 1: Data Scraping & Structuring (Current)

### 1. Web Scraping System (Completed)
- ✅ Set up project structure
- ✅ Create web scraping framework with Crawl4AI
- ✅ Implement rate limiting and error handling
- ✅ Parse researcher profiles into structured data
- ✅ Store data in raw HTML, Markdown, and JSON formats

### 2. Data Quality & Completeness
- Extract researcher data from all 127 profiles in Excel file
- Validate extracted data against schema
- Fix any parsing issues or edge cases
- Generate statistics on extracted fields (e.g., most common research areas)

### 3. Data Enrichment
- Cross-reference researchers with publications databases
- Identify collaborations between researchers based on co-authored papers
- Extract research topics using NLP techniques

## Phase 2: Retrieval-Augmented Generation System

### 1. Vector Database Setup (Week 1)
- Install and configure Chroma vector database
- Design schema for researcher embeddings
- Create metadata structure for filtering

### 2. Embedding Generation (Week 1-2)
- Select appropriate embedding model
- Generate embeddings for researcher profiles
- Store embeddings in vector database
- Implement efficient retrieval functions

### 3. Agentic Layer Development (Weeks 3-5)
- Build the orchestrator component for understanding queries
- Implement tool selection logic
- Create reasoning component for synthesizing answers
- Develop confidence evaluation and reflection mechanism

### 4. Frontend Interface (Weeks 6-7)
- Design simple chat interface with FastAPI and React
- Implement query processing pipeline
- Add citation display for transparency
- Create visualizations of researcher networks

## Implementation Approach

### Core Technologies
- **Web Scraping**: Crawl4AI, asyncio
- **Data Processing**: Python, pandas
- **Vector Database**: Chroma
- **Embeddings**: Sentence-transformers or OpenAI embeddings
- **Reasoning Agent**: Llama-4 Maverick
- **Frontend**: FastAPI + React

### Key Principles
1. **Respect for Sources**:
   - Rate-limit all requests to Moffitt's servers
   - Respect robots.txt directives
   - Store snapshots to minimize repeat crawling

2. **Data Quality**:
   - Validate all extracted data
   - Track provenance of all information
   - Implement content hashing for change detection

3. **Modular Architecture**:
   - Separate crawling, parsing, and storage components
   - Allow for component upgrades and replacements
   - Enable incremental improvements

4. **User-Focused Results**:
   - Prioritize accuracy in responses
   - Provide citations for all information
   - Support natural language queries

## Milestones & Timeline

| Milestone | Deliverable | Timeline |
|-----------|-------------|----------|
| ✅ Phase 1: Initial Framework | Web scraper & parser | Completed |
| Phase 1: Complete Dataset | researchers.jsonl with 127 profiles | 1 week |
| Phase 2: Vector Database | Searchable Chroma index | 1 week |
| Phase 2: Agentic Layer | Llama-powered orchestrator | 3 weeks |
| Phase 2: Frontend | FastAPI + React prototype | 2 weeks |

## Success Criteria

1. **Data Completeness**: >95% of researcher profiles successfully parsed with all key fields
2. **Query Accuracy**: >90% relevant answers for typical researcher queries
3. **System Performance**: <3 second response time for standard queries
4. **User Satisfaction**: Ability to answer complex questions about researcher expertise and collaborations

## Next Steps

1. Complete the scraping of all researcher profiles
2. Begin setting up the Chroma vector database
3. Generate embeddings for the researcher profiles
4. Develop the agentic orchestration layer