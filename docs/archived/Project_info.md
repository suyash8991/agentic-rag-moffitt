🧠 Agentic RAG System for Moffitt Cancer Center Research Data

Version: 1.0  Date: October 2025  Lead Engineer: Tejas Kumar Leelavathi

1. Project Summary

Moffitt Cancer Center maintains a large, publicly accessible catalog of researcher profiles across multiple programs and departments. Each page contains valuable information—biographies, research focus areas, publications, and contact details—but the information is spread across hundreds of static HTML pages.

The goal of this project is to build an Agentic Retrieval-Augmented Generation (RAG) system that can automatically gather, organize, and reason over these researcher profiles. The system will enable natural-language interaction such as:

“Who in BioEngineering studies cancer evolution?”
“Show researchers collaborating between Biostatistics and Cancer Epidemiology.”

2. Phase 1 — Data Understanding & Scraping

Before building any intelligence layer, we must understand and structure the source data.

2.1 Data Source

Website: Moffitt Cancer Center Research Portal

Content Types: Program → Department → Researcher profile pages

Fields of Interest:

Researcher name, title, program, department

Biography / overview

Research interests, publications

Contact info (email, phone, location)

Profile URL
 there could be more details

2.2 Approach

Automated Crawling:
 first lets start with one professors data and see final data how it woudl look and then proceed building for all.
 
Use an asynchronous crawler such as Crawl4AI to fetch all researcher profile URLs in parallel.

Respect robots.txt, apply rate limiting, and maintain HTML snapshots for reproducibility.

Content Extraction & Normalization:

Parse HTML into structured JSON using automatic readability extraction and section detection.

Convert main text blocks to Markdown for clarity and downstream embedding.

Validate records with a schema ensuring required fields (name, program, department).

Data Storage:

Store raw HTML → cleaned Markdown → validated JSONL under version control.

Compute content_hash for change detection and incremental re-crawling.

Output of Phase 1:
A clean, validated dataset of Moffitt researcher profiles, ready for semantic indexing and LLM training.

3. Phase 2 — Agentic AI Design & Reasoning Layer

With structured data in place, the next step is to make the system agentic—capable of understanding what information is needed and autonomously deciding how to get it.

3.1 Goals

Create an intelligent retrieval layer that can:

Parse the user’s intent (“find collaborators”, “list genomics experts”)

Select the appropriate tool or dataset (local vector DB, PubMed API, or web search)

Retrieve, reason, and summarize with citations

3.2 Core Components
Component	Function
Vector Database (Chroma)	Stores researcher text embeddings with program/department metadata
Embedding Model	Converts text into high-dimensional vectors for semantic similarity
Reasoning Agent (Llama-4 Maverick)	Decides retrieval steps, invokes tools, and generates grounded answers
Reflection Mechanism	Evaluates confidence; re-queries or augments if context is missing
Frontend Interface	Chat-like UI for natural queries and cited responses
3.3 Workflow Overview
User Query
   ↓
Agentic Orchestrator (decides search type)
   ↓
Vector Retriever + Metadata Filters
   ↓
(optional) Web or PubMed API tool
   ↓
LLM Synthesizer → Cited Answer

3.4 Expected Capabilities

Semantic researcher search across programs and departments

Cross-disciplinary collaboration discovery

Automatic summarization of departmental expertise

Extensible integration with external scientific databases

4. Milestones
Phase	Focus	Deliverable	Timeline
1. Data Scraping & Cleaning	Crawl & parse all profiles	researchers.jsonl dataset	2 weeks
2. Embedding & Vector DB	Build searchable Chroma index	moffitt_researchers collection	1 week
3. Agentic Layer	Add reasoning and tool routing	Llama-powered orchestrator	3 weeks
4. Demo Interface	Chat-based web UI + citations	FastAPI + React prototype	2 weeks
5. Outcome

By the end of Phase 2, the project will deliver a working prototype of an Agentic RAG Assistant that can:

Interpret open-ended questions about Moffitt researchers,

Retrieve accurate, cited information from scraped data, and

Reason dynamically about what additional information is required.

This will form the foundation for a scalable AI knowledge system that can later include NIH, PubMed, and clinical collaborations, supporting data-driven discovery at Moffitt Cancer Center.