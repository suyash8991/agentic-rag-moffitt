# QNA Seed Dataset - 50 Questions for Moffitt RAG System

## Overview

This CSV file contains 50 carefully curated questions designed to comprehensively test the Moffitt Agentic RAG system's ability to retrieve and synthesize information about cancer researchers.

## File Location

`data/qna_seed.csv`

## File Format

```csv
question,answer,category,difficulty
```

### Fields

- **question**: The query to be answered by the RAG system
- **answer**: Expected answer/guidance for evaluation
- **category**: Classification of question type
- **difficulty**: Complexity level (easy, medium, hard)

## Question Categories (50 total)

### 1. Research Interests (18 questions)
Questions about researchers' focus areas, methodologies, and research approaches.

**Examples:**
- "What are Alexander Anderson's research interests?"
- "What is Matthew Schabath's expertise in quantitative imaging?"
- "What is Keiran Smalley's work on PROTAC technology?"

### 2. Grants & Funding (9 questions)
Questions about funding sources, grant titles, and research support.

**Examples:**
- "What grants does Eric Haura have from the National Cancer Institute?"
- "Which researchers have grants from the Department of Defense?"
- "What collaborative grants involve multiple Moffitt researchers?"

### 3. Publications (5 questions)
Questions about research output and scholarly contributions.

**Examples:**
- "What recent publications has Alexander Anderson authored?"
- "Who has published in Nature Medicine?"
- "Which researchers published in 2025?"

### 4. Education & Training (5 questions)
Questions about educational background and academic credentials.

**Examples:**
- "Where did Matthew Schabath receive his PhD?"
- "Which researchers have both MD and PhD degrees?"
- "Who trained at Duke University?"

### 5. Cancer Type Specialization (5 questions)
Questions about specific cancer research focus.

**Examples:**
- "Who studies HPV-related cancers?"
- "Which researchers focus on prostate cancer?"
- "Who works on brain metastases?"

### 6. Associations & Centers (5 questions)
Questions about affiliations with programs and centers.

**Examples:**
- "Who is associated with the Melanoma & Skin Cancer Center of Excellence?"
- "What is the Center for Immunization & Infection Research in Cancer?"
- "What is the focus of the Molecular Medicine Program?"

### 7. Department & Program (3 questions)
Questions about organizational structure.

**Examples:**
- "Which researchers work in the Integrated Mathematical Oncology department?"
- "Who works in the Drug Discovery department?"

### 8. Cross-Cutting/Collaborative (2 questions)
Questions requiring synthesis across multiple researchers.

**Examples:**
- "Which researchers collaborate on evolutionary cancer research?"
- "What international collaborations does Anna Giuliano have?"

## Difficulty Distribution

### Easy (~12 questions)
Direct lookup questions requiring simple information retrieval.
- Educational background
- Department affiliations
- Basic research focus

### Medium (~23 questions)
Questions requiring moderate synthesis or filtering.
- Identifying researchers by topic
- Grant sources
- Publication venues

### Hard (~15 questions)
Complex questions requiring cross-referencing and deep synthesis.
- Multi-researcher collaborations
- Detailed methodology questions
- Comparative analyses
- Grant portfolios across researchers

## Researchers Featured

The questions primarily focus on these researchers (with diverse departments/specialties):

1. **Alexander Anderson** - Integrated Mathematical Oncology
2. **Joel Brown** - Integrated Mathematical Oncology
3. **Eric Haura** - Thoracic Oncology / Drug Discovery
4. **Anna Giuliano** - Cancer Epidemiology
5. **Matthew Schabath** - Cancer Epidemiology / Thoracic Oncology
6. **Keiran Smalley** - Tumor Microenvironment and Metastasis

Also referenced: Ahmad Tarhini, Alberto Chiappori, Robert Gatenby, Andriy Marusyk

## Question Types

### Direct Queries (30%)
"What are [Researcher]'s research interests?"

### Topic-Based Queries (25%)
"Which researchers work on [topic/cancer type]?"

### Comparative Queries (20%)
"Who has [specific characteristic across multiple researchers]?"

### Multi-Hop Reasoning (25%)
"What collaborative grants involve [multiple conditions]?"

## Usage

### For Evaluation
Use this dataset to:
1. Test RAG retrieval accuracy
2. Measure answer quality and completeness
3. Assess handling of varying difficulty levels
4. Evaluate cross-document synthesis capabilities

### For LangSmith Dataset Creation
Run: `python backend/scripts/create_langsmith_dataset.py`

This will upload the questions to LangSmith for evaluation tracking.

### For RAG vs LLM Comparison
Run: `python scripts/compare_rag_llm_answers.py`

This will generate an Excel file comparing RAG answers with direct LLM responses.

## Quality Considerations

- All questions are answerable from the processed researcher JSON files
- Answers include expected information types rather than exact text
- Questions test diverse RAG capabilities: retrieval, filtering, ranking, synthesis
- Difficulty levels allow performance assessment across query complexity

## Future Enhancements

Consider adding:
- More cross-researcher collaboration questions
- Temporal queries (recent vs. historical)
- Negation queries ("Who does NOT work on...")
- Quantitative questions ("How many researchers...")
- Comparison questions ("Compare research approaches of...")

---

**Created:** 2025-10-30
**Format Version:** 1.0
**Total Questions:** 50
