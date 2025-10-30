# RAG vs LLM Answer Comparison Tool

This script generates questions from professor JSON files and compares answers from the RAG system with direct LLM responses.

## Overview

The tool performs the following operations:

1. **Reads Professor Data**: Loads JSON files from `data/processed/` containing professor information
2. **Generates Questions**: Creates diverse questions based on:
   - Research interests
   - Publications
   - Department/Program affiliations
   - Education background
   - Grants and funding
   - Center associations
3. **Gets RAG Answers**: Queries the agentic RAG system for answers
4. **Gets LLM Answers**: Queries the LLM directly (without RAG) for comparison
5. **Gets LLM with Context Answers**: Queries the LLM with full professor data as context (optional)
6. **Creates Excel Report**: Generates a formatted Excel file with side-by-side comparison

## Installation

First, install the required dependency:

```bash
cd backend
pip install openpyxl
```

Or add it to requirements.txt and install:

```bash
echo "openpyxl==3.1.5" >> requirements.txt
pip install -r requirements.txt
```

## Usage

### Basic Usage (Process all professors)

```bash
python scripts/compare_rag_llm_answers.py
```

This will:
- Process all professor JSON files in `data/processed/`
- Generate questions for each professor
- Create an Excel file with timestamp: `data/rag_llm_comparison_YYYYMMDD_HHMMSS.xlsx`

### Process Limited Number of Professors

```bash
python scripts/compare_rag_llm_answers.py --limit 5
```

This processes only the first 5 professors (useful for testing).

### Custom Input/Output Paths

```bash
python scripts/compare_rag_llm_answers.py \
    --input-dir data/processed \
    --output-file results/comparison.xlsx \
    --limit 10
```

### Skip LLM with Context Column

If you want to save time and only compare RAG vs LLM without context:

```bash
python scripts/compare_rag_llm_answers.py --no-context
```

## Command-Line Arguments

- `--input-dir`: Directory containing professor JSON files (default: `data/processed`)
- `--output-file`: Output Excel file path (default: `data/rag_llm_comparison_YYYYMMDD_HHMMSS.xlsx`)
- `--limit`: Maximum number of professors to process (default: all)
- `--no-context`: Skip the "LLM Answer (With Context)" column to save processing time

## Output Format

The Excel file contains the following columns:

| Column | Description |
|--------|-------------|
| Researcher | Professor's name |
| Question | Generated question |
| Category | Question category (research_interests, publications, etc.) |
| Expected Information | What type of information should be in the answer |
| RAG Answer | Answer from the RAG system |
| RAG Time (s) | Time taken by RAG system |
| LLM Answer (No Context) | Direct LLM answer without any context |
| LLM Time (s) | Time taken by direct LLM |
| LLM Answer (With Context) | LLM answer with full professor data as context |
| LLM Context Time (s) | Time taken by LLM with context |

## Question Categories

The script generates questions in the following categories:

1. **research_interests**: Questions about research focus and expertise
2. **publications**: Questions about recent publications and specific works
3. **department**: Questions about departmental affiliation
4. **program**: Questions about primary research programs
5. **education**: Questions about educational background
6. **grants**: Questions about funding and active grants
7. **associations**: Questions about center and program associations

## Example Questions

For a professor like Ahmad Tarhini, the script generates questions like:

- "What are Ahmad Tarhini's research interests?"
- "What is Ahmad Tarhini's main area of research?"
- "What are some recent publications by Ahmad Tarhini?"
- "What department is Ahmad Tarhini affiliated with?"
- "Where did Ahmad Tarhini receive their education?"
- "What grants or funding does Ahmad Tarhini have?"

## Performance Considerations

- Each question requires 2-3 API calls (RAG + LLM + optional LLM with context)
- Processing includes a 0.5-second delay between questions to avoid rate limiting
- For testing, use `--limit` to process fewer professors
- Use `--no-context` to skip the third LLM call and process faster

## Example Output

```
================================================================================
RAG vs LLM Answer Comparison Tool
================================================================================
Input directory: data\processed
Output file: data\rag_llm_comparison_20251030_123456.xlsx
Professor limit: 5
================================================================================
Found 5 professor files to process

[1/5] Processing ahmad-tarhini...
  Researcher: Ahmad Tarhini
  Generated 8 questions
  [1/8] What are Ahmad Tarhini's research interests?
    Getting RAG answer...
    RAG answer received (2.34s)
    Getting LLM answer (no context)...
    LLM answer received (1.23s)
    Getting LLM answer (with context)...
    LLM with context answer received (1.45s)
  ...

Creating Excel file...
Excel file created successfully: data\rag_llm_comparison_20251030_123456.xlsx
Total questions processed: 40
================================================================================
```

## Troubleshooting

### Import Errors

If you encounter import errors, make sure you're running from the project root:

```bash
cd c:\Coding\Projects\moffitt-agentic-rag
python scripts/compare_rag_llm_answers.py
```

### API Key Errors

Ensure your environment variables are set:

```bash
# Check .env file or set environment variables
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### Database Errors

Make sure the vector database is initialized:

```bash
cd backend
python -m app.services.vector_db_builder
```

## Analyzing Results

After generating the Excel file:

1. **Compare Answer Quality**: Look at how well RAG answers match expected information vs direct LLM
2. **Check Response Times**: Compare processing times between RAG and direct LLM
3. **Evaluate Accuracy**: See if RAG provides more specific, grounded information
4. **Identify Gaps**: Find areas where RAG might need improvement

## Tips for Best Results

1. **Start Small**: Use `--limit 2` to test the script first
2. **Check Data Quality**: Ensure professor JSON files are properly formatted
3. **Monitor API Usage**: Be aware of API rate limits and costs
4. **Review Questions**: Examine generated questions to ensure they're relevant
5. **Compare Thoughtfully**: Consider that RAG should provide more specific, grounded answers
