"""
Script to compare RAG-generated answers with direct LLM answers.

This script:
1. Reads professor JSON files from data/processed
2. Generates questions based on professor data
3. Gets answers from both RAG system and direct LLM
4. Creates an Excel file comparing the results
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from datetime import datetime

# Set up correct paths BEFORE importing backend modules
# This ensures the backend uses the correct paths regardless of CWD
script_dir = Path(__file__).parent.parent.resolve()  # Project root
os.environ["VECTOR_DB_DIR"] = str(script_dir / "data" / "vector_db")
os.environ["PROCESSED_DATA_DIR"] = str(script_dir / "data" / "processed")

# Add backend to path
backend_path = script_dir / "backend"
sys.path.insert(0, str(backend_path))

from app.services.agent import process_query
from app.services.llm import generate_text
from app.services.query_status_service import QueryStatusService, InMemoryQueryStatusRepository


class QuestionGenerator:
    """Generates questions based on professor data."""

    @staticmethod
    def generate_questions(professor_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """
        Generate questions from professor data.

        Args:
            professor_data: Professor information from JSON file

        Returns:
            List of tuples: (question, category, expected_info)
        """
        questions = []
        researcher_name = professor_data.get("researcher_name", "Unknown")

        # Research interests questions
        if professor_data.get("research_interests"):
            questions.append((
                f"What are {researcher_name}'s research interests?",
                "research_interests",
                "Research focus areas and expertise"
            ))
            questions.append((
                f"What is {researcher_name}'s main area of research?",
                "research_interests",
                "Primary research focus"
            ))

        # Publications questions
        if professor_data.get("publications"):
            questions.append((
                f"What are some recent publications by {researcher_name}?",
                "publications",
                "Recent publication titles and years"
            ))
            pubs = professor_data.get("publications", [])
            if pubs and len(pubs) > 0:
                first_pub = pubs[0]
                if "title" in first_pub:
                    questions.append((
                        f"Tell me about {researcher_name}'s work on {first_pub['title'][:50]}",
                        "publications",
                        "Specific publication details"
                    ))

        # Department/Program questions
        if professor_data.get("department"):
            questions.append((
                f"What department is {researcher_name} affiliated with?",
                "department",
                "Department and organizational affiliation"
            ))

        if professor_data.get("primary_program"):
            questions.append((
                f"What is {researcher_name}'s primary research program?",
                "program",
                "Primary research program"
            ))

        # Education questions
        if professor_data.get("education"):
            questions.append((
                f"Where did {researcher_name} receive their education?",
                "education",
                "Educational background and degrees"
            ))

        # Grants questions
        if professor_data.get("grants"):
            questions.append((
                f"What grants or funding does {researcher_name} have?",
                "grants",
                "Active grants and funding sources"
            ))

        # Collaboration questions
        if professor_data.get("associations"):
            questions.append((
                f"What centers or programs is {researcher_name} associated with?",
                "associations",
                "Center and program associations"
            ))

        return questions


class AnswerComparator:
    """Compares RAG and LLM answers."""

    def __init__(self):
        # Initialize repository and service with proper dependency injection
        repository = InMemoryQueryStatusRepository()
        self.query_status_service = QueryStatusService(repository)

    async def get_rag_answer(self, question: str) -> Tuple[str, float]:
        """
        Get answer from RAG system.

        Args:
            question: Question to ask

        Returns:
            Tuple of (answer, processing_time)
        """
        import uuid
        import time

        start_time = time.time()
        query_id = str(uuid.uuid4())

        try:
            response = await process_query(
                query_id=query_id,
                query=question,
                query_status_service=self.query_status_service,
                query_type="general",
                streaming=False,
                max_results=5
            )

            processing_time = time.time() - start_time
            return response.answer, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            return f"Error: {str(e)}", processing_time

    async def get_llm_answer(self, question: str, professor_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Get answer from direct LLM (without RAG).

        Args:
            question: Question to ask
            professor_data: Professor data for context

        Returns:
            Tuple of (answer, processing_time)
        """
        import time

        start_time = time.time()

        # Create a prompt with minimal context (just the question)
        prompt = f"""Answer the following question about a researcher at Moffitt Cancer Center:

Question: {question}

Provide a concise and accurate answer based on your general knowledge. If you don't have specific information, say so."""

        try:
            answer = await generate_text(
                prompt=prompt,
                system_prompt="You are a helpful assistant that answers questions about cancer researchers.",
                temperature=0.7
            )

            processing_time = time.time() - start_time
            return answer, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            return f"Error: {str(e)}", processing_time

    async def get_llm_answer_with_context(self, question: str, professor_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Get answer from LLM with full professor context (for comparison).

        Args:
            question: Question to ask
            professor_data: Full professor data as context

        Returns:
            Tuple of (answer, processing_time)
        """
        import time

        start_time = time.time()

        # Create a prompt with the full professor data as context
        context = json.dumps(professor_data, indent=2)

        prompt = f"""Based on the following researcher information, answer the question:

RESEARCHER INFORMATION:
{context}

QUESTION: {question}

Provide a concise and accurate answer based on the information provided."""

        try:
            answer = await generate_text(
                prompt=prompt,
                system_prompt="You are a helpful assistant that answers questions based on provided context.",
                temperature=0.7
            )

            processing_time = time.time() - start_time
            return answer, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            return f"Error: {str(e)}", processing_time


async def process_professors(
    input_dir: Path,
    output_file: Path,
    limit: int = None,
    include_llm_with_context: bool = True
):
    """
    Process professor files and generate comparison Excel.

    Args:
        input_dir: Directory containing professor JSON files
        output_file: Output Excel file path
        limit: Maximum number of professors to process (None for all)
        include_llm_with_context: Whether to include LLM with full context column
    """
    print("=" * 80)
    print("RAG vs LLM Answer Comparison Tool")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Professor limit: {limit if limit else 'All'}")
    print("=" * 80)

    # Get all professor JSON files
    json_files = sorted(list(input_dir.glob("*.json")))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    # Exclude summary.json
    json_files = [f for f in json_files if f.stem != "summary"]

    # Apply limit if specified
    if limit:
        json_files = json_files[:limit]

    print(f"Found {len(json_files)} professor files to process")
    print()

    # Initialize components
    question_generator = QuestionGenerator()
    answer_comparator = AnswerComparator()

    # Collect all results
    results = []

    # Process each professor
    for idx, json_file in enumerate(json_files, 1):
        print(f"[{idx}/{len(json_files)}] Processing {json_file.stem}...")

        try:
            # Load professor data
            with open(json_file, 'r', encoding='utf-8') as f:
                professor_data = json.load(f)

            researcher_name = professor_data.get("researcher_name", json_file.stem)
            print(f"  Researcher: {researcher_name}")

            # Generate questions
            questions = question_generator.generate_questions(professor_data)
            print(f"  Generated {len(questions)} questions")

            # Process each question
            for q_idx, (question, category, expected_info) in enumerate(questions, 1):
                print(f"  [{q_idx}/{len(questions)}] {question}")

                # Get RAG answer
                print(f"    Getting RAG answer...")
                rag_answer, rag_time = await answer_comparator.get_rag_answer(question)
                print(f"    RAG answer received ({rag_time:.2f}s)")

                # Get LLM answer (without context)
                print(f"    Getting LLM answer (no context)...")
                llm_answer, llm_time = await answer_comparator.get_llm_answer(question, professor_data)
                print(f"    LLM answer received ({llm_time:.2f}s)")

                # Get LLM answer with context (optional)
                llm_context_answer = None
                llm_context_time = None
                if include_llm_with_context:
                    print(f"    Getting LLM answer (with context)...")
                    llm_context_answer, llm_context_time = await answer_comparator.get_llm_answer_with_context(
                        question, professor_data
                    )
                    print(f"    LLM with context answer received ({llm_context_time:.2f}s)")

                # Store result
                result = {
                    "Researcher": researcher_name,
                    "Question": question,
                    "Category": category,
                    "Expected Information": expected_info,
                    "RAG Answer": rag_answer,
                    "RAG Time (s)": f"{rag_time:.2f}",
                    "LLM Answer (No Context)": llm_answer,
                    "LLM Time (s)": f"{llm_time:.2f}",
                }

                if include_llm_with_context:
                    result["LLM Answer (With Context)"] = llm_context_answer
                    result["LLM Context Time (s)"] = f"{llm_context_time:.2f}"

                results.append(result)

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)

            print()

        except Exception as e:
            print(f"  Error processing {json_file.stem}: {e}")
            print()
            continue

    # Create DataFrame
    print("Creating Excel file...")
    df = pd.DataFrame(results)

    # Create Excel writer with formatting
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Comparison', index=False)

        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Comparison']

        # Set column widths
        worksheet.column_dimensions['A'].width = 25  # Researcher
        worksheet.column_dimensions['B'].width = 60  # Question
        worksheet.column_dimensions['C'].width = 20  # Category
        worksheet.column_dimensions['D'].width = 30  # Expected Information
        worksheet.column_dimensions['E'].width = 80  # RAG Answer
        worksheet.column_dimensions['F'].width = 15  # RAG Time
        worksheet.column_dimensions['G'].width = 80  # LLM Answer (No Context)
        worksheet.column_dimensions['H'].width = 15  # LLM Time

        if include_llm_with_context:
            worksheet.column_dimensions['I'].width = 80  # LLM Answer (With Context)
            worksheet.column_dimensions['J'].width = 15  # LLM Context Time

        # Enable text wrapping for answer columns
        from openpyxl.styles import Alignment
        for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical='top')

    print(f"Excel file created successfully: {output_file}")
    print(f"Total questions processed: {len(results)}")
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare RAG and LLM answers for professor questions"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing professor JSON files"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(f"data/rag_llm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"),
        help="Output Excel file path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of professors to process (default: all)"
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Skip LLM with context column"
    )

    args = parser.parse_args()

    # Run async processing
    asyncio.run(process_professors(
        input_dir=args.input_dir,
        output_file=args.output_file,
        limit=args.limit,
        include_llm_with_context=not args.no_context
    ))


if __name__ == "__main__":
    main()
