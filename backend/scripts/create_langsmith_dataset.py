"""
Create LangSmith evaluation dataset from qna_seed.csv

This script converts the Q&A pairs in qna_seed.csv into a LangSmith dataset
for evaluating researcher search quality.

Usage:
    python scripts/create_langsmith_dataset.py
"""

import csv
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import Client
from app.core.config import settings


def create_evaluation_dataset():
    """Create LangSmith dataset from qna_seed.csv"""

    if not settings.LANGCHAIN_TRACING_V2:
        print("❌ LangSmith tracing is not enabled. Set LANGCHAIN_TRACING_V2=true in your .env file.")
        return False

    if not settings.LANGCHAIN_API_KEY:
        print("❌ LANGCHAIN_API_KEY is not set. Please add your LangSmith API key to .env file.")
        return False

    # Initialize client
    client = Client(
        api_key=settings.LANGCHAIN_API_KEY,
        api_url=settings.LANGCHAIN_ENDPOINT
    )

    dataset_name = "moffitt-researcher-qa"
    description = "Q&A pairs for evaluating Moffitt researcher search quality"

    print(f"Creating LangSmith dataset: {dataset_name}")
    print(f"Project: {settings.LANGCHAIN_PROJECT}")
    print("-" * 60)

    # Check if dataset already exists
    try:
        existing_datasets = list(client.list_datasets(dataset_name=dataset_name))
        if existing_datasets:
            print(f"⚠️  Dataset '{dataset_name}' already exists.")
            response = input("Do you want to delete it and create a new one? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return False
            # Delete existing dataset
            for ds in existing_datasets:
                client.delete_dataset(dataset_id=ds.id)
                print(f"✓ Deleted existing dataset")
    except Exception as e:
        print(f"Note: Could not check for existing datasets: {e}")

    # Create dataset
    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description
        )
        print(f"✓ Created dataset with ID: {dataset.id}")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False

    # Load QA pairs from CSV
    csv_path = Path(__file__).parent.parent.parent / "data" / "qna_seed.csv"

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return False

    print(f"Loading Q&A pairs from: {csv_path}")

    examples_added = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                question = row.get('question', '').strip()
                answer = row.get('answer', '').strip()

                if not question or not answer:
                    continue

                # Add to dataset
                client.create_example(
                    dataset_id=dataset.id,
                    inputs={"query": question},
                    outputs={"answer": answer}
                )
                examples_added += 1

                # Show progress
                if examples_added % 10 == 0:
                    print(f"  Added {examples_added} examples...")

        print(f"✓ Added {examples_added} examples to dataset")
        print()
        print("Dataset created successfully!")
        print(f"View it at: https://smith.langchain.com/datasets")
        return True

    except Exception as e:
        print(f"❌ Failed to add examples: {e}")
        return False


if __name__ == "__main__":
    success = create_evaluation_dataset()
    sys.exit(0 if success else 1)
