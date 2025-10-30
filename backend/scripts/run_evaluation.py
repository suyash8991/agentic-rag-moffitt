"""
Run LangSmith evaluation on the moffitt-researcher-qa dataset

This script evaluates the agent's performance on the Q&A dataset
created from qna_seed.csv.

Usage:
    python scripts/run_evaluation.py
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import Client
from langsmith.evaluation import evaluate
from app.core.config import settings
from app.services.agent import process_query


async def evaluate_query(inputs: dict) -> dict:
    """
    Evaluation function to process a single query.

    Args:
        inputs: Dictionary with 'query' key

    Returns:
        Dictionary with 'answer' key
    """
    query = inputs["query"]

    try:
        # Generate a unique query ID for this evaluation
        query_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Process the query
        response = await process_query(
            query_id=query_id,
            query=query,
            query_type="researcher",
            streaming=False,
            max_results=5
        )

        return {"answer": response.answer}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}


def run_evaluation():
    """Run evaluation on the dataset"""

    if not settings.LANGCHAIN_TRACING_V2:
        print("❌ LangSmith tracing is not enabled. Set LANGCHAIN_TRACING_V2=true in your .env file.")
        return False

    if not settings.LANGCHAIN_API_KEY:
        print("❌ LANGCHAIN_API_KEY is not set. Please add your LangSmith API key to .env file.")
        return False

    dataset_name = "moffitt-researcher-qa"
    experiment_prefix = f"moffitt-rag-eval-{datetime.now().strftime('%Y%m%d_%H%M')}"

    print(f"Running LangSmith evaluation")
    print(f"Dataset: {dataset_name}")
    print(f"Experiment: {experiment_prefix}")
    print(f"Project: {settings.LANGCHAIN_PROJECT}")
    print("-" * 60)

    # Initialize client
    client = Client(
        api_key=settings.LANGCHAIN_API_KEY,
        api_url=settings.LANGCHAIN_ENDPOINT
    )

    # Check if dataset exists
    try:
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        if not datasets:
            print(f"❌ Dataset '{dataset_name}' not found.")
            print(f"   Run 'python scripts/create_langsmith_dataset.py' first to create the dataset.")
            return False

        print(f"✓ Found dataset: {dataset_name}")

        # Get dataset info
        dataset = datasets[0]
        examples_count = len(list(client.list_examples(dataset_id=dataset.id)))
        print(f"✓ Dataset contains {examples_count} examples")

    except Exception as e:
        print(f"❌ Error accessing dataset: {e}")
        return False

    # Wrapper to handle async function
    def sync_evaluate_query(inputs: dict) -> dict:
        """Synchronous wrapper for async evaluate_query"""
        return asyncio.run(evaluate_query(inputs))

    # Run evaluation
    print()
    print("Starting evaluation...")
    print("This may take several minutes depending on the number of examples.")
    print()

    try:
        results = evaluate(
            sync_evaluate_query,
            data=dataset_name,
            experiment_prefix=experiment_prefix,
            # Add custom evaluators here if needed
        )

        print()
        print("=" * 60)
        print("Evaluation complete!")
        print(f"Experiment: {experiment_prefix}")
        print(f"View results at: https://smith.langchain.com/")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
