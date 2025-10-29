#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from typing import Optional

try:
    import requests  # type: ignore
except Exception:
    requests = None


def call_backend(question: str, base_url: str, api_key: Optional[str], max_results: int = 5) -> Optional[str]:
    if requests is None:
        return None
    url = base_url.rstrip("/") + "/query"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = {
        "query": question,
        "query_type": "general",
        "streaming": False,
        "max_results": max_results,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer")
    except Exception as e:
        sys.stderr.write(f"Error calling backend for question: {question[:60]}... -> {e}\n")
        return None


def main():
    p = argparse.ArgumentParser(description="Fill LLM responses for questions CSV by calling backend /api/query")
    p.add_argument("--input", default="data/qna_seed.csv", help="Input CSV with columns: question,answer[,llm_response]")
    p.add_argument("--output", default="data/qna_with_llm.csv", help="Output CSV path")
    p.add_argument("--base-url", default="http://localhost:8000/api", help="Backend base URL (prefix before /query)")
    p.add_argument("--api-key", default=os.getenv("API_KEY"), help="API key for backend (env API_KEY by default)")
    p.add_argument("--max-results", type=int, default=5)
    p.add_argument("--sleep", type=float, default=0.2, help="Sleep between calls (seconds)")
    p.add_argument("--force", action="store_true", help="Overwrite existing llm_response values")
    args = p.parse_args()

    # Read input
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Ensure llm_response column exists
    for r in rows:
        if "llm_response" not in r or r["llm_response"] is None:
            r["llm_response"] = ""

    # Process
    for r in rows:
        if r.get("llm_response") and not args.force:
            continue
        q = r.get("question", "").strip()
        if not q:
            continue
        ans = call_backend(q, args.base_url, args.api_key, args.max_results)
        if ans is not None:
            r["llm_response"] = ans
        time.sleep(args.sleep)

    # Write output
    fieldnames = ["question", "answer", "llm_generated", "llm_response"]
    # include any extra columns from input
    for k in rows[0].keys():
        if k not in fieldnames:
            fieldnames.append(k)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

