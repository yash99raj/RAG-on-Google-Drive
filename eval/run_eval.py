#!/usr/bin/env python3
"""Eval harness — run against a live server: uvicorn src.api.main:create_app --factory --port 8000"""
import json
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
GOLDEN_PATH = Path(__file__).parent / "golden_set.json"


def run_eval() -> None:
    golden = json.loads(GOLDEN_PATH.read_text())

    results = []
    for item in golden:
        question = item["question"]
        keywords = item["keywords"]

        t0 = time.monotonic()
        try:
            resp = httpx.post(
                f"{BASE_URL}/ask",
                json={"query": question},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.monotonic() - t0) * 1000

            answer: str = data.get("answer", "")
            sources: list = data.get("sources", [])
            has_answer = len(answer) > 50
            has_sources = len(sources) > 0
            keyword_hit = any(kw.lower() in answer.lower() for kw in keywords)
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            has_answer = has_sources = keyword_hit = False
            print(f"[ERROR] {question!r}: {exc}", file=sys.stderr)

        results.append(
            {
                "question": question,
                "has_answer": has_answer,
                "has_sources": has_sources,
                "keyword_hit": keyword_hit,
                "latency_ms": round(latency_ms, 1),
            }
        )

    col_q = 45
    header = f"{'question':<{col_q}} | has_answer | has_sources | keyword_hit | latency_ms"
    print(header)
    print("-" * len(header))
    for r in results:
        q = r["question"][:col_q].ljust(col_q)
        print(
            f"{q} | {'yes' if r['has_answer'] else 'no ':>10} "
            f"| {'yes' if r['has_sources'] else 'no ':>11} "
            f"| {'yes' if r['keyword_hit'] else 'no ':>11} "
            f"| {r['latency_ms']:>10.1f}"
        )

    n = len(results)
    retrieval_rate = sum(r["has_sources"] for r in results) / n
    keyword_accuracy = sum(r["keyword_hit"] for r in results) / n
    avg_latency = sum(r["latency_ms"] for r in results) / n

    print()
    print(f"retrieval_rate:   {retrieval_rate:.0%}  ({sum(r['has_sources'] for r in results)}/{n})")
    print(f"keyword_accuracy: {keyword_accuracy:.0%}  ({sum(r['keyword_hit'] for r in results)}/{n})")
    print(f"avg_latency_ms:   {avg_latency:.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        sys.exit(0)
    run_eval()
