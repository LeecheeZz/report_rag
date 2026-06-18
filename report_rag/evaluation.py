from __future__ import annotations

import argparse
import json
from pathlib import Path

from .search import SearchSession
from .storage import read_jsonl


TOP_K = 10


def evaluate_retrieval(args: argparse.Namespace) -> None:
    eval_path = Path(args.eval_set).resolve()
    output_path = Path(args.output).resolve()
    rows = read_jsonl(eval_path)
    if not rows:
        raise RuntimeError(f"The eval set is empty: {eval_path}")

    args.top_k = max(args.top_k, TOP_K)
    session = SearchSession(args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            results, _ = session.search(row["query"])
            chunk_texts = [result["text"] for result in results[:TOP_K]]
            payload = {
                "query": row["query"],
                "ground-truth": row.get("answer"),
                "retrieved_chunks": chunk_texts,
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Retrieved top {TOP_K} chunks for {len(rows)} queries with route={args.route}")
    print(f"Details written to {output_path}")
