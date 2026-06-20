from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .bm25 import BM25Index
from .generation import (
    UNCERTAIN_ANSWER,
    QwenGenerator,
    format_citation_sources,
    validate_citations,
)
from .models import BgeEncoder, BgeReranker
from .storage import read_jsonl
from .text_utils import lexical_tokens
from .vectors import minmax, normalize_vectors


class SearchSession:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Missing faiss-cpu. Install dependencies first.") from exc

        self.args = args
        self.index_dir = Path(args.index_dir).resolve()
        self.chunks = read_jsonl(self.index_dir / "chunks.jsonl")
        if not self.chunks:
            raise RuntimeError("The index contains no chunks.")

        print(f"Loading chunks and BM25 from {self.index_dir}", flush=True)
        tokenized_documents = [lexical_tokens(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Index(tokenized_documents)
        self.vector_scores_template = np.zeros(len(self.chunks), dtype=np.float32)

        self.encoder = None
        self.faiss_index = None
        if args.route in {"vector", "hybrid"}:
            manifest = json.loads(
                (self.index_dir / "manifest.json").read_text(encoding="utf-8")
            )
            print(f"Loading recall model {manifest['model']}", flush=True)
            self.encoder = BgeEncoder(manifest["model"], use_fp16=not args.no_fp16)
            self.faiss_index = faiss.read_index(str(self.index_dir / "vectors.faiss"))
            self.configure_faiss_index(manifest)

        self.reranker = None
        if args.rerank:
            print(f"Loading reranker {args.reranker_model}", flush=True)
            self.reranker = BgeReranker(args.reranker_model, use_fp16=not args.no_fp16)

        self.generator = None
        if args.generate:
            print(f"Loading generator {args.llm_model}", flush=True)
            self.generator = QwenGenerator(args.llm_model, use_fp16=not args.no_fp16)

    def configure_faiss_index(self, manifest: dict) -> None:
        if self.faiss_index is None:
            return
        index_type = manifest.get("index_type", "flat")
        if index_type == "ivf":
            self.faiss_index.nprobe = self.args.ivf_nprobe
        elif index_type == "hnsw":
            self.faiss_index.hnsw.efSearch = self.args.hnsw_ef_search

    def search(self, query: str) -> tuple[list[dict], str | None]:
        args = self.args
        bm25_scores = self.bm25.scores(lexical_tokens(query))

        vector_scores = self.vector_scores_template.copy()
        if args.route in {"vector", "hybrid"}:
            if self.encoder is None or self.faiss_index is None:
                raise RuntimeError("Vector search requested but vector index is not loaded.")
            query_vector = normalize_vectors(self.encoder.encode_query(query))
            vector_scores.fill(-1.0)
            if args.route == "vector":
                search_k = args.top_k
                if args.rerank:
                    search_k = max(search_k, args.rerank_top_n)
            else:
                search_k = len(self.chunks)
            # faiss_start = time.perf_counter()
            scores, indices = self.faiss_index.search(query_vector, search_k)
            # faiss_elapsed = time.perf_counter() - faiss_start
            # print(f"FAISS search time: {faiss_elapsed:.6f}s")
            for score, index in zip(scores[0], indices[0]):
                if index >= 0:
                    vector_scores[index] = score

        if args.route == "vector":
            final_scores = vector_scores
        elif args.route == "bm25":
            final_scores = bm25_scores
        else:
            vector_weight = args.vector_weight
            final_scores = (
                vector_weight * minmax(vector_scores)
                + (1 - vector_weight) * minmax(bm25_scores)
            )

        candidate_count = args.top_k
        if args.rerank:
            candidate_count = max(candidate_count, args.rerank_top_n)
        candidate_count = min(candidate_count, len(self.chunks))
        ranking = np.argsort(-final_scores, kind="stable")[:candidate_count]

        rerank_scores_by_index: dict[int, float] = {}
        if args.rerank and len(ranking):
            if self.reranker is None:
                raise RuntimeError("Rerank requested but reranker is not loaded.")
            print(
                f"Reranking {len(ranking)} candidates with {args.reranker_model}",
                flush=True,
            )
            pairs = [
                (query, self.chunks[int(index)]["text"])
                for index in ranking
            ]
            rerank_scores = self.reranker.score(pairs, args.rerank_batch_size)
            rerank_scores_by_index = {
                int(index): float(score)
                for index, score in zip(ranking, rerank_scores)
            }
            ranking = np.asarray(
                sorted(
                    (int(index) for index in ranking),
                    key=lambda index: rerank_scores_by_index[index],
                    reverse=True,
                ),
                dtype=np.int64,
            )
        ranking = ranking[: args.top_k]

        results = build_results(
            ranking,
            self.chunks,
            final_scores,
            vector_scores,
            bm25_scores,
            rerank_scores_by_index,
        )

        answer = None
        if args.generate:
            if self.generator is None:
                raise RuntimeError("Generation requested but generator is not loaded.")
            context_results = generation_context_results(results, args)
            if not context_results:
                answer = UNCERTAIN_ANSWER
            else:
                answer = self.generator.generate_answer(
                    query,
                    context_results,
                    len(context_results),
                    args.max_input_tokens,
                    args.max_new_tokens,
                    args.temperature,
                )
                if args.citation_check and not validate_citations(
                    answer,
                    context_results,
                    len(context_results),
                ):
                    print(
                        "\n=== LLM Answer rejected === citation validation failed",
                        flush=True,
                    )
                    print(f"Raw answer: {answer}", flush=True)
                    answer = UNCERTAIN_ANSWER
                else:
                    answer = format_citation_sources(
                        answer,
                        context_results,
                        len(context_results),
                    )
        return results, answer


def is_low_confidence(results: Sequence[dict], args: argparse.Namespace) -> bool:
    return not generation_context_results(results, args)


def generation_context_results(
    results: Sequence[dict],
    args: argparse.Namespace,
) -> list[dict]:
    context_results = []
    for result in results:
        if passes_generation_thresholds(result, args):
            context_results.append(result)
        if len(context_results) >= args.context_chunks:
            break
    return context_results


def passes_generation_thresholds(result: dict, args: argparse.Namespace) -> bool:
    return not llm_exclusion_reasons(result, args)


def llm_exclusion_reasons(result: dict, args: argparse.Namespace) -> list[str]:
    reasons = []
    if args.min_recall_score is not None:
        recall_score = result.get("recall_score")
        if recall_score is None:
            reasons.append("recall score is missing")
        elif recall_score < args.min_recall_score:
            reasons.append(
                f"recall score {recall_score:.4f} < threshold {args.min_recall_score:.4f}"
            )

    if args.rerank and args.min_rerank_score is not None:
        rerank_score = result.get("rerank_score")
        if rerank_score is None:
            reasons.append("rerank score is missing")
        elif rerank_score < args.min_rerank_score:
            reasons.append(
                f"rerank score {rerank_score:.4f} < threshold {args.min_rerank_score:.4f}"
            )

    return reasons


def search_index(args: argparse.Namespace) -> None:
    session = SearchSession(args)
    results, answer = session.search(args.query)
    print_search_output(results, answer, args)


def interactive_search(args: argparse.Namespace) -> None:
    session = SearchSession(args)
    print("Interactive search is ready. Type 'exit', 'quit', or ':q' to stop.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except EOFError:
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            break
        start_time = time.perf_counter()
        results, answer = session.search(query)
        print_search_output(results, answer, args)
        elapsed = time.perf_counter() - start_time
        print(f"\nElapsed: {elapsed:.2f}s")


def build_results(
    ranking: Sequence[int],
    chunks: Sequence[dict],
    final_scores: np.ndarray,
    vector_scores: np.ndarray,
    bm25_scores: np.ndarray,
    rerank_scores_by_index: dict[int, float],
) -> list[dict]:
    results = []
    for rank, index in enumerate(ranking, 1):
        chunk = chunks[int(index)]
        rerank_score = rerank_scores_by_index.get(int(index))
        display_score = rerank_score if rerank_score is not None else float(final_scores[index])
        result = {
            "rank": rank,
            "score": round(display_score, 4),
            "recall_score": round(float(final_scores[index]), 4),
            "vector_score": round(float(vector_scores[index]), 4),
            "bm25_score": round(float(bm25_scores[index]), 4),
            "rerank_score": round(rerank_score, 4) if rerank_score is not None else None,
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "source": chunk["source"],
            "pages": chunk["pages"],
            "element_type": chunk["element_type"],
            "text": chunk["text"],
        }
        results.append(result)
    return results


def print_search_output(
    results: Sequence[dict],
    answer: str | None,
    args: argparse.Namespace,
) -> None:
    if args.json:
        payload = {"results": list(results)}
        if answer is not None:
            payload["answer"] = answer
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    fed_context_ids = set()
    if getattr(args, "generate", False):
        fed_context_ids = {
            result["chunk_id"]
            for result in generation_context_results(results, args)
        }

    for result in results:
        if result["chunk_id"] in fed_context_ids:
            print("\n=== Fed into the LLM ===")
        else:
            if getattr(args, "generate", False):
                reasons = llm_exclusion_reasons(result, args)
                reason = "; ".join(reasons) if reasons else "context_chunks limit reached"
            else:
                reason = "generation disabled"
            print(f"\n=== Exclude from LLM processing === {reason}")
        print(
            f"\n[{result['rank']}] score={result['score']:.4f}  "
            f"recall={result['recall_score']:.4f}  "
            f"vector={result['vector_score']:.4f}  bm25={result['bm25_score']:.4f}  "
            f"rerank={result['rerank_score']}"
        )
        print(
            f"source:{result['source']}  pages:{result['pages']}  "
            f"type:{result['element_type']}"
        )
        print(result["text"][: args.max_chars])

    if answer is not None:
        print("\n=== LLM Answer ===")
        print(answer)
