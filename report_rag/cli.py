from __future__ import annotations

import argparse
import os

from .config import DEFAULT_LLM_MODEL, DEFAULT_MODEL, DEFAULT_RERANKER_MODEL
from .evaluation import evaluate_retrieval
from .indexer import build_index
from .search import interactive_search, search_index


def add_search_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_query: bool = True,
) -> None:
    if require_query:
        parser.add_argument("query")
    parser.add_argument("--index-dir", default="index")
    parser.add_argument(
        "--route",
        choices=["vector", "bm25", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--vector-weight", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--context-chunks", type=int, default=5)
    parser.add_argument("--max-input-tokens", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse PDF reports and build vector/BM25/hybrid retrieval indexes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Parse PDFs and build the FAISS index.")
    build.add_argument("--pdf-dir", default="pdf_dataset")
    build.add_argument("--index-dir", default="index")
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--chunk-tokens", type=int, default=512)
    build.add_argument("--overlap-tokens", type=int, default=100)
    build.add_argument("--min-chunk-tokens", type=int, default=30)
    build.add_argument("--batch-size", type=int, default=16)
    build.add_argument("--header-ratio", type=float, default=0.08)
    build.add_argument("--footer-ratio", type=float, default=0.08)
    build.add_argument("--repeated-page-ratio", type=float, default=0.35)
    build.add_argument("--no-fp16", action="store_true")
    build.set_defaults(handler=build_index)

    search = subparsers.add_parser(
        "search", 
        help="Search an existing index."
    )
    add_search_arguments(search)
    search.set_defaults(handler=search_index)

    interactive = subparsers.add_parser(
        "interactive",
        help="Start a persistent search session and reuse loaded models.",
    )
    add_search_arguments(interactive, require_query=False)
    interactive.set_defaults(handler=interactive_search)

    evaluate = subparsers.add_parser(
        "eval",
        help="Evaluate retrieval recall with a JSONL eval set.",
    )
    add_search_arguments(evaluate, require_query=False)
    evaluate.add_argument("--eval-set", default="eval_set.jsonl")
    evaluate.add_argument("--output", default="eval_results_hybrid.jsonl")
    evaluate.set_defaults(handler=evaluate_retrieval)
    
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if hasattr(args, "vector_weight") and not 0 <= args.vector_weight <= 1:
        parser.error("--vector-weight must be between 0 and 1")
    if hasattr(args, "rerank_top_n") and args.rerank_top_n <= 0:
        parser.error("--rerank-top-n must be greater than 0")
    if hasattr(args, "context_chunks") and args.context_chunks <= 0:
        parser.error("--context-chunks must be greater than 0")
    if hasattr(args, "temperature") and args.temperature < 0:
        parser.error("--temperature must be greater than or equal to 0")
    if getattr(args, "command", None) == "eval":
        if args.generate:
            parser.error("eval does not support --generate")
        if args.output is None:
            args.output = f"eval_results_{args.route}.jsonl"


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = create_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    args.handler(args)
