from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import ChunkBuilder
from .models import BgeEncoder
from .pdf_parser import PdfLayoutParser
from .storage import write_jsonl
from .vectors import normalize_vectors


def create_faiss_index(vectors: np.ndarray, args: argparse.Namespace):
    import faiss

    dimension = int(vectors.shape[1])
    if args.index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
    elif args.index_type == "ivf":
        if len(vectors) < args.ivf_nlist:
            raise RuntimeError(
                f"IVF nlist ({args.ivf_nlist}) cannot exceed vector count ({len(vectors)})."
            )
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            args.ivf_nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(vectors)
    elif args.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(
            dimension,
            args.hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = args.hnsw_ef_construction
    else:
        raise RuntimeError(f"Unsupported FAISS index type: {args.index_type}")

    index.add(vectors)
    return index


def build_index(args: argparse.Namespace) -> None:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Missing faiss-cpu. Install dependencies first.") from exc

    pdf_dir = Path(args.pdf_dir).resolve()
    index_dir = Path(args.index_dir).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    parser = PdfLayoutParser(
        header_ratio=args.header_ratio,
        footer_ratio=args.footer_ratio,
        repeated_page_ratio=args.repeated_page_ratio,
    )
    elements = parser.parse_directory(pdf_dir)
    chunk_builder = ChunkBuilder(
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
    )
    chunks = chunk_builder.build(elements)
    if not chunks:
        raise RuntimeError("No chunks were produced from the PDF dataset.")

    print(f"Embedding {len(chunks)} chunks with {args.model}", flush=True)
    encoder = BgeEncoder(args.model, use_fp16=not args.no_fp16)
    embedding_start = time.perf_counter()
    vectors = normalize_vectors(
        encoder.encode_corpus([chunk.text for chunk in chunks], args.batch_size)
    )
    embedding_elapsed = time.perf_counter() - embedding_start
    print(f"Embedding time: {embedding_elapsed:.2f}s", flush=True)

    index_start = time.perf_counter()
    faiss_index = create_faiss_index(vectors, args)
    index_elapsed = time.perf_counter() - index_start
    print(f"FAISS index build time ({args.index_type}): {index_elapsed:.2f}s", flush=True)
    faiss.write_index(faiss_index, str(index_dir / "vectors.faiss"))
    np.save(index_dir / "vectors.npy", vectors)
    write_jsonl(index_dir / "chunks.jsonl", (asdict(chunk) for chunk in chunks))

    manifest = {
        "model": args.model,
        "dimension": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "pdf_dir": str(pdf_dir),
        "chunk_tokens": args.chunk_tokens,
        "overlap_tokens": args.overlap_tokens,
        "index_type": args.index_type,
        "metric": "inner_product",
        "normalized": True,
        "ivf_nlist": args.ivf_nlist if args.index_type == "ivf" else None,
        "hnsw_m": args.hnsw_m if args.index_type == "hnsw" else None,
        "hnsw_ef_construction": (
            args.hnsw_ef_construction if args.index_type == "hnsw" else None
        ),
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Index written to {index_dir}")

