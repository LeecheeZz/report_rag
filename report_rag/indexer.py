from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import ChunkBuilder
from .models import BgeEncoder
from .pdf_parser import PdfLayoutParser
from .storage import write_jsonl
from .vectors import normalize_vectors


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
    vectors = normalize_vectors(
        encoder.encode_corpus([chunk.text for chunk in chunks], args.batch_size)
    )

    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)
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
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Index written to {index_dir}")

