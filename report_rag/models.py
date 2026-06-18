from __future__ import annotations

from typing import Sequence

import numpy as np


class BgeEncoder:
    def __init__(self, model_name: str, use_fp16: bool = True) -> None:
        try:
            from FlagEmbedding import FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "Missing FlagEmbedding. Install dependencies before building/searching."
            ) from exc
        self.model = FlagModel(
            model_name,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=use_fp16,
        )

    def encode_corpus(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        return np.asarray(
            self.model.encode_corpus(list(texts), batch_size=batch_size),
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        vector = self.model.encode_queries([query])
        return np.asarray(vector, dtype=np.float32)


class BgeReranker:
    def __init__(self, model_name: str, use_fp16: bool = True) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "Missing FlagEmbedding. Install dependencies before reranking."
            ) from exc
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def score(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int,
    ) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        sentence_pairs = [[query, text] for query, text in pairs]
        try:
            scores = self.model.compute_score(sentence_pairs, batch_size=batch_size)
        except TypeError:
            scores = self.model.compute_score(sentence_pairs)
        if isinstance(scores, (float, int)):
            scores = [scores]
        return np.asarray(scores, dtype=np.float32)

