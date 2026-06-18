from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np


class BM25Index:
    def __init__(self, tokenized_documents: Sequence[Sequence[str]], k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = [list(document) for document in tokenized_documents]
        self.lengths = np.asarray(
            [len(document) for document in self.documents],
            dtype=np.float32,
        )
        self.avgdl = float(self.lengths.mean()) if len(self.lengths) else 0.0
        self.term_frequencies = [Counter(document) for document in self.documents]
        document_frequencies: Counter[str] = Counter()
        for document in self.documents:
            document_frequencies.update(set(document))
        count = len(self.documents)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(len(self.documents), dtype=np.float32)
        if not query_tokens or self.avgdl == 0:
            return scores
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self.term_frequencies):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / self.avgdl
                )
                scores[index] += idf * frequency * (self.k1 + 1) / denominator
        return scores

