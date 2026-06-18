from __future__ import annotations

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def minmax(scores: np.ndarray) -> np.ndarray:
    if not len(scores):
        return scores
    minimum = float(scores.min())
    maximum = float(scores.max())
    if np.isclose(minimum, maximum):
        return np.zeros_like(scores)
    return (scores - minimum) / (maximum - minimum)

