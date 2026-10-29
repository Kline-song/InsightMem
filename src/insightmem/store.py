"""Records exchanged by the memory API and the lexical scorer."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .text import lexical_tokens


@dataclass(frozen=True)
class MemoryMessage:
    role: str
    content: str
    timestamp: int | None = None


@dataclass(frozen=True)
class MemoryHit:
    id: str
    content: str
    score: float
    created_at: str | None


class RequestConflictError(ValueError):
    """A request id was reused with a different payload."""


def bm25_scores(documents: list[str], query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Corpus-wide IDF BM25 over the exact document set of one read snapshot."""
    tokenized = [lexical_tokens(document) for document in documents]
    query_terms = set(lexical_tokens(query))
    scores = np.zeros(len(documents), dtype=np.float32)
    if not documents or not query_terms:
        return scores

    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    average_length = average_length or 1.0
    document_frequency = Counter(
        term for tokens in tokenized for term in query_terms.intersection(tokens)
    )
    document_count = len(documents)
    for index, tokens in enumerate(tokenized):
        frequencies = Counter(tokens)
        length_normalization = 1.0 - b + b * len(tokens) / average_length
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1.0 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            scores[index] += inverse_document_frequency * (
                frequency * (k1 + 1.0) / (frequency + k1 * length_normalization)
            )
    return scores
