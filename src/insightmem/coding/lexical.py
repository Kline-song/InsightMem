"""Bounded, immutable lexical statistics for one read snapshot.

The exact ordered text tuple is part of the key, so Add, cross-process writes
and reordered rows cannot serve stale statistics. Entries are namespaced by
user and retrieval leg; queries and results are never cached.
"""
from __future__ import annotations

import math
from collections import Counter, OrderedDict
from threading import RLock

import numpy as np

from ..text import lexical_tokens


class LexicalFeatureCache:
    def __init__(self, max_entries: int = 4, max_units: int = 16_000_000) -> None:
        # Units conservatively weight characters and posting/dict entries; this
        # is a logical capacity bound, not a promise about Python RSS bytes.
        self.max_entries = max_entries
        self.max_units = max_units
        self._entries = OrderedDict()
        self._units = 0
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    def _prepare(self, scope, documents):
        texts = tuple(documents)
        key = (scope, texts)
        with self._lock:
            if key in self._entries:
                self.hits += 1
                self._entries.move_to_end(key)
                return self._entries[key][0]
            self.misses += 1
            rows = tuple(Counter(lexical_tokens(text)) for text in texts)
            lengths = tuple(sum(row.values()) for row in rows)
            frequency = Counter(term for row in rows for term in row)
            prepared = (rows, lengths, frequency, (sum(lengths) / len(rows) or 1.0) if rows else 1.0)
            units = sum(map(len, texts)) * 4 + 128 * (len(rows) + len(frequency) + sum(map(len, rows)))
            if self.max_entries > 0 and units <= self.max_units:
                while self._entries and (len(self._entries) >= self.max_entries or self._units + units > self.max_units):
                    _, (_, old_units) = self._entries.popitem(last=False)
                    self._units -= old_units
                self._entries[key] = (prepared, units)
                self._units += units
            return prepared

    def score(self, scope, documents, query, k1: float = 1.5, b: float = 0.75, dtype=np.float32) -> np.ndarray:
        rows, lengths, frequency, average = self._prepare(scope, documents)
        terms = set(lexical_tokens(query))
        scores = np.zeros(len(rows), dtype=dtype)
        # Retain the original term iteration order, arithmetic, float32 update
        # and corpus-wide IDF. Candidate-local IDF would change the ranking.
        for index, row in enumerate(rows):
            normalization = 1.0 - b + b * lengths[index] / average
            for term in terms:
                frequency_in_document = row.get(term, 0)
                if frequency_in_document:
                    inverse_document_frequency = math.log(
                        1.0 + (len(rows) - frequency[term] + 0.5) / (frequency[term] + 0.5)
                    )
                    scores[index] += inverse_document_frequency * (
                        frequency_in_document * (k1 + 1.0) / (frequency_in_document + k1 * normalization)
                    )
        return scores
