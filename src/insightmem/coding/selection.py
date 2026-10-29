"""Deterministic utility ordering: relevance first, novelty as a tie-break."""
from __future__ import annotations

from .content import query_features, typed_factor
from .evidence import duplicate_key, spans_covered, spans_overlap


def select(query, pool, top_k, provenance, bundles, type_factors):
    """Pick the highest-utility non-redundant candidates up to ``top_k``."""
    metadata = [
        (hit, query_features(hit.content, query), max(1, len(hit.content)))
        for hit in pool
    ]
    maximum = max((hit.score for hit in pool), default=1.0) or 1.0
    selected, covered, exact = [], set(), set()
    strict_keys, selected_spans = set(), []

    while metadata and len(selected) < top_k:
        candidates = []
        for index, (hit, info, cost) in enumerate(metadata):
            if hit.content in exact:
                continue
            spans = provenance.get(hit.id, ())
            if (duplicate_key(hit.content) in strict_keys
                    or spans_covered(spans, selected_spans)
                    or (hit.id in bundles and spans_overlap(spans, selected_spans))):
                continue
            # Relevance is the primary key; normalized novelty is a bounded
            # tie-break, so unrelated line numbers cannot overpower a match.
            gain = len(info - covered) / max(1, len(info))
            factor = type_factors.get(hit.id, typed_factor(hit.content))
            utility = factor * (hit.score / maximum)
            candidates.append((utility, gain, -cost, hit.id, index))
        if not candidates:
            break
        hit, info, _ = metadata.pop(max(candidates)[-1])
        selected.append(hit)
        covered.update(info)
        exact.add(hit.content)
        strict_keys.add(duplicate_key(hit.content))
        selected_spans.extend(provenance.get(hit.id, ()))
    return selected
