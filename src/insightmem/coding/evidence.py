"""Deterministic evidence assembly. Every rendered body is a verbatim source slice."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone

from ..store import MemoryHit
from .config import stable_id

SOURCE_LABEL = re.compile(r"^\[source role=[^\]\n]+ kind=[^\]\n]+\] ")


def body(text: str) -> str:
    """Remove only the renderer-owned label; literal body characters are kept."""
    return SOURCE_LABEL.sub("", text, count=1)


def duplicate_key(text: str):
    """Exact bodies only, with the source label retained as part of identity."""
    label = SOURCE_LABEL.match(text)
    return (label.group(0) if label else "", body(text))


def spans_covered(spans, selected) -> bool:
    """Suppress fully redundant spans; partial overlap can contain new conditions."""
    if not spans:
        return False
    for message_id, start, end in spans:
        covered_end = start
        for _, other_start, other_end in sorted((s for s in selected if s[0] == message_id), key=lambda s: s[1]):
            if other_start > covered_end:
                break
            if other_end > covered_end:
                covered_end = other_end
            if covered_end >= end:
                break
        if covered_end < end:
            return False
    return True


def spans_overlap(spans, selected) -> bool:
    """True when any candidate span was already returned.

    Bundles are convenience views of sources that stay available individually,
    so returning one after part of it was selected only duplicates context.
    """
    return any(message_id == selected_message and start < selected_end and selected_start < end
               for message_id, start, end in spans
               for selected_message, selected_start, selected_end in selected)


def candidate_order(ranked, chunks_by_id):
    """Move exact duplicate bodies behind unique ones without dropping any."""
    seen, unique, repeated = set(), [], []
    for chunk_id in ranked:
        chunk = chunks_by_id[chunk_id]
        key = (chunk["role"], chunk["kind"], chunk["raw_content"])
        (repeated if key in seen else unique).append(chunk_id)
        seen.add(key)
    return unique + repeated


def _stamp(chunks) -> str | None:
    values = [chunk["timestamp_ms"] for chunk in chunks if chunk["timestamp_ms"] is not None]
    return datetime.fromtimestamp(max(values) / 1000, timezone.utc).isoformat() if values else None


def _render(chunks, score):
    """Merge only overlapping or touching slices of the same source message."""
    chunks = sorted(chunks, key=lambda s: (s["ingest_seq"], s["message_index"], s["start_char"], s["chunk_id"]))
    pieces, spans, last = [], [], None
    for chunk in chunks:
        message_id, start, end, text = chunk["message_id"], chunk["start_char"], chunk["end_char"], chunk["raw_content"]
        if last is not None and message_id == last["message_id"] and start <= spans[-1][2]:
            old_start, old_end = spans[-1][1:]
            overlap = max(0, min(old_end, end) - start)
            if pieces[-1][start - old_start:start - old_start + overlap] != text[:overlap]:
                raise ValueError("inconsistent source overlap")
            if end > old_end:
                pieces[-1] += text[old_end - start:]
                spans[-1] = (message_id, old_start, end)
        else:
            pieces.append(text)
            spans.append((message_id, start, end))
        last = chunk
    labels = {chunk["message_id"]: f"[source role={chunk['role']} kind={chunk['kind']}] " for chunk in chunks}
    content = "\n\n".join(labels[message_id] + text for (message_id, _, _), text in zip(spans, pieces))
    identity = chunks[0]["chunk_id"] if len(chunks) == 1 else stable_id("evidence-v1", spans)
    return MemoryHit(identity, content, float(score), _stamp(chunks)), tuple(spans)


def prepare_sources(ranked, chunks, scores, config):
    """Merge touching slices of one message into ordered evidence candidates.

    A bundle is an optional consolidation, never a replacement for its members:
    long bundles can rank below individual slices, so members stay available and
    span deduplication removes them only if the bundle is actually returned.
    """
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    seeds = [by_id[chunk_id] for chunk_id in ranked]
    groups = []
    ordered = sorted(seeds, key=lambda s: (s["ingest_seq"], s["message_index"], s["start_char"], s["chunk_id"]))
    for chunk in ordered:
        if groups and groups[-1][-1]["message_id"] == chunk["message_id"]:
            first = min(item["start_char"] for item in groups[-1])
            last = max(max(item["end_char"] for item in groups[-1]), chunk["end_char"])
            touching = chunk["start_char"] <= max(item["end_char"] for item in groups[-1])
            if touching and last - first <= config.evidence_bundle_chars:
                groups[-1].append(chunk)
                continue
        groups.append([chunk])

    pool, provenance, emitted, bundles = [], {}, set(), set()

    def emit(group, score, bundle=False):
        hit, spans = _render(group, score)
        if hit.id in emitted:
            return True
        limit = config.evidence_bundle_chars if bundle else config.evidence_max_chars
        if len(hit.content) > limit:
            return False
        pool.append(hit)
        provenance[hit.id] = spans
        emitted.add(hit.id)
        if len(group) > 1:
            bundles.add(hit.id)
        return True

    for group in groups:
        score = max(scores.get(chunk["chunk_id"], 0.0) for chunk in group)
        if len(group) == 1:
            emit(group, score)
            continue
        emit(group, score, bundle=True)
        for chunk in group:
            emit([chunk], scores.get(chunk["chunk_id"], 0.0))
    return pool, provenance, bundles
