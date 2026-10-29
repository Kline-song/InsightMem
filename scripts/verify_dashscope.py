"""Minimal live compatibility check for Alibaba Bailian text-embedding-v4.

Uses at most ~14 embedding requests on a small synthetic corpus.  It never
prints the API key and stores no evaluation data.  Run with the project venv:

    PYTHONPATH=src python scripts/verify_dashscope.py

Exit codes: 0 all critical probes passed, 1 a critical probe failed,
2 no key configured (nothing was called).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np

from insightmem.coding.config import CodingConfig
from insightmem.coding.service import CodingMemoryService
from insightmem.embeddings import DashScopeEmbedder, EmbeddingServiceError
from insightmem.store import MemoryMessage as Message

BEIJING = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PROBES = []
STATE = {}
COUNTER = {"batches": 0, "raw": 0}


def mask(text: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", text)


def record(name, critical, fn):
    started = time.perf_counter()
    try:
        detail = fn() or {}
        entry = {"name": name, "critical": critical, "ok": True, **detail}
    except Exception as error:
        status = getattr(error, "status_code", None)
        entry = {
            "name": name,
            "critical": critical,
            "ok": False,
            "error_type": type(error).__name__,
            "error": mask(str(error))[:300],
            **({"http_status": status} if type(status) is int else {}),
        }
    entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    PROBES.append(entry)
    print(f"[{'ok' if entry['ok'] else 'FAIL'}] {name} {entry['elapsed_ms']}ms"
          + ("" if entry["ok"] else f" :: {entry.get('error', '')[:120]}"))
    return entry


def cosine(left, right):
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def make_embedder(key, base_url, dimensions, concurrency):
    return DashScopeEmbedder(
        api_key=key, base_url=base_url, dimensions=dimensions,
        batch_size=10, timeout=60, max_retries=5, concurrency=concurrency,
    )


def raw_call(embedder, text):
    COUNTER["raw"] += 1
    response = embedder._client.embeddings.create(
        model="text-embedding-v4", input=[text], dimensions=embedder.dimensions,
        encoding_format="float",
    )
    return {"vectors": len(response.data), "dimension": len(response.data[0].embedding)}


def count_batches(embedder):
    original = embedder._encode_batch

    def counted(batch):
        COUNTER["batches"] += 1
        return original(batch)

    embedder._encode_batch = counted
    return embedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", BEIJING))
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default="")
    parser.add_argument("--real-db", default="", help="optional read-only view database to sample")
    parser.add_argument("--skip-raw-probes", action="store_true")
    args = parser.parse_args()

    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("MEMORY_EMBED_API_KEY", "")
    if not key:
        print(json.dumps({"status": "skipped", "reason": "no DASHSCOPE_API_KEY/MEMORY_EMBED_API_KEY"}))
        return 2

    print(f"endpoint={args.base_url} dimensions={args.dimensions} concurrency={args.concurrency}")
    embedder = count_batches(make_embedder(key, args.base_url, args.dimensions, args.concurrency))
    english = "The HTTP client retries status 408, 425, 429 and 5xx; values outside 100-599 raise ValueError."
    chinese = "解析器在遇到转义字段时会丢失反斜杠，修复后 99 和 600 两个边界值都需要通过检查。"
    code = "def unescape(value):\n    return value.replace('\\\\n', '\\n')  # keep backslashes"
    long_english = ("parser escape regression " * 80)[:2000]
    long_chinese = ("转义字段解析回归测试 " * 120)[:1200]
    docs = [
        english, chinese, code, "emoji check: ok 😃 é — 中文标点：，。！",
        "line one\nline two\tindented\r\nline three", english,
        "src/parser.py::unescape ValueError 100 599", "status 408 425 429 500",
        long_english, long_chinese,
    ]
    queries = [
        "which HTTP status codes are retried?",
        "解析器转义字段为什么会失败？",
    ]

    def documents_ten():
        vectors = embedder.encode_documents(docs[:10])
        STATE["docs"] = vectors
        norms = np.linalg.norm(vectors, axis=1)
        return {
            "shape": list(vectors.shape),
            "norms_min": float(norms.min()),
            "norms_max": float(norms.max()),
            "deterministic_cosine": cosine(vectors[0], vectors[5]),
        }

    def queries_two():
        vectors = embedder.encode_queries(queries + [english])
        return {
            "shape": list(vectors.shape),
            "query_document_cosine": cosine(vectors[2], STATE["docs"][0]),
        }

    def batch_eleven():
        vectors = embedder.encode_documents(docs)
        return {"shape": list(vectors.shape)}

    def dimension_variant():
        variant = count_batches(make_embedder(key, args.base_url, 512, args.concurrency))
        try:
            vectors = variant.encode_documents([english])
            return {"shape": list(vectors.shape)}
        finally:
            variant.close()

    def raw_probe(name, text):
        def run():
            if args.skip_raw_probes:
                return {"skipped": True}
            return raw_call(embedder, text)
        return run

    def wrapper_controls():
        vectors = embedder.encode_documents(["alpha\x00beta\x01gamma\x7f"])
        return {"shape": list(vectors.shape)}

    def wrapper_blank():
        try:
            embedder.encode_documents([" "])
        except ValueError:
            return {"rejected_locally": True}
        raise AssertionError("blank input was not rejected")

    def service_round_trip():
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coding.db"
            config = CodingConfig()
            service = CodingMemoryService(path, embedder, config)
            messages = [
                Message("user", "Investigate why the parser fails on escaped fields."),
                Message("assistant", '[tool_use Read] {"path": "src/parser.py"}'),
                Message("assistant", "[tool_result] def unescape(value): return value.replace('\\n', '\\n')"),
                Message("assistant", "The bug is in unescape(): it drops backslashes."),
                Message("user", "Please fix it and test the 99 and 600 edge cases."),
                Message("assistant", "Fixed unescape(); all edge-case tests pass."),
            ]
            service.add("r1", "u1", "s1", messages)
            first = service.search("u1", "why does escaped field parsing fail", 10)
            second = service.search("u1", "which edge cases were tested", 10)
            CodingMemoryService(path, embedder, config)
            third = service.search("u1", "which function was fixed", 10)
            with service._connect() as connection:
                count, dimension = connection.execute(
                    "SELECT count(*), min(dimension) FROM view_embeddings"
                ).fetchone()
            return {
                "hits": [len(first), len(second), len(third)],
                "embeddings": count,
                "dimension": dimension,
            }

    def real_views():
        if not args.real_db:
            return {"skipped": True}
        import sqlite3

        connection = sqlite3.connect(f"file:{args.real_db}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT search_text FROM index_views WHERE dense_enabled=1 ORDER BY random() LIMIT 7"
            ).fetchall()
            nul = connection.execute(
                "SELECT search_text FROM index_views WHERE dense_enabled=1 "
                "AND instr(search_text, char(0)) > 0 LIMIT 2"
            ).fetchall()
            longest = connection.execute(
                "SELECT search_text FROM index_views WHERE dense_enabled=1 "
                "ORDER BY length(search_text) DESC LIMIT 1"
            ).fetchall()
        finally:
            connection.close()
        texts = [row[0] for row in rows + nul + longest]
        vectors = embedder.encode_documents(texts)
        return {
            "views": len(texts),
            "shape": list(vectors.shape),
            "max_chars": max(len(text) for text in texts),
            "nul_views": len(nul),
        }

    record("documents_10", True, documents_ten)
    record("queries_2", True, queries_two)
    record("batch_11_as_2_requests", True, batch_eleven)
    record("dimension_512", False, dimension_variant)
    record("wrapper_control_chars", True, wrapper_controls)
    record("wrapper_blank_rejected", True, wrapper_blank)
    if not args.skip_raw_probes:
        record("raw_nul_provider", False, raw_probe("raw_nul_provider", "alpha\x00beta"))
        record("raw_control_provider", False, raw_probe("raw_control_provider", "alpha\x01beta"))
        record("raw_whitespace_provider", False, raw_probe("raw_whitespace_provider", " "))
    record("service_add_search_reopen", True, service_round_trip)
    record("real_views", bool(args.real_db), real_views)

    embedder.close()
    critical_failures = [probe for probe in PROBES if probe["critical"] and not probe["ok"]]
    report = {
        "status": "ok" if not critical_failures else "failed",
        "endpoint": args.base_url,
        "dimensions": args.dimensions,
        "concurrency": args.concurrency,
        "requests_planned": (13 if not args.skip_raw_probes else 10) + (1 if args.real_db else 0),
        "requests_actual": COUNTER["batches"] + COUNTER["raw"],
        "batch_requests": COUNTER["batches"],
        "raw_requests": COUNTER["raw"],
        "probes": PROBES,
    }
    output = Path(args.output) if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report={output}")
    print(json.dumps({"status": report["status"], "failed": [p["name"] for p in critical_failures]}))
    return 0 if not critical_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
