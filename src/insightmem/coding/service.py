"""Coding memory service: immutable sources plus deterministic evidence retrieval."""
from __future__ import annotations

import logging
import sqlite3
import threading
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..embeddings import EmbeddingServiceError
from ..store import RequestConflictError, bm25_scores
from .config import PROFILE_VERSION, CodingConfig, digest, stable_id
from .content import typed_factor
from .evidence import candidate_order, prepare_sources
from .filter import low_information
from .lexical import LexicalFeatureCache
from .parse import chunk_spans, code_tokens, parse
from .selection import select

SCHEMA = """
CREATE TABLE coding_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE coding_requests(user_id TEXT,request_id TEXT,session_id TEXT,payload_hash TEXT,ingest_seq INTEGER,result_counts INTEGER,PRIMARY KEY(user_id,request_id),UNIQUE(user_id,ingest_seq));
CREATE INDEX request_session ON coding_requests(user_id,session_id,ingest_seq);
CREATE TABLE source_messages(user_id TEXT,message_id TEXT,request_id TEXT,message_index INTEGER,role TEXT,timestamp_ms INTEGER,raw_content TEXT,kind TEXT,PRIMARY KEY(user_id,message_id),FOREIGN KEY(user_id,request_id) REFERENCES coding_requests(user_id,request_id));
CREATE INDEX message_request ON source_messages(user_id,request_id,message_index);
CREATE TABLE source_chunks(user_id TEXT,chunk_id TEXT,message_id TEXT,ordinal INTEGER,start_char INTEGER,end_char INTEGER,raw_content TEXT,PRIMARY KEY(user_id,chunk_id),FOREIGN KEY(user_id,message_id) REFERENCES source_messages(user_id,message_id));
CREATE TABLE index_views(user_id TEXT,view_id TEXT,kind TEXT,search_text TEXT,dense_enabled INTEGER,PRIMARY KEY(user_id,view_id));
CREATE TABLE view_sources(user_id TEXT,view_id TEXT,source_ordinal INTEGER,message_id TEXT,start_char INTEGER,end_char INTEGER,PRIMARY KEY(user_id,view_id,source_ordinal),FOREIGN KEY(user_id,view_id) REFERENCES index_views(user_id,view_id),FOREIGN KEY(user_id,message_id) REFERENCES source_messages(user_id,message_id));
CREATE TABLE view_embeddings(user_id TEXT,view_id TEXT,vector BLOB,dimension INTEGER,identity TEXT,PRIMARY KEY(user_id,view_id),FOREIGN KEY(user_id,view_id) REFERENCES index_views(user_id,view_id));
"""


class CodingMemoryService:
    def __init__(self, path, embedder, config: CodingConfig | None = None) -> None:
        if embedder is None:
            raise ValueError("the evidence service requires an embedder")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config or CodingConfig()
        self.embedder = embedder
        self.identity = str(getattr(embedder, "identity", getattr(embedder, "model_name", type(embedder).__qualname__)))
        self.fingerprint = self.config.fingerprint(self.identity)
        self.lock = threading.RLock()
        self.lexical_cache = LexicalFeatureCache()
        with closing(self._connect()) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and "coding_meta" not in tables:
                raise ValueError("refusing legacy or unknown database")
            if not tables:
                connection.executescript("BEGIN IMMEDIATE;" + SCHEMA)
                connection.executemany(
                    "INSERT INTO coding_meta VALUES (?,?)",
                    [("schema", "1"), ("build", self.fingerprint), ("embedding", self.identity)],
                )
                connection.commit()
            if dict(connection.execute("SELECT key,value FROM coding_meta")) != {
                "schema": "1", "build": self.fingerprint, "embedding": self.identity,
            }:
                raise ValueError("coding build/embedding mismatch")
            connection.execute("PRAGMA journal_mode=WAL")

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _existing(connection, user, request, payload_hash):
        row = connection.execute(
            "SELECT payload_hash,result_counts FROM coding_requests WHERE user_id=? AND request_id=?",
            (user, request),
        ).fetchone()
        if row:
            if row["payload_hash"] != payload_hash:
                raise RequestConflictError("request_id already used with different payload")
            return row["result_counts"]
        return None

    def _vectors(self, texts, query=False):
        try:
            with self.lock:
                encoder = self.embedder.encode_queries if query else self.embedder.encode_documents
                vectors = np.asarray(encoder(texts), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] < 1 or not np.isfinite(vectors).all():
                raise ValueError("invalid shape")
            norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("invalid norms")
            return np.asarray(vectors / norms, dtype=np.float32)
        except Exception:
            raise EmbeddingServiceError("coding embedding failed or returned invalid vectors") from None

    def add(self, request_id, user_id, session_id, messages):
        if not messages or any(not isinstance(m.content, str) or not m.content for m in messages):
            raise ValueError("nonempty messages required")
        payload_hash = digest([session_id, [asdict(m) for m in messages]])
        with closing(self._connect()) as connection:
            previous = self._existing(connection, user_id, request_id, payload_hash)
            if previous is not None:
                return previous

        parsed = parse(user_id, request_id, messages)
        chunks, views, sources = [], [], []
        for message, parsed_message in zip(messages, parsed):
            for ordinal, (start, end) in enumerate(
                chunk_spans(message.content, self.config.chunk_units, self.config.overlap_units, self.config.max_chunk_chars)
            ):
                chunk_id = stable_id("chunk", parsed_message.message_id, start, end, PROFILE_VERSION)
                text = message.content[start:end]
                chunks.append((user_id, chunk_id, parsed_message.message_id, ordinal, start, end, text))
                # Every source occurrence stays stored; only the retrieval
                # representation is deduplicated by chunk identity.
                view_id = stable_id("source-view", chunk_id)
                views.append((user_id, view_id, "source", text, int(bool(text.strip()))))
                sources.append((user_id, view_id, 0, parsed_message.message_id, start, end))

        dense_views = [view for view in views if view[4]]
        vectors = self._vectors([view[3] for view in dense_views]) if dense_views else []
        with closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                previous = self._existing(connection, user_id, request_id, payload_hash)
                if previous is not None:
                    connection.rollback()
                    return previous
                sequence = connection.execute(
                    "SELECT COALESCE(MAX(ingest_seq),0)+1 FROM coding_requests WHERE user_id=?", (user_id,)
                ).fetchone()[0]
                dimensions = {row[0] for row in connection.execute("SELECT DISTINCT dimension FROM view_embeddings")}
                if len(vectors) and dimensions and dimensions != {vectors.shape[1]}:
                    raise EmbeddingServiceError("embedding dimension mismatch")
                connection.execute(
                    "INSERT INTO coding_requests VALUES (?,?,?,?,?,?)",
                    (user_id, request_id, session_id, payload_hash, sequence, len(chunks)),
                )
                connection.executemany(
                    "INSERT INTO source_messages VALUES (?,?,?,?,?,?,?,?)",
                    [
                        (user_id, parsed_message.message_id, request_id, index, message.role, message.timestamp,
                         message.content, parsed_message.kind)
                        for index, (message, parsed_message) in enumerate(zip(messages, parsed))
                    ],
                )
                connection.executemany("INSERT INTO source_chunks VALUES (?,?,?,?,?,?,?)", chunks)
                connection.executemany("INSERT INTO index_views VALUES (?,?,?,?,?)", views)
                connection.executemany("INSERT INTO view_sources VALUES (?,?,?,?,?,?)", sources)
                for view, vector in zip(dense_views, vectors):
                    connection.execute(
                        "INSERT INTO view_embeddings VALUES (?,?,?,?,?)",
                        (user_id, view[1], vector.tobytes(), len(vector), self.identity),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return len(chunks)

    def search(self, user_id, query, top_k, options=None):
        if top_k <= 0 or top_k > 100:
            raise ValueError("top_k out of range")
        with closing(self._connect()) as connection:
            # A Search sees one committed snapshot even while another Add commits.
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            chunks = [dict(row) for row in connection.execute(
                "SELECT s.*,m.role,m.kind,m.timestamp_ms,m.request_id,m.message_index,r.ingest_seq "
                "FROM source_chunks s JOIN source_messages m USING(user_id,message_id) "
                "JOIN coding_requests r USING(user_id,request_id) WHERE s.user_id=? "
                "ORDER BY r.ingest_seq,m.message_index,s.ordinal",
                (user_id,),
            )]
            if not chunks:
                return []
            views = [dict(row) for row in connection.execute(
                "SELECT * FROM index_views WHERE user_id=? ORDER BY view_id", (user_id,)
            )]
            references = [dict(row) for row in connection.execute(
                "SELECT * FROM view_sources WHERE user_id=?", (user_id,)
            )]
            vectors = [dict(row) for row in connection.execute(
                "SELECT * FROM view_embeddings WHERE user_id=? ORDER BY view_id", (user_id,)
            )]
            typed_messages = {
                row["message_id"]: typed_factor(row["raw_content"])
                for row in connection.execute(
                    "SELECT message_id,raw_content FROM source_messages WHERE user_id=?", (user_id,)
                )
            }
            blocked = {
                row["message_id"]
                for row in connection.execute(
                    "SELECT message_id,raw_content,kind FROM source_messages WHERE user_id=?", (user_id,)
                )
                if row["kind"] in {"assistant_text", "thinking"} and low_information(row["raw_content"])
            }

        # Message filtering is query-only and reversible: hidden messages keep
        # contributing to BM25 corpus statistics and occupy RRF rank positions,
        # but they never become candidates or evidence items.
        blocked_chunks = {chunk["chunk_id"] for chunk in chunks if chunk["message_id"] in blocked}
        depth = min(self.config.leg_max, max(50, 2 * top_k))
        scores = defaultdict(float)
        best = defaultdict(lambda: 10 ** 9)
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        chunk_index = {chunk["chunk_id"]: index for index, chunk in enumerate(chunks)}
        by_message = defaultdict(list)
        for chunk in chunks:
            by_message[chunk["message_id"]].append(chunk)
        view_chunks = defaultdict(set)
        for reference in references:
            for chunk in by_message[reference["message_id"]]:
                if chunk["start_char"] < reference["end_char"] and reference["start_char"] < chunk["end_char"]:
                    view_chunks[reference["view_id"]].add(chunk["chunk_id"])

        def lexical_scores(texts, leg, lexical_query):
            try:
                return self.lexical_cache.score((user_id, leg), texts, lexical_query)
            except Exception as error:
                logging.getLogger(__name__).warning("lexical cache fallback error_type=%s", type(error).__name__)
                return bm25_scores(texts, lexical_query)

        def fuse(ranked):
            seen, rank, unique = set(), 0, set()
            scan_limit = min(self.config.leg_max, depth * 4)
            for ids in ranked:
                fresh = [chunk_id for chunk_id in ids if chunk_id not in seen]
                if not fresh:
                    continue
                rank += 1
                for chunk_id in fresh:
                    scores[chunk_id] += 1 / (self.config.rrf_k + rank)
                    best[chunk_id] = min(best[chunk_id], rank)
                    seen.add(chunk_id)
                    chunk = chunk_by_id[chunk_id]
                    unique.add((chunk["role"], chunk["kind"], chunk["raw_content"]))
                    if len(seen) >= scan_limit or len(unique) >= depth:
                        return
                if rank >= scan_limit:
                    break

        raw = lexical_scores([chunk["raw_content"] for chunk in chunks], "raw", query)
        fuse([[chunks[i]["chunk_id"]] for i in sorted(range(len(chunks)), key=lambda i: (-float(raw[i]), i)) if raw[i] > 0])
        if views:
            texts = [" ".join(code_tokens(view["search_text"])) for view in views]
            expanded_query = " ".join(code_tokens(query))
            lexical = lexical_scores(texts, "views", expanded_query)
            fuse([
                sorted(view_chunks[views[i]["view_id"]], key=lambda chunk_id: (
                    -float(raw[chunk_index[chunk_id]]), chunk_by_id[chunk_id]["message_index"], chunk_by_id[chunk_id]["ordinal"]))
                for i in sorted(range(len(views)), key=lambda i: (-float(lexical[i]), views[i]["view_id"]))
                if lexical[i] > 0
            ])
        if vectors:
            query_vector = self._vectors([query], query=True)[0]
            matrix = np.stack([np.frombuffer(vector["vector"], np.float32) for vector in vectors])
            if matrix.shape[1] != len(query_vector):
                raise EmbeddingServiceError("query dimension mismatch")
            dense = matrix @ query_vector
            fuse([
                sorted(view_chunks[vectors[i]["view_id"]])
                for i in sorted(range(len(vectors)), key=lambda i: (-float(dense[i]), vectors[i]["view_id"]))
            ])

        def order(chunk_id):
            chunk = chunk_by_id[chunk_id]
            return (-scores[chunk_id], best[chunk_id], chunk["ingest_seq"], chunk["message_index"], chunk["ordinal"], chunk_id)

        ranked = sorted(scores, key=order)
        ranked = candidate_order(ranked, chunk_by_id)
        if blocked_chunks:
            ranked = [chunk_id for chunk_id in ranked if chunk_id not in blocked_chunks]
        pool, provenance, bundles = prepare_sources(ranked[:depth], chunks, scores, self.config)
        type_factors = {}
        for hit in pool:
            message_ids = [span[0] for span in provenance.get(hit.id, ())]
            if not message_ids and hit.id in chunk_by_id:
                message_ids = [chunk_by_id[hit.id]["message_id"]]
            type_factors[hit.id] = max((typed_messages.get(message_id, 1.0) for message_id in message_ids), default=1.0)
        return select(query, pool, top_k, provenance, bundles, type_factors)

    def stats(self):
        with closing(self._connect()) as connection:
            return {
                "requests": connection.execute("SELECT count(*) FROM coding_requests").fetchone()[0],
                "memories": connection.execute("SELECT count(*) FROM source_chunks").fetchone()[0],
                "users": connection.execute("SELECT count(DISTINCT user_id) FROM coding_requests").fetchone()[0],
            }
