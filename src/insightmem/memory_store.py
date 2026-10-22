from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from calendar import monthrange
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

from .chunking import TOKEN_PATTERN
from .embeddings import Embedder
from .reranking import Reranker


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
    created_at: str


class RequestConflictError(ValueError):
    pass


def lexical_tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in TOKEN_PATTERN.findall(text)
        if any(character.isalnum() or character == "_" for character in token)
    ]


def _shift_months(value: datetime, months: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _shift_years(value: datetime, years: int) -> datetime:
    year = value.year + years
    day = min(value.day, monthrange(year, value.month)[1])
    return value.replace(year=year, day=day)


def _format_date(value: datetime) -> str:
    return f"{value:%B} {value.day}, {value.year}"


def _format_month(value: datetime) -> str:
    return f"{value:%B} {value.year}"


def _format_week(value: datetime) -> str:
    start = value - timedelta(days=value.weekday())
    end = start + timedelta(days=6)
    return f"{_format_date(start)} through {_format_date(end)}"


def _relative_time_annotations(content: str, message_time: datetime) -> list[str]:
    rules = (
        (r"\blast night\b", _format_date(message_time - timedelta(days=1))),
        (r"\byesterday\b", _format_date(message_time - timedelta(days=1))),
        (r"\btomorrow\b", _format_date(message_time + timedelta(days=1))),
        (r"\b(?:today|tonight)\b", _format_date(message_time)),
        (r"\blast week\b", _format_week(message_time - timedelta(weeks=1))),
        (r"\bthis week\b", _format_week(message_time)),
        (r"\bnext week\b", _format_week(message_time + timedelta(weeks=1))),
        (r"\blast month\b", _format_month(_shift_months(message_time, -1))),
        (r"\bthis month\b", _format_month(message_time)),
        (r"\bnext month\b", _format_month(_shift_months(message_time, 1))),
        (r"\blast year\b", str(message_time.year - 1)),
        (r"\bthis year\b", str(message_time.year)),
        (r"\bnext year\b", str(message_time.year + 1)),
    )
    annotations: list[str] = []
    seen: set[str] = set()
    for pattern, resolved in rules:
        for match in re.finditer(pattern, content, flags=re.IGNORECASE):
            phrase = match.group(0).casefold()
            if phrase not in seen:
                annotations.append(f"{match.group(0)} = {resolved}")
                seen.add(phrase)

    quantities = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}
    ago_pattern = re.compile(
        r"\b(?P<count>a|an|one|two|three|\d+)\s+"
        r"(?P<unit>day|week|month|year)s?\s+ago\b",
        flags=re.IGNORECASE,
    )
    for match in ago_pattern.finditer(content):
        phrase = match.group(0).casefold()
        if phrase in seen:
            continue
        raw_count = match.group("count").casefold()
        count = quantities.get(raw_count, int(raw_count) if raw_count.isdigit() else 1)
        unit = match.group("unit").casefold()
        if unit == "day":
            resolved_time = message_time - timedelta(days=count)
        elif unit == "week":
            resolved_time = message_time - timedelta(weeks=count)
        elif unit == "month":
            resolved_time = _shift_months(message_time, -count)
        else:
            resolved_time = _shift_years(message_time, -count)
        annotations.append(f"{match.group(0)} = {_format_date(resolved_time)}")
        seen.add(phrase)
    return annotations


def bm25_scores(documents: list[str], query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
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


def split_memory_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be >= 0 and smaller than chunk_tokens")
    matches = list(TOKEN_PATTERN.finditer(text))
    if not matches:
        return []
    chunks: list[str] = []
    start = 0
    step = chunk_tokens - overlap_tokens
    while start < len(matches):
        end = min(start + chunk_tokens, len(matches))
        chunk = text[matches[start].start() : matches[end - 1].end()].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(matches):
            break
        start += step
    return chunks


class SQLiteMemoryStore:
    """Persistent, user-isolated vector store for the competition API."""

    def __init__(
        self,
        path: str | Path,
        embedder: Embedder,
        chunk_tokens: int = 320,
        overlap_tokens: int = 40,
        retrieval_mode: str = "dense",
        rrf_k: int = 60,
        lexical_weight: float = 1.0,
        context_messages: int = 0,
        option_view_weight: float = 0.2,
        reranker: Reranker | None = None,
        rerank_candidates: int = 200,
        temporal_enrichment: bool = False,
        result_window: int = 0,
        result_window_seed_k: int = 30,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        embedding_identity = str(
            getattr(
                embedder,
                "identity",
                getattr(embedder, "model_name", f"{embedder.__class__.__module__}.{embedder.__class__.__qualname__}"),
            )
        )
        self.embedding_identity = (
            embedding_identity
            if context_messages == 0
            else f"{embedding_identity}|context=v1:{context_messages}"
        )
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        if retrieval_mode not in {"dense", "hybrid", "enhanced"}:
            raise ValueError("retrieval_mode must be 'dense', 'hybrid', or 'enhanced'")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if lexical_weight < 0:
            raise ValueError("lexical_weight must be non-negative")
        if context_messages < 0:
            raise ValueError("context_messages must be non-negative")
        if not 0.0 <= option_view_weight <= 1.0:
            raise ValueError("option_view_weight must be between 0 and 1")
        if rerank_candidates <= 0:
            raise ValueError("rerank_candidates must be positive")
        if result_window < 0:
            raise ValueError("result_window must be non-negative")
        if result_window_seed_k <= 0:
            raise ValueError("result_window_seed_k must be positive")
        self.retrieval_mode = retrieval_mode
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self.context_messages = context_messages
        self.option_view_weight = option_view_weight
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.temporal_enrichment = temporal_enrichment
        self.result_window = result_window
        self.result_window_seed_k = result_window_seed_k
        self._embedding_lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 60000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS add_requests (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_timestamp INTEGER,
                    created_at TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES add_requests(request_id)
                );

                CREATE INDEX IF NOT EXISTS memories_user_id_idx
                ON memories(user_id);

                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            existing_identity = connection.execute(
                "SELECT value FROM store_metadata WHERE key = 'embedding_identity'"
            ).fetchone()
            if existing_identity is None:
                connection.execute(
                    "INSERT INTO store_metadata(key, value) VALUES ('embedding_identity', ?)",
                    (self.embedding_identity,),
                )
            elif existing_identity[0] != self.embedding_identity:
                raise ValueError(
                    "Configured embedding model does not match the model used by this memory database"
                )
            connection.commit()

    @staticmethod
    def _payload_hash(user_id: str, session_id: str, messages: list[MemoryMessage]) -> str:
        value = {
            "user_id": user_id,
            "session_id": session_id,
            "messages": [message.__dict__ for message in messages],
        }
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _timestamp(value: int | None) -> str:
        if value is None:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    def _embedding_text(
        self,
        messages: list[MemoryMessage],
        message_index: int,
        content: str,
    ) -> str:
        if self.context_messages == 0:
            return content
        current = messages[message_index]
        lines = [f"{current.role}: {content}"]
        excerpt_tokens = max(16, min(96, self.chunk_tokens // 3))
        for distance in range(1, self.context_messages + 1):
            previous_index = message_index - distance
            if previous_index >= 0:
                previous = messages[previous_index]
                body = self._context_excerpt(previous.content, excerpt_tokens, from_end=True)
                if body:
                    lines.append(f"Previous {previous.role}: {body}")
            next_index = message_index + distance
            if next_index < len(messages):
                following = messages[next_index]
                body = self._context_excerpt(following.content, excerpt_tokens, from_end=False)
                if body:
                    lines.append(f"Next {following.role}: {body}")
        return "\n".join(lines)

    @staticmethod
    def _context_excerpt(text: str, token_limit: int, from_end: bool) -> str:
        matches = list(TOKEN_PATTERN.finditer(text))
        if not matches:
            return ""
        if len(matches) <= token_limit:
            return text.strip()
        if from_end:
            return text[matches[-token_limit].start() : matches[-1].end()].strip()
        return text[matches[0].start() : matches[token_limit - 1].end()].strip()

    def add(self, request_id: str, user_id: str, session_id: str, messages: list[MemoryMessage]) -> int:
        payload_hash = self._payload_hash(user_id, session_id, messages)
        with closing(self._connect()) as connection:
            existing = connection.execute(
                "SELECT user_id, session_id, payload_hash FROM add_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["user_id"] != user_id
                    or existing["session_id"] != session_id
                    or existing["payload_hash"] != payload_hash
                ):
                    raise RequestConflictError("request_id was already used with a different payload")
                return int(
                    connection.execute(
                        "SELECT COUNT(*) FROM memories WHERE request_id = ?", (request_id,)
                    ).fetchone()[0]
                )

        pending: list[tuple[str, str, str, int | None, str, str]] = []
        for message_index, message in enumerate(messages):
            chunks = split_memory_text(message.content, self.chunk_tokens, self.overlap_tokens)
            for chunk_index, content in enumerate(chunks):
                identity = f"{request_id}:{message_index}:{chunk_index}"
                memory_id = "mem_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                pending.append(
                    (
                        memory_id,
                        message.role,
                        content,
                        message.timestamp,
                        self._timestamp(message.timestamp),
                        self._embedding_text(messages, message_index, content),
                    )
                )
        if not pending:
            raise ValueError("messages produced no searchable content")

        with self._embedding_lock:
            vectors = self.embedder.encode_documents([item[5] for item in pending])
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(pending):
            raise ValueError("Embedder returned an invalid document vector matrix")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT user_id, session_id, payload_hash FROM add_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["user_id"] != user_id
                    or existing["session_id"] != session_id
                    or existing["payload_hash"] != payload_hash
                ):
                    raise RequestConflictError("request_id was already used with a different payload")
                count = connection.execute(
                    "SELECT COUNT(*) FROM memories WHERE request_id = ?", (request_id,)
                ).fetchone()[0]
                connection.commit()
                return int(count)

            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            connection.execute(
                "INSERT INTO add_requests(request_id, user_id, session_id, payload_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (request_id, user_id, session_id, payload_hash, now),
            )
            for item, vector in zip(pending, vectors):
                memory_id, role, content, source_timestamp, created_at, _ = item
                connection.execute(
                    """
                    INSERT INTO memories(
                        id, request_id, user_id, session_id, role, content,
                        source_timestamp, created_at, embedding, dimension
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        request_id,
                        user_id,
                        session_id,
                        role,
                        content,
                        source_timestamp,
                        created_at,
                        np.ascontiguousarray(vector, dtype=np.float32).tobytes(),
                        int(vector.shape[0]),
                    ),
                )
            connection.commit()
            return len(pending)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def search(self, user_id: str, query: str, top_k: int, options: list[str] | None = None) -> list[MemoryHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, role, content, source_timestamp, created_at, embedding, dimension
                FROM memories WHERE user_id = ?
                ORDER BY rowid
                """,
                (user_id,),
            ).fetchall()
        if not rows:
            return []

        retrieval_query = query
        query_views = [query]
        if options:
            retrieval_query += "\nOptions:\n" + "\n".join(options)
            if self.retrieval_mode == "enhanced":
                query_views.extend(f"{query}\nCandidate answer: {option}" for option in options)
            else:
                query_views = [retrieval_query]
        with self._embedding_lock:
            query_vectors = np.asarray(self.embedder.encode_queries(query_views), dtype=np.float32)
        if query_vectors.ndim != 2 or query_vectors.shape[0] != len(query_views):
            raise ValueError("Embedder returned an invalid query vector matrix")
        dimension = int(rows[0]["dimension"])
        if query_vectors.shape[1] != dimension:
            raise ValueError("Query embedding dimension does not match stored memories")

        matrix = np.vstack(
            [np.frombuffer(row["embedding"], dtype=np.float32, count=dimension) for row in rows]
        )
        view_scores = matrix @ query_vectors.T
        dense_scores = view_scores[:, 0]
        if self.retrieval_mode == "enhanced" and len(query_views) > 1:
            option_scores = np.max(view_scores[:, 1:], axis=1)
            dense_scores = dense_scores + self.option_view_weight * np.maximum(
                option_scores - dense_scores,
                0.0,
            )
        result_scores = dense_scores
        positions = np.argsort(-dense_scores, kind="stable")
        if self.retrieval_mode in {"hybrid", "enhanced"} and self.lexical_weight > 0:
            lexical = bm25_scores([str(row["content"]) for row in rows], retrieval_query)
            fused = np.zeros(len(rows), dtype=np.float32)
            for rank, position in enumerate(positions):
                fused[int(position)] += 1.0 / (self.rrf_k + rank + 1)
            lexical_positions = np.flatnonzero(lexical > 0)
            lexical_positions = lexical_positions[
                np.argsort(-lexical[lexical_positions], kind="stable")
            ]
            for rank, position in enumerate(lexical_positions):
                fused[int(position)] += self.lexical_weight / (self.rrf_k + rank + 1)
            positions = np.lexsort((-dense_scores, -fused))
            result_scores = fused

        if self.reranker is not None:
            candidate_count = min(len(rows), max(top_k, self.rerank_candidates))
            candidate_positions = np.asarray(positions[:candidate_count], dtype=np.int64)
            candidate_documents = [
                self._result_content(rows[int(position)])
                for position in candidate_positions
            ]
            rerank_scores = np.asarray(
                self.reranker.score(query, candidate_documents),
                dtype=np.float32,
            ).reshape(-1)
            if rerank_scores.shape[0] != candidate_count or not np.all(np.isfinite(rerank_scores)):
                raise ValueError("Reranker returned invalid scores")
            rerank_order = np.lexsort(
                (-result_scores[candidate_positions], -rerank_scores)
            )
            positions = candidate_positions[rerank_order]
            result_scores = np.full(len(rows), -np.inf, dtype=np.float32)
            result_scores[candidate_positions] = rerank_scores

        limit = min(top_k, len(rows))
        if self.result_window:
            positions = self._expand_result_positions(positions, rows, limit)
        else:
            positions = positions[:limit]
        results: list[MemoryHit] = []
        for position in positions:
            row = rows[int(position)]
            timestamp = self._timestamp(row["source_timestamp"]) if row["source_timestamp"] is not None else row["created_at"]
            score = float(result_scores[int(position)])
            if not math.isfinite(score):
                score = 0.0
            results.append(
                MemoryHit(
                    id=str(row["id"]),
                    content=self._result_content(row),
                    score=score,
                    created_at=str(timestamp),
                )
            )
        return results

    def _expand_result_positions(
        self,
        positions: np.ndarray,
        rows: list[sqlite3.Row],
        limit: int,
    ) -> np.ndarray:
        expanded: list[int] = []
        seen: set[int] = set()

        def add(position: int) -> None:
            if 0 <= position < len(rows) and position not in seen and len(expanded) < limit:
                seen.add(position)
                expanded.append(position)

        seed_count = min(self.result_window_seed_k, len(positions))
        for raw_position in positions[:seed_count]:
            position = int(raw_position)
            add(position)
            for distance in range(1, self.result_window + 1):
                for neighbor in (position - distance, position + distance):
                    if (
                        0 <= neighbor < len(rows)
                        and rows[neighbor]["session_id"] == rows[position]["session_id"]
                    ):
                        add(neighbor)

        for raw_position in positions:
            add(int(raw_position))
        return np.asarray(expanded, dtype=np.int64)

    def _result_content(self, row: sqlite3.Row) -> str:
        if self.temporal_enrichment:
            if row["source_timestamp"] is None:
                return f"[{row['role']}] {row['content']}"
            message_time = datetime.fromtimestamp(
                row["source_timestamp"] / 1000,
                tz=timezone.utc,
            )
            rendered = (
                f"[{row['role']} | Message date: {_format_date(message_time)} "
                f"at {message_time:%H:%M} UTC] {row['content']}"
            )
            annotations = _relative_time_annotations(str(row["content"]), message_time)
            if annotations:
                rendered += f" [Resolved relative dates: {'; '.join(annotations)}]"
            return rendered
        timestamp = (
            self._timestamp(row["source_timestamp"])
            if row["source_timestamp"] is not None
            else row["created_at"]
        )
        return f"[{row['role']} | {timestamp}] {row['content']}"

    def stats(self) -> dict[str, int]:
        with closing(self._connect()) as connection:
            requests = int(connection.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0])
            memories = int(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
            users = int(connection.execute("SELECT COUNT(DISTINCT user_id) FROM memories").fetchone()[0])
        return {"requests": requests, "memories": memories, "users": users}
