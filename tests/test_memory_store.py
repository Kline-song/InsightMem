import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from insightmem.memory_store import (
    MemoryMessage,
    RequestConflictError,
    SQLiteMemoryStore,
    split_memory_text,
)


class DeterministicEmbedder:
    dimension = 64

    def _encode(self, texts):
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vectors[row, int.from_bytes(digest[:2], "big") % self.dimension] += 1
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors

    def encode_documents(self, texts):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode(texts)


class MisleadingEmbedder:
    dimension = 2

    def encode_documents(self, texts):
        return np.asarray(
            [[0.0, 1.0] if "PROJECT_ZEPHYR" in text else [1.0, 0.0] for text in texts],
            dtype=np.float32,
        )

    def encode_queries(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class RecordingEmbedder(DeterministicEmbedder):
    def __init__(self):
        self.document_batches = []
        self.query_batches = []

    def encode_documents(self, texts):
        self.document_batches.append(list(texts))
        return super().encode_documents(texts)

    def encode_queries(self, texts):
        self.query_batches.append(list(texts))
        return super().encode_queries(texts)


class RecordingReranker:
    def __init__(self):
        self.calls = []

    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        return np.asarray(
            [1.0 if "PROJECT_ZEPHYR" in document else 0.0 for document in documents],
            dtype=np.float32,
        )


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "memory.db"
        self.store = SQLiteMemoryStore(self.path, DeterministicEmbedder(), chunk_tokens=20, overlap_tokens=2)

    def tearDown(self):
        self.directory.cleanup()

    def test_add_is_idempotent_and_persistent(self):
        messages = [MemoryMessage("user", "Ada Lovelace wrote analytical engine notes", 1704067200000)]
        first = self.store.add("r1", "u1", "s1", messages)
        second = self.store.add("r1", "u1", "s1", messages)
        self.assertEqual(first, second)
        reopened = SQLiteMemoryStore(self.path, DeterministicEmbedder())
        self.assertEqual(reopened.stats(), {"requests": 1, "memories": 1, "users": 1})
        self.assertIn("Ada Lovelace", reopened.search("u1", "Ada Lovelace", 5)[0].content)

    def test_user_isolation(self):
        self.store.add("r1", "u1", "s1", [MemoryMessage("user", "alpha private fact")])
        self.store.add("r2", "u2", "s2", [MemoryMessage("user", "beta private marker")])
        self.assertTrue(self.store.search("u1", "beta", 10))
        self.assertTrue(all("beta private marker" not in hit.content for hit in self.store.search("u1", "beta", 10)))
        self.assertEqual(self.store.search("missing", "anything", 10), [])

    def test_temporal_enrichment_resolves_relative_dates(self):
        path = Path(self.directory.name) / "temporal.db"
        store = SQLiteMemoryStore(
            path,
            DeterministicEmbedder(),
            temporal_enrichment=True,
        )
        timestamp = int(datetime(2023, 1, 20, 16, 4, tzinfo=timezone.utc).timestamp() * 1000)
        store.add(
            "temporal-r1",
            "u1",
            "s1",
            [MemoryMessage("user", "I lost my job yesterday and will travel next month.", timestamp)],
        )

        content = store.search("u1", "job travel", 1)[0].content

        self.assertIn("Message date: January 20, 2023 at 16:04 UTC", content)
        self.assertIn("yesterday = January 19, 2023", content)
        self.assertIn("next month = February 2023", content)

    def test_temporal_enrichment_does_not_present_ingestion_time_as_event_time(self):
        path = Path(self.directory.name) / "missing-timestamp.db"
        store = SQLiteMemoryStore(
            path,
            DeterministicEmbedder(),
            temporal_enrichment=True,
        )
        store.add("temporal-r1", "u1", "s1", [MemoryMessage("user", "No event date supplied.")])

        content = store.search("u1", "event date", 1)[0].content

        self.assertEqual(content, "[user] No event date supplied.")

    def test_result_window_adds_neighbors_and_respects_top_k(self):
        path = Path(self.directory.name) / "result-window.db"
        store = SQLiteMemoryStore(
            path,
            MisleadingEmbedder(),
            retrieval_mode="hybrid",
            lexical_weight=1.0,
            result_window=1,
            result_window_seed_k=1,
        )
        store.add(
            "window-r1",
            "u1",
            "s1",
            [
                MemoryMessage("user", "The preceding local detail."),
                MemoryMessage("assistant", "The MIDDLE_WINDOW_MARKER appears here."),
                MemoryMessage("user", "The following local detail."),
                MemoryMessage("assistant", "An unrelated tail message."),
            ],
        )

        results = store.search("u1", "MIDDLE_WINDOW_MARKER", 3)

        self.assertEqual(len(results), 3)
        self.assertIn("MIDDLE_WINDOW_MARKER", results[0].content)
        self.assertIn("preceding local detail", results[1].content)
        self.assertIn("following local detail", results[2].content)

    def test_conflicting_request_id_is_rejected(self):
        self.store.add("r1", "u1", "s1", [MemoryMessage("user", "original")])
        with self.assertRaises(RequestConflictError):
            self.store.add("r1", "u1", "s1", [MemoryMessage("user", "changed")])

    def test_concurrent_idempotent_add(self):
        messages = [MemoryMessage("user", "the same retry payload")]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.store.add("retry", "u1", "s1", messages), range(4)))
        self.assertEqual(results, [1, 1, 1, 1])
        self.assertEqual(self.store.stats(), {"requests": 1, "memories": 1, "users": 1})

    def test_chunking_preserves_original_cjk_text(self):
        chunks = split_memory_text("北京搬到上海以后喜欢喝咖啡。", chunk_tokens=6, overlap_tokens=2)
        self.assertGreater(len(chunks), 1)
        self.assertNotIn("北 京", chunks[0])

    def test_hybrid_search_recovers_exact_lexical_match(self):
        path = Path(self.directory.name) / "hybrid.db"
        dense_store = SQLiteMemoryStore(path, MisleadingEmbedder(), retrieval_mode="dense")
        dense_store.add(
            "hybrid-r1",
            "u1",
            "s1",
            [
                MemoryMessage("user", "PROJECT_ZEPHYR launch code is 7319"),
                MemoryMessage("user", "The weekly status meeting moved to Tuesday"),
            ],
        )
        self.assertNotIn("PROJECT_ZEPHYR", dense_store.search("u1", "PROJECT_ZEPHYR", 1)[0].content)

        hybrid_store = SQLiteMemoryStore(path, MisleadingEmbedder(), retrieval_mode="hybrid")
        self.assertIn("PROJECT_ZEPHYR", hybrid_store.search("u1", "PROJECT_ZEPHYR", 1)[0].content)

    def test_contextual_embeddings_include_neighboring_messages(self):
        path = Path(self.directory.name) / "context.db"
        embedder = RecordingEmbedder()
        store = SQLiteMemoryStore(path, embedder, context_messages=1)
        store.add(
            "context-r1",
            "u1",
            "s1",
            [
                MemoryMessage("user", "The launch code is 7319"),
                MemoryMessage("assistant", "I will remember that code"),
            ],
        )

        self.assertEqual(
            embedder.document_batches[0],
            [
                "user: The launch code is 7319\nNext assistant: I will remember that code",
                "assistant: I will remember that code\nPrevious user: The launch code is 7319",
            ],
        )
        returned = [hit.content.lower() for hit in store.search("u1", "remember", 2)]
        self.assertTrue(any("launch code" in content for content in returned))
        self.assertTrue(any("remember that code" in content for content in returned))
        self.assertTrue(all("\nassistant:" not in content for content in returned))
        with self.assertRaisesRegex(ValueError, "embedding model does not match"):
            SQLiteMemoryStore(path, RecordingEmbedder(), context_messages=0)

    def test_enhanced_search_encodes_options_as_separate_views(self):
        path = Path(self.directory.name) / "multi-view.db"
        embedder = RecordingEmbedder()
        store = SQLiteMemoryStore(
            path,
            embedder,
            retrieval_mode="enhanced",
            lexical_weight=0,
            option_view_weight=0.2,
        )
        store.add("multi-r1", "u1", "s1", [MemoryMessage("user", "Ada wrote the notes")])
        store.search(
            "u1",
            "Who wrote the notes?",
            1,
            ["A. Ada Lovelace", "B. Grace Hopper"],
        )

        self.assertEqual(
            embedder.query_batches[-1],
            [
                "Who wrote the notes?",
                "Who wrote the notes?\nCandidate answer: A. Ada Lovelace",
                "Who wrote the notes?\nCandidate answer: B. Grace Hopper",
            ],
        )

    def test_enhanced_search_recovers_exact_lexical_match(self):
        path = Path(self.directory.name) / "enhanced.db"
        store = SQLiteMemoryStore(
            path,
            MisleadingEmbedder(),
            retrieval_mode="enhanced",
            lexical_weight=0.25,
        )
        store.add(
            "enhanced-r1",
            "u1",
            "s1",
            [
                MemoryMessage("user", "PROJECT_ZEPHYR launch code is 7319"),
                MemoryMessage("user", "The weekly status meeting moved to Tuesday"),
            ],
        )

        self.assertIn("PROJECT_ZEPHYR", store.search("u1", "PROJECT_ZEPHYR", 1)[0].content)

    def test_reranker_reorders_initial_candidates(self):
        path = Path(self.directory.name) / "reranker.db"
        reranker = RecordingReranker()
        store = SQLiteMemoryStore(
            path,
            MisleadingEmbedder(),
            retrieval_mode="dense",
            reranker=reranker,
            rerank_candidates=2,
        )
        store.add(
            "reranker-r1",
            "u1",
            "s1",
            [
                MemoryMessage("user", "PROJECT_ZEPHYR launch code is 7319"),
                MemoryMessage("user", "The weekly status meeting moved to Tuesday"),
            ],
        )

        result = store.search("u1", "What is the project launch code?", 1)

        self.assertIn("PROJECT_ZEPHYR", result[0].content)
        self.assertEqual(reranker.calls[0][0], "What is the project launch code?")
        self.assertEqual(len(reranker.calls[0][1]), 2)


if __name__ == "__main__":
    unittest.main()
