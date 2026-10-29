import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from insightmem.coding.config import CodingConfig
from insightmem.coding.service import CodingMemoryService
from insightmem.store import MemoryMessage as M, RequestConflictError


def vector_for(text, dimension=64):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * (dimension // len(digest) + 1))[:dimension]
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float32) + 1.0


class HashEmbedder:
    def __init__(self, dimension=64):
        self.dimension = dimension
        self.identity = f"hash|dimensions={dimension}"

    def _encode(self, texts):
        return np.asarray([vector_for(text, self.dimension) for text in texts], dtype=np.float32)

    def encode_documents(self, texts):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode(texts)


class CodingServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "coding.db"
        self.embedder = HashEmbedder()
        self.service = CodingMemoryService(self.path, self.embedder, CodingConfig())

    def tearDown(self):
        self.temporary.cleanup()

    def add(self, request_id, messages, user="u", session="s"):
        return self.service.add(request_id, user, session, [M(role, text) for role, text in messages])

    def test_add_is_idempotent_and_conflicts_on_reuse(self):
        messages = [("user", "The parser drops backslashes in escaped fields.")]
        first = self.add("r1", messages)
        self.assertEqual(first, self.add("r1", messages))
        with self.assertRaises(RequestConflictError):
            self.add("r1", [("user", "A different payload.")])

    def test_search_is_user_isolated(self):
        self.add("r1", [("user", "alpha parser failure")], user="a")
        self.add("r2", [("user", "alpha parser failure")], user="b")
        self.assertTrue(self.service.search("a", "alpha", 5))
        self.assertFalse(self.service.search("c", "alpha", 5))

    def test_search_respects_top_k_and_returns_labels(self):
        self.add("r1", [("user", "alpha one"), ("assistant", "alpha two"), ("user", "alpha three")])
        hits = self.service.search("u", "alpha", 2)
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(hit.content.startswith("[source role=") for hit in hits))

    def test_process_only_assistant_messages_are_not_selectable(self):
        self.add("r1", [
            ("user", "Where is the parser?"),
            ("assistant", "Let me inspect the implementation and run the relevant tests."),
            ("assistant", "The parser lives in src/parser.py and uses unescape()."),
        ])
        hits = self.service.search("u", "inspect the implementation and run the relevant tests", 10)
        self.assertTrue(hits)
        self.assertFalse(any("Let me inspect the implementation" in hit.content for hit in hits))

    def test_long_message_is_merged_into_one_evidence_item(self):
        long_text = " ".join(f"token{index}" for index in range(700))
        self.add("r1", [("assistant", long_text)])
        hits = self.service.search("u", "token300", 10)
        contents = "\n".join(hit.content for hit in hits)
        self.assertIn("token0 ", contents)
        self.assertIn("token699", contents)

    def test_reopening_an_existing_database_keeps_results(self):
        self.add("r1", [("user", "alpha parser failure")])
        reopened = CodingMemoryService(self.path, self.embedder, CodingConfig())
        self.assertEqual(len(reopened.search("u", "alpha", 5)), len(self.service.search("u", "alpha", 5)))

    def test_dimension_change_is_rejected(self):
        self.add("r1", [("user", "alpha")])
        with self.assertRaises(ValueError):
            CodingMemoryService(self.path, HashEmbedder(dimension=128), CodingConfig())

    def test_empty_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.add("r1", "u", "s", [])
        with self.assertRaises(ValueError):
            self.service.add("r1", "u", "s", [M("user", "")])

    def test_top_k_bounds(self):
        for value in (0, 101):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.service.search("u", "alpha", value)

    def test_stats_counts_requests_and_chunks(self):
        self.add("r1", [("user", "alpha beta")])
        stats = self.service.stats()
        self.assertEqual(stats["requests"], 1)
        self.assertGreaterEqual(stats["memories"], 1)


if __name__ == "__main__":
    unittest.main()
