import unittest

from insightmem.coding.config import CodingConfig
from insightmem.coding.evidence import (
    candidate_order,
    duplicate_key,
    prepare_sources,
    spans_covered,
    spans_overlap,
)


def chunk(chunk_id, message_id, start, end, text, ordinal=0, index=0, sequence=1):
    return {
        "chunk_id": chunk_id,
        "message_id": message_id,
        "start_char": start,
        "end_char": end,
        "raw_content": text,
        "ordinal": ordinal,
        "message_index": index,
        "ingest_seq": sequence,
        "role": "assistant",
        "kind": "assistant_text",
        "timestamp_ms": 1000,
        "request_id": "r",
    }


class CandidateOrderTests(unittest.TestCase):
    def test_unique_bodies_precede_repeated_ones(self):
        chunks = {
            "a": chunk("a", "m1", 0, 5, "same"),
            "b": chunk("b", "m2", 0, 5, "same"),
            "c": chunk("c", "m3", 0, 8, "different"),
        }
        self.assertEqual(candidate_order(["a", "b", "c"], chunks), ["a", "c", "b"])

    def test_duplicate_key_keeps_label(self):
        self.assertNotEqual(duplicate_key("[source role=user kind=issue_prompt] x"),
                            duplicate_key("[source role=assistant kind=assistant_text] x"))


class SpanTests(unittest.TestCase):
    def test_covered_requires_full_span(self):
        self.assertTrue(spans_covered([("m", 0, 5)], [("m", 0, 5)]))
        self.assertFalse(spans_covered([("m", 0, 6)], [("m", 0, 5)]))
        self.assertFalse(spans_covered([("m", 0, 5)], [("m", 0, 2), ("m", 2, 5)][:1]))

    def test_overlap_detects_shared_range(self):
        self.assertTrue(spans_overlap([("m", 0, 5)], [("m", 4, 9)]))
        self.assertFalse(spans_overlap([("m", 0, 5)], [("m", 5, 9)]))


class PrepareSourcesTests(unittest.TestCase):
    def test_touching_chunks_of_one_message_merge(self):
        chunks = [
            chunk("a", "m1", 0, 10, "0123456789"),
            chunk("b", "m1", 10, 20, "abcdefghij", ordinal=1),
        ]
        pool, provenance, bundles = prepare_sources(["a", "b"], chunks, {"a": 1.0, "b": 0.5}, CodingConfig())
        merged = [hit for hit in pool if hit.id in bundles]
        self.assertEqual(len(merged), 1)
        self.assertIn("0123456789abcdefghij", merged[0].content)
        self.assertEqual(provenance[merged[0].id], (("m1", 0, 20),))
        # Members remain available even though the bundle was emitted first.
        self.assertIn("a", {hit.id for hit in pool})
        self.assertIn("b", {hit.id for hit in pool})

    def test_non_touching_chunks_stay_separate(self):
        chunks = [
            chunk("a", "m1", 0, 5, "alpha"),
            chunk("b", "m1", 90, 95, "omega", ordinal=1),
        ]
        pool, _, bundles = prepare_sources(["a", "b"], chunks, {"a": 1.0, "b": 1.0}, CodingConfig())
        self.assertEqual(bundles, set())
        self.assertEqual(len(pool), 2)

    def test_distinct_messages_are_never_merged(self):
        chunks = [
            chunk("a", "m1", 0, 5, "alpha"),
            chunk("b", "m2", 5, 10, "beta"),
        ]
        pool, _, bundles = prepare_sources(["a", "b"], chunks, {"a": 1.0, "b": 1.0}, CodingConfig())
        self.assertEqual(bundles, set())
        self.assertEqual(len(pool), 2)


if __name__ == "__main__":
    unittest.main()
