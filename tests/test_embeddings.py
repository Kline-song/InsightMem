import hashlib
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from insightmem.coding.config import CodingConfig
from insightmem.coding.service import CodingMemoryService
from insightmem.embeddings import DashScopeEmbedder, EmbeddingServiceError, build_embedder
from insightmem.store import MemoryMessage as M

BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class Entry:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class Response:
    def __init__(self, entries):
        self.data = entries


def vector_for(text, dimension):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * (dimension // len(digest) + 1))[:dimension]
    return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) + 1.0).tolist()


class FakeEmbeddings:
    def __init__(self, dimension=1024, error=None, sleep=0.0, mutate=None):
        self.dimension = dimension
        self.error = error
        self.sleep = sleep
        self.mutate = mutate
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(list(kwargs["input"]))
        if self.sleep:
            time.sleep(self.sleep)
        if self.error:
            raise self.error
        entries = [
            Entry(index, vector_for(text, self.dimension))
            for index, text in enumerate(kwargs["input"])
        ]
        if self.mutate:
            entries = self.mutate(entries)
        return Response(list(reversed(entries)))


class FakeClient:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def close(self):
        pass


def build(dimension=1024, concurrency=1, **fake_kwargs):
    embedder = DashScopeEmbedder(
        api_key="x" * 32, base_url=BASE, dimensions=dimension, concurrency=concurrency
    )
    fake = FakeEmbeddings(dimension=dimension, **fake_kwargs)
    embedder._client = FakeClient(fake)
    return embedder, fake


class DashScopeEmbedderTests(unittest.TestCase):
    def test_rejects_invalid_configuration(self):
        cases = [
            dict(api_key="", base_url=BASE),
            dict(api_key="a b", base_url=BASE),
            dict(api_key="k", base_url="http://dashscope.aliyuncs.com/compatible-mode/v1"),
            dict(api_key="k", base_url="https://example.com/compatible-mode/v1"),
            dict(api_key="k", base_url=BASE + "?region=x"),
            dict(api_key="k", base_url=BASE, dimensions=100),
            dict(api_key="k", base_url=BASE, batch_size=11),
            dict(api_key="k", base_url=BASE, max_retries=6),
            dict(api_key="k", base_url=BASE, concurrency=0),
            dict(api_key="k", base_url=BASE, concurrency=17),
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    DashScopeEmbedder(**case)

    def test_identity_records_endpoint_and_dimensions(self):
        embedder, _ = build(dimension=512)
        self.assertEqual(
            embedder.identity,
            f"dashscope|text-embedding-v4|dimensions=512|endpoint={BASE}|normalization=l2-v1|sanitize=c0-v1",
        )

    def test_empty_input_makes_no_request(self):
        embedder, fake = build()
        result = embedder.encode_documents([])
        self.assertEqual(result.shape, (0, 1024))
        self.assertEqual(fake.calls, [])

    def test_batches_of_ten_preserve_order_and_normalize(self):
        embedder, fake = build()
        texts = [f"text-{index}" for index in range(11)]
        result = embedder.encode_documents(texts)
        self.assertEqual([len(call) for call in fake.calls], [10, 1])
        self.assertEqual(result.shape, (11, 1024))
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, rtol=1e-5)
        expected = np.asarray(vector_for("text-3", 1024), dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result[3], expected, rtol=1e-5, atol=1e-6)

    def test_concurrent_batches_preserve_order(self):
        embedder, fake = build(concurrency=4, sleep=0.01)
        texts = [f"concurrent-{index}" for index in range(25)]
        result = embedder.encode_documents(texts)
        self.assertEqual([len(call) for call in fake.calls], [10, 10, 5])
        for index in (0, 9, 10, 24):
            expected = np.asarray(vector_for(texts[index], 1024), dtype=np.float32)
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(result[index], expected, rtol=1e-5, atol=1e-6)

    def test_query_path_does_not_add_bge_prefix(self):
        embedder, fake = build()
        embedder.encode_queries(["where is the parser?"])
        self.assertEqual(fake.calls, [["where is the parser?"]])

    def test_control_characters_are_sanitized_for_embedding_only(self):
        embedder, fake = build()
        source = "alpha\x00beta\x01gamma\x7f"
        result = embedder.encode_documents([source])
        self.assertEqual(fake.calls, [["alpha beta gamma "]])
        self.assertEqual(source, "alpha\x00beta\x01gamma\x7f")
        self.assertEqual(result.shape, (1, 1024))

    def test_control_only_input_becomes_neutral_space(self):
        embedder, fake = build()
        result = embedder.encode_documents(["\x00\x01"])
        self.assertEqual(fake.calls, [[" "]])
        self.assertEqual(result.shape, (1, 1024))

    def test_blank_input_is_rejected_before_request(self):
        embedder, fake = build()
        for text in ("", " ", "\n\t"):
            with self.subTest(text=repr(text)):
                with self.assertRaises(ValueError):
                    embedder.encode_documents([text])
        self.assertEqual(fake.calls, [])

    def test_provider_error_maps_without_leaking_text(self):
        error = RuntimeError("secret-provider-detail")
        error.status_code = 429
        embedder, _ = build(error=error)
        with self.assertRaises(EmbeddingServiceError) as context:
            embedder.encode_documents(["private-query-text"])
        message = str(context.exception)
        self.assertIn("HTTP 429", message)
        self.assertEqual(context.exception.status_code, 429)
        self.assertNotIn("private-query-text", message)
        self.assertNotIn("secret-provider-detail", message)

    def test_invalid_provider_vectors_are_rejected(self):
        mutations = {
            "duplicate-index": lambda entries: [
                Entry(0, entries[0].embedding), Entry(0, entries[-1].embedding)
            ],
            "wrong-dimension": lambda entries: [
                Entry(entry.index, entry.embedding[:-1]) for entry in entries
            ],
            "nan": lambda entries: [
                Entry(entry.index, [float("nan")] + entry.embedding[1:]) for entry in entries
            ],
            "zero": lambda entries: [
                Entry(entry.index, [0.0] * len(entry.embedding)) for entry in entries
            ],
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                embedder, _ = build(mutate=mutate)
                with self.assertRaises(EmbeddingServiceError):
                    embedder.encode_documents(["alpha", "beta"])

    def test_key_alias_and_auto_backend(self):
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith("EMBEDDING_")
            and key not in {"DASHSCOPE_API_KEY", "MEMORY_EMBED_API_KEY", "DASHSCOPE_BASE_URL"}
        }
        environment.update(
            MEMORY_EMBED_API_KEY="z" * 32,
            EMBEDDING_BACKEND="auto",
            EMBEDDING_MODEL="text-embedding-v4",
            DASHSCOPE_BASE_URL=BASE,
        )
        with patch.dict(os.environ, environment, clear=True):
            embedder = build_embedder("text-embedding-v4")
        self.assertIsInstance(embedder, DashScopeEmbedder)
        self.assertEqual(embedder.dimensions, 1024)
        self.assertEqual(embedder.concurrency, 1)

    def test_dashscope_rejects_bge_query_prefix(self):
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith("EMBEDDING_")
            and key not in {"DASHSCOPE_API_KEY", "MEMORY_EMBED_API_KEY", "DASHSCOPE_BASE_URL"}
        }
        environment.update(
            DASHSCOPE_API_KEY="z" * 32,
            EMBEDDING_BACKEND="dashscope",
            DASHSCOPE_BASE_URL=BASE,
        )
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(ValueError):
                build_embedder("text-embedding-v4", query_prefix="Represent this sentence: ")


class ServiceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "coding.db"

    def tearDown(self):
        self.temporary.cleanup()

    def test_dense_service_round_trip_and_identity_guard(self):
        embedder, fake = build()
        service = CodingMemoryService(self.path, embedder, CodingConfig())
        service.add("r1", "u", "s", [M("user", "alpha beta gamma")])
        self.assertTrue(service.search("u", "alpha", 10))
        self.assertTrue(fake.calls)
        CodingMemoryService(self.path, embedder, CodingConfig())

        other, _ = build(dimension=512)
        with self.assertRaises(ValueError):
            CodingMemoryService(self.path, other, CodingConfig())

    def test_whitespace_only_chunk_is_not_embedded(self):
        embedder, fake = build()
        service = CodingMemoryService(self.path, embedder, CodingConfig())
        service.add("r1", "u", "s", [M("user", "   \n\t ")])
        self.assertEqual(fake.calls, [])
        with service._connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM view_embeddings").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
