import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import insightmem.api as api


def vector_for(text, dimension=64):
    import hashlib
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * (dimension // len(digest) + 1))[:dimension]
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float32) + 1.0


class HashEmbedder:
    identity = "hash|dimensions=64"

    def encode_documents(self, texts):
        return np.asarray([vector_for(text) for text in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray([vector_for(text) for text in texts], dtype=np.float32)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = {
            "MEMORY_API_KEY": "k" * 32,
            "MEMORY_DB_PATH": str(Path(self.temporary.name) / "memory.db"),
            "EMBEDDING_MODEL": "test-model",
        }
        api.get_store.cache_clear()
        self.patch = patch.dict(os.environ, self.environment, clear=False)
        self.patch.start()
        self.embedder_patch = patch("insightmem.api.build_embedder", return_value=HashEmbedder())
        self.embedder_patch.start()
        self.client = TestClient(api.app)

    def tearDown(self):
        api.get_store.cache_clear()
        self.embedder_patch.stop()
        self.patch.stop()
        self.temporary.cleanup()

    def headers(self):
        return {"X-Api-Key": "k" * 32}

    def add(self, request_id="r1", user="u"):
        return self.client.post("/add", headers=self.headers(), json={
            "request_id": request_id,
            "user_id": user,
            "session_id": "s1",
            "messages": [{"role": "user", "content": "The parser drops backslashes."}],
        })

    def test_health_is_open(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_add_and_search_require_a_key(self):
        unauthenticated = self.client.post("/add", json={
            "request_id": "r1", "user_id": "u", "session_id": "s1",
            "messages": [{"role": "user", "content": "x"}],
        })
        self.assertEqual(unauthenticated.status_code, 401)
        with patch.dict(os.environ, {"MEMORY_API_KEY": ""}, clear=False):
            response = self.client.post("/search", json={"query": "q", "user_id": "u", "top_k": 1})
        self.assertEqual(response.status_code, 503)
        response = self.client.post("/search", headers={"X-Api-Key": "wrong"},
                                    json={"query": "q", "user_id": "u", "top_k": 1})
        self.assertEqual(response.status_code, 401)

    def test_add_then_search(self):
        added = self.add()
        self.assertEqual(added.status_code, 200)
        self.assertEqual(added.json()["request_id"], "r1")
        found = self.client.post("/search", headers=self.headers(),
                                 json={"query": "parser", "user_id": "u", "top_k": 5})
        self.assertEqual(found.status_code, 200)
        self.assertTrue(found.json()["data"])

    def test_conflicting_request_id_returns_409(self):
        self.add()
        response = self.client.post("/add", headers=self.headers(), json={
            "request_id": "r1",
            "user_id": "u",
            "session_id": "s1",
            "messages": [{"role": "user", "content": "Different."}],
        })
        self.assertEqual(response.status_code, 409)

    def test_invalid_payload_returns_422(self):
        response = self.client.post("/add", headers=self.headers(), json={"request_id": "r1"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
