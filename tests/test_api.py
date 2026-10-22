import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


class ApiEmbedder:
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


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is installed only in the server environment")
class ApiContractTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from insightmem.api import app, get_store
        from insightmem.memory_store import SQLiteMemoryStore

        self.directory = tempfile.TemporaryDirectory()
        self.store = SQLiteMemoryStore(Path(self.directory.name) / "api.db", ApiEmbedder())
        app.dependency_overrides[get_store] = lambda: self.store
        os.environ["MEMORY_API_KEY"] = "test-secret"
        os.environ["SKIP_MODEL_STARTUP"] = "true"
        self.app = app
        self.client = TestClient(app)
        self.headers = {"X-Api-Key": "test-secret"}

    def tearDown(self):
        self.client.close()
        self.app.dependency_overrides.clear()
        os.environ.pop("MEMORY_API_KEY", None)
        os.environ.pop("SKIP_MODEL_STARTUP", None)
        self.directory.cleanup()

    def test_add_search_and_health_contract(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        payload = {
            "request_id": "r1",
            "messages": [{"role": "user", "content": "Ada Lovelace wrote notes."}],
            "user_id": "u1",
            "session_id": "s1",
        }
        response = self.client.post("/add", json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"success": True, "request_id": "r1", "user_id": "u1", "session_id": "s1"},
        )
        result = self.client.post(
            "/search",
            json={"query": "Ada Lovelace", "user_id": "u1", "top_k": 100},
            headers=self.headers,
        )
        self.assertEqual(result.status_code, 200)
        self.assertIn("Ada Lovelace", result.json()["data"][0]["content"])

    def test_add_accepts_full_contract_roles_and_large_batches(self):
        messages = [
            {
                "role": "system" if index == 0 else "tool",
                "content": f"coding trace {index}",
                "timestamp": 1704067200000 + index,
            }
            for index in range(21)
        ]

        response = self.client.post(
            "/add",
            json={
                "request_id": "full-contract-r1",
                "messages": messages,
                "user_id": "full-contract-u1",
                "session_id": "full-contract-s1",
            },
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["request_id"], "full-contract-r1")

    def test_validation_diagnostics_do_not_expose_request_content(self):
        private_content = "PRIVATE-CONTENT-MUST-NOT-APPEAR"

        with self.assertLogs(
            "insightmem.validation",
            level="WARNING",
        ) as captured:
            response = self.client.post(
                "/add",
                json={
                    "request_id": "invalid-contract-r1",
                    "messages": [
                        {
                            "role": "",
                            "content": private_content,
                        }
                    ],
                    "user_id": "invalid-contract-u1",
                    "session_id": "invalid-contract-s1",
                },
                headers=self.headers,
            )

        self.assertEqual(response.status_code, 422)
        combined = response.text + "\n".join(captured.output)
        self.assertNotIn(private_content, combined)

        errors = response.json()["detail"]
        self.assertTrue(errors)
        for error in errors:
            self.assertEqual(set(error), {"loc", "type", "msg"})

    def test_authentication_and_validation(self):
        response = self.client.post(
            "/search", json={"query": "q", "user_id": "u1", "top_k": 100}
        )
        self.assertEqual(response.status_code, 401)
        response = self.client.post(
            "/search",
            json={"query": "q", "user_id": "u1", "top_k": 101},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
