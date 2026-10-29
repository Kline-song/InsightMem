from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol
from urllib.parse import urlsplit

import numpy as np

from .config import load_project_env


class Embedder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local Hugging Face embedding backend for server-side benchmark runs."""

    def __init__(self, model_name: str, device: str | None = None, query_prefix: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        if query_prefix is None and "bge" in model_name.lower() and "-en-" in model_name.lower():
            query_prefix = "Represent this sentence for searching relevant passages: "
        self.query_prefix = query_prefix or ""
        self.identity = f"{model_name}|query_prefix={self.query_prefix}"
        self.model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + text for text in texts])


class EmbeddingServiceError(RuntimeError):
    """Provider failure with no credentials, input text, or raw response attached."""


class DashScopeEmbedder:
    """Dense text-embedding-v4 over the official OpenAI-compatible HTTPS API."""

    def __init__(
        self, *, api_key: str, base_url: str, dimensions: int = 1024,
        batch_size: int = 10, timeout: float = 60, max_retries: int = 2,
        concurrency: int = 1,
    ) -> None:
        from openai import OpenAI

        if not api_key or any(c.isspace() for c in api_key):
            raise ValueError("DASHSCOPE_API_KEY/MEMORY_EMBED_API_KEY is missing or malformed")
        try:
            url = urlsplit(base_url)
            valid_url = (
                url.scheme == "https" and (url.hostname or "").endswith(".aliyuncs.com")
                and not url.username and not url.password and not url.query and not url.fragment
                and url.path.rstrip("/") == "/compatible-mode/v1"
            )
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError("DASHSCOPE_BASE_URL must be an official HTTPS compatible-mode/v1 endpoint")
        if dimensions not in {64, 128, 256, 512, 768, 1024, 1536, 2048}:
            raise ValueError("Unsupported text-embedding-v4 dimension")
        if not 1 <= batch_size <= 10 or timeout <= 0 or not 0 <= max_retries <= 5:
            raise ValueError("Invalid embedding batch size, timeout, or retry limit")
        if type(concurrency) is not int or not 1 <= concurrency <= 16:
            raise ValueError("Invalid embedding concurrency")
        self.model_name = "text-embedding-v4"
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.identity = (
            f"dashscope|{self.model_name}|dimensions={dimensions}"
            f"|endpoint={base_url.rstrip('/')}|normalization=l2-v1|sanitize=c0-v1"
        )
        try:
            self._client = OpenAI(
                api_key=api_key, base_url=base_url.rstrip("/"),
                timeout=timeout, max_retries=max_retries,
            )
        except Exception:
            raise ValueError("Could not initialize DashScope embedding client") from None

    def _prepare(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input must contain non-empty strings")
        # Traces may legitimately contain NUL/DEL/C0 controls.  They carry no
        # embedding signal and some providers reject them in JSON input.
        cleaned = "".join(
            " " if (ord(ch) < 32 and ch not in "\n\t\r") or ord(ch) == 127 else ch
            for ch in text
        )
        return cleaned if cleaned.strip() else " "

    def _encode_batch(self, batch: list[str]) -> np.ndarray:
        try:
            response = self._client.embeddings.create(
                model=self.model_name, input=batch, dimensions=self.dimensions,
                encoding_format="float",
            )
        except Exception as error:
            status = getattr(error, "status_code", None)
            suffix = f" (HTTP {status})" if type(status) is int else ""
            mapped = EmbeddingServiceError("DashScope embedding request failed" + suffix)
            if type(status) is int:
                mapped.status_code = status
            raise mapped from None
        try:
            entries = response.data
            indices = [item.index for item in entries]
            if (len(entries) != len(batch) or any(type(i) is not int for i in indices)
                    or set(indices) != set(range(len(batch)))):
                raise ValueError()
            vectors = np.asarray(
                [item.embedding for item in sorted(entries, key=lambda item: item.index)],
                dtype=np.float32,
            )
            if vectors.shape != (len(batch), self.dimensions) or not np.isfinite(vectors).all():
                raise ValueError()
            norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError()
            return (vectors / norms).astype(np.float32)
        except Exception:
            raise EmbeddingServiceError("DashScope returned invalid embedding vectors") from None

    def _encode(self, texts: list[str]) -> np.ndarray:
        prepared = [self._prepare(text) for text in texts]
        if not prepared:
            return np.empty((0, self.dimensions), dtype=np.float32)
        batches = [
            prepared[start:start + self.batch_size]
            for start in range(0, len(prepared), self.batch_size)
        ]
        if self.concurrency == 1 or len(batches) == 1:
            results = [self._encode_batch(batch) for batch in batches]
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                # map preserves batch order, so row order matches the input.
                results = list(pool.map(self._encode_batch, batches))
        return np.concatenate(results, axis=0)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        # Do not carry over the BGE-specific query instruction.
        return self._encode(texts)

    def close(self) -> None:
        self._client.close()


def build_embedder(model_name: str, device: str | None = None, query_prefix: str | None = None) -> Embedder:
    load_project_env()
    backend = os.environ.get("EMBEDDING_BACKEND", "auto").strip().lower()
    if backend == "auto":
        backend = "dashscope" if model_name == "text-embedding-v4" else "local"
    if backend == "dashscope":
        if model_name != "text-embedding-v4":
            raise ValueError("DashScope backend requires EMBEDDING_MODEL=text-embedding-v4")
        if query_prefix:
            raise ValueError("DashScope backend does not accept the local-model query prefix")
        return DashScopeEmbedder(
            api_key=(os.environ.get("DASHSCOPE_API_KEY")
                     or os.environ.get("MEMORY_EMBED_API_KEY", "")),
            base_url=os.environ.get("DASHSCOPE_BASE_URL", ""),
            dimensions=int(os.environ.get("EMBEDDING_DIMENSIONS", "1024")),
            batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "10")),
            timeout=float(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.environ.get("EMBEDDING_MAX_RETRIES", "2")),
            concurrency=int(os.environ.get("EMBEDDING_CONCURRENCY", "1")),
        )
    if backend != "local" or model_name == "text-embedding-v4":
        raise ValueError("Invalid EMBEDDING_BACKEND for the selected model")
    return SentenceTransformerEmbedder(model_name=model_name, device=device, query_prefix=query_prefix)
