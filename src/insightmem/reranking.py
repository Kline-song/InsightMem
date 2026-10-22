from __future__ import annotations

import threading
from typing import Protocol

import numpy as np


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        if batch_size <= 0:
            raise ValueError("reranker batch_size must be positive")
        if max_length <= 0:
            raise ValueError("reranker max_length must be positive")
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs = {"torch_dtype": torch.float16} if resolved_device.startswith("cuda") else {}
        self.model = CrossEncoder(
            model_name,
            device=resolved_device,
            max_length=max_length,
            model_kwargs=model_kwargs,
            trust_remote_code=False,
        )
        self.batch_size = batch_size
        self._lock = threading.RLock()

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float32)
        pairs = [(query, document) for document in documents]
        with self._lock:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def build_reranker(
    model_name: str,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 512,
) -> Reranker:
    return CrossEncoderReranker(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
