"""Chunking and ranking limits for the evidence service."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

PROFILE_VERSION = "coding-pilot-v2"


@dataclass(frozen=True)
class CodingConfig:
    """Tunable limits only; retrieval behavior itself is fixed."""

    chunk_units: int = 320
    overlap_units: int = 40
    max_chunk_chars: int = 6000
    rrf_k: int = 60
    leg_max: int = 200
    evidence_bundle_chars: int = 4000
    evidence_max_chars: int = 12000

    def __post_init__(self) -> None:
        if type(self.chunk_units) is not int or self.chunk_units <= 0:
            raise ValueError("chunk units must be a positive integer")
        if type(self.overlap_units) is not int or not 0 <= self.overlap_units < self.chunk_units:
            raise ValueError("chunk overlap must be smaller than chunk units")
        if type(self.max_chunk_chars) is not int or self.max_chunk_chars <= 0:
            raise ValueError("chunk character limit must be positive")
        if type(self.rrf_k) is not int or self.rrf_k <= 0:
            raise ValueError("rrf k must be positive")
        if type(self.leg_max) is not int or self.leg_max <= 0:
            raise ValueError("leg limit must be positive")
        if type(self.evidence_max_chars) is not int or self.evidence_max_chars <= 0:
            raise ValueError("evidence item limit must be positive")
        if type(self.evidence_bundle_chars) is not int or not 0 < self.evidence_bundle_chars <= self.evidence_max_chars:
            raise ValueError("evidence bundle limit must not exceed the item limit")

    @classmethod
    def load(cls, path) -> "CodingConfig":
        return cls(**json.loads(open(path).read()))

    def fingerprint(self, identity: str) -> str:
        return digest([asdict(self), PROFILE_VERSION, identity])


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def stable_id(*values) -> str:
    return "c1-" + digest(values)
