"""Tokenization shared by lexical scoring and chunking."""
from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[一-鿿]|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)


def lexical_tokens(text: str) -> list[str]:
    """Case-folded word, CJK and punctuation tokens for BM25."""
    return [
        token.casefold()
        for token in TOKEN_PATTERN.findall(text)
        if any(character.isalnum() or character == "_" for character in token)
    ]
