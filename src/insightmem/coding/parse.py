"""Deterministic message parsing. Historical commands are never executed."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..text import TOKEN_PATTERN
from .config import stable_id

CALL = re.compile(r"^\[tool_use\s+([^\]]+)\]\s*")

# Whole-message transitions that only narrate the next step.
TRANSITION = re.compile(
    r"(?:Now |Next,? )?(?:let me|I will|I\x27ll) "
    r"(?:run the tests|inspect the code|look at the code|check the implementation|check the tests)\.?",
    re.I,
)
# Concrete content that a transition must never swallow.
PROTECT = re.compile(
    r"(?:[\w.-]+[/\\][\w./\\-]+|\w+_\w+|\w+\.\w+|\d|\b(?:must|unless|only|not|never|because|error|failed|failure|regression|constraint|None|zero|empty)\b)",
    re.I,
)


@dataclass(frozen=True)
class ParsedMessage:
    message_id: str
    kind: str


def parse(user: str, request: str, messages) -> list[ParsedMessage]:
    """Classify messages and assign stable ids; content is never rewritten."""
    result = []
    for index, message in enumerate(messages):
        message_id = stable_id("message", user, request, index)
        kind = "unknown"
        if CALL.match(message.content):
            kind = "tool_call"
        elif message.content.startswith("[tool_result]"):
            kind = "tool_result"
        elif message.content.startswith("[thinking]"):
            kind = "thinking"
        elif index == 0 and message.role == "user":
            kind = "issue_prompt"
        elif message.role == "assistant":
            kind = "assistant_text"
        result.append(ParsedMessage(message_id, kind))
    return result


def chunk_spans(text: str, units: int, overlap: int, max_chars: int) -> list[tuple[int, int]]:
    """Cover every character, including Unicode and whitespace; precise offsets."""
    if not text:
        return []
    matches = list(TOKEN_PATTERN.finditer(text))
    ranges = []
    position = 0
    while position < len(text):
        end = min(len(text), position + max_chars)
        tokens = [x for x in matches if position <= x.start() < end]
        if len(tokens) > units:
            end = tokens[units].start()
        if end <= position:
            end = min(len(text), position + max_chars)
        ranges.append((position, end))
        if end == len(text):
            break
        eligible = [x for x in tokens if x.start() < end]
        following = eligible[-overlap].start() if overlap and len(eligible) > overlap else end
        position = following if following > position else end
    return ranges


def code_tokens(text: str) -> list[str]:
    """Split identifiers and paths so a lexical leg can match code vocabulary."""
    words = re.findall(r"[\w]+(?:[./\\:-][\w.-]+)*", text, re.UNICODE)
    result = []
    for word in words:
        result.append(word.replace("\\", "/").casefold())
        parts = re.split(r"[/\\.:_-]+", word)
        for part in parts:
            result.append(part.casefold())
            result.extend(x.casefold() for x in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+", part))
    return result
