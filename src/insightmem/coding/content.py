"""Deterministic content types, not factual truth or execution-success labels."""
from __future__ import annotations

import json
import re

from .parse import code_tokens

LABEL = re.compile(r"^\[source role=[^\]\n]+ kind=[^\]\n]+\] ")
CALL = re.compile(r"^\[tool_use\s+([^\]]+)\]\s*")
OBSERVATION = re.compile(
    r"^\s*(?:=+\s*)?\d+\s+(?:passed|failed|errors?)\b"
    r"|^\s*(?:Traceback \(most recent call last\):|FAILED\b|OK\s*$)"
    r"|^[A-Za-z_][A-Za-z0-9_()., /-]{0,100}\s*(?:=|:)\s*(?:None|True|False|FiniteSet\([^\n]+\)|\{[-\d, ]+\})\s*$",
    re.M,
)
DIFF = re.compile(r"^(?:diff --git |@@ .+ @@|\*\*\* Begin Patch)|^--- [ab]/|^\+\+\+ [ab]/", re.M)
LINE_NUMBER = re.compile(r"^\s*\d+[\t:|]\s?", re.M)
TRANSITION = re.compile(
    r"^(?:\[thinking\]\s*)?(?:(?:Now|Next)[, ]+)?(?:Let me|I will|I.ll) "
    r"(?:inspect|read|look|check|run|find|search)\b[^\n`=:<>]{0,140}[.!]?$",
    re.I,
)

STOP = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "in", "to", "for", "from",
    "with", "on", "this", "that", "it", "as", "be", "by", "source", "role", "kind", "assistant",
    "user", "tool", "result", "historical", "memory", "scope", "summary",
}


def content_type(text: str) -> str:
    text = LABEL.sub("", text, count=1)
    match = CALL.match(text)
    if match:
        tool = match[1].casefold()
        try:
            payload = json.loads(text[match.end():])
        except (TypeError, ValueError):
            # A chunk can cut serialized parameters. Unknown is retained.
            return "unknown_call"
        if not isinstance(payload, dict):
            return "unknown_call"
        if tool in {"edit", "multiedit", "write", "apply_patch"}:
            if any(isinstance(payload.get(key), str) and payload[key] for key in ("new_string", "old_string", "content", "patch")) or isinstance(payload.get("edits"), list) and payload["edits"]:
                return "patch"
        command = payload.get("command")
        if isinstance(command, str) and DIFF.search(command):
            return "patch"
        return "invocation"
    if DIFF.search(text):
        return "patch"
    if text.startswith("[tool_result]") and observation_text(text):
        return "observation"
    if TRANSITION.fullmatch(text.strip()):
        return "transition"
    return "source"


def observation_text(text: str) -> bool:
    """Visible stdout/stderr only; numbered source dumps are not observations."""
    for line in text.removeprefix("[tool_result]").splitlines():
        line = line.strip()
        if re.match(r"^\d+[\t:|]|^(?:assert |def |if |return |#|\[tool_use)", line):
            continue
        if OBSERVATION.search(line):
            return True
    return False


def typed_factor(text: str) -> float:
    """Down-weight bare invocations and transitions during selection."""
    return 0.15 if content_type(text) in {"invocation", "transition"} else 1.0


def query_features(text: str, query: str) -> set[str]:
    """Query terms present in the candidate, ignoring presentation line numbers."""
    wanted = set(code_tokens(query)) - STOP
    words = set(code_tokens(LINE_NUMBER.sub("", text))) - STOP
    return {"q:" + token for token in wanted & words}
