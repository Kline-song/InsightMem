from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from .local_smoke import request_json
except ImportError:
    from local_smoke import request_json


def split_oversized_message(message: dict[str, object], max_words: int) -> list[dict[str, object]]:
    content = str(message["content"])
    matches = list(re.finditer(r"\S+", content))
    if len(matches) <= max_words:
        return [message]
    parts: list[dict[str, object]] = []
    for start in range(0, len(matches), max_words):
        end = min(start + max_words, len(matches))
        part = dict(message)
        part["content"] = content[matches[start].start() : matches[end - 1].end()]
        parts.append(part)
    return parts


def chunk_messages(messages: list[dict[str, object]], max_messages: int = 20, max_words: int = 2000) -> list[list[dict[str, object]]]:
    batches: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    words = 0
    expanded = [part for message in messages for part in split_oversized_message(message, max_words)]
    for message in expanded:
        message_words = len(str(message["content"]).split())
        if current and (len(current) >= max_messages or words + message_words > max_words):
            batches.append(current)
            current = []
            words = 0
        current.append(message)
        words += message_words
    if current:
        batches.append(current)
    return batches


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay normalized memory benchmark records against the HTTP API")
    parser.add_argument("--input", required=True, help="Normalized JSONL benchmark file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--run-label", default=None, help="Namespace used to isolate repeated benchmark runs")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    output = Path(args.output)
    run_label = args.run_label or Path(args.input).stem
    output.parent.mkdir(parents=True, exist_ok=True)
    question_count = 0
    answer_in_context = 0
    answer_count = 0
    evidence_recall_total = 0.0
    evidence_recall_count = 0

    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            record_id = str(record["id"])
            user_id = f"local-benchmark:{run_label}:{record_id}"
            for chunk_index, messages in enumerate(chunk_messages(record["messages"])):
                payload = {
                    "request_id": f"local-benchmark:{run_label}:{record_id}:chunk-{chunk_index}",
                    "messages": messages,
                    "user_id": user_id,
                    "session_id": str(record.get("session_id", record_id)),
                }
                status, body = request_json("POST", f"{base_url}/add", payload, args.api_key, args.timeout)
                if status != 200 or body.get("success") is not True:
                    raise RuntimeError(f"Add failed for {record_id} chunk {chunk_index}: HTTP {status} {body}")

            for question in record["questions"]:
                payload = {
                    "query": question["question"],
                    "user_id": user_id,
                    "top_k": args.top_k,
                }
                if question.get("options"):
                    payload["options"] = question["options"]
                status, body = request_json("POST", f"{base_url}/search", payload, args.api_key, args.timeout)
                if status != 200:
                    raise RuntimeError(f"Search failed for {record_id}: HTTP {status} {body}")
                context = "\n".join(item["content"] for item in body["data"])
                answer = str(question.get("answer", ""))
                hit = bool(answer and normalize(answer) in normalize(context))
                answer_count += int(bool(answer))
                evidence = [str(value) for value in question.get("evidence_texts", []) if str(value)]
                evidence_recall = None
                if evidence:
                    evidence_recall = sum(normalize(value) in normalize(context) for value in evidence) / len(evidence)
                    evidence_recall_total += evidence_recall
                    evidence_recall_count += 1
                question_count += 1
                answer_in_context += int(hit)
                stream.write(
                    json.dumps(
                        {
                            "record_id": record_id,
                            "question_id": question.get("id"),
                            "question": question["question"],
                            "answer": answer,
                            "answer_in_context": hit if answer else None,
                            "evidence_recall_at_k": evidence_recall,
                            "retrieved": body["data"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        "records": len(records),
        "questions": question_count,
        "answer_in_context_rate": answer_in_context / answer_count if answer_count else None,
        "evidence_recall_at_k": evidence_recall_total / evidence_recall_count if evidence_recall_count else None,
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
