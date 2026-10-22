from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import chunk_documents
from .datasets import convert_hotpotqa, load_jsonl_documents, load_jsonl_queries
from .embeddings import build_embedder
from .evaluation import AggregateMetrics
from .generation import OpenAIAnswerGenerator
from .index import FaissChunkIndex
from .pipeline import InsightMemRAG


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def add_embedding_args(parser: argparse.ArgumentParser, default_model: str | None = None) -> None:
    parser.add_argument("--embedding-model", default=default_model)
    parser.add_argument("--device", default=None, help="SentenceTransformer device, e.g. cuda or cpu")


def build_generator(kind: str, chat_model: str):
    if kind == "none":
        return None
    return OpenAIAnswerGenerator(model=chat_model)


def cmd_index(args: argparse.Namespace) -> None:
    documents = load_jsonl_documents(args.documents)
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    embedder = build_embedder(args.embedding_model, args.device)
    index = FaissChunkIndex.build(chunks, embedder, args.embedding_model)
    index.metadata.update({"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap, "document_count": len(documents)})
    index.save(args.output)
    print(json.dumps({"index": str(args.output), "documents": len(documents), "chunks": len(chunks), **index.metadata}, ensure_ascii=False, indent=2))


def load_rag(index_dir: str, embedding_model: str | None, device: str | None, generator: str, chat_model: str) -> InsightMemRAG:
    index = FaissChunkIndex.load(index_dir)
    model = embedding_model or str(index.metadata["embedding_model"])
    return InsightMemRAG(index=index, embedder=build_embedder(model, device), generator=build_generator(generator, chat_model))


def serialise_result(result) -> dict[str, object]:
    return {
        "question": result.question,
        "answer": result.answer,
        "context": result.context_text,
        "retrieved": [
            {"chunk_id": chunk.id, "document_id": chunk.document_id, "title": chunk.title, "score": score, "text": chunk.text}
            for chunk, score in result.contexts
        ],
    }


def cmd_ask(args: argparse.Namespace) -> None:
    rag = load_rag(args.index, args.embedding_model, args.device, args.generator, args.chat_model)
    print(json.dumps(serialise_result(rag.ask(args.question, args.top_k)), ensure_ascii=False, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    rag = load_rag(args.index, args.embedding_model, args.device, args.generator, args.chat_model)
    queries = load_jsonl_queries(args.questions)
    metrics = AggregateMetrics()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for query in queries:
            result = rag.ask(query.question, args.top_k)
            row = serialise_result(result)
            row.update({"query_id": query.id, "gold_answer": query.answer, "relevant_doc_ids": query.relevant_doc_ids})
            row["metrics"] = metrics.add(query, [chunk.document_id for chunk, _ in result.contexts], result.context_text, result.answer)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics.summary(), ensure_ascii=False, indent=2))


def cmd_prepare_hotpot(args: argparse.Namespace) -> None:
    docs_path, questions_path = convert_hotpotqa(args.input, args.output_dir)
    print(json.dumps({"documents": str(docs_path), "questions": str(questions_path)}, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="InsightMem command-line tools")
    commands = root.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Chunk documents, embed them, and build a FAISS index")
    index.add_argument("--documents", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--chunk-size", type=int, default=320)
    index.add_argument("--chunk-overlap", type=int, default=60)
    add_embedding_args(index, DEFAULT_EMBEDDING_MODEL)
    index.set_defaults(func=cmd_index)

    ask = commands.add_parser("ask", help="Retrieve context and optionally generate an answer")
    ask.add_argument("--index", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--generator", choices=["none", "openai"], default="none")
    ask.add_argument("--chat-model", default="gpt-4o-mini")
    add_embedding_args(ask)
    ask.set_defaults(func=cmd_ask)

    evaluate = commands.add_parser("evaluate", help="Run retrieval or end-to-end evaluation")
    evaluate.add_argument("--index", required=True)
    evaluate.add_argument("--questions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--generator", choices=["none", "openai"], default="none")
    evaluate.add_argument("--chat-model", default="gpt-4o-mini")
    add_embedding_args(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    hotpot = commands.add_parser("prepare-hotpot", help="Convert official HotpotQA JSON to project JSONL")
    hotpot.add_argument("--input", required=True)
    hotpot.add_argument("--output-dir", required=True)
    hotpot.set_defaults(func=cmd_prepare_hotpot)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
