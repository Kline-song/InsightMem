# InsightMem

Version: **v0.1**

InsightMem provides persistent memory storage and retrieval through an
Agent Memory Leaderboard compatible synchronous Add/Search API.
It supports cloud `text-embedding-v4` embeddings, SQLite persistence,
dense retrieval, BM25/RRF hybrid retrieval, and optional local embedding models.
The API returns retrieved evidence; the evaluation platform handles Answer/Eval.

## Quick start

Requires Python 3.10 or later. For the cloud embedding API:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-api.txt
cp .env.example .env
```

Configure the ignored `.env` locally:

```dotenv
EMBEDDING_BACKEND=dashscope
EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_API_KEY=your-private-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_DIMENSIONS=1024
MEMORY_API_KEY=your-independent-random-service-key
MEMORY_DB_PATH=artifacts/memory.db
MEMORY_RETRIEVAL_MODE=hybrid
MEMORY_RERANKER_MODEL=
```

The DashScope key must match the endpoint region. Process environment variables
take precedence over `.env`. Changing embedding models or dimensions requires
a fresh database and re-indexing.

```bash
PYTHONPATH=src uvicorn insightmem.api:app --host 127.0.0.1 --port 8000 --workers 1
```

For local embedding models, reranking, and offline benchmark tools, install
`requirements.txt` instead. Docker is also supported:

```bash
docker build -t insightmem:0.1 .
docker compose up -d
```

## API

| Method | Path | Authentication |
|---|---|---|
| GET | `/health` | None |
| POST | `/add` | Bearer, Token, or X-Api-Key |
| POST | `/search` | Bearer, Token, or X-Api-Key |

Use `Authorization: Bearer <MEMORY_API_KEY>`. Never expose the DashScope key.

```json
{
  "request_id": "unique-write-id",
  "user_id": "user-1",
  "session_id": "session-1",
  "messages": [{"role": "user", "content": "I prefer jasmine tea."}]
}
```

Add returns HTTP 200 after persistence and echoes the submitted identifiers.
Identical `request_id` retries are idempotent; conflicting payloads return 409.

```json
{"user_id": "user-1", "query": "What tea do I prefer?", "top_k": 100}
```

Search returns `{"data": [{"id": "...", "content": "...", "score": 0.0}]}`.
Retrieval is isolated by exact `user_id`, across that user's sessions.

## Deployment

Run one API worker initially, behind an HTTPS reverse proxy. Keep the database
on persistent storage, credentials outside Git, and service access authenticated.
Use a service manager for restart recovery and automatic certificate renewal.
Start with Add concurrency=1 and Search concurrency=1; size the deployment
using measured memory, storage, throughput, and upstream API limits.
Local checks do not constitute official AML Smoke/Full results.

## Tools and validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m insightmem.cli --help
python scripts/local_smoke.py --help
python scripts/memory_admin.py --help
```

Do not commit `.env`, API keys, databases, model caches, or evaluation payloads.
See [the API contract notes](docs/competition-contract.md) and
[third-party provenance](NOTICE.md).
