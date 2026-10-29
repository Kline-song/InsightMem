# InsightMem

InsightMem is a synchronous Agent Memory Leaderboard (AML) compatible memory
service for coding-agent sessions. It stores every submitted message verbatim,
indexes evidence at chunk level, and returns the most useful evidence for a
query. Answers are produced by the evaluation platform; this service only
performs Add and Search. Current release: `0.2`.

## What it does

- `POST /add` persists a request atomically: messages, chunks, index views and
  their embeddings become searchable as soon as the response returns.
- `POST /search` fuses three retrieval legs — source-chunk BM25, code-expanded
  view BM25 and dense cosine similarity — with reciprocal rank fusion, merges
  adjacent slices of one message, drops exact duplicates and pure process
  narration, then orders the remaining evidence by relevance and query coverage.
- Retrieval is deterministic: the same database and the same query produce the
  same result, independent of insertion timing.
- Message filtering is query-only. Hidden assistant narration stays in the BM25
  corpus and keeps its rank contributions, so surviving evidence keeps the
  scores it would have had without filtering.

## Quick start

Requires Python 3.10 or later. For the cloud embedding backend:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-api.txt -c constraints-api.txt
cp .env.example .env
```

Fill `.env` locally (it is ignored by git):

```dotenv
EMBEDDING_BACKEND=dashscope
EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_API_KEY=your-private-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_DIMENSIONS=1024
EMBEDDING_BATCH_SIZE=10
EMBEDDING_CONCURRENCY=4
MEMORY_API_KEY=your-independent-random-service-key
```

Then start the API:

```bash
PYTHONPATH=src uvicorn insightmem.api:app --host 127.0.0.1 --port 8000 --workers 1
```

For a local sentence-transformers model install `requirements.txt` and leave
`EMBEDDING_BACKEND=auto` with `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`.

### API

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/add \
  -H "X-Api-Key: $MEMORY_API_KEY" -H 'Content-Type: application/json' \
  -d '{"request_id":"r1","user_id":"u1","session_id":"s1",
       "messages":[{"role":"user","content":"The parser drops backslashes."}]}'

curl -s -X POST localhost:8000/search \
  -H "X-Api-Key: $MEMORY_API_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"why does the parser fail","user_id":"u1","top_k":10}'
```

`user_id` is the only retrieval-isolation boundary. See
[docs/api-contract.md](docs/api-contract.md) for the full contract.

## Retrieval profile

`configs/coding/default.json` sets chunking, fusion and evidence limits:

| Key | Default | Meaning |
|---|---:|---|
| `chunk_units` | 320 | tokens per source chunk |
| `overlap_units` | 40 | overlap between adjacent chunks |
| `max_chunk_chars` | 6000 | hard character cap per chunk |
| `rrf_k` | 60 | reciprocal-rank-fusion constant |
| `leg_max` | 200 | maximum candidates per retrieval leg |
| `evidence_bundle_chars` | 4000 | merge limit for adjacent slices |
| `evidence_max_chars` | 12000 | maximum characters per evidence item |

Set `CODING_CONFIG_PATH` to load a different profile. A database records the
profile fingerprint and the embedding identity; a mismatch is refused at
startup, so changing either requires a fresh database.

## Embeddings

The DashScope `text-embedding-v4` backend splits inputs into batches of ten,
issues batches concurrently, normalizes vectors locally, rejects the BGE query
instruction and replaces control characters for embedding only. Its identity
(`model|dimensions|endpoint|normalization|sanitize`) is part of the database
build fingerprint. `MEMORY_EMBED_API_KEY` is accepted as a fallback for
`DASHSCOPE_API_KEY`.

A small live check (about 13 requests) verifies endpoint, dimensions, batching,
concurrency, control characters and a dense Add/Search round trip:

```bash
PYTHONPATH=src python scripts/verify_dashscope.py
```

Deployment preflight (no API calls):

```bash
PYTHONPATH=src python scripts/server_preflight.py
```

See [docs/text-embedding-v4.md](docs/text-embedding-v4.md) for the interface
details and operational notes.

## Docker

```bash
docker compose up --build
```

The image installs the pinned API dependency set (`requirements-api.txt` with
`constraints-api.txt`), runs `uvicorn insightmem.api:app` with
`PYTHONPATH=/app/src` and persists the database under `/app/artifacts`. The
local sentence-transformers backend is installed from `requirements.txt` in a
virtual environment, not in this image.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite covers lexical scoring, parsing and chunking, the assistant process
filter, evidence merging and deduplication, the embedding client (with a mocked
provider), the service round trip and the HTTP contract.

## Operations

Deployment parameters, logging, data retention and the purge procedure are in
[docs/operations.md](docs/operations.md).

## License and third-party provenance

Original contributions are licensed under the MIT license in [LICENSE](LICENSE).
Portions derived from third-party work are identified in [NOTICE.md](NOTICE.md)
and are not relicensed by that file.
