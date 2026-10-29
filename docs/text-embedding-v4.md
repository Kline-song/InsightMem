# text-embedding-v4 backend

`DashScopeEmbedder` talks to the OpenAI-compatible DashScope endpoint. The
local sentence-transformers backend remains available for offline work.

## Interface

| Requirement | InsightMem implementation |
|---|---|
| OpenAI-compatible embedding endpoint | `openai` SDK against `.../compatible-mode/v1` |
| Model name | `EMBEDDING_MODEL=text-embedding-v4`; `EMBEDDING_BACKEND=auto` selects DashScope only for this exact name |
| Dimensions | `EMBEDDING_DIMENSIONS` (default 1024); allowed 64-2048 |
| Batch limit 10 | `EMBEDDING_BATCH_SIZE` is capped at 10 and inputs are split automatically |
| Query/doc asymmetry | no BGE query instruction is added; a configured `EMBEDDING_QUERY_PREFIX` is rejected |
| Normalization | vectors are L2-normalized locally before storage and search |
| Identity | `dashscope\|text-embedding-v4\|dimensions=...\|endpoint=...\|normalization=l2-v1\|sanitize=c0-v1` is stored in the database build fingerprint |
| Key source | `DASHSCOPE_API_KEY`, falling back to `MEMORY_EMBED_API_KEY` |
| Concurrency | `EMBEDDING_CONCURRENCY` (1-16, default 1); results keep input order |
| Retries | `EMBEDDING_MAX_RETRIES` (0-5, default 2) |

## Operational notes

1. **A fresh database is required when the embedding identity changes.** The
   fingerprint covers model, dimensions, endpoint and sanitization version.
   Opening a database built with another identity fails with
   `coding build/embedding mismatch`.
2. **Add latency is dominated by embedding requests.** Each Add issues one
   request per ten chunks. Use `EMBEDDING_CONCURRENCY=4` to overlap batches, and
   lower it if the account hits request-per-minute limits. A retry budget of 5
   is appropriate for long ingestion runs.
3. **Control characters and blank chunks.** NUL/DEL/C0 characters are replaced
   with spaces for embedding input only; stored text is unchanged.
   Whitespace-only chunks are kept for lexical retrieval but not embedded, so a
   blank message cannot turn an Add into a provider error.
4. **No local fallback for the cloud model.** `EMBEDDING_BACKEND=auto` with the
   default BGE model uses the local backend. Set the DashScope backend
   explicitly in production.
5. **Cost.** Inputs are billed per token; a persistent local exact-text cache
   avoids re-embedding repeated chunks across rebuilds.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/verify_dashscope.py   # about 13 live requests
```

The mocked suite covers batching, order preservation under concurrency,
normalization, response-shape rejection, error mapping without text or key
leaks, control-character sanitization, blank-input rejection, the database
identity guard and a dense Add/Search round trip.

## Live checklist

1. Confirm the endpoint region matches the key (Beijing or Singapore).
2. Run `verify_dashscope.py`; expect a query/document cosine near 1.0, a
   deterministic cosine near 1.0 and the 512-dimension variant accepted.
3. Record per-request latency and any 429 responses; choose concurrency.
4. Confirm the database fingerprint reports `dimensions=1024` and the
   DashScope endpoint before ingesting a large corpus.
