# Architecture

InsightMem keeps an immutable log of submitted messages and answers queries
from evidence extracted out of that log. There is no generation step and no
background worker: Add and Search are synchronous and deterministic.

## Storage

One SQLite database per service, isolated by `user_id` in every table.

| Table | Contents |
|---|---|
| `coding_requests` | request payload hash, per-user ingest sequence, chunk count |
| `source_messages` | verbatim message text, role, kind and timestamp |
| `source_chunks` | character spans of each message, used for lexical retrieval |
| `index_views` | retrieval representation and its text (currently the source slice) |
| `view_sources` | view-to-message span mapping |
| `view_embeddings` | normalized dense vector, dimension and embedding identity |

`coding_meta` stores the schema version, the retrieval-profile fingerprint and
the embedding identity. A database created with another profile or another
embedding model is refused at startup; the data must be rebuilt instead.

## Add

1. Hash `session_id` and the message list. A repeated `request_id` with the
   same payload returns the stored chunk count; a different payload raises a
   conflict.
2. Classify each message (issue prompt, assistant text, thinking, tool call,
   tool result) and assign stable ids.
3. Split every message into overlapping token-bounded chunks, capped by
   characters. Chunks are stored once; the retrieval view shares the same span.
4. Embed all non-whitespace chunks in one batched call and store the normalized
   vectors.
5. Commit everything in one transaction so a Search never observes a partial
   request.

## Search

1. Read one committed snapshot of the user's chunks, views, sources, vectors
   and message kinds.
2. Mark assistant messages that only narrate process. Their chunks stay in the
   corpus; only their eligibility for the result list is removed.
3. Score three legs:
   - source-chunk BM25 over raw message text;
   - view BM25 over code-expanded identifiers;
   - dense cosine similarity between the query vector and view vectors.
4. Fuse the legs with reciprocal rank fusion, counting each leg once per rank
   and keeping the best rank of every candidate. Duplicate bodies are moved
   behind unique ones rather than dropped.
5. Merge touching chunks of the same message into one evidence item, bounded by
   `evidence_bundle_chars`. Merged items never replace their members; span
   deduplication removes a member only when the bundle is actually returned.
6. Exclude messages marked in step 2 from the selectable candidates.
7. Order candidates by relevance, with query-term coverage as a tie-break, and
   down-weight bare tool invocations and transitions. Candidate text is never
   rewritten.

## Determinism

Ranking uses only the stored text, the stored vectors and the query. There is
no sampling, no wall-clock dependence and no learned model in the selection
path. Embeddings are the only non-deterministic component; with a fixed
provider and model, the same input yields the same output.
