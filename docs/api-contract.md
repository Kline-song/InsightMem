# AML integration contract

This document records the public Agent Memory Leaderboard (AML) contract used by
this service. It was rechecked on 2026-10-29 against the organizer guide saved on
2026-10-11. Recheck the official pages before a formal run because operational
limits can change.

Official sources:

- https://agentmemories.ai/api-guide
- https://agentmemories.ai/docs
- https://agentmemories.ai/rules

## Transport and authentication

- The platform calls participant-hosted Add and Search endpoints; it does not
  deploy repositories or containers.
- HTTPS is recommended. Endpoint URLs must not contain credentials and must not
  point at private, loopback or link-local addresses.
- Add and Search accept `X-Api-Key`, `Authorization: Bearer ...` and
  `Authorization: Token ...`. This service supports all three. `none` is only
  for public smoke.
- `GET /health` is unauthenticated and returns any HTTP 2xx when the service is
  ready. It must not expose keys, database paths or credentials.
- Keep the memory service key out of the repository, logs, screenshots and
  public email.

## Add

`POST /add` is synchronous. HTTP 200 is returned only after every submitted
message is persisted and immediately searchable. Queue-only or task-id-only
responses do not satisfy the contract.

```json
{
  "request_id": "eval:run_abc123:conv-0:chunk-0",
  "user_id": "eval:run_abc123:conv-0",
  "session_id": "eval:run_abc123:sample:0",
  "messages": [
    {"role": "user", "timestamp": 1704067200000, "content": "memory text"}
  ]
}
```

- `request_id`: required; unique per logical write; echoed exactly.
- `messages`: required; ordered; each entry has a `role` and non-empty
  `content`. Text and code tracks use strings.
- `timestamp`: optional, Unix milliseconds.
- `user_id`: required; the only retrieval-isolation boundary; Add and Search
  must use the same value.
- `session_id`: required; identifies the source conversation but is not a
  Search filter.
- The current contract does not send `metadata`, `app_id`, `agent_id` or
  `async_mode`.
- Add is idempotent on `request_id`: the same request and payload returns 200
  again; a different payload with the same `request_id` returns 409. Idempotency
  records survive a service restart.
- HTTP 202 is not supported by this service because no Add-status endpoint is
  bound.

Success response:

```json
{
  "success": true,
  "request_id": "eval:run_abc123:conv-0:chunk-0",
  "user_id": "eval:run_abc123:conv-0",
  "session_id": "eval:run_abc123:sample:0"
}
```

`success` must be boolean `true`; all three identifiers must be present and
match the request exactly.

## Search

`POST /search` returns the evidence list for one `user_id`.

```json
{
  "query": "Which answer best matches the memory?",
  "options": ["A. First answer", "B. Second answer"],
  "user_id": "eval:run_abc123:conv-0",
  "top_k": 100
}
```

- `query`: required; use the benchmark text, not a generated final answer.
- `options`: optional top-level choice list; it does not contain a gold label.
  Open-ended questions omit it. The service may use or ignore it, but must not
  write the whole option list into memory.
- `user_id`: required; search only within that user's memories.
- `top_k`: required; formal external evaluation uses 100; the response must
  never contain more items than `top_k`.
- The current contract does not send `filters`, `rerank` or
  `keyword_search`.

Success response:

```json
{
  "data": [
    {
      "id": "mem_1",
      "content": "remembered evidence text",
      "score": 0.87,
      "created_at": "2026-07-01T12:00:00Z"
    }
  ]
}
```

- `data`: required array; do not wrap it in `items` or return a top-level
  array.
- `id`: required non-empty stable string.
- `content`: required non-empty string.
- `score`: optional number; higher means more relevant. Scores need not be
  comparable across queries.
- `created_at`: optional.
- Returned order is preserved by the platform and is more important than
  `score`. A count above `top_k` or a malformed item is a contract error, not
  something the platform silently truncates.
- No matching memory returns `{"data": []}`.

## Limits and retries

- One Add or Search request may take up to 30 minutes.
- The platform form requires declared Add and Search concurrency of at least
  16. This service declares 16 for both.
- On rate limiting, return HTTP 429 with a valid `Retry-After` header; the
  platform honors up to 60 seconds. This service does not currently apply
  active rate limiting, so it normally does not emit 429.
- The platform retries Add on network errors and 408, 409, 425, 429, 500, 502,
  503, 504 and 524. A 409 alone is not success; the logical write keeps the same
  `request_id` and payload.
- The platform retries Search on network errors and 408, 425, 429, 500, 502,
  503 and 504. Search 409 and permanent 4xx responses are not retried.
- Invalid payloads, bad credentials and forbidden scopes (400/401/403/404/409/
  422) are not retried.

## Error responses

| Situation | Response |
|---|---|
| Missing service key configuration | `503` |
| Missing or wrong service key | `401` |
| Reused `request_id` with a different payload | `409` |
| Invalid request body | `422` with field locations only |
| Embedding provider unavailable or invalid | `503`, without provider text |
| Unexpected server failure | `500` |

Error bodies contain a short `detail`; they never contain API keys, database
paths or provider response text.

## Data handling

Evaluation payloads and derived data may only be used for the current run.
They must not be used for training, analytics, reconstruction or sharing and
must be deleted within 30 days after the run unless the organizer grants a
different retention period in writing. The operator procedure is in
[operations.md](operations.md).

Stored messages are never modified or deleted by retrieval options; assistant
process filtering is applied per query and is fully reversible.

## Version changes

A Full run is bound to the registered system version. Changing retrieval
logic, models, prompts or other implementation behavior requires registering a
new version and passing Smoke on every track that will be used. Updating a
version does not migrate or retarget an existing Full task.
