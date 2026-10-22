# AML integration contract

This document records the public Agent Memory Leaderboard contract checked on
2026-08-04. Recheck the official documentation before a formal run because the
organizers may update operational limits.

Official sources:

- https://agentmemories.ai/api-guide
- https://agentmemories.ai/docs
- https://agentmemories.ai/rules

## Endpoints

- `GET /health` is unauthenticated and returns any HTTP 2xx when ready.
- `POST /add` is synchronous. HTTP 200 means every submitted message is
  persisted and immediately searchable.
- `POST /search` returns `{"data": [...]}` in relevance order and never returns
  more than the requested `top_k` (100 in formal external evaluations).

`user_id` is the sole retrieval-isolation boundary. `session_id` identifies a
source conversation but is not a Search filter. Add responses must echo
`request_id`, `user_id`, and `session_id` exactly.

The hosted API accepts `Authorization: Bearer`, `Authorization: Token`, or
`X-Api-Key`. This implementation supports all three. Keep the configured key
out of the repository and logs.

## Data handling

Evaluation payloads and derived data may only be used for the current run.
They must not be used for training, analytics, reconstruction, or sharing and
must be deleted within 30 days after the run unless the organizer grants a
different retention period in writing.

## Local versus official results

The scripts in this repository validate the public HTTP contract and replay
normalized public benchmark data. They do not reproduce the private suite,
locked Answer model, evaluator configuration, or official aggregate score.
Only the organizer's Smoke and Full jobs are official.
