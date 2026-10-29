# Operations

This file records the deployment profile, logging and data-retention procedure
for the hosted service. It contains no credentials.

## Runtime profile

- Python 3.10.12; API dependencies are pinned in
  `requirements-api.txt` + `constraints-api.txt`.
- One `uvicorn` worker. The service is synchronous: Add returns only after the
  messages, chunks, views and embeddings are committed.
- SQLite in WAL mode. `MEMORY_DB_PATH` must point at a database created with the
  same config fingerprint and embedding identity; the service refuses a
  mismatched database at startup.
- A database records the profile fingerprint and embedding identity. Changing
  `configs/coding/default.json` or the embedding identity requires a fresh
  database.

Environment variables (names only; values belong in the server environment or
an ignored `.env`):

| Variable | Purpose |
|---|---|
| `MEMORY_API_KEY` | service key for Add/Search |
| `MEMORY_DB_PATH` | SQLite database path |
| `CODING_CONFIG_PATH` | retrieval profile JSON |
| `EMBEDDING_BACKEND` | `dashscope` for the hosted API |
| `EMBEDDING_MODEL` | `text-embedding-v4` |
| `EMBEDDING_DIMENSIONS` | `1024` |
| `EMBEDDING_BATCH_SIZE` | `10` |
| `EMBEDDING_CONCURRENCY` | parallel embedding batches inside one Add; `4` |
| `EMBEDDING_TIMEOUT_SECONDS` | `60` |
| `EMBEDDING_MAX_RETRIES` | `2` |
| `DASHSCOPE_BASE_URL` | compatible-mode endpoint |
| `DASHSCOPE_API_KEY` | embedding provider key; never logged |

Gateway and platform limits:

- Add/Search request timeout: 30 minutes (platform).
- Reverse-proxy read timeout for Add/Search: 1800 seconds; Health: 30 seconds.
- Request-body cap: 10 MiB.
- Declared concurrency: Add 16, Search 16. Reference measurements on the
  deployment host (2 vCPU, 3.5 GiB RAM): 16 concurrent Adds of 6 messages all
  returned 200 with P95 18.4 s; 16 concurrent Searches over 460 messages all
  returned 200 with P95 20.5 s and 100 results. These are reference numbers, not
  a Full-scale soak test.
- The service does not actively rate-limit. If a 429 is introduced, it must
  include `Retry-After`.

## Logging

- Nginx access/error logs: request path, status and bytes; kept for 60 days.
- systemd journal: application startup, validation warnings and errors; kept
  for 90 days with a 1 GiB cap.
- Request bodies, authorization headers and memory content are not written to
  logs. Validation warnings contain field locations only.
- The service does not run uvicorn access logging; Nginx provides the
  request-level record.

## Data purge after a run

Evaluation payloads and derived data are deleted within 30 days after the run,
unless the organizer grants a different retention period in writing. Keep a
deletion audit record (time, user/run identifier, row counts, operator, reason)
for at least the same period. Audit records must not contain message content or
credentials.

### A. Whole-database reset

Use this when every user belongs to a finished run.

1. Confirm the task has ended and no retry, checkpoint resume or review is
   pending.
2. Stop the service so no writer is active.
3. Copy the database aside for the audit archive if required, then remove the
   database and its `-wal`/`-shm` files, or move them out of the service path.
4. Start the service. It creates an empty database and checks the fingerprint.
5. Verify `/health` returns 200 and the counters are zero.

```bash
systemctl stop insightmem
cp -a "$MEMORY_DB_PATH" "/var/backups/insightmem-$(date +%Y%m%dT%H%M%SZ).db"
rm -f "$MEMORY_DB_PATH" "$MEMORY_DB_PATH-wal" "$MEMORY_DB_PATH-shm"
systemctl start insightmem
curl -fsS http://127.0.0.1:8000/health
```

### B. Per-user purge

Use this when only one user or run must be removed and other data stays.

1. Confirm the task has ended and no retry is pending.
2. Stop the service, or run the command against a copy.
3. Run the dry-run, then the apply, with an audit log:

```bash
python scripts/purge_user.py --db "$MEMORY_DB_PATH" --user-id "$USER_ID"
python scripts/purge_user.py --db "$MEMORY_DB_PATH" --user-id "$USER_ID" \
  --apply --audit-log /var/log/insightmem/purge-audit.jsonl
```

The command deletes in foreign-key order (`view_embeddings`, `view_sources`,
`source_chunks`, `index_views`, `source_messages`, `coding_requests`) inside one
transaction, then verifies that zero rows remain. It refuses a database that is
not an InsightMem coding database.
4. Restart the service and verify `/health`.

## Release and rollback

- Keep each release in its own directory (for example
  `/opt/insightmem/releases/<commit>`); point the unit's `WorkingDirectory` and
  `PYTHONPATH` at that directory.
- Back up the unit file and environment file before changing release paths.
- A version change requires registering the fixed commit and passing Smoke on
  each track that will be used. Existing Full tasks remain bound to their
  original version.
- To roll back, restore the unit and environment backups, reload systemd,
  restart the service and verify `/health`. Keep the previous database until
  the rollback decision is final; never point an old build at a database with
  a different schema or fingerprint.
