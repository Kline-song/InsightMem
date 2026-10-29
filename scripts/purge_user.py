#!/usr/bin/env python3
"""Operator-only deletion of one user's stored memories.

The API has no Clear endpoint. Run this only after the evaluation task has
finished and no retry or review is pending. Stop the service first, or run
against a copy, so no writer is active.

Without --apply the command only reports row counts. With --apply it deletes the
rows for one user in foreign-key order and verifies that none remain.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

# Child tables first, then the request row that owns the messages.
TABLES = (
    "view_embeddings",
    "view_sources",
    "source_chunks",
    "index_views",
    "source_messages",
    "coding_requests",
)


def row_counts(connection: sqlite3.Connection, user_id: str) -> dict[str, int]:
    return {
        table: connection.execute(
            f"SELECT count(*) FROM {table} WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        for table in TABLES
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="path to the SQLite database")
    parser.add_argument("--user-id", required=True, help="user_id to purge")
    parser.add_argument("--apply", action="store_true", help="perform the deletion")
    parser.add_argument("--audit-log", help="append one JSON record to this file")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print(f"database not found: {args.db}", file=sys.stderr)
        return 1

    connection = sqlite3.connect(args.db, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if "coding_meta" not in tables or not set(TABLES).issubset(tables):
        print("refusing: not an InsightMem coding database", file=sys.stderr)
        return 1

    before = row_counts(connection, args.user_id)
    result = {
        "db": os.path.abspath(args.db),
        "user_id": args.user_id,
        "before": before,
        "applied": False,
    }
    if not args.apply:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0

    try:
        connection.execute("BEGIN IMMEDIATE")
        for table in TABLES:
            connection.execute(f"DELETE FROM {table} WHERE user_id = ?", (args.user_id,))
        connection.commit()
    except BaseException:
        connection.rollback()
        raise

    after = row_counts(connection, args.user_id)
    result["after"] = after
    result["applied"] = True
    if any(after.values()):
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        print("refusing: rows remain after deletion", file=sys.stderr)
        return 1

    if args.audit_log:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            **result,
            "operator": os.environ.get("USER", "unknown"),
        }
        descriptor = os.open(
            args.audit_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
