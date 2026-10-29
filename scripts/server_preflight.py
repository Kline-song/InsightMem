from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def main() -> None:
    model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    backend = os.environ.get("EMBEDDING_BACKEND", "auto").strip().lower()
    if backend == "auto":
        backend = "dashscope" if model == "text-embedding-v4" else "local"
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "embedding_model": model,
        "embedding_backend": backend,
        "embedding_dimensions": os.environ.get("EMBEDDING_DIMENSIONS", "1024" if backend == "dashscope" else "model default"),
        "embedding_concurrency": os.environ.get("EMBEDDING_CONCURRENCY", "1"),
    }
    errors: list[str] = []
    versions: dict[str, str] = {}
    modules = ["fastapi", "uvicorn", "numpy", "openai"]
    if backend == "local":
        modules += ["sentence_transformers", "torch"]
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = str(getattr(module, "__version__", "installed"))
        except Exception as error:
            errors.append(f"cannot import {module_name}: {error}")
    report["dependencies"] = versions

    if backend == "dashscope":
        if model != "text-embedding-v4":
            errors.append("DashScope backend requires EMBEDDING_MODEL=text-embedding-v4")
        key_value = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("MEMORY_EMBED_API_KEY", "")
        report["dashscope_key_configured"] = bool(key_value)
        if not key_value:
            errors.append("DASHSCOPE_API_KEY or MEMORY_EMBED_API_KEY is required")
        if os.environ.get("EMBEDDING_QUERY_PREFIX"):
            errors.append("EMBEDDING_QUERY_PREFIX must be unset for text-embedding-v4")

    key = os.environ.get("MEMORY_API_KEY", "")
    if len(key) < 24 or key == "replace-with-a-long-random-secret":
        errors.append("MEMORY_API_KEY must be a non-placeholder secret of at least 24 characters")

    database = Path(os.environ.get("MEMORY_DB_PATH", "artifacts/memory.db"))
    database.parent.mkdir(parents=True, exist_ok=True)
    probe = database.parent / ".write-probe"
    try:
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
    except OSError as error:
        errors.append(f"database directory is not writable: {error}")
    usage = shutil.disk_usage(database.parent)
    report["database"] = str(database.resolve())
    report["free_disk_gb"] = round(usage.free / (1024**3), 2)

    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception:
        report["cuda_available"] = False

    report["status"] = "passed" if not errors else "failed"
    report["errors"] = errors
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

