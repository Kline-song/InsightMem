from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def main() -> None:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    }
    errors: list[str] = []
    versions: dict[str, str] = {}
    for module_name in ("fastapi", "uvicorn", "numpy", "sentence_transformers", "faiss", "torch"):
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = str(getattr(module, "__version__", "installed"))
        except Exception as error:
            errors.append(f"cannot import {module_name}: {error}")
    report["dependencies"] = versions

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

