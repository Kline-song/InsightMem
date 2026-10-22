"""Load only this project's ignored .env; process environment takes precedence."""
from pathlib import Path

from dotenv import load_dotenv


def load_project_env(path: Path | None = None) -> None:
    load_dotenv(
        path if path is not None else Path(__file__).resolve().parents[2] / ".env",
        override=False,
        encoding="utf-8-sig",
        interpolate=False,
    )
