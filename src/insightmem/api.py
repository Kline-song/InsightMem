"""Synchronous Add/Search HTTP API for the coding memory service."""
from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .coding.config import CodingConfig
from .coding.service import CodingMemoryService
from .config import load_project_env
from .embeddings import EmbeddingServiceError, build_embedder
from .store import MemoryMessage, RequestConflictError

load_project_env()


class MessagePayload(BaseModel):
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)
    timestamp: int | None = None


class AddRequest(BaseModel):
    request_id: str = Field(min_length=1)
    messages: list[MessagePayload] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(gt=0, le=100)


class SearchItem(BaseModel):
    id: str
    content: str
    score: float | None = None
    created_at: str | None = None


class SearchResponse(BaseModel):
    data: list[SearchItem]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    requests: int
    memories: int
    users: int


@lru_cache(maxsize=1)
def get_store() -> CodingMemoryService:
    config_path = os.environ.get("CODING_CONFIG_PATH")
    config = CodingConfig.load(config_path) if config_path else CodingConfig()
    model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedder = build_embedder(model, os.environ.get("EMBEDDING_DEVICE") or None)
    database = os.environ.get("MEMORY_DB_PATH", "artifacts/memory.db")
    return CodingMemoryService(database, embedder, config)


def require_api_key(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.environ.get("MEMORY_API_KEY", "")
    if not expected:
        if os.environ.get("ALLOW_UNAUTHENTICATED", "false").lower() == "true":
            return
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="MEMORY_API_KEY is not configured")
    provided = x_api_key
    if not provided and authorization:
        scheme, separator, credential = authorization.partition(" ")
        if separator and scheme.lower() in {"bearer", "token"}:
            provided = credential
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("SKIP_MODEL_STARTUP", "false").lower() != "true":
        get_store()
    yield


app = FastAPI(title="InsightMem API", version=__version__, lifespan=lifespan)

validation_logger = logging.getLogger("insightmem.validation")


@app.exception_handler(EmbeddingServiceError)
async def embedding_service_exception_handler(request: Request, exc: EmbeddingServiceError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Embedding provider unavailable; retry later"})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    safe_errors = [
        {"loc": list(error.get("loc", ())), "type": error.get("type"), "msg": error.get("msg")}
        for error in exc.errors()
    ]
    validation_logger.warning(
        "request validation failed path=%s errors=%s", request.url.path, safe_errors
    )
    return JSONResponse(status_code=422, content={"detail": safe_errors})


@app.get("/health", response_model=HealthResponse)
def health(store: Annotated[CodingMemoryService, Depends(get_store)]) -> HealthResponse:
    return HealthResponse(**store.stats())


@app.post("/add", response_model=AddResponse, dependencies=[Depends(require_api_key)])
def add_memory(payload: AddRequest, store: Annotated[CodingMemoryService, Depends(get_store)]) -> AddResponse:
    messages = [MemoryMessage(role=item.role, content=item.content, timestamp=item.timestamp) for item in payload.messages]
    try:
        store.add(payload.request_id, payload.user_id, payload.session_id, messages)
    except RequestConflictError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return AddResponse(request_id=payload.request_id, user_id=payload.user_id, session_id=payload.session_id)


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
def search_memory(payload: SearchRequest, store: Annotated[CodingMemoryService, Depends(get_store)]) -> SearchResponse:
    hits = store.search(payload.user_id, payload.query, payload.top_k, payload.options)
    return SearchResponse(data=[SearchItem(**hit.__dict__) for hit in hits])
