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
from .config import load_project_env
from .embeddings import EmbeddingServiceError, build_embedder
from .memory_store import MemoryMessage, RequestConflictError, SQLiteMemoryStore
from .reranking import build_reranker

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
def get_store() -> SQLiteMemoryStore:
    model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    device = os.environ.get("EMBEDDING_DEVICE") or None
    query_prefix = os.environ.get("EMBEDDING_QUERY_PREFIX")
    database = os.environ.get("MEMORY_DB_PATH", "artifacts/memory.db")
    chunk_tokens = int(os.environ.get("MEMORY_CHUNK_TOKENS", "320"))
    overlap_tokens = int(os.environ.get("MEMORY_CHUNK_OVERLAP", "40"))
    retrieval_mode = os.environ.get("MEMORY_RETRIEVAL_MODE", "hybrid").lower()
    rrf_k = int(os.environ.get("MEMORY_RRF_K", "60"))
    lexical_weight = float(os.environ.get("MEMORY_LEXICAL_WEIGHT", "0.5"))
    context_messages = int(os.environ.get("MEMORY_CONTEXT_MESSAGES", "0"))
    option_view_weight = float(os.environ.get("MEMORY_OPTION_VIEW_WEIGHT", "0"))
    reranker_model = os.environ.get("MEMORY_RERANKER_MODEL", "").strip()
    reranker_device = os.environ.get("MEMORY_RERANKER_DEVICE") or device
    reranker_batch_size = int(os.environ.get("MEMORY_RERANKER_BATCH_SIZE", "32"))
    reranker_max_length = int(os.environ.get("MEMORY_RERANKER_MAX_LENGTH", "512"))
    rerank_candidates = int(os.environ.get("MEMORY_RERANK_CANDIDATES", "200"))
    temporal_enrichment = os.environ.get("MEMORY_TEMPORAL_ENRICHMENT", "false").lower() == "true"
    result_window = int(os.environ.get("MEMORY_RESULT_WINDOW", "1"))
    result_window_seed_k = int(os.environ.get("MEMORY_RESULT_WINDOW_SEED_K", "20"))
    reranker = (
        build_reranker(
            reranker_model,
            device=reranker_device,
            batch_size=reranker_batch_size,
            max_length=reranker_max_length,
        )
        if reranker_model
        else None
    )
    return SQLiteMemoryStore(
        database,
        build_embedder(model, device, query_prefix),
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        retrieval_mode=retrieval_mode,
        rrf_k=rrf_k,
        lexical_weight=lexical_weight,
        context_messages=context_messages,
        option_view_weight=option_view_weight,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
        temporal_enrichment=temporal_enrichment,
        result_window=result_window,
        result_window_seed_k=result_window_seed_k,
    )


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
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    safe_errors = [
        {
            "loc": list(error.get("loc", ())),
            "type": error.get("type"),
            "msg": error.get("msg"),
        }
        for error in exc.errors()
    ]
    validation_logger.warning(
        "request validation failed path=%s errors=%s",
        request.url.path,
        safe_errors,
    )
    return JSONResponse(status_code=422, content={"detail": safe_errors})


@app.get("/health", response_model=HealthResponse)
def health(store: Annotated[SQLiteMemoryStore, Depends(get_store)]) -> HealthResponse:
    return HealthResponse(**store.stats())


@app.post("/add", response_model=AddResponse, dependencies=[Depends(require_api_key)])
def add_memory(payload: AddRequest, store: Annotated[SQLiteMemoryStore, Depends(get_store)]) -> AddResponse:
    messages = [MemoryMessage(role=item.role, content=item.content, timestamp=item.timestamp) for item in payload.messages]
    try:
        store.add(payload.request_id, payload.user_id, payload.session_id, messages)
    except RequestConflictError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return AddResponse(
        request_id=payload.request_id,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
def search_memory(payload: SearchRequest, store: Annotated[SQLiteMemoryStore, Depends(get_store)]) -> SearchResponse:
    hits = store.search(payload.user_id, payload.query, payload.top_k, payload.options)
    return SearchResponse(data=[SearchItem(**hit.__dict__) for hit in hits])
