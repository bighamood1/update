"""Minimal local HTTP adapter for the existing NMU RAG pipeline.

This server is a thin, read-only facade over the RAG project. It reuses the
existing ``RAGPipeline`` (retrieval + reranking + Ollama generation) and
exposes a single chat endpoint. It never modifies RAG code, documents, or the
vector index.

Clients (e.g. the GUI in ``gui/``) talk to this server over HTTP only and
never touch ChromaDB, embeddings, or Ollama directly.

Security: binds to ``127.0.0.1`` only (local development). No auth, no data
leaves this machine.

Error contract
--------------
Every failure returns a structured body with NO stack traces and safe
``answer``/``sources`` defaults the GUI can render::

    {"success": false, "error": true, "error_type": "...", "message": "...",
     "answer": "", "sources": [], "error": {"type": "...", "message": "..."}}

The nested ``error`` object is kept for backward compatibility.

Error ``type`` values: ``invalid_request`` (422), ``busy`` (409),
``connection_error`` / ``model_unavailable`` (503), ``generation_timeout``
(504), ``generation_error`` (502), ``backend_error`` (503),
``unexpected_error`` (500).

Concurrency: Ollama generation concurrency is bounded by
``MAX_CONCURRENT_GENERATIONS`` (default 1 on this CPU-only machine). A request
beyond the limit gets an immediate ``busy`` (409) response instead of queueing
on a deadlock-prone lock.

Additional endpoints
--------------------
``POST /feedback`` stores a user rating (``useful`` / ``somewhat`` /
``not_useful``) plus an optional per-rating reason for a given
``question_id``. ``GET /stats`` exposes lightweight runtime analytics.

Run::

    python api/server.py                 # or
    uvicorn api.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag.generation.ollama_client import OllamaError
from rag.pipeline.rag import RAGPipeline
from rag.utils.logging_utils import setup_logging

setup_logging()
_api_logger = logging.getLogger("api")

# Built once at process start: loads the embedder + reranker + vector store.
# Individual /chat calls then take as long as retrieval + generation need.
pipeline = RAGPipeline()


def _generation_limit() -> int:
    """Concurrency bound for Ollama generations (configurable, default 1)."""
    try:
        from rag.config import get_config

        return max(1, int(get_config().get("max_concurrent_generations", 1) or 1))
    except Exception:  # pragma: no cover - defensive
        return 1


# Non-blocking acquire => busy (409) when the generation slots are saturated.
_generation_semaphore = threading.BoundedSemaphore(_generation_limit())


def _startup_checks() -> None:
    """Validate the environment at startup and log clearly (no crash)."""
    t0 = time.perf_counter()
    _api_logger.info("RAG API starting...")
    built = False
    try:
        if not pipeline.vectorstore.is_built():
            _api_logger.error(
                "[startup] Vector index is NOT built. Run: python scripts/build_index.py"
            )
        else:
            built = True
            for warn in pipeline.vectorstore.compatibility_warnings():
                _api_logger.warning("[startup] %s", warn)
    except Exception:  # pragma: no cover - defensive
        _api_logger.exception("[startup] Vector store check failed")
    try:
        model = pipeline.ollama.resolve_model()
        _api_logger.info("[OLLAMA] Using model: %s", model)
    except OllamaError as exc:
        _api_logger.error("[OLLAMA] %s", exc)
    except Exception:  # pragma: no cover - defensive
        _api_logger.exception("[startup] Ollama model resolution failed")
    try:
        _api_logger.info(
            "[MODEL] embedder=%s device=%s | reranker=%s device=%s enabled=%s",
            pipeline.embedder.model_name, pipeline.embedder.device,
            getattr(pipeline.retriever.reranker, "model_name", "-"),
            getattr(pipeline.retriever.reranker, "device", "-"),
            bool(pipeline.retriever.reranker_enabled),
        )
    except Exception:  # pragma: no cover - defensive
        pass
    # Warm the embedding model + BM25 lexical index once at startup so the
    # first /chat does not pay a cold-start penalty (models load once, then
    # stay resident for the process lifetime).
    if built:
        try:
            pipeline.embedder.dimension()
            pipeline.retriever.warmup()
        except Exception:  # pragma: no cover - warm-up must never crash startup
            _api_logger.warning(
                "[startup] Model warm-up skipped; models will load lazily on first use"
            )
    _api_logger.info("[PERF] startup completed in %.2fs", time.perf_counter() - t0)


def _shutdown() -> None:
    """Release external resources (HTTP clients) on shutdown."""
    try:
        pipeline.ollama.close()
    except Exception:  # pragma: no cover - defensive
        _api_logger.exception("[shutdown] Ollama client close failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """FastAPI lifespan context manager (replaces the deprecated on_event hook)."""
    _startup_checks()
    try:
        yield
    finally:
        _shutdown()


app = FastAPI(
    title="NMU RAG API",
    description="Local HTTP facade over the NMU AI Robot Assistant RAG pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] | None = None


class SourceOut(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    success: bool = True
    answer: str
    sources: list[SourceOut]
    response_id: str = ""
    question_id: str = ""
    cache_hit: bool = False
    cached_question_id: str | None = None


class FeedbackRequest(BaseModel):
    question_id: str | None = None
    response_id: str | None = None
    rating: str | None = None
    feedback: str | None = None
    reason: str | None = None


def _error(status: int, error_type: str, message: str) -> JSONResponse:
    """Structured error body (never leaks stack traces or internals).

    Always carries GUI-safe ``answer``/``sources`` defaults plus both the
    top-level fields (``error_type``/``message``) and the legacy nested
    ``error`` object, so old and new clients can parse it.
    """
    return JSONResponse(
        status_code=status,
        content={
            "success": False,
            "error": True,
            "error_type": error_type,
            "message": message,
            "answer": "",
            "sources": [],
            "error": {"type": error_type, "message": message},
        },
    )


def _clean_answer(answer: Any) -> str:
    """Normalize the answer to a stripped non-empty string (defensive)."""
    if answer is None:
        return ""
    return str(answer).strip()


def _normalize_sources(sources: Any) -> list[SourceOut]:
    """Normalize + dedupe source entries; silently drop malformed ones."""
    out: list[SourceOut] = []
    seen: set[tuple[str, str]] = set()
    for item in sources or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "Source").strip()
        url = str(item.get("url") or "").strip()
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        out.append(SourceOut(title=title or "Source", url=url))
    return out


def _ollama_error_response(exc: OllamaError) -> JSONResponse:
    error_type = getattr(exc, "error_type", "generation_error")
    if error_type == "connection":
        status, kind = 503, "connection_error"
    elif error_type == "model_unavailable":
        status, kind = 503, "model_unavailable"
    elif error_type == "generation_timeout":
        status, kind = 504, "generation_timeout"
    else:
        status, kind = 502, "generation_error"
    return _error(status, kind, str(exc))


def _log_timings(
    message: str,
    timings: dict,
    elapsed_total: float,
    *,
    model: str = "",
    llm_used: bool = False,
    cache_hit: bool = False,
) -> None:
    """[PERF] breakdown to the backend console only (never the GUI)."""
    retrieval = float(timings.get("retrieval_time", 0.0))
    reranking = float(timings.get("reranking_time", 0.0))
    ollama = float(timings.get("ollama_request_time", 0.0))
    fast = float(timings.get("fast_path_time", 0.0))
    total = round(elapsed_total, 3)

    def _fmt(s: float) -> str:
        if s >= 60:
            return f"{s / 60:,.1f}m"
        if s >= 1:
            return f"{s:,.2f}s"
        return f"{s * 1000:,.0f}ms"

    bar = "-" * 64
    _api_logger.info("\n%s", bar)
    _api_logger.info(
        "[PERF] %s",
        (message[:80] + "…") if len(message) > 80 else message,
    )
    _api_logger.info("  model    : %s | llm_used=%s cache_hit=%s", model or "-", llm_used, cache_hit)
    _api_logger.info("  retrieval : %s", _fmt(retrieval))
    _api_logger.info("  rerank    : %s", _fmt(reranking))
    _api_logger.info("  fast_path : %s", _fmt(fast))
    _api_logger.info("  Ollama    : %s", _fmt(ollama))
    _api_logger.info("  TOTAL     : %s", _fmt(total))
    _api_logger.info("%s\n", bar)


@app.get("/health")
def health() -> dict:
    """Liveness check for clients / load balancers."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse | JSONResponse:
    message = (req.message or "").strip()
    if not message:
        return _error(422, "invalid_request", "Please enter a question.")

    if not _generation_semaphore.acquire(blocking=False):
        return _error(
            409,
            "busy",
            "The assistant is already answering another question. "
            "Please wait and try again.",
        )

    t_wall_start = time.perf_counter()
    result: Any = None
    try:
        history = None
        if req.history:
            history = [
                {"role": t.role, "content": t.content} for t in req.history
            ]
        result = pipeline.ask(message, history=history)
    except OllamaError as exc:
        _api_logger.error("Ollama error (%s): %s", getattr(exc, "error_type", "?"), exc)
        return _ollama_error_response(exc)
    except ValueError as exc:
        return _error(422, "invalid_request", str(exc))
    except RuntimeError as exc:
        _api_logger.error("Retrieval/index error: %s", exc)
        return _error(503, "backend_error", str(exc))
    except Exception as exc:  # noqa: BLE001 - log cause, never leak it
        _api_logger.exception("RAG pipeline failed for: %s", message[:120])
        return _error(
            500,
            "unexpected_error",
            "The NMU assistant failed to answer this question. Please make "
            "sure the local RAG backend (ChromaDB + Ollama) is running.",
        )
    finally:
        _generation_semaphore.release()

    # Success path only: every failure above returned before this point, so
    # `result` is bound here and never touched on error paths.
    if result is None:
        _api_logger.error("RAG pipeline returned None for: %s", message[:120])
        return _error(
            500,
            "unexpected_error",
            "The NMU assistant returned an empty answer. Please try again.",
        )
    answer = _clean_answer(result.answer)
    if not answer:
        _api_logger.error("RAG pipeline returned an empty answer for: %s", message[:120])
        return _error(
            500,
            "unexpected_error",
            "The NMU assistant returned an empty answer. Please try again.",
        )

    elapsed_wall = time.perf_counter() - t_wall_start
    try:
        _log_timings(
            message,
            getattr(result, "timings", None) or {},
            elapsed_wall,
            model=getattr(pipeline.ollama, "model", "") or "auto",
            llm_used=bool(getattr(result, "llm_used", False)),
            cache_hit=bool(getattr(result, "cache_hit", False)),
        )
    except Exception:  # pragma: no cover - never let logging break the response
        pass

    return ChatResponse(
        answer=answer,
        sources=_normalize_sources(result.sources),
        response_id=str(getattr(result, "question_id", "") or ""),
        question_id=str(getattr(result, "question_id", "") or ""),
        cache_hit=bool(getattr(result, "cache_hit", False)),
        cached_question_id=getattr(result, "cached_question_id", None),
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """Record a user rating for a previously answered question."""
    from rag.feedback.store import FeedbackStore

    response_id = (req.response_id or req.question_id or "").strip()
    rating = (req.feedback or req.rating or "").strip()
    record = FeedbackStore().submit(
        question_id=response_id, rating=rating, reason=req.reason
    )
    if not record.ok:
        if record.message == "unknown question_id":
            status, kind = 404, "unknown_question"
        elif record.message == "question_id is required":
            status, kind = 422, "invalid_request"
        else:
            status, kind = 422, "invalid_feedback"
        return JSONResponse(
            status_code=status,
            content={
                "success": False,
                "error": True,
                "error_type": kind,
                "message": record.message,
                "error": {"type": kind, "message": record.message},
            },
        )
    return {
        "success": True,
        "feedback_id": record.feedback_id,
        "response_id": response_id,
        "feedback": record.rating,
    }


@app.get("/stats")
def stats() -> dict:
    """Lightweight runtime analytics (questions, ratings, cache stats)."""
    try:
        from rag.cache.store import get_runtime_store

        return get_runtime_store().stats()
    except Exception:  # pragma: no cover - analytics must never break the API
        _api_logger.exception("Failed to build /stats")
        return {"success": False, "error_type": "stats_unavailable", "message": "stats unavailable"}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
