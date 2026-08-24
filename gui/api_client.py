"""HTTP client for the local NMU RAG API.

This module is the ONLY place in the GUI that talks to the backend. It knows
nothing about ChromaDB, embeddings, chunks, reranking, or Ollama — it sends a
message and receives ``answer`` + ``sources``.

The API is designed so a streaming variant (``stream_message``) can be added
later without touching the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


class APIError(Exception):
    """Base class for all client-side API errors (already user-friendly)."""


class APIConnectionError(APIError):
    """Server unreachable (connection refused, DNS, network)."""


class APITimeoutError(APIError):
    """Server took too long to answer."""


class APIServerError(APIError):
    """Server answered with a 5xx error."""


class APIResponseError(APIError):
    """Server answered 2xx but the payload was not usable."""


@dataclass(frozen=True)
class Source:
    title: str
    url: str


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    response_id: str = ""
    question_id: str = ""
    cache_hit: bool = False


class APIClient:
    """Thin, stateless client for ``POST /chat``, ``POST /feedback`` and
    ``GET /health``/``GET /stats``."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 1200.0,
        *,
        verify: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._verify = verify

    def is_reachable(self) -> bool:
        """True if the API answers /health. Never raises."""
        try:
            with httpx.Client(verify=self._verify, timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def send_message(
        self, message: str, history: list[dict] | None = None
    ) -> ChatResult:
        """Send one message and return a clean ``ChatResult``.

        ``history`` is an optional ordered list of ``{"role", "content"}``
        turns (conversation context for follow-up questions). Raises one of the
        APIError subclasses with a user-friendly message.
        """
        payload: dict = {"message": message}
        if history:
            payload["history"] = [
                {"role": str(t.get("role") or ""), "content": str(t.get("content") or "")}
                for t in history
                if isinstance(t, dict) and t.get("role") in ("user", "assistant")
                and (t.get("content") or "").strip()
            ]
        try:
            with httpx.Client(verify=self._verify, timeout=self._timeout) as client:
                resp = client.post(f"{self.base_url}/chat", json=payload)
        except (httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            raise APITimeoutError(
                "The local model is taking longer than expected. Please try again."
            ) from exc
        except httpx.ConnectTimeout as exc:
            raise APIConnectionError(
                "Unable to connect to the NMU assistant. Please make sure the "
                "RAG server is running."
            ) from exc
        except httpx.ConnectError as exc:
            raise APIConnectionError(
                "Unable to connect to the NMU assistant. Please make sure the "
                "RAG server is running."
            ) from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(
                "The local model is taking longer than expected. Please try again."
            ) from exc
        except httpx.HTTPError as exc:
            raise APIConnectionError(
                "Unable to reach the NMU assistant. Please check the connection "
                "and that the RAG server is running."
            ) from exc

        if resp.status_code == 404:
            raise APIConnectionError(
                "The RAG server is running but does not expose the /chat "
                "endpoint. Please start the correct server."
            )
        if resp.status_code >= 400:
            raise APIServerError(self._friendly_error(resp))

        return self._parse(resp)

    @staticmethod
    def _friendly_error(resp: httpx.Response) -> str:
        """Turn a structured error body into a user-friendly message.

        Accepts both the legacy nested shape (``error.type``/``error.message``)
        and the top-level shape (``error_type``/``message``).
        """
        error_type: str = ""
        message: str = ""
        try:
            body = resp.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                error_type = str(err.get("type") or "")
                message = str(err.get("message") or "")
            if not error_type:
                error_type = str(body.get("error_type") or "")
            if not message:
                message = str(body.get("message") or "")

        canned = {
            "busy": "The assistant is already answering another question. "
            "Please wait for it to finish and try again.",
            "generation_timeout": "The local model took too long to answer. "
            "Try asking a more specific question, or restart the backend.",
            "model_unavailable": "The configured AI model is not available on "
            "this computer. Please check the backend configuration.",
            "connection_error": "The local AI engine (Ollama) is not running. "
            "Please start it and try again.",
            "generation_error": "The local model failed to generate an answer. "
            "Please try again.",
            "backend_error": "The knowledge base could not be searched. "
            "Please check that the index is built.",
            "invalid_request": "Please enter a valid question.",
            "unexpected_error": "The NMU assistant failed to answer this question. "
            "Please make sure the local RAG backend (ChromaDB + Ollama) is "
            "running and try again.",
            # legacy category names from older server builds
            "ollama_connection": "The local AI engine (Ollama) is not running. "
            "Please start it and try again.",
            "retrieval_error": "The knowledge base could not be searched. "
            "Please check that the index is built.",
            "internal_error": "The NMU assistant failed to answer this question. "
            "Please make sure the local RAG backend (ChromaDB + Ollama) is "
            "running and try again.",
        }
        if error_type in canned:
            return canned[error_type]
        if message and message not in ("Please enter a question.",):
            return message
        return "The NMU assistant could not answer this question. Please try again."

    @staticmethod
    def _parse(resp: httpx.Response) -> ChatResult:
        try:
            data = resp.json()
        except ValueError as exc:
            raise APIResponseError(
                "The NMU assistant returned an unreadable response. Please try again."
            ) from exc
        if not isinstance(data, dict):
            raise APIResponseError(
                "The NMU assistant returned an unexpected response. Please try again."
            )
        if data.get("success") is False:
            err = data.get("error")
            msg = err.get("message") if isinstance(err, dict) else None
            msg = msg or data.get("message") or ""
            raise APIResponseError(
                msg or "The NMU assistant could not answer this question. Please try again."
            )
        answer = data.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise APIResponseError(
                "The NMU assistant returned an empty answer. Please try again."
            )
        raw_sources = data.get("sources")
        if not isinstance(raw_sources, list):
            raw_sources = []
        sources: list[Source] = []
        for item in raw_sources:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or "Source"
            url = item.get("url") or ""
            sources.append(Source(title=str(title), url=str(url)))
        return ChatResult(
            answer=answer.strip(),
            sources=sources,
            response_id=str(data.get("response_id") or data.get("question_id") or ""),
            question_id=str(data.get("question_id") or ""),
            cache_hit=bool(data.get("cache_hit")),
        )

    def send_feedback(
        self, question_id: str, rating: str, reason: str | None = None
    ) -> bool:
        """Record a user rating for a question. Returns True on success.

        Never raises for a rejected rating (unknown id / bad value): the server
        returns a structured error body which is swallowed here so the GUI's
        feedback buttons never block the chat.
        """
        question_id = (question_id or "").strip()
        if not question_id:
            return False
        payload: dict = {"question_id": question_id, "rating": rating}
        payload["response_id"] = question_id
        payload["feedback"] = rating
        if reason:
            payload["reason"] = reason
        try:
            with httpx.Client(verify=self._verify, timeout=30.0) as client:
                resp = client.post(f"{self.base_url}/feedback", json=payload)
        except httpx.HTTPError:
            return False
        if resp.status_code >= 400:
            return False
        try:
            return bool((resp.json() or {}).get("success"))
        except ValueError:
            return False
