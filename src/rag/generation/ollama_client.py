"""Ollama API client for local LLM generation.

Communicates with Ollama exclusively over its HTTP API
(``http://localhost:11434``). Never touches the local model files.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..config import get_config
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class OllamaError(RuntimeError):
    """Raised for Ollama connectivity / generation failures.

    ``error_type`` is a stable machine-readable category used by the HTTP API
    to build structured error responses:
        - "connection"           Ollama unreachable
        - "model_unavailable"    configured model not installed locally
        - "generation_timeout"   read timeout while generating
        - "generation_error"     Ollama returned an error or empty payload

    ``details`` carries optional machine metadata (model, timeout) for logs.
    """

    def __init__(
        self,
        message: str,
        error_type: str = "generation_error",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


# How long a successful connectivity/model check stays valid. Avoids paying
# a /api/tags round-trip on every generation request (which on a busy CPU is
# measurable); failures bypass the cache so errors surface immediately.
_CHECK_TTL_SECONDS = 300.0


class OllamaClient:
    """Thin, typed client for the Ollama chat API.

    Uses a persistent ``httpx.Client`` (connection reuse — no new connection
    objects per request) with split timeouts: a short connect timeout, a long
    read timeout for generation, and bounded write/pool timeouts. ``keep_alive``
    keeps the model resident between requests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        cfg = get_config()
        self.base_url = (base_url or cfg["ollama_base_url"]).rstrip("/")
        self.model = model or cfg["ollama_model"]
        self.timeout = timeout or cfg["ollama_timeout"]
        self.options = options or cfg["ollama_options"]
        self.keep_alive = cfg["ollama_keep_alive"]
        self._checked_at = 0.0
        self._verified = False
        self._closed = False
        # (connect, read, write, pool) timeout tuple — read stays long so long
        # generations are never killed prematurely.
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=self.timeout,
                write=60.0,
                pool=10.0,
            ),
        )

    def close(self) -> None:
        """Close the persistent HTTP client (safe to call more than once)."""
        if not self._closed:
            self._client.close()
            self._closed = True

    # -- connectivity ------------------------------------------------------

    def check_connection(self) -> dict:
        """Verify Ollama is reachable. Raises OllamaError otherwise."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. Is the Ollama service running?",
                error_type="connection",
            ) from exc

    def list_models(self) -> list[str]:
        return [m["name"] for m in self.check_connection().get("models", [])]

    def resolve_model(self, installed: list[str] | None = None) -> str:
        """Pick the model to use, then verify it is installed.

        - ``OLLAMA_MODEL`` explicitly set -> must be installed, otherwise a
          clear ``model_unavailable`` error lists what is available.
        - ``OLLAMA_MODEL`` empty -> auto-select the first INSTALLED model from
          ``OLLAMA_PREFERRED_MODELS`` (faster text models first). Never
          downloads anything.
        """
        if installed is None:
            installed = self.list_models()
        names = set(installed)
        if self.model:
            if self.model in names or self.model.split(":")[0] in names:
                return self.model
            raise OllamaError(
                f"Configured model '{self.model}' is not installed in Ollama. "
                f"Available models: {installed or '(none)'}. Install it with: "
                f"ollama pull {self.model}  (or set OLLAMA_MODEL to an "
                "available model)",
                error_type="model_unavailable",
            )
        if not installed:
            raise OllamaError(
                "Ollama is running but no models are installed. Install one with: "
                "ollama pull qwen3:4b",
                error_type="model_unavailable",
            )
        for candidate in get_config().get("ollama_preferred_models", []):
            if candidate in names or candidate.split(":")[0] in names:
                if candidate != self.model:
                    logger.info("[OLLAMA] Auto-selected model: %s", candidate)
                self.model = candidate
                return candidate
        # Fallback: use the only (or first) installed model.
        selected = installed[0]
        logger.warning(
            "[OLLAMA] No preferred model installed; using available model '%s'. "
            "For faster answers install qwen3:4b and set OLLAMA_MODEL.",
            selected,
        )
        self.model = selected
        return selected

    def ensure_model(self, installed: list[str] | None = None) -> None:
        """Verify the model is installed locally (auto-selects when unset)."""
        self.resolve_model(installed)

    def _verify_connection(self) -> None:
        """Reachability + model check, cached for ``_CHECK_TTL_SECONDS``.

        A successful check is reused for the TTL so repeated generation calls
        do not hit ``/api/tags`` each time. Any failure invalidates the cache
        so the next call retries immediately and surfaces the error.
        """
        now = time.time()
        if self._verified and (now - self._checked_at) < _CHECK_TTL_SECONDS:
            return
        data = self.check_connection()
        names = [m["name"] for m in data.get("models", [])]
        self.ensure_model(installed=names)
        self._verified = True
        self._checked_at = time.time()

    def _invalidate_check(self) -> None:
        self._verified = False
        self._checked_at = 0.0

    # -- generation ----------------------------------------------------------

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Send a chat request and return the assistant text.

        ``kwargs`` may override the default temperature / num_ctx options.
        """
        self._verify_connection()

        options = {**self.options, **kwargs}
        # `think` is a top-level request flag for Qwen3-style reasoning models.
        think = options.pop("think", None)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if think is not None:
            payload["think"] = bool(think)

        logger.info("[OLLAMA] Sending generation request (model=%s)", self.model)
        try:
            resp = self._client.post(
                "/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            # Ollama returns Qwen3's private reasoning in ``message.thinking``.
            # With a long RAG context the reasoning can consume the whole
            # prediction budget and leave ``message.content`` empty. In that
            # case, preserve the requested reasoning pass, then perform one
            # bounded local finalization pass with thinking disabled. Only the
            # final answer is returned; private reasoning is never logged or
            # exposed to the API/GUI.
            message = data.get("message") or {}
            content = str(message.get("content") or "").strip()
            thinking_text = str(message.get("thinking") or "").strip()
            if think is True and not content and thinking_text and "error" not in data:
                logger.warning(
                    "[OLLAMA] Thinking pass returned no visible content; "
                    "running one final-answer pass."
                )
                final_payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "assistant", "content": thinking_text},
                        {
                            "role": "user",
                            "content": (
                                "Using the reasoning you just completed, return only "
                                "the final user-facing answer in the user's language. "
                                "Do not mention reasoning, evidence, context, sources, "
                                "or internal steps."
                            ),
                        },
                    ],
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "think": False,
                    "options": {
                        **options,
                        "num_predict": min(
                            1200, max(512, int(options.get("num_predict", 1200)))
                        ),
                    },
                }
                final_resp = self._client.post("/api/chat", json=final_payload)
                final_resp.raise_for_status()
                data = final_resp.json()
        except httpx.ReadTimeout as exc:
            self._invalidate_check()
            raise OllamaError(
                f"Generation timed out after {self.timeout}s (model {self.model}). "
                "The answer may be too long for this model on this machine. "
                "Reduce OLLAMA_MAX_OUTPUT_TOKENS, try a faster model "
                "(OLLAMA_MODEL=qwen3:4b), or raise OLLAMA_TIMEOUT.",
                error_type="generation_timeout",
                details={
                    "model": self.model,
                    "timeout_s": self.timeout,
                    "category": "generation_timeout",
                },
            ) from exc
        except (httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
            self._invalidate_check()
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}: {type(exc).__name__}. "
                "Is the Ollama service running?",
                error_type="connection",
                details={
                    "model": self.model,
                    "timeout_s": self.timeout,
                    "category": "connection",
                },
            ) from exc
        except httpx.HTTPError as exc:
            self._invalidate_check()
            raise OllamaError(
                f"Ollama request failed: {exc}", error_type="generation_error"
            ) from exc

        if "error" in data:
            raise OllamaError(
                f"Ollama returned an error: {data['error']}",
                error_type="generation_error",
            )

        return data.get("message", {}).get("content", "")
