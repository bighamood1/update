"""Unit tests for the Ollama client connectivity cache and context sizing."""

from __future__ import annotations

import time

import httpx

from rag.generation.ollama_client import _CHECK_TTL_SECONDS, OllamaClient
from rag.pipeline.rag import ContextBuilder


class _FakeOllama(OllamaClient):
    """OllamaClient with no network: counts verification calls."""

    def __init__(self) -> None:
        self.calls = 0
        self._verified = False
        self._checked_at = 0.0

    def check_connection(self) -> dict:
        self.calls += 1
        return {"models": []}

    def ensure_model(self, installed=None) -> None:
        pass

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        return "ok"


def test_connection_check_is_cached():
    client = _FakeOllama()
    client._verify_connection()
    client._verify_connection()
    client._verify_connection()
    assert client.calls == 1  # only the first call hits /api/tags


def test_connection_check_invalidates_after_ttl():
    client = _FakeOllama()
    client._verify_connection()
    client._checked_at -= _CHECK_TTL_SECONDS + 1
    client._verify_connection()
    assert client.calls == 2


def test_context_builder_reads_config_defaults():
    builder = ContextBuilder()
    # FINAL_CONTEXT_CHUNKS defaults to 4, CONTEXT_MAX_CHARS to 4000.
    assert builder.max_chunks == 4
    assert builder.max_chars == 4000


def test_context_builder_explicit_args_win():
    builder = ContextBuilder(max_chunks=2, max_chars=100)
    assert builder.max_chunks == 2
    assert builder.max_chars == 100


def test_thinking_only_response_gets_local_final_answer_pass():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(__import__("json").loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(
                200,
                json={"message": {"content": "", "thinking": "The dean is Wael."}},
            )
        return httpx.Response(
            200,
            json={"message": {"content": "عميد كلية الهندسة هو أ.د وائل صديق."}},
        )

    client = OllamaClient.__new__(OllamaClient)
    client.model = "qwen3:4b"
    client.options = {"think": True, "num_predict": 2500}
    client.keep_alive = "30m"
    client.timeout = 60.0
    client.base_url = "http://test"
    client._verified = True
    client._checked_at = time.time()
    client._closed = False
    client._client = httpx.Client(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
    )
    try:
        answer = client.generate("system", "question")
    finally:
        client.close()

    assert answer == "عميد كلية الهندسة هو أ.د وائل صديق."
    assert len(payloads) == 2
    assert payloads[0]["think"] is True
    assert payloads[1]["think"] is False
