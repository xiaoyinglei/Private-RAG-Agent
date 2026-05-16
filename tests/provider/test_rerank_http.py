from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from rag.providers.rerank_http import RerankHttpClient


class _FakeTransport(httpx.BaseTransport):
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler = handler

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)


def _json_response(status_code: int, data: object) -> httpx.Response:
    return httpx.Response(status_code, json=data)


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> RerankHttpClient:
    return RerankHttpClient(
        "http://127.0.0.1:9091",
        client=httpx.Client(transport=_FakeTransport(handler)),
    )


# ── Health check failures ──


def test_health_check_fails_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "internal error"})

    with pytest.raises(RuntimeError, match="health check failed"):
        _make_client(handler)


def test_health_check_fails_on_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {})

    with pytest.raises(RuntimeError, match="health check"):
        _make_client(handler)


# ── Successful construction ──


def test_health_check_success_stores_model_name() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"model": "bge-reranker-v2-m3"})

    client = _make_client(handler)
    assert client.rerank_model_name == "bge-reranker-v2-m3"


# ── rerank ──


def test_rerank_empty_documents_returns_empty() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/health":
            call_count += 1
            return _json_response(200, {"model": "m"})
        raise AssertionError("unexpected request")

    client = _make_client(handler)
    assert client.rerank("query", []) == []


def test_rerank_returns_correct_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m"})
        if request.url.path == "/v1/rerank":
            body = json.loads(request.content)
            n = len(body["documents"])
            return _json_response(200, {"scores": [0.9] * n})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = _make_client(handler)
    result = client.rerank("query", ["doc1", "doc2", "doc3"])
    assert len(result) == 3
    assert result == [0.9, 0.9, 0.9]


def test_rerank_count_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m"})
        if request.url.path == "/v1/rerank":
            return _json_response(200, {"scores": [0.9]})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = _make_client(handler)
    with pytest.raises(RuntimeError, match="count mismatch"):
        client.rerank("query", ["doc1", "doc2"])


def test_rerank_service_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m"})
        if request.url.path == "/v1/rerank":
            return _json_response(500, {"detail": "rerank failed"})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = _make_client(handler)
    with pytest.raises(RuntimeError, match="Rerank service error"):
        client.rerank("query", ["doc1"])


def test_rerank_request_body_has_expected_shape() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m"})
        if request.url.path == "/v1/rerank":
            nonlocal captured
            captured = json.loads(request.content)
            return _json_response(200, {"scores": [0.5, 0.8]})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = _make_client(handler)
    client.rerank("what is AI?", ["doc about AI", "doc about birds"])
    assert captured["query"] == "what is AI?"
    assert captured["documents"] == ["doc about AI", "doc about birds"]
