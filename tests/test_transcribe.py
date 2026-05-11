"""Tests for the OpenRouter transcription client (port of llm_client/remote.rs).

The wire contract is fixed (see docs/plan-voxd.md). These tests assert:

* the request body shape matches the Rust reference bit-for-bit;
* the model catalogue dispatches `language`/`prompt` correctly;
* HTTP status codes are classified as documented (401 -> AuthInvalid,
  429/5xx -> Retryable, other 4xx -> Permanent, network -> Retryable);
* the retry policy executes the exponential backoff schedule (250ms * 2^(n-1))
  and gives up after MAX_RETRIES additional attempts.

We avoid an extra ``pytest-asyncio`` dependency: each test is synchronous and
calls a tiny ``_run`` helper that drives the coroutine on a fresh event loop.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
import pytest

from voxd import transcribe
from voxd.transcribe import (
    MAX_RETRIES,
    MODELS,
    AuthInvalid,
    Model,
    Permanent,
    Retryable,
    backoff_delay,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


# A tiny but valid-looking WAV header — body-shape tests check the magic bytes.
_FAKE_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"


def _run(coro: Awaitable[T]) -> T:
    """Run an awaitable on a fresh event loop and return its result."""
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise ``asyncio.sleep`` so retry tests finish in milliseconds.

    Backoff correctness is verified separately by ``test_backoff_schedule``
    (pure function). All retry-path tests only need to assert the number
    of attempts and the final outcome — they should not spend seconds
    sleeping between attempts.
    """

    async def _noop(_delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop)


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    """Build an ``AsyncClient`` whose every request hits ``handler``."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _do_transcribe(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str = "sk-test",
    model_id: str = "openai/whisper-large-v3-turbo",
    language: str = "auto",
    prompt: str = "",
    wav: bytes = _FAKE_WAV,
) -> str:
    """Drive ``transcribe.transcribe`` against an in-process mock transport."""
    async with _make_client(handler) as client:
        return await transcribe.transcribe(
            wav,
            api_key=api_key,
            model_id=model_id,
            language=language,
            prompt=prompt,
            client=client,
        )


def _body_of(captured: dict[str, object]) -> dict[str, object]:
    """Decode the JSON body recorded by a capturing handler."""
    raw = captured["body"]
    assert isinstance(raw, (bytes, bytearray))
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_stt_200_returns_text() -> None:
    """A 200 OK with a ``text`` field surfaces directly as the return value."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/audio/transcriptions")
        return httpx.Response(200, json={"text": "hello world"})

    result = _run(_do_transcribe(handler))
    assert result == "hello world"


def test_body_shape() -> None:
    """Body is ``{model, input_audio:{data:b64, format:wav}, ...}`` with WAV magic intact."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"text": "ok"})

    _run(_do_transcribe(handler))

    body = _body_of(captured)
    assert body["model"] == "openai/whisper-large-v3-turbo"
    input_audio = body["input_audio"]
    assert isinstance(input_audio, dict)
    assert input_audio["format"] == "wav"
    decoded = base64.b64decode(input_audio["data"])  # type: ignore[arg-type]
    assert decoded.startswith(b"RIFF")
    assert decoded == _FAKE_WAV

    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer sk-test"
    assert headers["content-type"].startswith("application/json")
    assert "voxd" in headers["user-agent"].lower()
    assert headers["x-title"] == "voxd"


def test_language_omitted_when_auto() -> None:
    """``language="auto"`` means *let the provider auto-detect* — key absent."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "ok"})

    _run(_do_transcribe(handler, language="auto"))
    assert "language" not in _body_of(captured)


def test_language_present_when_supported() -> None:
    """Whisper supports ``supports_language``; ``language='fr'`` ends up in the body."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "ok"})

    _run(_do_transcribe(handler, language="fr"))
    assert _body_of(captured)["language"] == "fr"


def test_language_omitted_when_model_does_not_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model with ``supports_language=False`` never gets a ``language`` key."""
    monkeypatch.setitem(
        MODELS,
        "test/no-lang",
        Model(id="test/no-lang", supports_prompt=True, supports_language=False),
    )

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "ok"})

    _run(_do_transcribe(handler, model_id="test/no-lang", language="fr"))
    assert "language" not in _body_of(captured)


def test_prompt_omitted_when_empty() -> None:
    """Empty user prompt -> no ``prompt`` key even on models that support it."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "ok"})

    _run(_do_transcribe(handler, prompt=""))
    assert "prompt" not in _body_of(captured)


def test_prompt_omitted_when_model_does_not_support() -> None:
    """Chirp-3 has ``supports_prompt=False`` — user prompt is dropped silently."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "ok"})

    _run(_do_transcribe(handler, model_id="google/chirp-3", prompt="vocabulary: foo"))
    assert "prompt" not in _body_of(captured)


def test_prompt_present_when_supported() -> None:
    """Whisper has ``supports_prompt=True`` — user prompt is forwarded verbatim."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "ok"})

    _run(_do_transcribe(handler, prompt="vocabulary: foo bar"))
    assert _body_of(captured)["prompt"] == "vocabulary: foo bar"


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


def test_401_raises_auth_invalid_no_retry() -> None:
    """401 means a bad key — surface AuthInvalid immediately, never retry."""
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(401, json={"error": "invalid_api_key"})

    with pytest.raises(AuthInvalid):
        _run(_do_transcribe(handler))
    assert call_count == 1


def test_400_raises_permanent_no_retry() -> None:
    """400 (and any other 4xx ≠ 401, 429) is a permanent client error, no retry."""
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(400, json={"error": "bad_request"})

    with pytest.raises(Permanent):
        _run(_do_transcribe(handler))
    assert call_count == 1


def test_429_then_200_retries_and_succeeds() -> None:
    """A single transient 429 is retried — the 2nd attempt's 200 wins."""
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, json={"error": "rate_limited"})
        return httpx.Response(200, json={"text": "after retry"})

    result = _run(_do_transcribe(handler))
    assert result == "after retry"
    assert call_count == 2


def test_5_consecutive_500_raises_retryable() -> None:
    """``MAX_RETRIES=5`` -> 6 attempts total; persistent 5xx ends in Retryable."""
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(500, json={"error": "server_error"})

    with pytest.raises(Retryable):
        _run(_do_transcribe(handler))
    assert call_count == MAX_RETRIES + 1 == 6


def test_network_error_raises_retryable() -> None:
    """A connection-level error (no HTTP response) is Retryable, retried 6 times."""
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("no route to host")

    with pytest.raises(Retryable):
        _run(_do_transcribe(handler))
    assert call_count == MAX_RETRIES + 1 == 6


def test_empty_api_key_raises_auth_invalid() -> None:
    """An empty key short-circuits before any HTTP call (port of Rust check)."""
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"text": "never reached"})

    with pytest.raises(AuthInvalid):
        _run(_do_transcribe(handler, api_key=""))
    assert call_count == 0


def test_unknown_model_raises_permanent() -> None:
    """Asking for a model id absent from the catalogue is a programmer error."""

    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return httpx.Response(200, json={"text": "nope"})

    with pytest.raises(Permanent):
        _run(_do_transcribe(handler, model_id="does/not-exist"))


def test_200_missing_text_field_raises_permanent() -> None:
    """If the server returns 200 but no ``text`` field, treat it as a bug."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {"tokens": 42}})

    with pytest.raises(Permanent):
        _run(_do_transcribe(handler))


# ---------------------------------------------------------------------------
# Backoff schedule (pure function)
# ---------------------------------------------------------------------------


def test_backoff_schedule() -> None:
    """Schedule from the plan: 0, 250, 500, 1000, 2000, 4000 ms (no jitter)."""
    no_jitter: Callable[[float], float] = lambda d: d  # noqa: E731

    assert backoff_delay(0, jitter_fn=no_jitter) == 0.0
    assert backoff_delay(1, jitter_fn=no_jitter) == pytest.approx(0.250)
    assert backoff_delay(2, jitter_fn=no_jitter) == pytest.approx(0.500)
    assert backoff_delay(3, jitter_fn=no_jitter) == pytest.approx(1.000)
    assert backoff_delay(4, jitter_fn=no_jitter) == pytest.approx(2.000)
    assert backoff_delay(5, jitter_fn=no_jitter) == pytest.approx(4.000)


def test_backoff_jitter_bound() -> None:
    """Default jitter must keep every delay within ±20 % of the schedule."""
    schedule_ms = {1: 250, 2: 500, 3: 1000, 4: 2000, 5: 4000}
    for attempt, expected_ms in schedule_ms.items():
        for _ in range(50):
            delay = backoff_delay(attempt)
            assert delay >= 0.0
            lo = expected_ms * 0.8 / 1000.0
            hi = expected_ms * 1.2 / 1000.0
            assert lo <= delay <= hi, f"attempt={attempt} delay={delay} not in [{lo}, {hi}]"


def test_backoff_zero_attempt_is_zero_regardless_of_jitter() -> None:
    """Attempt 0 always returns 0 — no wait before the first try."""
    assert backoff_delay(0, jitter_fn=lambda d: d * 1000) == 0.0
