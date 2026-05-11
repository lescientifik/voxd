"""OpenRouter ``/audio/transcriptions`` client.

Port of ``llm_client/{remote,error,catalog,stt_body,parse}.rs`` from the
Rust ``handy`` reference codebase.

What this module does
---------------------

* Builds the OpenRouter STT request body (model id + base64-encoded WAV +
  optional ``language`` / ``prompt`` keys, gated by the model's capabilities).
* POSTs to ``{base_url}/audio/transcriptions`` with the same headers the Rust
  reference sends (``Authorization``, ``Content-Type``, ``User-Agent``,
  ``X-Title``).
* Classifies failures into three exception types — ``AuthInvalid`` (401),
  ``Permanent`` (other 4xx, malformed responses) and ``Retryable`` (429, 5xx,
  network) — and retries the last on an exponential backoff schedule with
  ±20 % jitter, up to ``MAX_RETRIES`` extra attempts.

The retry budget, backoff schedule and error taxonomy mirror the Rust
implementation bit-for-bit; the only Python-specific choices are the
async machinery (``asyncio.sleep``, ``httpx.AsyncClient``) and the
testability hooks (injectable ``client`` and ``jitter_fn``).
"""

from __future__ import annotations

import asyncio
import base64
import random
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from voxd import __version__

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


# Exception names are fixed by the plan contract (port of the Rust
# ``TranscriptionError`` variants); ``N818`` would force ``*Error`` suffixes.
class AuthInvalid(Exception):  # noqa: N818
    """The OpenRouter API key is missing, empty, or rejected with 401.

    No retry happens for this case — the user must fix their config.
    """


class Retryable(Exception):  # noqa: N818
    """Transient failure (429, 5xx, network error) — eligible for retry.

    Raised only after the retry budget has been exhausted.
    """


class Permanent(Exception):  # noqa: N818
    """Definitive failure with no useful retry semantics.

    Covers 4xx (other than 401/429), malformed 2xx responses, and unknown
    model ids.
    """


@dataclass(frozen=True)
class Model:
    """Static descriptor for an OpenRouter-exposed transcription model.

    The ``endpoint`` field is kept for forward compatibility: V1 only ships
    STT models, but the Rust catalogue dispatches between STT and Chat
    endpoints. Keeping the field here means a future Chat-endpoint addition
    is a one-line catalogue change rather than a schema migration.
    """

    id: str
    supports_prompt: bool
    supports_language: bool
    endpoint: str = "stt"


MODELS: dict[str, Model] = {
    "openai/whisper-large-v3-turbo": Model(
        id="openai/whisper-large-v3-turbo",
        supports_prompt=True,
        supports_language=True,
    ),
    "openai/whisper-large-v3": Model(
        id="openai/whisper-large-v3",
        supports_prompt=True,
        supports_language=True,
    ),
    "openai/gpt-4o-transcribe": Model(
        id="openai/gpt-4o-transcribe",
        supports_prompt=True,
        supports_language=True,
    ),
    "openai/gpt-4o-mini-transcribe": Model(
        id="openai/gpt-4o-mini-transcribe",
        supports_prompt=True,
        supports_language=True,
    ),
    "google/chirp-3": Model(
        id="google/chirp-3",
        supports_prompt=False,
        supports_language=True,
    ),
}

DEFAULT_MODEL = "openai/whisper-large-v3-turbo"

#: Maximum number of *additional* retry attempts after the first try.
#: Total attempts the client will make: ``MAX_RETRIES + 1`` (= 6).
MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# Backoff schedule (pure function — directly testable)
# ---------------------------------------------------------------------------


def _default_jitter(delay: float) -> float:
    """Return ``delay`` perturbed by ±20 %.

    Mirrors the Rust default jitter: pick a factor uniformly in
    ``[-20, +20]`` percent and apply it. The Rust version derives entropy
    from the current nanosecond; we use ``random.uniform`` — identical for
    our purposes, since jitter only needs to break tie-locked retries.
    """
    factor = random.uniform(-0.20, 0.20)
    return max(0.0, delay * (1.0 + factor))


def backoff_delay(
    attempt: int,
    *,
    jitter_fn: Callable[[float], float] | None = None,
) -> float:
    """Return the sleep duration (seconds) before retry attempt ``attempt``.

    Schedule (no jitter): ``0, 0.250, 0.500, 1.000, 2.000, 4.000`` seconds
    for attempts ``0..=5``. Attempts beyond ``MAX_RETRIES`` are clamped to
    the last entry — they should never happen in practice because the
    retry loop stops at ``MAX_RETRIES + 1`` total attempts.

    Args:
        attempt: Zero-based retry index. ``0`` is the very first try and
            always returns ``0`` (no pre-wait).
        jitter_fn: Optional override used by tests to make the result
            deterministic. The default applies ±20 % uniform jitter.

    Returns:
        Number of seconds to sleep before the next request.
    """
    if attempt <= 0:
        return 0.0
    clamped = min(attempt, MAX_RETRIES)
    base = 0.250 * (1 << (clamped - 1))  # 0.250 * 2^(clamped-1)
    fn = jitter_fn if jitter_fn is not None else _default_jitter
    return fn(base)


# ---------------------------------------------------------------------------
# Body / headers
# ---------------------------------------------------------------------------


_USER_AGENT = f"voxd/{__version__} (+https://github.com/lescientifik/voxd)"


def _build_headers(api_key: str) -> dict[str, str]:
    """Return the request headers for ``/audio/transcriptions``."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
        "X-Title": "voxd",
    }


def _build_stt_body(
    model: Model,
    wav_bytes: bytes,
    language: str,
    user_prompt: str,
) -> dict[str, object]:
    """Build the JSON body — base64-encoded WAV plus gated language/prompt keys."""
    body: dict[str, object] = {
        "model": model.id,
        "input_audio": {
            "data": base64.b64encode(wav_bytes).decode("ascii"),
            "format": "wav",
        },
    }
    if model.supports_language and language != "auto":
        body["language"] = language
    if model.supports_prompt and user_prompt:
        body["prompt"] = user_prompt
    return body


# ---------------------------------------------------------------------------
# Error classification + response parsing
# ---------------------------------------------------------------------------


def _classify_status(status: int) -> type[Exception]:
    """Map a non-2xx HTTP status to the exception class to raise.

    Returns one of ``AuthInvalid``, ``Retryable``, ``Permanent`` (the type
    object itself, not an instance — the caller adds the message).
    """
    if status == 401:
        return AuthInvalid
    if status == 429 or 500 <= status < 600:
        return Retryable
    return Permanent


def _parse_stt_response(data: object) -> str:
    """Extract the top-level ``text`` field from a parsed JSON response."""
    if not isinstance(data, dict):
        raise Permanent("STT response is not a JSON object")
    text = data.get("text")  # ty: ignore[invalid-argument-type]
    if not isinstance(text, str):
        raise Permanent("STT response missing 'text' field")
    return text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def transcribe(
    wav_bytes: bytes,
    *,
    api_key: str,
    model_id: str,
    language: str,
    prompt: str,
    base_url: str = "https://openrouter.ai/api/v1",
    client: httpx.AsyncClient | None = None,
    jitter_fn: Callable[[float], float] | None = None,
) -> str:
    """Transcribe ``wav_bytes`` via OpenRouter and return the recognised text.

    Retries 429/5xx/network errors with exponential backoff (see
    ``backoff_delay``), up to ``MAX_RETRIES`` additional attempts. 401 and
    other 4xx surface immediately as ``AuthInvalid`` / ``Permanent``.

    Args:
        wav_bytes: A complete WAV file (16 kHz mono PCM int16 — see
            ``voxd.wav.samples_to_wav_bytes``).
        api_key: OpenRouter API key. Empty string raises ``AuthInvalid``
            without making any HTTP call.
        model_id: Catalogue key (see ``MODELS``). Unknown ids raise
            ``Permanent``.
        language: ISO 639-1 code or ``"auto"``. Forwarded only if the
            model declares ``supports_language``.
        prompt: User-provided prompt. Forwarded only if non-empty *and*
            the model declares ``supports_prompt``.
        base_url: OpenRouter API root. Override for tests / proxies.
        client: Optional ``httpx.AsyncClient`` to reuse. Tests inject one
            wired to ``httpx.MockTransport``; production code lets us build
            a fresh client per call.
        jitter_fn: Optional override for the backoff jitter function.

    Returns:
        The transcribed text (top-level ``text`` field of the response).

    Raises:
        AuthInvalid: Missing/invalid API key or HTTP 401.
        Permanent: Unknown model id, malformed response, or any 4xx other
            than 401/429.
        Retryable: Retry budget exhausted after persistent 429/5xx/network
            failures.
    """
    if not api_key:
        raise AuthInvalid("OpenRouter API key is missing")

    model = MODELS.get(model_id)
    if model is None:
        raise Permanent(f"unknown model id: {model_id!r}")

    headers = _build_headers(api_key)
    body = _build_stt_body(model, wav_bytes, language, prompt)
    url = f"{base_url}/audio/transcriptions"

    owns_client = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=30.0)
    try:
        return await _post_with_retry(http, url, headers, body, jitter_fn=jitter_fn)
    finally:
        if owns_client:
            await http.aclose()


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, object],
    *,
    jitter_fn: Callable[[float], float] | None,
) -> str:
    """Drive the retry loop. Returns the parsed text or raises a typed error.

    The loop is structured so each iteration sleeps ``backoff_delay(attempt)``
    *before* the request — that puts ``0`` seconds before attempt 0 and the
    full exponential schedule before subsequent attempts. The last
    classified error is preserved so the final raise carries a useful
    message.
    """
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        delay = backoff_delay(attempt, jitter_fn=jitter_fn)
        if delay > 0:
            await asyncio.sleep(delay)

        try:
            response = await client.post(url, headers=headers, json=body)
        except httpx.RequestError as exc:
            # No HTTP response at all (DNS, connect, read, write timeouts).
            last_error = Retryable(f"network error: {exc}")
            continue

        if response.is_success:
            try:
                data = response.json()
            except ValueError as exc:
                raise Permanent(f"response is not valid JSON: {exc}") from exc
            return _parse_stt_response(data)

        exc_cls = _classify_status(response.status_code)
        message = f"HTTP {response.status_code}"
        if exc_cls is Retryable:
            last_error = exc_cls(message)
            continue
        # AuthInvalid / Permanent: give up immediately.
        raise exc_cls(message)

    # Retry budget exhausted.
    if last_error is None:  # pragma: no cover - defensive
        raise Retryable("retry budget exhausted")
    raise last_error


__all__ = [
    "DEFAULT_MODEL",
    "MAX_RETRIES",
    "MODELS",
    "AuthInvalid",
    "Model",
    "Permanent",
    "Retryable",
    "backoff_delay",
    "transcribe",
]
