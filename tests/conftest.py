"""Shared pytest fixtures for voxd tests.

This module provides the building blocks used by the daemon end-to-end tests:

* :func:`monkey_xdg_runtime` — point ``$XDG_RUNTIME_DIR`` at a tmp dir.
* :func:`capture_subprocess` — record every ``subprocess.run`` call across the
  inject/indicator modules; supports awaiting a specific binary invocation.
* :func:`mock_openrouter_server` — programmable httpx ``MockTransport`` that
  counts requests and returns a queued response stack.
* :func:`fake_audio_input` — patches :class:`voxd.audio.AudioCapture` with a
  fake that replays a fixture WAV via the daemon's ``on_frame`` callback in a
  background thread (mimicking PortAudio).
* :func:`make_test_config` — helper to assemble a minimal in-memory
  :class:`voxd.config.Config`.

The fixtures only kick in when explicitly requested by a test, so the existing
unit-test suite is unaffected.
"""

from __future__ import annotations

import asyncio
import io
import subprocess
import sys
import threading
import time
import types
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sounddevice stub — same trick as test_audio.py so the package imports on a
# CI box without PortAudio.
# ---------------------------------------------------------------------------


def _install_fake_sounddevice() -> None:
    """Insert a stub ``sounddevice`` module if PortAudio is absent."""
    if "sounddevice" in sys.modules:
        return
    fake = types.ModuleType("sounddevice")

    def _raise(*args: object, **kwargs: object) -> Any:  # noqa: ARG001
        raise RuntimeError("sounddevice stub: monkeypatch the real symbol in your test")

    fake.InputStream = _raise  # type: ignore[attr-defined]
    fake.query_devices = _raise  # type: ignore[attr-defined]
    sys.modules["sounddevice"] = fake


_install_fake_sounddevice()


FIXTURES = Path(__file__).parent / "fixtures"
SPEECH_FR_WAV = FIXTURES / "speech_fr.wav"
SILENCE_WAV = FIXTURES / "silence.wav"


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def make_test_config(
    tmp_path: Path,  # noqa: ARG001 — kept for symmetry / future use (e.g. cfg paths)
    *,
    api_key: str = "test-key",
    language: str = "auto",
    prompt: str = "",
    model: str = "openai/whisper-large-v3-turbo",
    type_inject: bool = True,
    clipboard: bool = True,
) -> Any:
    """Return a :class:`voxd.config.Config` with overridable test values."""
    from voxd.config import AudioConfig, Config, InjectConfig, OpenRouterConfig

    return Config(
        openrouter=OpenRouterConfig(
            api_key=api_key, model=model, language=language, prompt=prompt
        ),
        audio=AudioConfig(device=""),
        inject=InjectConfig(type=type_inject, clipboard=clipboard),
    )


# ---------------------------------------------------------------------------
# XDG runtime dir
# ---------------------------------------------------------------------------


@pytest.fixture
def monkey_xdg_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point ``$XDG_RUNTIME_DIR`` at ``tmp_path`` for the duration of the test."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    yield tmp_path


# ---------------------------------------------------------------------------
# subprocess.run capture
# ---------------------------------------------------------------------------


@dataclass
class _SubprocessCall:
    """Snapshot of a single :func:`subprocess.run` invocation."""

    argv: list[str]
    stdin: str | None
    kwargs: dict[str, Any] = field(default_factory=dict)


class _SubprocessRecorder:
    """Capture every ``subprocess.run`` call (across all modules).

    The fake ``run`` returns a fake :class:`subprocess.CompletedProcess` with
    ``returncode=0``. Callers needing different behaviour (e.g. simulating
    ``notify-send -p`` printing an id) can register a per-binary handler via
    :meth:`set_handler`.
    """

    def __init__(self) -> None:
        self._calls: list[_SubprocessCall] = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._handlers: dict[
            str, Callable[[_SubprocessCall], subprocess.CompletedProcess[str]]
        ] = {}

    # ---- registration ----

    def set_handler(
        self,
        binary: str,
        handler: Callable[[_SubprocessCall], subprocess.CompletedProcess[str]],
    ) -> None:
        """Override the response for ``binary``."""
        self._handlers[binary] = handler

    # ---- query ----

    def calls(self, binary: str) -> list[_SubprocessCall]:
        """Return all recorded calls whose argv[0] basename equals ``binary``."""
        with self._lock:
            return [c for c in self._calls if Path(c.argv[0]).name == binary]

    def all_calls(self) -> list[_SubprocessCall]:
        """Return every recorded call in invocation order."""
        with self._lock:
            return list(self._calls)

    async def wait_for_call(self, binary: str, timeout: float = 5.0) -> _SubprocessCall:
        """Asynchronously wait until at least one call to ``binary`` is recorded.

        Returns the first matching call (subsequent ones are still accessible
        via :meth:`calls`). Raises :class:`TimeoutError` if the deadline elapses.
        """
        deadline = time.monotonic() + timeout
        while True:
            for call in self.calls(binary):
                return call
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no call to {binary!r} within {timeout}s")
            await asyncio.sleep(0.01)

    # ---- internal ----

    def _record(self, argv: list[str], stdin: str | None, kwargs: dict[str, Any]) -> None:
        call = _SubprocessCall(argv=argv, stdin=stdin, kwargs=kwargs)
        with self._cond:
            self._calls.append(call)
            self._cond.notify_all()

    def _run(self, *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        argv = list(args[0]) if args else list(kwargs.get("args", []))
        stdin_input = kwargs.get("input")
        if isinstance(stdin_input, bytes):
            stdin_input = stdin_input.decode("utf-8", errors="replace")
        call = _SubprocessCall(argv=argv, stdin=stdin_input, kwargs=dict(kwargs))
        with self._cond:
            self._calls.append(call)
            self._cond.notify_all()

        binary = Path(argv[0]).name if argv else ""
        handler = self._handlers.get(binary)
        if handler is not None:
            return handler(call)
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")


@pytest.fixture
def capture_subprocess(monkeypatch: pytest.MonkeyPatch) -> Iterator[_SubprocessRecorder]:
    """Capture every ``subprocess.run`` issued by voxd (inject, indicator)."""
    rec = _SubprocessRecorder()
    # Patch the symbol on each module that imported it at module level.
    import voxd.indicator
    import voxd.inject

    monkeypatch.setattr(voxd.inject.subprocess, "run", rec._run)
    monkeypatch.setattr(voxd.indicator.subprocess, "run", rec._run)
    yield rec


# ---------------------------------------------------------------------------
# Mock OpenRouter server (httpx.MockTransport wrapper)
# ---------------------------------------------------------------------------


@dataclass
class _MockResponse:
    """Programmable response queue entry."""

    status: int
    body: dict[str, Any] | None = None
    body_bytes: bytes | None = None


class MockOpenRouter:
    """Programmable mock for OpenRouter's ``/audio/transcriptions`` endpoint.

    Tests configure responses via :meth:`set_responses` and read back the
    request count, last body, etc. The mock exposes an :class:`httpx.AsyncClient`
    via :attr:`client` that the daemon (or anything calling
    :func:`voxd.transcribe.transcribe`) can use directly.
    """

    DEFAULT_URL = "https://mock.local/api/v1"

    def __init__(self, *, default_text: str = "hello world") -> None:
        self.url = self.DEFAULT_URL
        self.request_count = 0
        self.requests: list[httpx.Request] = []
        self._responses: list[_MockResponse] = []
        self._default_text = default_text
        self.client: httpx.AsyncClient = httpx.AsyncClient(
            transport=httpx.MockTransport(self._handle)
        )

    def set_responses(self, *responses: tuple[int, dict[str, Any] | None]) -> None:
        """Queue a list of ``(status, body)`` responses, consumed in order."""
        self._responses = [_MockResponse(status=s, body=b) for s, b in responses]

    def queue_text(self, *texts: str) -> None:
        """Convenience: queue N 200 OK responses each returning ``{"text": t}``."""
        self._responses = [
            _MockResponse(status=200, body={"text": t}) for t in texts
        ]

    def reset(self) -> None:
        """Clear request log and response queue."""
        self.request_count = 0
        self.requests.clear()
        self._responses.clear()

    async def aclose(self) -> None:
        """Close the underlying ``AsyncClient``."""
        await self.client.aclose()

    # ---- internal ----

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.request_count += 1
        self.requests.append(request)
        if self._responses:
            resp = self._responses.pop(0)
        else:
            resp = _MockResponse(status=200, body={"text": self._default_text})
        if resp.body_bytes is not None:
            return httpx.Response(resp.status, content=resp.body_bytes)
        if resp.body is None:
            return httpx.Response(resp.status)
        return httpx.Response(resp.status, json=resp.body)


@pytest.fixture
def mock_openrouter_server() -> Iterator[MockOpenRouter]:
    """Provide a :class:`MockOpenRouter` for the duration of one test."""
    mock = MockOpenRouter()
    try:
        yield mock
    finally:
        # Close the AsyncClient on a fresh loop — the test loop may already be
        # closed at teardown time.
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(mock.aclose())
            finally:
                loop.close()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Fake audio input — replaces voxd.audio.AudioCapture with a WAV replayer.
# ---------------------------------------------------------------------------


def _read_wav_mono_16k(path: Path) -> np.ndarray:
    """Load a 16 kHz mono int16 WAV into float32 samples in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000, f"{path} must be 16 kHz"
        assert w.getnchannels() == 1, f"{path} must be mono"
        assert w.getsampwidth() == 2, f"{path} must be int16"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


FRAME_SAMPLES = 480  # 30 ms @ 16 kHz — matches voxd.audio FRAME_SAMPLES


class FakeAudioInput:
    """Drop-in replacement for :class:`voxd.audio.AudioCapture` used by tests.

    The fake replays a pre-recorded ``samples`` numpy array (mono, 16 kHz,
    float32) into the daemon's ``on_frame`` callback, chunked into 480-sample
    frames — i.e. the exact contract a real AudioCapture exposes. The replay
    happens on a background daemon thread that yields between frames so the
    asyncio loop can interleave its own coroutines (mirroring PortAudio
    semantics).
    """

    def __init__(self) -> None:
        # Default to silence so an unparameterised test produces an empty
        # recording (used by ``test_empty_recording_no_upload``).
        self._samples: np.ndarray = np.zeros(0, dtype=np.float32)
        self._created: list[_FakeCapture] = []

    # ---- test API ----

    def set_samples(self, samples: np.ndarray) -> None:
        """Set the float32 mono 16 kHz audio buffer to replay on next start."""
        self._samples = np.asarray(samples, dtype=np.float32)

    def load_wav(self, path: Path) -> None:
        """Load a 16 kHz mono int16 WAV fixture into the replay buffer."""
        self.set_samples(_read_wav_mono_16k(path))

    def set_silence(self, duration_s: float = 1.0) -> None:
        """Replace the buffer with ``duration_s`` of pure silence."""
        n = int(round(duration_s * 16000))
        self.set_samples(np.zeros(n, dtype=np.float32))

    def set_speech_then_silence(
        self, duration_s: float, *, source_wav: Path = SPEECH_FR_WAV
    ) -> None:
        """Replace the buffer with the first ``duration_s`` of a speech fixture."""
        full = _read_wav_mono_16k(source_wav)
        n = int(round(duration_s * 16000))
        self.set_samples(full[:n])

    async def replay_finished(self, timeout: float = 5.0) -> None:
        """Wait until every currently-running fake capture finishes replaying."""
        deadline = time.monotonic() + timeout
        while True:
            active = [c for c in self._created if c.started and not c.replay_done]
            if not active:
                return
            if time.monotonic() > deadline:
                raise TimeoutError("fake audio replay did not finish in time")
            await asyncio.sleep(0.01)

    # ---- internal: capture factory ----

    def _spawn(
        self,
        device: str,  # noqa: ARG002
        on_frame: Callable[[np.ndarray], None],
        loop: asyncio.AbstractEventLoop,
    ) -> _FakeCapture:
        """Return a fresh :class:`_FakeCapture` bound to this fixture."""
        capture = _FakeCapture(self._samples, on_frame=on_frame, loop=loop)
        self._created.append(capture)
        return capture


class _FakeCapture:
    """Lightweight fake of :class:`voxd.audio.AudioCapture`.

    Holds a snapshot of the samples to replay, plus the daemon-provided
    ``on_frame`` callback and event loop. ``start()`` kicks off a daemon
    thread that pushes frames to the loop via ``call_soon_threadsafe``;
    ``stop()`` joins the thread (with a generous timeout).
    """

    def __init__(
        self,
        samples: np.ndarray,
        *,
        on_frame: Callable[[np.ndarray], None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._samples = samples
        self._on_frame = on_frame
        self._loop = loop
        self._thread: threading.Thread | None = None
        self.started = False
        self.stopped = False
        self.replay_done = False

    def start(self) -> None:
        """Begin replaying ``samples`` on a background thread."""
        if self.started:
            return
        self.started = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Join the replay thread (blocking, with a short safety timeout)."""
        if not self.started or self.stopped:
            return
        self.stopped = True
        # Replay loop checks `stopped` between frames, so it will exit promptly.
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ---- internals ----

    def _run(self) -> None:
        """Drive frames into the event loop, then mark replay done."""
        try:
            data = self._samples
            n = data.size
            # Emit complete 480-sample frames. Trailing samples (< 480) drop
            # — same behaviour as the real AudioCapture which keeps them in
            # `_pending` until `stop()` flushes the resampler.
            for start in range(0, n - n % FRAME_SAMPLES, FRAME_SAMPLES):
                if self.stopped:
                    break
                frame = np.ascontiguousarray(
                    data[start : start + FRAME_SAMPLES], dtype=np.float32
                )
                try:
                    self._loop.call_soon_threadsafe(self._on_frame, frame)
                except RuntimeError:
                    # Loop closed underneath us — bail out.
                    break
                # Yield so the asyncio loop can drain. Without this the
                # threadsafe queue can grow huge and the loop never sees
                # individual frames between two recordings.
                time.sleep(0.0005)
        finally:
            self.replay_done = True


@pytest.fixture
def fake_audio_input(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeAudioInput]:
    """Replace :class:`voxd.audio.AudioCapture` with a WAV replayer.

    The patch targets both ``voxd.audio.AudioCapture`` and
    ``voxd.daemon.AudioCapture`` (the daemon binds the symbol at import time
    via ``from voxd.audio import AudioCapture``).
    """
    fake = FakeAudioInput()

    def _factory(
        device: str,
        on_frame: Callable[[np.ndarray], None],
        loop: asyncio.AbstractEventLoop,
    ) -> _FakeCapture:
        return fake._spawn(device, on_frame, loop)

    import voxd.audio
    import voxd.daemon

    monkeypatch.setattr(voxd.audio, "AudioCapture", _factory)
    monkeypatch.setattr(voxd.daemon, "AudioCapture", _factory)
    yield fake


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def wav_bytes_to_duration_s(wav_bytes: bytes) -> float:
    """Return the playback duration (seconds) of a PCM WAV byte string."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# Re-export the most useful names so tests can ``from conftest import ...``.
__all__ = [
    "FakeAudioInput",
    "MockOpenRouter",
    "make_test_config",
    "wav_bytes_to_duration_s",
]
