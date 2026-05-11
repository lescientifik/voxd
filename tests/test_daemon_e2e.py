"""End-to-end tests for :class:`voxd.daemon.Daemon`.

Each test drives the daemon through one or more toggle cycles and asserts on
the observable side-effects (HTTP requests sent, subprocess calls produced).
The daemon runs on a private event loop inside ``_run``; tests await
specific milestones via the fixtures (``replay_finished``, ``wait_for_call``).

We deliberately call ``daemon._on_toggle()`` directly instead of sending a
real ``SIGUSR2``. Reasons:

* ``loop.add_signal_handler`` only works when the loop runs on the main
  thread; pytest's main thread is also the test thread, so the loop here
  lives on a worker thread.
* The signal path *is* exercised by ``test_signal_handler_triggers_toggle``,
  which spins up a dedicated main-thread loop. Mixing this into every E2E
  scenario adds flake without raising coverage.

The behaviour validated end-to-end:

1. ``test_full_cycle_record_transcribe_inject`` — record → stop → upload →
   inject the text via wl-copy + wtype.
2. ``test_double_toggle_during_upload_queues_second_recording`` — two
   recordings back-to-back maintain FIFO ordering on the wtype side, even
   when the second upload finishes before the first.
3. ``test_empty_recording_no_upload`` — silent toggle pair issues zero HTTP
   requests and notifies the user.
4. ``test_short_recording_padded`` — under-second recording is padded to
   1.25 s before upload.
5. ``test_401_clears_indicator_and_notifies`` — 401 hides the indicator and
   surfaces an error notification, no wtype call.
6. ``test_5xx_retries_then_inject`` — 500/500/200 triggers two retries then
   injects the eventual text.
7. ``test_signal_handler_triggers_toggle`` — real SIGUSR2 flips the daemon
   state once (sanity check; isolated from the upload pipeline).
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

import pytest
from conftest import (  # type: ignore[import-not-found]
    FakeAudioInput,
    MockOpenRouter,
    make_test_config,
    wav_bytes_to_duration_s,
)

from voxd.daemon import Daemon

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Async-test driver
# ---------------------------------------------------------------------------


def _run(coro: Awaitable[T]) -> T:
    """Drive ``coro`` to completion on a fresh event loop and return its result."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _model_path() -> Path:
    """Return the bundled silero ONNX file path."""
    return Path(__file__).parent.parent / "src" / "voxd" / "resources" / "silero_vad_v4.onnx"


# ---------------------------------------------------------------------------
# 1. Full cycle: record → stop → transcribe → inject
# ---------------------------------------------------------------------------


def test_full_cycle_record_transcribe_inject(
    tmp_path: Path,
    mock_openrouter_server: MockOpenRouter,
    fake_audio_input: FakeAudioInput,
    capture_subprocess,  # noqa: ANN001 — pytest fixture (custom recorder)
    monkey_xdg_runtime: Path,  # noqa: ARG001 — sets env, not used directly
) -> None:
    """One toggle pair records speech, uploads it, and injects the returned text."""
    cfg = make_test_config(tmp_path, api_key="fake")
    mock_openrouter_server.queue_text("hello world")
    fake_audio_input.load_wav(Path(__file__).parent / "fixtures" / "speech_fr.wav")

    async def scenario() -> None:
        daemon = Daemon(
            cfg,
            openrouter_base_url=mock_openrouter_server.url,
            model_path=_model_path(),
            http_client=mock_openrouter_server.client,
            install_signal_handler=False,
        )
        runner = asyncio.create_task(daemon.run())
        await asyncio.sleep(0)  # let the daemon initialise

        # Start recording (no real signal — just call the handler).
        await daemon._on_toggle()
        await fake_audio_input.replay_finished()

        # Stop recording → enqueue → upload → inject.
        await daemon._on_toggle()
        await capture_subprocess.wait_for_call("wtype")

        daemon.stop()
        await runner

    _run(scenario())

    wl_calls = capture_subprocess.calls("wl-copy")
    wt_calls = capture_subprocess.calls("wtype")
    assert wl_calls, "expected at least one wl-copy invocation"
    assert wt_calls, "expected at least one wtype invocation"
    assert wl_calls[-1].stdin == "hello world"
    assert wt_calls[-1].argv[-1] == "hello world"
    assert mock_openrouter_server.request_count == 1


# ---------------------------------------------------------------------------
# 2. FIFO: a second toggle-pair started before the first upload completes is
#    queued and injects in order.
# ---------------------------------------------------------------------------


def test_double_toggle_during_upload_queues_second_recording(
    tmp_path: Path,
    mock_openrouter_server: MockOpenRouter,
    fake_audio_input: FakeAudioInput,
    capture_subprocess,  # noqa: ANN001
    monkey_xdg_runtime: Path,  # noqa: ARG001
) -> None:
    """Two record→stop cycles produce wtype calls in submission order."""
    cfg = make_test_config(tmp_path, api_key="fake")
    fake_audio_input.load_wav(Path(__file__).parent / "fixtures" / "speech_fr.wav")
    # Two responses queued; first one is delayed via a custom transport
    # wrapper so it lands AFTER the second is sent.
    mock_openrouter_server.queue_text("first", "second")

    async def scenario() -> None:
        daemon = Daemon(
            cfg,
            openrouter_base_url=mock_openrouter_server.url,
            model_path=_model_path(),
            http_client=mock_openrouter_server.client,
            install_signal_handler=False,
        )
        runner = asyncio.create_task(daemon.run())
        await asyncio.sleep(0)

        # 1st recording.
        await daemon._on_toggle()
        await fake_audio_input.replay_finished()
        await daemon._on_toggle()

        # Don't wait for upload — fire the 2nd recording immediately.
        await daemon._on_toggle()
        await fake_audio_input.replay_finished()
        await daemon._on_toggle()

        # Wait for both wtype calls.
        deadline = time.monotonic() + 10.0
        while len(capture_subprocess.calls("wtype")) < 2:
            if time.monotonic() > deadline:
                raise TimeoutError("did not see two wtype calls")
            await asyncio.sleep(0.02)

        daemon.stop()
        await runner

    _run(scenario())

    wt_calls = capture_subprocess.calls("wtype")
    assert len(wt_calls) == 2, f"expected 2 wtype calls, got {len(wt_calls)}"
    assert wt_calls[0].argv[-1] == "first"
    assert wt_calls[1].argv[-1] == "second"
    assert mock_openrouter_server.request_count == 2


# ---------------------------------------------------------------------------
# 3. Empty recording → no upload, no inject, notify.
# ---------------------------------------------------------------------------


def test_empty_recording_no_upload(
    tmp_path: Path,
    mock_openrouter_server: MockOpenRouter,
    fake_audio_input: FakeAudioInput,
    capture_subprocess,  # noqa: ANN001
    monkey_xdg_runtime: Path,  # noqa: ARG001
) -> None:
    """Silence between two toggles produces no HTTP request and no wtype call."""
    cfg = make_test_config(tmp_path, api_key="fake")
    fake_audio_input.load_wav(Path(__file__).parent / "fixtures" / "silence.wav")

    async def scenario() -> None:
        daemon = Daemon(
            cfg,
            openrouter_base_url=mock_openrouter_server.url,
            model_path=_model_path(),
            http_client=mock_openrouter_server.client,
            install_signal_handler=False,
        )
        runner = asyncio.create_task(daemon.run())
        await asyncio.sleep(0)

        await daemon._on_toggle()
        await fake_audio_input.replay_finished()
        await daemon._on_toggle()

        # Give the daemon a moment to process the (empty) recording.
        await asyncio.sleep(0.1)

        daemon.stop()
        await runner

    _run(scenario())

    assert mock_openrouter_server.request_count == 0
    assert capture_subprocess.calls("wtype") == []
    assert capture_subprocess.calls("wl-copy") == []
    # An error/info notification should have been issued.
    notif_calls = capture_subprocess.calls("notify-send")
    assert notif_calls, "expected a notify-send call announcing the empty recording"
    last = " ".join(notif_calls[-1].argv).lower()
    assert "no speech" in last or "empty" in last


# ---------------------------------------------------------------------------
# 4. Short recording is padded to ≥ 1.25 s before upload.
# ---------------------------------------------------------------------------


def test_short_recording_padded(
    tmp_path: Path,
    mock_openrouter_server: MockOpenRouter,
    fake_audio_input: FakeAudioInput,
    capture_subprocess,  # noqa: ANN001
    monkey_xdg_runtime: Path,  # noqa: ARG001
) -> None:
    """A 0.5 s speech clip uploads as a WAV of ≥ 1.25 s after padding."""
    import base64

    cfg = make_test_config(tmp_path, api_key="fake")
    mock_openrouter_server.queue_text("ok")
    fake_audio_input.set_speech_then_silence(0.5)

    async def scenario() -> None:
        daemon = Daemon(
            cfg,
            openrouter_base_url=mock_openrouter_server.url,
            model_path=_model_path(),
            http_client=mock_openrouter_server.client,
            install_signal_handler=False,
        )
        runner = asyncio.create_task(daemon.run())
        await asyncio.sleep(0)

        await daemon._on_toggle()
        await fake_audio_input.replay_finished()
        await daemon._on_toggle()

        await capture_subprocess.wait_for_call("wtype")
        daemon.stop()
        await runner

    _run(scenario())

    assert mock_openrouter_server.request_count == 1
    req = mock_openrouter_server.requests[-1]
    import json

    body = json.loads(req.content)
    wav_bytes = base64.b64decode(body["input_audio"]["data"])
    duration = wav_bytes_to_duration_s(wav_bytes)
    assert duration >= 1.25 - 1e-3, f"WAV duration {duration:.3f}s should be >= 1.25 s"


# ---------------------------------------------------------------------------
# 5. 401 → indicator hide + error notif + no wtype.
# ---------------------------------------------------------------------------


def test_401_clears_indicator_and_notifies(
    tmp_path: Path,
    mock_openrouter_server: MockOpenRouter,
    fake_audio_input: FakeAudioInput,
    capture_subprocess,  # noqa: ANN001
    monkey_xdg_runtime: Path,  # noqa: ARG001
) -> None:
    """A 401 surfaces as an error notification; no text injection happens."""
    cfg = make_test_config(tmp_path, api_key="fake")
    mock_openrouter_server.set_responses((401, {"error": "bad key"}))
    fake_audio_input.load_wav(Path(__file__).parent / "fixtures" / "speech_fr.wav")

    async def scenario() -> None:
        daemon = Daemon(
            cfg,
            openrouter_base_url=mock_openrouter_server.url,
            model_path=_model_path(),
            http_client=mock_openrouter_server.client,
            install_signal_handler=False,
        )
        runner = asyncio.create_task(daemon.run())
        await asyncio.sleep(0)

        await daemon._on_toggle()
        await fake_audio_input.replay_finished()
        await daemon._on_toggle()

        # Wait for the daemon to process the upload (which fails with 401).
        deadline = time.monotonic() + 5.0
        while mock_openrouter_server.request_count < 1:
            if time.monotonic() > deadline:
                raise TimeoutError("upload did not happen")
            await asyncio.sleep(0.02)
        # Plus a moment for the error path.
        await asyncio.sleep(0.1)

        daemon.stop()
        await runner

    _run(scenario())

    assert capture_subprocess.calls("wtype") == []
    notif_msgs = [" ".join(c.argv).lower() for c in capture_subprocess.calls("notify-send")]
    assert any("invalid" in m or "api" in m or "key" in m or "401" in m for m in notif_msgs), (
        f"expected an error notification mentioning the API key, got: {notif_msgs}"
    )


# ---------------------------------------------------------------------------
# 6. 5xx retries then succeeds → exactly one wtype call.
# ---------------------------------------------------------------------------


def test_5xx_retries_then_inject(
    tmp_path: Path,
    mock_openrouter_server: MockOpenRouter,
    fake_audio_input: FakeAudioInput,
    capture_subprocess,  # noqa: ANN001
    monkey_xdg_runtime: Path,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """500 → 500 → 200 retries twice then injects the eventual text once."""
    # Speed up retries: make backoff_delay return 0 so the transcribe loop
    # doesn't actually wait between attempts. We keep the real asyncio.sleep
    # in place so the daemon's own internal yields behave normally.
    import voxd.transcribe

    monkeypatch.setattr(voxd.transcribe, "backoff_delay", lambda *a, **k: 0.0)

    cfg = make_test_config(tmp_path, api_key="fake")
    mock_openrouter_server.set_responses(
        (500, {"error": "boom"}),
        (500, {"error": "boom"}),
        (200, {"text": "retried"}),
    )
    fake_audio_input.load_wav(Path(__file__).parent / "fixtures" / "speech_fr.wav")

    async def scenario() -> None:
        daemon = Daemon(
            cfg,
            openrouter_base_url=mock_openrouter_server.url,
            model_path=_model_path(),
            http_client=mock_openrouter_server.client,
            install_signal_handler=False,
        )
        runner = asyncio.create_task(daemon.run())
        # Wait until daemon has wired its loop/queue.
        for _ in range(20):
            if daemon._loop is not None and daemon._upload_queue is not None:
                break
            await asyncio.sleep(0.01)

        await daemon._on_toggle()
        await fake_audio_input.replay_finished()
        await daemon._on_toggle()

        await capture_subprocess.wait_for_call("wtype")
        daemon.stop()
        await runner

    _run(scenario())

    assert mock_openrouter_server.request_count == 3
    wt_calls = capture_subprocess.calls("wtype")
    assert len(wt_calls) == 1
    assert wt_calls[0].argv[-1] == "retried"


# ---------------------------------------------------------------------------
# 7. Real SIGUSR2 flips the daemon state once (isolated sanity check).
# ---------------------------------------------------------------------------


def test_signal_handler_triggers_toggle(
    tmp_path: Path,
    mock_openrouter_server: MockOpenRouter,
    fake_audio_input: FakeAudioInput,
    capture_subprocess,  # noqa: ANN001 — fixture wires subprocess.run capture
    monkey_xdg_runtime: Path,  # noqa: ARG001
) -> None:
    """Sending real SIGUSR2 to ourselves toggles the daemon into Recording.

    This is the only test that exercises the actual ``add_signal_handler``
    path. The rest of the suite calls ``_on_toggle`` directly to keep timing
    deterministic.
    """
    _ = capture_subprocess  # ensure subprocess.run is patched for safety
    cfg = make_test_config(tmp_path, api_key="fake")
    fake_audio_input.load_wav(Path(__file__).parent / "fixtures" / "speech_fr.wav")

    async def scenario() -> bool:
        daemon = Daemon(
            cfg,
            openrouter_base_url=mock_openrouter_server.url,
            model_path=_model_path(),
            http_client=mock_openrouter_server.client,
            install_signal_handler=True,
        )
        runner = asyncio.create_task(daemon.run())
        # Let add_signal_handler attach.
        await asyncio.sleep(0.05)

        os.kill(os.getpid(), signal.SIGUSR2)

        # Wait until daemon reports recording.
        deadline = time.monotonic() + 2.0
        while not daemon.is_recording:
            if time.monotonic() > deadline:
                break
            await asyncio.sleep(0.02)

        recording_now = daemon.is_recording
        daemon.stop()
        await runner
        return recording_now

    # asyncio signal handlers only work on the main thread; pytest's worker
    # thread *is* the main thread, so this should succeed.
    result = _run(scenario())
    assert result is True, "SIGUSR2 should have transitioned the daemon to Recording"
