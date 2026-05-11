"""End-to-end user journey: ``voxd setup`` -> daemon cycle -> clean shutdown.

The plan spec for Step 12 calls for an E2E that spawns ``voxd`` in a real
subprocess. We deliberately diverge to an *in-process* journey instead
(option (a) in the briefing): the subprocess path defeats every fixture we
rely on (``fake_audio_input`` monkey-patches ``voxd.audio.AudioCapture`` at
import time, ``mock_openrouter_server`` plugs an ``httpx.MockTransport``,
``capture_subprocess`` rewires ``subprocess.run``). All three live in the
test process; they can't reach into a fresh ``python -m voxd`` subprocess.

The test still validates the same chain of guarantees the plan asked for:

1. ``voxd setup`` (monkey-patched API-key prompt) produces a config file
   *and* a sway snippet in the right XDG locations, with no daemon spawn.
2. The daemon is constructed from that *real on-disk config* (loaded via
   ``voxd.config.load``) — proving the round-trip from setup output to
   daemon input is wired correctly.
3. A toggle pair drives the full pipeline (audio capture -> VAD ->
   transcribe -> inject).
4. After ``daemon.stop()`` the worker is cancelled cleanly and the HTTP
   client closed (no warnings, no leaked tasks).

This is the closure test for v0.1.0: if it passes, every Step-1..Step-11
module composes correctly under realistic-ish conditions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

import pytest
from conftest import (  # type: ignore[import-not-found]
    FakeAudioInput,
    MockOpenRouter,
)

from voxd import config as cfg_module
from voxd import setup as voxd_setup  # avoid pytest's "setup" hook name
from voxd.daemon import Daemon

T = TypeVar("T")


def _run(coro: Awaitable[T]) -> T:
    """Drive ``coro`` on a fresh event loop (mirrors test_daemon_e2e._run)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _model_path() -> Path:
    """Return the bundled silero ONNX file path."""
    return Path(__file__).parent.parent / "src" / "voxd" / "resources" / "silero_vad_v4.onnx"


def test_full_user_journey(
    tmp_path: Path,
    mock_openrouter_server: MockOpenRouter,
    fake_audio_input: FakeAudioInput,
    capture_subprocess,  # noqa: ANN001 — pytest fixture (custom recorder)
    monkey_xdg_runtime: Path,  # noqa: ARG001 — env-side-effect only
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User runs setup, daemon starts, records, transcribes, shuts down cleanly.

    Walks through the four observable milestones of v0.1.0:

    * ``voxd setup`` writes a 0600 config.toml and a sway snippet in the
      expected XDG paths and prints restart instructions.
    * The daemon, constructed from ``config.load(config_path)``, picks up
      the API key written by setup.
    * One SIGUSR2-toggle pair (driven via ``_on_toggle`` to keep the test
      thread-safe) records fake audio, posts it to the mock OpenRouter
      server, and produces the expected ``wl-copy`` + ``wtype`` calls.
    * ``daemon.stop()`` cancels the upload worker without leaking tasks
      (the test would hang otherwise).
    """
    # --- 1. Pretend HOME and XDG_CONFIG_HOME both point at tmp_path ----
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake_home / ".config"))

    # --- 2. Run voxd setup with a mocked getpass prompt -----------------
    monkeypatch.setattr(voxd_setup.getpass, "getpass", lambda _prompt: "sk-or-v1-journey")
    rc = voxd_setup.main()
    assert rc == 0, "voxd setup should succeed"

    config_path = cfg_module.default_path()
    assert config_path.exists(), "config.toml should have been created by setup"
    sway_snippet = fake_home / ".config" / "sway" / "config.d" / "voxd.conf"
    assert sway_snippet.exists(), "sway snippet should have been created by setup"
    assert "pkill -USR2 voxd" in sway_snippet.read_text(encoding="utf-8")

    # --- 3. Reload the config from disk — this is the daemon's view -----
    cfg = cfg_module.load(config_path)
    assert cfg.openrouter.api_key == "sk-or-v1-journey", (
        "config loaded by daemon must reflect what setup wrote"
    )

    # --- 4. Drive a full record/transcribe/inject cycle -----------------
    mock_openrouter_server.queue_text("journey complete")
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

        # Start recording (in lieu of SIGUSR2 — see test_daemon_e2e for rationale).
        await daemon._on_toggle()
        await fake_audio_input.replay_finished()

        # Stop -> enqueue -> upload -> inject.
        await daemon._on_toggle()
        await capture_subprocess.wait_for_call("wtype")

        # Clean shutdown (analogue of SIGTERM in this in-process journey).
        daemon.stop()
        await runner

    _run(scenario())

    # --- 5. Assertions on observable side effects -----------------------
    wl_calls = capture_subprocess.calls("wl-copy")
    wt_calls = capture_subprocess.calls("wtype")
    assert wl_calls, "wl-copy should have been called"
    assert wt_calls, "wtype should have been called"
    assert wl_calls[-1].stdin == "journey complete"
    assert wt_calls[-1].argv[-1] == "journey complete"
    assert mock_openrouter_server.request_count == 1, (
        "exactly one transcription request should have been sent"
    )

    # --- 6. The request carries the API key we wrote in step 2 ----------
    req = mock_openrouter_server.requests[-1]
    assert req.headers["authorization"] == "Bearer sk-or-v1-journey"
