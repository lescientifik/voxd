"""Headless voxd daemon — asyncio loop, SIGUSR2 toggle, FIFO upload queue.

This module is the *only* place where every other voxd module is composed:

* :class:`voxd.audio.AudioCapture` produces 480-sample frames @ 16 kHz on the
  PortAudio callback thread, which hop to the event loop via
  ``call_soon_threadsafe``.
* :class:`voxd.vad.SmoothedVad` (wrapping :class:`voxd.vad.SileroV4`) classifies
  each frame and emits speech buffers (pre-roll + frame on onset, single frame
  during steady speech, hangover frames at the trailing edge).
* :func:`voxd.transcribe.transcribe` POSTs the encoded WAV to OpenRouter.
* :func:`voxd.inject.inject` synthesises ``wl-copy`` + ``wtype`` subprocesses.
* :class:`voxd.indicator.Indicator` flips a single ``notify-send`` bubble
  between Recording / Transcribing / Error / hidden.

Threading model
---------------

All daemon state lives on the asyncio loop. The PortAudio callback runs on a
separate thread but never touches state directly: it only calls
``loop.call_soon_threadsafe(_on_audio_frame, frame)``. The upload pipeline is
a single coroutine consuming an :class:`asyncio.Queue` — the FIFO ordering of
``wtype`` calls is a direct consequence of having exactly one worker.

A separate :class:`asyncio.Lock` (``_wtype_lock``) guards the
``inject``-time blocking subprocess calls. Since the worker is the only
producer of wtype calls, the lock is currently single-threaded; we keep it
because :func:`voxd.inject.inject` is a synchronous (blocking) call run via
``await loop.run_in_executor`` and we don't want any future contention if a
secondary path (e.g. shutdown drain) ever shares the same channel.

State model
-----------

The daemon is a two-state machine:

* ``Idle`` — no AudioCapture running, ``_speech_buffers`` empty.
* ``Recording`` — AudioCapture is running and pushing frames; speech-classified
  buffers accumulate in ``_speech_buffers``.

A toggle in ``Idle`` starts recording; a toggle in ``Recording`` stops the
capture, drains the speech buffer, encodes the WAV, and enqueues the upload.
"Upload in flight" is *independent* from this state: a new toggle pair can
start a fresh recording while the previous upload is still pending — the
new transcription simply lands behind it in the queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from voxd.audio import AudioCapture
from voxd.config import Config, InjectConfig
from voxd.indicator import Indicator
from voxd.inject import InjectError, inject
from voxd.transcribe import AuthInvalid, Permanent, Retryable, transcribe
from voxd.vad import SileroV4, SmoothedVad
from voxd.wav import samples_to_wav_bytes

_log = logging.getLogger(__name__)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TARGET_RATE = 16000
PAD_SAMPLES = TARGET_RATE * 5 // 4  # 1.25 s — minimum upload length


def _default_model_path() -> Path:
    """Return the bundled silero_vad_v4.onnx path."""
    return Path(__file__).parent / "resources" / "silero_vad_v4.onnx"


@dataclass
class _UploadJob:
    """One pending OpenRouter request: a WAV blob and nothing more.

    All configuration is read from the daemon at job submission time; the
    worker only needs the bytes.
    """

    wav_bytes: bytes


class Daemon:
    """voxd headless daemon — composes audio, VAD, transcribe, inject.

    Lifetime:

    1. Caller constructs the daemon with a :class:`voxd.config.Config`.
    2. ``await daemon.run()`` — installs the SIGUSR2 handler, spins up the
       upload worker, and blocks until :meth:`stop` is called.
    3. ``daemon.stop()`` — cancels the worker and unblocks ``run``. The
       method is safe to call from any context (it just sets an event).

    Tests can drive the toggle path directly via :meth:`_on_toggle` to avoid
    the flakiness of real signal delivery.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        model_path: Path | None = None,
        http_client: httpx.AsyncClient | None = None,
        install_signal_handler: bool = True,
        inject_fn: Callable[[str, InjectConfig], None] = inject,
        transcribe_fn: Callable[..., Awaitable[str]] = transcribe,
    ) -> None:
        """Create a daemon — does not start anything yet.

        Args:
            cfg: Loaded :class:`Config` (OpenRouter, audio, inject sections).
            openrouter_base_url: API root; overridable for tests / proxies.
            model_path: Path to the Silero v4 ONNX file. Defaults to the
                bundled resource.
            http_client: Optional pre-built :class:`httpx.AsyncClient`; if
                provided, the daemon does NOT close it on shutdown (the
                caller owns it). When ``None`` the daemon creates and owns
                its own client.
            install_signal_handler: When ``True`` (default), :meth:`run`
                installs an asyncio SIGUSR2 handler. Tests that drive
                :meth:`_on_toggle` directly should set this to ``False``.
            inject_fn: Override for the inject implementation (test seam).
            transcribe_fn: Override for the transcribe implementation
                (test seam).
        """
        self._cfg = cfg
        self._base_url = openrouter_base_url
        self._model_path = model_path if model_path is not None else _default_model_path()
        self._install_signal_handler = install_signal_handler

        self._inject_fn = inject_fn
        self._transcribe_fn = transcribe_fn

        self._owns_http_client = http_client is None
        # When the caller injects a client, reuse it as-is. When we own one,
        # build it lazily inside ``run`` so the timeout settings live close
        # to where the loop spins up.
        self._http_client: httpx.AsyncClient | None = http_client

        # State exclusively touched from the event loop.
        self._is_recording: bool = False
        self._capture: AudioCapture | None = None
        self._vad: SileroV4 | None = None
        self._smoothed: SmoothedVad | None = None
        self._speech_buffers: list[np.ndarray] = []
        self._toggle_lock = asyncio.Lock()
        self._wtype_lock = asyncio.Lock()

        # Communication primitives — created in ``run`` so they bind to the
        # right loop.
        self._upload_queue: asyncio.Queue[_UploadJob] | None = None
        self._stop_event: asyncio.Event | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._indicator = Indicator()

    # ------------------------------------------------------------------ #
    # Public lifecycle                                                   #
    # ------------------------------------------------------------------ #

    @property
    def is_recording(self) -> bool:
        """Whether the daemon is currently in the Recording state."""
        return self._is_recording

    async def run(self) -> None:
        """Drive the daemon until :meth:`stop` is called.

        Loads the Silero VAD model, installs the SIGUSR2 handler (unless
        opted out via ``install_signal_handler=False``), starts the upload
        worker, and blocks on a stop event. On shutdown, cancels the worker
        and (if owned) closes the HTTP client.
        """
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._upload_queue = asyncio.Queue()
        self._stop_event = asyncio.Event()

        # Load VAD lazily here so construction is cheap and tests that
        # don't reach the audio path don't pay the ONNX load cost.
        self._vad = SileroV4(self._model_path)
        self._smoothed = SmoothedVad(self._vad)

        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)

        if self._install_signal_handler:
            try:
                loop.add_signal_handler(signal.SIGUSR2, self._on_signal)
            except (NotImplementedError, RuntimeError) as exc:
                # Windows / loops on non-main threads — keep going without
                # a signal handler. Tests that need a deterministic toggle
                # call _on_toggle directly anyway.
                _log.warning("signal handler not installed: %s", exc)

        self._worker_task = asyncio.create_task(self._upload_worker())

        try:
            await self._stop_event.wait()
        finally:
            await self._shutdown()

    def stop(self) -> None:
        """Request shutdown — safe from any thread / context.

        Sets the stop event. ``run`` will observe it on its next loop tick
        and drive the cleanup path.
        """
        if self._stop_event is None or self._loop is None:
            return
        # Loop closed -> nothing to do.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._stop_event.set)

    # ------------------------------------------------------------------ #
    # Signal path                                                        #
    # ------------------------------------------------------------------ #

    def _on_signal(self) -> None:
        """SIGUSR2 sync handler — schedule the async toggle on the loop."""
        if self._loop is None:
            return
        # add_signal_handler runs us on the loop already, but we still need
        # to spawn a task because _on_toggle is a coroutine.
        self._loop.create_task(self._on_toggle())

    async def _on_toggle(self) -> None:
        """Toggle: start recording if Idle, stop & enqueue if Recording.

        Serialises through ``_toggle_lock`` so two rapid toggles can never
        interleave their state mutations. Each branch is responsible for
        flipping ``self._is_recording`` exactly once.
        """
        async with self._toggle_lock:
            if not self._is_recording:
                await self._start_recording()
            else:
                await self._stop_and_enqueue()

    # ------------------------------------------------------------------ #
    # Start / Stop recording                                             #
    # ------------------------------------------------------------------ #

    async def _start_recording(self) -> None:
        """Transition Idle → Recording: reset VAD, start AudioCapture."""
        assert self._loop is not None and self._vad is not None and self._smoothed is not None

        self._speech_buffers.clear()
        self._vad.reset()
        self._smoothed.reset()

        self._indicator.show_recording()

        capture = AudioCapture(
            device=self._cfg.audio.device,
            on_frame=self._on_audio_frame,
            loop=self._loop,
        )
        # ``start`` is synchronous but cheap; running it inline keeps the
        # state machine ordered (no chance of a second toggle observing
        # ``_is_recording`` mid-init).
        capture.start()
        self._capture = capture
        self._is_recording = True

    async def _stop_and_enqueue(self) -> None:
        """Transition Recording → Idle: stop, drain, encode, enqueue."""
        assert self._upload_queue is not None

        # Stop the capture FIRST so the audio thread stops feeding frames.
        # Any frames still in flight after this point get classified by the
        # VAD as usual; we accept them as part of the recording.
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        self._is_recording = False

        # Let any pending call_soon_threadsafe callbacks fire.
        await asyncio.sleep(0)

        self._indicator.show_transcribing()

        # Drain accumulated speech samples.
        buffers = self._speech_buffers
        self._speech_buffers = []
        if not buffers:
            self._indicator.hide()
            self._notify("No speech detected")
            return
        samples = np.concatenate(buffers).astype(np.float32, copy=False)

        if samples.size == 0:
            self._indicator.hide()
            self._notify("No speech detected")
            return

        # Pad to 1.25 s if under one second — mirrors the Rust handy daemon.
        if samples.size < TARGET_RATE:
            samples = np.pad(samples, (0, PAD_SAMPLES - samples.size))

        wav_bytes = samples_to_wav_bytes(samples)
        await self._upload_queue.put(_UploadJob(wav_bytes=wav_bytes))

    # ------------------------------------------------------------------ #
    # Audio callback (runs on the loop, posted from the audio thread)    #
    # ------------------------------------------------------------------ #

    def _on_audio_frame(self, frame: np.ndarray) -> None:
        """Receive a single 480-sample frame and feed it through the VAD.

        Runs on the event loop (the audio thread schedules us via
        ``call_soon_threadsafe``). If the VAD emits speech samples for this
        frame, they're appended to ``_speech_buffers``; otherwise the frame
        is silently dropped.

        Frames arriving after ``_stop_and_enqueue`` has flipped
        ``_is_recording`` to ``False`` are ignored — they belong to the
        previous recording's tail flush and the VAD state machine isn't
        in a useful condition anymore.
        """
        if not self._is_recording or self._smoothed is None:
            return
        try:
            out = self._smoothed.push(frame)
        except Exception:
            _log.exception("VAD push failed")
            return
        if out is not None and out.size > 0:
            self._speech_buffers.append(out)

    # ------------------------------------------------------------------ #
    # Upload worker (single consumer = FIFO order)                       #
    # ------------------------------------------------------------------ #

    async def _upload_worker(self) -> None:
        """Single-consumer FIFO worker for the upload queue.

        Each job goes through:

        1. ``transcribe(wav_bytes)`` — async POST + retry policy.
        2. ``inject(text, cfg.inject)`` — synchronous subprocess fan-out,
           held under ``_wtype_lock`` so it stays serialised even if a
           future change introduces a second producer.
        3. Indicator state update — hide on success, error message on
           failure.
        """
        assert self._upload_queue is not None

        while True:
            try:
                job = await self._upload_queue.get()
            except asyncio.CancelledError:
                return
            try:
                await self._process_job(job)
            except Exception:
                _log.exception("upload worker loop failed unexpectedly")
            finally:
                self._upload_queue.task_done()

    async def _process_job(self, job: _UploadJob) -> None:
        """Transcribe one job and inject the resulting text.

        Failure handling:

        * :class:`AuthInvalid` → ``show_error("Invalid API key")`` + notif.
        * :class:`Permanent` → ``show_error("Transcription failed: …")``.
        * :class:`Retryable` → ``show_error("Transcription failed (network)")``
          — retry budget already exhausted at this layer.
        * :class:`InjectError` → indicator stays in error state with the
          subprocess error message.
        """
        assert self._http_client is not None
        try:
            text = await self._transcribe_fn(
                job.wav_bytes,
                api_key=self._cfg.openrouter.api_key,
                model_id=self._cfg.openrouter.model,
                language=self._cfg.openrouter.language,
                prompt=self._cfg.openrouter.prompt,
                base_url=self._base_url,
                client=self._http_client,
            )
        except AuthInvalid as exc:
            _log.warning("transcription failed (auth): %s", exc)
            self._indicator.hide()
            self._indicator.show_error("Invalid API key")
            return
        except Permanent as exc:
            _log.warning("transcription failed (permanent): %s", exc)
            self._indicator.hide()
            self._indicator.show_error(f"Transcription failed: {exc}")
            return
        except Retryable as exc:
            _log.warning("transcription failed (retryable, budget exhausted): %s", exc)
            self._indicator.hide()
            self._indicator.show_error("Transcription failed (network)")
            return

        await self._inject_text(text)

    async def _inject_text(self, text: str) -> None:
        """Hide the indicator and inject ``text`` via the configured channels.

        The actual ``inject`` call is blocking subprocess work, so we run it
        on the default executor. Wrapping it under ``_wtype_lock`` keeps the
        contract single-writer even if a future caller adds a second path.
        """
        loop = asyncio.get_running_loop()
        async with self._wtype_lock:
            self._indicator.hide()
            try:
                await loop.run_in_executor(None, self._inject_fn, text, self._cfg.inject)
            except InjectError as exc:
                _log.warning("inject failed: %s", exc)
                self._indicator.show_error(f"Inject failed: {exc}")

    # ------------------------------------------------------------------ #
    # Misc                                                                #
    # ------------------------------------------------------------------ #

    def _notify(self, message: str) -> None:
        """Surface a transient info message via the indicator's error channel.

        ``notify-send`` is the only user-facing surface we have. Reusing
        ``show_error`` here keeps the path simple even though the message is
        informational ("No speech detected" isn't really an error).
        """
        self._indicator.show_error(message)

    # ------------------------------------------------------------------ #
    # Shutdown                                                            #
    # ------------------------------------------------------------------ #

    async def _shutdown(self) -> None:
        """Tear down the daemon: stop capture, cancel worker, close client."""
        # Stop any live capture.
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception:
                _log.exception("error stopping capture during shutdown")
            self._capture = None
        self._is_recording = False

        # Cancel the worker. We don't drain the queue: pending uploads are
        # dropped, matching the "no cancel in v1" decision (the user toggled
        # us off, they don't expect more wtype activity).
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

        # Remove the signal handler so a follow-up SIGUSR2 doesn't try to
        # interact with a dead loop. ``remove_signal_handler`` is idempotent
        # but raises if the loop is closed; guard accordingly.
        if self._install_signal_handler and self._loop is not None:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                self._loop.remove_signal_handler(signal.SIGUSR2)

        if self._owns_http_client and self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                _log.exception("error closing http client during shutdown")
            self._http_client = None

        self._indicator.hide()


__all__ = ["Daemon"]
