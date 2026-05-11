"""Microphone capture with streaming resampling to 16 kHz @ 480-sample frames.

Pipeline (PortAudio callback thread)::

    sd.InputStream.callback(indata) -> downmix to mono -> soxr.ResampleStream
        -> accumulate -> emit 480-sample frames via loop.call_soon_threadsafe(on_frame, frame)

The frame size (480 = 30 ms @ 16 kHz) matches the Silero v4 contract, so the
event-loop side can feed frames straight into ``SmoothedVad`` without any
buffering of its own.

Threading rules:

* The PortAudio callback runs on a non-Python audio thread. It owns the
  resampler and the internal accumulator — *no* event-loop code touches them.
* The handoff to the event loop is exclusively through
  ``loop.call_soon_threadsafe(on_frame, frame)``: the callback is responsible
  for thread safety, the user-supplied ``on_frame`` runs on the loop.
* ``start`` / ``stop`` are called from the event-loop thread. ``stop`` flushes
  the resampler (``last=True``) so trailing samples don't get lost.

We deliberately do *not* import sounddevice at module load: it pulls PortAudio
shared libraries on import, and tests need to monkey-patch ``sd.InputStream``
before any capture instance is created. We import it on demand in ``start()``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import soxr

TARGET_RATE = 16000
FRAME_SAMPLES = 480  # 30 ms @ 16 kHz — Silero v4 contract
_BLOCKSIZE = 1024  # PortAudio block size; small enough for low latency
_RESAMPLER_QUALITY = "HQ"  # default; matches handy's libsoxr usage


class AudioCapture:
    """Streaming microphone capture that emits 480-sample @ 16 kHz frames.

    The on_frame callback is invoked on the event loop, never on the audio
    thread. Frames are emitted continuously while the stream is running; the
    caller decides what to do with each (typically: push into an asyncio.Queue
    or directly into the VAD).
    """

    def __init__(
        self,
        device: str,
        on_frame: Callable[[np.ndarray], None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Create a capture wired to ``loop``.

        Args:
            device: Sounddevice device identifier (name or index). Empty string
                or ``""`` means the default input device.
            on_frame: Callback invoked on the event loop for every 480-sample
                frame at 16 kHz. Receives a 1-D float32 numpy array.
            loop: The asyncio event loop on which ``on_frame`` will be scheduled.
        """
        self._device = device
        self._on_frame = on_frame
        self._loop = loop

        self._stream: Any = None
        # Capture-thread-only state. Created in ``start`` once we know the
        # device's native rate and channel count.
        self._resampler: soxr.ResampleStream | None = None
        self._native_rate: int = 0
        self._channels: int = 0
        # 1-D float32 buffer of post-resample samples that haven't filled a
        # 480-sample frame yet. Lives entirely in the audio thread.
        self._pending: np.ndarray = np.zeros(0, dtype=np.float32)
        # Guards against concurrent start/stop calls from the event loop side.
        self._lifecycle_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public lifecycle                                                   #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Open the input stream and begin emitting frames.

        Queries the device's native sample rate and channel count, instantiates
        the streaming resampler, and starts the PortAudio stream. Safe to call
        only once; calling ``start`` on an already-started capture is a no-op.
        """
        with self._lifecycle_lock:
            if self._stream is not None:
                return

            import sounddevice as sd  # local import: see module docstring

            device_arg: str | None = self._device if self._device else None
            info = sd.query_devices(device_arg, kind="input") if device_arg else sd.query_devices(
                kind="input"
            )
            # ``query_devices`` returns a dict-like for a single device.
            native_rate = int(round(float(info["default_samplerate"])))
            max_channels = int(info.get("max_input_channels", 1))
            # Open mono whenever the device allows it (most laptop mics are
            # mono); otherwise open 2 channels and let downmix collapse it.
            channels = 1 if max_channels <= 1 else min(max_channels, 2)

            self._native_rate = native_rate
            self._channels = channels
            self._pending = np.zeros(0, dtype=np.float32)
            self._resampler = (
                soxr.ResampleStream(
                    native_rate,
                    TARGET_RATE,
                    num_channels=1,
                    dtype="float32",
                    quality=_RESAMPLER_QUALITY,
                )
                if native_rate != TARGET_RATE
                else None
            )

            self._stream = sd.InputStream(
                samplerate=native_rate,
                channels=channels,
                dtype="float32",
                blocksize=_BLOCKSIZE,
                device=device_arg,
                callback=self._on_audio_callback,
            )
            self._stream.start()

    def stop(self) -> None:
        """Stop the input stream and flush any trailing samples.

        Pushes a final ``last=True`` chunk into the resampler so its internal
        buffer drains. Frames produced by this flush are dispatched normally
        via ``call_soon_threadsafe`` and will be visible to ``on_frame`` once
        the event loop drains its queue.
        """
        with self._lifecycle_lock:
            if self._stream is None:
                return
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
            self._flush_resampler()

    # ------------------------------------------------------------------ #
    # Audio thread internals                                             #
    # ------------------------------------------------------------------ #

    def _on_audio_callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002 — sounddevice contract
        time_info: object,  # noqa: ARG002
        status: object,  # noqa: ARG002
    ) -> None:
        """PortAudio callback — runs on the audio thread.

        Downmixes to mono (channel mean), resamples to 16 kHz if needed,
        accumulates, and emits 480-sample frames to the event loop.
        """
        # sounddevice always hands us a 2-D array of shape (frames, channels)
        # when channels was set explicitly. Downmix via channel mean.
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1).astype(np.float32, copy=False)
        elif indata.ndim == 2:
            mono = indata[:, 0].astype(np.float32, copy=False)
        else:
            mono = indata.astype(np.float32, copy=False)

        self._process_mono(mono, last=False)

    def _process_mono(self, mono: np.ndarray, *, last: bool) -> None:
        """Resample ``mono`` (or pass through), accumulate, emit frames.

        ``last=True`` flushes libsoxr's internal latency buffer; only call it
        from ``stop``.
        """
        if self._resampler is not None:
            resampled = self._resampler.resample_chunk(mono, last=last)
        else:
            resampled = mono

        if resampled.size:
            self._pending = (
                np.concatenate([self._pending, resampled])
                if self._pending.size
                else np.asarray(resampled, dtype=np.float32)
            )

        # Emit as many full 480-sample frames as we have.
        n = self._pending.size
        full_frames = n // FRAME_SAMPLES
        if full_frames == 0:
            return

        cut = full_frames * FRAME_SAMPLES
        chunk = self._pending[:cut]
        self._pending = self._pending[cut:].copy()  # avoid retaining big view
        for i in range(full_frames):
            frame = np.ascontiguousarray(
                chunk[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES],
                dtype=np.float32,
            )
            self._loop.call_soon_threadsafe(self._on_frame, frame)

    def _flush_resampler(self) -> None:
        """Drain libsoxr's internal latency by feeding an empty last chunk.

        Any partial frame remaining after the flush stays in ``_pending`` and
        is *not* emitted — the daemon pads recordings explicitly, so we don't
        want phantom zero-padded frames slipping through here.
        """
        if self._resampler is None:
            return
        self._process_mono(np.zeros(0, dtype=np.float32), last=True)
