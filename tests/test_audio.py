"""Tests for the audio capture pipeline.

Four scenarios:

1. ``soxr.ResampleStream`` resamples a 48 kHz sinusoid to 16 kHz with
   frequency content preserved (FFT peak unchanged within ±5 Hz).
2. Chunking the same signal differently before resampling yields a
   numerically identical output (chunk boundaries do not ring).
3. ``AudioCapture`` driven by a fake ``sounddevice.InputStream`` that replays
   a 44.1 kHz WAV emits 480-sample frames at 16 kHz, in the right quantity,
   with the right content.
4. A stereo input with the right channel silent is downmixed to mono = L/2,
   matching the channel-mean used everywhere else.
"""

from __future__ import annotations

import asyncio
import sys
import types
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _install_fake_sounddevice() -> types.ModuleType:
    """Insert a stub ``sounddevice`` module before voxd.audio imports it.

    The real ``sounddevice`` package fails to import on systems without the
    PortAudio shared library — CI machines, containers, etc. Our tests never
    hit real PortAudio (we monkey-patch ``InputStream`` and ``query_devices``
    inside each test), so a stub module is enough to let the import succeed.
    """
    fake = types.ModuleType("sounddevice")

    def _raise(*args: object, **kwargs: object) -> Any:  # noqa: ARG001
        raise RuntimeError("sounddevice stub: monkeypatch the real symbol in your test")

    fake.InputStream = _raise  # type: ignore[attr-defined]
    fake.query_devices = _raise  # type: ignore[attr-defined]
    sys.modules.setdefault("sounddevice", fake)
    return sys.modules["sounddevice"]


_install_fake_sounddevice()

from voxd.audio import AudioCapture  # noqa: E402 — must follow the stub install

FIXTURES = Path(__file__).parent / "fixtures"
SPEECH_44K_WAV = FIXTURES / "speech_44k.wav"

TARGET_RATE = 16000
FRAME_SAMPLES = 480  # 30 ms @ 16 kHz


def _read_int16_wav(path: Path) -> tuple[np.ndarray, int, int]:
    """Read a PCM int16 WAV, return float32 samples in [-1, 1], rate, channels."""
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2, "fixture must be int16 PCM"
        rate = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    samples = pcm.astype(np.float32) / 32768.0
    return samples, rate, channels


def _make_sine(freq_hz: float, duration_s: float, rate: int) -> np.ndarray:
    """Return a float32 sine wave at ``freq_hz`` with amplitude 0.5."""
    n = int(rate * duration_s)
    t = np.arange(n, dtype=np.float32) / float(rate)
    return (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _fft_peak_hz(signal: np.ndarray, rate: int) -> float:
    """Return the frequency of the largest FFT magnitude (excluding DC)."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / rate)
    spectrum[0] = 0.0  # ignore DC
    return float(freqs[int(np.argmax(spectrum))])


# ---------------------------------------------------------------------------
# 1. ResampleStream 48k -> 16k preserves spectral content
# ---------------------------------------------------------------------------


def test_resample_streaming_48k_to_16k_preserves_signal() -> None:
    """A 440 Hz tone @ 48 kHz fed in 1024-sample chunks resamples to a 440 Hz
    tone @ 16 kHz (FFT peak within ±5 Hz)."""
    import soxr

    in_rate = 48000
    out_rate = TARGET_RATE
    duration_s = 1.0
    sine = _make_sine(440.0, duration_s, in_rate)

    stream = soxr.ResampleStream(in_rate, out_rate, num_channels=1, dtype="float32")
    pieces: list[np.ndarray] = []
    chunk = 1024
    for start in range(0, sine.size, chunk):
        out = stream.resample_chunk(sine[start : start + chunk])
        if out.size:
            pieces.append(out)
    tail = stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
    if tail.size:
        pieces.append(tail)

    resampled = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    # Ratio is 16/48 = 1/3, so expect ~16000 samples for 1 s of input.
    assert abs(resampled.size - out_rate) <= 64

    peak = _fft_peak_hz(resampled, out_rate)
    assert abs(peak - 440.0) <= 5.0, f"expected 440 Hz peak, got {peak}"


# ---------------------------------------------------------------------------
# 2. Chunk boundaries don't add ringing — output is independent of chunking
# ---------------------------------------------------------------------------


def _resample_in_chunks(sine: np.ndarray, in_rate: int, out_rate: int, chunk: int) -> np.ndarray:
    import soxr

    stream = soxr.ResampleStream(in_rate, out_rate, num_channels=1, dtype="float32")
    pieces: list[np.ndarray] = []
    for start in range(0, sine.size, chunk):
        out = stream.resample_chunk(sine[start : start + chunk])
        if out.size:
            pieces.append(out)
    tail = stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
    if tail.size:
        pieces.append(tail)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def test_resample_chunk_boundaries_dont_introduce_ringing() -> None:
    """Resampling the same sine in 100-sample versus 10000-sample chunks yields
    numerically identical output."""
    in_rate = 48000
    out_rate = TARGET_RATE
    sine = _make_sine(440.0, 1.0, in_rate)

    out_small = _resample_in_chunks(sine, in_rate, out_rate, chunk=100)
    out_large = _resample_in_chunks(sine, in_rate, out_rate, chunk=10000)

    # libsoxr is deterministic for the same signal+config — expect close to
    # bit-equality. We allow a tiny numerical tolerance to keep the test robust
    # across libsoxr versions.
    assert out_small.shape == out_large.shape
    np.testing.assert_allclose(out_small, out_large, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# Fake sounddevice.InputStream for AudioCapture tests
# ---------------------------------------------------------------------------


class _FakeInputStream:
    """Replay a fixed numpy array as if it were live mic input.

    Drives the registered callback synchronously on ``start()`` so that tests
    don't depend on real PortAudio threading. Mirrors the subset of the
    ``sd.InputStream`` API that ``AudioCapture`` calls.
    """

    def __init__(
        self,
        data: np.ndarray,
        chunk: int,
        rate: int,
        channels: int,
        callback: Callable[..., None],
    ) -> None:
        self._data = data
        self._chunk = chunk
        self._rate = rate
        self._channels = channels
        self._callback = callback
        self.started = False
        self.stopped = False

    def start(self) -> None:
        """Synchronously push the whole fixture through the callback."""
        self.started = True
        # Reshape so each callback receives shape (frames, channels), matching
        # sounddevice's contract.
        data = self._data.reshape(-1, 1) if self._channels == 1 else self._data
        n = data.shape[0]
        for start in range(0, n, self._chunk):
            block = data[start : start + self._chunk]
            # sounddevice passes (indata, frames, time_info, status)
            self._callback(block, block.shape[0], None, None)

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 3. End-to-end capture: fake mic at 44.1 kHz mono -> 480-sample @ 16 kHz frames
# ---------------------------------------------------------------------------


def test_capture_pipeline_with_wav_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capture pipeline turns a 44.1 kHz WAV into 480-sample frames @ 16 kHz."""
    samples, rate, channels = _read_int16_wav(SPEECH_44K_WAV)
    assert rate == 44100 and channels == 1, "fixture must be 44.1 kHz mono"

    # Capture frames into a list. Use a real running loop so we can verify the
    # call_soon_threadsafe hand-off works end-to-end.
    loop = asyncio.new_event_loop()
    try:
        collected: list[np.ndarray] = []

        def on_frame(frame: np.ndarray) -> None:
            collected.append(frame)

        fake_stream_box: dict[str, _FakeInputStream] = {}

        def fake_input_stream_ctor(**kwargs: object) -> _FakeInputStream:
            # AudioCapture should pick the device's native rate. We patch
            # query_devices below, so we know it'll be 44100.
            assert kwargs.get("samplerate") == 44100
            cb = kwargs["callback"]
            assert callable(cb)
            stream = _FakeInputStream(
                data=samples,
                chunk=2048,
                rate=int(kwargs["samplerate"]),  # type: ignore[arg-type]
                channels=int(kwargs["channels"]),  # type: ignore[arg-type]
                callback=cb,
            )
            fake_stream_box["stream"] = stream
            return stream

        import sounddevice as sd

        monkeypatch.setattr(sd, "InputStream", fake_input_stream_ctor)
        monkeypatch.setattr(
            sd,
            "query_devices",
            lambda device=None, kind=None: {  # noqa: ARG005
                "default_samplerate": 44100.0,
                "max_input_channels": 1,
                "name": "fake-mic",
            },
        )

        capture = AudioCapture(device="", on_frame=on_frame, loop=loop)
        capture.start()
        # Fake stream pushed all samples synchronously. Now flush the resampler
        # tail and let queued frames drain on the event loop.
        capture.stop()

        async def _drain() -> None:
            # Yield several times so call_soon_threadsafe callbacks fire.
            for _ in range(5):
                await asyncio.sleep(0)

        loop.run_until_complete(_drain())

        # 4.30 s @ 16 kHz = 68800 samples ≈ 143 frames of 480.
        expected_frames = samples.size * TARGET_RATE // rate // FRAME_SAMPLES
        assert abs(len(collected) - expected_frames) <= 2, (
            f"got {len(collected)} frames, expected ~{expected_frames}"
        )
        # Every emitted frame is exactly 480 float32 samples.
        for frame in collected:
            assert frame.shape == (FRAME_SAMPLES,)
            assert frame.dtype == np.float32
        # Speech should produce non-trivial energy somewhere.
        total = np.concatenate(collected)
        assert np.abs(total).max() > 0.01
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# 4. Stereo -> mono downmix is the channel mean
# ---------------------------------------------------------------------------


def test_downmix_stereo_to_mono(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stereo input with L = sine, R = 0 downmixes to mono = sine/2 (channel mean)."""
    in_rate = TARGET_RATE  # no resampling — isolate the downmix
    sine = _make_sine(440.0, 0.5, in_rate)  # amplitude 0.5
    left = sine
    right = np.zeros_like(sine)
    stereo = np.stack([left, right], axis=1)  # shape (N, 2)

    loop = asyncio.new_event_loop()
    try:
        collected: list[np.ndarray] = []

        def on_frame(frame: np.ndarray) -> None:
            collected.append(frame)

        def fake_input_stream_ctor(**kwargs: object) -> _FakeInputStream:
            cb = kwargs["callback"]
            assert callable(cb)
            return _FakeInputStream(
                data=stereo,
                chunk=1024,
                rate=int(kwargs["samplerate"]),  # type: ignore[arg-type]
                channels=int(kwargs["channels"]),  # type: ignore[arg-type]
                callback=cb,
            )

        import sounddevice as sd

        monkeypatch.setattr(sd, "InputStream", fake_input_stream_ctor)
        monkeypatch.setattr(
            sd,
            "query_devices",
            lambda device=None, kind=None: {  # noqa: ARG005
                "default_samplerate": float(in_rate),
                "max_input_channels": 2,
                "name": "fake-stereo-mic",
            },
        )

        capture = AudioCapture(device="", on_frame=on_frame, loop=loop)
        capture.start()
        capture.stop()

        async def _drain() -> None:
            for _ in range(5):
                await asyncio.sleep(0)

        loop.run_until_complete(_drain())

        assert collected, "expected at least one frame"
        merged = np.concatenate(collected)
        # We only check the bulk of the signal; the resampler is bypassed here
        # so we expect mono == L/2 exactly. Allow a tiny tolerance for f32.
        expected = (sine / 2.0)[: merged.size]
        np.testing.assert_allclose(merged, expected, atol=1e-6, rtol=0.0)
    finally:
        loop.close()
