"""Tests for Silero v4 VAD wrapper and the SmoothedVad state machine.

These tests are scenario-driven: we feed real WAV fixtures (speech + silence)
through the pipeline and assert on observable behavior — never on internal
state or library specifics.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from voxd.vad import SileroV4, SmoothedVad

FRAME_SAMPLES = 480  # 30 ms @ 16 kHz
SAMPLE_RATE = 16000

FIXTURES = Path(__file__).parent / "fixtures"
SPEECH_WAV = FIXTURES / "speech_fr.wav"
SILENCE_WAV = FIXTURES / "silence.wav"
MODEL_PATH = Path(__file__).parents[1] / "src" / "voxd" / "resources" / "silero_vad_v4.onnx"


def _read_wav_int16_to_float32(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM int16 mono WAV, return float32 samples in [-1, 1] and the rate."""
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1, f"expected mono, got {w.getnchannels()} channels"
        assert w.getsampwidth() == 2, f"expected int16, got {w.getsampwidth()} bytes"
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    samples_i16 = np.frombuffer(raw, dtype=np.int16)
    samples_f32 = samples_i16.astype(np.float32) / 32768.0
    return samples_f32, rate


def _frames_480(samples: np.ndarray) -> list[np.ndarray]:
    """Slice a float32 mono signal into non-overlapping 480-sample frames."""
    n = (len(samples) // FRAME_SAMPLES) * FRAME_SAMPLES
    truncated = samples[:n]
    return [truncated[i : i + FRAME_SAMPLES] for i in range(0, n, FRAME_SAMPLES)]


@pytest.fixture(scope="session")
def silero_session() -> SileroV4:
    """Load the Silero ONNX once for the whole test session (load is ~300 ms)."""
    return SileroV4(MODEL_PATH)


@pytest.fixture(scope="session")
def speech_samples() -> np.ndarray:
    """French speech, float32, mono, 16 kHz."""
    samples, rate = _read_wav_int16_to_float32(SPEECH_WAV)
    assert rate == SAMPLE_RATE
    return samples


@pytest.fixture(scope="session")
def silence_samples() -> np.ndarray:
    """Pure silence, float32, mono, 16 kHz."""
    samples, rate = _read_wav_int16_to_float32(SILENCE_WAV)
    assert rate == SAMPLE_RATE
    return samples


# ---------------------------------------------------------------------------
# SileroV4 — raw ONNX wrapper
# ---------------------------------------------------------------------------


def test_silero_returns_high_prob_on_speech(
    silero_session: SileroV4, speech_samples: np.ndarray
) -> None:
    """At least one frame inside a 4 s speech WAV should yield prob > 0.5."""
    silero_session.reset()
    probs = [silero_session.push(f) for f in _frames_480(speech_samples)]
    assert max(probs) > 0.5, f"max prob was {max(probs):.3f}"


def test_silero_returns_low_prob_on_silence(silero_session: SileroV4) -> None:
    """A frame of pure zeros should yield prob < 0.1."""
    silero_session.reset()
    zeros = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    prob = silero_session.push(zeros)
    assert prob < 0.1, f"silence prob was {prob:.3f}"


def test_silero_state_is_carried_between_frames(
    silero_session: SileroV4, speech_samples: np.ndarray
) -> None:
    """Feeding the same frame twice produces a different prob (h,c changed)."""
    silero_session.reset()
    # Pick a frame inside the speech region (~1 s in).
    frame = speech_samples[16000 : 16000 + FRAME_SAMPLES].astype(np.float32)
    p1 = silero_session.push(frame)
    p2 = silero_session.push(frame)
    assert p1 != p2, "expected probs to differ across calls — LSTM state not carried"


def test_silero_reset_clears_state(
    silero_session: SileroV4, speech_samples: np.ndarray
) -> None:
    """After reset, the same frame yields the same prob as the very first call."""
    silero_session.reset()
    frame = speech_samples[16000 : 16000 + FRAME_SAMPLES].astype(np.float32)
    first = silero_session.push(frame)
    # Drift the state a bit.
    for _ in range(5):
        silero_session.push(frame)
    silero_session.reset()
    after_reset = silero_session.push(frame)
    assert first == pytest.approx(after_reset, abs=1e-6)


def test_silero_rejects_wrong_frame_size(silero_session: SileroV4) -> None:
    """Feeding a frame of the wrong length must raise — silent reshape would mask bugs."""
    silero_session.reset()
    with pytest.raises(ValueError):
        silero_session.push(np.zeros(479, dtype=np.float32))


# ---------------------------------------------------------------------------
# SmoothedVad — state machine
# ---------------------------------------------------------------------------


def test_smoothed_vad_with_real_speech_wav(
    silero_session: SileroV4, speech_samples: np.ndarray
) -> None:
    """Real speech in: bounded amount of audio out (pre-roll + speech + hangover)."""
    silero_session.reset()
    smoothed = SmoothedVad(silero_session)

    out_chunks: list[np.ndarray] = []
    for frame in _frames_480(speech_samples):
        out = smoothed.push(frame)
        if out is not None:
            out_chunks.append(out)

    assert out_chunks, "expected at least one speech chunk on a 4 s speech WAV"
    total_out = sum(len(c) for c in out_chunks)
    speech_duration_s = len(speech_samples) / SAMPLE_RATE

    # Lower bound: we should have captured most of the speech (allow generous slack
    # because the first/last 100 ms of espeak-ng output can be near-silence).
    lower = int(speech_duration_s * SAMPLE_RATE * 0.5)
    # Upper bound: cannot exceed input + pre-roll + hangover padding.
    upper = len(speech_samples) + (15 + 15) * FRAME_SAMPLES
    assert lower <= total_out <= upper, (
        f"total_out={total_out}, expected in [{lower}, {upper}], "
        f"input len={len(speech_samples)}"
    )


def test_smoothed_vad_with_pure_silence_emits_nothing(
    silero_session: SileroV4, silence_samples: np.ndarray
) -> None:
    """Pure silence in → no speech frames out."""
    silero_session.reset()
    smoothed = SmoothedVad(silero_session)
    for frame in _frames_480(silence_samples):
        assert smoothed.push(frame) is None


def test_smoothed_vad_short_blip_rejected(silero_session: SileroV4) -> None:
    """A 50 ms 440 Hz blip is too short to clear onset_frames=2 with a real-speech threshold.

    The blip is followed by silence, so the onset counter resets without emission.
    """
    silero_session.reset()
    smoothed = SmoothedVad(silero_session)

    # 50 ms @ 16 kHz = 800 samples ≈ 1.67 frames of 480. Use 1 full frame to be safe.
    t = np.arange(FRAME_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    blip = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    silence_frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)

    # 1 frame of blip, then plenty of silence to let onset reset.
    chunks = [blip] + [silence_frame] * 30
    out = [smoothed.push(f) for f in chunks]
    assert all(o is None for o in out), "did not expect any speech emission"


def test_smoothed_vad_reset_clears_state(
    silero_session: SileroV4, speech_samples: np.ndarray, silence_samples: np.ndarray
) -> None:
    """After speech then reset, pure silence yields nothing — no hangover leak."""
    silero_session.reset()
    smoothed = SmoothedVad(silero_session)

    for frame in _frames_480(speech_samples):
        smoothed.push(frame)

    # Force a reset: this should clear hangover_counter, in_speech, frame_buffer.
    smoothed.reset()
    silero_session.reset()

    for frame in _frames_480(silence_samples):
        assert smoothed.push(frame) is None, "leak: speech detected after reset"


def test_smoothed_vad_emits_preroll_on_speech_onset(
    silero_session: SileroV4, speech_samples: np.ndarray
) -> None:
    """The first emitted chunk after onset must be a multi-frame pre-roll.

    Pre-roll = frame_buffer at the time of transition (≤ prefill_frames + 1 frames).
    """
    silero_session.reset()
    smoothed = SmoothedVad(silero_session, prefill=15, hangover=15, onset=2, threshold=0.3)

    first_chunk: np.ndarray | None = None
    for frame in _frames_480(speech_samples):
        out = smoothed.push(frame)
        if out is not None:
            first_chunk = out
            break

    assert first_chunk is not None, "no speech ever emitted on the speech WAV"
    # First chunk is the pre-roll bundle: must be at least 2 frames long
    # (could be up to prefill+1 = 16 frames).
    assert len(first_chunk) >= 2 * FRAME_SAMPLES, (
        f"first emission was {len(first_chunk) // FRAME_SAMPLES} frames, "
        "expected multi-frame pre-roll bundle"
    )
    assert len(first_chunk) <= 16 * FRAME_SAMPLES
