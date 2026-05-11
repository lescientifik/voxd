"""Tests for WAV PCM int16 encoding.

These validate the wire contract used to POST audio to OpenRouter:
- 16-bit signed PCM, mono, 16 kHz by default
- Float32 input in [-1, 1] saturates at ±32767 (Rust `as` cast semantics — NOT
  the NumPy `astype(int16)` wrap-around that would silently corrupt audio).
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from voxd.wav import samples_to_wav_bytes


def _decode_wav_bytes(data: bytes) -> tuple[np.ndarray, int, int, int]:
    """Decode WAV bytes via stdlib ``wave``, returning (samples_i16, rate, channels, sampwidth)."""
    with wave.open(io.BytesIO(data), "rb") as r:
        channels = r.getnchannels()
        sampwidth = r.getsampwidth()
        rate = r.getframerate()
        n = r.getnframes()
        raw = r.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16) if raw else np.array([], dtype=np.int16)
    return samples, rate, channels, sampwidth


# ---------------------------------------------------------------------------
# Saturation contract — the central reason this function exists.
# ---------------------------------------------------------------------------


def test_samples_to_wav_bytes_clips_amplitude() -> None:
    """sample=1.5 → must saturate to +32767 (NOT wrap to -32767)."""
    data = samples_to_wav_bytes(np.array([1.5], dtype=np.float32))
    decoded, _, _, _ = _decode_wav_bytes(data)
    assert decoded.tolist() == [32767]


def test_samples_to_wav_bytes_clips_negative() -> None:
    """sample=-1.5 → must saturate to -32767 (matches Rust `i16::MAX` scaling)."""
    data = samples_to_wav_bytes(np.array([-1.5], dtype=np.float32))
    decoded, _, _, _ = _decode_wav_bytes(data)
    assert decoded.tolist() == [-32767]


# ---------------------------------------------------------------------------
# Roundtrip — does the bytestream we produce decode back to the same signal?
# ---------------------------------------------------------------------------


def test_roundtrip_decodes_to_close_values() -> None:
    """A float32 signal in [-1, 1] roundtrips through encode/decode within 1 LSB."""
    rng = np.random.default_rng(seed=42)
    original = rng.uniform(-0.9, 0.9, size=1024).astype(np.float32)

    data = samples_to_wav_bytes(original)
    decoded_i16, rate, channels, sampwidth = _decode_wav_bytes(data)

    assert rate == 16000
    assert channels == 1
    assert sampwidth == 2  # bytes → 16 bits
    assert len(decoded_i16) == len(original)

    # Recovered float32: should match within quantization noise (≤ 1 / 32767).
    recovered = decoded_i16.astype(np.float32) / 32767.0
    np.testing.assert_allclose(recovered, original, atol=1.0 / 32767.0)


# ---------------------------------------------------------------------------
# Header / format sanity.
# ---------------------------------------------------------------------------


def test_header_riff_wave() -> None:
    """A valid WAV file starts with 'RIFF' at offset 0 and 'WAVE' at offset 8."""
    data = samples_to_wav_bytes(np.array([0.0], dtype=np.float32))
    assert data[0:4] == b"RIFF"
    assert data[8:12] == b"WAVE"


def test_mono_16k_16bit() -> None:
    """Default spec: mono, 16 kHz, 16-bit signed PCM."""
    data = samples_to_wav_bytes(np.zeros(100, dtype=np.float32))
    _, rate, channels, sampwidth = _decode_wav_bytes(data)
    assert channels == 1
    assert rate == 16000
    assert sampwidth == 2


def test_empty_input() -> None:
    """Encoding an empty array must still produce a parseable WAV with 0 frames."""
    data = samples_to_wav_bytes(np.array([], dtype=np.float32))
    decoded, rate, channels, sampwidth = _decode_wav_bytes(data)
    assert len(decoded) == 0
    assert rate == 16000
    assert channels == 1
    assert sampwidth == 2


# ---------------------------------------------------------------------------
# Bonus: explicit sample_rate override behaves.
# ---------------------------------------------------------------------------


def test_custom_sample_rate_is_honored() -> None:
    """Caller can override the default 16 kHz (used by tests at native capture rates)."""
    data = samples_to_wav_bytes(np.zeros(10, dtype=np.float32), sample_rate=44100)
    _, rate, _, _ = _decode_wav_bytes(data)
    assert rate == 44100


@pytest.mark.parametrize("amplitude", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_boundary_values_roundtrip(amplitude: float) -> None:
    """Boundary samples encode to ``trunc(amplitude * 32767)`` (Rust ``as i16`` semantics)."""
    data = samples_to_wav_bytes(np.array([amplitude], dtype=np.float32))
    decoded, _, _, _ = _decode_wav_bytes(data)
    # Truncation toward zero, matching Rust `(sample * i16::MAX as f32) as i16`
    # and NumPy `astype(int16)`. round() would be wrong here.
    expected = int(np.trunc(amplitude * 32767))
    assert decoded.tolist() == [expected]
