"""WAV PCM int16 encoder — port of ``audio_toolkit/audio/utils.rs::samples_to_wav_bytes``.

The function exists for one reason: OpenRouter's ``/audio/transcriptions``
endpoint takes WAV-encoded audio in the request body. We need a deterministic,
bit-exact encoder that matches what the Rust ``handy`` codebase produces, so
that wire-level behaviour is identical across the two implementations.

Key subtlety: Rust's ``as i16`` cast saturates on overflow, while NumPy's
``ndarray.astype(np.int16)`` *wraps around*. Without explicit clipping, a
float sample of ``1.5`` would silently become ``-32767`` instead of ``+32767``,
corrupting audio with no diagnostic. We clip before casting.
"""

from __future__ import annotations

import io
import wave

import numpy as np

_PCM_MAX_I16 = 32767  # i16::MAX in Rust; matches the scale factor used by `handy`.


def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encode float32 samples (range [-1, 1]) as 16-bit PCM mono WAV bytes.

    Saturates samples outside [-1, 1] at ±32767 — mirrors Rust ``as i16``
    casting semantics. Empty input is allowed and produces a valid (zero-frame)
    WAV file.

    Args:
        samples: 1-D float array of audio samples in [-1, 1]. Values outside
            this range are clipped before quantisation.
        sample_rate: Sample rate in Hz to embed in the WAV header. Defaults to
            16000 because that is what OpenRouter STT models expect.

    Returns:
        A bytestring containing a complete RIFF/WAVE file: mono, 16-bit signed
        little-endian PCM, at ``sample_rate`` Hz.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * _PCM_MAX_I16).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()
