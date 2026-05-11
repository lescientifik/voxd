"""Silero v4 VAD wrapper and SmoothedVad state machine.

Port of `handy/src-tauri/src/audio_toolkit/vad/{silero,smoothed}.rs`. The wire
contract (frame size, ONNX input names, dtype, hidden-state shape) and the
production parameters (threshold=0.3, onset=2, prefill=15, hangover=15) are
verbatim from the Rust reference.

Public surface:

    SileroV4(model_path).push(frame_480) -> float       # raw probability
    SmoothedVad(SileroV4(...)).push(frame_480) -> np.ndarray | None
        # None on silence/onset-incomplete, np.ndarray on speech.
        # On the very first speech emission, the array bundles the rolling
        # pre-roll buffer (up to prefill+1 frames) followed by the current
        # frame; subsequent emissions are just the current frame.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

FRAME_SAMPLES = 480  # 30 ms @ 16 kHz — Silero v4 fixed frame length
SAMPLE_RATE = 16000
_LSTM_STATE_SHAPE = (2, 1, 64)


class SileroV4:
    """Streaming Silero v4 voice activity detector.

    Wraps the ONNX session; carries the LSTM hidden/cell state across
    consecutive `push` calls. Call `reset()` between independent recordings.
    """

    def __init__(self, model_path: Path) -> None:
        """Load the Silero v4 ONNX model from `model_path`.

        Args:
            model_path: Path to ``silero_vad_v4.onnx``.
        """
        # CPU-only — voxd is a headless desktop daemon, no GPU dependency.
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        # `sr` is a scalar int64 in the ONNX graph; pre-build it once.
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self._h: np.ndarray
        self._c: np.ndarray
        self.reset()

    def reset(self) -> None:
        """Zero out the LSTM hidden/cell state. Call between recordings."""
        self._h = np.zeros(_LSTM_STATE_SHAPE, dtype=np.float32)
        self._c = np.zeros(_LSTM_STATE_SHAPE, dtype=np.float32)

    def push(self, frame_480: np.ndarray) -> float:
        """Feed one 30 ms frame and return the speech probability.

        Args:
            frame_480: float32 mono samples of length 480. Values are expected
                in [-1, 1] but the model is robust to mild out-of-range input.

        Returns:
            Probability in [0, 1] that the frame contains speech.

        Raises:
            ValueError: if `frame_480` is not exactly 480 samples.
        """
        if frame_480.shape != (FRAME_SAMPLES,):
            raise ValueError(
                f"expected frame of shape ({FRAME_SAMPLES},), got {frame_480.shape}"
            )
        # ONNX expects float32 [1, 480]. astype(copy=False) is a no-op if already f32.
        x = frame_480.astype(np.float32, copy=False).reshape(1, FRAME_SAMPLES)
        outputs = self._session.run(
            None,
            {"input": x, "sr": self._sr, "h": self._h, "c": self._c},
        )
        # `session.run` returns a heterogeneous union per ty; we know the
        # graph yields three dense float32 arrays (output, hn, cn).
        out = np.asarray(outputs[0], dtype=np.float32)
        self._h = np.asarray(outputs[1], dtype=np.float32)
        self._c = np.asarray(outputs[2], dtype=np.float32)
        return float(out[0, 0])


class SmoothedVad:
    """Hysteresis layer over a `SileroV4`: pre-roll, onset, hangover.

    Port of `vad/smoothed.rs`. Maintains a rolling buffer of recent silence
    frames; once `onset` consecutive speech frames are seen, emits the buffered
    pre-roll plus the current frame in a single chunk, then emits each
    subsequent frame individually for as long as speech (or hangover) lasts.
    """

    def __init__(
        self,
        vad: SileroV4,
        prefill: int = 15,
        hangover: int = 15,
        onset: int = 2,
        threshold: float = 0.3,
    ) -> None:
        """Wrap a `SileroV4` with hysteresis smoothing.

        Args:
            vad: The underlying frame-level VAD.
            prefill: Number of past silence frames to prepend at speech onset.
            hangover: Number of frames to keep emitting after speech ends.
            onset: Consecutive speech-probability frames required to trigger.
            threshold: Probability threshold above which a frame is voice.
        """
        self._vad = vad
        self._prefill = prefill
        self._hangover = hangover
        self._onset = onset
        self._threshold = threshold

        # Rolling buffer of the last (prefill + 1) frames. Matches the Rust
        # behavior of capping the deque at prefill_frames + 1.
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=prefill + 1)
        self._hangover_counter = 0
        self._onset_counter = 0
        self._in_speech = False

    def reset(self) -> None:
        """Clear all smoothing state. Caller is responsible for resetting the inner VAD."""
        self._frame_buffer.clear()
        self._hangover_counter = 0
        self._onset_counter = 0
        self._in_speech = False

    def push(self, frame_480: np.ndarray) -> np.ndarray | None:
        """Feed one frame; return concatenated speech samples or ``None``.

        Returns:
            ``np.ndarray`` of float32 samples to keep (single frame in the
            steady-state speech regime, multi-frame pre-roll bundle on the very
            first emission after onset), or ``None`` if the frame is classified
            as silence / not-yet-confirmed onset.
        """
        # 1. Always buffer the frame for possible pre-roll. Store a copy so the
        #    caller can mutate `frame_480` without disturbing our ring buffer.
        self._frame_buffer.append(np.asarray(frame_480, dtype=np.float32).copy())

        # 2. Delegate the keep/drop decision to the inner VAD.
        prob = self._vad.push(frame_480)
        is_voice = prob > self._threshold

        # 3. Four-way state machine — direct port of smoothed.rs:51-95.
        if not self._in_speech and is_voice:
            # Potential onset: need `onset` consecutive voice frames.
            self._onset_counter += 1
            if self._onset_counter >= self._onset:
                self._in_speech = True
                self._hangover_counter = self._hangover
                self._onset_counter = 0
                # Emit the full rolling buffer (pre-roll + current frame).
                return np.concatenate(list(self._frame_buffer))
            return None

        if self._in_speech and is_voice:
            # Ongoing speech.
            self._hangover_counter = self._hangover
            return np.asarray(frame_480, dtype=np.float32)

        if self._in_speech and not is_voice:
            # End of speech — keep emitting through the hangover window.
            if self._hangover_counter > 0:
                self._hangover_counter -= 1
                return np.asarray(frame_480, dtype=np.float32)
            self._in_speech = False
            return None

        # Silence or broken onset sequence.
        self._onset_counter = 0
        return None
