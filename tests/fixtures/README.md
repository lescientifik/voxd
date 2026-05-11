---
description: Audio fixtures for voxd tests — generation procedure and contracts.
---

# Test fixtures

All WAV fixtures are **PCM int16, mono, 16 kHz** to match the Silero v4 contract.

## `speech_fr.wav`

~4.3 s of French speech via `espeak-ng`, resampled to 16 kHz mono via `sox`.

```bash
espeak-ng -v fr -s 145 -w /tmp/_speech_fr_raw.wav \
    "Bonjour, ceci est un test de transcription vocale française."
sox /tmp/_speech_fr_raw.wav -r 16000 -c 1 -b 16 tests/fixtures/speech_fr.wav
```

## `silence.wav`

3 s of silence (zero samples), 16 kHz mono int16. Generated with:

```python
import wave
import numpy as np

samples = np.zeros(3 * 16000, dtype=np.int16)
with wave.open("tests/fixtures/silence.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(samples.tobytes())
```

## Regeneration

If you ever delete these files, rerun the snippets above. They are deterministic
modulo the espeak-ng version (the speech timing may shift by tens of ms).
