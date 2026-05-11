---
description: Plan d'implémentation de voxd — daemon Python headless de transcription vocale via OpenRouter, contrôlé par signaux pkill, intégré à sway, exécuté en TDD par des subagents Opus séquentiels.
---

# voxd — Plan d'implémentation

## Objectif

Remplacer Handy (Tauri/Rust + GUI) par un daemon Python minimal pour usage Wayland/sway, où le GUI n'apporte rien. Pipeline complet :

```
SIGUSR2 → daemon → sounddevice capture (44.1k/48k) → soxr resample 16k
       → Silero v4 VAD (trim silences) → WAV PCM int16
       → POST /audio/transcriptions @ OpenRouter
       → wl-copy <text> + wtype <text>   (FIFO-serialized)
```

Toutes les décisions de pipeline (modèles, format wire, params VAD, retries) reprennent à l'identique ce qui est déjà tranché et testé dans `feat/openrouter-transcription` côté Rust — c'est une réécriture portable, pas une refonte fonctionnelle.

## Décisions architecturales (issues de la review)

Verrouillées avant écriture de code. Tout subagent qui veut s'en écarter doit escalader.

| Sujet | Décision |
|---|---|
| Modèle d'exécution | `exec voxd` dans `~/.config/sway/config.d/voxd.conf`. Aucun systemd, aucun `sway-session.target`, aucun `import-environment`. Le daemon hérite naturellement de l'env sway. |
| IPC | `pkill -USR2 voxd` envoyé par le bindsym. Aucun pidfile, aucun unix socket. SIGUSR2 = toggle. Pas de cancel en v1. |
| Format config | TOML, lu en stdlib `tomllib` (3.11+), édité via `tomlkit` (préserve commentaires). Path `$XDG_CONFIG_HOME/voxd/config.toml`, mode 0600. |
| Resampler | `soxr` (libsoxr Python binding), API streaming, état porté entre chunks. |
| VAD | `onnxruntime` brut avec `silero_vad_v4.onnx` bundlé. État `(h, c)` shape `[2,1,64]` float32 porté entre frames. **Ne pas** utiliser le paquet `silero-vad` PyPI (tire torch, v5-only). |
| Injection | `wl-copy` + `wtype` en parallèle, chacun désactivable. Pas de fallback paste-via-ctrl-V en v1. |
| Threading | asyncio main loop. `loop.add_signal_handler(SIGUSR2, ...)`. Callback PortAudio → `loop.call_soon_threadsafe(queue.put_nowait, frame)`. Mutations d'état dans des coroutines. |
| `voxd setup` | Pure édition de fichiers. Ne lance pas voxd. Ne touche pas à systemd. Idempotent. |
| Reload config | Restart manuel (`pkill voxd && voxd &`). Documenté dans le message de fin de setup. |

## Contrats wire (bit-exact avec le Rust)

### WAV encoding (port de `audio/utils.rs::samples_to_wav_bytes`)

```python
def samples_to_wav_bytes(samples: np.ndarray) -> bytes:
    """Encode float32 samples (range [-1, 1]) as 16-bit PCM mono WAV at 16 kHz.

    Piège: Rust `as` casting sature, NumPy `astype(int16)` enroule.
    Sans clip, sample=1.5 devient -32767 au lieu de 32767. Clip obligatoire.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    # ... écrit RIFF + fmt chunk (PCM, mono, 16k, 16bit) + data chunk
```

Référence Rust : `src-tauri/src/audio_toolkit/audio/utils.rs`. Test attendu : sample=1.5 → encode → décode → `[32767]`.

### Silero v4 ONNX — contrat I/O

D'après `cjpais/vad-rs` (le crate utilisé par handy) :

```
Inputs :
  input  : float32 [1, 480]        (30 ms @ 16 kHz)
  sr     : int64   [1]             = 16000
  h      : float32 [2, 1, 64]      (LSTM hidden state)
  c      : float32 [2, 1, 64]      (LSTM cell state)

Outputs :
  output : float32 [1, 1]          (probabilité de speech)
  hn     : float32 [2, 1, 64]      → recycle comme h au tour suivant
  cn     : float32 [2, 1, 64]      → recycle comme c au tour suivant
```

Init : `h = c = np.zeros([2, 1, 64], dtype=np.float32)`. Reset entre enregistrements.

### SmoothedVad — params et machine à états

Identiques au Rust (`src-tauri/src/managers/audio.rs:124` + `vad/smoothed.rs`) :

| Paramètre | Valeur | Rôle |
|---|---|---|
| `threshold` | 0.3 | output Silero > 0.3 → speech |
| `onset_frames` | 2 | 2 frames consécutives requises pour passer en speech |
| `prefill_frames` | 15 | Pré-roll de 15 frames (~450 ms) prependées au début |
| `hangover_frames` | 15 | Post-roll de 15 frames gardées après silence détecté |
| `frame_samples` | 480 | 30 ms @ 16 kHz |

Machine à états (4 transitions) : voir `vad/smoothed.rs:51-95` côté Rust. Port direct.

### Recording terminé — padding et empty-skip

```python
if len(samples) == 0:
    # Pas d'upload, notif "no speech detected"
    return
if len(samples) < TARGET_RATE:               # < 1 s
    samples = np.pad(samples, (0, TARGET_RATE * 5 // 4 - len(samples)))
# upload
```

Le Rust handy fait pareil (`managers/audio.rs:474-478`). NE PAS uploader 0 samples.

### OpenRouter `/audio/transcriptions` — body et endpoint

Endpoint documenté : <https://openrouter.ai/docs/guides/overview/multimodal/stt>

```python
body = {
    "model": "openai/whisper-large-v3-turbo",
    "input_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"},
}
if model.supports_language and language != "auto":
    body["language"] = language          # "fr", "en", ...
if model.supports_prompt and user_prompt:
    body["prompt"] = user_prompt         # NOTE: pas dans le schéma officiel
                                         # OpenRouter — forward best-effort vers
                                         # provider sous-jacent. À documenter.

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "User-Agent": "voxd/0.1.0 (+https://github.com/.../voxd)",
    "X-Title": "voxd",
}
```

Response success : `{"text": "...", "usage": {...}}`. Top-level `text`.

### Retry policy (port de `llm_client/{error.rs, remote.rs}`)

```
MAX_RETRIES = 5  → 6 tentatives au total (attempt ∈ 0..=5)
backoff = 250ms * 2^(attempt-1) ± jitter ±20 %
  attempt=0 → 0 ms
  attempt=1 → 250 ms ± 20 %
  attempt=2 → 500 ms ± 20 %
  attempt=3 → 1000 ms ± 20 %
  attempt=4 → 2000 ms ± 20 %
  attempt=5 → 4000 ms ± 20 %

401            → AuthInvalid (pas de retry)
429, 5xx       → Retryable
autres 4xx     → Permanent (pas de retry)
erreurs réseau → Retryable
```

### Catalogue de modèles (port de `llm_client/catalog.rs`)

V1 = STT endpoint seulement.

```python
MODELS = {
    "openai/whisper-large-v3-turbo": Model(endpoint="stt", supports_prompt=True, supports_language=True),
    "openai/whisper-large-v3":       Model(endpoint="stt", supports_prompt=True, supports_language=True),
    "openai/gpt-4o-transcribe":      Model(endpoint="stt", supports_prompt=True, supports_language=True),
    "openai/gpt-4o-mini-transcribe": Model(endpoint="stt", supports_prompt=True, supports_language=True),
    "google/chirp-3":                Model(endpoint="stt", supports_prompt=False, supports_language=True),
}
DEFAULT_MODEL = "openai/whisper-large-v3-turbo"
```

Gemini chat endpoint reporté à plus tard.

## Stack technique

- **Python ≥ 3.11** (tomllib stdlib, `Self` type)
- **uv** pour deps + `uv tool install`
- **ruff** + **ty** + **pytest** (short stdout)

### Dépendances runtime (pyproject.toml)

```toml
[project]
name = "voxd"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "sounddevice>=0.4",       # audio capture (PortAudio)
    "numpy>=2.0",             # buffer math
    "onnxruntime>=1.18",      # Silero v4 inference
    "httpx>=0.27",            # async HTTP client
    "soxr>=0.5",              # streaming resampler (libsoxr)
    "tomlkit>=0.13",          # TOML édition avec préservation de commentaires
]

[project.scripts]
voxd = "voxd.cli:main"
```

### Dépendances système externes

- `wtype` — injection clavier Wayland (`zwp_virtual_keyboard_v1`)
- `wl-clipboard` — `wl-copy` pour le clipboard
- `notify-send` — indicateur (paquet `libnotify`)
- PortAudio shared lib (sounddevice runtime dep)

## Layout du repo

```
voxd/
├── pyproject.toml
├── README.md
├── LICENSE
├── CLAUDE.md
├── docs/
│   ├── plan-voxd.md             (ce fichier)
│   └── review-plan-voxd.md      (review adversariale qui a produit ce plan)
├── src/voxd/
│   ├── __init__.py
│   ├── __main__.py              # entry point CLI
│   ├── cli.py                   # parsing + dispatch
│   ├── daemon.py                # main loop, signaux, FIFO upload queue
│   ├── audio.py                 # sounddevice capture + soxr resample
│   ├── vad.py                   # Silero ONNX + smoothed (port du Rust)
│   ├── transcribe.py            # client OpenRouter (httpx)
│   ├── inject.py                # subprocess wtype + wl-copy
│   ├── indicator.py             # notify-send avec replace_id
│   ├── config.py                # load/save TOML via tomlkit
│   ├── setup.py                 # `voxd setup` : config + sway snippet
│   ├── doctor.py                # `voxd doctor` : check deps
│   └── resources/
│       └── silero_vad_v4.onnx   # ~2 MB, bundlé via package_data
└── tests/
    ├── conftest.py              # fixtures partagées (WAV samples, mock servers)
    ├── fixtures/
    │   ├── speech_fr.wav        # ~3 s de parole française @ 16 kHz
    │   ├── speech_en.wav        # ~3 s de parole anglaise @ 16 kHz
    │   ├── silence.wav          # silence pur
    │   └── speech_44k.wav       # speech au taux non-natif (test resample)
    ├── test_config.py
    ├── test_vad.py
    ├── test_audio.py
    ├── test_wav.py
    ├── test_transcribe.py
    ├── test_inject.py
    ├── test_indicator.py
    ├── test_setup.py
    ├── test_doctor.py
    ├── test_cli.py
    └── test_daemon_e2e.py       # le test de référence end-to-end
```

## Threading model

```
            ┌─────────────────────────────────┐
            │  Main thread (asyncio loop)     │
            │                                 │
   SIGUSR2 →│  loop.add_signal_handler        │
            │   ↓ on_toggle (coroutine)       │
            │   ↓                             │
            │  upload_queue (asyncio.Queue)   │←──┐
            │   ↓                             │   │
            │  upload_worker (coroutine)      │   │
            │   ↓ httpx async POST            │   │
            │   ↓ async with wtype_lock:      │   │
            │   ↓   subprocess wl-copy + wtype│   │
            └─────────────────────────────────┘   │
                                                  │
            ┌─────────────────────────────────┐   │
            │  PortAudio callback thread      │   │
            │                                 │   │
   mic →    │  on_audio(indata, ...)          │   │
            │   ↓ soxr.ResampleStream         │   │
            │   ↓ smoothed_vad.push(frame)    │   │
            │   ↓ if speech →                 │   │
            │      loop.call_soon_threadsafe( │   │
            │        speech_queue.put_nowait, │   │
            │        frame)                   │───┘
            └─────────────────────────────────┘
```

- Aucune mutation d'état partagée hors event loop
- `speech_queue` (intra-loop) accumule les frames pendant l'enregistrement
- À l'arrêt : la coroutine consomme `speech_queue`, concat, pad, encode, push dans `upload_queue`
- `upload_queue` est traitée FIFO par un unique worker — ordre des `wtype` garanti

## Philosophie de tests

> Tests validate features (the *what*), not that libraries work as advertised (the *how*). — CLAUDE.md user

**Chaque test doit jouer un scénario d'usage réaliste**, pas wrapper un mock. Les tests unitaires purs sont OK pour les fonctions pures (encoder WAV, classifier statut HTTP), mais le cœur de la valeur est dans les tests d'intégration par module et le test end-to-end.

### Principes opérationnels

1. **Préfère le fixture réel au mock**. WAV fixtures avec vraie parole > sinusoïdes synthétiques. La parole synthétique est OK pour vérifier les bordures (silence pur, sinusoïde 50 ms trop courte), pas comme cas central.
2. **Mock à la frontière externe seulement**. OpenRouter HTTP → `httpx.MockTransport`. `subprocess.run` pour wtype/wl-copy → monkeypatch qui capture argv. Tout ce qui est à l'intérieur de voxd → vrai code.
3. **Tests par module ≠ tests unitaires**. `test_vad.py` n'est pas "appeler `push_frame` 100 fois", c'est "jouer un WAV de parole, vérifier que le SmoothedVad produit les bonnes frames de speech (avec pré-roll, hangover, etc.)".
4. **Le test E2E (`test_daemon_e2e.py`) est l'oracle ultime**. Si lui passe, voxd marche dans la vraie vie.
5. **Tests rapides**. Le suite complète doit tourner en < 10 s. ONNX session load = ~300 ms, donc fixture session-scoped.

### Patterns par type

**Tests audio/VAD** : `pytest.fixture` qui charge un WAV fixture (`scipy.io.wavfile`), feed via le pipeline, assert sur les frames produites. Pas de mock de `onnxruntime` — on tourne le vrai ONNX en CI (le fichier est bundlé, taille ~2 MB).

**Tests transcribe** : `httpx.MockTransport` configuré par cas (200, 401, 429+200, 5×500, network error). Assert sur le body envoyé (model, input_audio.format, language omitted/present, etc.) ET sur le résultat.

**Tests inject** : `monkeypatch.setattr(subprocess, "run", capture)` qui stocke chaque appel. Assert que `wl-copy` reçoit le texte ET que `wtype` reçoit le texte (ou pas, selon flags config).

**Tests daemon E2E** : spawn voxd avec config tmp + mock OpenRouter local (httpx mock ou aiohttp server sur port libre) + monkeypatch sounddevice qui replay un WAV fixture + monkeypatch subprocess pour wtype/wl-copy. Envoie SIGUSR2 → joue audio → SIGUSR2 → attend → assert wtype.run a été appelé avec le texte attendu.

## Méthodologie d'exécution — subagents Opus séquentiels

Chaque step de la roadmap est exécuté par **un** subagent Opus, **séquentiellement** (jamais en parallèle), avec autonomie maximale.

### Le briefing-type pour un subagent

Quand on lance un subagent sur le step N, il reçoit :

1. **Le plan complet** (ce fichier) — pour comprendre le contexte global
2. **Le step à exécuter** — section roadmap correspondante, copiée
3. **Les fichiers à toucher** — paths exacts
4. **L'instruction d'autonomie** :

   > Tu as autorité pour décider : structure interne du module, nommage des variables/fonctions privées, factorisation interne, tests d'intégration supplémentaires au-delà du minimum requis, formulations exactes des messages d'erreur, choix de bibliothèques stdlib pour les détails.
   >
   > Tu N'AS PAS autorité pour : changer une décision architecturale du plan, modifier le contrat wire (Silero I/O, OpenRouter body, WAV format, retry policy), ajouter une dépendance non listée dans pyproject.toml sans justification dans un commit séparé.
   >
   > Tu escalades (rends la main avec un rapport clair) SI :
   >   - Une API externe se comporte différemment de ce qui est spec'd (ex: OpenRouter rejette la requête, soxr a une API différente)
   >   - Un test d'intégration ne passe pas après 2 tentatives de fix et tu suspectes un problème de design
   >   - Tu identifies un risque que le plan n'a pas couvert
   >
   > Tu NE poses PAS de questions clarificatrices "préventives". Le plan a été reviewé, les décisions sont fermes. Tu implémentes.

5. **Le DoD du step** — critères de "fini"
6. **L'instruction de commit** — un commit conventional par step, message format `<type>(<scope>): <description>` (ex: `feat(vad): port smoothed VAD from handy`)

### Lancement

```
Pour chaque step de la roadmap, lancer un subagent Opus en mode foreground.
Attendre son rapport avant de lancer le suivant. Si le subagent escalade,
on traite ensemble avant de relancer (potentiellement le même step amendé).
```

Pas de parallélisation, pas de race, pas de mauvaises surprises de merge.

## Roadmap TDD (red/green) — tâches atomiques

Chaque step suit le pattern :
1. **Red** : écrire un test qui échoue
2. **Green** : minimum de code pour le faire passer
3. **DoD** : critères vérifiables avant de committer
4. **Commit** : `<type>(<scope>): <description>`

### Step 0 — Bootstrap repo

**Test rouge** : `uv run pytest` retourne 0 (aucun test mais pytest discover OK). `uv run voxd --version` print `voxd 0.1.0`. `uv run ruff check src tests` et `uv run ty src` passent.

**Délivrable** :
- `pyproject.toml` avec les deps déclarées + `[tool.ruff]`, `[tool.ty]`, `[tool.pytest.ini_options]`
- `src/voxd/__init__.py` avec `__version__ = "0.1.0"`
- `src/voxd/__main__.py` qui appelle `cli.main()`
- `src/voxd/cli.py` minimal : argparse avec `--version` (sous-commandes ajoutées au step 9)
- `tests/conftest.py` vide
- `.python-version` = `3.11`
- `.gitignore` standard Python + uv
- `README.md` minimal (titre + 2 lignes)

**DoD** : les 3 commandes ci-dessus passent en local.

**Commit** : `chore: bootstrap repo with uv + pytest + ruff + ty`

### Step 1 — Config TOML (load/save/defaults)

**Test rouge** (`tests/test_config.py`) :
- `test_load_with_all_defaults` : config fichier inexistant → renvoie les valeurs par défaut sans crash
- `test_roundtrip_preserves_comments` : écrit un config avec un commentaire, le relit avec `tomlkit`, le ré-écrit, le commentaire est encore là
- `test_file_mode_is_0600_on_create` : après save, `stat().st_mode & 0o777 == 0o600`
- `test_load_with_missing_field_falls_back` : config avec `[openrouter]` mais sans `prompt` → `prompt = ""` (défaut), pas de crash
- `test_load_with_utf8_prompt` : prompt = "café, naïve, façon" → roundtrip exact

**Délivrable** :
- `src/voxd/config.py` avec :
  ```python
  @dataclass(frozen=True)
  class OpenRouterConfig:
      api_key: str = ""
      model: str = "openai/whisper-large-v3-turbo"
      language: str = "auto"
      prompt: str = ""

  @dataclass(frozen=True)
  class AudioConfig:
      device: str = ""

  @dataclass(frozen=True)
  class InjectConfig:
      type: bool = True       # wtype
      clipboard: bool = True  # wl-copy

  @dataclass(frozen=True)
  class Config:
      openrouter: OpenRouterConfig
      audio: AudioConfig
      inject: InjectConfig

  def default_path() -> Path: ...        # respecte XDG_CONFIG_HOME
  def load(path: Path | None = None) -> Config: ...
  def save(cfg: Config, path: Path | None = None) -> None: ...
  ```

**DoD** : tous les tests `test_config.py` verts. Edition manuelle du `config.toml` avec un commentaire, relance, commentaire toujours là.

**Commit** : `feat(config): TOML load/save with tomlkit and comment preservation`

### Step 2 — Silero v4 VAD + SmoothedVad

**Test rouge** (`tests/test_vad.py`) :
- Fixture session-scoped : `silero_session` qui charge l'ONNX une fois
- `test_silero_returns_high_prob_on_speech` : feed une frame de 480 samples extraite d'un WAV de parole → `prob > 0.5`
- `test_silero_returns_low_prob_on_silence` : feed une frame de zéros → `prob < 0.1`
- `test_silero_state_is_carried_between_frames` : feeder la même frame deux fois → `prob` différent au tour 2 (preuve que `h, c` ont changé)
- `test_smoothed_vad_with_real_speech_wav` : feed un WAV fr de ~3 s → recevoir un buffer de speech qui contient pré-roll + tout le speech + hangover. Assert `len(out_samples) ≥ speech_duration * 16000 - tolerance` et `≤ speech_duration * 16000 + (prefill + hangover) * 480`
- `test_smoothed_vad_with_pure_silence_emits_nothing` : feed silence pur → aucune frame Speech
- `test_smoothed_vad_short_blip_rejected` : feed 50 ms de sinusoïde 440 Hz → onset_frames=2 pas atteint → rien émis
- `test_smoothed_vad_reset_clears_state` : feed speech, reset, feed silence → pas de speech leak

**Délivrable** :
- `src/voxd/vad.py` avec :
  ```python
  class SileroV4:
      def __init__(self, model_path: Path): ...
      def reset(self) -> None: ...
      def push(self, frame_480: np.ndarray) -> float: ...  # retourne prob

  class VadFrame:
      Speech: np.ndarray
      Noise: None

  class SmoothedVad:
      def __init__(self, vad: SileroV4, prefill=15, hangover=15, onset=2, threshold=0.3): ...
      def push(self, frame_480: np.ndarray) -> np.ndarray | None: ...
      def reset(self) -> None: ...
  ```
- `src/voxd/resources/silero_vad_v4.onnx` copié depuis `/home/thenry/Projects/handy/src-tauri/resources/models/silero_vad_v4.onnx`
- `tests/fixtures/speech_fr.wav` et `tests/fixtures/silence.wav` créés (ou TTS-generated, doc dans `tests/fixtures/README.md`)

**DoD** : tests verts. Manual sanity-check : `uv run python -c "import voxd.vad; ..."` produit un résultat cohérent sur le fixture.

**Commit** : `feat(vad): port Silero v4 + SmoothedVad from handy (bit-exact)`

### Step 3 — WAV PCM int16 encoding

**Test rouge** (`tests/test_wav.py`) :
- `test_samples_to_wav_bytes_clips_amplitude` : sample=1.5 → décode → `[32767]` (saturation, pas wrap)
- `test_samples_to_wav_bytes_clips_negative` : sample=-1.5 → décode → `[-32767]`
- `test_roundtrip_via_scipy` : encode un array float32 connu → décode avec `scipy.io.wavfile.read` → assert proche (tolerance flottante)
- `test_header_riff_wave` : bytes commencent par `b'RIFF'` puis `b'WAVE'` à offset 8
- `test_mono_16k_16bit` : décode → spec channels=1, rate=16000, bits=16
- `test_empty_input` : encode `np.array([])` → bytes valides parsables comme WAV vide

**Délivrable** : `src/voxd/wav.py` (ou fonction dans `audio.py`) :
```python
def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encode float32 [-1, 1] samples as 16-bit PCM mono WAV. Saturates beyond ±1."""
```

**DoD** : tests verts.

**Commit** : `feat(wav): encode float32 samples to 16-bit PCM mono WAV with saturation`

### Step 4 — Audio capture + resample streaming

**Test rouge** (`tests/test_audio.py`) :
- `test_resample_streaming_48k_to_16k_preserves_signal` : feed une sinusoïde 440 Hz @ 48 kHz par chunks de 1024 samples via `soxr.ResampleStream` → concat output → FFT → peak à 440 Hz ± 5 Hz
- `test_resample_chunk_boundaries_dont_introduce_ringing` : feed la même sinusoïde en deux passes (chunks de 100 puis chunks de 10000) → output identique à tolérance numérique
- `test_capture_pipeline_with_wav_fixture` : monkey-patcher `sounddevice.InputStream` pour qu'il replay un WAV 48 kHz fixture, faire tourner le pipeline, récupérer les frames 480 samples @ 16 kHz produites, vérifier nombre et content (un buffer assez long pour ne pas être tronqué)
- `test_downmix_stereo_to_mono` : feed un signal stéréo (canal L = sinusoïde, R = silence) → output mono = sinusoïde/2

**Délivrable** :
- `src/voxd/audio.py` avec :
  ```python
  class AudioCapture:
      def __init__(self, device: str, on_frame: Callable[[np.ndarray], None], loop: asyncio.AbstractEventLoop): ...
      def start(self) -> None: ...
      def stop(self) -> None: ...
      # Internally:
      #   - opens sd.InputStream with native rate/channels
      #   - in callback: downmix to mono, push to soxr.ResampleStream
      #   - emit 480-sample frames via loop.call_soon_threadsafe(on_frame, ...)
  ```
- `tests/fixtures/speech_44k.wav` ajouté

**DoD** : tests verts. Test manuel : `python -c "..."` qui lance la capture sur le vrai micro pendant 1 s et imprime `len(frames_produced)`.

**Commit** : `feat(audio): streaming capture with soxr resampling and mono downmix`

### Step 5 — OpenRouter client + retry

**Test rouge** (`tests/test_transcribe.py`) : reproduction exacte des 6 cas du Rust (`remote.rs::tests`).

- `test_stt_200_returns_text` : mock POST /audio/transcriptions → 200 `{"text": "hello world"}` → returns "hello world"
- `test_body_shape` : mock → capture request body → assert `body["model"]`, `body["input_audio"]["format"] == "wav"`, `body["input_audio"]["data"]` est base64 décodable et commence par `RIFF`
- `test_language_omitted_when_auto` : `language="auto"` → `"language"` absent du body
- `test_language_present_when_supported` : `language="fr"`, model whisper → `body["language"] == "fr"`
- `test_language_omitted_when_model_does_not_support` : model = chirp-3 (supports_language=True), `language="fr"` → effectivement présent ; test inverse avec un modèle hypothétique `supports_language=False`
- `test_prompt_omitted_when_empty` : `user_prompt=""` → absent
- `test_prompt_omitted_when_model_does_not_support` : chirp-3 + prompt non-vide → absent
- `test_401_raises_auth_invalid_no_retry` : mock 401 → raises `AuthInvalid`, mock called exactly once
- `test_429_then_200_retries_and_succeeds` : mock 429 puis 200 → returns text, mock called twice
- `test_5_consecutive_500_raises_retryable` : mock 500 toujours → raises `Retryable`, mock called 6 fois (MAX_RETRIES + 1)
- `test_network_error_raises_retryable` : URL invalide → raises `Retryable`
- `test_backoff_schedule` : test pur de `backoff_delay(attempt, jitter)` — voir Rust pour valeurs exactes

**Délivrable** :
- `src/voxd/transcribe.py` avec :
  ```python
  class AuthInvalid(Exception): ...
  class Retryable(Exception): ...
  class Permanent(Exception): ...

  @dataclass(frozen=True)
  class Model:
      id: str
      supports_prompt: bool
      supports_language: bool

  MODELS: dict[str, Model] = {...}

  async def transcribe(
      wav_bytes: bytes,
      *,
      api_key: str,
      model_id: str,
      language: str,
      prompt: str,
      base_url: str = "https://openrouter.ai/api/v1",
      client: httpx.AsyncClient | None = None,
  ) -> str: ...
  ```
- Retry interne avec backoff exponentiel + jitter ±20 %

**DoD** : 12 tests verts. Le body shape doit être bit-exact identique à ce que produit le Rust (vérifier en lançant le test Rust côté handy et en comparant un dump).

**Commit** : `feat(transcribe): OpenRouter client with retry policy (port from handy)`

### Step 6 — Inject (wtype + wl-copy)

**Test rouge** (`tests/test_inject.py`) :
- `test_inject_default_calls_both_wlcopy_and_wtype` : config par défaut → `subprocess.run` appelé deux fois, argv0=`wl-copy` puis argv0=`wtype`, texte arrive en stdin (`wl-copy`) ou argv (`wtype`)
- `test_inject_clipboard_false_skips_wlcopy` : config `clipboard=false` → seul `wtype` appelé
- `test_inject_type_false_skips_wtype` : config `type=false` → seul `wl-copy`
- `test_inject_utf8_french_accents` : texte = `"café, naïve, façon"` → les deux subprocess reçoivent le texte intact (encode UTF-8 vers stdin pour `wl-copy`, argv pour `wtype`)
- `test_inject_empty_text_is_noop` : texte vide → aucun subprocess appelé
- `test_inject_wlcopy_called_before_wtype` : si les deux sont activés, ordre = wl-copy d'abord, wtype ensuite (pour que le clipboard soit prêt si wtype foire)
- `test_inject_propagates_subprocess_error` : monkeypatch `subprocess.run` pour lever `FileNotFoundError` (wtype absent) → raise `InjectError` avec message clair

**Délivrable** :
- `src/voxd/inject.py` :
  ```python
  class InjectError(Exception): ...

  def inject(text: str, cfg: InjectConfig) -> None:
      """Synchronously injects text via wl-copy (if enabled) then wtype (if enabled).
      wl-copy first so clipboard is set even if wtype fails."""
  ```

**DoD** : tests verts.

**Commit** : `feat(inject): wl-copy + wtype subprocess with config flags`

### Step 7 — Indicator (notify-send)

**Test rouge** (`tests/test_indicator.py`) :
- `test_show_recording_calls_notify_send` : argv contient `-t 0` (persistent), `-u low` (urgency low), message attendu, `-p` (print id)
- `test_show_transcribing_replaces_recording` : appelle show_recording → capture le `notif_id` retourné (fake la sortie de notify-send -p) → appelle show_transcribing → argv contient `-r <notif_id>`
- `test_hide_closes_notification` : appelle hide → soit gdbus CloseNotification soit notify-send avec `-t 1` court
- `test_indicator_survives_notify_send_failure` : monkeypatch subprocess pour lever → indicator log + continue, ne crash pas le daemon

**Délivrable** :
- `src/voxd/indicator.py` :
  ```python
  class Indicator:
      def show_recording(self) -> None: ...
      def show_transcribing(self) -> None: ...
      def show_error(self, msg: str) -> None: ...
      def hide(self) -> None: ...
  ```

**DoD** : tests verts.

**Commit** : `feat(indicator): notify-send wrapper with replace_id lifecycle`

### Step 8 — Daemon (signal + concurrence + FIFO upload queue)

C'est **le step critique**. Le subagent qui prend ce step doit avoir lu `/home/thenry/Projects/handy/src-tauri/src/managers/audio.rs` et `audio/recorder.rs` pour comprendre le state model du Rust.

**Test rouge** (`tests/test_daemon_e2e.py`) :

Le scénario E2E de référence :

```python
async def test_full_cycle_record_transcribe_inject(
    tmp_path,
    mock_openrouter_server,    # local httpx server returning {"text": "hello world"}
    fake_audio_input,           # monkeypatched sd.InputStream that replays speech_fr.wav
    capture_subprocess,         # captures wl-copy + wtype invocations
    monkey_xdg_runtime,         # XDG_RUNTIME_DIR = tmp_path
):
    cfg = make_test_config(tmp_path, api_key="fake")
    daemon = Daemon(cfg, openrouter_base_url=mock_openrouter_server.url)

    task = asyncio.create_task(daemon.run())
    await asyncio.sleep(0.1)  # let it start

    # First toggle: start recording
    os.kill(os.getpid(), signal.SIGUSR2)
    await fake_audio_input.replay_finished()

    # Second toggle: stop + upload + inject
    os.kill(os.getpid(), signal.SIGUSR2)
    await capture_subprocess.wait_for_call("wtype")

    # Assertions
    assert capture_subprocess.calls("wl-copy")[-1].stdin == "hello world"
    assert capture_subprocess.calls("wtype")[-1].argv[-1] == "hello world"
    assert mock_openrouter_server.request_count == 1

    daemon.stop()
    await task
```

Autres tests dans le même fichier :
- `test_double_toggle_during_upload_queues_second_recording` : SIGUSR2 → audio → SIGUSR2 → audio → SIGUSR2 (avant que le premier upload finisse) → audio → SIGUSR2. Mock OpenRouter renvoie text différent par appel ("first", "second"). Assert wtype reçoit "first" PUIS "second" (FIFO).
- `test_empty_recording_no_upload` : SIGUSR2 → silence pur → SIGUSR2. Assert `mock_openrouter_server.request_count == 0`, notif "no speech detected".
- `test_short_recording_padded_to_1.25s` : SIGUSR2 → 0.5 s de speech → SIGUSR2. Assert que le WAV envoyé a une durée >= 1.25 s.
- `test_401_clears_indicator_and_notifies` : mock OpenRouter renvoie 401 → assert indicator hide + error notif.
- `test_5xx_retries_then_inject` : mock 500 deux fois puis 200 → assert wtype appelé une fois avec le texte.

**Délivrable** :
- `src/voxd/daemon.py` :
  ```python
  class Daemon:
      def __init__(self, cfg: Config, *, openrouter_base_url: str = "...", model_path: Path = ...): ...
      async def run(self) -> None: ...
      def stop(self) -> None: ...
      # Internals:
      #   - asyncio.Queue speech_frames (filled by audio callback via call_soon_threadsafe)
      #   - asyncio.Queue upload_queue (FIFO, consumed by worker)
      #   - asyncio.Lock wtype_lock
      #   - loop.add_signal_handler(SIGUSR2, self._on_toggle)
      #   - _on_toggle: dispatches between _start_recording and _stop_and_enqueue
      #   - _upload_worker: consumes upload_queue, calls transcribe, inject under wtype_lock
  ```
- `tests/conftest.py` enrichi avec les fixtures `mock_openrouter_server`, `fake_audio_input`, `capture_subprocess`

**DoD** : tous les tests E2E verts. Test manuel : `voxd` avec config dev, `pkill -USR2 voxd`, parle, `pkill -USR2 voxd`, vérifie que le texte s'injecte dans une fenêtre.

**Commit** : `feat(daemon): asyncio loop with SIGUSR2 toggle and FIFO upload queue`

### Step 9 — CLI dispatcher (sous-commandes)

**Test rouge** (`tests/test_cli.py`) :
- `test_version_flag_prints_version`
- `test_no_args_starts_daemon` : monkey-patcher Daemon.run pour assert qu'il est appelé
- `test_setup_dispatches_to_setup_main`
- `test_doctor_dispatches_to_doctor_main`
- `test_config_prints_path`
- `test_config_edit_opens_editor` : monkey-patch `subprocess.run` qui capture l'argv (`$EDITOR <path>`)
- `test_unknown_command_shows_help`

**Délivrable** :
- `src/voxd/cli.py` étendu avec argparse subparsers : `setup`, `doctor`, `config [--edit]`. Pas de `toggle` (le bindsym appelle `pkill` directement).

**DoD** : tests verts.

**Commit** : `feat(cli): subcommand dispatch (setup, doctor, config)`

### Step 10 — `voxd setup`

**Test rouge** (`tests/test_setup.py`) :
- `test_setup_creates_config_with_provided_api_key` : monkey-patch `getpass.getpass` pour retourner "sk-or-v1-test", tmp HOME → après setup, `~/.config/voxd/config.toml` existe avec mode 0600 et contient la clé
- `test_setup_writes_sway_snippet` : `~/.config/sway/config.d/voxd.conf` contient `exec voxd` ET `bindsym ... exec pkill -USR2 voxd`
- `test_setup_adds_include_line_if_missing` : sway config sans `include ~/.config/sway/config.d/*` → après setup, la ligne est présente une fois
- `test_setup_does_not_duplicate_include_line` : run twice → la ligne est présente UNE seule fois
- `test_setup_preserves_existing_config_comments` : pre-create un config avec commentaires → run setup → commentaires toujours là
- `test_setup_does_not_overwrite_existing_api_key_without_confirm` : config existe avec clé non-vide → setup ne l'écrase pas (ou prompt confirme)
- `test_setup_does_not_spawn_voxd` : assert que `subprocess.run` ou `Popen` n'a pas été appelé pour lancer voxd
- `test_setup_prints_final_message` : capture stdout → contient `pkill voxd && voxd` (instruction de restart)

**Délivrable** :
- `src/voxd/setup.py` :
  ```python
  def main() -> int:
      """Interactive: API key prompt, write config.toml, write sway snippet,
      ensure include line, print restart instructions. Never spawns voxd."""
  ```

**DoD** : tous tests verts. Test manuel : run dans un tmpfs HOME, vérifier les fichiers produits sont conformes.

**Commit** : `feat(setup): interactive config + sway snippet generator (idempotent)`

### Step 11 — `voxd doctor`

**Test rouge** (`tests/test_doctor.py`) :
- `test_doctor_reports_missing_wtype` : monkey-patch `shutil.which("wtype")` → None → output contient "wtype: MISSING"
- `test_doctor_reports_present_tools` : tous présents → output `wtype: OK`, `wl-copy: OK`, `notify-send: OK`
- `test_doctor_handles_portaudio_import_error` : monkey-patch `import sounddevice` pour lever `OSError` → output `PortAudio: MISSING (...)`, pas de crash
- `test_doctor_warns_on_non_sway_desktop` : `os.environ["XDG_CURRENT_DESKTOP"] = "GNOME"` → output contient un warning sur wtype/GNOME
- `test_doctor_checks_mic_access` : tente d'ouvrir un `InputStream` 100 ms, ferme. Si OSError → `Mic access: DENIED`
- `test_doctor_exit_code` : retourne 0 si tout OK, 1 si une dep critique manque

**Délivrable** :
- `src/voxd/doctor.py` :
  ```python
  def main() -> int: ...
  ```

**DoD** : tests verts.

**Commit** : `feat(doctor): check wtype, wl-copy, notify-send, PortAudio, mic access`

### Step 12 — Polish + README + smoke test final

**Test rouge** :
- `test_full_user_journey` : un seul test qui simule l'expérience utilisateur complète :
  1. Run `voxd setup` (avec mocks) → fichiers de config créés
  2. Spawn `voxd` en subprocess avec config tmp
  3. Send SIGUSR2 → fake audio → SIGUSR2
  4. Assert wtype + wl-copy ont reçu le texte du mock OpenRouter
  5. Send SIGTERM → daemon exit propre (queue drainée)

**Délivrable** :
- `README.md` complet : install (`uv tool install voxd`), `voxd setup`, troubleshooting (`voxd doctor`), config reference
- Code review sweep : enlever les `print` de debug, vérifier docstrings, ruff/ty sans warnings
- Vérifier que `silero_vad_v4.onnx` est bien inclus dans le wheel (`tool.hatch.build` ou équivalent)
- Vérifier que `voxd --version` print la bonne version

**DoD** : test final E2E vert. Install sur la machine sway de l'utilisateur via `uv tool install --editable .`, run, parle dans Slack/foot/firefox, vérifie que ça marche.

**Commit** : `chore: README, polish, prepare v0.1.0`

## Conventions

- **Commits** : conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`)
- **Type hints** : partout, `ty` enforced
- **Docstrings** : sur fonctions publiques et classes (peut être 1 ligne)
- **ruff** : config par défaut + `select = ["E", "F", "I", "N", "UP", "B", "SIM"]`
- **Tests** : pas de skip sauf justification écrite, pas de mock excessif (voir philosophie ci-dessus)

## Références

- **Codebase Rust de référence** : `/home/thenry/Projects/handy`, branche `feat/openrouter-transcription`. Fichiers à lire pour chaque port :
  - VAD : `src-tauri/src/audio_toolkit/vad/{silero,smoothed,mod}.rs` + `src-tauri/src/managers/audio.rs:120-130` (params prod)
  - Audio capture/resample : `src-tauri/src/audio_toolkit/audio/{recorder,resampler}.rs`
  - WAV : `src-tauri/src/audio_toolkit/audio/utils.rs`
  - OpenRouter : `src-tauri/src/llm_client/{remote,stt_body,catalog,error,parse}.rs`
- **Silero v4 ONNX contract** : <https://github.com/snakers4/silero-vad/wiki/FAQ> et le crate <https://github.com/cjpais/vad-rs>
- **soxr Python** : <https://github.com/dofuuz/python-soxr>
- **OpenRouter STT docs** : <https://openrouter.ai/docs/guides/overview/multimodal/stt>
- **Prior art (lecture obligatoire avant Step 8)** : <https://github.com/sevos/waystt> — Rust, SIGUSR1/USR2 toggle, archi quasi-identique
- **Review qui a produit ce plan** : [docs/review-plan-voxd.md](review-plan-voxd.md)
