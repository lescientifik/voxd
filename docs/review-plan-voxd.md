---
description: Review adversariale du plan voxd — failles, écarts vis-à-vis du Rust de référence, et pièges de portage Python avec citations.
---

# Review adversariale — `plan-voxd.md`

Verdict d'ensemble : le plan est bien charpenté et l'idée d'un port direct des décisions Rust est saine. Mais plusieurs hypothèses doivent être ajustées avant de coder. Les findings sont classés par sévérité.

## CRITIQUE — à corriger avant d'écrire la première ligne

### C1. Silero v4 ONNX — contrat I/O à expliciter, et **NE PAS** prendre `silero-vad` PyPI

Le plan dit "Silero v4 a un state interne (h, c) à porter entre push_frame — vérifier en lisant `vad-rs`" en question ouverte. Réponse confirmée :

- v4 ONNX inputs : `input` (f32 `[1, N]`), `sr` (i64 `[1]`), `h` (f32 `[2,1,64]`), `c` (f32 `[2,1,64]`)
- v4 ONNX outputs : `output` (proba speech), `hn`, `cn` — à recycler comme `h`/`c` du tour suivant
- Init : `h = c = zeros([2,1,64], float32)`, `sr = 16000`
- v4 accepte 480 samples (30 ms) sans padding — c'est un format d'entraînement first-class. Le restriction "512 samples obligatoires" concerne v5, pas v4. Source : [snakers4/silero-vad FAQ](https://github.com/snakers4/silero-vad/wiki/FAQ), [cjpais/vad-rs](https://github.com/cjpais/vad-rs).

**Ne pas** utiliser le paquet PyPI `silero-vad` : il tire `torch` (~200 Mo), ne ship que v5 (512-strict), et son `VADIterator` n'a pas de chemin 30 ms. Le plan doit dire explicitement : appel onnxruntime brut, état porté manuellement, comme `vad-rs`.

### C2. Le `systemd --user` unit est partiellement faux

Tel que rédigé, `wtype` plantera car `WAYLAND_DISPLAY` n'est pas importé dans l'environnement du user manager. Trois corrections :

1. `WantedBy=default.target` ⇒ doit être `WantedBy=sway-session.target` (ou similaire), avec `PartOf=graphical-session.target` pour que le service s'arrête quand la session sway s'arrête. `After=graphical-session.target` seul n'est qu'un ordering, pas une dépendance. Source : [Arch wiki systemd/User — Xorg and Wayland sessions](https://wiki.archlinux.org/title/Systemd/User#Xorg_and_Wayland_sessions).
2. `voxd setup` doit s'assurer que `~/.config/sway/config` contient :
   ```
   exec systemctl --user import-environment WAYLAND_DISPLAY XDG_CURRENT_DESKTOP
   exec dbus-update-activation-environment --systemd WAYLAND_DISPLAY XDG_CURRENT_DESKTOP
   exec systemctl --user start sway-session.target
   ```
   Source : [sway wiki — Systemd integration](https://github.com/swaywm/sway/wiki/Systemd-integration).
3. `sway-session.target` n'existe pas par défaut. Soit voxd l'installe lui-même, soit on documente la dépendance à [alebastr/sway-systemd](https://github.com/alebastr/sway-systemd).

### C3. Sway n'inclut pas `~/.config/sway/config.d/*` par défaut

Le `config` shipped n'inclut que `/etc/sway/config.d/*`. Si l'utilisateur a copié `/etc/sway/config` vers `~/.config/sway/config`, la directive `include ~/.config/sway/config.d/*` doit y être ajoutée — sinon `voxd.conf` est ignoré silencieusement. `voxd setup` doit :

1. Vérifier l'existence et le contenu de `~/.config/sway/config`
2. Y ajouter la ligne `include ~/.config/sway/config.d/*` si absente
3. Sinon écrire le bindsym directement dans le main config

Source : [sway/config.in](https://github.com/swaywm/sway/blob/master/config.in).

### C4. float32 → int16 PCM : piège de portage NumPy

Le Rust fait `(sample * i16::MAX as f32) as i16` qui **sature** (sémantique `as` Rust). En Python, `(samples * 32767).astype(np.int16)` **enroule** : un échantillon de valeur 1.5 devient -32767 au lieu de 32767. Le plan ne flagge pas ce piège. Code correct :

```python
clipped = np.clip(samples, -1.0, 1.0)
pcm16 = (clipped * 32767.0).astype(np.int16)
```

## MAJEUR — design gaps

### M1. `scipy.signal.resample_poly` par chunk introduit des artefacts au bord

Le Rust utilise `rubato::FftFixedIn` qui est un resampler **streaming** (état interne entre `process()`). Le plan propose `scipy.signal.resample_poly` appelé par chunk de 30 ms. À chaque appel, le filtre FIR n'a aucun contexte du chunk précédent : ringing en début/fin de chaque chunk, alimente le VAD avec un signal bruité aux bordures de chunk.

Le plan flagge le problème (Q ouverte #2) mais le défère. C'est mandatory avant de pouvoir tester le VAD honnêtement. Options :

- `soxr` (PyPI `soxr`) — wrapper officiel libsoxr, supporte le streaming via `ResampleStream`
- `samplerate` (PyPI `samplerate`) — wrapper libsamplerate, streaming
- Buffer ≥1 s en amont puis resample, ré-émettre 30 ms — accepte une latence d'1 s en plus
- Polyphase FIR avec état porté à la main (peu code, mais redécoupe rubato)

### M2. Threading model asyncio + PortAudio callback + signal — non explicité

Le pseudocode `daemon.py` mélange trois mondes sans dire comment ils se croisent :

- Callback sounddevice tourne dans un thread PortAudio
- `signal.signal()` ne déclenche le handler que sur le main thread
- `asyncio.Queue` n'est pas thread-safe pour `put_nowait` depuis un autre thread

Le plan doit fixer :

1. Boucle asyncio sur le main thread
2. `loop.add_signal_handler(SIGUSR2, ...)` (pas `signal.signal`)
3. Callback PortAudio fait `loop.call_soon_threadsafe(queue.put_nowait, frame)` (ou utilise `janus.Queue`)
4. Toutes les mutations d'état dans des coroutines pour éviter les locks

### M3. Pidfile + SIGUSR2 — race et PID recycling

- `atexit` ne déclenche pas sur SIGKILL → pidfile orphelin après crash
- PID recyclé : `kill -USR2 <pid>` peut signaler un processus quelconque
- Pas de protection contre `voxd toggle` lancé en parallèle de `voxd setup`

Plus propre : unix socket à `$XDG_RUNTIME_DIR/voxd.sock`. Le socket meurt avec le processus, pas de stale state, et extensible (status, version, cancel à terme).

Si on garde le pidfile : avant `kill`, lire `/proc/<pid>/comm` et vérifier que c'est bien `voxd`.

### M4. Enregistrement vide envoyé au cloud

Le pseudocode plan pad `< 1 s` à `1.25 s`. Mais si `len(samples) == 0` (aucun speech détecté par VAD), il pad **0 samples** à 1.25 s de silence, et upload. Coût + transcription poubelle.

Le Rust handy évite ça : `if s_len < WHISPER_SAMPLE_RATE && s_len > 0 { pad }`. Voxd doit faire pareil et notifier "aucune voix détectée" si 0 samples.

### M5. UX path d'erreur du upload — l'indicateur peut rester collé

Si `transcribe()` lève (auth, réseau, 5×500), le worker prend l'exception. Le plan ne dit pas comment l'indicateur revient à un état propre. Risque : notification "… transcribing" qui reste à l'écran pour toujours.

À spécifier explicitement : chaque chemin de sortie du worker (succès, échec, cancel) clear l'indicateur et notifie l'utilisateur.

### M6. `prompt` non documenté côté OpenRouter STT

Le schéma documenté `/audio/transcriptions` chez OpenRouter liste : `model`, `input_audio`, `language`, `temperature`, `provider`. **Pas** `prompt`. Voir [OpenRouter STT docs](https://openrouter.ai/docs/guides/overview/multimodal/stt). Le Rust handy envoie quand même `prompt` — vraisemblablement forwardé vers OpenAI mais comportement non garanti par OpenRouter, et déjà ignoré par chirp (`supports_prompt=false`).

À documenter dans le plan et dans la config : "feature non-officielle, support dépend du provider sous-jacent".

## MINEUR

### m1. `voxd doctor` et l'import sounddevice

`import sounddevice` lève une `OSError` au moment de l'import si la lib PortAudio shared n'est pas installée. `voxd doctor` doit wrapper l'import dans try/except sinon il crash avant de pouvoir reporter quoi que ce soit.

### m2. Tests `test_daemon.py` — simulation SIGUSR2

`os.kill(os.getpid(), SIGUSR2)` en pytest est racy et perturbe d'autres tests. Mieux : extraire la logique du handler dans une coroutine pure `async def handle_toggle()` testable directement, et tester en bout `add_signal_handler` qu'avec un seul smoke test.

### m3. `voxd setup` — non idempotence cachée

Le plan dit "écritures idempotentes". Il faut détailler par fichier :

- `config.toml` existant avec clé API → ne pas écraser, fusionner par champ
- `voxd.service` existant → diff + demande confirmation si contenu différent
- Sway snippet existant → idem
- Sway main config : ajouter `include` line une seule fois (idempotent)

### m4. Hotplug / déconnexion device mid-recording

USB ou Bluetooth débranché en cours d'enregistrement → PortAudio lève dans le callback thread. Pas traité. Au minimum : try/except autour du callback, retour idle propre, notify-send "audio device disconnected".

### m5. Compositor non-sway → `wtype` muet sans message

`wtype` ne fonctionne pas sous GNOME-Wayland ([wtype#29](https://github.com/atx/wtype/issues/29)) et est cassé sous Plasma 6 ([wtype#53](https://github.com/atx/wtype/issues/53)). Le plan cible sway. OK. Mais `voxd doctor` doit checker `$XDG_CURRENT_DESKTOP` et avertir si ≠ sway/wlroots, pour éviter l'échec silencieux.

### m6. Fallback paste pour Unicode edge cases

D'après [wtype#45](https://github.com/atx/wtype/issues/45) et la prior art (voxtype, hyprvoice), certaines apps Electron/GTK4 rejettent les keysyms synthétisés pour caractères hors BMP / combinaisons. Fallback éprouvé :

```bash
wl-copy "$text" && wtype -M ctrl -P v -m ctrl
```

À envisager comme `inject.method = "wtype" | "paste"` dans la config (défaut wtype).

## Prior art à lire avant de coder

Le projet le plus proche architecturalement : **[sevos/waystt](https://github.com/sevos/waystt)** — Rust, SIGUSR1 transcribe + SIGUSR2 cancel, PipeWire + OpenAI Whisper API, conçu pour `bindsym → pkill --signal`. Lire son signal handler et sa sémantique "second signal pendant upload = cancel partial". Vraiment proche.

Autres références : [LeonardoTrapani/hyprvoice](https://github.com/LeonardoTrapani/hyprvoice) (Go, 26 backends STT, fallbacks d'injection), [peteonrails/voxtype](https://github.com/peteonrails/voxtype) (Python, push-to-talk).

## Synthèse des points qui restent vraiment ouverts (décisions design)

1. **Sway session integration** — dépendre de `alebastr/sway-systemd`, ou hand-roll dans `voxd setup` ?
2. **Resampler streaming** — `soxr`, `samplerate`, ou polyphase FIR maison ?
3. **Cancel** — le plan dit no. Mais waystt et handy l'ont. Revoir ou trancher définitif ?
4. **Inject fallback** — wtype-only, ou paste-via-wl-copy en fallback configurable ?
5. **IPC daemon/CLI** — pidfile+signal, ou unix socket ?
