# voxd

Headless voice-to-text daemon for Wayland/sway. Transcribes speech via OpenRouter
and injects the result into the focused window. No GUI, no overlay, no tray —
just a toggle bindsym.

## How it works

```
SIGUSR2 -> daemon -> sounddevice capture -> Silero VAD (trims silence)
       -> POST /audio/transcriptions @ OpenRouter
       -> wl-copy <text> + wtype <paste-keystroke>   (FIFO-serialized)
```

One `Ctrl+Alt+Space` starts recording; the next one stops it, uploads the
silence-trimmed WAV to OpenRouter, drops the transcription on the Wayland
clipboard, and synthesises the paste keystroke for the focused window.

## Tested on

voxd is developed and tested only on Fedora 43 (Sway spin). Other platforms are
not officially supported, but contributions are welcome — see
[Contributing](#contributing).

| Component       | Version                       |
|-----------------|-------------------------------|
| OS              | Fedora Linux 43 (Sway spin)   |
| Kernel          | `6.17.1-300.fc43.x86_64`      |
| Compositor      | sway 1.11                     |
| `wl-clipboard`  | 2.2.1                         |
| `notify-send`   | 0.8.8 (libnotify)             |
| PortAudio       | 19.7.0-2.fc43                 |
| Python          | 3.14.4 (system); voxd needs ≥ 3.11 |
| uv              | 0.9.28                        |

## Install

### System dependencies

voxd shells out to a few binaries and links against PortAudio at runtime.
Install these via your distro package manager:

| Tool          | Fedora                          | Debian / Ubuntu             | Arch                     |
|---------------|---------------------------------|-----------------------------|--------------------------|
| `wtype`       | `dnf install wtype`             | `apt install wtype`         | `pacman -S wtype`        |
| `wl-clipboard`| `dnf install wl-clipboard`      | `apt install wl-clipboard`  | `pacman -S wl-clipboard` |
| `libnotify`   | `dnf install libnotify`         | `apt install libnotify-bin` | `pacman -S libnotify`    |
| PortAudio     | `dnf install portaudio`         | `apt install libportaudio2` | `pacman -S portaudio`    |

### voxd

voxd is distributed as a [uv](https://docs.astral.sh/uv/) tool installed
directly from GitHub:

```
uv tool install git+https://github.com/lescientifik/voxd
```

`uv tool install` creates an isolated venv and drops the `voxd` entry point on
your `$PATH`. Requires Python 3.11+ (uv will fetch one if needed).

## Setup

### 1. Get an OpenRouter API key

Create one at [openrouter.ai/keys](https://openrouter.ai/keys). You will need
credits on your account — OpenRouter forwards transcription requests to the
underlying provider and bills per audio second.

### 2. Run `voxd setup`

```
voxd setup
```

The command is idempotent and never spawns the daemon. It does three things:

1. Prompts for your OpenRouter API key (echo-hidden) and writes it to
   `$XDG_CONFIG_HOME/voxd/config.toml` at mode `0600`.
2. Writes `~/.config/sway/config.d/voxd.conf` with an `exec voxd` line and a
   `bindsym Ctrl+Alt+Space exec pkill -USR2 voxd`.
3. Ensures `~/.config/sway/config` includes `config.d/*` (and copies the
   distro default `/etc/sway/config` first if no user config exists yet).

### 3. Reload sway

```
swaymsg reload
```

Sway picks up the new bindsym.

### 4. Start the daemon for the current session

```
swaymsg exec voxd
```

This launches voxd as a child of sway, so it survives the terminal it was
started from (unlike `voxd &`). At the next sway login, the `exec voxd` line
in the generated snippet starts the daemon automatically.

### 5. Toggle recording

Press `Ctrl+Alt+Space` to start, speak, then `Ctrl+Alt+Space` again to stop.
voxd uploads the audio, copies the transcription, and synthesises a paste
keystroke into the focused window.

## Troubleshooting

```
voxd doctor
```

Prints a one-line verdict per dependency:

- `wtype` — required for the paste keystroke (**critical**).
- `wl-copy` — required to land the transcription on the clipboard before the
  paste keystroke fires; without it `Ctrl+V` would paste the previous clipboard
  contents (**critical**).
- `notify-send` — used to nudge the user to paste manually on Xwayland apps
  (warning).
- `PortAudio` — required for mic capture (**critical**).
- `Mic access` — exercises an actual `InputStream` open (warning if denied).
- `Desktop` — warns when `$XDG_CURRENT_DESKTOP` is not `sway`.

Exit code is `0` iff every critical check passes.

## Config reference

`$XDG_CONFIG_HOME/voxd/config.toml`:

```toml
[openrouter]
api_key  = "sk-or-v1-…"                       # required
model    = "openai/whisper-large-v3-turbo"    # see voxd.transcribe.MODELS
language = "auto"                             # ISO 639-1 or "auto"
prompt   = ""                                 # optional hint forwarded to STT

[audio]
device   = ""                                 # empty = system default

[inject]
type      = true                              # invoke wtype (paste keystroke)
clipboard = true                              # invoke wl-copy
```

`inject.type = true` requires `inject.clipboard = true` — the paste keystroke
needs the transcription on the clipboard first. The daemon refuses to start
with `type=true, clipboard=false` (exit code `2`).

Edit by hand with `voxd config --edit` (opens `$EDITOR`, falls back to `nano`).
Comments are preserved across edits.

## How injection works

Injection runs in two steps: `wl-copy` puts the transcription on the Wayland
clipboard, then `wtype` synthesises the paste keystroke appropriate to the
focused window. voxd inspects `swaymsg -t get_tree` to pick the right combo:

- **Wayland-native terminal** (`foot`, `Alacritty`, `kitty`, `wezterm`,
  `ghostty`) — sends `Ctrl+Shift+V` (terminals reserve plain `Ctrl+V` for
  their own bindings).
- **Other Wayland-native app** (Firefox, Slack, VS Code Wayland, …) — sends
  `Ctrl+V`.
- **Xwayland app** (older apps, some Electron builds) — voxd does *not* send
  any keystroke (`wtype` cannot drive Xwayland windows, see
  [`atx/wtype#62`](https://github.com/atx/wtype/issues/62)). Instead it pops a
  `notify-send` reminder; the transcription is on the clipboard, just paste
  manually with `Ctrl+V` or `Ctrl+Shift+V`.
- **Unknown focus / non-sway compositor** (Hyprland, labwc, edge cases where
  `swaymsg` is absent or returns no focused container) — falls back to plain
  `Ctrl+V`, which works on most wlroots-based compositors.

No text is ever typed character by character — paste-via-keystroke avoids the
character-loss bugs in `wtype`
([`atx/wtype#46`](https://github.com/atx/wtype/issues/46)).

## Sub-commands

| Command                | Purpose                                                |
|------------------------|--------------------------------------------------------|
| `voxd`                 | Run the daemon (foreground).                           |
| `voxd setup`           | Interactive setup: config + sway snippet.              |
| `voxd doctor`          | Diagnose missing system dependencies.                  |
| `voxd config`          | Print the config file path.                            |
| `voxd config --edit`   | Open the config in `$EDITOR` (default `nano`).         |
| `pkill -USR2 voxd`     | Toggle recording (the sway bindsym does this for you). |

## Architecture

voxd is a Python port of the audio pipeline from the Rust
[handy](https://github.com/cjpais/handy) project (branch
`feat/openrouter-transcription`). The wire contract — Silero v4 ONNX I/O,
OpenRouter request body, WAV encoding, retry policy — matches the Rust
reference bit-for-bit. Runtime stack: Python ≥ 3.11, `sounddevice`,
`onnxruntime`, `soxr`, `httpx`, `tomlkit`.

See [`docs/plan-voxd.md`](docs/plan-voxd.md) for the full design and decision
log, and [`docs/plan-inject-refactor.md`](docs/plan-inject-refactor.md) for the
paste-via-keystroke refactor.

## Contributing

voxd is built and supported for Fedora 43 + sway only. PRs adapting it to
other distros, compositors, or STT providers are welcome but won't be
maintained by the author. File issues and PRs at
[github.com/lescientifik/voxd/issues](https://github.com/lescientifik/voxd/issues).

Dev stack: `uv`, `ruff`, `ty`, `pytest` — see [CLAUDE.md](CLAUDE.md) for
conventions.

## License

MIT. See [LICENSE](LICENSE).
