# voxd

Headless voice-to-text daemon for Wayland/sway. Transcribes via OpenRouter's
`/audio/transcriptions` endpoint and injects the result with `wtype`.
Controlled by SIGUSR2 (toggle start/stop) — no GUI, no overlay, no tray.

## How it works

```
SIGUSR2 -> daemon -> sounddevice capture -> Silero VAD (trims silence)
       -> POST /audio/transcriptions @ OpenRouter
       -> wl-copy <text> + wtype <paste-keystroke>   (FIFO-serialized)
```

A single toggle binding (`Ctrl+Alt+Space`) starts recording; the next one stops
it, uploads the speech-trimmed WAV to OpenRouter, copies the transcription onto
the clipboard, and synthesises the paste keystroke for the focused window.

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
  [`atx/wtype#62`](https://github.com/atx/wtype/issues/62)). Instead it pops
  a `notify-send` reminder; the transcription is on the clipboard, just paste
  manually with `Ctrl+V` or `Ctrl+Shift+V`.
- **Unknown focus / non-sway compositor** (Hyprland, labwc, edge cases where
  `swaymsg` is absent or returns no focused container) — falls back to plain
  `Ctrl+V`, which works on most wlroots-based compositors.

No text is ever typed character by character — paste-via-keystroke avoids the
character-loss bugs in `wtype` ([`atx/wtype#46`](https://github.com/atx/wtype/issues/46)).

## Install

### System dependencies

`voxd` shells out to a few binaries and links against PortAudio at runtime.
Install these via your distro package manager:

| Tool          | Fedora                          | Debian / Ubuntu             | Arch                |
|---------------|---------------------------------|-----------------------------|---------------------|
| `wtype`       | `dnf install wtype`             | `apt install wtype`         | `pacman -S wtype`   |
| `wl-clipboard`| `dnf install wl-clipboard`      | `apt install wl-clipboard`  | `pacman -S wl-clipboard` |
| `libnotify`   | `dnf install libnotify`         | `apt install libnotify-bin` | `pacman -S libnotify` |
| PortAudio     | `dnf install portaudio`         | `apt install libportaudio2` | `pacman -S portaudio` |

### voxd itself

```
uv tool install voxd
```

(Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+. `uv tool install`
creates an isolated venv and drops the `voxd` entry point on your `$PATH`.)

## Setup

```
voxd setup
```

Walks through three things:

1. Prompts for your OpenRouter API key (echo-hidden) and writes it to
   `$XDG_CONFIG_HOME/voxd/config.toml` at mode `0600`.
2. Drops `~/.config/sway/config.d/voxd.conf` with the `exec voxd` line and
   the `Ctrl+Alt+Space` bindsym.
3. Ensures `~/.config/sway/config` sources `config.d/*` (idempotent).

`voxd setup` never spawns the daemon. After it returns, run:

```
swaymsg reload                 # picks up the new bindsym
voxd &                         # start the daemon in the background
```

Toggle recording with **Ctrl+Alt+Space**.

## Troubleshooting

```
voxd doctor
```

Prints a one-line verdict per dependency:

- `wtype` — required for the paste keystroke (critical).
- `wl-copy` — required to land the transcription on the clipboard before
  the paste keystroke fires; without it, `Ctrl+V` would paste the previous
  clipboard contents (critical).
- `notify-send` — used to nudge the user to paste manually on Xwayland
  apps (warning).
- `PortAudio` — required for mic capture (critical).
- `Mic access` — exercises an actual `InputStream` open (warning if denied).
- `Desktop` — warns when `$XDG_CURRENT_DESKTOP` is not `sway`.

Exit code is `0` iff every *critical* check passes.

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
Comments are preserved across edits made through the daemon (e.g. by
`voxd setup`).

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

`voxd` is a Python port of the audio pipeline from the Rust
[handy](https://github.com/cjpais/handy) project (branch
`feat/openrouter-transcription`). The wire contract (Silero v4 ONNX I/O,
OpenRouter request body, WAV encoding, retry policy) matches the Rust
reference bit-for-bit. See [`docs/plan-voxd.md`](docs/plan-voxd.md) for the
design, decision log, and TDD roadmap.

## License

MIT. See [LICENSE](LICENSE).
