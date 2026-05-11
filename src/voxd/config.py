"""voxd configuration: TOML load/save with comment preservation.

Reading uses the stdlib ``tomllib`` (Python 3.11+); writing uses ``tomlkit``
so user-authored comments survive a round-trip when the file exists. When
the file does not exist, ``save()`` builds a fresh ``tomlkit`` document from
the dataclass values.

The on-disk file lives at ``$XDG_CONFIG_HOME/voxd/config.toml`` (falling
back to ``~/.config/voxd/config.toml``) and is created with mode ``0600``.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import tomlkit
import tomlkit.items
from tomlkit import TOMLDocument

# --- Public dataclasses -----------------------------------------------------


@dataclass(frozen=True)
class OpenRouterConfig:
    """OpenRouter transcription endpoint settings."""

    api_key: str = ""
    model: str = "openai/whisper-large-v3-turbo"
    language: str = "auto"
    prompt: str = ""


@dataclass(frozen=True)
class AudioConfig:
    """Audio capture device selection.

    Empty string means "let PortAudio pick the system default".
    """

    device: str = ""


@dataclass(frozen=True)
class InjectConfig:
    """Toggle which injection channels are used after transcription."""

    type: bool = True  # wtype
    clipboard: bool = True  # wl-copy


@dataclass(frozen=True)
class Config:
    """Top-level voxd configuration, composed of three sections."""

    openrouter: OpenRouterConfig
    audio: AudioConfig
    inject: InjectConfig


# --- Path resolution --------------------------------------------------------


def default_path() -> Path:
    """Return the canonical config file path, honouring ``$XDG_CONFIG_HOME``.

    Per the XDG Base Directory specification, an unset *or empty*
    ``XDG_CONFIG_HOME`` falls back to ``$HOME/.config``.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "voxd" / "config.toml"


# --- Load -------------------------------------------------------------------


def _coerce_str(value: object, default: str) -> str:
    """Return ``value`` as a str, falling back if absent or wrong type."""
    return value if isinstance(value, str) else default


def _coerce_bool(value: object, default: bool) -> bool:
    """Return ``value`` as a bool, falling back if absent or wrong type."""
    return value if isinstance(value, bool) else default


def load(path: Path | None = None) -> Config:
    """Load the voxd config from ``path`` (or ``default_path()`` if ``None``).

    Missing file -> returns an all-defaults ``Config``. Missing sections or
    fields fall back to their defaults individually; we never crash on a
    partially-written config. Unknown keys are ignored (but preserved on
    write via ``tomlkit``).
    """
    target = path if path is not None else default_path()

    defaults = Config(
        openrouter=OpenRouterConfig(),
        audio=AudioConfig(),
        inject=InjectConfig(),
    )

    if not target.exists():
        return defaults

    raw = tomllib.loads(target.read_text(encoding="utf-8"))

    or_section = raw.get("openrouter", {})
    audio_section = raw.get("audio", {})
    inject_section = raw.get("inject", {})

    openrouter = OpenRouterConfig(
        api_key=_coerce_str(or_section.get("api_key"), defaults.openrouter.api_key),
        model=_coerce_str(or_section.get("model"), defaults.openrouter.model),
        language=_coerce_str(or_section.get("language"), defaults.openrouter.language),
        prompt=_coerce_str(or_section.get("prompt"), defaults.openrouter.prompt),
    )
    audio = AudioConfig(
        device=_coerce_str(audio_section.get("device"), defaults.audio.device),
    )
    inject = InjectConfig(
        type=_coerce_bool(inject_section.get("type"), defaults.inject.type),
        clipboard=_coerce_bool(inject_section.get("clipboard"), defaults.inject.clipboard),
    )
    return replace(defaults, openrouter=openrouter, audio=audio, inject=inject)


# --- Save -------------------------------------------------------------------


def _section_doc(doc: TOMLDocument, name: str) -> tomlkit.items.Table:
    """Return the named table from ``doc``, creating it if missing."""
    existing = doc.get(name)
    if existing is None:
        table = tomlkit.table()
        doc.add(name, table)
        return table
    # tomlkit stores tables as items.Table; trust it.
    return existing  # type: ignore[return-value]


def _set_field(table: tomlkit.items.Table, key: str, value: str | bool) -> None:
    """Set ``table[key] = value``, replacing or inserting as needed."""
    if key in table:
        table[key] = value
    else:
        table.add(key, value)


def _build_document(cfg: Config, existing: TOMLDocument | None) -> TOMLDocument:
    """Return a tomlkit document reflecting ``cfg``.

    When ``existing`` is provided, fields are updated in place so comments
    and unknown user keys survive. Otherwise, a fresh document is built.
    """
    doc = existing if existing is not None else tomlkit.document()

    openrouter = _section_doc(doc, "openrouter")
    _set_field(openrouter, "api_key", cfg.openrouter.api_key)
    _set_field(openrouter, "model", cfg.openrouter.model)
    _set_field(openrouter, "language", cfg.openrouter.language)
    _set_field(openrouter, "prompt", cfg.openrouter.prompt)

    audio = _section_doc(doc, "audio")
    _set_field(audio, "device", cfg.audio.device)

    inject = _section_doc(doc, "inject")
    _set_field(inject, "type", cfg.inject.type)
    _set_field(inject, "clipboard", cfg.inject.clipboard)

    return doc


def save(cfg: Config, path: Path | None = None) -> None:
    """Write ``cfg`` to ``path`` (or ``default_path()`` if ``None``).

    Preserves comments and unknown keys when the target file already exists.
    The parent directory is created with mode ``0700`` if missing, and the
    file itself is forced to mode ``0600`` (whether newly created or not).
    The write is atomic: contents land in a temp file in the same directory
    then ``os.replace`` swaps it in.
    """
    target = path if path is not None else default_path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    existing_doc: TOMLDocument | None = None
    if target.exists():
        existing_doc = tomlkit.parse(target.read_text(encoding="utf-8"))

    doc = _build_document(cfg, existing_doc)
    serialized = tomlkit.dumps(doc)

    # Atomic write via temp file in the same directory, then chmod to 0600.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".voxd-config-", suffix=".toml.tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
    except Exception:
        # Best-effort cleanup; do not mask the original error.
        tmp_path.unlink(missing_ok=True)
        raise
    # os.replace preserves the destination's inode permissions on some
    # filesystems but here we've replaced via rename of a 0600 file, so the
    # result is 0600. Make it explicit just in case (re-chmod is cheap).
    os.chmod(target, 0o600)


__all__ = [
    "AudioConfig",
    "Config",
    "InjectConfig",
    "OpenRouterConfig",
    "default_path",
    "load",
    "save",
]
