"""Tests for the voxd config module (TOML load/save with comment preservation)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from voxd import config as cfg_module
from voxd.config import (
    AudioConfig,
    Config,
    InjectConfig,
    OpenRouterConfig,
    default_path,
    load,
    save,
)


def _make_default_config() -> Config:
    """Return a Config whose sections are entirely default values."""
    return Config(
        openrouter=OpenRouterConfig(),
        audio=AudioConfig(),
        inject=InjectConfig(),
    )


def test_load_with_all_defaults(tmp_path: Path) -> None:
    """Loading a non-existent config file returns all default values, no crash."""
    missing = tmp_path / "does-not-exist.toml"
    loaded = load(missing)

    assert loaded == _make_default_config()
    # And the defaults are exactly the documented values
    assert loaded.openrouter.api_key == ""
    assert loaded.openrouter.model == "openai/whisper-large-v3-turbo"
    assert loaded.openrouter.language == "auto"
    assert loaded.openrouter.prompt == ""
    assert loaded.audio.device == ""
    assert loaded.inject.type is True
    assert loaded.inject.clipboard is True


def test_roundtrip_preserves_comments(tmp_path: Path) -> None:
    """A comment present in the on-disk TOML survives load -> save."""
    path = tmp_path / "config.toml"
    path.write_text(
        "# top-of-file comment that must survive\n"
        "\n"
        "[openrouter]\n"
        "# explains why this key is set\n"
        'api_key = "sk-test"\n'
        'model = "openai/whisper-large-v3-turbo"\n'
        'language = "auto"\n'
        'prompt = ""\n'
        "\n"
        "[audio]\n"
        'device = ""\n'
        "\n"
        "[inject]\n"
        "type = true\n"
        "clipboard = true\n",
        encoding="utf-8",
    )

    loaded = load(path)
    save(loaded, path)

    rewritten = path.read_text(encoding="utf-8")
    assert "# top-of-file comment that must survive" in rewritten
    assert "# explains why this key is set" in rewritten


def test_file_mode_is_0600_on_create(tmp_path: Path) -> None:
    """Saving to a fresh path creates the file with 0600 permissions."""
    path = tmp_path / "subdir" / "config.toml"

    save(_make_default_config(), path)

    assert path.exists()
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_load_with_missing_field_falls_back(tmp_path: Path) -> None:
    """A config with an [openrouter] section but no `prompt` key uses the default."""
    path = tmp_path / "config.toml"
    path.write_text(
        "[openrouter]\n"
        'api_key = "sk-test"\n'
        'model = "openai/whisper-large-v3-turbo"\n'
        'language = "fr"\n',
        encoding="utf-8",
    )

    loaded = load(path)

    assert loaded.openrouter.api_key == "sk-test"
    assert loaded.openrouter.language == "fr"
    assert loaded.openrouter.prompt == ""  # default kicked in
    # And the entirely-missing [audio] / [inject] sections fall back too
    assert loaded.audio == AudioConfig()
    assert loaded.inject == InjectConfig()


def test_load_with_utf8_prompt(tmp_path: Path) -> None:
    """A UTF-8 prompt with accents round-trips byte-for-byte (write -> load)."""
    path = tmp_path / "config.toml"
    prompt = "café, naïve, façon"

    initial = Config(
        openrouter=OpenRouterConfig(api_key="sk-test", prompt=prompt),
        audio=AudioConfig(),
        inject=InjectConfig(),
    )
    save(initial, path)

    loaded = load(path)
    assert loaded.openrouter.prompt == prompt


# --- Additional reasonable tests --------------------------------------------


def test_default_path_respects_xdg_config_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """default_path() honours $XDG_CONFIG_HOME and falls back to ~/.config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert default_path() == tmp_path / "xdg" / "voxd" / "config.toml"

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert default_path() == tmp_path / "home" / ".config" / "voxd" / "config.toml"


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    """save() creates missing parent directories rather than crashing."""
    nested = tmp_path / "a" / "b" / "c" / "config.toml"

    save(_make_default_config(), nested)

    assert nested.exists()
    # Parent dir restricted to user, per XDG good-practice
    parent_mode = nested.parent.stat().st_mode & 0o777
    assert parent_mode & 0o077 == 0, (
        f"parent dir should not be group/other accessible, got {oct(parent_mode)}"
    )


def test_load_uses_default_path_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling load() with no argument uses default_path()."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    # No config file at default path -> all defaults, no crash
    loaded = load()
    assert loaded == _make_default_config()


def test_roundtrip_preserves_user_added_key(tmp_path: Path) -> None:
    """A user-added unknown key in the file is preserved by load -> save.

    Rationale: tomlkit roundtrip is the whole point; we must not silently drop
    fields we don't model yet.
    """
    path = tmp_path / "config.toml"
    path.write_text(
        "[openrouter]\n"
        'api_key = "sk-test"\n'
        'experimental = "yes"\n',
        encoding="utf-8",
    )

    loaded = load(path)
    save(loaded, path)

    rewritten = path.read_text(encoding="utf-8")
    assert 'experimental = "yes"' in rewritten


def test_module_exposes_public_api() -> None:
    """The module surface is the one announced in the plan."""
    # Sanity-check, also gives `ty` something to chew on.
    assert callable(cfg_module.load)
    assert callable(cfg_module.save)
    assert callable(cfg_module.default_path)
    # frozen dataclasses are immutable: attempting mutation must raise
    from dataclasses import FrozenInstanceError

    c = OpenRouterConfig()
    with pytest.raises(FrozenInstanceError):
        c.api_key = "nope"  # type: ignore[misc]


def test_xdg_config_home_empty_falls_back_to_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty XDG_CONFIG_HOME is treated as unset (per XDG spec)."""
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert default_path() == tmp_path / "home" / ".config" / "voxd" / "config.toml"


def test_save_overwrites_existing_file_preserving_mode(tmp_path: Path) -> None:
    """save() onto an existing file keeps it at 0600 (not loosened)."""
    path = tmp_path / "config.toml"
    save(_make_default_config(), path)
    # Tamper: open up perms, then re-save
    os.chmod(path, 0o644)
    save(_make_default_config(), path)
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600
