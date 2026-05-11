"""Tests for ``voxd setup`` — interactive config + sway snippet generator.

The setup command is pure file editing: it prompts for an OpenRouter API key,
writes ``$XDG_CONFIG_HOME/voxd/config.toml`` (mode 0600), drops a sway snippet
at ``~/.config/sway/config.d/voxd.conf``, ensures the main ``~/.config/sway/config``
contains an ``include`` directive for ``config.d/*``, and prints restart
instructions. It must NEVER spawn the voxd daemon — restart is the user's job.

These tests pin every observable behaviour from the plan's Step 10 spec:

* fresh-install behaviour (config + snippet + include line + mode bits)
* idempotence (run twice -> same end state, no duplicates)
* non-destructive overwrite (existing API key preserved unless user confirms)
* user-authored comments survive
* no subprocess spawn for ``voxd``
* final restart message guides the user
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from voxd import setup as setup_mod

# ---------------------------------------------------------------------------
# Common fixture: isolate HOME / XDG so we don't touch the real user config.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point ``$HOME`` at ``tmp_path`` and clear ``$XDG_CONFIG_HOME``.

    The config module resolves to ``$XDG_CONFIG_HOME/voxd/config.toml`` when
    set, otherwise ``$HOME/.config/voxd/config.toml``. Clearing XDG_CONFIG_HOME
    keeps the test path predictable: ``tmp_path/.config/...``.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    yield tmp_path


def _set_inputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str = "sk-or-v1-test",
    confirm: str = "",
) -> None:
    """Patch ``getpass.getpass`` (returns ``api_key``) and ``input`` (returns ``confirm``)."""
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": api_key)
    monkeypatch.setattr("builtins.input", lambda _prompt="": confirm)


# ---------------------------------------------------------------------------
# 1. Fresh install — config file, mode, content
# ---------------------------------------------------------------------------


def test_setup_creates_config_with_provided_api_key(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """getpass returns a key -> config.toml exists, mode 0600, contains the key."""
    _set_inputs(monkeypatch, api_key="sk-or-v1-test")

    rc = setup_mod.main()

    assert rc == 0
    config_path = isolated_home / ".config" / "voxd" / "config.toml"
    assert config_path.exists()
    mode = config_path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
    contents = config_path.read_text(encoding="utf-8")
    assert "sk-or-v1-test" in contents


# ---------------------------------------------------------------------------
# 2. Sway snippet contents
# ---------------------------------------------------------------------------


def test_setup_writes_sway_snippet(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sway snippet contains ``exec voxd`` AND the Ctrl+Alt+Space bindsym."""
    _set_inputs(monkeypatch)

    setup_mod.main()

    snippet = isolated_home / ".config" / "sway" / "config.d" / "voxd.conf"
    assert snippet.exists()
    body = snippet.read_text(encoding="utf-8")
    assert "exec voxd" in body
    assert "bindsym" in body
    assert "Ctrl+Alt+Space" in body
    assert "pkill -USR2 voxd" in body


# ---------------------------------------------------------------------------
# 3. Include line is added when missing
# ---------------------------------------------------------------------------


def test_setup_adds_include_line_if_missing(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sway main config without the include directive gets one appended."""
    sway_dir = isolated_home / ".config" / "sway"
    sway_dir.mkdir(parents=True)
    main_config = sway_dir / "config"
    main_config.write_text(
        "# my sway config\nset $mod Mod4\n", encoding="utf-8"
    )

    _set_inputs(monkeypatch)
    setup_mod.main()

    contents = main_config.read_text(encoding="utf-8")
    # The pre-existing content is preserved
    assert "set $mod Mod4" in contents
    # The include line is now present
    assert "include ~/.config/sway/config.d/*" in contents


# ---------------------------------------------------------------------------
# 4. Idempotence: include line not duplicated on re-run
# ---------------------------------------------------------------------------


def test_setup_does_not_duplicate_include_line(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running setup twice keeps the include line exactly once."""
    _set_inputs(monkeypatch)

    setup_mod.main()
    setup_mod.main()

    main_config = isolated_home / ".config" / "sway" / "config"
    contents = main_config.read_text(encoding="utf-8")
    assert contents.count("include ~/.config/sway/config.d/*") == 1


# ---------------------------------------------------------------------------
# 5. User comments in the existing config are preserved
# ---------------------------------------------------------------------------


def test_setup_preserves_existing_config_comments(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-existing comments in the user's TOML survive setup."""
    config_path = isolated_home / ".config" / "voxd" / "config.toml"
    config_path.parent.mkdir(parents=True, mode=0o700)
    config_path.write_text(
        "# I wrote this comment myself\n"
        "[openrouter]\n"
        "# and this one too\n"
        'api_key = ""\n'
        'model = "openai/whisper-large-v3-turbo"\n'
        'language = "auto"\n'
        'prompt = ""\n'
        "\n[audio]\ndevice = \"\"\n"
        "\n[inject]\ntype = true\nclipboard = true\n",
        encoding="utf-8",
    )
    os.chmod(config_path, 0o600)

    _set_inputs(monkeypatch, api_key="sk-or-v1-newkey")
    setup_mod.main()

    body = config_path.read_text(encoding="utf-8")
    assert "# I wrote this comment myself" in body
    assert "# and this one too" in body
    assert "sk-or-v1-newkey" in body


# ---------------------------------------------------------------------------
# 6. Existing non-empty API key is NOT overwritten without confirmation
# ---------------------------------------------------------------------------


def test_setup_does_not_overwrite_existing_api_key_without_confirm(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a key is already set and user declines, the old key survives."""
    config_path = isolated_home / ".config" / "voxd" / "config.toml"
    config_path.parent.mkdir(parents=True, mode=0o700)
    config_path.write_text(
        "[openrouter]\n"
        'api_key = "sk-or-v1-EXISTING"\n'
        'model = "openai/whisper-large-v3-turbo"\n'
        'language = "auto"\n'
        'prompt = ""\n'
        "[audio]\ndevice = \"\"\n"
        "[inject]\ntype = true\nclipboard = true\n",
        encoding="utf-8",
    )
    os.chmod(config_path, 0o600)

    # User answers "" (i.e. accept default = No) to the overwrite prompt.
    # getpass MUST NOT be called when the user declines — guard with a
    # function that fails loudly if called.
    def _no_getpass(_prompt: str = "") -> str:
        raise AssertionError(
            "getpass should not be called when user declines overwrite"
        )

    monkeypatch.setattr("getpass.getpass", _no_getpass)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")  # default No

    setup_mod.main()

    body = config_path.read_text(encoding="utf-8")
    assert "sk-or-v1-EXISTING" in body


# ---------------------------------------------------------------------------
# 7. The setup never tries to spawn voxd itself
# ---------------------------------------------------------------------------


def test_setup_does_not_spawn_voxd(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No subprocess.run / Popen / os.system call should target ``voxd``."""
    import subprocess

    spawn_log: list[list[str]] = []

    def _trip(*args: object, **kwargs: object) -> None:
        # Accept argv positional or via kwargs; flatten what we got
        argv_obj: object = args[0] if args else kwargs.get("args", [])
        argv = list(argv_obj) if not isinstance(argv_obj, (str, bytes)) else [argv_obj]
        spawn_log.append([str(a) for a in argv])
        # Pretend success — but also fail if voxd appears in argv0.
        return None

    monkeypatch.setattr(subprocess, "run", _trip)
    monkeypatch.setattr(subprocess, "Popen", _trip)
    monkeypatch.setattr(os, "system", lambda cmd: spawn_log.append(["sh", "-c", str(cmd)]) or 0)

    _set_inputs(monkeypatch)
    setup_mod.main()

    for argv in spawn_log:
        if not argv:
            continue
        argv0 = Path(argv[0]).name
        assert argv0 != "voxd", f"setup spawned voxd via {argv!r}"
        # Defensive: also reject anyone sneaking it in via `sh -c ...`
        joined = " ".join(argv)
        assert "voxd &" not in joined and "exec voxd" not in joined, (
            f"setup contains a spawn invocation: {joined!r}"
        )


# ---------------------------------------------------------------------------
# 8. Final message guides the user to restart manually
# ---------------------------------------------------------------------------


def test_setup_prints_final_message(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The terminal output instructs the user to run ``pkill voxd && voxd``."""
    _set_inputs(monkeypatch)

    setup_mod.main()

    captured = capsys.readouterr().out
    assert "pkill voxd" in captured
    assert "voxd" in captured
    # The plan suggests "pkill voxd && voxd" — we accept any phrasing that
    # mentions both pkill and a fresh voxd invocation on the same screen.
    assert "swaymsg reload" in captured  # also told to reload sway


# ---------------------------------------------------------------------------
# Bonus: when the sway main config is missing entirely, we create it with
# just the include directive (and a comment header).
# ---------------------------------------------------------------------------


def test_setup_creates_sway_main_config_when_missing(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No sway main config -> create one with just the include line + comment."""
    _set_inputs(monkeypatch)

    setup_mod.main()

    main_config = isolated_home / ".config" / "sway" / "config"
    assert main_config.exists()
    body = main_config.read_text(encoding="utf-8")
    assert "include ~/.config/sway/config.d/*" in body
    # We expect at least one comment line explaining the file
    assert body.lstrip().startswith("#"), (
        "freshly created main sway config should start with a comment"
    )


# ---------------------------------------------------------------------------
# Bonus: snippet is not rewritten when content is already up to date.
# (Pure idempotence at the file level: mtime should not change on re-run.)
# ---------------------------------------------------------------------------


def test_setup_does_not_rewrite_snippet_when_already_correct(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second run with identical content should not touch the snippet file."""
    _set_inputs(monkeypatch)

    setup_mod.main()
    snippet = isolated_home / ".config" / "sway" / "config.d" / "voxd.conf"
    first_mtime = snippet.stat().st_mtime_ns

    # Force a different observable mtime if a write happens.
    os.utime(snippet, ns=(first_mtime - 1_000_000_000, first_mtime - 1_000_000_000))
    before = snippet.stat().st_mtime_ns

    setup_mod.main()
    after = snippet.stat().st_mtime_ns
    assert after == before, "snippet was rewritten even though content was identical"
