"""Tests for the voxd CLI dispatcher.

Step 9 wires :func:`voxd.cli.main` to ``argparse`` subparsers:

* no args → run the daemon
* ``setup`` → delegate to :func:`voxd.setup.main`
* ``doctor`` → delegate to :func:`voxd.doctor.main`
* ``config`` → print the resolved config path
* ``config --edit`` → ``$EDITOR <path>``
* ``--version`` → print the version
* unknown command → argparse error / help

The daemon path is exercised by monkey-patching :meth:`voxd.daemon.Daemon.run`
to a stubbed coroutine; that keeps the test offline and synchronous.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from voxd import cli

# ---------------------------------------------------------------------------
# --version / unknown commands
# ---------------------------------------------------------------------------


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    """``voxd --version`` prints ``voxd <version>`` and exits 0."""
    from voxd import __version__

    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert f"voxd {__version__}" in combined


def test_unknown_command_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown subcommand exits non-zero with an error mentioning the command."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["nope"])
    # argparse uses code 2 for usage errors.
    assert exc.value.code != 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # argparse prints the offending token and usage.
    assert "nope" in combined or "usage" in combined.lower()


# ---------------------------------------------------------------------------
# no args → daemon
# ---------------------------------------------------------------------------


def test_no_args_starts_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no subcommand, ``main`` constructs a :class:`Daemon` and runs it.

    We monkey-patch ``Daemon.run`` (the async entry point) so the test stays
    fast and offline, and we patch ``config.load`` to skip filesystem I/O.
    Asserts that ``run`` was awaited exactly once.
    """
    from voxd import config as config_mod
    from voxd import daemon as daemon_mod

    cfg = config_mod.Config(
        openrouter=config_mod.OpenRouterConfig(api_key="sk-test"),
        audio=config_mod.AudioConfig(),
        inject=config_mod.InjectConfig(),
    )
    monkeypatch.setattr(config_mod, "load", lambda *a, **kw: cfg)

    run_calls: list[int] = []

    async def fake_run(self: Any) -> None:  # noqa: ARG001
        run_calls.append(1)

    monkeypatch.setattr(daemon_mod.Daemon, "run", fake_run, raising=True)

    rc = cli.main([])
    assert rc == 0
    assert run_calls == [1]


# ---------------------------------------------------------------------------
# setup / doctor subcommands
# ---------------------------------------------------------------------------


def test_setup_dispatches_to_setup_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """``voxd setup`` forwards to :func:`voxd.setup.main` and returns its code."""
    import voxd.setup as setup_mod

    calls: list[int] = []

    def fake_main() -> int:
        calls.append(1)
        return 42

    monkeypatch.setattr(setup_mod, "main", fake_main, raising=True)
    rc = cli.main(["setup"])
    assert rc == 42
    assert calls == [1]


def test_doctor_dispatches_to_doctor_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """``voxd doctor`` forwards to :func:`voxd.doctor.main` and returns its code."""
    import voxd.doctor as doctor_mod

    calls: list[int] = []

    def fake_main() -> int:
        calls.append(1)
        return 7

    monkeypatch.setattr(doctor_mod, "main", fake_main, raising=True)
    rc = cli.main(["doctor"])
    assert rc == 7
    assert calls == [1]


# ---------------------------------------------------------------------------
# config subcommand
# ---------------------------------------------------------------------------


def test_config_prints_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``voxd config`` (no flag) prints the resolved config path on stdout."""
    expected = tmp_path / "voxd" / "config.toml"
    monkeypatch.setattr(cli.config, "default_path", lambda: expected)

    rc = cli.main(["config"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == str(expected)


def test_config_edit_opens_editor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``voxd config --edit`` invokes ``$EDITOR <path>`` via subprocess.run.

    Captures the argv passed to ``subprocess.run`` and asserts both entries
    (editor binary + config path).
    """
    expected_path = tmp_path / "voxd" / "config.toml"
    monkeypatch.setattr(cli.config, "default_path", lambda: expected_path)
    monkeypatch.setenv("EDITOR", "vim")

    captured: dict[str, Any] = {}

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = list(args)
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    rc = cli.main(["config", "--edit"])
    assert rc == 0
    assert captured["argv"] == ["vim", str(expected_path)]


def test_config_edit_defaults_to_nano_when_editor_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ``$EDITOR`` is unset, ``config --edit`` falls back to ``nano``."""
    expected_path = tmp_path / "voxd" / "config.toml"
    monkeypatch.setattr(cli.config, "default_path", lambda: expected_path)
    monkeypatch.delenv("EDITOR", raising=False)

    captured: dict[str, Any] = {}

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = list(args)
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    rc = cli.main(["config", "--edit"])
    assert rc == 0
    assert captured["argv"] == ["nano", str(expected_path)]


# ---------------------------------------------------------------------------
# no-args daemon path: missing API key is fail-fast
# ---------------------------------------------------------------------------


def test_no_args_with_missing_api_key_fails_fast(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the config has no ``api_key``, the daemon path exits non-zero.

    The error message must point the user at ``voxd setup`` so they know
    how to recover.
    """
    from voxd import config as config_mod
    from voxd import daemon as daemon_mod

    cfg = config_mod.Config(
        openrouter=config_mod.OpenRouterConfig(api_key=""),
        audio=config_mod.AudioConfig(),
        inject=config_mod.InjectConfig(),
    )
    monkeypatch.setattr(config_mod, "load", lambda *a, **kw: cfg)

    run_called: list[int] = []

    async def fake_run(self: Any) -> None:  # noqa: ARG001
        run_called.append(1)

    monkeypatch.setattr(daemon_mod.Daemon, "run", fake_run, raising=True)

    rc = cli.main([])
    assert rc != 0
    assert run_called == []
    err = capsys.readouterr().err
    assert "voxd setup" in err


def test_cli_invalid_inject_config_exits_2_with_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``inject.type=true && clipboard=false`` exits 2 with a clear message.

    The daemon's ``__init__`` raises ``ValueError`` on this config; the CLI
    must catch it, print to stderr (mentioning ``clipboard`` and the config
    file path so the user can fix it), and exit 2 — distinct from 1 (runtime)
    and from argparse's own 2 (usage error).
    """
    from voxd import config as config_mod
    from voxd import daemon as daemon_mod

    cfg = config_mod.Config(
        openrouter=config_mod.OpenRouterConfig(api_key="sk-test"),
        audio=config_mod.AudioConfig(),
        inject=config_mod.InjectConfig(type=True, clipboard=False),
    )
    monkeypatch.setattr(config_mod, "load", lambda *a, **kw: cfg)

    expected_path = tmp_path / "voxd" / "config.toml"
    monkeypatch.setattr(config_mod, "default_path", lambda: expected_path)

    run_called: list[int] = []

    async def fake_run(self: Any) -> None:  # noqa: ARG001
        run_called.append(1)

    monkeypatch.setattr(daemon_mod.Daemon, "run", fake_run, raising=True)

    rc = cli.main([])
    assert rc == 2, f"expected exit code 2 for invalid config, got {rc}"
    assert run_called == [], "Daemon.run must not be called when validation fails"
    err = capsys.readouterr().err
    assert "clipboard" in err, f"stderr should mention 'clipboard'; got: {err!r}"
    assert "config.toml" in err or str(expected_path) in err, (
        f"stderr should point at the config file; got: {err!r}"
    )
