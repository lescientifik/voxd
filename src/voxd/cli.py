"""Command-line entry point for voxd.

Dispatches the top-level executable to one of three subcommands or, when no
subcommand is given, runs the daemon in the foreground.

Subcommands:

* ``setup``   — interactive: prompt for API key, write config + sway snippet
                (delegates to :func:`voxd.setup.main`).
* ``doctor``  — diagnose missing system dependencies
                (delegates to :func:`voxd.doctor.main`).
* ``config``  — print the resolved config path. With ``--edit``, open the
                file in ``$EDITOR`` (defaulting to ``nano``).

No ``toggle`` subcommand: the sway bindsym calls ``pkill -USR2 voxd``
directly, so adding one would just be dead code.

Fail-fast on missing API key: if the loaded config has an empty
``openrouter.api_key`` we exit with a clear message pointing at ``voxd setup``
rather than starting the daemon and hitting an :class:`AuthInvalid` on the
first transcription.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from collections.abc import Sequence

from voxd import __version__, config
from voxd.daemon import Daemon


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``argparse.ArgumentParser`` with subparsers."""
    parser = argparse.ArgumentParser(
        prog="voxd",
        description="Headless voice-to-text daemon for Wayland/sway.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"voxd {__version__}",
    )

    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    sub.add_parser("setup", help="Interactive setup: config + sway snippet")
    sub.add_parser("doctor", help="Diagnose missing dependencies")

    p_config = sub.add_parser("config", help="Show or edit config")
    p_config.add_argument(
        "--edit",
        action="store_true",
        help="Open config in $EDITOR (default: nano)",
    )

    return parser


def _run_daemon() -> int:
    """Load config and drive :class:`Daemon` until SIGINT or completion.

    Returns ``0`` on a clean shutdown (Ctrl+C, ``stop()``), ``1`` if the
    loaded config is missing an API key (we point the user at ``voxd setup``
    rather than producing an opaque ``AuthInvalid`` on the first
    transcription), and ``2`` if :class:`Daemon` rejects the config as
    internally inconsistent (e.g. ``inject.type=true`` with ``clipboard=false``).
    """
    cfg = config.load()
    if not cfg.openrouter.api_key:
        print(
            "voxd: missing OpenRouter API key. Run `voxd setup` to configure.",
            file=sys.stderr,
        )
        return 1

    try:
        daemon = Daemon(cfg)
    except ValueError as exc:
        print(f"voxd: {exc}", file=sys.stderr)
        print(f"  config file: {config.default_path()}", file=sys.stderr)
        return 2

    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        daemon.stop()
    return 0


def _cmd_config(*, edit: bool) -> int:
    """Handle ``voxd config`` / ``voxd config --edit``.

    Without ``--edit``, prints the resolved config path so it composes with
    shell scripts (``vim "$(voxd config)"``). With ``--edit``, spawns
    ``$EDITOR <path>`` (defaulting to ``nano``) and returns 0 regardless of
    the editor's exit code — refusing to save is a normal flow, not an error
    we should surface.
    """
    path = config.default_path()
    if edit:
        editor = os.environ.get("EDITOR") or "nano"
        subprocess.run([editor, str(path)], check=False)
        return 0
    print(path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (defaulting to :data:`sys.argv[1:]`) and dispatch.

    Returns the process exit code. Sub-commands return whatever their handler
    returns; the no-arg path returns the daemon's exit code; ``--version``
    raises :class:`SystemExit` via ``argparse`` before we get a chance to
    return.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        return _run_daemon()
    if args.cmd == "setup":
        # Import lazily so tests can monkey-patch ``voxd.setup.main`` without
        # triggering the real setup module at CLI import time.
        from voxd import setup as setup_mod

        return setup_mod.main()
    if args.cmd == "doctor":
        from voxd import doctor as doctor_mod

        return doctor_mod.main()
    if args.cmd == "config":
        return _cmd_config(edit=args.edit)

    # Unreachable: argparse rejects unknown subcommands before we get here.
    parser.error(f"unknown command: {args.cmd}")
    return 2  # pragma: no cover
