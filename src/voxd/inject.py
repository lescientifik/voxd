"""Synchronous text injection into the active Wayland session.

After a successful transcription, the daemon hands the text off to
``inject()`` which fires up to two subprocesses:

* ``wl-copy`` — populates the Wayland clipboard. Text is piped on stdin.
* ``wtype``   — synthesises keystrokes into the focused window. Text is
  passed as ``argv`` after a ``--`` sentinel so any leading dash in the
  user's transcription is not parsed as a flag.

Both channels are independently toggleable via :class:`InjectConfig`.

Ordering matters: ``wl-copy`` runs **first** so that even if ``wtype``
fails (no focused window accepting input, binary missing, etc.) the
user still has the transcription on the clipboard and can paste it
manually. On failure of either channel we raise :class:`InjectError`
with a message naming the offending command — the daemon's caller
turns that into a ``notify-send`` error popup.
"""

from __future__ import annotations

import subprocess

from voxd.config import InjectConfig


class InjectError(Exception):
    """Raised when a subprocess used by :func:`inject` fails.

    Wraps either :class:`FileNotFoundError` (binary missing from
    ``$PATH``) or :class:`subprocess.CalledProcessError` (non-zero exit
    code). The original exception is chained via ``__cause__``.
    """


def _run_wlcopy(text: str) -> None:
    """Pipe ``text`` to ``wl-copy`` via stdin.

    Raises :class:`InjectError` on missing binary or non-zero exit.
    """
    try:
        subprocess.run(
            ["wl-copy"],
            input=text,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except FileNotFoundError as exc:
        raise InjectError("wl-copy not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise InjectError(f"wl-copy failed: exit {exc.returncode}") from exc


def _run_wtype(text: str) -> None:
    """Invoke ``wtype -- <text>`` to synthesise keystrokes.

    The ``--`` sentinel guards against transcriptions starting with a
    dash being misread as wtype flags. Raises :class:`InjectError` on
    missing binary or non-zero exit.
    """
    try:
        subprocess.run(
            ["wtype", "--", text],
            check=True,
        )
    except FileNotFoundError as exc:
        raise InjectError("wtype not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise InjectError(f"wtype failed: exit {exc.returncode}") from exc


def inject(text: str, cfg: InjectConfig) -> None:
    """Synchronously inject ``text`` per the channels enabled in ``cfg``.

    Order of operations when both flags are true:

    1. ``wl-copy`` populates the clipboard;
    2. ``wtype`` synthesises keystrokes into the focused window.

    Empty text is a no-op (no subprocesses spawned). On failure of any
    enabled channel, raises :class:`InjectError`; the clipboard channel
    runs first so a ``wtype`` failure still leaves the user with a
    pasteable clipboard.
    """
    if not text:
        return

    if cfg.clipboard:
        _run_wlcopy(text)

    if cfg.type:
        _run_wtype(text)


__all__ = ["InjectError", "inject"]
