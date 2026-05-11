"""Synchronous text injection into the active Wayland session.

After a successful transcription, the daemon hands the text off to
:func:`inject` which fires up to three subprocesses:

* ``wl-copy`` — populates the Wayland clipboard. Text is piped on stdin.
* ``wtype``   — synthesises a paste keystroke (``Ctrl+V`` or
  ``Ctrl+Shift+V``, picked from the focused window's identity).
* ``notify-send`` — informational fallback when the focused window can't
  be typed into (Xwayland clients — wtype can't drive X11, see
  `atx/wtype#62 <https://github.com/atx/wtype/issues/62>`_).

We **never** type the transcribed text character-by-character: the
upstream wtype typing path drops characters under load
(`atx/wtype#46 <https://github.com/atx/wtype/issues/46>`_, repo abandoned
since 2021). Paste-via-keystroke is reliable because wtype only has to
send three or four scan codes, never the payload itself.

Routing decisions, all driven by :func:`voxd.focus.detect_focus`:

* Wayland-native terminal (``app_id`` in :data:`_TERMINAL_APP_IDS`) →
  ``wtype -M ctrl -M shift -k v -m shift -m ctrl`` (terminals bind paste
  to Ctrl+Shift+V because bare Ctrl+V is a literal control char).
* Wayland-native non-terminal → ``wtype -M ctrl -k v -m ctrl``.
* Xwayland → skip wtype entirely, fire a best-effort notify-send asking
  the user to paste manually.
* Unknown focus (no sway, parse fail, no focused container) → ``Ctrl+V``
  as a conservative cross-compositor default.

Ordering: ``wl-copy`` runs first in every branch so the clipboard is
already populated before any paste keystroke or notif goes out. On
failure of the required channels we raise :class:`InjectError` with a
message naming the offending command — the daemon's caller turns that
into a notify-send error popup. ``notify-send`` itself is best-effort
and its failures are swallowed (the clipboard is already set).
"""

from __future__ import annotations

import subprocess

from voxd.config import InjectConfig
from voxd.focus import FocusKind, detect_focus

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Wayland ``app_id`` strings of terminals known to paste via Ctrl+Shift+V.
# Casing matches what sway actually emits in ``swaymsg -t get_tree`` — note
# the capital "A" in ``Alacritty`` and the reverse-DNS app_ids for wezterm
# and ghostty. Hardcoded by design (5 entries, churn-free, no TOML knob).
_TERMINAL_APP_IDS = frozenset(
    {
        "foot",
        "Alacritty",
        "kitty",
        "org.wezfurlong.wezterm",
        "com.mitchellh.ghostty",
    }
)

# notify-send wording when we hit an Xwayland client. Lower-cased
# "paste manually" must remain in the message — tests pin it as the
# minimum guarantee that the user gets actionable instructions.
_XWAYLAND_NOTIF_SUMMARY = "voxd"
_XWAYLAND_NOTIF_BODY = (
    "Paste manually — XWayland app detected (Ctrl+V or Ctrl+Shift+V)"
)


class InjectError(Exception):
    """Raised when a *required* subprocess used by :func:`inject` fails.

    Wraps either :class:`FileNotFoundError` (binary missing from
    ``$PATH``) or :class:`subprocess.CalledProcessError` (non-zero exit).
    The original exception is chained via ``__cause__``.

    ``notify-send`` failures are intentionally *not* raised through this
    type — the notif is informational and the clipboard channel has
    already succeeded by the time we attempt it.
    """


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run_wlcopy(text: str) -> None:
    """Pipe ``text`` to ``wl-copy`` via stdin.

    Raises :class:`InjectError` on missing binary or non-zero exit — the
    clipboard is the load-bearing channel of the whole paste-via-keystroke
    design, so failure here must be surfaced to the user.
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


def _run_wtype_combo(modifiers: list[str], key: str) -> None:
    """Synthesise a modifier+key chord via ``wtype``.

    Builds an argv of the form::

        wtype -M m1 -M m2 ... -k key -m mN -m mN-1 ... -m m1

    Modifiers are pressed in the given order and released in the reverse
    order (the LIFO idiom from the wtype README). Raises
    :class:`InjectError` on a missing binary or a non-zero exit.
    """
    argv: list[str] = ["wtype"]
    for mod in modifiers:
        argv.extend(("-M", mod))
    argv.extend(("-k", key))
    for mod in reversed(modifiers):
        argv.extend(("-m", mod))

    try:
        subprocess.run(argv, check=True)
    except FileNotFoundError as exc:
        raise InjectError("wtype not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise InjectError(f"wtype failed: exit {exc.returncode}") from exc


def _run_notify_send(summary: str, body: str) -> None:
    """Fire a best-effort ``notify-send`` call; swallow every failure.

    The notif is purely informational: by the time we get here the
    clipboard already holds the text. If libnotify is missing or the
    daemon isn't running, there's nothing actionable to surface — the
    user can still paste manually.
    """
    try:
        subprocess.run(["notify-send", summary, body], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        # Intentionally swallowed — notif is informational, not load-bearing.
        return


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def inject(text: str, cfg: InjectConfig) -> None:
    """Synchronously inject ``text`` per the channels enabled in ``cfg``.

    Order of operations:

    1. Empty text → no-op (no subprocesses spawned).
    2. ``cfg.clipboard`` → ``wl-copy`` populates the clipboard (raises
       :class:`InjectError` on failure).
    3. ``cfg.type`` → probe focus via :func:`voxd.focus.detect_focus` and
       route to one of three branches (Xwayland notif, terminal
       ``Ctrl+Shift+V``, generic ``Ctrl+V``). When ``cfg.type`` is false
       the focus detector is *not* called — it spawns a subprocess and
       there's no point paying for it just to discard the verdict.

    Raises :class:`InjectError` when a required subprocess fails;
    ``notify-send`` failures are swallowed silently.
    """
    if not text:
        return

    if cfg.clipboard:
        _run_wlcopy(text)

    if not cfg.type:
        return

    focus = detect_focus()

    if focus.kind == FocusKind.XWAYLAND:
        _run_notify_send(_XWAYLAND_NOTIF_SUMMARY, _XWAYLAND_NOTIF_BODY)
        return

    if focus.kind == FocusKind.WAYLAND and focus.app_id in _TERMINAL_APP_IDS:
        _run_wtype_combo(["ctrl", "shift"], "v")
    else:
        # WAYLAND non-terminal, or UNKNOWN: plain Ctrl+V. Conservative
        # default that works on Hyprland / labwc / sway-without-focus-info.
        _run_wtype_combo(["ctrl"], "v")


__all__ = ["InjectError", "inject"]
