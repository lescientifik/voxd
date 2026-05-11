"""Single-slot desktop notification driven by ``notify-send``.

voxd has no GUI, no tray, no overlay. The only user-facing feedback is a
desktop notification that mutates through the recording → transcribing
→ done lifecycle. We keep exactly **one** notification on screen at any
time by capturing the id printed by ``notify-send -p`` on the first
show, then reusing it via ``notify-send -r <id>`` for every subsequent
state change. There is no notification stack: each state replaces the
previous bubble in-place.

Design choices
--------------

* ``-t 0`` (no auto-hide) so the user sees the current state for as long
  as the daemon is in it. We're the one in charge of dismissing.
* ``-u low`` for the recording / transcribing states — they're status,
  not alerts. ``-u critical`` for ``show_error`` so failures are visible
  even on a busy desktop.
* ``hide`` is implemented as ``notify-send -r <id> -t 1 voxd ""``: a
  replace with a 1 ms timeout. The bubble disappears immediately and we
  avoid pulling in ``gdbus`` (or any extra dependency) just to call
  ``org.freedesktop.Notifications.CloseNotification``.
* Every subprocess call is wrapped: a missing or broken ``notify-send``
  must **never** crash the daemon. Errors are logged at WARNING and
  swallowed.
"""

from __future__ import annotations

import logging
import subprocess

_log = logging.getLogger(__name__)

_APP_NAME = "voxd"


def _run_notify(args: list[str]) -> str | None:
    """Invoke ``notify-send`` with ``args`` and return its stdout, or ``None``
    on failure.

    All exceptions from ``subprocess.run`` (missing binary, non-zero exit,
    OS errors) are caught, logged at WARNING, and turned into a ``None``
    return value. Callers must treat ``None`` as "ignore, keep going".
    """
    try:
        proc = subprocess.run(
            ["notify-send", *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        _log.warning("notify-send not found in PATH; indicator disabled")
        return None
    except subprocess.CalledProcessError as exc:
        _log.warning("notify-send failed: exit %d (stderr=%r)", exc.returncode, exc.stderr)
        return None
    except OSError as exc:
        _log.warning("notify-send OS error: %s", exc)
        return None
    return proc.stdout


class Indicator:
    """Single-slot desktop notification controller.

    Tracks the id printed by the first successful ``notify-send -p`` call
    in ``self._notif_id`` and reuses it across state transitions with
    ``-r <id>``. None of the public methods ever raise: the daemon must
    keep working even if the user has no notification daemon running or
    ``libnotify`` is missing entirely.
    """

    def __init__(self) -> None:
        """Initialise with no live notification.

        ``_notif_id`` becomes a real id after the first successful
        ``show_*`` call; it stays ``None`` if ``notify-send`` failed or
        was never invoked.
        """
        self._notif_id: int | None = None

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def show_recording(self) -> None:
        """Show / refresh the "Recording…" bubble (low urgency, persistent)."""
        self._post_status("Recording…")

    def show_transcribing(self) -> None:
        """Replace the current bubble with "Transcribing…" (low urgency)."""
        self._post_status("Transcribing…")

    def show_error(self, msg: str) -> None:
        """Surface an error to the user with ``-u critical``.

        Replaces the current bubble if one is live, otherwise opens a new
        one. The message is forwarded verbatim as the notification body.
        """
        args = ["-u", "critical", "-t", "0", "-p"]
        if self._notif_id is not None:
            args += ["-r", str(self._notif_id)]
        args += [_APP_NAME, msg]
        self._capture_id(_run_notify(args))

    def hide(self) -> None:
        """Dismiss the current bubble, if any.

        Implemented as ``notify-send -r <id> -t 1 voxd ""`` — a replace
        with a 1 ms timeout, which makes the bubble vanish without any
        external dependency beyond ``notify-send`` itself. No-op if no
        notification id has been captured yet.
        """
        if self._notif_id is None:
            return
        _run_notify(["-r", str(self._notif_id), "-t", "1", _APP_NAME, ""])
        self._notif_id = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _post_status(self, message: str) -> None:
        """Post or replace a low-urgency, persistent status notification."""
        args = ["-u", "low", "-t", "0", "-p"]
        if self._notif_id is not None:
            args += ["-r", str(self._notif_id)]
        args += [_APP_NAME, message]
        self._capture_id(_run_notify(args))

    def _capture_id(self, stdout: str | None) -> None:
        """Parse ``notify-send -p`` stdout into ``self._notif_id``.

        Silently ignores empty / malformed output: an indicator with no
        captured id simply skips ``-r`` on the next call, which posts a
        fresh notification instead of replacing — a benign degradation.
        """
        if stdout is None:
            return
        stripped = stdout.strip()
        if not stripped:
            return
        try:
            self._notif_id = int(stripped)
        except ValueError:
            _log.warning("notify-send returned non-integer id: %r", stripped)


__all__ = ["Indicator"]
