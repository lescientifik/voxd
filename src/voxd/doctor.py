"""``voxd doctor`` — environment self-check.

Runs through the system-level dependencies voxd needs at runtime and prints a
one-line verdict for each. Designed to be the first thing a user runs when
``voxd`` doesn't work — the output should answer "is my box even capable of
running this daemon?" without making the user dig through logs.

The checks fall in two buckets:

* **Critical** (missing → exit code 1, voxd cannot function):
    - ``wtype`` — without it, no synthetic keystrokes, no text injection.
    - PortAudio (``sounddevice`` importable) — without it, no audio capture.

* **Warning** (missing → exit code stays 0, voxd will work in a degraded mode):
    - ``wl-copy`` — clipboard mirroring is a fallback, not the primary path.
    - ``notify-send`` — the daemon still works without on-screen notifications.
    - Desktop ≠ sway — wtype actually requires a Wayland compositor with the
      virtual-keyboard protocol; sway is the only one we test against, but
      Hyprland and labwc work too. Warn, don't fail.
    - Mic access denied — the user may simply not have a mic plugged in
      *right now*, or PulseAudio/Pipewire may need a permission grant.

The PortAudio import is wrapped in ``_import_sounddevice`` so tests can
substitute a fake module without monkey-patching the import system. This also
keeps the doctor robust on machines that lack ``libportaudio2`` entirely —
importing :mod:`sounddevice` there raises :class:`OSError`, which we catch
and report as a clean "MISSING" rather than letting the whole subcommand
crash.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from types import ModuleType

# 16 kHz mono is what the real capture pipeline opens. We re-use the same
# parameters here so the mic probe exercises the same code path.
_PROBE_SAMPLERATE = 16000
_PROBE_CHANNELS = 1
_PROBE_BLOCKSIZE = 160  # 10 ms — small enough that open-close is near-instant


class _Status(Enum):
    """Outcome of a single diagnostic check."""

    OK = "OK"
    MISSING = "MISSING"
    DENIED = "DENIED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class _Check:
    """One diagnostic line: a name, a status, an optional detail, criticality.

    ``critical`` decides whether a non-OK status flips the overall exit code.
    The output line always reads ``<name>: <STATUS>`` so tests (and ``grep``)
    can rely on a stable substring like ``"wtype: OK"`` or ``"wtype: MISSING"``.
    """

    name: str
    status: _Status
    detail: str = ""
    critical: bool = False

    def format(self) -> str:
        """Render the check as a single human-readable line."""
        marker = {
            _Status.OK: "[OK]  ",
            _Status.MISSING: "[FAIL]",
            _Status.DENIED: "[FAIL]",
            _Status.WARNING: "[WARN]",
            _Status.SKIPPED: "[SKIP]",
        }[self.status]
        base = f"{marker} {self.name}: {self.status.value}"
        if self.detail:
            return f"{base} ({self.detail})"
        return base

    def failed_critically(self) -> bool:
        """Return True iff this check should make the overall exit code 1."""
        return self.critical and self.status is not _Status.OK


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _import_sounddevice() -> ModuleType:
    """Import :mod:`sounddevice` lazily.

    Wrapped in a function so tests can monkey-patch this single seam to inject
    either a fake module (mic-access tests) or an ``OSError`` (PortAudio-
    missing test). Importing :mod:`sounddevice` at module level would defeat
    that — and would also crash the doctor on a box without libportaudio.
    """
    import sounddevice  # noqa: PLC0415 — intentionally deferred

    return sounddevice


def _check_binary(name: str, *, critical: bool, what_for: str) -> _Check:
    """Look up ``name`` on ``$PATH`` and report OK / MISSING.

    ``what_for`` becomes the failure detail so the user knows *why* this
    binary matters (and which package to install).
    """
    path = shutil.which(name)
    if path is None:
        return _Check(
            name=name,
            status=_Status.MISSING,
            detail=what_for,
            critical=critical,
        )
    return _Check(name=name, status=_Status.OK, detail=path, critical=critical)


def _check_portaudio() -> tuple[_Check, ModuleType | None]:
    """Verify :mod:`sounddevice` can be imported (i.e. libportaudio is present).

    Returns the check *and* the imported module (or ``None`` on failure). The
    module is handed to :func:`_check_mic_access` so we don't import twice and
    so the mic check can be cleanly skipped when the import failed.
    """
    try:
        mod = _import_sounddevice()
    except OSError as exc:
        return (
            _Check(
                name="PortAudio",
                status=_Status.MISSING,
                detail=str(exc) or "libportaudio2 not installed",
                critical=True,
            ),
            None,
        )
    return _Check(name="PortAudio", status=_Status.OK, critical=True), mod


def _check_mic_access(sounddevice_mod: ModuleType | None) -> _Check:
    """Try to open a 16 kHz mono ``InputStream`` for a single block.

    If the open succeeds (and the context-manager exit cleans up cleanly),
    the user's mic is reachable. An ``OSError`` here usually means "no input
    device available" or "permission denied" — both warnings, not critical.

    Skipped entirely if PortAudio couldn't be imported (otherwise we'd just
    crash on the first attribute access).
    """
    if sounddevice_mod is None:
        return _Check(
            name="Mic access",
            status=_Status.SKIPPED,
            detail="PortAudio unavailable",
            critical=False,
        )
    try:
        stream_cls = sounddevice_mod.InputStream  # type: ignore[attr-defined]
        with stream_cls(
            samplerate=_PROBE_SAMPLERATE,
            channels=_PROBE_CHANNELS,
            blocksize=_PROBE_BLOCKSIZE,
        ):
            # Just opening + closing is enough: PortAudio negotiates with the
            # backend during ``__enter__`` and raises if the device is busy
            # or the user lacks permission.
            pass
    except OSError as exc:
        return _Check(
            name="Mic access",
            status=_Status.DENIED,
            detail=str(exc),
            critical=False,
        )
    return _Check(name="Mic access", status=_Status.OK, critical=False)


def _check_desktop() -> _Check:
    """Inspect ``$XDG_CURRENT_DESKTOP`` and warn on non-sway environments.

    ``wtype`` and ``wl-copy`` require a Wayland compositor with the
    ``zwp_virtual_keyboard_v1`` protocol. sway is the only one we ship
    instructions for; others (Hyprland, labwc, Niri) typically work too. We
    warn rather than fail because we don't want to lock users out of an
    environment that's likely fine — they can always confirm with the
    binary checks above.
    """
    raw = os.environ.get("XDG_CURRENT_DESKTOP", "")
    if not raw:
        return _Check(
            name="Desktop",
            status=_Status.WARNING,
            detail="XDG_CURRENT_DESKTOP unset — wtype/wl-copy need a Wayland session",
            critical=False,
        )
    if "sway" in raw.lower():
        return _Check(name="Desktop", status=_Status.OK, detail=raw, critical=False)
    return _Check(
        name="Desktop",
        status=_Status.WARNING,
        detail=(
            f"{raw} — wtype/wl-copy require a Wayland compositor with the "
            "virtual-keyboard protocol (sway, Hyprland, labwc, …)"
        ),
        critical=False,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_checks() -> list[_Check]:
    """Execute every diagnostic and return the ordered list of results.

    Order is chosen for human readability: binaries first (cheapest, most
    likely culprits), then PortAudio + mic (more involved), then the
    environmental hint about the desktop session.
    """
    portaudio_check, sounddevice_mod = _check_portaudio()

    checks: list[_Check] = [
        _check_binary(
            "wtype",
            critical=True,
            what_for="install `wtype` — required to inject keystrokes",
        ),
        _check_binary(
            "wl-copy",
            critical=False,
            what_for="install `wl-clipboard` — clipboard mirroring will be disabled",
        ),
        _check_binary(
            "notify-send",
            critical=False,
            what_for="install `libnotify` — on-screen status hints will be disabled",
        ),
        portaudio_check,
        _check_mic_access(sounddevice_mod),
        _check_desktop(),
    ]
    return checks


def _print_checks(checks: Iterable[_Check]) -> None:
    """Print each check on its own line in the order produced by :func:`_run_checks`."""
    for c in checks:
        print(c.format())


def main() -> int:
    """Run every diagnostic, print the report, return 0/1 for shell scripts.

    Returns:
        0 if no *critical* dependency is missing (wtype + PortAudio both OK).
        1 otherwise. Warnings (wl-copy, notify-send, desktop, mic) never
        flip the exit code — they are informational so the user knows what
        to expect, but they do not block voxd from running.
    """
    checks = _run_checks()
    _print_checks(checks)
    failed_critical = any(c.failed_critically() for c in checks)
    return 1 if failed_critical else 0


__all__ = ["main"]
