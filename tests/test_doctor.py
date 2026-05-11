"""Tests for ``voxd doctor`` — environment diagnostics.

``voxd doctor`` is the user-facing self-check for the system-level
dependencies voxd needs to function: the ``wtype`` and ``wl-copy``
Wayland helpers, ``notify-send`` (libnotify), the PortAudio shared
library backing ``sounddevice``, and effective microphone access.

These tests pin the contract:

* missing/present binaries are reported with stable substrings
  (``"wtype: OK"`` / ``"wtype: MISSING"``) the user (and grep) can rely on;
* a PortAudio ``OSError`` at import time is *not* fatal — the doctor
  must catch it, surface it as a critical failure, and skip the mic check;
* a non-sway desktop produces a warning but does not flip the exit code
  (the user may run another wlroots compositor where wtype works fine);
* microphone access failures are warnings (the user may simply not have
  plugged a mic in), not exit-code-flipping errors;
* the exit code is 0 iff no *critical* check failed (``wtype``,
  ``wl-copy`` and PortAudio are the three critical ones — without them
  voxd cannot function at all: ``wl-copy`` is critical because the
  injection pipeline pastes via Ctrl+V, so the transcript must be on the
  clipboard first).

We import sounddevice lazily through ``doctor._import_sounddevice`` so
the import-error test can monkey-patch a single, stable seam.
"""

from __future__ import annotations

import shutil
import sys
import types
from collections.abc import Callable

import pytest

from voxd import doctor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_which_stub(missing: set[str]) -> Callable[[str], str | None]:
    """Return a ``shutil.which`` replacement that reports ``missing`` as None."""

    def fake(name: str, *args: object, **kwargs: object) -> str | None:  # noqa: ARG001
        if name in missing:
            return None
        return f"/usr/bin/{name}"

    return fake


def _fake_sounddevice_module(open_raises: BaseException | None = None) -> types.ModuleType:
    """Build a fake ``sounddevice`` module compatible with the mic-access check."""
    mod = types.ModuleType("sounddevice")

    class _Stream:
        def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: ARG002
            if open_raises is not None:
                raise open_raises

        def __enter__(self) -> _Stream:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    mod.InputStream = _Stream  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def all_binaries_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every binary lookup succeed."""
    monkeypatch.setattr(shutil, "which", _make_which_stub(missing=set()))


@pytest.fixture
def sway_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend we're running under sway so the desktop check passes."""
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "sway")


@pytest.fixture
def good_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make PortAudio import succeed AND mic-stream open succeed."""
    mod = _fake_sounddevice_module(open_raises=None)
    monkeypatch.setattr(doctor, "_import_sounddevice", lambda: mod)


# ---------------------------------------------------------------------------
# Binary checks
# ---------------------------------------------------------------------------


def test_doctor_reports_missing_wtype(
    monkeypatch: pytest.MonkeyPatch,
    sway_desktop: None,
    good_sounddevice: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``shutil.which("wtype") -> None`` produces a 'wtype: MISSING' line."""
    monkeypatch.setattr(shutil, "which", _make_which_stub(missing={"wtype"}))

    rc = doctor.main()
    out = capsys.readouterr().out

    assert "wtype: MISSING" in out
    # wtype is a critical dependency: missing it must flip the exit code.
    assert rc == 1


def test_doctor_reports_present_tools(
    all_binaries_present: None,
    sway_desktop: None,
    good_sounddevice: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All three binaries present → 'wtype: OK', 'wl-copy: OK', 'notify-send: OK'."""
    rc = doctor.main()
    out = capsys.readouterr().out

    assert "wtype: OK" in out
    assert "wl-copy: OK" in out
    assert "notify-send: OK" in out
    assert rc == 0


# ---------------------------------------------------------------------------
# PortAudio check
# ---------------------------------------------------------------------------


def test_doctor_handles_portaudio_import_error(
    monkeypatch: pytest.MonkeyPatch,
    all_binaries_present: None,
    sway_desktop: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If importing ``sounddevice`` raises ``OSError``, doctor reports MISSING.

    The whole point of wrapping the import in ``_import_sounddevice`` is to
    survive a PortAudio-less box — the daemon won't work there, but the
    doctor must tell us *that* rather than crashing.
    """

    def boom() -> types.ModuleType:
        raise OSError("libportaudio2 not found")

    monkeypatch.setattr(doctor, "_import_sounddevice", boom)

    rc = doctor.main()
    out = capsys.readouterr().out

    assert "PortAudio: MISSING" in out
    assert "libportaudio2 not found" in out
    # PortAudio is a critical dependency.
    assert rc == 1


def test_doctor_skips_mic_check_when_portaudio_missing(
    monkeypatch: pytest.MonkeyPatch,
    all_binaries_present: None,
    sway_desktop: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If PortAudio import fails, the mic-access check is skipped, not retried.

    Otherwise the user sees a confusing duplicate error.
    """

    def boom() -> types.ModuleType:
        raise OSError("nope")

    monkeypatch.setattr(doctor, "_import_sounddevice", boom)

    rc = doctor.main()
    out = capsys.readouterr().out

    assert "PortAudio: MISSING" in out
    assert "Mic access: SKIPPED" in out
    assert rc == 1


# ---------------------------------------------------------------------------
# Desktop check
# ---------------------------------------------------------------------------


def test_doctor_warns_on_non_sway_desktop(
    monkeypatch: pytest.MonkeyPatch,
    all_binaries_present: None,
    good_sounddevice: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A GNOME desktop yields a non-fatal warning about wtype compatibility."""
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")

    rc = doctor.main()
    out = capsys.readouterr().out

    assert "Desktop:" in out
    assert "GNOME" in out
    # The warning mentions wtype / Wayland so the user knows what's at stake.
    assert "wtype" in out.lower() or "wayland" in out.lower()
    # Desktop is a warning, not a critical failure: exit code stays 0.
    assert rc == 0


def test_doctor_recognises_sway_desktop(
    monkeypatch: pytest.MonkeyPatch,
    all_binaries_present: None,
    good_sounddevice: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``XDG_CURRENT_DESKTOP=sway`` is reported as OK."""
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "sway")

    rc = doctor.main()
    out = capsys.readouterr().out

    assert "Desktop: OK" in out
    assert rc == 0


# ---------------------------------------------------------------------------
# Mic access check
# ---------------------------------------------------------------------------


def test_doctor_checks_mic_access(
    monkeypatch: pytest.MonkeyPatch,
    all_binaries_present: None,
    sway_desktop: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If opening the input stream raises ``OSError``, mic access is DENIED.

    Mic denial is a warning (the user may not have a mic plugged in right
    now), so it does NOT flip the exit code.
    """
    mod = _fake_sounddevice_module(open_raises=OSError("permission denied"))
    monkeypatch.setattr(doctor, "_import_sounddevice", lambda: mod)

    rc = doctor.main()
    out = capsys.readouterr().out

    assert "Mic access: DENIED" in out
    assert "permission denied" in out
    assert rc == 0


def test_doctor_reports_mic_ok_when_stream_opens(
    all_binaries_present: None,
    sway_desktop: None,
    good_sounddevice: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If the input stream opens cleanly, mic access is OK."""
    rc = doctor.main()
    out = capsys.readouterr().out

    assert "Mic access: OK" in out
    assert rc == 0


# ---------------------------------------------------------------------------
# Exit code matrix
# ---------------------------------------------------------------------------


def test_doctor_exit_code_zero_when_all_ok(
    all_binaries_present: None,
    sway_desktop: None,
    good_sounddevice: None,
) -> None:
    """All critical checks OK → exit code 0."""
    assert doctor.main() == 0


def test_doctor_exit_code_one_when_critical_dep_missing(
    monkeypatch: pytest.MonkeyPatch,
    sway_desktop: None,
    good_sounddevice: None,
) -> None:
    """A missing critical dep (wtype) → exit code 1."""
    monkeypatch.setattr(shutil, "which", _make_which_stub(missing={"wtype"}))
    assert doctor.main() == 1


def test_doctor_exit_code_zero_when_only_warnings(
    monkeypatch: pytest.MonkeyPatch,
    good_sounddevice: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-critical issues (notify-send missing, GNOME, mic denied) → still 0.

    Documents the critical-vs-warning split post-paste-via-Ctrl+V refactor:
    ``wtype``, ``wl-copy`` and PortAudio are critical (the injection pipeline
    can't function without any of them). ``notify-send`` is graceful-
    degradation (only used to nudge the user on Xwayland apps), and
    desktop / mic are environmental warnings.
    """
    monkeypatch.setattr(shutil, "which", _make_which_stub(missing={"notify-send"}))
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    mod = _fake_sounddevice_module(open_raises=OSError("no device"))
    monkeypatch.setattr(doctor, "_import_sounddevice", lambda: mod)

    rc = doctor.main()
    out = capsys.readouterr().out

    # Everything we expected to be a warning is reported as such.
    assert "notify-send: MISSING" in out
    assert "Desktop:" in out and "GNOME" in out
    assert "Mic access: DENIED" in out
    # And yet — none of these is critical, so exit code stays 0.
    assert rc == 0


def test_wlcopy_missing_is_critical(
    monkeypatch: pytest.MonkeyPatch,
    sway_desktop: None,
    good_sounddevice: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``shutil.which("wl-copy") -> None`` flips the exit code to 1.

    The injection pipeline pastes via Ctrl+V, so the transcript must land
    on the clipboard first via ``wl-copy``. Without it, the Ctrl+V keystroke
    pastes whatever was previously on the clipboard — actively wrong, not a
    graceful degradation — so ``wl-copy`` is a hard critical dependency.
    """
    monkeypatch.setattr(shutil, "which", _make_which_stub(missing={"wl-copy"}))

    checks = doctor._run_checks()
    wlcopy = next(c for c in checks if c.name == "wl-copy")
    assert wlcopy.critical is True
    assert wlcopy.status is doctor._Status.MISSING

    rc = doctor.main()
    out = capsys.readouterr().out

    assert "wl-copy: MISSING" in out
    assert rc == 1


# ---------------------------------------------------------------------------
# CLI integration (smoke)
# ---------------------------------------------------------------------------


def test_cli_doctor_dispatches_to_doctor_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """``voxd doctor`` returns whatever ``doctor.main`` returns.

    This is already covered indirectly by ``test_cli.py``; re-asserted here
    as a sanity check that the wiring still holds after we rewrite the
    module.
    """
    from voxd import cli

    called: dict[str, int] = {}

    def fake_main() -> int:
        called["n"] = called.get("n", 0) + 1
        return 0

    # Patch the module attribute the CLI looks up via ``from voxd import doctor``.
    monkeypatch.setattr(sys.modules["voxd.doctor"], "main", fake_main)

    rc = cli.main(["doctor"])
    assert rc == 0
    assert called["n"] == 1
