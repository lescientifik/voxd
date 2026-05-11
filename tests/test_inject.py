"""Tests for ``voxd.inject``: wl-copy + wtype paste-via-keystroke wrapper.

After the v0.2 refactor, injection no longer types the transcribed text
character-by-character (the upstream ``wtype`` typing path has known data
loss — atx/wtype#46). Instead, ``inject()`` populates the clipboard with
``wl-copy`` and then asks ``wtype`` to synthesise a paste keystroke
(``Ctrl+V`` or ``Ctrl+Shift+V``), picking the right combo from the focused
window's identity (queried via :mod:`voxd.focus`).

The contract these tests pin:

* ``wl-copy`` always runs first (so a Xwayland or unknown-focus fallback
  still leaves a pasteable clipboard);
* the focus detector is only consulted when ``cfg.type`` is true (perf);
* Wayland-native terminals get ``Ctrl+Shift+V`` (their standard paste);
* every other Wayland-native or unknown focus gets plain ``Ctrl+V``;
* Xwayland focus skips ``wtype`` entirely (wtype can't drive Xwayland —
  atx/wtype#62) and fires a best-effort ``notify-send`` instead;
* the released modifiers come out in reverse order of the pressed ones,
  matching the wtype README idiom (``-M ctrl -M shift -k v -m shift -m ctrl``);
* subprocess failures of the required channels raise :class:`InjectError`;
* ``notify-send`` failures are swallowed (the notif is informational).

We do not exec real binaries — ``subprocess.run`` is monkeypatched and
every call is captured for assertion.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

import pytest

import voxd.inject
from voxd.config import InjectConfig
from voxd.focus import FocusInfo, FocusKind
from voxd.inject import InjectError, inject

# ---------------------------------------------------------------------------
# Capture harness for subprocess.run
# ---------------------------------------------------------------------------


@dataclass
class Call:
    """One captured invocation of ``subprocess.run``."""

    argv: list[str]
    stdin: str | None
    kwargs: dict[str, Any]


@pytest.fixture
def capture_run(monkeypatch: pytest.MonkeyPatch) -> list[Call]:
    """Replace ``subprocess.run`` with a recorder that returns a fake result.

    Returns the list of captured calls in invocation order. The list is
    mutated by the patched function; tests just read it after calling
    ``inject``.
    """
    calls: list[Call] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(Call(argv=list(argv), stdin=kwargs.get("input"), kwargs=kwargs))
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.fixture
def force_focus(monkeypatch: pytest.MonkeyPatch) -> _FocusForcer:
    """Replace :func:`voxd.inject.detect_focus` with a controllable stub.

    The default verdict is a non-terminal Wayland app so tests that don't care
    about the focus branch get the plain ``Ctrl+V`` path. Tests that need a
    different verdict call ``.set(...)``.
    """
    return _FocusForcer(monkeypatch)


class _FocusForcer:
    """Helper that swaps in a fake :func:`detect_focus` and counts calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._verdict = FocusInfo(kind=FocusKind.WAYLAND, app_id="google-chrome")
        self.call_count = 0

        def fake_detect() -> FocusInfo:
            self.call_count += 1
            return self._verdict

        monkeypatch.setattr(voxd.inject, "detect_focus", fake_detect)

    def set(self, verdict: FocusInfo) -> None:
        """Update the verdict returned by future ``detect_focus()`` calls."""
        self._verdict = verdict


# ---------------------------------------------------------------------------
# wl-copy plumbing (unchanged by the refactor)
# ---------------------------------------------------------------------------


def test_inject_clipboard_false_skips_wlcopy(
    capture_run: list[Call], force_focus: _FocusForcer
) -> None:
    """With ``clipboard=False`` ``wl-copy`` is never invoked."""
    _ = force_focus  # non-terminal Wayland default → plain Ctrl+V via wtype
    inject("hi", InjectConfig(type=True, clipboard=False))

    binaries = [c.argv[0] for c in capture_run]
    assert "wl-copy" not in binaries
    assert "wtype" in binaries


def test_inject_type_false_skips_wtype(capture_run: list[Call]) -> None:
    """With ``type=False`` only ``wl-copy`` is invoked.

    No focus detection, no wtype, no notify-send — the clipboard path is
    completely independent of the focused window.
    """
    inject("hi", InjectConfig(type=False, clipboard=True))

    assert len(capture_run) == 1
    assert capture_run[0].argv[0] == "wl-copy"
    assert capture_run[0].stdin == "hi"


def test_inject_empty_text_is_noop(capture_run: list[Call]) -> None:
    """Empty text returns immediately without spawning any subprocess."""
    inject("", InjectConfig())

    assert capture_run == []


def test_inject_wlcopy_called_before_wtype(
    capture_run: list[Call], force_focus: _FocusForcer
) -> None:
    """When both channels are enabled, ``wl-copy`` runs first.

    This invariant lets every downstream branch (Xwayland notif, unknown
    fallback, terminal Ctrl+Shift+V) rely on the clipboard already holding
    the text — that's what makes the paste keystroke meaningful.
    """
    _ = force_focus
    inject("ordered", InjectConfig(type=True, clipboard=True))

    binaries = [c.argv[0] for c in capture_run]
    assert binaries.index("wl-copy") < binaries.index("wtype")


def test_inject_utf8_text_still_reaches_wlcopy(
    capture_run: list[Call], force_focus: _FocusForcer
) -> None:
    """French diacritics survive byte-for-byte through ``wl-copy``.

    The text path goes clipboard-only now (wtype no longer receives the
    transcription as argv), so the clipboard is the canonical text channel.
    """
    _ = force_focus
    text = "café, naïve, façon"
    inject(text, InjectConfig())

    wlcopy = next(c for c in capture_run if c.argv[0] == "wl-copy")
    assert wlcopy.stdin == text


# ---------------------------------------------------------------------------
# Focus-driven routing
# ---------------------------------------------------------------------------


def test_inject_wayland_non_terminal_uses_ctrl_v(
    capture_run: list[Call], force_focus: _FocusForcer
) -> None:
    """A non-terminal Wayland app gets a plain ``Ctrl+V`` keystroke combo.

    The combo is ``-M ctrl -k v -m ctrl`` — Shift must NOT appear.
    """
    force_focus.set(FocusInfo(kind=FocusKind.WAYLAND, app_id="google-chrome"))
    inject("hi", InjectConfig())

    wtype = next(c for c in capture_run if c.argv[0] == "wtype")
    assert wtype.argv == ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]
    assert "shift" not in wtype.argv


@pytest.mark.parametrize(
    "terminal_app_id",
    [
        "foot",
        "Alacritty",
        "kitty",
        "org.wezfurlong.wezterm",
        "com.mitchellh.ghostty",
    ],
)
def test_inject_wayland_terminal_uses_ctrl_shift_v(
    capture_run: list[Call], force_focus: _FocusForcer, terminal_app_id: str
) -> None:
    """Each known terminal app_id triggers ``Ctrl+Shift+V`` (its paste binding).

    Terminals interpret bare ``Ctrl+V`` as a literal control sequence (SIGTERM,
    visual block, etc.), so we have to send the shifted variant for them.
    """
    force_focus.set(FocusInfo(kind=FocusKind.WAYLAND, app_id=terminal_app_id))
    inject("hi", InjectConfig())

    wtype = next(c for c in capture_run if c.argv[0] == "wtype")
    assert wtype.argv == [
        "wtype",
        "-M", "ctrl",
        "-M", "shift",
        "-k", "v",
        "-m", "shift",
        "-m", "ctrl",
    ]


def test_inject_unknown_focus_uses_ctrl_v(
    capture_run: list[Call], force_focus: _FocusForcer
) -> None:
    """Unknown focus (no sway, parse fail, etc.) falls back to plain ``Ctrl+V``.

    Conservative default: it works on every Wayland compositor with a paste
    binding (Hyprland, labwc, GNOME, …) and accepts the (small) cost that a
    focused terminal in the unknown branch would need a manual paste.
    """
    force_focus.set(FocusInfo(kind=FocusKind.UNKNOWN, app_id=None))
    inject("hi", InjectConfig())

    wtype = next(c for c in capture_run if c.argv[0] == "wtype")
    assert wtype.argv == ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]


def test_inject_xwayland_skips_wtype_and_notifies(
    capture_run: list[Call], force_focus: _FocusForcer
) -> None:
    """Xwayland focus → no ``wtype`` (it can't drive X11), a notif instead.

    The user still has the clipboard; the notif tells them to paste manually.
    """
    force_focus.set(FocusInfo(kind=FocusKind.XWAYLAND, app_id=None))
    inject("hi", InjectConfig())

    binaries = [c.argv[0] for c in capture_run]
    assert "wl-copy" in binaries
    assert "wtype" not in binaries
    assert "notify-send" in binaries

    notif = next(c for c in capture_run if c.argv[0] == "notify-send")
    assert any("paste manually" in arg.lower() for arg in notif.argv), (
        f"notify-send message must mention paste manually; got argv={notif.argv}"
    )


def test_inject_xwayland_still_runs_wlcopy_first(
    capture_run: list[Call], force_focus: _FocusForcer
) -> None:
    """The clipboard is set even when ``wtype`` is skipped on Xwayland focus.

    Without it the notify-send branch would tell the user to paste an empty
    clipboard — strictly worse than typing nothing.
    """
    force_focus.set(FocusInfo(kind=FocusKind.XWAYLAND, app_id=None))
    inject("hi", InjectConfig())

    wlcopy = next(c for c in capture_run if c.argv[0] == "wl-copy")
    assert wlcopy.stdin == "hi"


def test_inject_type_false_skips_focus_detection(
    capture_run: list[Call], force_focus: _FocusForcer
) -> None:
    """``cfg.type=False`` short-circuits the focus probe (no swaymsg spawn).

    The detector spawns a subprocess on every call, so guarding the call on
    ``cfg.type`` is both a correctness and a perf concern — clipboard-only
    users shouldn't pay for swaymsg.
    """
    _ = capture_run
    inject("hi", InjectConfig(type=False, clipboard=True))

    assert force_focus.call_count == 0


def test_inject_wtype_combo_release_modifiers_reverse_order(
    capture_run: list[Call], force_focus: _FocusForcer
) -> None:
    """Modifier releases (``-m``) come in reverse press (``-M``) order.

    The wtype README example for a chord follows the LIFO pattern: press
    ctrl, press shift, type key, release shift, release ctrl. Pin the exact
    order so a refactor can't accidentally reverse it.
    """
    force_focus.set(FocusInfo(kind=FocusKind.WAYLAND, app_id="foot"))
    inject("hi", InjectConfig())

    wtype = next(c for c in capture_run if c.argv[0] == "wtype")
    # Releases occur after the -k key event; check their relative ordering.
    release_indices = [
        (i, wtype.argv[i + 1]) for i, tok in enumerate(wtype.argv) if tok == "-m"
    ]
    assert [mod for _, mod in release_indices] == ["shift", "ctrl"]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_inject_propagates_subprocess_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing ``wl-copy`` binary surfaces as :class:`InjectError`.

    The exception message names the failing command so the daemon can show
    the user a useful notif.
    """

    def boom(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", argv[0])

    monkeypatch.setattr(subprocess, "run", boom)

    with pytest.raises(InjectError) as excinfo:
        inject("nope", InjectConfig())

    assert "wl-copy" in str(excinfo.value)


def test_inject_called_process_error_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero exit from ``wl-copy`` is also wrapped as :class:`InjectError`."""

    def fail(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(returncode=1, cmd=argv, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(InjectError) as excinfo:
        inject("nope", InjectConfig(type=False, clipboard=True))

    assert "wl-copy" in str(excinfo.value)


def test_inject_notify_send_failure_is_swallowed(
    monkeypatch: pytest.MonkeyPatch, force_focus: _FocusForcer
) -> None:
    """``notify-send`` failures must NOT raise — the notif is best-effort.

    The clipboard already holds the text by the time we attempt the notif,
    so swallowing the failure leaves the user no worse off than if libnotify
    weren't installed at all.
    """
    force_focus.set(FocusInfo(kind=FocusKind.XWAYLAND, app_id=None))

    def maybe_fail(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[0] == "notify-send":
            raise FileNotFoundError(2, "No such file or directory", "notify-send")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", maybe_fail)

    # Must not raise.
    inject("hi", InjectConfig())
