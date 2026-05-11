"""Tests for ``voxd.inject``: wl-copy + wtype subprocess wrapper.

The injection step has two side channels: the Wayland clipboard
(``wl-copy``, reading the text from stdin) and synthetic keystrokes
(``wtype``, taking the text as argv). Both are optional via
``InjectConfig.clipboard`` / ``InjectConfig.type``.

These tests pin the contract:

* default config exercises both channels in the right order
  (clipboard first, so a manual Ctrl+V fallback remains usable if
  ``wtype`` blows up);
* the boolean flags individually gate each channel;
* empty text short-circuits with zero subprocess calls;
* UTF-8 (French accents) round-trips byte-for-byte;
* subprocess failures are wrapped in ``InjectError`` with a clear message.

We do not exec real binaries — ``subprocess.run`` is monkeypatched and
every call is captured for assertion.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

import pytest

from voxd.config import InjectConfig
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_inject_default_calls_both_wlcopy_and_wtype(capture_run: list[Call]) -> None:
    """Default config (both flags True) issues exactly two subprocess calls:
    wl-copy first (text on stdin), then wtype (text as argv)."""
    inject("hello world", InjectConfig())

    assert len(capture_run) == 2

    wlcopy = capture_run[0]
    assert wlcopy.argv[0] == "wl-copy"
    assert wlcopy.stdin == "hello world"

    wtype = capture_run[1]
    assert wtype.argv[0] == "wtype"
    # text reaches wtype via argv; '--' is recommended to swallow leading dashes
    assert "hello world" in wtype.argv
    assert wtype.stdin is None


def test_inject_clipboard_false_skips_wlcopy(capture_run: list[Call]) -> None:
    """With ``clipboard=False`` only ``wtype`` is invoked."""
    inject("hi", InjectConfig(type=True, clipboard=False))

    assert len(capture_run) == 1
    assert capture_run[0].argv[0] == "wtype"


def test_inject_type_false_skips_wtype(capture_run: list[Call]) -> None:
    """With ``type=False`` only ``wl-copy`` is invoked."""
    inject("hi", InjectConfig(type=False, clipboard=True))

    assert len(capture_run) == 1
    assert capture_run[0].argv[0] == "wl-copy"
    assert capture_run[0].stdin == "hi"


def test_inject_utf8_french_accents(capture_run: list[Call]) -> None:
    """French diacritics survive intact through both channels."""
    text = "café, naïve, façon"
    inject(text, InjectConfig())

    # wl-copy gets the exact text on stdin
    assert capture_run[0].argv[0] == "wl-copy"
    assert capture_run[0].stdin == text

    # wtype gets the exact text in argv (last element after the optional '--')
    assert capture_run[1].argv[0] == "wtype"
    assert text in capture_run[1].argv


def test_inject_empty_text_is_noop(capture_run: list[Call]) -> None:
    """Empty text returns immediately without spawning any subprocess."""
    inject("", InjectConfig())

    assert capture_run == []


def test_inject_wlcopy_called_before_wtype(capture_run: list[Call]) -> None:
    """When both channels are enabled, clipboard is populated first.

    This matters so that if ``wtype`` fails (no focused window, missing
    binary, etc.) the user can still recover the text with a manual paste.
    """
    inject("ordered", InjectConfig(type=True, clipboard=True))

    assert [c.argv[0] for c in capture_run] == ["wl-copy", "wtype"]


def test_inject_propagates_subprocess_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing binary (FileNotFoundError) is re-raised as ``InjectError``
    with a message naming the failing command."""

    def boom(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", argv[0])

    monkeypatch.setattr(subprocess, "run", boom)

    with pytest.raises(InjectError) as excinfo:
        inject("nope", InjectConfig())

    # Message must point at the failing command so the daemon can notify clearly.
    assert "wl-copy" in str(excinfo.value)


def test_inject_called_process_error_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero exit code is wrapped as ``InjectError`` too."""

    def fail(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(returncode=1, cmd=argv, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(InjectError) as excinfo:
        inject("nope", InjectConfig(type=False, clipboard=True))

    assert "wl-copy" in str(excinfo.value)


def test_inject_wtype_failure_after_wlcopy_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """If wl-copy succeeds but wtype fails, the error is raised AFTER the
    clipboard has been populated — the caller can still rely on the user
    pasting manually.

    We assert this by counting calls before the exception fires.
    """
    calls: list[Call] = []

    def maybe_fail(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(Call(argv=list(argv), stdin=kwargs.get("input"), kwargs=kwargs))
        if argv[0] == "wtype":
            raise FileNotFoundError(2, "No such file or directory", "wtype")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", maybe_fail)

    with pytest.raises(InjectError) as excinfo:
        inject("text", InjectConfig())

    assert [c.argv[0] for c in calls] == ["wl-copy", "wtype"]
    assert "wtype" in str(excinfo.value)
