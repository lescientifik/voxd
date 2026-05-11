"""Tests for ``voxd.indicator``: notify-send wrapper with replace lifecycle.

The indicator is the user's only feedback channel: a single notification
that mutates state across the recording → transcribing → done cycle. Its
contract is:

* ``show_recording`` posts a persistent (``-t 0``), low-urgency (``-u low``)
  notification and captures the id printed by ``notify-send -p`` so later
  state changes can reuse the same notification slot via ``-r <id>``.
* ``show_transcribing`` replaces the current notification (no stack).
* ``show_error`` surfaces a problem to the user without crashing.
* ``hide`` makes the notification disappear quickly.
* No call ever raises — a missing or broken ``notify-send`` must never
  bring the daemon down. Failures are logged and swallowed.

We never exec the real ``notify-send`` binary; ``subprocess.run`` is
monkeypatched and every call is captured for assertion. The fake also
emulates ``-p`` by writing a deterministic id to ``stdout``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any

import pytest

from voxd.indicator import Indicator

# ---------------------------------------------------------------------------
# Capture harness for subprocess.run
# ---------------------------------------------------------------------------


@dataclass
class Call:
    """One captured invocation of ``subprocess.run``."""

    argv: list[str]
    kwargs: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def capture_run(monkeypatch: pytest.MonkeyPatch) -> list[Call]:
    """Replace ``subprocess.run`` with a recorder that emulates ``notify-send -p``.

    When ``-p`` is present in the argv, the fake returns ``"42\\n"`` on
    stdout (the printed notification id). Otherwise it returns an empty
    stdout. Always exit 0.
    """
    calls: list[Call] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(Call(argv=list(argv), kwargs=kwargs))
        stdout = "42\n" if "-p" in argv else ""
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_show_recording_calls_notify_send(capture_run: list[Call]) -> None:
    """``show_recording`` posts a persistent, low-urgency notification and
    asks ``notify-send`` to print the id (so we can replace it later)."""
    indicator = Indicator()
    indicator.show_recording()

    assert len(capture_run) == 1
    argv = capture_run[0].argv
    assert argv[0] == "notify-send"
    # Persistent: timeout 0 → no auto-hide.
    assert "-t" in argv
    assert argv[argv.index("-t") + 1] == "0"
    # Low urgency: doesn't steal focus.
    assert "-u" in argv
    assert argv[argv.index("-u") + 1] == "low"
    # Print id (so we can capture it for later -r).
    assert "-p" in argv
    # The recording message reaches the user.
    joined = " ".join(argv)
    assert "Recording" in joined


def test_show_transcribing_replaces_recording(capture_run: list[Call]) -> None:
    """A second state change reuses the captured id via ``-r <id>`` so
    notifications never stack — there is always exactly one voxd bubble."""
    indicator = Indicator()
    indicator.show_recording()  # fake prints id "42"
    indicator.show_transcribing()

    assert len(capture_run) == 2
    second = capture_run[1].argv
    assert second[0] == "notify-send"
    # Replace the previously captured id.
    assert "-r" in second
    assert second[second.index("-r") + 1] == "42"
    # Still persistent and low-urgency.
    assert "-t" in second and second[second.index("-t") + 1] == "0"
    assert "-u" in second and second[second.index("-u") + 1] == "low"
    # Carries the transcribing message.
    assert "Transcribing" in " ".join(second)


def test_hide_closes_notification(capture_run: list[Call]) -> None:
    """``hide`` makes the bubble disappear. We implement this as a replace
    with a tiny timeout (``-t 1``) so we avoid pulling in ``gdbus`` — the
    same ``notify-send`` binary we already require is enough."""
    indicator = Indicator()
    indicator.show_recording()  # captures id 42
    indicator.hide()

    assert len(capture_run) == 2
    hide_argv = capture_run[1].argv
    assert hide_argv[0] == "notify-send"
    # Same slot.
    assert "-r" in hide_argv
    assert hide_argv[hide_argv.index("-r") + 1] == "42"
    # Tiny timeout → vanishes immediately.
    assert "-t" in hide_argv
    assert hide_argv[hide_argv.index("-t") + 1] == "1"


def test_hide_without_prior_show_is_noop(capture_run: list[Call]) -> None:
    """Calling ``hide`` before any ``show_*`` must not fail and must not
    fabricate a fake id — there's nothing to close."""
    Indicator().hide()
    assert capture_run == []


def test_show_error_uses_critical_urgency(capture_run: list[Call]) -> None:
    """Errors get ``-u critical`` so they're visible even on a busy desktop,
    and the message text is forwarded verbatim."""
    indicator = Indicator()
    indicator.show_error("API key invalid")

    assert len(capture_run) == 1
    argv = capture_run[0].argv
    assert argv[0] == "notify-send"
    assert "-u" in argv
    assert argv[argv.index("-u") + 1] == "critical"
    assert "API key invalid" in argv


def test_indicator_survives_notify_send_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``notify-send`` is missing or fails, ``Indicator`` must log and
    keep going. The daemon's lifecycle does not depend on libnotify."""

    def boom(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", "notify-send")

    monkeypatch.setattr(subprocess, "run", boom)

    indicator = Indicator()
    # None of these must raise.
    indicator.show_recording()
    indicator.show_transcribing()
    indicator.show_error("oops")
    indicator.hide()


def test_indicator_survives_called_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero exit from ``notify-send`` is also swallowed."""

    def fail(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(returncode=1, cmd=argv, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fail)

    indicator = Indicator()
    indicator.show_recording()
    indicator.show_transcribing()
    indicator.show_error("oops")
    indicator.hide()


def test_show_recording_idempotent_reuses_id(capture_run: list[Call]) -> None:
    """Calling ``show_recording`` a second time replaces the existing bubble
    rather than creating a new one — useful if the daemon re-arms after an
    error path."""
    indicator = Indicator()
    indicator.show_recording()
    indicator.show_recording()

    assert len(capture_run) == 2
    second = capture_run[1].argv
    assert "-r" in second
    assert second[second.index("-r") + 1] == "42"
