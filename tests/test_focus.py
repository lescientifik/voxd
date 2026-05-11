"""Tests for ``voxd.focus``: sway focused-window detector.

The detector calls ``swaymsg -t get_tree``, parses the JSON, walks the tree
to find the focused ``con`` (container) node, and returns a typed verdict:

* :class:`FocusKind.WAYLAND` — Wayland-native client (``app_id`` is set)
* :class:`FocusKind.XWAYLAND` — Xwayland client (``window_properties`` is set
  while ``app_id`` is ``None``)
* :class:`FocusKind.UNKNOWN` — anything we can't classify (swaymsg missing,
  exit non-zero, JSON parse failure, no focused container in the tree)

These tests pin the *contract* — they describe what callers can rely on, not
how the walk is implemented. ``subprocess.run`` is monkeypatched, no real
swaymsg is invoked.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from voxd.focus import FocusInfo, FocusKind, detect_focus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_run_returning(tree: dict[str, Any]) -> Any:
    """Build a fake ``subprocess.run`` returning ``tree`` as JSON on stdout."""

    payload = json.dumps(tree)

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout=payload, stderr=""
        )

    return fake_run


def _con(
    *,
    focused: bool,
    app_id: str | None,
    window_properties: dict[str, str] | None = None,
    nodes: list[dict[str, Any]] | None = None,
    floating_nodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Minimal ``type=con`` node shape — only the fields the detector reads."""
    return {
        "type": "con",
        "focused": focused,
        "app_id": app_id,
        "window_properties": window_properties,
        "nodes": nodes or [],
        "floating_nodes": floating_nodes or [],
    }


def _wrap_in_root(*containers: dict[str, Any]) -> dict[str, Any]:
    """Wrap a list of containers into a realistic sway tree skeleton."""
    return {
        "type": "root",
        "focused": False,
        "nodes": [
            {
                "type": "output",
                "focused": False,
                "nodes": [
                    {
                        "type": "workspace",
                        "focused": False,
                        "nodes": list(containers),
                        "floating_nodes": [],
                    }
                ],
                "floating_nodes": [],
            }
        ],
        "floating_nodes": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect_focus_wayland_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """A focused ``con`` node with ``app_id`` set → WAYLAND with that app_id."""
    tree = _wrap_in_root(_con(focused=True, app_id="foot", window_properties=None))
    monkeypatch.setattr(subprocess, "run", _ok_run_returning(tree))

    assert detect_focus() == FocusInfo(kind=FocusKind.WAYLAND, app_id="foot")


def test_detect_focus_xwayland(monkeypatch: pytest.MonkeyPatch) -> None:
    """A focused ``con`` with ``app_id=None`` and ``window_properties`` set → XWAYLAND.

    The ``app_id`` on the returned :class:`FocusInfo` is ``None`` — callers
    that need the X11 class read ``window_properties`` themselves, but voxd
    doesn't, hence we don't surface it.
    """
    tree = _wrap_in_root(
        _con(focused=True, app_id=None, window_properties={"class": "XTerm"})
    )
    monkeypatch.setattr(subprocess, "run", _ok_run_returning(tree))

    assert detect_focus() == FocusInfo(kind=FocusKind.XWAYLAND, app_id=None)


def test_detect_focus_no_focused_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """No node has ``focused=true`` among containers → UNKNOWN."""
    tree = _wrap_in_root(
        _con(focused=False, app_id="foot"),
        _con(focused=False, app_id="firefox"),
    )
    monkeypatch.setattr(subprocess, "run", _ok_run_returning(tree))

    assert detect_focus() == FocusInfo(kind=FocusKind.UNKNOWN, app_id=None)


def test_detect_focus_swaymsg_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``swaymsg`` binary absent (``FileNotFoundError``) → UNKNOWN."""

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", "swaymsg")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert detect_focus() == FocusInfo(kind=FocusKind.UNKNOWN, app_id=None)


def test_detect_focus_swaymsg_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``swaymsg`` exits non-zero (e.g. not running under sway) → UNKNOWN."""

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv, returncode=1, stdout="", stderr="not running sway"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert detect_focus() == FocusInfo(kind=FocusKind.UNKNOWN, app_id=None)


def test_detect_focus_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stdout is not valid JSON → UNKNOWN (never raises)."""

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="not json", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert detect_focus() == FocusInfo(kind=FocusKind.UNKNOWN, app_id=None)


def test_detect_focus_nested_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The focused container can be arbitrarily deep — the walk is recursive.

    Layout: root > output > workspace > parent con > **focused leaf con**.
    A focused descendant of a non-focused container must still be found.
    """
    leaf = _con(focused=True, app_id="kitty", window_properties=None)
    parent = _con(focused=False, app_id=None, window_properties=None, nodes=[leaf])
    tree = _wrap_in_root(parent)

    monkeypatch.setattr(subprocess, "run", _ok_run_returning(tree))

    assert detect_focus() == FocusInfo(kind=FocusKind.WAYLAND, app_id="kitty")


def test_detect_focus_ignores_focused_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``type=workspace`` with ``focused=true`` doesn't qualify as a window.

    Only ``type=con`` nodes are window containers; workspaces report focused
    when they are the active workspace but they aren't applications.
    """
    tree = {
        "type": "root",
        "focused": False,
        "nodes": [
            {
                "type": "output",
                "focused": False,
                "nodes": [
                    {
                        # The workspace itself is "focused" (active workspace)
                        # but it's not a window — must be ignored.
                        "type": "workspace",
                        "focused": True,
                        "app_id": None,
                        "window_properties": None,
                        "nodes": [],
                        "floating_nodes": [],
                    }
                ],
                "floating_nodes": [],
            }
        ],
        "floating_nodes": [],
    }

    monkeypatch.setattr(subprocess, "run", _ok_run_returning(tree))

    assert detect_focus() == FocusInfo(kind=FocusKind.UNKNOWN, app_id=None)


def test_detect_focus_floating_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Floating windows live under ``floating_nodes`` — the walk must visit them."""
    floating = _con(focused=True, app_id="org.wezfurlong.wezterm", window_properties=None)
    workspace = {
        "type": "workspace",
        "focused": False,
        "nodes": [],
        "floating_nodes": [floating],
    }
    tree = {
        "type": "root",
        "focused": False,
        "nodes": [
            {
                "type": "output",
                "focused": False,
                "nodes": [workspace],
                "floating_nodes": [],
            }
        ],
        "floating_nodes": [],
    }

    monkeypatch.setattr(subprocess, "run", _ok_run_returning(tree))

    assert detect_focus() == FocusInfo(
        kind=FocusKind.WAYLAND, app_id="org.wezfurlong.wezterm"
    )
