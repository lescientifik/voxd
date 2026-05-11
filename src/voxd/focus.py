"""sway focused-window detector.

This module is a thin, defensive wrapper around ``swaymsg -t get_tree``: it
invokes the binary, parses the JSON, walks the tree to find the focused
*window container* (a node with ``type == "con"`` and ``focused == True``),
and returns a typed verdict.

The verdict distinguishes three cases used downstream by the injection
pipeline:

* :class:`FocusKind.WAYLAND` — Wayland-native client (``app_id`` is set).
* :class:`FocusKind.XWAYLAND` — Xwayland client (``window_properties`` is
  populated, ``app_id`` is ``None``).
* :class:`FocusKind.UNKNOWN` — anything we can't classify. Every failure
  mode (binary missing, exit non-zero, JSON parse failure, no focused
  container in the tree) collapses to UNKNOWN; callers pick a safe fallback.

The module is pure plumbing: it knows nothing about wtype, terminals, or
voxd's injection policy.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any


class FocusKind(Enum):
    """What kind of client owns the currently-focused container."""

    WAYLAND = "wayland"
    """Wayland-native client — ``app_id`` is set on the sway node."""

    XWAYLAND = "xwayland"
    """Xwayland client — ``window_properties`` is set, ``app_id`` is null."""

    UNKNOWN = "unknown"
    """Indeterminate — swaymsg failed, parse failed, or no focused container."""


@dataclass(frozen=True)
class FocusInfo:
    """Verdict of :func:`detect_focus`.

    Attributes:
        kind: The classification of the focused container.
        app_id: The Wayland ``app_id`` of the focused window. Only set when
            ``kind == FocusKind.WAYLAND``; ``None`` otherwise (including for
            Xwayland — voxd doesn't currently look at the X11 class).
    """

    kind: FocusKind
    app_id: str | None = None


# Subprocess timeout for `swaymsg -t get_tree`. The call is normally
# sub-millisecond; the timeout is just a guardrail against a wedged IPC
# socket so we don't block the injection pipeline indefinitely.
_SWAYMSG_TIMEOUT_S = 2.0


def detect_focus() -> FocusInfo:
    """Return the focused window's classification.

    Queries ``swaymsg -t get_tree``, parses the JSON, and walks the tree.
    All failure modes (swaymsg missing, exit non-zero, invalid JSON,
    no focused ``con`` node anywhere) collapse to
    :class:`FocusKind.UNKNOWN` — callers must handle UNKNOWN as the
    "we can't tell" branch, not as an error to surface.

    Returns:
        A :class:`FocusInfo` describing the focused window.
    """
    tree = _query_tree()
    if tree is None:
        return FocusInfo(kind=FocusKind.UNKNOWN)

    focused = _find_focused_con(tree)
    if focused is None:
        return FocusInfo(kind=FocusKind.UNKNOWN)

    return _classify(focused)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _query_tree() -> dict[str, Any] | None:
    """Invoke ``swaymsg -t get_tree`` and return the parsed JSON.

    Returns ``None`` on any failure (binary missing, non-zero exit,
    timeout, invalid JSON, or non-dict root payload). Never raises.
    """
    try:
        result = subprocess.run(
            ["swaymsg", "-t", "get_tree"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_SWAYMSG_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    try:
        tree = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(tree, dict):
        return None
    return tree


def _iter_nodes(root: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield every dict node reachable from ``root`` (DFS).

    Walks both ``nodes`` (tiled children) and ``floating_nodes`` (floating
    children) at each level. Non-dict entries are skipped silently — the
    parser is intentionally lenient so a malformed-but-recoverable tree
    doesn't crash the detector.
    """
    stack: list[dict[str, Any]] = [root]
    while stack:
        node = stack.pop()
        yield node
        for key in ("nodes", "floating_nodes"):
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        stack.append(child)


def _find_focused_con(tree: dict[str, Any]) -> dict[str, Any] | None:
    """Find the focused window container in the tree.

    Only ``type == "con"`` nodes qualify — workspaces and outputs report
    ``focused`` flags that don't correspond to a window, and we must not
    pick them up.

    Returns:
        The first focused ``con`` node found, or ``None`` if none exists.
    """
    for node in _iter_nodes(tree):
        if node.get("type") == "con" and node.get("focused") is True:
            return node
    return None


def _classify(node: dict[str, Any]) -> FocusInfo:
    """Map a focused ``con`` node to a :class:`FocusInfo`.

    A Wayland-native client populates ``app_id``; an Xwayland client leaves
    ``app_id`` null and populates ``window_properties``. If neither hint is
    present we fall back to UNKNOWN — better a conservative fallback than
    a wrong classification.
    """
    app_id = node.get("app_id")
    if isinstance(app_id, str) and app_id:
        return FocusInfo(kind=FocusKind.WAYLAND, app_id=app_id)

    if isinstance(node.get("window_properties"), dict):
        return FocusInfo(kind=FocusKind.XWAYLAND)

    return FocusInfo(kind=FocusKind.UNKNOWN)


__all__ = ["FocusInfo", "FocusKind", "detect_focus"]
