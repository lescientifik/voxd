"""``voxd doctor`` diagnostics — placeholder, implemented in Step 11.

The full implementation will probe for ``wtype``, ``wl-copy``, ``notify-send``,
PortAudio, and mic access. For now this stub lets :mod:`voxd.cli` register the
subcommand without an ``ImportError``.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Print a "not implemented" message and exit non-zero.

    Returns 1 so callers can detect the placeholder state programmatically.
    """
    print("voxd doctor: not implemented yet (coming in Step 11)", file=sys.stderr)
    return 1


__all__ = ["main"]
