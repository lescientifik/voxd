"""Interactive ``voxd setup`` — placeholder, implemented in Step 10.

The full implementation will prompt for an OpenRouter API key, write the
TOML config, and drop a sway snippet at ``~/.config/sway/config.d/voxd.conf``.
For now this stub exists so the CLI dispatcher in :mod:`voxd.cli` can wire
the subcommand without an ``ImportError``.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Print a "not implemented" message and exit non-zero.

    Returns 1 so a script piping ``voxd setup || ...`` notices the failure.
    """
    print("voxd setup: not implemented yet (coming in Step 10)", file=sys.stderr)
    return 1


__all__ = ["main"]
