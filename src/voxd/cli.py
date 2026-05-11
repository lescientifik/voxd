"""Command-line entry point for voxd."""

import argparse

from voxd import __version__


def main() -> int:
    """Parse arguments and dispatch to the appropriate sub-command.

    Returns the process exit code. Sub-commands (``setup``, ``doctor``,
    ``config``) are wired up in a later step; for now only ``--version``
    is supported and invoking ``voxd`` with no arguments is a no-op that
    exits cleanly.
    """
    parser = argparse.ArgumentParser(
        prog="voxd",
        description="Headless voice-to-text daemon for Wayland/sway.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"voxd {__version__}",
    )
    parser.parse_args()
    # Sub-commands (setup, doctor, config) wired at step 9; daemon at step 8.
    return 0
