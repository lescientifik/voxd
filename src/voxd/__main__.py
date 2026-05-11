"""Allow `python -m voxd` to dispatch to the CLI entry point."""

from voxd.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
