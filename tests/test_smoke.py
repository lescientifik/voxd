"""Smoke test ensuring the package imports and exposes a version."""

from voxd import __version__


def test_version_is_a_non_empty_string() -> None:
    """voxd should advertise a semantic version string at import time."""
    assert isinstance(__version__, str)
    assert __version__
