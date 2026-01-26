"""Compatibility layer for comparing and wrapping rmesh with trimesh.

This module provides two modes:
- "compare": Run both, log timing/equality, return trimesh values (for testing)
- "wrap": Return rmesh values when available, fallback to trimesh (for users)

Example usage:
    # Compare mode (testing)
    from rmesh._compatibility import enable
    enable("compare")

    import trimesh
    mesh = trimesh.load("model.stl")
    print(mesh.volume)  # Runs both, logs comparison, returns trimesh value

    # Wrap mode (users)
    from rmesh._compatibility import enable
    enable("wrap")

    import trimesh
    mesh = trimesh.load("model.stl")
    print(mesh.volume)  # Uses rmesh (faster) when available
"""

from contextlib import contextmanager
from typing import Literal

from ._results import (
    TestSessionResults,
    get_session_results,
    reset_session_results,
    save_results,
)
from ._wrapper import patch_trimesh, unpatch_trimesh

__all__ = [
    "TestSessionResults",
    "compatibility_mode",
    "disable",
    "enable",
    "get_mode",
    "get_results",
    "is_enabled",
    "reset_results",
    "save_report",
]

# Current mode state
_current_mode: Literal["compare", "wrap"] | None = None


def enable(mode: Literal["compare", "wrap"] = "wrap") -> None:
    """Enable compatibility layer.

    Args:
        mode: "compare" - run both, log timing, return trimesh values
              "wrap" - return rmesh values, fallback to trimesh
    """
    global _current_mode

    if _current_mode is not None:
        if _current_mode == mode:
            return  # Already enabled with same mode
        disable()  # Disable first if switching modes

    patch_trimesh(mode=mode)
    _current_mode = mode


def disable() -> None:
    """Disable compatibility layer, restore original trimesh."""
    global _current_mode

    if _current_mode is None:
        return

    unpatch_trimesh()
    _current_mode = None


@contextmanager
def compatibility_mode(mode: Literal["compare", "wrap"] = "wrap"):
    """Context manager for scoped compatibility mode.

    Args:
        mode: "compare" or "wrap"

    Example:
        with compatibility_mode("wrap"):
            mesh = trimesh.load("model.stl")
            print(mesh.volume)  # Uses rmesh
    """
    previous_mode = _current_mode
    try:
        enable(mode)
        yield
    finally:
        disable()
        if previous_mode is not None:
            enable(previous_mode)


def is_enabled() -> bool:
    """Check if compatibility mode is enabled."""
    return _current_mode is not None


def get_mode() -> Literal["compare", "wrap"] | None:
    """Get the current compatibility mode."""
    return _current_mode


def get_results() -> TestSessionResults:
    """Get the current session results (compare mode only)."""
    return get_session_results()


def reset_results() -> None:
    """Reset all comparison results."""
    reset_session_results()


def save_report(path: str = "comparison.md", json_path: str | None = None) -> None:
    """Save comparison report to file.

    Args:
        path: Path for markdown output (default: comparison.md)
        json_path: Optional path for JSON output
    """
    save_results(path, json_path)
