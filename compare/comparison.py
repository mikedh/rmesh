"""Comparison utilities for trimesh vs rmesh results."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ComparisonResult:
    """Result of comparing a single property/method between trimesh and rmesh."""

    name: str
    trimesh_value: Any = None
    rmesh_value: Any = None
    trimesh_time: float = 0.0
    rmesh_time: float = 0.0
    identical: bool | None = None
    error: str | None = None
    rmesh_implemented: bool = True

    @property
    def speedup(self) -> float | None:
        """Calculate speedup ratio (trimesh_time / rmesh_time)."""
        if self.trimesh_time > 0 and self.rmesh_time > 0:
            return self.trimesh_time / self.rmesh_time
        return None

    def to_row(self) -> list:
        """Convert to a table row."""
        speedup = self.speedup
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"

        if not self.rmesh_implemented:
            identical_str = "Not impl"
        elif self.error:
            identical_str = f"Error: {self.error[:20]}"
        elif self.identical is None:
            identical_str = "N/A"
        else:
            identical_str = "Yes" if self.identical else "No"

        return [
            self.name,
            f"{self.trimesh_time:.6f}" if self.trimesh_time > 0 else "N/A",
            f"{self.rmesh_time:.6f}" if self.rmesh_time > 0 else "N/A",
            speedup_str,
            identical_str,
        ]


def arrays_equal(
    a: np.ndarray, b: np.ndarray, rtol: float = 1e-7, atol: float = 1e-10
) -> bool:
    """Compare two numpy arrays with tolerance.

    Args:
        a: First array
        b: Second array
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if arrays are equal within tolerance
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        # Shape mismatch - can't directly compare
        # This is common for vertices/faces due to processing differences
        return False

    # Handle empty arrays
    if a.size == 0 and b.size == 0:
        return True

    # For integer arrays, compare as integers
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        return np.array_equal(a.astype(np.int64), b.astype(np.int64))

    # For float arrays, convert to same dtype and use allclose
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    return np.allclose(a_float, b_float, rtol=rtol, atol=atol)


def values_equal(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:  # noqa: C901
    """Type-aware comparison of values within tolerance.

    Args:
        a: First value
        b: Second value
        rtol: Relative tolerance for floats
        atol: Absolute tolerance for floats

    Returns:
        True (Python bool) if values are equal within tolerance
    """
    # Handle None
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    # Handle numpy arrays
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(arrays_equal(a, b, rtol, atol))

    # Handle floats
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return True
        return bool(np.isclose(a, b, rtol=rtol, atol=atol))

    # Handle booleans
    if isinstance(a, (bool, np.bool_)) and isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)

    # Handle integers
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)

    # Handle lists/tuples - convert to arrays if numeric
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        try:
            return bool(arrays_equal(np.array(a), np.array(b), rtol, atol))
        except (ValueError, TypeError):
            return a == b

    # Handle matrices (3x3 for inertia tensor)
    if hasattr(a, "shape") and hasattr(b, "shape"):
        return bool(arrays_equal(np.asarray(a), np.asarray(b), rtol, atol))

    # Default comparison
    return bool(a == b)
