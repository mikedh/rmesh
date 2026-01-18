"""Compare package for trimesh vs rmesh comparison testing.

This package provides utilities to run trimesh tests while comparing
results with rmesh implementations. It generates timing and compatibility
reports.

Example usage:

    # Direct usage
    from compare import TrimeshComparison

    mesh = TrimeshComparison.from_file("model.stl")
    print(mesh.volume)          # Compares both, returns trimesh result
    print(mesh.is_watertight)   # Compares both
    print(mesh.get_results_table())  # Show comparison table

    # Via pytest
    pytest --compare-rmesh test/trimesh/tests/test_inertia.py
    # -> Generates comparison_results.md
"""

from .comparison import ComparisonResult, arrays_equal, values_equal
from .mapping import API_MAPPING, get_mapping, is_comparable
from .results import (
    TestSessionResults,
    get_session_results,
    reset_session_results,
)
from .wrapper import TrimeshComparison

__all__ = [
    "API_MAPPING",
    "ComparisonResult",
    "TestSessionResults",
    "TrimeshComparison",
    "arrays_equal",
    "get_mapping",
    "get_session_results",
    "is_comparable",
    "reset_session_results",
    "values_equal",
]
