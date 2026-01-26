"""Compare package for trimesh vs rmesh comparison testing.

This package provides utilities to run trimesh tests while comparing
results with rmesh implementations. It generates timing and compatibility
reports.

Example usage:

    # Via pytest (recommended)
    pytest --compare-rmesh test/trimesh/tests/test_inertia.py
    # -> Generates comparison.md

    # Programmatic patching
    from compare import patch_trimesh
    patch_trimesh()  # Now trimesh.Trimesh properties log rmesh comparisons

    import trimesh
    mesh = trimesh.load("model.stl")
    print(mesh.volume)  # Compares both, returns trimesh result
"""

from .mapping import API_MAPPING, get_mapping, is_comparable
from .results import (
    ComparisonResult,
    TestSessionResults,
    arrays_equal,
    get_session_results,
    reset_session_results,
    values_equal,
)
from .wrapper import patch_trimesh

__all__ = [
    "API_MAPPING",
    "ComparisonResult",
    "TestSessionResults",
    "arrays_equal",
    "get_mapping",
    "get_session_results",
    "is_comparable",
    "patch_trimesh",
    "reset_session_results",
    "values_equal",
]
