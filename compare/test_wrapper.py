"""Unit tests for the compare wrapper."""

import numpy as np
import pytest


def test_from_arrays():
    """Test creating a comparison mesh from arrays."""
    from compare import TrimeshComparison

    # Simple tetrahedron
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], dtype=np.int64)

    mesh = TrimeshComparison.from_arrays(vertices, faces)

    # Check basic properties
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (4, 3)

    # Check results were recorded
    results = mesh.get_results()
    assert len(results) >= 2  # At least vertices and faces


def test_comparison_result():
    """Test ComparisonResult dataclass."""
    from compare import ComparisonResult

    result = ComparisonResult(
        name="test",
        trimesh_time=0.001,
        rmesh_time=0.0001,
        identical=True,
    )

    assert result.speedup == 10.0
    row = result.to_row()
    assert row[0] == "test"
    assert "10.00x" in row[3]
    assert row[4] == "Yes"


def test_arrays_equal():
    """Test array comparison utilities."""
    from compare import arrays_equal

    # Same arrays
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert arrays_equal(a, b)

    # Within tolerance
    c = np.array([1.0, 2.0, 3.0 + 1e-10])
    assert arrays_equal(a, c)

    # Different shapes
    d = np.array([1.0, 2.0])
    assert not arrays_equal(a, d)

    # Integer arrays
    e = np.array([1, 2, 3], dtype=np.int32)
    f = np.array([1, 2, 3], dtype=np.int64)
    assert arrays_equal(e, f)


def test_session_results():
    """Test session-level result aggregation."""
    from compare import ComparisonResult, get_session_results, reset_session_results

    reset_session_results()
    results = get_session_results()

    # Add some results
    results.add_result(
        ComparisonResult(
            name="volume", trimesh_time=0.01, rmesh_time=0.001, identical=True
        )
    )
    results.add_result(
        ComparisonResult(
            name="volume", trimesh_time=0.02, rmesh_time=0.002, identical=True
        )
    )

    summary = results.summary()
    assert summary["total"] == 2
    assert summary["identical"] == 2
    assert summary["speedup_avg"] == 10.0

    # Test JSON output
    json_str = results.to_json()
    assert "volume" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
