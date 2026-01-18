"""Pytest fixtures and hooks for trimesh/rmesh comparison testing."""

from pathlib import Path

import pytest

from .results import get_session_results, reset_session_results
from .wrapper import TrimeshComparison


def pytest_addoption(parser):
    """Add custom pytest options for comparison mode."""
    parser.addoption(
        "--compare-rmesh",
        action="store_true",
        default=False,
        help="Enable rmesh comparison mode: run tests with both trimesh and rmesh",
    )
    parser.addoption(
        "--comparison-output",
        type=str,
        default="comparison.md",
        help="Output file for comparison results (default: comparison.md)",
    )
    parser.addoption(
        "--comparison-json",
        type=str,
        default=None,
        help="Output JSON file for comparison results (optional)",
    )


def pytest_configure(config):
    """Configure pytest with comparison mode if enabled."""
    if config.getoption("--compare-rmesh"):
        # Reset session results at the start
        reset_session_results()

        # Monkey-patch trimesh module
        _patch_trimesh_module()


def pytest_sessionfinish(session, exitstatus):
    """Write comparison results at the end of the test session."""
    if session.config.getoption("--compare-rmesh"):
        from .results import get_session_results, save_results

        # Write markdown output
        output_file = session.config.getoption("--comparison-output")
        json_file = session.config.getoption("--comparison-json")
        save_results(output_file, json_file)

        # Print summary to console
        results = get_session_results()
        summary = results.summary()
        print("\n" + "=" * 60)
        print("rmesh Comparison Summary")
        print("=" * 60)
        print(f"Total comparisons: {summary['total']}")
        print(f"Identical results: {summary['identical']}")
        if summary["speedup_avg"]:
            avg = summary["speedup_avg"]
            min_s = summary["speedup_min"]
            max_s = summary["speedup_max"]
            print(f"Speedup: {avg:.1f}x avg ({min_s:.1f}x min, {max_s:.1f}x max)")
        print("=" * 60)


def _patch_trimesh_module():  # noqa: C901
    """Monkey-patch trimesh to use TrimeshComparison wrapper.

    Only patches trimesh.load, not trimesh.Trimesh (to avoid breaking isinstance checks).
    """
    import trimesh

    # Store original functions at module level
    global _original_load, _original_Trimesh
    _original_load = trimesh.load
    _original_Trimesh = trimesh.Trimesh

    def patched_load(file_obj, *args, **kwargs):
        """Patched trimesh.load that returns TrimeshComparison wrapper."""
        # Call original to get trimesh result
        result = _original_load(file_obj, *args, **kwargs)

        # Wrap the result
        return _wrap_geometry(result, file_obj)

    def _wrap_geometry(geom, source=None):  # noqa: C901
        """Wrap a trimesh geometry object in TrimeshComparison if possible."""
        source_path = str(source) if source else "unknown"

        # Check if it's a Trimesh using the original class
        if isinstance(geom, _original_Trimesh):
            # Try to load with rmesh too
            rmesh_mesh = None
            if source and not hasattr(source, "read"):
                try:
                    import rmesh

                    path = Path(source) if not isinstance(source, Path) else source
                    if path.exists():
                        rmesh_mesh = rmesh.load_mesh(str(path))
                        # Apply cleanup to match trimesh's default vertex merging behavior
                        rmesh_mesh = rmesh_mesh.cleanup(merge_vertices=8)
                except BaseException as e:
                    # Catch PanicException from Rust (inherits BaseException)
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    pass
            return TrimeshComparison(geom, rmesh_mesh, source_path=source_path)

        elif hasattr(geom, "geometry"):
            # Scene - wrap first geometry if there's only one
            if len(geom.geometry) == 1:
                _, g = next(iter(geom.geometry.items()))
                if isinstance(g, _original_Trimesh):
                    # Try to load with rmesh too
                    rmesh_mesh = None
                    if source and not hasattr(source, "read"):
                        try:
                            import rmesh

                            path = (
                                Path(source) if not isinstance(source, Path) else source
                            )
                            if path.exists():
                                rmesh_mesh = rmesh.load_mesh(str(path))
                                # Apply cleanup to match trimesh's vertex merging
                                rmesh_mesh = rmesh_mesh.cleanup(merge_vertices=8)
                        except BaseException as e:
                            # Catch PanicException from Rust (inherits BaseException)
                            if isinstance(e, KeyboardInterrupt):
                                raise
                            pass
                    return TrimeshComparison(g, rmesh_mesh, source_path=source_path)

        return geom

    # Only patch load - leave Trimesh class as-is to preserve isinstance checks
    trimesh.load = patched_load


# Store originals
_original_load = None
_original_Trimesh = None


@pytest.fixture
def comparison_wrapper():
    """Fixture that provides the TrimeshComparison class."""
    return TrimeshComparison


@pytest.fixture
def comparison_results():
    """Fixture that provides access to session results."""
    return get_session_results()
