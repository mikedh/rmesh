"""Pytest fixtures and hooks for trimesh/rmesh comparison testing."""

import pytest

from .results import get_session_results, reset_session_results


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

        # Monkey-patch trimesh.Trimesh properties to log comparisons
        from .wrapper import patch_trimesh

        patch_trimesh()


def pytest_sessionfinish(session, exitstatus):
    """Write comparison results at the end of the test session."""
    if session.config.getoption("--compare-rmesh"):
        from .results import save_results

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


@pytest.fixture
def comparison_results():
    """Fixture that provides access to session results."""
    return get_session_results()
