"""Pytest integration for rmesh/trimesh compatibility testing.

Patches ``trimesh.load`` and ``trimesh.load_mesh`` to return
WrappedTrimesh / WrappedScene objects so the trimesh test suite
exercises rmesh transparently.
"""

import pytest

from ._results import get_session_results, reset_session_results


def pytest_addoption(parser):
    """Add CLI options for compatibility mode."""
    parser.addoption(
        "--rmesh-compat",
        choices=["compare", "wrap", "off"],
        default="off",
        help=(
            "Enable rmesh compatibility mode: "
            "compare, wrap, or off (default: off)"
        ),
    )
    # Deprecated alias kept for backward compatibility
    parser.addoption(
        "--compare-rmesh",
        action="store_true",
        default=False,
        help="[Deprecated] Use --rmesh-compat=compare instead",
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
    parser.addoption(
        "--comparison-db",
        type=str,
        default="comparison_results.db",
        help="SQLite database path for comparison results (default: comparison_results.db)",
    )


def pytest_configure(config):
    """Patch trimesh loaders when compatibility mode is requested."""
    compat_mode = config.getoption("--rmesh-compat")
    if compat_mode == "off" and config.getoption("--compare-rmesh"):
        compat_mode = "compare"

    if compat_mode not in ("compare", "wrap"):
        return

    db_path = config.getoption("--comparison-db")
    reset_session_results(db_path=db_path)

    import trimesh

    from ._wrapper import enable_benchmark, wrap

    _original_load = trimesh.load
    _original_load_mesh = getattr(trimesh, "load_mesh", None)

    def patched_load(*args, **kwargs):
        return wrap(_original_load(*args, **kwargs))

    trimesh.load = patched_load

    if _original_load_mesh is not None:

        def patched_load_mesh(*args, **kwargs):
            return wrap(_original_load_mesh(*args, **kwargs))

        trimesh.load_mesh = patched_load_mesh

    if compat_mode == "compare":
        enable_benchmark()


def pytest_sessionfinish(session, exitstatus):
    """Write comparison results at the end of the test session."""
    compat_mode = session.config.getoption("--rmesh-compat")
    if compat_mode == "off" and session.config.getoption("--compare-rmesh"):
        compat_mode = "compare"

    if compat_mode != "compare":
        return

    from ._results import save_results

    output_file = session.config.getoption("--comparison-output")
    json_file = session.config.getoption("--comparison-json")
    save_results(output_file, json_file)

    results = get_session_results()
    summary = results.summary()
    print("\n" + "=" * 60)
    print("rmesh Comparison Summary")
    print("=" * 60)
    print(f"Total comparisons: {summary['total']}")
    print(f"Identical results: {summary['identical']}")
    if summary["speedup_avg"]:
        print(
            f"Speedup: {summary['speedup_avg']:.1f}x avg "
            f"({summary['speedup_min']:.1f}x min, "
            f"{summary['speedup_max']:.1f}x max)"
        )
    print("=" * 60)

    results.close()


@pytest.fixture
def comparison_results():
    """Fixture that provides access to session results."""
    return get_session_results()
