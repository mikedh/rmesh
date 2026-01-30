"""Pytest integration for rmesh/trimesh compatibility testing.

All monkey-patching and benchmark infrastructure lives here.  In
**compare** mode ``trimesh.Trimesh.__getattribute__`` is patched so that
every public property access that rmesh also implements triggers a timed
comparison.  In **wrap** mode the patch is installed but benchmarking is
disabled (only rmesh routing is active).
"""

import threading
import timeit
import weakref

import numpy as np
import pytest

from .results import ComparisonResult, get_session_results, reset_session_results

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_benchmark: bool = False
_original_getattr = None
_rmesh_props: frozenset | None = None
_guard = threading.local()
_companions: dict = {}  # id(trimesh_obj) -> rmesh.Trimesh

# Types that can be meaningfully compared between implementations.
_COMPARABLE_TYPES = (np.ndarray, np.generic, bool, int, float, tuple, list)

# Normalizers applied to *both* sides before comparison.
_NORMALIZERS: dict = {
    "face_adjacency": lambda adj: (
        np.array([], dtype=np.int64).reshape((0, 2))
        if adj is None or len(adj) == 0
        else (
            lambda a: a[np.lexsort(a.T[::-1])]
        )(np.sort(np.asarray(adj, dtype=np.int64), axis=1))
    ),
}


# ---------------------------------------------------------------------------
# Benchmark recording
# ---------------------------------------------------------------------------


def _record_benchmark(tm, rm, name):
    """
    Time both implementations, compare results, and record.

    Parameters
    ----------
    tm : trimesh.Trimesh
        The trimesh mesh instance.
    rm : rmesh.Trimesh
        The rmesh mesh instance.
    name : str
        The property/attribute name being benchmarked.
    """
    from .results import values_equal

    # Get the trimesh value via the original, unpatched accessor.
    try:
        trimesh_value = _original_getattr(tm, name)
    except Exception:
        return

    # Only benchmark data properties, not bound methods / complex objects.
    if callable(trimesh_value):
        return
    if trimesh_value is not None and not isinstance(
        trimesh_value, _COMPARABLE_TYPES
    ):
        return

    # --- time trimesh (single shot) ---
    try:
        t0 = timeit.default_timer()
        _original_getattr(tm, name)
        trimesh_time = timeit.default_timer() - t0
    except Exception:
        return

    # --- time rmesh (single shot) ---
    rmesh_value = None
    rmesh_time = 0.0
    error = None
    try:
        t0 = timeit.default_timer()
        rmesh_value = getattr(rm, name)
        rmesh_time = timeit.default_timer() - t0
    except Exception as exc:
        error = str(exc)

    # --- compare ---
    identical = None
    if error is None and rmesh_value is not None:
        try:
            norm = _NORMALIZERS.get(name)
            tv = norm(trimesh_value) if norm else trimesh_value
            rv = norm(rmesh_value) if norm else rmesh_value
            identical = values_equal(tv, rv)
        except Exception as exc:
            error = f"Comparison failed: {exc}"

    get_session_results().add_result(
        ComparisonResult(
            name=name,
            trimesh_time=trimesh_time,
            rmesh_time=rmesh_time,
            identical=identical,
            error=error,
        )
    )


# ---------------------------------------------------------------------------
# Monkey-patching
# ---------------------------------------------------------------------------


def _patch_trimesh():
    """
    Patch ``trimesh.Trimesh.__getattribute__`` to intercept property access.

    Idempotent: returns immediately if already patched.
    """
    global _original_getattr, _rmesh_props

    if _original_getattr is not None:
        return

    import trimesh

    import rmesh

    _rmesh_props = frozenset(
        n for n in dir(rmesh.Trimesh) if not n.startswith("_")
    )
    _original_getattr = trimesh.Trimesh.__getattribute__

    def _patched(self, name):
        if (
            name.startswith("_")
            or name not in _rmesh_props
            or not _benchmark
            or getattr(_guard, "active", False)
        ):
            return _original_getattr(self, name)

        _guard.active = True
        try:
            # Get or create rmesh.Trimesh companion
            obj_id = id(self)
            rm = _companions.get(obj_id)
            if rm is None:
                try:
                    vertices = np.asarray(
                        _original_getattr(self, "vertices"), dtype=np.float64
                    )
                    faces = np.asarray(
                        _original_getattr(self, "faces"), dtype=np.int64
                    )
                    rm = rmesh.Trimesh(vertices, faces)
                except Exception:
                    return _original_getattr(self, name)
                except BaseException:
                    # pyo3 PanicException inherits BaseException, not Exception
                    return _original_getattr(self, name)
                _companions[obj_id] = rm
                weakref.finalize(self, _companions.pop, obj_id, None)

            _record_benchmark(self, rm, name)
            # Always return trimesh value so tests see trimesh results.
            return _original_getattr(self, name)
        finally:
            _guard.active = False

    trimesh.Trimesh.__getattribute__ = _patched


def _unpatch_trimesh():
    """Restore the original ``trimesh.Trimesh.__getattribute__``."""
    global _original_getattr
    if _original_getattr is not None:
        import trimesh

        trimesh.Trimesh.__getattribute__ = _original_getattr
        _original_getattr = None


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


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
        help="SQLite database path for comparison results",
    )


def _disable_broken_blender_boolean():
    """Unconditionally remove ``'blender'`` from boolean engines."""
    import trimesh.boolean as _bool
    from trimesh.exceptions import ExceptionWrapper

    engine = _bool._engines.get("blender")
    if engine is None or isinstance(engine, ExceptionWrapper):
        return

    _bool._engines["blender"] = ExceptionWrapper(
        ImportError("blender boolean engine disabled")
    )
    _bool.engines_available.discard("blender")


def pytest_configure(config):
    """Patch trimesh for compatibility / comparison mode."""
    global _benchmark

    _disable_broken_blender_boolean()

    compat_mode = config.getoption("--rmesh-compat")
    if compat_mode == "off" and config.getoption("--compare-rmesh"):
        compat_mode = "compare"

    if compat_mode not in ("compare", "wrap"):
        return

    db_path = config.getoption("--comparison-db")
    reset_session_results(db_path=db_path)

    if compat_mode == "compare":
        _benchmark = True
    _patch_trimesh()


def pytest_sessionfinish(session, exitstatus):
    """Write comparison results at the end of the test session."""
    compat_mode = session.config.getoption("--rmesh-compat")
    if compat_mode == "off" and session.config.getoption("--compare-rmesh"):
        compat_mode = "compare"

    if compat_mode != "compare":
        return

    from .results import save_results

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
