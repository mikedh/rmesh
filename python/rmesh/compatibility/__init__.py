"""Public compatibility layer for using rmesh as a trimesh drop-in.

Wraps trimesh objects with composition-based delegation to rmesh.
Set ``BENCHMARK_RMESH=1`` to enable dual-execution benchmark mode.
"""

from ._results import (
    TestSessionResults,
    get_session_results,
    reset_session_results,
    save_results,
)
from ._wrapper import (
    WrappedScene,
    WrappedTrimesh,
    disable_benchmark,
    enable_benchmark,
    from_trimesh,
    is_benchmarking,
    load_scene,
    to_trimesh,
    wrap,
)

# Public aliases matching the plan's API surface.
get_results = get_session_results
reset_results = reset_session_results
save_report = save_results

__all__ = [
    "TestSessionResults",
    "WrappedScene",
    "WrappedTrimesh",
    "disable_benchmark",
    "enable_benchmark",
    "from_trimesh",
    "get_results",
    "is_benchmarking",
    "load_scene",
    "reset_results",
    "save_report",
    "to_trimesh",
    "wrap",
]
