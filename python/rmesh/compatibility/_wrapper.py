"""Composition-based wrapper for comparing and wrapping rmesh with trimesh.

Uses WrappedTrimesh with __getattr__ delegation instead of monkey-patching.
"""

import os
import timeit

import numpy as np

# ---------------------------------------------------------------------------
# Benchmark state
# ---------------------------------------------------------------------------

_benchmark_flag: bool = bool(os.environ.get("BENCHMARK_RMESH"))


def enable_benchmark() -> None:
    """Enable benchmark mode (dual execution, timing, comparison)."""
    global _benchmark_flag
    _benchmark_flag = True


def disable_benchmark() -> None:
    """Disable benchmark mode."""
    global _benchmark_flag
    _benchmark_flag = False


def is_benchmarking() -> bool:
    """Return True if benchmark mode is active."""
    return _benchmark_flag


# ---------------------------------------------------------------------------
# Converters: rmesh value -> trimesh-compatible value
# Only needed when PyO3 returns a type trimesh users don't expect.
# ---------------------------------------------------------------------------

_CONVERTERS: dict = {
    "center_mass": lambda v: np.array([v[0], v[1], v[2]]),
}

# Normalizers applied to *both* sides before comparison (benchmark only).
_COMPARISON_NORMALIZERS: dict = {
    "face_adjacency": lambda adj: (
        np.array([], dtype=np.int64).reshape((0, 2))
        if adj is None or len(adj) == 0
        else (
            lambda a: a[np.lexsort(a.T[::-1])]
        )(np.sort(np.asarray(adj, dtype=np.int64), axis=1))
    ),
}


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def from_trimesh(trimesh_mesh):
    """Create an rmesh.Trimesh from a fully-processed trimesh.Trimesh."""
    import rmesh

    vertices = np.asarray(trimesh_mesh.vertices, dtype=np.float64)
    faces = np.asarray(trimesh_mesh.faces, dtype=np.int64)

    # pass vertex normals if trimesh has computed them
    vertex_normals = None
    try:
        vn = trimesh_mesh.vertex_normals
        if vn is not None and len(vn) == len(vertices):
            vertex_normals = np.asarray(vn, dtype=np.float64)
    except Exception:
        pass

    # pass face colors if present (RGBA uint8)
    face_colors = None
    try:
        if hasattr(trimesh_mesh, "visual") and hasattr(
            trimesh_mesh.visual, "face_colors"
        ):
            fc = trimesh_mesh.visual.face_colors
            if fc is not None and len(fc) == len(faces):
                face_colors = np.asarray(fc, dtype=np.uint8)
                if face_colors.ndim != 2 or face_colors.shape[1] != 4:
                    face_colors = None
    except Exception:
        pass

    return rmesh.Trimesh.from_arrays(
        vertices,
        faces,
        vertex_normals=vertex_normals,
        face_colors=face_colors,
    )


def to_trimesh(rmesh_mesh):
    """Create a trimesh.Trimesh from an rmesh.Trimesh without reprocessing."""
    import trimesh

    return trimesh.Trimesh(
        vertices=np.asarray(rmesh_mesh.vertices),
        faces=np.asarray(rmesh_mesh.faces),
        process=False,
    )


# ---------------------------------------------------------------------------
# WrappedTrimesh
# ---------------------------------------------------------------------------


class WrappedTrimesh:
    """Wraps a trimesh.Trimesh, delegating to rmesh where available."""

    __slots__ = ("_rmesh", "_rmesh_failed", "_trimesh")

    def __init__(self, trimesh_mesh):
        object.__setattr__(self, "_trimesh", trimesh_mesh)
        object.__setattr__(self, "_rmesh", None)
        object.__setattr__(self, "_rmesh_failed", False)

    # Expose the wrapped class so isinstance(obj, trimesh.Trimesh) works.
    @property
    def __class__(self):
        return self._trimesh.__class__

    # -- lazy rmesh creation ------------------------------------------------

    def _ensure_rmesh(self):
        """Return cached rmesh.Trimesh or create one; None on failure."""
        rm = self._rmesh
        if rm is not None:
            return rm
        if self._rmesh_failed:
            return None
        try:
            rm = from_trimesh(self._trimesh)
            object.__setattr__(self, "_rmesh", rm)
            return rm
        except Exception:
            object.__setattr__(self, "_rmesh_failed", True)
            return None

    # -- attribute delegation -----------------------------------------------

    def __getattr__(self, name):
        # Private / dunder -> always trimesh
        if name.startswith("_"):
            return getattr(self._trimesh, name)

        rmesh_mesh = self._ensure_rmesh()

        if rmesh_mesh is not None and hasattr(rmesh_mesh, name):
            if is_benchmarking():
                self._benchmark(name, rmesh_mesh)
                return getattr(self._trimesh, name)
            try:
                val = getattr(rmesh_mesh, name)
                conv = _CONVERTERS.get(name)
                return conv(val) if conv else val
            except Exception:
                pass  # fall through to trimesh

        return getattr(self._trimesh, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._trimesh, name, value)

    def __repr__(self):
        return repr(self._trimesh)

    def __dir__(self):
        return dir(self._trimesh)

    def __len__(self):
        return len(self._trimesh)

    # -- benchmark helper ---------------------------------------------------

    def _benchmark(self, name, rmesh_mesh):
        """Run both implementations, compare, log result."""
        from ._results import ComparisonResult, get_session_results, values_equal

        tm = self._trimesh

        # Time trimesh
        try:
            t_timer = timeit.Timer(lambda: getattr(tm, name))
            trimesh_time = min(t_timer.repeat(repeat=3, number=1))
            trimesh_value = getattr(tm, name)
        except Exception:
            return  # can't benchmark if trimesh fails

        # Time rmesh
        rmesh_value = None
        rmesh_time = 0.0
        error = None
        try:
            r_timer = timeit.Timer(lambda: getattr(rmesh_mesh, name))
            rmesh_time = min(r_timer.repeat(repeat=3, number=1))
            raw = getattr(rmesh_mesh, name)
            conv = _CONVERTERS.get(name)
            rmesh_value = conv(raw) if conv else raw
        except Exception as exc:
            error = str(exc)

        # Compare
        identical = None
        if error is None and rmesh_value is not None:
            try:
                norm = _COMPARISON_NORMALIZERS.get(name)
                tv = norm(trimesh_value) if norm else trimesh_value
                rv = norm(rmesh_value) if norm else rmesh_value
                identical = values_equal(tv, rv)
            except Exception as exc:
                error = f"Comparison failed: {exc}"

        result = ComparisonResult(
            name=name,
            trimesh_time=trimesh_time,
            rmesh_time=rmesh_time,
            identical=identical,
            error=error,
        )
        get_session_results().add_result(result)


# ---------------------------------------------------------------------------
# WrappedScene
# ---------------------------------------------------------------------------


class WrappedScene:
    """Wraps a trimesh.Scene, wrapping each Trimesh geometry."""

    __slots__ = ("_scene", "_wrapped_geometry")

    def __init__(self, scene):
        import trimesh

        object.__setattr__(self, "_scene", scene)
        wrapped = {}
        for key, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                wrapped[key] = WrappedTrimesh(geom)
            else:
                wrapped[key] = geom
        object.__setattr__(self, "_wrapped_geometry", wrapped)

    @property
    def __class__(self):
        return self._scene.__class__

    @property
    def geometry(self):
        """Return geometry dict with wrapped Trimesh instances."""
        return self._wrapped_geometry

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def __repr__(self):
        return repr(self._scene)

    def __dir__(self):
        return dir(self._scene)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def wrap(obj):
    """Wrap a trimesh.Trimesh or trimesh.Scene with rmesh delegation.

    Returns the original object unchanged if it is not a recognized type.
    """
    import trimesh

    if isinstance(obj, trimesh.Trimesh):
        return WrappedTrimesh(obj)
    if isinstance(obj, trimesh.Scene):
        return WrappedScene(obj)
    return obj


def load_scene(path, **kwargs):
    """Load a mesh/scene via trimesh, then wrap with rmesh delegation."""
    import trimesh

    return wrap(trimesh.load(path, **kwargs))
