"""Monkey-patch trimesh.Trimesh to compare with or wrap rmesh implementations."""

import functools
import timeit
from typing import Literal

import numpy as np

from ._mapping import API_MAPPING, get_mapping
from ._results import ComparisonResult, get_session_results, values_equal

# Track patching state
_patched = False
_current_mode: Literal["compare", "wrap"] | None = None
_originals: dict = {}


def _get_or_create_rmesh(trimesh_mesh, originals: dict):
    """Get or create an rmesh mesh from a trimesh mesh.

    Args:
        trimesh_mesh: The trimesh.Trimesh instance
        originals: Dict of original property/method references

    Returns:
        rmesh.Trimesh instance or None if creation fails
    """
    rmesh_mesh = getattr(trimesh_mesh, "_rmesh_cache", None)
    if rmesh_mesh is not None:
        return rmesh_mesh

    if getattr(trimesh_mesh, "_rmesh_failed", False):
        return None

    try:
        import rmesh

        # Use originals to avoid recursion when getting vertices/faces
        verts_prop = originals.get("vertices")
        faces_prop = originals.get("faces")
        if verts_prop and faces_prop:
            verts = (
                verts_prop.fget(trimesh_mesh)
                if isinstance(verts_prop, property)
                else verts_prop(trimesh_mesh)
            )
            faces = (
                faces_prop.fget(trimesh_mesh)
                if isinstance(faces_prop, property)
                else faces_prop(trimesh_mesh)
            )
            rmesh_mesh = rmesh.Trimesh(
                np.asarray(verts, dtype=np.float64),
                np.asarray(faces, dtype=np.int64),
            )
            # Cache it
            object.__setattr__(trimesh_mesh, "_rmesh_cache", rmesh_mesh)
            return rmesh_mesh
    except Exception:
        object.__setattr__(trimesh_mesh, "_rmesh_failed", True)

    return None


def make_comparing_property(name: str, original, originals: dict):
    """Create a property that compares trimesh and rmesh (compare mode).

    Returns trimesh values but logs comparison results.
    """

    @functools.wraps(original.fget if isinstance(original, property) else original)
    def comparing_getter(self):
        # Get trimesh value with timing
        try:
            if isinstance(original, property):
                timer = timeit.Timer(lambda: original.fget(self))
            else:
                timer = timeit.Timer(lambda: original(self))
            times = timer.repeat(repeat=3, number=1)
            trimesh_time = min(times)
            trimesh_value = (
                original.fget(self) if isinstance(original, property) else original(self)
            )
        except Exception:
            return (
                original.fget(self) if isinstance(original, property) else original(self)
            )

        # Try to get rmesh value
        mapping = get_mapping(name)
        if mapping is None or not mapping.rmesh_implemented:
            return trimesh_value

        rmesh_value = None
        rmesh_time = 0.0
        error = None

        rmesh_mesh = _get_or_create_rmesh(self, originals)

        if rmesh_mesh is not None:
            try:
                rmesh_name = mapping.rmesh_name
                attr = getattr(rmesh_mesh, rmesh_name)
                is_callable = callable(attr) and not mapping.is_property

                if is_callable:
                    timer = timeit.Timer(lambda: attr())
                else:
                    timer = timeit.Timer(lambda: getattr(rmesh_mesh, rmesh_name))

                times = timer.repeat(repeat=3, number=1)
                rmesh_time = min(times)
                rmesh_value = attr() if is_callable else attr

                if mapping.converter and rmesh_value is not None:
                    rmesh_value = mapping.converter(rmesh_value)
            except Exception as e:
                error = str(e)

        # Compare and log
        identical = None
        if error is None and rmesh_value is not None:
            try:
                compare_trimesh = trimesh_value
                if mapping.trimesh_converter and trimesh_value is not None:
                    compare_trimesh = mapping.trimesh_converter(trimesh_value)
                identical = values_equal(compare_trimesh, rmesh_value)
            except Exception as e:
                error = f"Comparison failed: {e}"

        if rmesh_mesh is not None or error:
            result = ComparisonResult(
                name=name,
                trimesh_value=trimesh_value,
                rmesh_value=rmesh_value,
                trimesh_time=trimesh_time,
                rmesh_time=rmesh_time,
                identical=identical,
                error=error,
                rmesh_implemented=mapping.rmesh_implemented and error is None,
            )
            get_session_results().add_result(result)

        return trimesh_value

    if isinstance(original, property):
        return property(comparing_getter, original.fset, original.fdel, original.__doc__)
    return comparing_getter


def make_wrapping_property(name: str, original, originals: dict):
    """Create a property that uses rmesh when available, fallback to trimesh (wrap mode).

    Returns rmesh values when available for better performance.
    """

    @functools.wraps(original.fget if isinstance(original, property) else original)
    def wrapping_getter(self):
        mapping = get_mapping(name)
        if mapping is None or not mapping.rmesh_implemented:
            # No rmesh mapping, use trimesh
            return (
                original.fget(self) if isinstance(original, property) else original(self)
            )

        rmesh_mesh = _get_or_create_rmesh(self, originals)

        if rmesh_mesh is not None:
            try:
                rmesh_name = mapping.rmesh_name
                attr = getattr(rmesh_mesh, rmesh_name)
                is_callable = callable(attr) and not mapping.is_property

                result = attr() if is_callable else attr

                if mapping.converter and result is not None:
                    result = mapping.converter(result)

                return result
            except Exception:
                pass  # Fall through to trimesh

        # Fallback to trimesh
        return (
            original.fget(self) if isinstance(original, property) else original(self)
        )

    if isinstance(original, property):
        return property(wrapping_getter, original.fset, original.fdel, original.__doc__)
    return wrapping_getter


def patch_trimesh(mode: Literal["compare", "wrap"] = "compare") -> None:
    """Monkey-patch trimesh.Trimesh to add comparison or wrapping behavior.

    Args:
        mode: "compare" - run both, log timing, return trimesh values
              "wrap" - return rmesh values, fallback to trimesh
    """
    global _patched, _current_mode, _originals

    if _patched:
        if _current_mode == mode:
            return  # Already patched with same mode
        unpatch_trimesh()  # Unpatch first if switching modes

    import trimesh

    # Store original property getters
    _originals = {}

    for name in API_MAPPING:
        if hasattr(trimesh.Trimesh, name):
            _originals[name] = getattr(trimesh.Trimesh, name)

    # Choose the property maker based on mode
    if mode == "compare":
        make_property = make_comparing_property
    else:
        make_property = make_wrapping_property

    # Patch each mapped property
    for name, original in _originals.items():
        try:
            setattr(trimesh.Trimesh, name, make_property(name, original, _originals))
        except Exception:
            pass  # Some attributes can't be patched, that's ok

    _patched = True
    _current_mode = mode


def unpatch_trimesh() -> None:
    """Remove monkey-patches and restore original trimesh behavior."""
    global _patched, _current_mode, _originals

    if not _patched:
        return

    import trimesh

    # Restore originals
    for name, original in _originals.items():
        try:
            setattr(trimesh.Trimesh, name, original)
        except Exception:
            pass

    _originals = {}
    _patched = False
    _current_mode = None
