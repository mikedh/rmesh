"""Monkey-patch trimesh.Trimesh to compare with rmesh implementations."""

import functools
import timeit

import numpy as np

from .mapping import API_MAPPING, get_mapping
from .results import ComparisonResult, get_session_results, values_equal

# Track whether we've already patched
_patched = False


def patch_trimesh():
    """Monkey-patch trimesh.Trimesh to add comparison behavior.

    This patches property getters to also call rmesh and log comparisons.
    The trimesh behavior is unchanged - we just add logging.
    """
    global _patched
    if _patched:
        return

    import trimesh

    # Store original property getters
    _originals = {}

    for name in API_MAPPING:
        if hasattr(trimesh.Trimesh, name):
            _originals[name] = getattr(trimesh.Trimesh, name)

    def make_comparing_property(name: str, original):
        """Create a property that compares trimesh and rmesh."""

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
                trimesh_value = original.fget(self) if isinstance(original, property) else original(self)
            except Exception:
                return original.fget(self) if isinstance(original, property) else original(self)

            # Try to get rmesh value
            mapping = get_mapping(name)
            if mapping is None or not mapping.rmesh_implemented:
                return trimesh_value

            rmesh_value = None
            rmesh_time = 0.0
            error = None

            # Get or create rmesh mesh from cache
            rmesh_mesh = getattr(self, '_rmesh_cache', None)
            if rmesh_mesh is None and not getattr(self, '_rmesh_failed', False):
                try:
                    import rmesh
                    # Use _originals to avoid recursion when getting vertices/faces
                    verts_prop = _originals.get('vertices')
                    faces_prop = _originals.get('faces')
                    if verts_prop and faces_prop:
                        verts = verts_prop.fget(self) if isinstance(verts_prop, property) else verts_prop(self)
                        faces = faces_prop.fget(self) if isinstance(faces_prop, property) else faces_prop(self)
                        rmesh_mesh = rmesh.Trimesh(
                            np.asarray(verts, dtype=np.float64),
                            np.asarray(faces, dtype=np.int64),
                        )
                        # Cache it
                        object.__setattr__(self, '_rmesh_cache', rmesh_mesh)
                except Exception as e:
                    object.__setattr__(self, '_rmesh_failed', True)
                    error = str(e)

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

    # Patch each mapped property
    for name, original in _originals.items():
        try:
            setattr(trimesh.Trimesh, name, make_comparing_property(name, original))
        except Exception:
            pass  # Some attributes can't be patched, that's ok

    _patched = True
