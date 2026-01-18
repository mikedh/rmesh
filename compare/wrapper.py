"""TrimeshComparison wrapper class that compares trimesh and rmesh implementations."""

import time
from pathlib import Path
from typing import Any

import numpy as np

from .comparison import ComparisonResult, values_equal
from .mapping import API_MAPPING, get_mapping
from .results import get_session_results


class TrimeshComparison:
    """Wrapper that runs both trimesh and rmesh, compares results.

    Always returns trimesh results to maintain compatibility with existing tests.
    Logs comparison results for analysis.
    """

    def __init__(
        self,
        trimesh_mesh: Any,
        rmesh_mesh: Any = None,
        source_path: str | None = None,
    ):
        """Initialize with trimesh and optionally rmesh mesh objects.

        Args:
            trimesh_mesh: The trimesh.Trimesh object
            rmesh_mesh: The rmesh.Trimesh object (optional)
            source_path: Path the mesh was loaded from (for logging)
        """
        self._trimesh = trimesh_mesh
        self._rmesh = rmesh_mesh
        self._source_path = source_path
        self._results: list[ComparisonResult] = []

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "TrimeshComparison":
        """Load a mesh from file using both trimesh and rmesh.

        Args:
            path: Path to mesh file
            **kwargs: Additional arguments passed to trimesh.load

        Returns:
            TrimeshComparison wrapping both meshes
        """
        # Import the original load function to avoid recursion
        # when conftest patches trimesh.load
        import trimesh
        from trimesh.exchange.load import load_mesh as trimesh_load_mesh

        path = Path(path)
        path_str = str(path)

        # Load with trimesh using internal loader
        trimesh_mesh = trimesh_load_mesh(path_str, **kwargs)

        # Handle Scene objects - get first geometry
        if hasattr(trimesh_mesh, "geometry"):
            if len(trimesh_mesh.geometry) > 0:
                trimesh_mesh = next(iter(trimesh_mesh.geometry.values()))
            else:
                trimesh_mesh = trimesh.Trimesh()

        # Try to load with rmesh
        rmesh_mesh = None
        try:
            import rmesh

            rmesh_mesh = rmesh.load_mesh(path_str)
            # Apply cleanup to match trimesh's default vertex merging behavior
            rmesh_mesh = rmesh_mesh.cleanup(merge_vertices=8)
        except Exception:
            # Log but don't fail - we can still return trimesh results
            pass

        return cls(trimesh_mesh, rmesh_mesh, path_str)

    @classmethod
    def from_arrays(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        **kwargs,
    ) -> "TrimeshComparison":
        """Create from vertex and face arrays using both libraries.

        Args:
            vertices: (n, 3) array of vertex positions
            faces: (m, 3) array of face indices
            **kwargs: Additional arguments passed to trimesh.Trimesh

        Returns:
            TrimeshComparison wrapping both meshes
        """
        import trimesh

        # Create with trimesh
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, **kwargs)

        # Try to create with rmesh
        rmesh_mesh = None
        try:
            import rmesh

            rmesh_mesh = rmesh.Trimesh(
                np.asarray(vertices, dtype=np.float64),
                np.asarray(faces, dtype=np.int64),
            )
        except Exception:
            pass

        return cls(trimesh_mesh, rmesh_mesh, source_path="from_arrays")

    def _get_trimesh_value(self, name: str) -> tuple[Any, float]:
        """Get a value from trimesh and measure time.

        Returns:
            (value, time_seconds)
        """
        start = time.perf_counter()
        try:
            attr = getattr(self._trimesh, name)
            if callable(attr):
                value = attr()
            else:
                value = attr
        except Exception:
            return None, 0.0
        elapsed = time.perf_counter() - start
        return value, elapsed

    def _get_rmesh_value(self, name: str, mapping) -> tuple[Any, float, str | None]:
        """Get a value from rmesh and measure time.

        Returns:
            (value, time_seconds, error_message)
        """
        if self._rmesh is None:
            return None, 0.0, "rmesh not loaded"

        if not mapping.rmesh_implemented:
            return None, 0.0, "Not exposed to Python"

        rmesh_name = mapping.rmesh_name
        start = time.perf_counter()
        try:
            attr = getattr(self._rmesh, rmesh_name)
            if callable(attr) and not mapping.is_property:
                value = attr()
            else:
                value = attr

            # Apply converter if specified
            if mapping.converter and value is not None:
                value = mapping.converter(value)

        except AttributeError:
            return None, 0.0, f"rmesh has no attribute '{rmesh_name}'"
        except Exception as e:
            return None, 0.0, str(e)

        elapsed = time.perf_counter() - start
        return value, elapsed, None

    def _compare_and_log(self, name: str) -> Any:
        """Compare trimesh and rmesh values, log result, return trimesh value."""
        mapping = get_mapping(name)

        # Get trimesh value (always)
        trimesh_value, trimesh_time = self._get_trimesh_value(name)

        # If no mapping, just return trimesh value
        if mapping is None:
            return trimesh_value

        # Get rmesh value
        rmesh_value, rmesh_time, error = self._get_rmesh_value(name, mapping)

        # Compare values (apply converters for comparison only)
        identical = None
        if error is None and rmesh_value is not None:
            try:
                # Apply trimesh converter if specified (for comparison only)
                compare_trimesh = trimesh_value
                if mapping.trimesh_converter and trimesh_value is not None:
                    compare_trimesh = mapping.trimesh_converter(trimesh_value)
                identical = values_equal(compare_trimesh, rmesh_value)
            except Exception as e:
                error = f"Comparison failed: {e}"

        # Only record result if rmesh actually loaded (skip when rmesh failed)
        if self._rmesh is not None:
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
            self._results.append(result)
            get_session_results().add_result(result)

        return trimesh_value

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access to compare implementations.

        This is called when an attribute is not found on the wrapper itself.
        """
        # Check if this is a comparable property
        if name in API_MAPPING:
            return self._compare_and_log(name)

        # Otherwise, just forward to trimesh
        attr = getattr(self._trimesh, name)

        # If it's a method, wrap it to return wrapper for chaining
        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # If result is a Trimesh, wrap it
                if hasattr(result, "vertices") and hasattr(result, "faces"):
                    return TrimeshComparison(result, None, self._source_path)
                return result

            return wrapper

        return attr

    def get_results(self) -> list[ComparisonResult]:
        """Get all comparison results for this mesh."""
        return self._results.copy()

    def get_results_table(self) -> str:
        """Format results as a markdown table."""
        if not self._results:
            return "No comparison results yet."

        lines = [
            "| Property | trimesh (s) | rmesh (s) | Speedup | Identical? |",
            "|----------|-------------|-----------|---------|------------|",
        ]

        for result in self._results:
            row = result.to_row()
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")

        return "\n".join(lines)

    # Properties that should be directly forwarded without comparison
    @property
    def visual(self):
        """Forward visual property directly."""
        return self._trimesh.visual

    @visual.setter
    def visual(self, value):
        """Forward visual setter directly."""
        self._trimesh.visual = value

    @property
    def metadata(self):
        """Forward metadata property directly."""
        return self._trimesh.metadata

    @metadata.setter
    def metadata(self, value):
        """Forward metadata setter directly."""
        self._trimesh.metadata = value

    # Make wrapper work like the underlying trimesh object
    def __len__(self):
        return len(self._trimesh.faces)

    def __repr__(self):
        return f"TrimeshComparison({self._trimesh!r})"

    def copy(self):
        """Create a copy of the wrapper."""
        return TrimeshComparison(
            self._trimesh.copy(),
            None,  # Don't copy rmesh for now
            self._source_path,
        )
