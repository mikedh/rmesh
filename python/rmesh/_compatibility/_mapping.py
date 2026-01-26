"""API mapping between trimesh and rmesh."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class APIMapping:
    """Mapping of a trimesh property/method to its rmesh equivalent."""

    rmesh_name: str
    is_property: bool = True
    converter: Callable[[Any], Any] | None = None  # Applied to rmesh value
    trimesh_converter: Callable[[Any], Any] | None = None  # Applied to trimesh value
    rmesh_implemented: bool = True  # Currently exposed to Python


# Converter functions for different data types


def convert_bounds(rmesh_bounds: tuple) -> np.ndarray:
    """Convert rmesh bounds (min, max) points to trimesh (2, 3) array."""
    if rmesh_bounds is None:
        return None
    min_pt, max_pt = rmesh_bounds
    return np.array(
        [[min_pt[0], min_pt[1], min_pt[2]], [max_pt[0], max_pt[1], max_pt[2]]]
    )


def convert_center_mass(rmesh_cm: Any) -> np.ndarray:
    """Convert rmesh center_mass Vector3 to numpy array."""
    if rmesh_cm is None:
        return None
    return np.array([rmesh_cm[0], rmesh_cm[1], rmesh_cm[2]])


def sort_adjacency_pairs(adj: np.ndarray) -> np.ndarray:
    """Sort face_adjacency pairs for consistent comparison.

    Face adjacency pairs can be in any order, so we sort each pair
    and then sort the array lexicographically for consistent comparison.
    """
    if adj is None or len(adj) == 0:
        return np.array([], dtype=np.int64).reshape((0, 2))
    arr = np.asarray(adj, dtype=np.int64)
    # Sort each pair so (3,1) becomes (1,3)
    arr = np.sort(arr, axis=1)
    # Sort rows lexicographically
    arr = arr[np.lexsort(arr.T[::-1])]
    return arr


def convert_inertia_matrix(rmesh_inertia: Any) -> np.ndarray:
    """Convert rmesh 3x3 inertia matrix to numpy array."""
    if rmesh_inertia is None:
        return None
    # rmesh returns a 3x3 matrix, convert to numpy
    return np.array(rmesh_inertia)


# API Mapping: trimesh_name -> APIMapping
# Properties and methods that should be compared between trimesh and rmesh
API_MAPPING: dict[str, APIMapping] = {
    # Core geometry (already exposed)
    "vertices": APIMapping("vertices", is_property=True),
    "faces": APIMapping("faces", is_property=True),
    # Properties currently exposed to Python
    "vertex_normals": APIMapping("vertex_normals", is_property=True),
    "face_colors": APIMapping("face_colors", is_property=True),
    # Basic geometry properties (exposed to Python)
    "face_normals": APIMapping("face_normals", is_property=True, rmesh_implemented=True),
    "edges": APIMapping("edges", is_property=True, rmesh_implemented=True),
    "bounds": APIMapping("bounds", is_property=True, rmesh_implemented=True),
    "extents": APIMapping("extents", is_property=True, rmesh_implemented=True),
    "area": APIMapping("area", is_property=True, rmesh_implemented=True),
    "area_faces": APIMapping("area_faces", is_property=True, rmesh_implemented=True),
    # Face adjacency (exposed to Python)
    "face_adjacency": APIMapping(
        "face_adjacency",
        is_property=True,
        converter=sort_adjacency_pairs,
        trimesh_converter=sort_adjacency_pairs,
        rmesh_implemented=True,
    ),
    "face_adjacency_angles": APIMapping(
        "face_adjacency_angles", is_property=False, rmesh_implemented=True
    ),
    "face_adjacency_unshared": APIMapping(
        "face_adjacency_unshared", is_property=True, rmesh_implemented=True
    ),
    "face_adjacency_projections": APIMapping(
        "face_adjacency_projections", is_property=True, rmesh_implemented=True
    ),
    "face_adjacency_convex": APIMapping(
        "face_adjacency_convex", is_property=True, rmesh_implemented=True
    ),
    # Mass properties (exposed to Python)
    "volume": APIMapping("volume", is_property=True, rmesh_implemented=True),
    "mass": APIMapping("mass", is_property=True, rmesh_implemented=True),
    "center_mass": APIMapping(
        "center_mass",
        is_property=True,
        converter=convert_center_mass,
        rmesh_implemented=True,
    ),
    "moment_inertia": APIMapping(
        "moment_inertia",
        is_property=True,
        converter=convert_inertia_matrix,
        rmesh_implemented=True,
    ),
    # Topology checks (exposed to Python)
    "is_watertight": APIMapping(
        "is_watertight", is_property=True, rmesh_implemented=True
    ),
    "is_winding_consistent": APIMapping(
        "is_winding_consistent", is_property=True, rmesh_implemented=True
    ),
    "is_volume": APIMapping("is_volume", is_property=True, rmesh_implemented=True),
    "is_convex": APIMapping("is_convex", is_property=True, rmesh_implemented=True),
    # Topology
    "euler_number": APIMapping("euler_number", is_property=True, rmesh_implemented=True),
    "centroid": APIMapping("centroid", is_property=True, rmesh_implemented=True),
    # Triangles
    "triangles": APIMapping("triangles", is_property=True, rmesh_implemented=True),
    "triangles_center": APIMapping(
        "triangles_center", is_property=True, rmesh_implemented=True
    ),
    # Principal inertia (eigenvalues/eigenvectors of inertia tensor)
    "principal_inertia_components": APIMapping(
        "principal_inertia_components", is_property=True, rmesh_implemented=True
    ),
    "principal_inertia_vectors": APIMapping(
        "principal_inertia_vectors", is_property=True, rmesh_implemented=True
    ),
    # Edge properties
    "edges_sorted": APIMapping("edges_sorted", is_property=True, rmesh_implemented=True),
    "edges_unique_length": APIMapping(
        "edges_unique_length", is_property=True, rmesh_implemented=True
    ),
    "edges_unique_inverse": APIMapping(
        "edges_unique_inverse", is_property=True, rmesh_implemented=True
    ),
}

# Sets for quick lookup
RMESH_PYTHON_EXPOSED = {
    "vertices",
    "faces",
    "uv",
    "vertex_normals",
    "face_colors",
    "simplify",
    "from_arrays",
    # Geometry properties
    "face_normals",
    "edges",
    "edges_unique",
    "edges_sorted",
    "edges_unique_length",
    "edges_unique_inverse",
    "bounds",
    "extents",
    "area",
    "area_faces",
    "faces_cross",
    "centroid",
    "triangles",
    "triangles_center",
    # Mass properties
    "volume",
    "mass",
    "center_mass",
    "moment_inertia",
    "principal_inertia_components",
    "principal_inertia_vectors",
    # Topology
    "is_watertight",
    "is_winding_consistent",
    "is_volume",
    "is_convex",
    "euler_number",
    # Face adjacency
    "face_adjacency",
    "face_adjacency_angles",
    "face_adjacency_unshared",
    "face_adjacency_projections",
    "face_adjacency_convex",
}

RMESH_NOT_IMPLEMENTED = {
    "face_adjacency_edges",
    "split",
    "merge_vertices",
}


def get_mapping(trimesh_name: str) -> APIMapping | None:
    """Get the API mapping for a trimesh property/method name."""
    return API_MAPPING.get(trimesh_name)


def is_comparable(trimesh_name: str) -> bool:
    """Check if a trimesh property/method has an rmesh equivalent we can compare."""
    return trimesh_name in API_MAPPING
