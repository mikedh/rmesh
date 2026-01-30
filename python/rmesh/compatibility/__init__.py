"""Public API for rmesh/trimesh compatibility."""

from .wrapper import WrappedMesh, WrappedScene, from_trimesh, load_scene

__all__ = [
    "WrappedMesh",
    "WrappedScene",
    "from_trimesh",
    "load_scene",
]
