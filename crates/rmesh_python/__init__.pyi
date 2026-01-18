from numpy import float64, int64
from numpy.typing import NDArray

class Trimesh:
    """A triangle mesh."""

    def __init__(self, vertices: NDArray[float64], faces: NDArray[int64]):
        """Create a new Trimesh from vertices and faces."""
        ...

def load_mesh(file_data: bytes, file_type: str) -> Trimesh:
    """Load a mesh from a file, doing no initial processing."""
    ...
