"""Public rmesh/trimesh compatibility wrappers.

Provides ``from_trimesh()`` for data conversion, ``WrappedMesh`` and
``WrappedScene`` as drop-in replacements backed by rmesh, and
``load_scene()`` as a convenience loader.
"""

import numpy as np


def _cascadio_to_rmesh_surface(d):
    """Remap a cascadio surface dict to the format expected by rmesh.

    Cascadio uses ``"type"`` (lowercase) while rmesh expects ``"kind"``
    (capitalized).  All other keys are passed through since the Rust
    parser ignores unknown fields via ``serde(default)``.
    """
    out = {k: v for k, v in d.items() if k != "type"}
    out["kind"] = d["type"].capitalize()
    return out


def from_trimesh(mesh):
    """
    Convert a ``trimesh.Trimesh`` to an ``rmesh.Trimesh``.

    If the source mesh carries cascadio BREP metadata
    (``mesh.metadata['cascadio']['brep_faces']`` and
    ``mesh.face_attributes['brep_index']``), it is forwarded
    as ``face_surfaces`` so that ``project()`` can emit
    analytical circles and arcs.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Source mesh.

    Returns
    -------
    rmesh.Trimesh
    """
    import rmesh

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    face_surfaces = None
    cascadio_meta = getattr(mesh, "metadata", {}).get("cascadio", {})
    brep_faces = cascadio_meta.get("brep_faces")
    fa = getattr(mesh, "face_attributes", {})
    brep_index = fa.get("brep_index") if isinstance(fa, dict) else None
    if brep_faces is not None and brep_index is not None:
        # Filter out None entries and remap face indices
        valid = {old: new for new, old in enumerate(
            i for i, f in enumerate(brep_faces) if f is not None)}
        clean_faces = [_cascadio_to_rmesh_surface(f)
                       for f in brep_faces if f is not None]
        raw_index = np.asarray(brep_index, dtype=np.int64)
        # Remap indices; faces referencing a None surface get index -1
        remapped = np.array([valid.get(int(i), -1) for i in raw_index],
                            dtype=np.int64)
        if clean_faces and (remapped >= 0).any():
            # Map unmapped faces (-1) to max int64 so the Rust side
            # sees them as usize::MAX == UNSET after the i64→usize cast.
            remapped = np.where(
                remapped >= 0, remapped, np.iinfo(np.int64).max)
            face_surfaces = (clean_faces, remapped)

    return rmesh.Trimesh(vertices, faces, face_surfaces=face_surfaces)


class WrappedMesh:
    """
    Drop-in replacement for ``trimesh.Trimesh`` backed by ``rmesh.Trimesh``.

    Explicit properties handle type conversions between rmesh and trimesh
    return types.  Unrecognized attributes fall through to the underlying
    rmesh mesh via ``__getattr__``.

    Parameters
    ----------
    rmesh_mesh : rmesh.Trimesh
        The backing mesh.
    """

    __slots__ = ("_rmesh",)

    def __init__(self, rmesh_mesh):
        object.__setattr__(self, "_rmesh", rmesh_mesh)

    @classmethod
    def from_trimesh(cls, mesh):
        """
        Create from a ``trimesh.Trimesh`` via data copy.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Source mesh.

        Returns
        -------
        WrappedMesh
        """
        return cls(from_trimesh(mesh))

    # --- Properties requiring type conversion ---

    @property
    def vertex_attributes(self):
        """dict : Vertex attributes (uv, normals, colors)."""
        va = self._rmesh.vertex_attributes
        result = {}
        for key in ("uv", "normals", "colors"):
            val = getattr(va, key, None)
            if val is not None and len(val) > 0:
                result[key] = np.asarray(val)
        return result

    @property
    def face_attributes(self):
        """dict : Face attributes (uv, normals, colors)."""
        fa = self._rmesh.face_attributes
        result = {}
        for key in ("uv", "normals", "colors"):
            val = getattr(fa, key, None)
            if val is not None and len(val) > 0:
                result[key] = np.asarray(val)
        return result

    @property
    def convex_hull(self):
        """WrappedMesh : Convex hull of this mesh."""
        return WrappedMesh(self._rmesh.convex_hull)

    # --- Methods returning meshes (wrap result) ---

    def simplify(self, target_faces, aggressiveness=None):
        """
        Return a simplified mesh.

        Parameters
        ----------
        target_faces : int
            Target number of faces.
        aggressiveness : float, optional
            Simplification aggressiveness.

        Returns
        -------
        WrappedMesh
        """
        return WrappedMesh(self._rmesh.simplify(target_faces, aggressiveness))

    def cleanup(self, **kwargs):
        """
        Return a cleaned mesh.

        Parameters
        ----------
        **kwargs
            Passed to ``rmesh.Trimesh.cleanup()``.

        Returns
        -------
        WrappedMesh
        """
        return WrappedMesh(self._rmesh.cleanup(**kwargs))

    def decompose(self, max_hulls=64, resolution=400_000):
        """
        Return approximate convex decomposition.

        Parameters
        ----------
        max_hulls : int
            Maximum number of convex hulls.
        resolution : int
            Voxel resolution for decomposition.

        Returns
        -------
        list[WrappedMesh]
        """
        return [WrappedMesh(m) for m in self._rmesh.decompose(max_hulls, resolution)]

    # --- Fallback for direct pass-through properties ---

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        rm = object.__getattribute__(self, "_rmesh")
        try:
            return getattr(rm, name)
        except AttributeError:
            raise NotImplementedError(
                f"rmesh.Trimesh has no attribute '{name}'"
            ) from None

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"WrappedMesh attributes are read-only (tried to set '{name}')"
            )

    def __repr__(self):
        return repr(object.__getattribute__(self, "_rmesh"))

    def __dir__(self):
        return dir(object.__getattribute__(self, "_rmesh"))

    def __len__(self):
        return len(object.__getattribute__(self, "_rmesh"))


class WrappedScene:
    """
    Drop-in replacement for ``trimesh.Scene`` backed by ``rmesh.Scene``.

    Parameters
    ----------
    rmesh_scene : rmesh.Scene
        The backing scene from ``rmesh.load()``.
    """

    __slots__ = ("_geometry", "_scene")

    def __init__(self, rmesh_scene):
        object.__setattr__(self, "_scene", rmesh_scene)
        geo = rmesh_scene.geometry
        wrapped = {name: WrappedMesh(geo[name]) for name in geo.keys()}
        object.__setattr__(self, "_geometry", wrapped)

    @property
    def geometry(self):
        """dict[str, WrappedMesh] : Scene geometry by name."""
        return object.__getattribute__(self, "_geometry")

    def __len__(self):
        return len(object.__getattribute__(self, "_geometry"))

    def __repr__(self):
        return repr(object.__getattribute__(self, "_scene"))

    def __iter__(self):
        return iter(object.__getattribute__(self, "_geometry"))


def load_scene(path, **kwargs):
    """
    Load a mesh file and return a wrapped scene.

    Parameters
    ----------
    path : str or os.PathLike
        File path to load.
    **kwargs
        Additional keyword arguments (reserved for future use).

    Returns
    -------
    WrappedScene
    """
    import rmesh

    scene = rmesh.load(str(path))
    return WrappedScene(scene)
