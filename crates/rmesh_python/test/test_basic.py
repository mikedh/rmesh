import os

import numpy as np
import rmesh

# current directory this test is in
_cwd = os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))
# project root directory
_root = os.path.abspath(os.path.join(_cwd, "..", "..", ".."))
# where test models are stored
_models = os.path.abspath(os.path.join(_root, "test", "data"))


def test_load_stl():
    with open(os.path.join(_models, "unit_cube.STL"), "rb") as f:
        scene = rmesh.load(f.read(), "stl")

    # load() now returns a Scene
    assert isinstance(scene, rmesh.Scene)
    assert len(scene) == 1
    assert len(scene.geometry) == 1

    # geometry is now dict-like, access by name or iteration
    assert list(scene.geometry.keys())  # has keys

    # get first mesh by iterating
    first_name = list(scene.geometry.keys())[0]
    m = scene.geometry[first_name]

    # check the dtypes of our rust output
    assert m.vertices.dtype == np.float64
    assert m.faces.dtype == np.int64

    assert not m.vertices.flags.writeable
    assert not m.faces.flags.writeable

    # make sure this unit cube produces the correct shape of triangle soup
    assert m.vertices[m.faces].shape == (12, 3, 3)


def test_geometry_dict_api():
    """Test the dict-like geometry access API."""
    scene = rmesh.load(os.path.join(_models, "cube.glb"))

    # Test dict-like methods
    assert len(scene.geometry) >= 1
    keys = scene.geometry.keys()
    assert len(keys) >= 1

    # Test __contains__
    first_key = keys[0]
    assert first_key in scene.geometry
    assert "nonexistent" not in scene.geometry

    # Test __getitem__
    mesh = scene.geometry[first_key]
    assert mesh is not None
    assert hasattr(mesh, 'vertices')
    assert hasattr(mesh, 'faces')

    # Test iteration
    count = 0
    for name in scene.geometry:
        assert isinstance(name, str)
        count += 1
    assert count == len(scene.geometry)

    # Test values()
    values = scene.geometry.values()
    assert len(values) == len(scene.geometry)

    # Test items()
    items = scene.geometry.items()
    assert len(items) == len(scene.geometry)
    for name, mesh in items:
        assert isinstance(name, str)
        assert hasattr(mesh, 'vertices')

    # Test get() with default
    assert scene.geometry.get("nonexistent") is None
    assert scene.geometry.get(first_key) is not None


if __name__ == "__main__":
    test_load_stl()
