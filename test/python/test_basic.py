import os
import rmesh
import numpy as np

# get data path relative to this file
_cwd = os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))
_models = os.path.abspath(os.path.join(_cwd, "..", "data"))


def test_load_stl():
    with open(os.path.join(_models, "unit_cube.STL"), "rb") as f:
        m = rmesh.load_mesh(f.read(), "stl")

    # check the dtypes of our rust output
    assert m.vertices.dtype == np.float64
    assert m.faces.dtype == np.int64

    # todo : fix this
    # assert not m.vertices.flags.writeable
    # assert not m.faces.flags.writeable

    # make sure this unit cube produces the correct shape of triangle soup
    assert m.vertices[m.faces].shape == (12, 3, 3)


if __name__ == "__main__":
    test_load_stl()
