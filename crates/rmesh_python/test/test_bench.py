import os

import rmesh
import trimesh
from io import BytesIO
from timeit import timeit

# where test models are stored
_models = os.path.abspath(os.path.expanduser("~/trimesh/models"))


def compare(file_data: bytes, file_type: str):
    try:
        # make sure we can load in both libraries before benchmarking
        _r = rmesh.load_mesh(file_data=file_data, file_type=file_type)
        _t = trimesh.load_mesh(file_obj=BytesIO(file_data), file_type=file_type)
    except BaseException as E:
        print(E)
        return None

    count = 1
    tri = timeit(
        stmt="trimesh.load_mesh(file_obj=BytesIO(file_data), file_type=file_type)",
        setup="import trimesh; from io import BytesIO",
        number=count,
        globals={"file_data": file_data, "file_type": file_type},
    )
    rme = timeit(
        stmt="rmesh.load_mesh(file_data=file_data, file_type=file_type)",
        setup="import rmesh",
        number=count,
        globals={"file_data": file_data, "file_type": file_type},
    )

    return {"trimesh": tri / count, "rmesh": rme / count, "ratio": tri / rme}


def test_both():
    results = {}
    for file_name in os.listdir(_models):
        with open(os.path.join(_models, file_name), "rb") as f:
            file_data = f.read()
        file_type = trimesh.util.split_extension(file_name)

        results[file_name] = compare(file_data=file_data, file_type=file_type)
        print(file_name, results[file_name])


if __name__ == "__main__":
    test_both()
