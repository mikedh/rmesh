import os
from io import BytesIO
from timeit import timeit

import rmesh
import trimesh

# current working
_cwd = os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))
# root of checkout
_root = os.path.abspath(os.path.join(_cwd, "..", "..", ".."))
# if trimesh was cloned next to rmesh
_models = os.path.abspath(os.path.join(_root, "..", "trimesh", "models"))


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
        stmt="len(trimesh.load_mesh(file_obj=BytesIO(file_data), file_type=file_type).vertices)",
        setup="import trimesh; from io import BytesIO",
        number=count,
        globals={"file_data": file_data, "file_type": file_type},
    )
    rme = timeit(
        stmt="len(rmesh.load_mesh(file_data=file_data, file_type=file_type).vertices)",
        setup="import rmesh",
        number=count,
        globals={"file_data": file_data, "file_type": file_type},
    )

    return {"trimesh": tri / count, "rmesh": rme / count, "ratio": tri / rme}


def to_markdown(results: dict) -> str:
    """ """
    mark = [
        "| File Name | `trimesh` | `rmesh` | Speedup |",
        "|    --     |    --     |   --    |   --    |",
    ]
    mark.extend(
        f"| {k} | {r['trimesh']:0.4f}s | {r['rmesh']:0.4f}s | {r['ratio']:0.2f}x |"
        for k, r in results.items()
    )

    return "\n".join(mark)


def test_both():
    # if we didn't check out trimesh for the models corpus
    if not os.path.exists(_models):
        print(f"models not in expected location (`{_models}`), exiting!")
        return

    results = {}
    for file_name in os.listdir(_models):
        file_path = os.path.join(_models, file_name)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "rb") as f:
            file_data = f.read()
        file_type = trimesh.util.split_extension(file_name)

        if result := compare(file_data=file_data, file_type=file_type):
            results[file_name] = result
            print(file_name, result)

    print(to_markdown(results))


if __name__ == "__main__":
    test_both()
