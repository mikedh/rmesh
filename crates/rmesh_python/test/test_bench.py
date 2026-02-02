import json
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
# alternative location in home directory
_models_home = os.path.expanduser("~/trimesh/models")

# Supported file extensions
SUPPORTED_EXTENSIONS = {"stl", "obj", "glb", "ply"}


def compare(file_data: bytes, file_type: str):
    try:
        # make sure we can load in both libraries before benchmarking
        _r = rmesh.load(file_data, file_type=file_type)
        if not _r.geometry:
            print("  Skip: no geometry in scene")
            return None
        _t = trimesh.load_mesh(file_obj=BytesIO(file_data), file_type=file_type)
    except BaseException as E:
        print(f"  Skip: {E}")
        return None

    count = 1
    tri = timeit(
        stmt="len(trimesh.load_mesh("
        "file_obj=BytesIO(file_data), file_type=file_type).vertices)",
        setup="import trimesh; from io import BytesIO",
        number=count,
        globals={"file_data": file_data, "file_type": file_type},
    )
    rme = timeit(
        stmt="len(rmesh.load(file_data, file_type=file_type).geometry[0].vertices)",
        setup="import rmesh",
        number=count,
        globals={"file_data": file_data, "file_type": file_type},
    )

    return {"trimesh": tri / count, "rmesh": rme / count, "ratio": tri / rme}


def to_markdown(results: dict) -> str:
    """Convert results to markdown table."""
    mark = [
        "| File Name | `trimesh` | `rmesh` | Speedup |",
        "|    --     |    --     |   --    |   --    |",
    ]
    mark.extend(
        f"| {k} | {r['trimesh']:0.4f}s | {r['rmesh']:0.4f}s | {r['ratio']:0.2f}x |"
        for k, r in sorted(results.items())
    )

    return "\n".join(mark)


def to_json(results: dict) -> str:
    """Convert results to JSON for CI integration."""
    return json.dumps(results, indent=2)


def test_both():
    # Try multiple locations for models
    models_dir = None
    for path in [_models, _models_home]:
        if os.path.exists(path):
            models_dir = path
            break

    if models_dir is None:
        print("models not in expected location, tried:")
        print(f"  - {_models}")
        print(f"  - {_models_home}")
        return

    print(f"Using models from: {models_dir}")
    print(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
    print()

    results = {}
    skipped = []

    for file_name in sorted(os.listdir(models_dir)):
        file_path = os.path.join(models_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        # Get extension and check if supported
        file_type = trimesh.util.split_extension(file_name).lower()
        if file_type not in SUPPORTED_EXTENSIONS:
            skipped.append((file_name, f"unsupported format: {file_type}"))
            continue

        print(f"Testing: {file_name}")
        with open(file_path, "rb") as f:
            file_data = f.read()

        if result := compare(file_data=file_data, file_type=file_type):
            results[file_name] = result
            print(
                f"  trimesh: {result['trimesh']:.4f}s, "
                f"rmesh: {result['rmesh']:.4f}s, "
                f"speedup: {result['ratio']:.2f}x"
            )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60 + "\n")

    print("## Markdown Table\n")
    print(to_markdown(results))

    print("\n## Summary\n")
    if results:
        avg_speedup = sum(r["ratio"] for r in results.values()) / len(results)
        print(f"Files tested: {len(results)}")
        print(f"Files skipped: {len(skipped)}")
        print(f"Average speedup: {avg_speedup:.2f}x")

    if skipped:
        print(f"\n## Skipped Files ({len(skipped)})\n")
        for name, reason in skipped[:10]:  # Show first 10
            print(f"  - {name}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    # Write JSON results for CI
    json_path = os.path.join(_cwd, "bench_results.json")
    with open(json_path, "w") as f:
        f.write(to_json(results))
    print(f"\nJSON results written to: {json_path}")


if __name__ == "__main__":
    test_both()
