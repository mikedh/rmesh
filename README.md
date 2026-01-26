# rmesh
[![codecov](https://codecov.io/github/mikedh/rmesh/graph/badge.svg?token=Z33O31BODL)](https://codecov.io/github/mikedh/rmesh)

`rmesh` is an experimental reimplementation of [`trimesh`](https://trimesh.org) in Rust using [`PyO3`](https://pyo3.rs). It is not currently released and may never be. The general idea of `rmesh` is to be mostly API compatible and pass trimesh's [large test suite](https://app.codecov.io/gh/mikedh/trimesh) with the exception of behavior changes made for quality reasons. 

## Background

Trimesh was originally a [research codebase](https://github.com/mikedh/ifab_archive) that grew organically to solve problems [I care about](https://carvewizard.com). `trimesh` is probably [popular](https://pypistats.org/packages/trimesh) because it tries quite hard to avoid being annoying: `pip install` always works, expensive values are cached automatically on the mesh object, and it uses [occasionally convoluted indexing tricks](https://github.com/mikedh/trimesh/blob/4c83215f3ad749c4d4596598dbb6bcc26c0647cf/trimesh/exchange/obj.py#L137-L151) to avoid Evil Python Behavior such as loops. `trimesh` usually performs comparably to carefully written compiled mesh codebases for many common tasks, which is achieved with the careful use of numpy. It also is opinionated and maybe a little too magical sometimes, given my Very Strong Opinions that most applications that consume meshes should be like 10 lines.

`trimesh` also predates type hints and has a [larger than ideal number of optional dependencies](https://trimesh.org/install.html#dependency-overview), an upstream burden which requires [effort to maintain](https://github.com/trimesh). There's also been a revolution in the Python ecosystem where some great [Python libraries](https://github.com/astral-sh/ruff) are now written in Rust, thanks to the awesome work of [`PyO3`](https://pyo3.rs) and [`cibuildwheel`](https://github.com/pypa/cibuildwheel). Rmesh is intended to package similarly to [`polars`](https://pola.rs/), where it is written in Rust but usable from both Python and Rust. Rust is also a nice low-level language with great tooling and is fast enough to implement very sensitive iterative algorithms.

## Project Status

This is experimental, and doesn't do anything at the moment. It hasn't been released to package indexes and may never be if it turns out to be a dead end. `trimesh` isn't going anywhere as it has a ton of surface area. This won't be released to indexes until the following MVP feature list exists:

- Load an STL, OBJ, and PLY file from Python 3.0x faster than `trimesh` in Python.
- Implement the following for a mesh object: `edges`, `euler_number`, `merge_vertices`, `face_normals`, `face_adjacency`, `face_adjacency_angles`, `extents`, `bounds`, `split`, `is_watertight`, `is_winding_consistant`, `is_volume`, `is_convex`, mass properties using the [same](https://github.com/mikedh/trimesh/blob/76b2bd31d32a1231320f8151d94f99e77ac8dc5b/trimesh/triangles.py#L214-L329) [algorithm](http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf), `principal_inertia_components` (using nalgebra `hermitian_eigen`),

With Claude, this is more likely to actually happen. 


## Goals
- Targeting use as a Rust crate, a nicely type hinted Python module, and WASM. WASM is mostly because `wasm-pack` made it kind of easy, and keeping the build in CI from the start makes sure we don't add things that break WASM builds.
- Be generally faster than trimesh and pass many-to-most of trimesh's unit tests.
- Have a relatively small number of carefully chosen dependencies, and prefer to vendor/re-write the rest in Rust. Generally try to keep it to major crates, like `nalgebra`, `anyhow`, `bytemuck`, although if there's a well-maintained implementation of something in pure Rust we should use it (i.e. [earcut](https://github.com/ciscorn/earcut-rs)).
- Build Python wheels for every platform using cibuildwheel.


## Implementation Notes

- Caching
  - `trimesh` uses hashes of numpy arrays and dirty flags on an `ndarray` subclass to save expensive values, like `face_normals` (very often the slowest thing in a profile).
  - `rmesh` objects are immutable and mutations produce new objects with a new cache. Cached values are saved to a `RwLock` cache which can be accessed through convenience macros. TODO: does every value have to also be an `Arc`?
- Soft Dependencies
  - `trimesh` tries to package high-quality preferably header-only upstream C codebases using `cibuildwheel` into their own Python package.
  - `rmesh` tries to use a small number of compile-time dependencies. Most mesh algorithms should be done inside the codebase to avoid chasing mysterious upstream dependencies if at all possible. The Python install should have `numpy` as the only Python dependency, as all  heavy lifting will be implemented in Rust. 
- Basic Data Types
  - `trimesh` uses a `np.ndarray` object for vertices (`float64`) and faces (`int64`)
  - `rmesh` has the choice between `ndarray` vs `nalgebra`:  `ndarray::Array2<f64>` or `Vec<nalgebra::Point3<f64>>`
    - `rmesh` started with nalgebra, converted to `ndarray` on a branch, and then reverted back to nalgebra. [Forum posts indicate ndarray doesn't win on performance](https://users.rust-lang.org/t/is-it-possible-to-improve-ndarray-performance-vs-nalgebra/94114) like I kind of expected it to. The nalgebra objects have some nice properties: specifically you can constrain the number of columns (i.e `Vector3` vs `Vector4`) which I couldn't make work in ndarray although it's probably possible. I also generally preferred the nalgebra API but this is certainly a matter of taste.
- Vertex Attributes
  - `trimesh` kind of squirrels these away in multiple places: `mesh.visual.TextureVisuals.uv` which puts them in `mesh.visual.vertex_attributes` but not `mesh.vertex_attributes`. Which is a little weird. And what if you had multiple sets of UV's and colors?
  - `rmesh` intends to be more attribute-forward. Faces and vertices each get a flat `Vec` of `Attribute`, similar to the GLTF format. For instance if a function wanted vertex color they'd go through the vec and then take the first (or n-th) `Color` attribute. We could have a lookup helper if we really wanted to but most meshes as loaded rarely have more than ~3 attributes (with a median of 0).


### Project Layout: Crates

`rmesh` is set up as a Cargo workspace which is a [common](https://github.com/gfx-rs/wgpu) choice for a complex project. The workspace crates (in `./crates`) are:
  - `rmesh`
    - the basic crate where algorithms are implemented.
  - `rmesh_macro`
    - all `proc macros` must be their own crate for Reasons. This as of writing only contains the `cache_access` proc macro which handles some of the boilerplate for dealing with the `RwLock` cache.
  - `rmesh_python`
    - The crate that builds to `pip install rmesh`, and includes a dependency on `PyO3` and other Python plumbing. This should be 100% boilerplate for accessing `rmesh`.

### Setup for Mixed Rust-Python Development

After cloning, run once:
```bash
uv sync
```

Then `cargo test` and `uv run pytest` work from anywhere in the workspace. The `uv sync` must happen first because `pyo3-build-config` (a dependency) builds before our code, and it needs `.venv/bin/python` to exist when it runs. If you are working entirely in the `crates/rmesh` in a rust-only project you shouldn't need UV.

### Running Tests

Run the trimesh test suite with rmesh compatibility comparison:
```bash
# Basic test run (uses trimesh base install)
uv run --extra test pytest

# Full test run with all trimesh optional dependencies
uv run --extra test-compat pytest
```

The pytest configuration automatically enables `--rmesh-compat=compare` mode, which:
- Monkey-patches `trimesh.Trimesh` to run both trimesh and rmesh for each property access
- Compares results and logs timing differences
- Generates `comparison.md` after each test run

To regenerate `comparison.md`, simply run pytest:
```bash
uv run --extra test-compat pytest test/trimesh/tests/
```

### Compatibility Module

The `rmesh._compatibility` module provides two modes for integrating rmesh with trimesh:

**Compare mode** (for testing): Run both implementations, log timing/equality, return trimesh values.
```python
import rmesh._compatibility
rmesh._compatibility.enable("compare")
# Now trimesh.Trimesh properties run both and compare results
```

**Wrap mode** (for users): Return rmesh values when available, fallback to trimesh.
```python
import rmesh._compatibility
rmesh._compatibility.enable("wrap")
import trimesh
mesh = trimesh.load("model.stl")
print(mesh.volume)  # Uses faster rmesh implementation
```

### Concious API Differences

- `mesh.process` -> `mesh.cleanup(tol_merge:int ...)
  - remove_degenerate, remove_infinite, etc all go into cleanup
- `mesh.apply_transform` returns a transformed copy, no in-place
- `mesh.bounds`/ `mesh.extents` on empty meshes return None, not zeros
- `mesh.density` -> `mesh.MassSettings` (override density + COM)
- lists of structured stuff are on the mesh and flat:
  - `mesh.materials`
  - `mesh.patch_primitives`
- OBJ groups/smoothing/objects are face attributes not new meshes
  - can be used in like, `mesh.split_on`

