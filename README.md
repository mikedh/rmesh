# rmesh

`rmesh` is an experimental reimplementation of [`trimesh`](https://trimesh.org) in Rust using Pyo3. The general idea is to be mostly API-compatible and pass most of trimesh's large test suite, with the exception of behavior changes made for quality reasons. 

## Background

Trimesh was originally a [research codebase](ifab) and is probably popular because it tries to avoid being annoying: `pip install` always works, it caches expensive values automatically on the mesh object, and it uses [occasionally convoluted numpy indexing tricks]() to avoid Evil Python Behavior such as loops of any sort. It performs order-of-magnitude similar to compiled mesh codebases for many common tasks.

It also predates type hints and has a [larger than ideal number of optional dependencies]() which requires some effort to maintain. 

## Goals
- Targeting use as a clean Rust crate  (i.e. doesn't depend on pyo3), a nicely type hinted Python module, and a WASM build.
- Be faster than trimesh for every function call and pass many-to-most of trimesh's unit tests.
- Have a relatively small number of carefully chosen dependencies.  and vendor/re-write the rest in Rust. I.e. major crates only, like `nalgebra`, `anyhow`, `bytemuck`, etc. 
- Build Python wheels for every platform using cibuildwheel.



## Implementation Notes

- Caching
  - `trimesh` uses hashes of numpy arrays and dirty flags on an `ndarray` subclass to save expensive values, like `face_normals` (very often the slowest thing in a profile).
  - `rmesh` objects are immutable, and mutations produce new objects with a new cache. Cached values are saved to a `RwLock` cache which can be accessed through convience macros. TODO: does every value have to also be an `Arc`?
- Soft Dependencies
  - `trimesh` tries to package high-quality preferrably header-only upstream C codebases using `cibuildwheel` into their own Python package.
  - `rmesh` tries to use a small number of compile-time dependencies. Most mesh algorithms should be done inside the codebase to avoid chasing mysterious upstream dependencies if at all possible. The Python install should have `numpy` as the only dependancy (no scipy as all the heavy lifting will be done in Rust). 
- `ndarray` vs `nalgebra`: should the basic data (e.g. `(n, 3)` vertices) be a `ndarray::Array2<f64>` or a `Vec<nalgebra::Point3<f64>>`
  - This started with nalgebra, converted to ndarray on a branch, and then reverted back to nalgebra. [Forum posts indicate ndarray doesn't win on performance](https://users.rust-lang.org/t/is-it-possible-to-improve-ndarray-performance-vs-nalgebra/94114) like I kind of expected it to. The nalgebra objects have some nice properties: specifically you can constrain the number of columns (i.e `Vector3` vs `Vector4`) which I couldn't make work in ndarray although it's probably possible. I also generally preferred the nalgebra API but this is certainly a matter of taste.
