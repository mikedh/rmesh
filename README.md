# rmesh

`rmesh` is an experimental reimplementation of [`trimesh`](https://trimesh.org) in Rust using [`PyO3`](https://pyo3.rs). The general idea is to be mostly API compatible and pass trimesh's [large test suite](https://app.codecov.io/gh/mikedh/trimesh) with the exception of behavior changes made for quality reasons. 

## Background

Trimesh was originally a [research codebase](https://github.com/mikedh/ifab_archive) that grew organically to solve problems [I care about](https://carvewizard.com). It's probably popular because it tries quite hard to avoid being annoying: `pip install` always works, it caches expensive values automatically on the mesh object, and it uses [occasionally convoluted numpy indexing tricks](https://github.com/mikedh/trimesh/blob/4c83215f3ad749c4d4596598dbb6bcc26c0647cf/trimesh/exchange/obj.py#L137-L151) to avoid Evil Python Behavior such as loops of any sort. When I've benchmarked it, it often performs order-of-magnitude similar to carefully written compiled mesh codebases for many common tasks. It also is opinionated and maybe a little too magical sometimes, given my Very Strong Opinions that most applications that consume meshes should be like 10 lines.

It also predates type hints and has a [larger than ideal number of optional dependencies](https://trimesh.org/install.html#dependency-overview) which require some effort to maintain. There's also been a revolution in the Python ecosystem where some great [Python libraries](https://github.com/astral-sh/ruff) are now written in Rust, thanks to the awesome work of [`PyO3`](https://pyo3.rs) and [`cibuildwheel`](https://github.com/pypa/cibuildwheel). Rmesh is intended to package similarly to [`polars`](https://pola.rs/), where it is written in Rust but usable from both Python and Rust. Rust is also a nice low-level language with great tooling and is fast enough to implement very sensitive iterative algorithms.


## Goals
- Targeting use as a Rust crate, a nicely type hinted Python module, and WASM. WASM is mostly because `wasm-pack` made it kind of easy, and keeping the build in CI from the start makes sure we don't add things that break WASM builds.
- Be generally faster than trimesh and pass many-to-most of trimesh's unit tests.
- Have a relatively small number of carefully chosen dependencies, and prefer to vendor/re-write the rest in Rust. Generally try to keep it to major crates, like `nalgebra`, `anyhow`, `bytemuck`, although if there's a well-maintained implementation of something in pure Rust we should use it (i.e. [earcut](https://github.com/ciscorn/earcut-rs)).
- Build Python wheels for every platform using cibuildwheel.


## Implementation Notes

- Caching
  - `trimesh` uses hashes of numpy arrays and dirty flags on an `ndarray` subclass to save expensive values, like `face_normals` (very often the slowest thing in a profile).
  - `rmesh` objects are immutable and mutations produce new objects with a new cache. Cached values are saved to a `RwLock` cache which can be accessed through convience macros. TODO: does every value have to also be an `Arc`?
- Soft Dependencies
  - `trimesh` tries to package high-quality preferrably header-only upstream C codebases using `cibuildwheel` into their own Python package.
  - `rmesh` tries to use a small number of compile-time dependencies. Most mesh algorithms should be done inside the codebase to avoid chasing mysterious upstream dependencies if at all possible. The Python install should have `numpy` as the only dependancy (no scipy as all the heavy lifting will be done in Rust). 
- `ndarray` vs `nalgebra`: should the basic data (e.g. `(n, 3)` vertices) be `ndarray::Array2<f64>` or `Vec<nalgebra::Point3<f64>>`
  - This started with nalgebra, converted to ndarray on a branch, and then reverted back to nalgebra. [Forum posts indicate ndarray doesn't win on performance](https://users.rust-lang.org/t/is-it-possible-to-improve-ndarray-performance-vs-nalgebra/94114) like I kind of expected it to. The nalgebra objects have some nice properties: specifically you can constrain the number of columns (i.e `Vector3` vs `Vector4`) which I couldn't make work in ndarray although it's probably possible. I also generally preferred the nalgebra API but this is certainly a matter of taste.
