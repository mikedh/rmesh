# rmesh

An experimental reimplementation of `trimesh` in Rust using PyO3. The long-term goal is to be API-compatible and pass 100% of trimesh's large test suite, with the exception of behavior changes made for quality reasons. 

## Introduction

Trimesh was originally a [research codebase](ifab) and is probably popular because it tries to avoid being annoying: `pip install` always works, it caches expensive values automatically on the mesh object, and it uses [occasionally convoluted numpy indexing tricks]() to avoid Evil Python Behavior such as loops of any sort. It performs order-of-magnitude similar to compiled mesh codebases for many common tasks.

It also predates type hints and has a [larger than ideal number of optional dependencies]() which requires some effort to maintain. 

## Goals
- Targeting use as a clean (i.e. no PyO3 for pure Rust) rust crate, nicely hinted Python module, and WASM.
- Be faster than trimesh for every function call and pass many-to-most of trimesh's unit tests.
- Have a relatively small number of carefully chosen dependencies and vendor/re-write the rest in Rust. I.e. major crates only, like `nalgebra`, `anyhow`, `bytemuck`, etc. 
- Build Python wheels for every platform.



## Implementation Comparison

- Caching
  - `trimesh` uses hashes of numpy arrays and dirty flags on an `ndarray` subclass to save expensive values, like `face_normals` (very often the slowest thing in a profile).
  - `rmesh` objects are immutable, and mutations produce new objects with a new cache. Cached values are saved to a `RwLock` cache which can be accessed through convience macros. TODO: does every value have to also be an `Arc`?
- Soft Dependencies
  - `trimesh` tries to package high-quality preferrably header-only upstream C codebases using `cibuildwheel` into their own Python package.
  - `rmesh` tries to use a small number of compile-time dependencies. Most mesh algorithms should be done inside the codebase to avoid chasing mysterious upstream dependencies if at all possible.
- 

