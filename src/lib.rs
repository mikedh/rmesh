mod mesh;
mod simplify;
use pyo3::prelude::*;

use mesh::{Trimesh, py_load_mesh};

/// A Python module implemented in Rust.
#[pymodule]
fn rmesh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_load_mesh, m)?)?;
    m.add_class::<Trimesh>()?;
    Ok(())
}
