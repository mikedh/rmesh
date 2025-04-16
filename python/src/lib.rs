use pyo3::prelude::*;
mod mesh;
use mesh::{py_load_mesh, PyTrimesh};

/// A Python module implemented in Rust.
#[pymodule]
fn rmesh_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_load_mesh, m)?)?;
    m.add_class::<PyTrimesh>()?;
    Ok(())
}
