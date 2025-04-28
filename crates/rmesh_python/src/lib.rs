mod mesh;

pub use mesh::{PyTrimesh, py_load_mesh};

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn rmesh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_load_mesh, m)?)?;
    m.add_class::<PyTrimesh>()?;
    Ok(())
}
