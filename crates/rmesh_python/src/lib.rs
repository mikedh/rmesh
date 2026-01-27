mod mesh;

pub use mesh::{
    PyFaceAttributes, PyGrouping, PyGroupingCollection, PyTrimesh, PyVertexAttributes, py_load_mesh,
};

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn rmesh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_load_mesh, m)?)?;
    m.add_class::<PyTrimesh>()?;
    m.add_class::<PyVertexAttributes>()?;
    m.add_class::<PyFaceAttributes>()?;
    m.add_class::<PyGrouping>()?;
    m.add_class::<PyGroupingCollection>()?;
    Ok(())
}
