mod feature;
mod mesh;

pub use mesh::{
    PyFaceAttributes, PyGrouping, PyGroupingCollection, PyScene, PyTrimesh, PyVertexAttributes,
    PyVoxelGrid, py_load,
};

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn rmesh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Mesh functionality
    m.add_function(wrap_pyfunction!(py_load, m)?)?;
    m.add_class::<PyScene>()?;
    m.add_class::<PyTrimesh>()?;
    m.add_class::<PyVertexAttributes>()?;
    m.add_class::<PyFaceAttributes>()?;
    m.add_class::<PyGrouping>()?;
    m.add_class::<PyGroupingCollection>()?;
    m.add_class::<PyVoxelGrid>()?;

    // Starlark expression evaluation (top-level convenience function)
    m.add_function(wrap_pyfunction!(feature::py_evaluate, m)?)?;

    // Feature submodule
    feature::register_feature_module(m)?;

    Ok(())
}
