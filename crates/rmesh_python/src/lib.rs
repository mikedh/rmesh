mod feature;
mod mesh;
mod scene;
mod viewer;

pub use mesh::{
    PyFaceAttributes, PyGrouping, PyGroupingCollection, PyPath2D, PyPath3D, PyPolygon2D, PyTrimesh,
    PyVertexAttributes, PyVoxelGrid, py_load,
};
pub use scene::{PyGeometryDict, PyScene, PySceneGraph, PySceneNode};

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
    m.add_class::<PyPolygon2D>()?;
    m.add_class::<PyPath2D>()?;
    m.add_class::<PyPath3D>()?;
    m.add_class::<PySceneGraph>()?;
    m.add_class::<PySceneNode>()?;

    // Starlark expression evaluation (top-level convenience function)
    m.add_function(wrap_pyfunction!(feature::py_evaluate, m)?)?;

    // Feature submodule
    feature::register_feature_module(m)?;

    // Viewer functions
    viewer::register_viewer(m)?;

    Ok(())
}
