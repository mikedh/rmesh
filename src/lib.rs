mod mesh;
mod simplify;
use pyo3::prelude::*;

use mesh::Trimesh;

/// A Python module implemented in Rust.
#[pymodule]
fn rmesh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Trimesh>()?;

    Ok(())
}
