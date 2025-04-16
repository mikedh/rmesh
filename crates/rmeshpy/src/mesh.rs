use anyhow::Result;
use pyo3::prelude::*;

use rmesh::{load_mesh, MeshFormat, Trimesh};

//use crate::rmesh::mesh::{load_mesh, MeshFormat, Trimesh};

#[pyclass(name = "Trimesh")]
#[derive(Clone)]
pub struct PyTrimesh {
    data: Trimesh,
}

#[pymethods]
impl PyTrimesh {
    #[new]
    pub fn new(vertices: &[u8], faces: &[u8]) -> Result<Self> {
        let vertices: &[f64] = bytemuck::cast_slice::<u8, f64>(vertices);
        let faces: &[usize] = bytemuck::cast_slice::<u8, usize>(faces);
        let data = Trimesh::from_slice(vertices, faces)?;
        Ok(PyTrimesh { data })
    }

    pub fn py_check(&self) -> usize {
        10
    }
}

/// Load a mesh from a file, doing no initial processing.
#[pyfunction(name = "load_mesh")]
pub fn py_load_mesh(file_data: &[u8], file_type: String) -> Result<PyTrimesh> {
    let data = load_mesh(file_data, MeshFormat::from_string(&file_type)?)?;

    Ok(PyTrimesh { data })
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_mesh_python() {
        let m = PyTrimesh::new(
            &bytemuck::cast_slice::<f64, u8>(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            &bytemuck::cast_slice::<usize, u8>(&[0, 1, 2]),
        )
        .unwrap();

        assert_eq!(m.data.vertices.len(), 3);
        assert_eq!(m.data.faces.len(), 1);
        assert_eq!(m.py_check(), 10);
    }
}
