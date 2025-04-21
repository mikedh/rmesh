use anyhow::Result;
use nalgebra::Point3;
use pyo3::prelude::*;

use numpy::PyReadonlyArray2;

use rmesh::exchange::{load_mesh, MeshFormat};
use rmesh::mesh::Trimesh;

//use crate::rmesh::mesh::{load_mesh, MeshFormat, Trimesh};

#[pyclass(name = "Trimesh")]
#[derive(Clone)]
pub struct PyTrimesh {
    data: Trimesh,
}

#[pymethods]
impl PyTrimesh {

    #[new]
    /// (pyfunc) Create a new Trimesh from vertices and faces.
    pub fn new<'py>(
        vertices: PyReadonlyArray2<'py, f64>,
        faces: PyReadonlyArray2<'py, i64>,
    ) -> Result<Self> {
        let vertices: Vec<Point3<f64>> = vertices
            .as_array()
            .rows()
            .into_iter()
            .map(|x| Point3::new(x[0], x[1], x[2]))
            .collect::<Vec<_>>();

        let faces: Vec<(usize, usize, usize)> = faces
            .as_array()
            .rows()
            .into_iter()
            .map(|x| (x[0] as usize, x[1] as usize, x[2] as usize))
            .collect::<Vec<_>>();

        Ok(PyTrimesh {
            data: Trimesh::new(vertices, faces)?,
        })
    }

    pub fn py_check(&self) -> usize {
        10
    }
}

/// (pyfunc) Load a mesh from a file, doing no initial processing.
#[pyfunction(name = "load_mesh")]
pub fn py_load_mesh(file_data: &[u8], file_type: String) -> Result<PyTrimesh> {
    let data = load_mesh(file_data, MeshFormat::from_string(&file_type)?)?;

    Ok(PyTrimesh { data })
}

#[cfg(test)]
mod tests {

    use super::*;

    use rmesh::creation::create_box;

    #[test]
    fn test_mesh_python() {
        let data = create_box(&[1.0, 1.0, 1.0]);

        let m = PyTrimesh { data };

        assert_eq!(m.py_check(), 10);
    }
}
