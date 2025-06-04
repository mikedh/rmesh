use anyhow::Result;
use nalgebra::Point3;
use numpy::ndarray::Array2;
use pyo3::prelude::*;

use numpy::{PyArray2, PyReadonlyArray2};

use rmesh::exchange::{MeshFormat, load_mesh};
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
            data: Trimesh::new(vertices, faces, None, None)?,
        })
    }

    #[getter]
    pub fn get_vertices<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        // todo : is this the best way to do these conversions from Vec<Point3<f64>> to ndarray?
        // todo : the output array should be read-only
        // todo : should we cache this numpy conversion?
        let vertices = &self.data.vertices;
        let shape = (vertices.len(), 3);

        let arr = Array2::from_shape_vec(
            shape,
            vertices
                .iter()
                .flat_map(|p| p.coords.iter().cloned().collect::<Vec<_>>())
                .collect(),
        )
        .unwrap();

        PyArray2::from_array(py, &arr).to_owned().into()
    }

    #[getter]
    pub fn get_faces<'py>(&self, py: Python<'py>) -> Py<PyArray2<i64>> {
        let faces = &self.data.faces;
        let shape = (faces.len(), 3);

        let arr = Array2::from_shape_vec(
            shape,
            faces
                .iter()
                .flat_map(|&(a, b, c)| vec![a as i64, b as i64, c as i64])
                .collect(),
        )
        .unwrap();

        PyArray2::from_array(py, &arr).to_owned().into()
    }

    #[getter]
    pub fn get_uv<'py>(&self, py: Python<'py>) -> Option<Py<PyArray2<f64>>> {
        self.data.uv().as_ref().map(|uvs| {
            let shape = (uvs.len(), 2);
            let arr =
                Array2::from_shape_vec(shape, uvs.iter().flat_map(|p| vec![p.x, p.y]).collect())
                    .unwrap();
            PyArray2::from_array(py, &arr).to_owned().into()
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
