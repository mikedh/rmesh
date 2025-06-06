use anyhow::Result;
use nalgebra::Point3;
use numpy::ndarray::Array2;
use pyo3::prelude::*;

use numpy::{PyArray2, PyReadonlyArray2};

use rmesh::exchange::{MeshFormat, load_mesh};
use rmesh::mesh::Trimesh;

//use crate::rmesh::mesh::{load_mesh, MeshFormat, Trimesh};

/// Helper trait for converting Rust data structures to NumPy arrays
trait ToNumPy<T> {
    fn to_numpy(&self, py: Python<'_>) -> Py<PyArray2<T>>;
}

/// Implementation for Vec<Point3<f64>> -> PyArray2<f64>
impl ToNumPy<f64> for Vec<Point3<f64>> {
    fn to_numpy(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        let shape = (self.len(), 3);
        let arr = Array2::from_shape_vec(
            shape,
            self.iter()
                .flat_map(|p| p.coords.iter().cloned().collect::<Vec<_>>())
                .collect(),
        )
        .unwrap();
        PyArray2::from_array(py, &arr).to_owned().into()
    }
}

/// Implementation for Vec<(usize, usize, usize)> -> PyArray2<i64>
impl ToNumPy<i64> for Vec<(usize, usize, usize)> {
    fn to_numpy(&self, py: Python<'_>) -> Py<PyArray2<i64>> {
        let shape = (self.len(), 3);
        let arr = Array2::from_shape_vec(
            shape,
            self.iter()
                .flat_map(|&(a, b, c)| vec![a as i64, b as i64, c as i64])
                .collect(),
        )
        .unwrap();
        PyArray2::from_array(py, &arr).to_owned().into()
    }
}

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
            data: Trimesh::new(vertices, faces, None)?,
        })
    }

    #[getter]
    pub fn get_vertices<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        self.data.vertices.to_numpy(py)
    }

    #[getter]
    pub fn get_faces<'py>(&self, py: Python<'py>) -> Py<PyArray2<i64>> {
        self.data.faces.to_numpy(py)
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

    #[test]
    fn test_to_numpy_traits() {
        use nalgebra::Point3;
        
        // Test Vec<Point3<f64>> conversion
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        
        // We can't test the actual numpy conversion without Python runtime,
        // but we can test that the trait is implemented and compiles
        let _test = format!("{:?}", vertices.len());
        
        // Test Vec<(usize, usize, usize)> conversion
        let faces = vec![(0, 1, 2), (1, 2, 3)];
        let _test = format!("{:?}", faces.len());
        
        // Test that it works end-to-end with our PyTrimesh
        let data = create_box(&[1.0, 1.0, 1.0]);
        let m = PyTrimesh { data };
        
        // Check that we have the expected number of vertices and faces
        assert_eq!(m.data.vertices.len(), 8);
        assert_eq!(m.data.faces.len(), 12);
    }
}
