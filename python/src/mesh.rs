use pyo3::prelude::*;
use anyhow::{anyhow, Result};
use crate::rmesh::mesh::{load_mesh, BinaryStl, MeshFormat, Trimesh};

#[pymethods]
impl Trimesh {
    #[new]
    pub fn py_new(vertices: &[u8], faces: &[u8]) -> Result<Self> {
        let vertices: &[f64] = bytemuck::cast_slice::<u8, f64>(vertices);
        let faces: &[usize] = bytemuck::cast_slice::<u8, usize>(faces);

        Self::from_slice(vertices, faces)
    }

    pub fn py_check(&self) -> usize {
        10
    }
}

/// Load a mesh from a file, doing no initial processing.
#[pyfunction(name = "load_mesh")]
pub fn py_load_mesh(file_data: &[u8], file_type: String) -> Result<Trimesh> {
    load_mesh(file_data, MeshFormat::from_string(&file_type)?)
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::relative_eq;

    #[test]
    fn test_mesh_normals() {
        let m = Trimesh::from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[0, 1, 2])
            .unwrap();
        let normals = m.face_normals();
        assert_eq!(normals.len(), 1);
        assert!(relative_eq!(
            normals[0],
            Vector3::new(0.0, 0.0, 1.0),
            epsilon = 1e-6
        ));
    }
}
