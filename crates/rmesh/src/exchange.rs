use anyhow::{Result, anyhow};
use itertools::Itertools;
use nalgebra::{Point3, Vector3, convert};
use rayon::prelude::*;

use crate::mesh::Trimesh;

pub struct BinaryStl {
    header: String,
    triangles: Vec<BinaryStlTriangle>,
}
#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BinaryStlTriangle {
    pub normal: Vector3<f32>,
    pub vertices: [Point3<f32>; 3],
    pub attributes: u16,
}

impl BinaryStlTriangle {
    pub fn convert_normal(&self) -> Vector3<f64> {
        convert(self.normal)
    }

    pub fn convert_vertices(&self) -> [Point3<f64>; 3] {
        [
            convert(self.vertices[0]),
            convert(self.vertices[1]),
            convert(self.vertices[2]),
        ]
    }
}

impl BinaryStl {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 84 {
            return Err(anyhow::anyhow!("STL file too short"));
        }

        let header = String::from_utf8_lossy(&bytes[0..80]).to_string();
        // let triangle_count = u32::from_le_bytes(bytes[80..84].try_into().unwrap());

        let triangles: &[BinaryStlTriangle] = bytemuck::try_cast_slice(&bytes[84..])
            .map_err(|_e| anyhow!("Could not interpret bytes as STL triangles!"))?;

        Ok(Self {
            header,
            triangles: triangles.to_vec(),
        })
    }

    pub fn to_mesh(&self) -> Trimesh {
        // convert STL f32 vertices to f64
        let vertices: Vec<Point3<f64>> = self
            .triangles
            .par_iter()
            .flat_map(|triangle| triangle.convert_vertices())
            .collect();

        let faces: Vec<(usize, usize, usize)> = (0..vertices.len()).tuples().collect();

        Trimesh::new(vertices, faces)
    }
}

enum ObjLine {
    V(Point3<f64>),
    Vn(Vector3<f64>),
    Vt(Vector3<f64>),
    F(Vec<(usize, usize, usize)>),
    O(String),
    G(String),
}

// An enum to represent the different mesh file formats.
pub enum MeshFormat {
    STL,
    OBJ,
    PLY,
}

impl MeshFormat {
    /// Convert a string to a MeshFormat enum.
    pub fn from_string(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().trim() {
            "stl" => Ok(MeshFormat::STL),
            "obj" => Ok(MeshFormat::OBJ),
            "ply" => Ok(MeshFormat::PLY),
            _ => Err(anyhow::anyhow!("Unsupported file type: {}", s)),
        }
    }
}

pub fn load_mesh(file_data: &[u8], file_type: MeshFormat) -> Result<Trimesh> {
    match file_type {
        MeshFormat::STL => Ok(BinaryStl::from_bytes(file_data)?.to_mesh()),
        MeshFormat::OBJ => todo!(),
        MeshFormat::PLY => todo!(),
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_mesh_stl() {
        let stl_data = include_bytes!("../../../test/data/unit_cube.STL");

        let mesh = load_mesh(stl_data, MeshFormat::STL).unwrap();

        assert_eq!(mesh.vertices.len(), 36);
        assert_eq!(mesh.faces.len(), 12);
    }
}
