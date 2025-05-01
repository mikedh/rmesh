use anyhow::{Result, anyhow};
use rayon::prelude::*;

use crate::{attributes::LoadSource, mesh::Trimesh};

pub struct BinaryStl {
    header: String,
    triangles: Vec<BinaryStlTriangle>,
}
#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BinaryStlTriangle {
    pub normal: [f32; 3],
    pub vertices: [f32; 9],
    pub attributes: u16,
}

impl BinaryStl {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 84 {
            return Err(anyhow::anyhow!("STL file too short"));
        }

        let header = String::from_utf8_lossy(&bytes[0..80]).trim().to_string();
        // let triangle_count = u32::from_le_bytes(bytes[80..84].try_into().unwrap());

        let triangles: &[BinaryStlTriangle] = bytemuck::try_cast_slice(&bytes[84..])
            .map_err(|_e| anyhow!("Could not interpret bytes as STL triangles!"))?;

        Ok(Self {
            header,
            triangles: triangles.to_vec(),
        })
    }

    pub fn to_mesh(&self) -> Result<Trimesh> {
        // convert STL f32 vertices to f64
        let vertices: Vec<f64> = self
            .triangles
            .par_iter()
            .flat_map(|t| {
                let vertices = t.vertices; // Copy the packed field to a local variable
                vertices.iter().map(|&v| v as f64).collect::<Vec<_>>()
            })
            .collect();

        let faces: Vec<usize> = (0..(vertices.len() / 3)).collect();

        // if the header was empty ignore it
        let header: Option<String> = if self.header.is_empty() {
            None
        } else {
            Some(self.header.clone())
        };

        let source = LoadSource {
            header,
            format: super::MeshFormat::STL,
        };

        let mut result = Trimesh::from_slice(&vertices, &faces)?;
        result.source = source;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::exchange::{MeshFormat, load_mesh};

    #[test]
    fn test_mesh_stl() {
        let stl_data = include_bytes!("../../../../test/data/unit_cube.STL");

        let mesh = load_mesh(stl_data, MeshFormat::STL).unwrap();

        assert_eq!(mesh.vertices.len(), 36);
        assert_eq!(mesh.faces.len(), 12);
    }
}
