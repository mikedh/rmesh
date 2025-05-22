use anyhow::{Result, anyhow};
use rayon::prelude::*;

use crate::{attributes::LoadSource, mesh::Trimesh};

pub struct BinaryStl {
    header: String,
    triangles: Vec<StlTriangle>,
}
#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct StlTriangle {
    pub normal: [f32; 3],
    pub vertices: [f32; 9],
    pub attributes: u16,
}
// The size of each triangle in bytes
const STL_TRIANGLE_SIZE: usize = std::mem::size_of::<StlTriangle>();

impl BinaryStl {
    /// Parse a binary or ASCII STL file from the raw bytes. Note that binary STL files
    /// must exactly match the size specified in the header, or they will be parsed as
    /// ASCII STL files and error later.
    ///
    /// Parameters
    /// ------------
    /// bytes
    ///   Raw bytes of the STL file.
    ///
    /// Returns
    /// ------------
    /// Result<Self>
    ///   A Result containing the parsed STL file or an error.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 84 {
            return Err(anyhow::anyhow!("STL file too short"));
        }

        let header = String::from_utf8_lossy(&bytes[0..80]).trim().to_string();
        // the number of triangles is stored as a little-endian u32 at bytes 80-84
        let triangle_count = u32::from_le_bytes(bytes[80..84].try_into().unwrap());

        // if our passed bytes are not a
        if bytes.len() != 84 + (triangle_count as usize) * STL_TRIANGLE_SIZE {
            // this may be an ASCII STL file
            return Self::parse_ascii_stl(bytes);
            // return Err(anyhow::anyhow!("STL file size does not match header"));
        }
        // we are

        let triangles: &[StlTriangle] = bytemuck::try_cast_slice(&bytes[84..])
            .map_err(|_e| anyhow!("Could not interpret bytes as STL triangles!"))?;

        Ok(Self {
            header,
            triangles: triangles.to_vec(),
        })
    }

    /// Parse an ASCII STL file.
    fn parse_ascii_stl(bytes: &[u8]) -> Result<Self> {
        let text = String::from_utf8_lossy(bytes);

        let header = text
            .lines()
            .next()
            .ok_or_else(|| anyhow!("STL file is empty"))?
            .to_string();

        // split the text into chunks between the `facet` and `endfacet` keywords
        let chunks = text.split("facet").collect::<Vec<_>>();

        //println!("chunks: {:?}", chunks.clone());

        let triangles = chunks
            .par_iter()
            .map(|chunk| {
                let mut normal = [0.0f32; 3];
                let mut vertices = [0.0f32; 9];
                let mut vertex_count = 0;

                for line in chunk.lines() {
                    let mut parts = line.split_whitespace();
                    //println!("parts: {:?}", parts.clone().collect::<Vec<_>>());
                    match parts.next() {
                        Some("normal") => {
                            // Handles: "facet normal x y z"
                            for i in 0..3 {
                                normal[i] = match parts.next().and_then(|v| v.parse().ok()) {
                                    Some(val) => val,
                                    None => return None,
                                };
                            }
                        }
                        Some("vertex") => {
                            // Handles: "vertex x y z"
                            if vertex_count >= 3 {
                                break;
                            }
                            for i in 0..3 {
                                vertices[vertex_count * 3 + i] =
                                    match parts.next().and_then(|v| v.parse().ok()) {
                                        Some(val) => val,
                                        None => return None,
                                    };
                            }
                            vertex_count += 1;
                        }
                        _ => {}
                    }
                }

                if vertex_count == 3 {
                    Some(StlTriangle {
                        normal,
                        vertices,
                        attributes: 0,
                    })
                } else {
                    None
                }
            })
            .filter_map(|t| t)
            .collect::<Vec<_>>();
        //println!("triangles: {:?}", triangles.clone());

        Ok(Self { header, triangles })
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
            format: Some(super::MeshFormat::STL),
        };

        let mut result = Trimesh::from_slice(&vertices, &faces)?;
        result.source = source;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {

    use crate::exchange::{MeshFormat, load_mesh};

    #[test]
    fn test_mesh_binary_stl() {
        let stl_data = include_bytes!("../../../../test/data/unit_cube.STL");

        let mesh = load_mesh(stl_data, MeshFormat::STL).unwrap();

        assert_eq!(mesh.vertices.len(), 36);
        assert_eq!(mesh.faces.len(), 12);
    }

    #[test]
    fn test_mesh_ascii_stl() {
        let stl_data = include_bytes!("../../../../test/data/two_objects_mixed_case_names.stl");
        let mesh = load_mesh(stl_data, MeshFormat::STL).unwrap();

        //assert_eq!(mesh.vertices.len(), 36);
        assert_eq!(mesh.faces.len(), 24);
    }
}
