use anyhow::{Result, anyhow};
use itertools::Itertools;
use nalgebra::{Point3, Vector3, Vector4, convert};
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

/// The intermediate representation of a single line from an OBJ file,
/// which can later be turned into a more useful structure.
#[derive(Debug)]
enum ObjLine {
    // A vertex position
    V(Point3<f64>),
    // A vertex normal
    Vn(Vector3<f64>),
    // A vertex UV texture coordinate
    Vt(Vector3<f64>),
    // A vertex color
    Vc(Vector4<f64>),
    // An OBJ face
    F(Vec<Vec<Option<usize>>>),
    // A new-object command
    O(String),
    // A group command
    G(String),
    // A usemtl command
    UseMtl(String),
    // A mtllib command defining a particular material
    MtlLib(String),
    // Somethign we don't care about
    Ignore(String),
}

impl ObjLine {
    fn from_line(line: &str) -> Self {
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        match parts.as_slice() {
            ["v", x, y, z] => ObjLine::V(Point3::new(
                x.parse().unwrap(),
                y.parse().unwrap(),
                z.parse().unwrap(),
            )),
            ["vn", x, y, z] => ObjLine::Vn(Vector3::new(
                x.parse().unwrap(),
                y.parse().unwrap(),
                z.parse().unwrap(),
            )),
            ["vt", u, v] => ObjLine::Vt(Vector3::new(u.parse().unwrap(), v.parse().unwrap(), 0.0)),
            ["o", name] => ObjLine::O(name.to_string()),
            ["g", name] => ObjLine::G(name.to_string()),
            ["f", blob @ ..] => {
                // so keep them
                // the OBJ format allows v/vt/vn, v//vn, v/vt, v
                let payload: Vec<Vec<Option<usize>>> = blob
                    .iter()
                    .map(|f| f.split('/').map(|s| s.parse::<usize>().ok()).collect())
                    .collect();

                // println!("blob: {:?}", blob);
                // println!("payload: {:?}", payload);
                ObjLine::F(payload)
            }

            _ => ObjLine::Ignore(line.to_string()),
        }
    }
}

pub struct ObjMesh {
    lines: Vec<ObjLine>,
}

impl ObjMesh {
    pub fn from_string(data: &str) -> Result<Self> {
        let lines: Vec<ObjLine> = data
            .lines()
            .collect::<Vec<_>>()
            .iter() // TODO : check performance of par_iter ;)
            .map(|line| ObjLine::from_line(line))
            .collect();

        println!("lines: {:?}", lines);

        return Ok(Self { lines });
    }

    pub fn to_mesh(&self) -> Trimesh {
        // convert OBJ f32 vertices to f64
        let vertices: Vec<Point3<f64>> = vec![];
        let faces: Vec<(usize, usize, usize)> = vec![];
        Trimesh::new(vertices, faces)
    }
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
        MeshFormat::OBJ => Ok(ObjMesh::from_string(std::str::from_utf8(file_data)?)?.to_mesh()),
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

    #[test]
    fn test_mesh_obj() {
        let data = include_bytes!("../../../test/data/basic.obj");

        let mesh = load_mesh(data, MeshFormat::OBJ).unwrap();

        //assert_eq!(mesh.vertices.len(), 36);
        //assert_eq!(mesh.faces.len(), 12);
    }
}
