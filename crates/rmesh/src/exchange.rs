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
    pub normal: [f32; 3],
    pub vertices: [f32; 9],
    pub attributes: u16,
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

        Trimesh::from_slice(&vertices, &faces)
    }
}

/// The intermediate representation of a single line from an OBJ file,
/// which can later be turned into a more useful structure.
#[derive(Debug, PartialEq)]
enum ObjLine {
    // A vertex position and optionally a vertex color in some OBJ exporters.
    V(Point3<f64>, Option<Vector4<u8>>),
    // A vertex normal
    Vn(Vector3<f64>),
    // A vertex UV texture coordinate
    Vt(Vector3<f64>),
    // An OBJ face
    F(Vec<Vec<Option<usize>>>),
    // A new-object command
    O(String),
    // A group command
    G(String),
    // A smoothing group command
    S(String),
    // A usemtl command
    UseMtl(String),
    // A mtllib command defining a particular material
    MtlLib(String),

    // Something we don't care about
    Ignore(String),
}

impl ObjLine {
    fn from_line(line: &str) -> Self {
        // clean up a raw OBJ line: ignore anything after a comment then cleanly split it
        let parts: Vec<&str> = line
            .split('#')
            .next()
            .unwrap_or_default()
            .trim()
            .split_whitespace()
            .collect();

        match parts.as_slice() {
            ["v", x, y, z] => ObjLine::V(
                Point3::new(x.parse().unwrap(), y.parse().unwrap(), z.parse().unwrap()),
                None,
            ),
            ["v", x, y, z, color @ ..] => {
                // they've encoded some other color data after the vertex
                ObjLine::V(
                    Point3::new(x.parse().unwrap(), y.parse().unwrap(), z.parse().unwrap()),
                    float_to_rgba(color),
                )
            }
            ["vn", x, y, z] => ObjLine::Vn(Vector3::new(
                x.parse().unwrap(),
                y.parse().unwrap(),
                z.parse().unwrap(),
            )),
            ["vt", u, v, _garbage @ ..] => {
                ObjLine::Vt(Vector3::new(u.parse().unwrap(), v.parse().unwrap(), 0.0))
            }
            ["o", name @ ..] => ObjLine::O(name.join(" ")),
            ["s", name @ ..] => ObjLine::S(name.join(" ")),
            ["g", name @ ..] => ObjLine::G(name.join(" ")),
            ["usemtl", name @ ..] => ObjLine::UseMtl(name.join(" ")),
            ["mtllib", name @ ..] => ObjLine::MtlLib(name.join(" ")),
            ["f", blob @ ..] => ObjLine::F(
                // this way of parsing supports face references like:
                // 1/2/3, 1//3, 1/2, 1
                // and will return None for any missing values which can be analyzed later
                blob.iter()
                    .map(|f| f.split('/').map(|s| s.parse::<usize>().ok()).collect())
                    .collect(),
            ),

            _ => ObjLine::Ignore(line.to_string()),
        }
    }
}

pub struct ObjMesh {
    // the raw values, most people shouldn't
    pub lines: Vec<ObjLine>,
}

impl ObjMesh {
    /// Parse a loaded string into an ObjMesh.
    pub fn from_string(data: &str) -> Result<Self> {
        let lines: Vec<ObjLine> = data
            .lines()
            .collect::<Vec<_>>()
            .iter() // TODO : check performance of par_iter ;)
            .map(|line| ObjLine::from_line(line))
            .collect();

        // todo : this is for debug
        for line in lines.iter() {
            println!("{:?}", line);
        }
        return Ok(Self { lines });
    }

    pub fn to_mesh(&self) -> Result<Trimesh> {
        let mut vertices: Vec<Point3<f64>> = vec![];
        let mut vertex_normals: Vec<Vector3<f64>> = vec![];
        let mut vertex_uvs: Vec<Vector3<f64>> = vec![];
        let mut vertex_groups: Vec<Vec<usize>> = vec![];
        let mut vertex_materials: Vec<usize> = vec![];
        let mut vertex_colors: Vec<Vector4<u8>> = vec![];

        let mut faces: Vec<(usize, usize, usize)> = vec![];

        let mut current_material: Option<String> = None;
        let mut current_group: Option<String> = None;
        for line in self.lines.iter() {
            match line {
                ObjLine::V(p, color) => {
                    vertices.push(*p);
                    if let Some(c) = color {
                        vertex_colors.push(*c);
                    } else {
                        vertex_colors.push(Vector4::new(255, 255, 255, 255));
                    }
                }
                ObjLine::Vn(n) => vertex_normals.push(*n),
                ObjLine::Vt(t) => vertex_uvs.push(*t),
                ObjLine::F(faces_raw) => (),
                ObjLine::O(_) => (),
                ObjLine::G(_) => (),
                ObjLine::S(_) => (),
                ObjLine::UseMtl(name) => current_material = Some(name.to_string()),
                ObjLine::MtlLib(_) => (),
                ObjLine::Ignore(_) => (),
            }
        }
        Trimesh::new(vertices, faces)
    }
}

/// Convert a string slice containing float color values to a Vector4<u8>.
fn float_to_rgba(raw: &[&str]) -> Option<Vector4<u8>> {
    if raw.len() < 3 {
        return None;
    }

    // start with only alpha set
    let mut color = [0u8, 0u8, 0u8, 255u8];
    for (i, c) in raw.iter().enumerate() {
        if i > 4 {
            break;
        }
        let value = c.parse::<f64>();
        match value {
            Ok(v) => color[i] = (v * 255.0).round().clamp(0.0, 255.0) as u8,
            Err(_) => return None,
        }
    }

    Some(color.into())
}

#[derive(Debug, Clone, PartialEq)]
// An enum to represent the different mesh file formats.
pub enum MeshFormat {
    STL,
    OBJ,
    PLY,
}

impl MeshFormat {
    /// Convert a string to a MeshFormat enum.
    pub fn from_string(s: &str) -> Result<Self> {
        // clean up to match 'stl', '.stl', ' .STL ', etc
        let binding = s.to_ascii_lowercase();
        let clean = binding.trim().trim_start_matches('.');
        match clean {
            "stl" => Ok(MeshFormat::STL),
            "obj" => Ok(MeshFormat::OBJ),
            "ply" => Ok(MeshFormat::PLY),
            _ => Err(anyhow::anyhow!("Unsupported file type: `{}`", clean)),
        }
    }
}

pub fn load_mesh(file_data: &[u8], file_type: MeshFormat) -> Result<Trimesh> {
    match file_type {
        MeshFormat::STL => BinaryStl::from_bytes(file_data)?.to_mesh(),
        MeshFormat::OBJ => ObjMesh::from_string(std::str::from_utf8(file_data)?)?.to_mesh(),
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
    fn test_mesh_format_keys() {
        // check our string cleanup logic
        assert_eq!(MeshFormat::from_string("stl").unwrap(), MeshFormat::STL);
        assert_eq!(MeshFormat::from_string("STL").unwrap(), MeshFormat::STL);
        assert_eq!(MeshFormat::from_string(".stl").unwrap(), MeshFormat::STL);
        assert_eq!(MeshFormat::from_string(".STL").unwrap(), MeshFormat::STL);
        assert_eq!(MeshFormat::from_string("  .StL ").unwrap(), MeshFormat::STL);
        assert_eq!(MeshFormat::from_string("obj").unwrap(), MeshFormat::OBJ);

        assert_eq!(MeshFormat::from_string("obj").unwrap(), MeshFormat::OBJ);

        assert_eq!(MeshFormat::from_string("ply").unwrap(), MeshFormat::PLY);
        assert_eq!(MeshFormat::from_string("PLY").unwrap(), MeshFormat::PLY);
        assert_eq!(MeshFormat::from_string(".ply").unwrap(), MeshFormat::PLY);
        assert_eq!(MeshFormat::from_string(".PLY").unwrap(), MeshFormat::PLY);
        assert_eq!(MeshFormat::from_string("  .pLy ").unwrap(), MeshFormat::PLY);

        assert!(MeshFormat::from_string("foo").is_err());
    }

    #[test]
    fn test_mesh_obj() {
        let data = include_str!("../../../test/data/basic.obj");

        let parsed = ObjMesh::from_string(data).unwrap().lines;

        // check a few parse results of more difficult lines
        let required: Vec<ObjLine> = vec![ObjLine::O("cube for life!!!".to_string())];

        // make sure we implemented the PartialEq trait
        assert_eq!(required[0], required[0]);

        // we should
        for req in required.iter() {
            assert!(parsed.contains(&req), "missing line: {:?}", req);
        }

        // make sure the OBJ file was loadable into a mesh
        let mesh = load_mesh(data.as_bytes(), MeshFormat::OBJ).unwrap();

        // should have loaded a vertex for every occurance of 'v '
        assert_eq!(mesh.vertices.len(), data.matches("v ").count());
        // todo : implement faces
        // should have loaded a face for every occurance of 'f '
        // assert_eq!(mesh.faces.len(), data.matches("f ").count());
    }
}
