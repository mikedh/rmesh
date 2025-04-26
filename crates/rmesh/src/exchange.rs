use anyhow::{Result, anyhow};
use itertools::Itertools;
use nalgebra::{Point3, Vector3, Vector4, convert};
use rayon::prelude::*;

use crate::creation::{Triangulator, triangulate_fan};
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
    /// Parse a single raw OBJ line into native types
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
                    str_to_rgba(color),
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
    // each line of the OBJ file parsed into a native type
    // but not evaluated into a mesh
    lines: Vec<ObjLine>,
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

    /// Convert the parsed OBJ file into a mesh.
    ///
    ///
    /// Returns
    /// ---------
    /// mesh
    ///   A Trimesh object containing the vertices and faces of the mesh
    ///   and attributes for normals, colors, materials, and UVs.
    pub fn to_mesh(&self) -> Result<Trimesh> {
        // keep a bunch of mutable arrays as we go
        #[derive(Default)]
        struct Vertices {
            pub vertices: Vec<Point3<f64>>,
            pub normal: Vec<Vector3<f64>>,
            pub uv: Vec<Vector3<f64>>,
            pub group: Vec<Vec<usize>>,
            pub material: Vec<usize>,
            // collect colors as a vertex index and a color
            // so that if only one vertex has a color we can index it later
            // and in the majority of cases we can do nothing as there
            // are no vertex colors
            pub color: Vec<(usize, Vector4<u8>)>,
        }

        // in an OBJ file if there is a directive like "usemtl" or "g"
        // it means that the faces or vertices that follow it are part of that
        // directive until it's overridden by another directive
        // so we need to keep track of the current directive and apply it as we go.
        #[derive(Default, Clone)]
        struct State {
            // the index of the current value
            pub material: usize,
            pub group: usize,
            pub smooth: usize,
            pub object: usize,

            // now the actual collected values that the indices point to
            pub materials: Vec<String>,
            pub groups: Vec<String>,
            pub smooths: Vec<String>,
            pub objects: Vec<String>,
        }

        /// A helper function to upsert a value into a vector and return its index.
        fn upsert(name: &str, values: &mut Vec<String>) -> usize {
            if let Some(index) = values.iter().position(|m| m == name) {
                index
            } else {
                values.push(name.to_string());
                values.len() - 1
            }
        }

        impl State {
            pub fn upsert_material(&mut self, name: &str) {
                self.material = upsert(name, &mut self.materials);
            }
            pub fn upsert_group(&mut self, name: &str) {
                self.group = upsert(name, &mut self.groups);
            }
            pub fn upsert_smooth(&mut self, name: &str) {
                self.smooth = upsert(name, &mut self.smooths);
            }
            pub fn upsert_object(&mut self, name: &str) {
                self.object = upsert(name, &mut self.objects);
            }
        }

        // todo : this one sucks
        let mut faces: Vec<(usize, usize, usize)> = vec![];
        let mut vertex = Vertices::default();
        let mut state = State::default();

        // we may have to triangulate 3D polygon faces as we go
        let mut triangulator = Triangulator::new();

        for line in self.lines.iter() {
            match line {
                ObjLine::V(p, color) => {
                    vertex.vertices.push(*p);
                    if let Some(c) = color {
                        vertex.color.push((vertex.vertices.len(), *c));
                    }
                }
                ObjLine::Vn(n) => vertex.normal.push(*n),
                ObjLine::Vt(t) => vertex.uv.push(*t),
                ObjLine::F(raw) => {
                    // just take the vertex index for now
                    let f: Vec<usize> = raw.iter().map(|v| v[0].unwrap_or(0) - 1).collect();

                    if f.len() == 3 {
                        // if we have a triangle this is easy
                        faces.push((f[0], f[1], f[2]));
                    } else if f.len() == 4 {
                        // if we have a quad split it into two triangles
                        faces.push((f[0], f[1], f[2]));
                        faces.push((f[0], f[2], f[3]));
                    } else if f.len() > 4 {
                        // if we have a polygon triangulate it
                        // TODO : ugh do we have to do this in a second pass to avoid
                        // referencing vertices that haven't been added yet?
                        if let Ok(tri) = triangulator.triangulate_3d(&f, &[], &vertex.vertices) {
                            faces.extend(tri);
                        } else {
                            // if our fancy triangulator fails we can
                            // always fall back to a fan triangulation
                            faces.extend(triangulate_fan(&f));
                        }
                    }
                }
                ObjLine::O(name) => state.upsert_object(name),
                ObjLine::G(name) => state.upsert_group(name),
                ObjLine::S(name) => state.upsert_smooth(name),
                ObjLine::UseMtl(name) => state.upsert_material(name),

                ObjLine::MtlLib(_) => (),
                ObjLine::Ignore(_) => (),
            }
        }
        Trimesh::new(vertex.vertices, faces)
    }
}

/// Convert a string slice containing 0.0 to 1.0 float colors
/// to a Vector4<u8> color.
///
/// Parameters
/// -----------
/// raw
///   A slice of string slices containing the color values.
/// Returns
/// --------
///   An RGBA color or None if the input is invalid.
fn str_to_rgba(raw: &[&str]) -> Option<Vector4<u8>> {
    if raw.len() < 3 {
        return None;
    }

    // start with only alpha
    let mut color: Vector4<u8> = Vector4::new(0u8, 0u8, 0u8, 255u8);
    for (i, c) in raw.iter().take(4).enumerate() {
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
        let clean = binding.trim().trim_start_matches('.').trim();
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
    fn test_color_parse() {
        let raw = vec!["0.5", "0.5", "0.5", "0.5"];
        let color = str_to_rgba(&raw).unwrap();
        assert_eq!(color, Vector4::new(128, 128, 128, 128));

        let raw = vec!["0.5", "0.5", "0.5"];
        let color = str_to_rgba(&raw).unwrap();
        assert_eq!(color, Vector4::new(128, 128, 128, 255));
        let raw = vec!["0.5", "0.5"];
        let color = str_to_rgba(&raw);
        assert_eq!(color, None);
        let raw = vec!["1.0", "1", "1", "0.0"];
        let color = str_to_rgba(&raw).unwrap();
        assert_eq!(color, Vector4::new(255, 255, 255, 0));
    }

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

        // should have loaded a vertex for every occurrence of 'v '
        assert_eq!(mesh.vertices.len(), data.matches("v ").count());
        // todo : implement faces
        // should have loaded a face for every occurrence of 'f '
        assert_eq!(mesh.faces.len(), data.matches("f ").count());

        println!("mesh: {:?}", mesh);
    }
}
