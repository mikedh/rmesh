use anyhow::Result;
use nalgebra::{Point3, Vector2, Vector3, Vector4};
use rayon::prelude::*;

use crate::attributes::{Attribute, Material};
use crate::creation::{Triangulator, triangulate_fan};
use crate::mesh::Trimesh;

/// The intermediate representation of a single line from an OBJ file,
/// which can later be turned into a more useful structure.
#[derive(Debug, PartialEq)]
enum ObjLine {
    // A vertex position and optionally a vertex color in some OBJ exporters.
    V(Point3<f64>, Option<Vector4<u8>>),
    // A vertex normal
    Vn(Vector3<f64>),
    // A vertex UV texture coordinate
    Vt(Vector2<f64>),
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
                ObjLine::Vt(Vector2::new(u.parse().unwrap(), v.parse().unwrap()))
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

    fn load_materials(&self) -> Option<Vec<Material>> {
        match self {
            ObjLine::MtlLib(name) => {
                // TODO : load the materials from the file
                // and return them as a vector of Materials
                // for now just return an empty vector
                Some(vec![])
            }
            _ => None,
        }
    }
}

pub struct ObjMesh {
    // each line of the OBJ file parsed into a native type
    // but not evaluated into a mesh
    lines: Vec<ObjLine>,
    materials: Vec<Material>,
}

impl ObjMesh {
    /// Parse a loaded string into an ObjMesh.
    pub fn from_string(data: &str) -> Result<Self> {
        let lines: Vec<ObjLine> = data
            .lines()
            .collect::<Vec<_>>()
            .par_iter() // TODO : check performance of par_iter vs iter ;)
            .map(|line| ObjLine::from_line(line))
            .collect();
        Ok(Self {
            lines,
            materials: vec![],
        })
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
            pub uv: Vec<Vector2<f64>>,
            // collect colors as a vertex index and a color
            // so that if only one vertex has a color we can index it later
            // and in the majority of cases we can do nothing as there
            // are no vertex colors
            pub color: Vec<(usize, Vector4<u8>)>,
        }

        impl Vertices {
            /// Convert the vertex data into a vector of attributes
            /// for the Trimesh.
            pub fn to_attributes(&self) -> Option<Vec<Attribute>> {
                let mut attributes = vec![];

                // add the vertex colors
                if !self.color.is_empty() {
                    let mut color = vec![Vector4::new(0, 0, 0, 255); self.vertices.len()];
                    for (i, c) in self.color.iter() {
                        color[*i] = *c;
                    }
                    attributes.push(Attribute::Color(color));
                }

                // add the normals
                if !self.normal.is_empty() {
                    attributes.push(Attribute::Normal(self.normal.clone()));
                }

                // add the UVs
                if !self.uv.is_empty() {
                    attributes.push(Attribute::UV(self.uv.clone()));
                }

                if attributes.is_empty() {
                    None
                } else {
                    Some(attributes)
                }
            }
        }

        // in an OBJ file if there is a directive like "usemtl" or "g"
        // it means that the faces or vertices that follow it are part of that
        // directive until it's overridden by another directive
        // so we need to keep track of the current directive and apply it as we go.
        #[derive(Default, Clone)]
        struct ObjFaces {
            // the index of the current value
            pub material: usize,
            pub group: usize,
            pub smooth: usize,
            pub object: usize,

            // the faces we're collecting
            pub faces: Vec<(usize, usize, usize)>,
            pub faces_material: Vec<usize>,
            pub faces_group: Vec<usize>,
            pub faces_smooth: Vec<usize>,
            pub faces_object: Vec<usize>,

            // now the actual collected values
            // the *name* of the material that we will use for the index `material`
            pub materials: Vec<String>,
            pub groups: Vec<String>,
            pub smooths: Vec<String>,
            pub objects: Vec<String>,

            // the actual materials which may not match the order of `materials` name
            // until we load them from the file and re-order them at the end.
            pub materials_obj: Vec<Material>,
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

        impl ObjFaces {
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

            /// here's where we do the logic to add faces and keep track of attributes.
            pub fn extend(&mut self, faces: &[(usize, usize, usize)]) {
                self.faces.extend(faces);
                self.faces_material.extend(vec![self.material; faces.len()]);
            }
        }

        // todo : this one sucks
        let mut vertex = Vertices::default();
        let mut faces = ObjFaces::default();

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
                        faces.extend(&[(f[0], f[1], f[2])]);
                    } else if f.len() == 4 {
                        // if we have a quad split it into two triangles
                        faces.extend(&[(f[0], f[1], f[2]), (f[0], f[2], f[3])]);
                    } else if f.len() > 4 {
                        // if we have a polygon triangulate it
                        // TODO : do we have to do this in a second pass to avoid
                        // referencing vertices that haven't been added yet?
                        if let Ok(tri) = triangulator.triangulate_3d(&f, &[], &vertex.vertices) {
                            faces.extend(&tri);
                        } else {
                            // if our fancy triangulator fails we can
                            // always fall back to a fan triangulation
                            faces.extend(&triangulate_fan(&f));
                        }
                    }
                }
                ObjLine::O(name) => faces.upsert_object(name),
                ObjLine::G(name) => faces.upsert_group(name),
                ObjLine::S(name) => faces.upsert_smooth(name),
                ObjLine::UseMtl(name) => faces.upsert_material(name),
                ObjLine::MtlLib(_) => {
                    // try to load the materials from the `mtl` file specified
                    if let Some(materials) = line.load_materials() {
                        faces.materials_obj.extend(materials);
                    }
                }
                ObjLine::Ignore(_) => (),
            }
        }

        let vertex_attributes = vertex.to_attributes();

        Trimesh::new(vertex.vertices, faces.faces, vertex_attributes)
    }
}

/// Convert a string slice containing 0.0 to 1.0 float colors
/// to a vector color.
///
/// Parameters
/// -----------
/// raw
///   A slice of string slices containing the color values.
///
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
        if let Ok(value) = c.parse::<f64>() {
            color[i] = (value * 255.0).round().clamp(0.0, 255.0) as u8;
        } else {
            // if any of the values fail to parse return None
            return None;
        }
    }

    Some(color)
}

#[cfg(test)]
mod tests {

    use crate::exchange::{MeshFormat, load_mesh};

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
    fn test_mesh_obj_tex() {
        // has many of the test cases we need
        let data = include_str!("../../../../test/data/fuze.obj");
        // make sure the OBJ file was loadable into a mesh
        let mesh = load_mesh(data.as_bytes(), crate::exchange::MeshFormat::OBJ).unwrap();

        // should have loaded a vertex for every occurrence of 'v '
        assert_eq!(mesh.vertices.len(), data.matches("\nv ").count());
        // todo : implement faces
        // should have loaded a face for every occurrence of 'f '
        assert_eq!(mesh.faces.len(), data.matches("\nf ").count());

        assert!(mesh.uv().is_some());
        let uv = mesh.uv().unwrap();
        assert_eq!(uv.len(), data.matches("\nvt ").count());

        // here's the big tricky TODO
        // assert_eq!(uv.len(),mesh.vertices.len());
    }

    #[test]
    fn test_mesh_obj() {
        // has many of the test cases we need
        let data = include_str!("../../../../test/data/basic.obj");

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
        assert_eq!(mesh.vertices.len(), data.matches("\nv ").count());
        // todo : implement faces
        // should have loaded a face for every occurrence of 'f '
        assert_eq!(mesh.faces.len(), data.matches("\nf ").count());

        println!("mesh: {:?}", mesh);
    }
}
