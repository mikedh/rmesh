/// obj.rs
/// -----------
/// Parse OBJ files into a direct representation of their structure with separately indexed
/// vertices, normals, texture coordinates, and faces which can be used by users with specific
/// needs of the exact mapping. Or in most cases it is converted into a Trimesh with corresponding
/// attributes (i.e. `mesh.vertices[n]` corresponds to `mesh.attributes_vertex.color[n]`
use ahash::AHashMap;
use anyhow::Result;
use nalgebra::{Point3, Vector2, Vector3, Vector4};
use rayon::prelude::*;

use crate::attributes::{Attributes, DEFAULT_COLOR, Material};
use crate::creation::{Triangulator, triangulate_fan};
use crate::mesh::Trimesh;

/// The intermediate representation of a single line from an OBJ file,
/// which can later be turned into a more useful structure.
///
/// These can be evaluated in parallel as they are independent of each other.
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

/// A helper function to upsert a value into a vector and return its index.
///
/// Parameters
/// -----------
/// name
///   
fn upsert(name: &str, values: &mut Vec<String>) -> usize {
    if let Some(index) = values.iter().position(|m| m == name) {
        index
    } else {
        values.push(name.to_string());
        values.len() - 1
    }
}

// keep a bunch of mutable arrays as we go
#[derive(Default, Clone)]
struct ObjVertices {
    // the vertex positions from the `v` lines
    pub vertices: Vec<Point3<f64>>,

    // the non-corresponding normals from the `vn` lines
    pub normal: Vec<Vector3<f64>>,

    // the non-corresponding texture coordinates from the `vt` lines
    pub uv: Vec<Vector2<f64>>,

    // collect colors as a vertex index and a color
    // so that if only one vertex has a color we can index it later
    // and in the majority of cases we can do nothing as there
    // are no vertex colors
    pub color: Vec<(usize, Vector4<u8>)>,
}

impl ObjVertices {
    /// Convert the vertex data into a vector of attributes
    /// for the Trimesh.
    pub fn to_attributes(&self) -> Option<Attributes> {
        let mut attributes = Attributes::default();

        // Add vertex colors only if they exist
        if !self.color.is_empty() {
            // the colors are a tuple of (vertex index, color) pairs
            // since they may be  sparse and not all vertices have a color.
            // thus, start with a fully populated vector of the default color
            let mut color = vec![DEFAULT_COLOR; self.vertices.len()];
            for (i, c) in self.color.iter() {
                // replace just the color at the index
                color[*i] = *c;
            }
            // push our vertex-matching colors into the attributes
            attributes.colors.push(color);
        }

        // Add normals if any were populated.
        if !self.normal.is_empty() {
            attributes.normals.push(self.normal.clone());
        }

        // Add UVs
        if !self.uv.is_empty() {
            attributes.uv.push(self.uv.clone());
        }

        if attributes.colors.is_empty()
            && attributes.normals.is_empty()
            && attributes.uv.is_empty()
            && attributes.groupings.is_empty()
        {
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
    // the index of the current material set by `self.materials`
    pub material: usize,
    // the index of the current group set by `self.groups`
    pub group: usize,
    // the index of the current smoothing group set by `self.smooths`
    pub smooth: usize,
    // the index of the current object set by `self.objects`
    pub object: usize,

    // the indexes of `vertices.vertices`
    pub faces: Vec<(usize, usize, usize)>,
    pub faces_tex: Vec<Option<(usize, usize, usize)>>,
    pub face_normal: Vec<Option<(usize, usize, usize)>>,
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

impl ObjFaces {
    ///
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

    /// Implement the logic to triangulate raw face data which can contain any number
    /// of data points representing arbitrary polygons:
    ///   -- just vertex indices
    ///   -- vertex indices and texture coordinates
    ///   -- vertex indices, texture coordinates and normals
    ///   -- vertex indices and normals.
    pub fn extend(
        &mut self,
        raw: &[Vec<Option<usize>>],
        vertices: &[Point3<f64>],
        triangulator: &mut Triangulator,
    ) {
        // get just the indexes of the vertex positions, which we will use to triangulate
        // and the subsequent positions of normals and texture coordinates depend on this triangulation
        let f: Vec<usize> = raw.iter().map(|v| v[0].unwrap_or(0) - 1).collect();

        // get the triangles as indexes in our current face
        let (tri, index) = {
            // if we have a triangle this is easy
            if f.len() == 3 {
                (vec![(f[0], f[1], f[2])], vec![(0, 1, 2)])
            } else if f.len() == 4 {
                // if we have a quad split it into two triangles
                (
                    vec![(f[0], f[1], f[2]), (f[0], f[2], f[3])],
                    // we know the index
                    vec![(0, 1, 2), (0, 2, 3)],
                )
            } else if f.len() > 4 {
                // if we have a polygon triangulate it
                // TODO : do we have to do this in a second pass to avoid
                // referencing vertices that haven't been added yet?
                let triangles = triangulator
                    .triangulate_3d(&f, &[], vertices)
                    .unwrap_or_else(|_| triangulate_fan(&f));

                // we may have produced triangles in any order so we need to go from the full
                // vertex index to the index in our local group of faces `f`
                // this is a little indirect and could potentially be slow if you had a whole
                // lot of non-triangulated faces, but in practice I've never seen that
                // and if's a problem someone can refactor triangulate to use the original
                // vertex indexes instead of the local `f` indexes.
                let remap: AHashMap<usize, usize> =
                    f.iter().enumerate().map(|(i, &v)| (v, i)).collect();

                // map from triangles back into indexes of raw
                let index = triangles
                    .iter()
                    .map(|&(a, b, c)| {
                        (
                            remap.get(&a).copied().unwrap_or_default(),
                            remap.get(&b).copied().unwrap_or_default(),
                            remap.get(&c).copied().unwrap_or_default(),
                        )
                    })
                    .collect();

                (triangles, index)
            } else {
                // if we have less than 3 vertices we can't triangulate
                (vec![], vec![])
            }
        };

        // add the triangles referencing vertex positions
        self.faces.extend(tri);

        // add the triangles referencing texture coordinates if they exist
        self.faces_tex.extend(index.iter().map(|&(a, b, c)| {
            match (
                raw[a].get(1).cloned().unwrap_or(None),
                raw[b].get(1).cloned().unwrap_or(None),
                raw[c].get(1).cloned().unwrap_or(None),
            ) {
                (Some(u0), Some(u1), Some(u2)) => Some((u0, u1, u2)),
                _ => None,
            }
        }));
    }
}

pub struct ObjMesh {
    // the original indexed vertices from the OBJ file
    vertices: ObjVertices,

    // the indexed faces from the OBJ file
    faces: ObjFaces,

    // was this loaded in a "flattened" manner, which triangulated
    // every face and ensured that every vertex is unique?
    flattened: bool,
}

impl ObjMesh {
    /// Parse a string into an ObjMesh.
    pub fn from_string(data: &str, flatten: bool) -> Result<Self> {
        // parse the strings in parallel
        let lines: Vec<ObjLine> = data
            .lines()
            .collect::<Vec<_>>()
            .into_par_iter() // TODO : check performance of par_iter vs iter ;)
            .map(ObjLine::from_line)
            .collect();

        // the `vn``, `vt``, `v`` lines which are independent of each other
        let mut vertex = ObjVertices::default();
        // the `f` lines which may reference any of the `v`, `vn`, `vt` lines
        let mut faces = ObjFaces::default();

        // we may have to triangulate 3D polygon faces as we go
        // OBJ supports arbitrary polygons but we need triangles
        let mut triangulator = Triangulator::new();

        for line in lines.iter() {
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
                    faces.extend(raw, &vertex.vertices, &mut triangulator);
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

        Ok(ObjMesh {
            vertices: vertex,
            faces,
            flattened: flatten,
        })
    }

    pub fn to_mesh(self) -> Result<Trimesh> {
        // "flatten" the mesh to ensure each vertex matches
        let attributes_vertex = self.vertices.to_attributes().unwrap_or_default();

        Ok(Trimesh {
            vertices: self.vertices.vertices,
            faces: self.faces.faces,
            attributes_vertex,
            ..Default::default()
        })
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
        // parse the strings in parallel
        let parsed: Vec<ObjLine> = data
            .lines()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(ObjLine::from_line)
            .collect();

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
