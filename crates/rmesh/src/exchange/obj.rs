use std::collections::HashMap;

use anyhow::Result;
use nalgebra::{Point3, Vector2, Vector3, Vector4};
use rayon::prelude::*;

use crate::attributes::{Attributes, DEFAULT_COLOR, Grouping, GroupingKind, Material};
use crate::creation::Triangulator;
use crate::mesh::Trimesh;

use super::mtl::parse_mtl;
use crate::resolvers::Resolver;

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
    Ignore,
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
                // OBJ has V=0 at the bottom; GPU texture coords have V=0 at the top.
                ObjLine::Vt(Vector2::new(
                    u.parse().unwrap(),
                    1.0 - v.parse::<f64>().unwrap(),
                ))
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

            _ => ObjLine::Ignore,
        }
    }
}

/// Insert `name` into `values` if absent and return its index.
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

// In an OBJ file, directives like "usemtl", "g", "o", "s" apply to all
// subsequent faces until overridden. We track the current state and
// record per-face indices into the name lookup tables.
#[derive(Default, Clone)]
struct ObjFaces {
    // Current state indices (set by directives)
    pub material: usize,
    pub group: usize,
    pub smooth: usize,
    pub object: usize,

    // Face vertex indices
    pub faces: Vec<[usize; 3]>,

    // Per-face-vertex UV indices from `f v/vt/vn` lines
    pub face_uv: Vec<[Option<usize>; 3]>,

    // Per-face-vertex normal indices from `f v/vt/vn` lines
    pub face_normal: Vec<[Option<usize>; 3]>,

    // Per-face attribute indices (one entry per face)
    pub face_material: Vec<usize>,
    pub face_group: Vec<usize>,
    pub face_smooth: Vec<usize>,
    pub face_object: Vec<usize>,

    // Name lookup tables
    pub materials: Vec<String>,
    pub groups: Vec<String>,
    pub smooths: Vec<String>,
    pub objects: Vec<String>,
}

impl ObjFaces {
    /// Material operations for OBJ faces
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

    /// Triangulate raw face data and record per-face attributes.
    /// Raw faces can be arbitrary polygons with vertex/uv/normal indices.
    pub fn extend(
        &mut self,
        raw: &[Vec<Option<usize>>],
        vertices: &[Point3<f64>],
        triangulator: &mut Triangulator,
    ) {
        // Extract vertex, UV, and normal indices from the raw data.
        // OBJ format is `f v/vt/vn` where vt and vn are optional.
        // Indices are 1-based in OBJ, convert to 0-based here.
        let f: Vec<usize> = raw.iter().map(|v| v[0].unwrap_or(0) - 1).collect();
        let f_uv: Vec<Option<usize>> = raw
            .iter()
            .map(|v| v.get(1).copied().flatten().map(|i| i - 1))
            .collect();
        let f_normal: Vec<Option<usize>> = raw
            .iter()
            .map(|v| v.get(2).copied().flatten().map(|i| i - 1))
            .collect();

        // Triangulate the polygon into local indices (0..f.len()),
        // so we can use the same indices into f_uv and f_normal.
        let tri_indices: Vec<[usize; 3]> = if f.len() == 3 {
            vec![[0, 1, 2]]
        } else if f.len() == 4 {
            vec![[0, 1, 2], [0, 2, 3]]
        } else if f.len() > 4 {
            triangulator
                .triangulate_3d(&f, &[], vertices, true, true)
                .unwrap()
        } else {
            vec![]
        };

        // Record per-face attributes for each resulting triangle
        let num_tris = tri_indices.len();
        for ti in &tri_indices {
            self.faces.push([f[ti[0]], f[ti[1]], f[ti[2]]]);
            self.face_uv.push([f_uv[ti[0]], f_uv[ti[1]], f_uv[ti[2]]]);
            self.face_normal
                .push([f_normal[ti[0]], f_normal[ti[1]], f_normal[ti[2]]]);
        }
        self.face_object
            .extend(std::iter::repeat_n(self.object, num_tris));
        self.face_group
            .extend(std::iter::repeat_n(self.group, num_tris));
        self.face_material
            .extend(std::iter::repeat_n(self.material, num_tris));
        self.face_smooth
            .extend(std::iter::repeat_n(self.smooth, num_tris));
    }

    /// Check if any face has UV or normal indices.
    pub fn has_uv_or_normal(&self) -> bool {
        self.face_uv.iter().any(|f| f.iter().any(|i| i.is_some()))
            || self
                .face_normal
                .iter()
                .any(|f| f.iter().any(|i| i.is_some()))
    }

    /// Build face-level attributes (groupings) from the recorded per-face indices.
    pub fn to_attributes(&self) -> Option<Attributes> {
        let mut attributes = Attributes::default();

        if !self.objects.is_empty() {
            attributes.groupings.push(Grouping {
                kind: GroupingKind::Object,
                names: self.objects.clone(),
                indices: self.face_object.clone(),
            });
        }
        if !self.groups.is_empty() {
            attributes.groupings.push(Grouping {
                kind: GroupingKind::Group,
                names: self.groups.clone(),
                indices: self.face_group.clone(),
            });
        }
        if !self.materials.is_empty() {
            attributes.groupings.push(Grouping {
                kind: GroupingKind::Material,
                names: self.materials.clone(),
                indices: self.face_material.clone(),
            });
        }
        if !self.smooths.is_empty() {
            attributes.groupings.push(Grouping {
                kind: GroupingKind::Smoothing,
                names: self.smooths.clone(),
                indices: self.face_smooth.clone(),
            });
        }

        if attributes.groupings.is_empty() {
            None
        } else {
            Some(attributes)
        }
    }
}

/// Unmerge vertices so that position, UV, and normal arrays are vertex-aligned.
///
/// OBJ files index positions, UVs, and normals independently (e.g. `f v/vt/vn`).
/// A mesh requires all per-vertex attributes to share a single index.
/// This function creates a new vertex for each unique `(v, vt, vn)` combination.
struct UnmergedMesh {
    vertices: Vec<Point3<f64>>,
    faces: Vec<[usize; 3]>,
    attributes: Attributes,
}

fn unmerge_vertices(verts: &ObjVertices, faces: &ObjFaces) -> UnmergedMesh {
    // Map from (position_idx, uv_idx, normal_idx) → new vertex index
    let mut key_map: HashMap<(usize, Option<usize>, Option<usize>), usize> = HashMap::new();

    let estimated = faces.faces.len() * 3;
    let mut new_vertices: Vec<Point3<f64>> = Vec::with_capacity(estimated);
    let mut new_uv: Vec<Vector2<f64>> = Vec::with_capacity(estimated);
    let mut new_normal: Vec<Vector3<f64>> = Vec::with_capacity(estimated);
    let mut new_faces: Vec<[usize; 3]> = Vec::with_capacity(faces.faces.len());

    let has_uv = !verts.uv.is_empty();
    let has_normal = !verts.normal.is_empty();

    for (fi, face) in faces.faces.iter().enumerate() {
        let uv_face = &faces.face_uv[fi];
        let normal_face = &faces.face_normal[fi];

        let mut new_face = [0usize; 3];
        for corner in 0..3 {
            let v_idx = face[corner];
            let vt_idx = uv_face[corner];
            let vn_idx = normal_face[corner];

            let key = (v_idx, vt_idx, vn_idx);
            let new_idx = *key_map.entry(key).or_insert_with(|| {
                let idx = new_vertices.len();
                new_vertices.push(verts.vertices[v_idx]);
                if has_uv {
                    if let Some(ti) = vt_idx {
                        new_uv.push(verts.uv[ti]);
                    } else {
                        new_uv.push(Vector2::zeros());
                    }
                }
                if has_normal {
                    if let Some(ni) = vn_idx {
                        new_normal.push(verts.normal[ni]);
                    } else {
                        new_normal.push(Vector3::zeros());
                    }
                }
                idx
            });

            new_face[corner] = new_idx;
        }
        new_faces.push(new_face);
    }

    // Build vertex-aligned attributes
    let mut attributes = Attributes::default();

    if !new_uv.is_empty() {
        attributes.uv.push(new_uv);
    }
    if !new_normal.is_empty() {
        attributes.normals.push(new_normal);
    }

    // Remap vertex colors to the new vertex indices
    if !verts.color.is_empty() {
        let mut colors = vec![DEFAULT_COLOR; new_vertices.len()];
        let color_map: HashMap<usize, Vector4<u8>> = verts.color.iter().copied().collect();
        for (&(v_idx, _, _), &new_idx) in &key_map {
            if let Some(&c) = color_map.get(&v_idx) {
                colors[new_idx] = c;
            }
        }
        attributes.colors.push(colors);
    }

    UnmergedMesh {
        vertices: new_vertices,
        faces: new_faces,
        attributes,
    }
}

pub struct ObjMesh {
    // the original indexed vertices from the OBJ file
    vertices: ObjVertices,

    // the indexed faces from the OBJ file
    faces: ObjFaces,

    // materials loaded from MTL files
    materials: Vec<Material>,
}

impl ObjMesh {
    /// Parse a string into an ObjMesh with an optional resolver for external references.
    ///
    /// If `resolver` is `None`, external files (MTL, textures) are silently skipped.
    pub fn from_string(data: &str, resolver: Option<&dyn Resolver>) -> Self {
        // Handle OBJ line continuation: backslash at end of line joins with next line
        let data = data.replace("\\\n", " ").replace("\\\r\n", " ");

        // parse the strings in parallel
        let lines: Vec<ObjLine> = data
            .lines()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(ObjLine::from_line)
            .collect();

        // the `vn`, `vt`, `v` lines which are independent of each other
        let mut vertex = ObjVertices::default();
        // the `f` lines which may reference any of the `v`, `vn`, `vt` lines
        let mut faces = ObjFaces::default();
        // materials loaded from MTL files
        let mut materials = Vec::new();

        // we may have to triangulate 3D polygon faces as we go
        // OBJ supports arbitrary polygons but we need triangles
        let mut triangulator = Triangulator::new();

        for line in lines.iter() {
            match line {
                ObjLine::V(p, color) => {
                    vertex.vertices.push(*p);
                    if let Some(c) = color {
                        // Use len() - 1 since we already pushed the vertex
                        vertex.color.push((vertex.vertices.len() - 1, *c));
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
                ObjLine::MtlLib(path) => {
                    // Try to load the MTL file using the resolver (if provided)
                    if let Some(res) = resolver
                        && let Ok(mtl_bytes) = res.resolve(path)
                    {
                        let mtl_str = String::from_utf8_lossy(&mtl_bytes);
                        materials.extend(parse_mtl(&mtl_str, resolver));
                    }
                }
                ObjLine::Ignore => (),
            }
        }

        ObjMesh {
            vertices: vertex,
            faces,
            materials,
        }
    }

    /// Get the primary name for this OBJ mesh.
    /// Uses the first object name if available, otherwise returns empty string.
    pub fn primary_name(&self) -> &str {
        self.faces.objects.first().map_or("", |s| s.as_str())
    }

    pub fn into_mesh(self) -> Result<Trimesh> {
        let attributes_face = self.faces.to_attributes();

        let mut mesh = if self.faces.has_uv_or_normal() {
            // Faces reference UV/normal indices that don't align with vertex indices.
            // Unmerge vertices so each unique (v, vt, vn) combination gets its own vertex.
            let unmerged = unmerge_vertices(&self.vertices, &self.faces);
            Trimesh::new(
                unmerged.vertices,
                unmerged.faces,
                Some(unmerged.attributes),
                attributes_face,
            )?
        } else {
            // No UV/normal indices on faces — use the original vertex data directly.
            let attributes_vertex = self.vertices.to_attributes();
            Trimesh::new(
                self.vertices.vertices,
                self.faces.faces,
                attributes_vertex,
                attributes_face,
            )?
        };

        mesh.materials = self.materials;

        Ok(mesh)
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
            // Cast is safe: clamp guarantees 0.0..=255.0 which fits in u8
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let byte = (value * 255.0).round().clamp(0.0, 255.0) as u8;
            color[i] = byte;
        } else {
            // if any of the values fail to parse return None
            return None;
        }
    }

    Some(color)
}

#[cfg(test)]
mod tests {

    use crate::exchange::{FileType, load};
    use crate::geometry::Geometry;

    use super::*;

    /// Helper to extract a Trimesh from a Scene's first geometry.
    fn get_mesh(scene: &crate::scene::Scene) -> &Trimesh {
        match scene.geometry.values().next().unwrap() {
            Geometry::Mesh(mesh) => mesh,
            _ => panic!("Expected Mesh geometry"),
        }
    }

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
        // make sure the OBJ file was loadable into a mesh (no resolver)
        let scene = load(data.as_bytes(), Some(FileType::OBJ), None).unwrap();
        let mesh = get_mesh(&scene);

        // after unmerging each unique (v, vt, vn) combo becomes a vertex
        assert_eq!(mesh.vertices.len(), 664);
        // should have loaded a face for every occurrence of 'f '
        assert_eq!(mesh.faces.len(), data.matches("\nf ").count());

        // UVs are vertex-aligned after unmerging
        assert!(!mesh.attributes_vertex.uv.is_empty());
        let uv = &mesh.attributes_vertex.uv[0];
        assert_eq!(uv.len(), mesh.vertices.len());
        // at least some UV coordinates should be nonzero
        assert!(uv.iter().any(|v| v.x != 0.0 || v.y != 0.0));

        // normals should also be vertex-aligned
        assert!(!mesh.attributes_vertex.normals.is_empty());
        let normals = &mesh.attributes_vertex.normals[0];
        assert_eq!(normals.len(), mesh.vertices.len());
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
            assert!(parsed.contains(req), "missing line: {req:?}");
        }

        // make sure the OBJ file was loadable into a mesh (no resolver)
        let scene = load(data.as_bytes(), Some(FileType::OBJ), None).unwrap();
        let mesh = get_mesh(&scene);

        // after unmerging, vertex count >= original `v` count
        assert!(mesh.vertices.len() >= data.matches("\nv ").count());
        // should have loaded a face for every occurrence of 'f '
        assert_eq!(mesh.faces.len(), data.matches("\nf ").count());
    }

    #[test]
    fn test_obj_objects() {
        let data = include_str!("../../../../test/data/basic.obj");
        let scene = load(data.as_bytes(), Some(FileType::OBJ), None).unwrap();
        let mesh = get_mesh(&scene);

        // Find the object grouping in face attributes
        let objects = mesh
            .attributes_face
            .groupings
            .iter()
            .find(|g| matches!(g.kind, GroupingKind::Object))
            .expect("should have object groupings");

        // Verify 3 objects were found
        assert_eq!(objects.names, vec!["Cone", "cube for life!!!", "tetra"]);

        // Verify per-face indices: 4 faces (Cone) + 12 faces (cube) + 4 faces (tetra) = 20
        assert_eq!(objects.indices.len(), 20);

        // First 4 faces belong to Cone (index 0)
        assert_eq!(&objects.indices[0..4], &[0, 0, 0, 0]);
        // Next 12 faces belong to cube (index 1)
        assert_eq!(&objects.indices[4..16], &[1; 12]);
        // Last 4 faces belong to tetra (index 2)
        assert_eq!(&objects.indices[16..20], &[2, 2, 2, 2]);
    }

    #[test]
    fn test_obj_materials() {
        use crate::attributes::Material;
        use crate::exchange::InMemoryResolver;

        let obj_data = include_str!("../../../../test/data/fuze.obj");
        let mtl_data = include_str!("../../../../test/data/fuze.obj.mtl");

        // Create an in-memory resolver with the MTL file
        let mut resolver = InMemoryResolver::new();
        resolver.insert("./fuze.obj.mtl", mtl_data.as_bytes().to_vec());

        let scene = load(obj_data.as_bytes(), Some(FileType::OBJ), Some(&resolver)).unwrap();
        let mesh = get_mesh(&scene);

        // Should have loaded 1 material
        assert_eq!(mesh.materials.len(), 1);

        // Verify the material properties
        match &mesh.materials[0] {
            Material::Simple(mat) => {
                assert_eq!(mat.name, "material_0");
                assert!(mat.diffuse.is_some());
                // Tr is non-standard and ignored; alpha should be None (defaults to opaque)
                assert_eq!(mat.alpha, None);
            }
            _ => panic!("expected SimpleMaterial"),
        }
    }
}
