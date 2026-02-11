//! Bidirectional conversion between Collada schema types and rmesh Scene/Trimesh.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use nalgebra::{Matrix4, Point3, Rotation3, Unit, Vector2, Vector3};

use super::schema;
use crate::attributes::{Attributes, Grouping, GroupingKind, Material, SimpleMaterial, UNSET};
use crate::exchange::FileType;
use crate::geometry::Geometry;
use crate::image::LazyImage;
use crate::mesh::Trimesh;
use crate::resolvers::Resolver;
use crate::scene::{Scene, SceneNode, SceneNodeKind};

// ── Shared Helpers ──────────────────────────────────────────────────────────

/// Strip "#" prefix from URI fragment references.
fn strip_fragment(url: &str) -> &str {
    url.strip_prefix('#').unwrap_or(url)
}

/// Create a "#id" fragment reference.
fn fragment_ref(id: &str) -> String {
    format!("#{id}")
}

/// Parsed source data from a Collada `<source>` element.
struct SourceData<'a> {
    data: &'a [f64],
    stride: usize,
}

impl SourceData<'_> {
    /// Get the data for element at `index` as a slice of `stride` values.
    fn get(&self, index: usize) -> Option<&[f64]> {
        let base = index * self.stride;
        self.data.get(base..base + self.stride)
    }
}

/// Read float source data with stride from accessor.
fn read_source(source: &schema::SourceElementType) -> Option<SourceData<'_>> {
    let float_array = source.float_array()?;
    let tc = source.technique_common()?;
    Some(SourceData {
        data: &float_array.content.0,
        stride: tc.accessor.stride as usize,
    })
}

/// Compute the combined transform for a node from its ordered content elements.
fn node_transform(node: &schema::NodeElementType) -> Option<Matrix4<f64>> {
    let mut m = Matrix4::identity();
    let mut any = false;
    for item in &node.content {
        match item {
            schema::NodeElementTypeContent::Matrix(mat) => {
                if mat.content.0.len() >= 16 {
                    // Collada stores matrices in row-major order
                    m *= Matrix4::from_row_slice(&mat.content.0[..16]);
                    any = true;
                }
            }
            schema::NodeElementTypeContent::Translate(t) => {
                let v = &t.content.0;
                if v.len() >= 3 {
                    m *= Matrix4::new_translation(&Vector3::new(v[0], v[1], v[2]));
                    any = true;
                }
            }
            schema::NodeElementTypeContent::Rotate(r) => {
                let v = &r.content.0;
                if v.len() >= 4 {
                    let axis_vec = Vector3::new(v[0], v[1], v[2]);
                    if let Some(axis) = Unit::try_new(axis_vec, 1e-10) {
                        let rot = Rotation3::from_axis_angle(&axis, v[3].to_radians());
                        m *= rot.to_homogeneous();
                        any = true;
                    }
                }
            }
            schema::NodeElementTypeContent::Scale(s) => {
                let v = &s.content.0;
                if v.len() >= 3 {
                    m *= Matrix4::new_nonuniform_scaling(&Vector3::new(v[0], v[1], v[2]));
                    any = true;
                }
            }
            _ => {}
        }
    }
    any.then_some(m)
}

/// Compute up-axis correction matrix to convert to Y-up.
fn up_axis_correction(collada: &schema::Collada) -> Option<Matrix4<f64>> {
    let asset = collada.asset.as_ref()?;
    let axis = asset.up_axis.as_ref()?;
    match axis {
        schema::UpAxisType::ZUp => {
            // Rotate -90° around X: (x, y, z) → (x, z, -y)
            Some(Matrix4::new(
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0,
            ))
        }
        schema::UpAxisType::XUp => {
            // Rotate 90° around Z: (x, y, z) → (-y, x, z)
            Some(Matrix4::new(
                0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                1.0,
            ))
        }
        schema::UpAxisType::YUp => None,
    }
}

// ── Import ──────────────────────────────────────────────────────────────────

/// Convert a parsed Collada document to an rmesh Scene.
pub fn to_scene(collada: &schema::Collada, resolver: Option<&dyn Resolver>) -> Result<Scene> {
    let mut scene = Scene::new();

    // Build lookup tables
    let images = build_image_map(collada);
    let materials_by_id = build_material_map(collada);
    let material_bindings = collect_material_bindings(collada);

    // Load geometries → geometry_id → scene geometry index
    let mut geom_id_to_index: HashMap<String, usize> = HashMap::new();
    if let Some(lib_geom) = &collada.library_geometries {
        for geom_elem in &lib_geom.geometry {
            let geom_id = geom_elem.id.as_deref().unwrap_or("geometry");
            let geom_name = geom_elem.name.as_deref().unwrap_or(geom_id);

            let mesh = match geom_elem.mesh() {
                Some(m) => m,
                None => continue,
            };

            let trimesh = load_mesh(
                mesh,
                &materials_by_id,
                &material_bindings,
                &images,
                resolver,
            )?;
            let (_, idx) = scene.add_geometry(geom_name, Geometry::Mesh(Box::new(trimesh)));
            geom_id_to_index.insert(geom_id.to_string(), idx);
        }
    }

    // Build scene graph from visual scene
    let vs = resolve_visual_scene(collada);
    if let Some(vs) = vs {
        let root = SceneNode {
            name: vs.name.as_deref().unwrap_or("Scene").to_string(),
            children: Vec::new(),
            transform: up_axis_correction(collada),
            kind: SceneNodeKind::Geometry,
            index: vec![],
        };
        let root_idx = scene.graph.add_node(root);
        scene.graph.root = root_idx;

        for node in &vs.node {
            let child_idx = import_node(node, &geom_id_to_index, &mut scene)?;
            scene.graph.nodes[root_idx].children.push(child_idx);
        }
    } else if !scene.geometry.is_empty() {
        // No visual scene — create a flat graph with all geometries
        let root = SceneNode {
            name: "Scene".to_string(),
            children: Vec::new(),
            transform: up_axis_correction(collada),
            kind: SceneNodeKind::Geometry,
            index: vec![],
        };
        let root_idx = scene.graph.add_node(root);
        scene.graph.root = root_idx;

        for (name, geom_idx) in &geom_id_to_index {
            let child = SceneNode {
                name: name.clone(),
                children: Vec::new(),
                transform: None,
                kind: SceneNodeKind::Geometry,
                index: vec![*geom_idx],
            };
            let ci = scene.graph.add_node(child);
            scene.graph.nodes[root_idx].children.push(ci);
        }
    }

    Ok(scene)
}

/// Build image lookup: image_id → &ImageElementType
fn build_image_map(collada: &schema::Collada) -> HashMap<&str, &schema::ImageElementType> {
    let mut map = HashMap::new();
    if let Some(lib) = &collada.library_images {
        for img in &lib.image {
            if let Some(ref id) = img.id {
                map.insert(id.as_str(), img);
            }
        }
    }
    map
}

/// Build material lookup: material_id → &MaterialElementType
fn build_material_map(collada: &schema::Collada) -> HashMap<&str, &schema::MaterialElementType> {
    let mut map = HashMap::new();
    if let Some(lib) = &collada.library_materials {
        for mat in &lib.material {
            if let Some(ref id) = mat.id {
                map.insert(id.as_str(), mat);
            }
        }
    }
    map
}

/// Scan visual scene for material bindings: symbol → material_id.
fn collect_material_bindings(collada: &schema::Collada) -> HashMap<String, String> {
    let mut bindings = HashMap::new();
    let Some(lib_vs) = &collada.library_visual_scenes else {
        return bindings;
    };
    for vs in &lib_vs.visual_scene {
        for node in &vs.node {
            collect_bindings_from_node(node, &mut bindings);
        }
    }
    bindings
}

fn collect_bindings_from_node(
    node: &schema::NodeElementType,
    bindings: &mut HashMap<String, String>,
) {
    for ig in node.instance_geometries() {
        if let Some(ref bm) = ig.bind_material {
            for im in &bm.technique_common.instance_material {
                bindings
                    .entry(im.symbol.clone())
                    .or_insert_with(|| strip_fragment(&im.target).to_string());
            }
        }
    }
    for ic in node.instance_controllers() {
        if let Some(ref bm) = ic.bind_material {
            for im in &bm.technique_common.instance_material {
                bindings
                    .entry(im.symbol.clone())
                    .or_insert_with(|| strip_fragment(&im.target).to_string());
            }
        }
    }
    for child in node.child_nodes() {
        collect_bindings_from_node(child, bindings);
    }
}

/// Resolve the visual scene referenced by the <scene> element.
fn resolve_visual_scene<'a>(
    collada: &'a schema::Collada,
) -> Option<&'a schema::VisualSceneElementType> {
    let scene = collada.scene.as_ref()?;
    let ivs = scene.instance_visual_scene.as_ref()?;
    let url = ivs.url.as_deref()?;
    let vs_id = strip_fragment(url);

    let lib = collada.library_visual_scenes.as_ref()?;
    lib.visual_scene
        .iter()
        .find(|vs| vs.id.as_deref() == Some(vs_id))
        .or_else(|| lib.visual_scene.first())
}

/// Recursively process a Collada node into a SceneNode.
fn import_node(
    node: &schema::NodeElementType,
    geom_map: &HashMap<String, usize>,
    scene: &mut Scene,
) -> Result<usize> {
    let name = node
        .name
        .as_deref()
        .or(node.id.as_deref())
        .unwrap_or("node")
        .to_string();

    let transform = node_transform(node);

    // Collect geometry references from instance_geometry and instance_controller
    let mut geom_indices = Vec::new();
    for ig in node.instance_geometries() {
        let geom_id = strip_fragment(&ig.url);
        if let Some(&idx) = geom_map.get(geom_id) {
            geom_indices.push(idx);
        }
    }
    for ic in node.instance_controllers() {
        let geom_id = strip_fragment(&ic.url);
        if let Some(&idx) = geom_map.get(geom_id) {
            geom_indices.push(idx);
        }
    }

    let scene_node = SceneNode {
        name,
        children: Vec::new(),
        transform,
        kind: SceneNodeKind::Geometry,
        index: geom_indices,
    };
    let node_idx = scene.graph.add_node(scene_node);

    // Process child nodes
    for child in node.child_nodes() {
        let child_idx = import_node(child, geom_map, scene)?;
        scene.graph.nodes[node_idx].children.push(child_idx);
    }

    Ok(node_idx)
}

/// Intermediate result from processing a primitive group.
struct UnmergedPrimitive {
    vertices: Vec<Point3<f64>>,
    faces: Vec<[usize; 3]>,
    normals: Vec<Vector3<f64>>,
    uv: Vec<Vector2<f64>>,
}

/// Load a complete mesh from all its primitive groups.
fn load_mesh(
    mesh: &schema::MeshElementType,
    materials_by_id: &HashMap<&str, &schema::MaterialElementType>,
    material_bindings: &HashMap<String, String>,
    images: &HashMap<&str, &schema::ImageElementType>,
    resolver: Option<&dyn Resolver>,
) -> Result<Trimesh> {
    // Build source map: id → source element
    let source_map: HashMap<&str, &schema::SourceElementType> =
        mesh.sources().map(|s| (s.id.as_str(), s)).collect();

    // Get vertices element and find POSITION source
    let vertices = mesh
        .vertices()
        .context("mesh has no <vertices> element")?;
    let pos_source_id = vertices
        .input
        .iter()
        .find(|i| i.semantic == "POSITION")
        .map(|i| strip_fragment(&i.source))
        .context("vertices has no POSITION input")?;

    let mut parts: Vec<(UnmergedPrimitive, Option<String>)> = Vec::new();

    // Process <triangles>
    for tri in mesh.triangles() {
        let face_sizes: Vec<usize> = vec![3; tri.count as usize];
        let p = tri.p.as_ref().map(|p| &p.0[..]).unwrap_or(&[]);
        let prim = load_primitive(&tri.input, p, &face_sizes, &vertices.id, &source_map, pos_source_id)?;
        parts.push((prim, tri.material.clone()));
    }

    // Process <polylist>
    for poly in mesh.polylist() {
        let face_sizes: Vec<usize> = poly
            .vcount
            .as_ref()
            .map(|v| v.0.iter().map(|&n| n as usize).collect())
            .unwrap_or_default();
        let p = poly.p.as_ref().map(|p| &p.0[..]).unwrap_or(&[]);
        let prim = load_primitive(&poly.input, p, &face_sizes, &vertices.id, &source_map, pos_source_id)?;
        parts.push((prim, poly.material.clone()));
    }

    // Process <polygons>
    for polys in mesh.polygons() {
        let inputs: Vec<&schema::InputLocalOffsetType> = polys.inputs().collect();
        let stride = inputs
            .iter()
            .map(|i| i.offset as usize)
            .max()
            .unwrap_or(0)
            + 1;

        // Each <p> is one polygon
        let mut all_p: Vec<u64> = Vec::new();
        let mut face_sizes: Vec<usize> = Vec::new();
        for p_elem in polys.ps() {
            let n_verts = p_elem.0.len() / stride;
            face_sizes.push(n_verts);
            all_p.extend_from_slice(&p_elem.0);
        }

        if !face_sizes.is_empty() {
            let prim = load_primitive(&inputs, &all_p, &face_sizes, &vertices.id, &source_map, pos_source_id)?;
            parts.push((prim, polys.material.clone()));
        }
    }

    // Combine all primitive parts into a single mesh
    let mut all_verts: Vec<Point3<f64>> = Vec::new();
    let mut all_faces: Vec<[usize; 3]> = Vec::new();
    let mut all_normals: Vec<Vector3<f64>> = Vec::new();
    let mut all_uv: Vec<Vector2<f64>> = Vec::new();
    let mut face_mat_indices: Vec<usize> = Vec::new();
    let mut mat_names: Vec<String> = Vec::new();
    let mut mat_name_map: HashMap<String, usize> = HashMap::new();
    let mut has_any_normals = false;
    let mut has_any_uv = false;

    for (prim, _mat_symbol) in &parts {
        if !prim.normals.is_empty() {
            has_any_normals = true;
        }
        if !prim.uv.is_empty() {
            has_any_uv = true;
        }
    }

    for (prim, mat_symbol) in parts {
        let offset = all_verts.len();
        all_verts.extend(prim.vertices);

        // Ensure consistent normal/UV arrays (pad with zeros if a group is missing them)
        if has_any_normals {
            if prim.normals.is_empty() {
                all_normals.extend(std::iter::repeat_n(
                    Vector3::zeros(),
                    all_verts.len() - offset,
                ));
            } else {
                all_normals.extend(prim.normals);
            }
        }
        if has_any_uv {
            if prim.uv.is_empty() {
                all_uv.extend(std::iter::repeat_n(
                    Vector2::zeros(),
                    all_verts.len() - offset,
                ));
            } else {
                all_uv.extend(prim.uv);
            }
        }

        let mat_idx = if let Some(sym) = mat_symbol {
            *mat_name_map.entry(sym.clone()).or_insert_with(|| {
                let idx = mat_names.len();
                mat_names.push(sym);
                idx
            })
        } else {
            UNSET
        };

        for face in prim.faces {
            all_faces.push([face[0] + offset, face[1] + offset, face[2] + offset]);
            face_mat_indices.push(mat_idx);
        }
    }

    // Build attributes
    let mut attrs_vertex = Attributes::default();
    if has_any_normals && !all_normals.is_empty() {
        attrs_vertex.normals.push(all_normals);
    }
    if has_any_uv && !all_uv.is_empty() {
        attrs_vertex.uv.push(all_uv);
    }

    let mut attrs_face = Attributes::default();
    if !mat_names.is_empty() {
        attrs_face.groupings.push(Grouping {
            kind: GroupingKind::Material,
            names: mat_names.clone(),
            indices: face_mat_indices,
        });
    }

    let attrs_v = if attrs_vertex.normals.is_empty() && attrs_vertex.uv.is_empty() {
        None
    } else {
        Some(attrs_vertex)
    };
    let attrs_f = if attrs_face.groupings.is_empty() {
        None
    } else {
        Some(attrs_face)
    };

    let mut trimesh = Trimesh::new(all_verts, all_faces, attrs_v, attrs_f)?;

    // Create material objects from the symbols resolved through bindings
    for name in &mat_names {
        // Try to resolve through binding → material definition
        let mat_name = material_bindings
            .get(name.as_str())
            .and_then(|mat_id| materials_by_id.get(mat_id.as_str()))
            .and_then(|mat| mat.name.as_deref().or(mat.id.as_deref()))
            .unwrap_or(name.as_str());

        let mut simple = SimpleMaterial {
            name: mat_name.to_string(),
            ..Default::default()
        };

        // Try to load diffuse texture if we can resolve the image
        if let Some(resolver) = resolver {
            if let Some(texture) = try_load_texture(name, material_bindings, materials_by_id, images, resolver) {
                simple.diffuse_texture = Some(texture);
            }
        }

        trimesh
            .materials
            .push(Material::Simple(simple));
    }

    trimesh.source.format = Some(FileType::DAE);
    Ok(trimesh)
}

/// Try to load a texture for a material by following the binding chain.
fn try_load_texture(
    symbol: &str,
    bindings: &HashMap<String, String>,
    materials: &HashMap<&str, &schema::MaterialElementType>,
    images: &HashMap<&str, &schema::ImageElementType>,
    resolver: &dyn Resolver,
) -> Option<LazyImage> {
    // symbol → material_id → material → instance_effect → effect_id
    // We don't have effects parsed, but we can try to find an image with a similar name
    let mat_id = bindings.get(symbol)?;
    let _mat = materials.get(mat_id.as_str())?;

    // Heuristic: try to find an image whose id contains the material id
    for (img_id, img) in images {
        if img_id.contains(mat_id.as_str()) || mat_id.contains(*img_id) {
            if let Some(init_from) = img.init_from() {
                if let Ok(data) = resolver.resolve(init_from) {
                    return Some(LazyImage::new(data));
                }
            }
        }
    }

    // Try all images with init_from
    for img in images.values() {
        if let Some(init_from) = img.init_from() {
            if let Ok(data) = resolver.resolve(init_from) {
                return Some(LazyImage::new(data));
            }
        }
    }

    None
}

/// Process a single primitive group (triangles, polylist, or polygon batch).
fn load_primitive(
    inputs: &[impl AsInputLocalOffset],
    p_data: &[u64],
    face_sizes: &[usize],
    vertices_id: &str,
    source_map: &HashMap<&str, &schema::SourceElementType>,
    pos_source_id: &str,
) -> Result<UnmergedPrimitive> {
    // Determine stride (tuple width)
    let stride = inputs
        .iter()
        .map(|i| i.offset() as usize)
        .max()
        .unwrap_or(0)
        + 1;

    // Find source data for each semantic
    let pos_source = source_map
        .get(pos_source_id)
        .and_then(|s| read_source(s))
        .context("cannot read position source")?;

    let mut pos_offset = None;
    let mut normal_offset = None;
    let mut normal_source: Option<SourceData> = None;
    let mut uv_offset = None;
    let mut uv_source: Option<SourceData> = None;

    for input in inputs {
        let source_id = strip_fragment(input.source());
        match input.semantic() {
            "VERTEX" => {
                if source_id == vertices_id {
                    pos_offset = Some(input.offset() as usize);
                }
            }
            "NORMAL" => {
                normal_offset = Some(input.offset() as usize);
                if let Some(src) = source_map.get(source_id) {
                    normal_source = read_source(src);
                }
            }
            "TEXCOORD" => {
                if uv_offset.is_none() {
                    // Take first UV set
                    uv_offset = Some(input.offset() as usize);
                    if let Some(src) = source_map.get(source_id) {
                        uv_source = read_source(src);
                    }
                }
            }
            _ => {}
        }
    }

    let pos_offset = pos_offset.context("no VERTEX input found in primitive")?;

    // Validate p array length
    let total_verts: usize = face_sizes.iter().sum();
    let needed = total_verts * stride;
    if needed > p_data.len() {
        bail!(
            "p array too short: expected {needed} elements ({total_verts} verts × {stride} stride), got {}",
            p_data.len()
        );
    }

    let has_normals = normal_offset.is_some() && normal_source.is_some();
    let has_uv = uv_offset.is_some() && uv_source.is_some();

    // Unmerge: dedup unique (position, normal, uv) vertex combos
    let mut new_vertices: Vec<Point3<f64>> = Vec::new();
    let mut new_normals: Vec<Vector3<f64>> = Vec::new();
    let mut new_uv: Vec<Vector2<f64>> = Vec::new();
    let mut key_map: HashMap<(usize, Option<usize>, Option<usize>), usize> = HashMap::new();
    let mut new_faces: Vec<[usize; 3]> = Vec::new();

    let mut p_idx = 0;
    for &face_size in face_sizes {
        let mut face_indices = Vec::with_capacity(face_size);

        for _ in 0..face_size {
            let pi = p_data[p_idx + pos_offset] as usize;
            let ni = if has_normals {
                Some(p_data[p_idx + normal_offset.unwrap()] as usize)
            } else {
                None
            };
            let ti = if has_uv {
                Some(p_data[p_idx + uv_offset.unwrap()] as usize)
            } else {
                None
            };

            let key = (pi, ni, ti);
            let new_idx = *key_map.entry(key).or_insert_with(|| {
                let idx = new_vertices.len();

                // Read position
                if let Some(pos) = pos_source.get(pi) {
                    if pos.len() >= 3 {
                        new_vertices.push(Point3::new(pos[0], pos[1], pos[2]));
                    } else {
                        new_vertices.push(Point3::origin());
                    }
                } else {
                    new_vertices.push(Point3::origin());
                }

                // Read normal
                if let Some(ref src) = normal_source {
                    if let Some(n) = ni.and_then(|i| src.get(i)) {
                        if n.len() >= 3 {
                            new_normals.push(Vector3::new(n[0], n[1], n[2]));
                        } else {
                            new_normals.push(Vector3::zeros());
                        }
                    } else {
                        new_normals.push(Vector3::zeros());
                    }
                }

                // Read UV
                if let Some(ref src) = uv_source {
                    if let Some(uv) = ti.and_then(|i| src.get(i)) {
                        if uv.len() >= 2 {
                            new_uv.push(Vector2::new(uv[0], uv[1]));
                        } else {
                            new_uv.push(Vector2::zeros());
                        }
                    } else {
                        new_uv.push(Vector2::zeros());
                    }
                }

                idx
            });

            face_indices.push(new_idx);
            p_idx += stride;
        }

        // Fan-triangulate the face
        if face_size >= 3 {
            for i in 1..face_size - 1 {
                new_faces.push([face_indices[0], face_indices[i], face_indices[i + 1]]);
            }
        }
    }

    Ok(UnmergedPrimitive {
        vertices: new_vertices,
        faces: new_faces,
        normals: new_normals,
        uv: new_uv,
    })
}

/// Trait to abstract over InputLocalOffsetType and &InputLocalOffsetType.
trait AsInputLocalOffset {
    fn offset(&self) -> u64;
    fn semantic(&self) -> &str;
    fn source(&self) -> &str;
}

impl AsInputLocalOffset for schema::InputLocalOffsetType {
    fn offset(&self) -> u64 {
        self.offset
    }
    fn semantic(&self) -> &str {
        &self.semantic
    }
    fn source(&self) -> &str {
        &self.source
    }
}

impl AsInputLocalOffset for &schema::InputLocalOffsetType {
    fn offset(&self) -> u64 {
        self.offset
    }
    fn semantic(&self) -> &str {
        &self.semantic
    }
    fn source(&self) -> &str {
        &self.source
    }
}

// ── Export ───────────────────────────────────────────────────────────────────

/// Convert an rmesh Scene to Collada XML bytes.
pub fn from_scene(scene: &Scene) -> Result<Vec<u8>> {
    let mut geometries = Vec::new();
    let mut geom_id_map: HashMap<usize, String> = HashMap::new();

    for (idx, (name, geom)) in scene.geometry.iter().enumerate() {
        if let Geometry::Mesh(trimesh) = geom {
            let geom_id = format!("{name}-mesh");
            let geom_elem = build_geometry(&geom_id, name, trimesh);
            geometries.push(geom_elem);
            geom_id_map.insert(idx, geom_id);
        }
    }

    // Build visual scene from the scene graph
    let vs = build_visual_scene(scene, &geom_id_map);

    let collada = schema::Collada {
        version: Some("1.4.1".to_string()),
        asset: Some(schema::AssetElementType {
            contributor: vec![],
            created: None,
            keywords: None,
            modified: None,
            revision: None,
            subject: None,
            title: None,
            unit: Some(schema::AssetUnitElementType {
                meter: 1.0,
                name: "meter".to_string(),
            }),
            up_axis: Some(schema::UpAxisType::YUp),
        }),
        library_geometries: if geometries.is_empty() {
            None
        } else {
            Some(schema::LibraryGeometries {
                geometry: geometries,
            })
        },
        library_visual_scenes: Some(schema::LibraryVisualScenes {
            visual_scene: vec![vs],
        }),
        library_materials: None,
        library_images: None,
        library_effects: None,
        scene: Some(schema::Scene {
            instance_visual_scene: Some(schema::InstanceVisualScene {
                url: Some("#Scene".to_string()),
                sid: None,
                name: None,
            }),
        }),
    };

    let xml = quick_xml::se::to_string(&collada)
        .context("failed to serialize Collada to XML")?;
    let output = format!("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n{xml}");
    Ok(output.into_bytes())
}

/// Build a Collada geometry element from a Trimesh.
fn build_geometry(
    geom_id: &str,
    name: &str,
    trimesh: &Trimesh,
) -> schema::GeometryElementType {
    let pos_source_id = format!("{geom_id}-positions");
    let vertices_id = format!("{geom_id}-vertices");

    // Build position source
    let pos_data: Vec<f64> = trimesh
        .vertices
        .iter()
        .flat_map(|v| [v.x, v.y, v.z])
        .collect();
    let pos_source = build_source(&pos_source_id, &pos_data, 3, &["X", "Y", "Z"]);

    let mut content = vec![schema::MeshElementTypeContent::Source(pos_source)];
    let mut tri_inputs = Vec::new();
    let offset = 0u64;

    // Check for normals
    let has_normals = !trimesh.attributes_vertex.normals.is_empty()
        && !trimesh.attributes_vertex.normals[0].is_empty();
    if has_normals {
        let norm_source_id = format!("{geom_id}-normals");
        let norm_data: Vec<f64> = trimesh.attributes_vertex.normals[0]
            .iter()
            .flat_map(|n| [n.x, n.y, n.z])
            .collect();
        let norm_source = build_source(&norm_source_id, &norm_data, 3, &["X", "Y", "Z"]);
        content.push(schema::MeshElementTypeContent::Source(norm_source));

        tri_inputs.push(schema::InputLocalOffsetType {
            offset: offset + 1,
            semantic: "NORMAL".to_string(),
            source: fragment_ref(&norm_source_id),
            set: None,
        });
    }

    // Check for UVs
    let has_uv = !trimesh.attributes_vertex.uv.is_empty()
        && !trimesh.attributes_vertex.uv[0].is_empty();
    if has_uv {
        let uv_source_id = format!("{geom_id}-map-0");
        let uv_data: Vec<f64> = trimesh.attributes_vertex.uv[0]
            .iter()
            .flat_map(|uv| [uv.x, uv.y])
            .collect();
        let uv_source = build_source(&uv_source_id, &uv_data, 2, &["S", "T"]);
        content.push(schema::MeshElementTypeContent::Source(uv_source));

        let norm_offset = if has_normals { 1 } else { 0 };
        tri_inputs.push(schema::InputLocalOffsetType {
            offset: offset + 1 + norm_offset,
            semantic: "TEXCOORD".to_string(),
            source: fragment_ref(&uv_source_id),
            set: Some(0),
        });
    }

    // Vertices element
    let vertices = schema::VerticesElementType {
        id: vertices_id.clone(),
        name: None,
        input: vec![schema::InputLocalType {
            semantic: "POSITION".to_string(),
            source: fragment_ref(&pos_source_id),
        }],
        extra: vec![],
    };
    content.push(schema::MeshElementTypeContent::Vertices(vertices));

    // VERTEX input is always at offset 0
    let mut all_inputs = vec![schema::InputLocalOffsetType {
        offset: 0,
        semantic: "VERTEX".to_string(),
        source: fragment_ref(&vertices_id),
        set: None,
    }];
    all_inputs.extend(tri_inputs);

    let stride = all_inputs.len();

    // Build interleaved <p> array
    let mut p_data = Vec::with_capacity(trimesh.faces.len() * 3 * stride);
    for face in &trimesh.faces {
        for &vi in face {
            // All attributes are vertex-aligned, so they share the same index
            for _ in 0..stride {
                p_data.push(vi as u64);
            }
        }
    }

    let triangles = schema::TrianglesElementType {
        name: None,
        count: trimesh.faces.len() as u64,
        material: None,
        input: all_inputs,
        p: Some(schema::ListOfUIntsType(p_data)),
        extra: vec![],
    };
    content.push(schema::MeshElementTypeContent::Triangles(triangles));

    schema::GeometryElementType {
        id: Some(geom_id.to_string()),
        name: Some(name.to_string()),
        content: vec![schema::GeometryElementTypeContent::Mesh(
            schema::MeshElementType { content },
        )],
    }
}

/// Build a Collada source element from flat float data.
fn build_source(
    id: &str,
    data: &[f64],
    stride: usize,
    params: &[&str],
) -> schema::SourceElementType {
    let count = data.len() / stride;
    let array_id = format!("{id}-array");

    let float_array = schema::FloatArrayElementType {
        id: Some(array_id.clone()),
        name: None,
        count: data.len() as u64,
        digits: schema::FloatArrayElementType::default_digits(),
        magnitude: schema::FloatArrayElementType::default_magnitude(),
        content: schema::ListOfFloatsType(data.to_vec()),
    };

    let accessor = schema::AccessorElementType {
        count: count as u64,
        offset: 0,
        source: Some(fragment_ref(&array_id)),
        stride: stride as u64,
        param: params
            .iter()
            .map(|&name| schema::ParamElementType {
                name: Some(name.to_string()),
                sid: None,
                semantic: None,
                type_: "float".to_string(),
                content: String::new(),
            })
            .collect(),
    };

    let technique_common = schema::SourceTechniqueCommonElementType { accessor };

    schema::SourceElementType {
        id: id.to_string(),
        name: None,
        content: vec![
            schema::SourceElementTypeContent::FloatArray(float_array),
            schema::SourceElementTypeContent::TechniqueCommon(technique_common),
        ],
    }
}

/// Build the visual scene element from the scene graph.
fn build_visual_scene(
    scene: &Scene,
    geom_id_map: &HashMap<usize, String>,
) -> schema::VisualSceneElementType {
    let mut nodes = Vec::new();

    if !scene.graph.nodes.is_empty() {
        let root = &scene.graph.nodes[scene.graph.root];
        // Export root's children as top-level nodes
        for &child_idx in &root.children {
            nodes.push(build_export_node(&scene.graph, child_idx, geom_id_map));
        }
        // If root itself has geometry, export it too
        if !root.index.is_empty() {
            nodes.push(build_export_node(
                &scene.graph,
                scene.graph.root,
                geom_id_map,
            ));
        }
    }

    schema::VisualSceneElementType {
        id: Some("Scene".to_string()),
        name: Some("Scene".to_string()),
        asset: None,
        node: nodes,
        evaluate_scene: vec![],
        extra: vec![],
    }
}

/// Recursively build a Collada node from a SceneNode.
fn build_export_node(
    graph: &crate::scene::SceneGraph,
    node_idx: usize,
    geom_id_map: &HashMap<usize, String>,
) -> schema::NodeElementType {
    let node = &graph.nodes[node_idx];
    let mut content: Vec<schema::NodeElementTypeContent> = Vec::new();

    // Transform → matrix
    if let Some(ref matrix) = node.transform {
        // Collada uses row-major: emit rows left-to-right, top-to-bottom
        let mut row_major = Vec::with_capacity(16);
        for r in 0..4 {
            for c in 0..4 {
                row_major.push(matrix[(r, c)]);
            }
        }
        content.push(schema::NodeElementTypeContent::Matrix(
            schema::MatrixElementType {
                sid: Some("transform".to_string()),
                content: schema::Float4X4Type(row_major),
            },
        ));
    }

    // Instance geometry
    for &geom_idx in &node.index {
        if let Some(geom_id) = geom_id_map.get(&geom_idx) {
            content.push(schema::NodeElementTypeContent::InstanceGeometry(
                schema::InstanceGeometryElementType {
                    url: fragment_ref(geom_id),
                    sid: None,
                    name: None,
                    bind_material: None,
                    extra: vec![],
                },
            ));
        }
    }

    // Child nodes
    for &child_idx in &node.children {
        content.push(schema::NodeElementTypeContent::Node(build_export_node(
            graph,
            child_idx,
            geom_id_map,
        )));
    }

    schema::NodeElementType {
        id: Some(node.name.clone()),
        name: Some(node.name.clone()),
        sid: None,
        type_: schema::NodeType::Node,
        layer: None,
        content,
    }
}
