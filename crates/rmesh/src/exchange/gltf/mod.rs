//! GLTF/GLB loader for importing 3D scenes.

pub mod convert;
pub mod extensions;
pub mod schema;

use anyhow::{Context, Result, bail};
use nalgebra::{Matrix4, Point3, Quaternion, UnitQuaternion, Vector3, Vector4};
use rayon::prelude::*;

use crate::attributes::{AlphaMode, Grouping, GroupingKind, Material, PBRMaterial, UNSET};
use crate::boundary::Surface;
use crate::geometry::Geometry;
use crate::image::LazyImage;
use crate::mesh::Trimesh;
use crate::resolvers::Resolver;
use crate::scene::{
    Animation, AnimationChannel, AnimationPath, AnimationSampler, Camera, CameraProjection,
    Interpolation, Light, LightType, Scene, SceneGraph, SceneNode, SceneNodeKind,
};
use self::schema::{
    self as gltf_2, AccessorType, CameraType, GltfAlphaMode, GltfAnimationPath,
    GltfInterpolation, GltfLightType, COMPONENT_U8, COMPONENT_U16, COMPONENT_U32,
    GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP, GL_TRIANGLES, GlTf, GltfIndex, KhrLightsPunctual,
};

use self::extensions::ExtensionRegistry;

// GLB magic numbers
const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const GLB_JSON: u32 = 0x4E4F_534A; // "JSON"
const GLB_BIN: u32 = 0x004E_4942; // "BIN\0"

/// Resolved accessor data for reading buffer content.
struct AccessorReader<'a> {
    buffer: &'a [u8],
    start: usize,
    stride: usize,
    count: usize,
}

/// GLTF loader that holds parsed header and buffer data.
pub struct GltfLoader {
    header: GlTf,
    buffers: Vec<Vec<u8>>,
    extensions: ExtensionRegistry,
}

impl GltfLoader {
    /// Load from GLB binary data.
    pub fn from_glb(data: &[u8]) -> Result<Self> {
        if data.len() < 12 {
            bail!("GLB data too short for header");
        }

        // Read 12-byte header
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let _length = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

        if magic != GLB_MAGIC {
            bail!("Invalid GLB magic number");
        }
        if version != 2 {
            bail!("Unsupported GLB version: {}", version);
        }

        let mut offset = 12;
        let mut json_data: Option<&[u8]> = None;
        let mut bin_data: Option<&[u8]> = None;

        // Read chunks
        while offset + 8 <= data.len() {
            let chunk_length = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            let chunk_type = u32::from_le_bytes([
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            offset += 8;

            if offset + chunk_length > data.len() {
                bail!("GLB chunk extends past end of data");
            }

            let chunk_data = &data[offset..offset + chunk_length];
            offset += chunk_length;

            match chunk_type {
                GLB_JSON => json_data = Some(chunk_data),
                GLB_BIN => bin_data = Some(chunk_data),
                _ => {} // Ignore unknown chunks
            }
        }

        let json_data = json_data.context("GLB missing JSON chunk")?;
        let header: GlTf =
            serde_json::from_slice(json_data).context("Failed to parse GLTF JSON")?;

        let mut buffers = Vec::new();
        if let Some(bin) = bin_data {
            buffers.push(bin.to_vec());
        }

        Ok(Self {
            header,
            buffers,
            extensions: ExtensionRegistry::with_builtins(),
        })
    }

    /// Load from GLTF JSON with optional resolver for external buffers.
    pub fn from_gltf(json: &[u8], resolver: Option<&dyn Resolver>) -> Result<Self> {
        let header: GlTf = serde_json::from_slice(json).context("Failed to parse GLTF JSON")?;

        let mut buffers = Vec::new();
        for buffer in header.buffers.as_slice() {
            if let Some(uri) = &buffer.uri {
                if uri.starts_with("data:") {
                    // Data URI
                    let data = Self::decode_data_uri(uri)?;
                    buffers.push(data);
                } else if let Some(resolver) = resolver {
                    // External file
                    let data = resolver.resolve(uri)?;
                    buffers.push(data);
                } else {
                    bail!("External buffer '{}' but no resolver provided", uri);
                }
            } else {
                // Buffer without URI - should be in GLB binary chunk
                buffers.push(Vec::new());
            }
        }

        Ok(Self {
            header,
            buffers,
            extensions: ExtensionRegistry::with_builtins(),
        })
    }

    /// Decode a base64 data URI.
    fn decode_data_uri(uri: &str) -> Result<Vec<u8>> {
        // Format: data:[<mediatype>][;base64],<data>
        let parts: Vec<&str> = uri.splitn(2, ',').collect();
        if parts.len() != 2 {
            bail!("Invalid data URI format");
        }

        if parts[0].contains(";base64") {
            use base64::Engine;
            let data = base64::engine::general_purpose::STANDARD
                .decode(parts[1])
                .context("Failed to decode base64 data URI")?;
            Ok(data)
        } else {
            // URL-encoded data
            Ok(parts[1].as_bytes().to_vec())
        }
    }

    /// Get raw bytes for a buffer view.
    fn get_buffer_view_data(&self, buffer_view_index: usize) -> Option<&[u8]> {
        let buffer_view = self.header.buffer_views.get(buffer_view_index)?;
        let buffer = self.buffers.get(buffer_view.buffer as usize)?;

        let start = buffer_view.byte_offset as usize;
        let length = buffer_view.byte_length as usize;
        let end = start + length;

        if end > buffer.len() {
            return None;
        }

        Some(&buffer[start..end])
    }

    /// Load an image by index, returning a LazyImage containing raw bytes.
    fn load_image(&self, image_index: usize) -> Option<LazyImage> {
        let image = self.header.images.get(image_index)?;

        // Case 1: Image in buffer view (embedded in GLB)
        if let Some(buffer_view_index) = image.buffer_view {
            let bytes = self.get_buffer_view_data(buffer_view_index as usize)?;
            return Some(LazyImage::new(bytes.to_vec()));
        }

        // Case 2: Data URI (base64 embedded)
        if let Some(uri) = &image.uri
            && uri.starts_with("data:")
            && let Ok(bytes) = Self::decode_data_uri(uri)
        {
            return Some(LazyImage::new(bytes));
        }
        // Case 3: External file - would need resolver, skip for now
        // External images require file system access which we don't have here

        None
    }

    /// Load a texture by index, resolving through the texture -> image indirection.
    fn load_texture(&self, texture_index: usize) -> Option<LazyImage> {
        let texture = self.header.textures.get(texture_index)?;

        // Get the image source index
        let image_index = texture.source? as usize;
        self.load_image(image_index)
    }

    /// Convert the loaded GLTF to a Scene.
    pub fn to_scene(&self) -> Result<Scene> {
        let mut scene = Scene::new();

        // Load materials first so we can reference them from primitives
        let materials = self.load_materials();

        // Load each GLTF mesh (combining all its primitives into one Trimesh)
        let gltf_meshes = self.header.meshes.as_slice();
        let meshes_with_names: Vec<_> = gltf_meshes
            .par_iter()
            .enumerate()
            .filter_map(|(mesh_idx, mesh)| {
                let mesh_name = mesh
                    .name
                    .as_deref()
                    .map_or_else(|| format!("mesh_{}", mesh_idx), |s| s.to_string());
                self.load_mesh(mesh, &materials)
                    .ok()
                    .map(|trimesh| (mesh_name, trimesh))
            })
            .collect();

        for (name, trimesh) in meshes_with_names {
            scene.add_geometry(&name, Geometry::Mesh(Box::new(trimesh)));
        }

        // Load cameras
        for gltf_cam in self.header.cameras.as_slice() {
            let camera = self.convert_camera(gltf_cam);
            scene.add_camera(camera);
        }

        // Load lights from KHR_lights_punctual extension
        if let Some(lights_ext) = self.header.extensions.get("KHR_lights_punctual")
            && let Ok(lights_data) = serde_json::from_value::<KhrLightsPunctual>(lights_ext.clone())
        {
            for gltf_light in lights_data.lights {
                let light = self.convert_light(&gltf_light);
                scene.add_light(light);
            }
        }

        // Build scene graph
        scene.graph = self.build_scene_graph();

        // Load animations
        for gltf_anim in self.header.animations.as_slice() {
            if let Ok(animation) = self.convert_animation(gltf_anim) {
                scene.add_animation(animation);
            }
        }

        // Set active scene (root is already set by build_scene_graph)
        // Just validate the scene index exists
        if let Some(scene_idx) = self.header.scene {
            let scenes = self.header.scenes.as_slice();
            if scene_idx as usize >= scenes.len() {
                // Invalid scene index, but we continue anyway
            }
        }

        Ok(scene)
    }

    /// Load a GLTF mesh, combining all its primitives into a single Trimesh.
    /// Each primitive becomes a face grouping, similar to OBJ objects.
    fn load_mesh(&self, mesh: &gltf_2::Mesh, materials: &[Material]) -> Result<Trimesh> {
        if mesh.primitives.is_empty() {
            bail!("Mesh has no primitives");
        }

        // Load all primitives
        let loaded_primitives: Vec<_> = mesh
            .primitives
            .iter()
            .enumerate()
            .filter_map(|(idx, prim)| match self.load_primitive(prim, materials) {
                Ok(m) => Some((idx, m)),
                Err(e) => {
                    #[cfg(debug_assertions)]
                    eprintln!("[gltf] failed to load primitive {idx}: {e:?}");
                    None
                }
            })
            .collect();

        if loaded_primitives.is_empty() {
            bail!("No primitives could be loaded");
        }

        // If only one primitive, return it directly
        if loaded_primitives.len() == 1 {
            return Ok(loaded_primitives.into_iter().next().unwrap().1);
        }

        // Combine multiple primitives into one mesh
        let mut combined_vertices: Vec<Point3<f64>> = Vec::new();
        let mut combined_faces: Vec<[usize; 3]> = Vec::new();
        let mut primitive_indices: Vec<usize> = Vec::new(); // Per-face primitive index
        let mut primitive_names: Vec<String> = Vec::new();

        // Track materials across primitives
        let mut material_indices: Vec<usize> = Vec::new();
        let mut material_names: Vec<String> = Vec::new();
        let mut combined_materials: Vec<Material> = Vec::new();

        // Track BREP surfaces across primitives
        let mut combined_face_surfaces: Vec<Surface> = Vec::new();
        let mut surface_indices: Vec<usize> = Vec::new();
        let mut surface_names: Vec<String> = Vec::new();

        for (prim_idx, prim_mesh) in loaded_primitives {
            let vertex_offset = combined_vertices.len();
            let num_faces = prim_mesh.faces.len();

            // Add vertices
            combined_vertices.extend(prim_mesh.vertices.iter().copied());

            // Add faces with offset vertex indices
            for face in &prim_mesh.faces {
                combined_faces.push([
                    face[0] + vertex_offset,
                    face[1] + vertex_offset,
                    face[2] + vertex_offset,
                ]);
            }

            // Track which primitive each face belongs to
            let prim_name = format!("primitive_{}", prim_idx);
            let prim_name_idx = primitive_names.len();
            primitive_names.push(prim_name);
            primitive_indices.extend(std::iter::repeat_n(prim_name_idx, num_faces));

            // Merge face_surfaces from this primitive
            if !prim_mesh.face_surfaces.is_empty() {
                combined_face_surfaces.extend(prim_mesh.face_surfaces.iter().cloned());
                // Find the surface grouping to get per-face indices
                if let Some(sg) = prim_mesh
                    .attributes_face
                    .groupings
                    .iter()
                    .find(|g| g.kind == GroupingKind::Surface)
                {
                    let name_offset = surface_names.len();
                    surface_names.extend(sg.names.iter().cloned());
                    surface_indices.extend(
                        sg.indices.iter().map(
                            |&i| {
                                if i == UNSET { UNSET } else { i + name_offset }
                            },
                        ),
                    );
                }
            } else if !surface_indices.is_empty() {
                // Primitive has no BREP data but earlier primitives did;
                // pad with UNSET so the index array stays aligned with faces.
                surface_indices.extend(std::iter::repeat_n(UNSET, num_faces));
            }

            // Handle materials from this primitive
            if prim_mesh.materials.is_empty() {
                // No material - use a placeholder index
                let no_mat_name = String::new();
                let mat_idx =
                    if let Some(idx) = material_names.iter().position(|n| n == &no_mat_name) {
                        idx
                    } else {
                        let idx = material_names.len();
                        material_names.push(no_mat_name);
                        idx
                    };
                material_indices.extend(std::iter::repeat_n(mat_idx, num_faces));
            } else {
                // Find or add the material
                let mat = &prim_mesh.materials[0];
                let mat_name = mat.name().to_string();
                let mat_idx = if let Some(idx) = material_names.iter().position(|n| n == &mat_name)
                {
                    idx
                } else {
                    let idx = material_names.len();
                    material_names.push(mat_name);
                    combined_materials.push(mat.clone());
                    idx
                };
                material_indices.extend(std::iter::repeat_n(mat_idx, num_faces));
            }
        }

        let mut result = Trimesh::new(combined_vertices, combined_faces, None, None)?;

        // Add primitive grouping
        result.attributes_face.groupings.push(Grouping {
            kind: GroupingKind::Group, // Use Group for primitives
            names: primitive_names,
            indices: primitive_indices,
        });

        // Add material grouping if we have materials
        if !material_names.is_empty() && material_names.iter().any(|n| !n.is_empty()) {
            result.attributes_face.groupings.push(Grouping {
                kind: GroupingKind::Material,
                names: material_names,
                indices: material_indices,
            });
        }

        result.materials = combined_materials;

        // Add combined face surfaces and grouping
        if !combined_face_surfaces.is_empty() {
            result.face_surfaces = combined_face_surfaces;
            result.attributes_face.groupings.push(Grouping {
                kind: GroupingKind::Surface,
                names: surface_names,
                indices: surface_indices,
            });
        }

        Ok(result)
    }

    /// Load a single mesh primitive.
    fn load_primitive(
        &self,
        primitive: &gltf_2::MeshPrimitive,
        materials: &[Material],
    ) -> Result<Trimesh> {
        // Only handle triangles for now
        let mode = primitive.mode;
        if mode != GL_TRIANGLES && mode != GL_TRIANGLE_STRIP && mode != GL_TRIANGLE_FAN {
            bail!("Unsupported primitive mode: {}", mode);
        }

        // Get position accessor
        let pos_accessor_idx = primitive
            .attributes
            .get("POSITION")
            .context("Primitive missing POSITION attribute")?;
        let positions = self.read_accessor_vec3(*pos_accessor_idx as usize)?;

        let vertices: Vec<Point3<f64>> = positions
            .into_iter()
            .map(|[x, y, z]| Point3::new(f64::from(x), f64::from(y), f64::from(z)))
            .collect();

        // Get indices
        let faces = if let Some(indices_idx) = primitive.indices {
            let indices = self.read_accessor_indices(indices_idx as usize)?;
            match mode {
                GL_TRIANGLES => indices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
                GL_TRIANGLE_STRIP => {
                    let mut faces = Vec::with_capacity(indices.len().saturating_sub(2));
                    for i in 0..indices.len().saturating_sub(2) {
                        if i % 2 == 0 {
                            faces.push([indices[i], indices[i + 1], indices[i + 2]]);
                        } else {
                            faces.push([indices[i], indices[i + 2], indices[i + 1]]);
                        }
                    }
                    faces
                }
                GL_TRIANGLE_FAN => {
                    let mut faces = Vec::with_capacity(indices.len().saturating_sub(2));
                    for i in 1..indices.len().saturating_sub(1) {
                        faces.push([indices[0], indices[i], indices[i + 1]]);
                    }
                    faces
                }
                _ => Vec::new(),
            }
        } else {
            // Non-indexed: generate sequential indices
            match mode {
                GL_TRIANGLES => (0..vertices.len())
                    .step_by(3)
                    .map(|i| [i, i + 1, i + 2])
                    .collect(),
                _ => bail!("Non-indexed primitives only supported for TRIANGULAR mode"),
            }
        };

        let mut mesh = Trimesh::new(vertices, faces, None, None)?;

        // Load normals if present
        if let Some(&normal_idx) = primitive.attributes.get("NORMAL")
            && let Ok(normals) = self.read_accessor_vec3(normal_idx as usize)
        {
            let normals: Vec<Vector3<f64>> = normals
                .into_iter()
                .map(|[x, y, z]| Vector3::new(f64::from(x), f64::from(y), f64::from(z)))
                .collect();
            mesh.attributes_vertex.normals.push(normals);
        }

        // Load all UV sets (TEXCOORD_0, TEXCOORD_1, etc.)
        const TEXCOORD_NAMES: [&str; 8] = [
            "TEXCOORD_0",
            "TEXCOORD_1",
            "TEXCOORD_2",
            "TEXCOORD_3",
            "TEXCOORD_4",
            "TEXCOORD_5",
            "TEXCOORD_6",
            "TEXCOORD_7",
        ];
        for attr_name in TEXCOORD_NAMES {
            if let Some(&uv_idx) = primitive.attributes.get(attr_name) {
                if let Ok(uvs) = self.read_accessor_vec2(uv_idx as usize) {
                    let uvs: Vec<nalgebra::Vector2<f64>> = uvs
                        .into_iter()
                        .map(|[u, v]| nalgebra::Vector2::new(f64::from(u), f64::from(v)))
                        .collect();
                    mesh.attributes_vertex.uv.push(uvs);
                }
            } else {
                // Stop at first missing UV set
                break;
            }
        }

        // Load tangents if present (VEC4: xyz = tangent direction, w = handedness)
        if let Some(&tangent_idx) = primitive.attributes.get("TANGENT")
            && let Ok(tangents) = self.read_accessor_vec4(tangent_idx as usize)
        {
            let tangents: Vec<Vector4<f64>> = tangents
                .into_iter()
                .map(|[x, y, z, w]| {
                    Vector4::new(f64::from(x), f64::from(y), f64::from(z), f64::from(w))
                })
                .collect();
            mesh.attributes_vertex.tangents.push(tangents);
        }

        // Load vertex colors if present
        if let Some(&color_idx) = primitive.attributes.get("COLOR_0")
            && let Ok(colors) = self.read_accessor_vec4(color_idx as usize)
        {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let colors: Vec<Vector4<u8>> = colors
                .into_iter()
                .map(|[r, g, b, a]| {
                    Vector4::new(
                        (r * 255.0) as u8,
                        (g * 255.0) as u8,
                        (b * 255.0) as u8,
                        (a * 255.0) as u8,
                    )
                })
                .collect();
            mesh.attributes_vertex.colors.push(colors);
        }

        // Assign material to mesh if present
        if let Some(material_index) = primitive.material {
            if let Some(material) = materials.get(material_index as usize) {
                let material_name = material.name().to_string();
                let num_faces = mesh.faces.len();

                // Add material grouping to face attributes
                mesh.attributes_face.groupings.push(Grouping {
                    kind: GroupingKind::Material,
                    names: vec![material_name],
                    indices: vec![0; num_faces], // All faces use material index 0
                });

                // Clone the material for this mesh
                mesh.materials.push(material.clone());
            } else if cfg!(debug_assertions) {
                eprintln!(
                    "[gltf] warning: primitive has material index {} but only {} materials loaded",
                    material_index,
                    materials.len()
                );
            }
        }

        // Process primitive extensions (TM_brep_faces, etc.)
        if !primitive.extensions.is_empty() {
            let ext_result = self
                .extensions
                .handle_primitive(&primitive.extensions, &|accessor_idx| {
                    self.read_accessor_indices(accessor_idx)
                })?;
            if !ext_result.face_surfaces.is_empty() {
                mesh.face_surfaces = ext_result.face_surfaces;
            }
            if let Some(grouping) = ext_result.surface_grouping {
                mesh.attributes_face.groupings.push(grouping);
            }
        }

        Ok(mesh)
    }

    /// Resolve an accessor to buffer data and metadata.
    fn resolve_accessor(
        &self,
        index: GltfIndex,
        expected_type: Option<&AccessorType>,
        element_size: usize,
    ) -> Result<AccessorReader<'_>> {
        let accessor = self
            .header
            .accessors
            .get(index)
            .context("Invalid accessor index")?;

        if let Some(expected) = expected_type {
            if &accessor.type_ != expected {
                bail!("Expected {:?} accessor, got {:?}", expected, accessor.type_);
            }
        }

        let buffer_view_idx = accessor
            .buffer_view
            .context("Accessor missing buffer view")? as usize;
        let buffer_view = self
            .header
            .buffer_views
            .get(buffer_view_idx)
            .context("Invalid buffer view index")?;
        let buffer = self
            .buffers
            .get(buffer_view.buffer as usize)
            .context("Invalid buffer index")?;

        let start = buffer_view.byte_offset as usize + accessor.byte_offset as usize;
        let stride = buffer_view.byte_stride.map_or(element_size, |s| s as usize);

        Ok(AccessorReader {
            buffer,
            start,
            stride,
            count: accessor.count as usize,
        })
    }

    /// Read accessor data with a custom element parser.
    fn read_accessor_with<T, F>(
        &self,
        index: GltfIndex,
        expected_type: Option<&AccessorType>,
        element_size: usize,
        parse: F,
    ) -> Result<Vec<T>>
    where
        F: Fn(&[u8]) -> T,
    {
        let reader = self.resolve_accessor(index, expected_type, element_size)?;
        let mut result = Vec::with_capacity(reader.count);

        for i in 0..reader.count {
            let offset = reader.start + i * reader.stride;
            if offset + element_size > reader.buffer.len() {
                bail!("Buffer read out of bounds");
            }
            result.push(parse(&reader.buffer[offset..offset + element_size]));
        }

        Ok(result)
    }

    /// Read a VEC2 accessor as f32 arrays.
    fn read_accessor_vec2(&self, index: GltfIndex) -> Result<Vec<[f32; 2]>> {
        self.read_accessor_with(index, Some(&AccessorType::VEC2), 8, |bytes| {
            [
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            ]
        })
    }

    /// Read a VEC3 accessor as f32 arrays.
    fn read_accessor_vec3(&self, index: GltfIndex) -> Result<Vec<[f32; 3]>> {
        self.read_accessor_with(index, Some(&AccessorType::VEC3), 12, |bytes| {
            [
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            ]
        })
    }

    /// Read a VEC4 accessor as f32 arrays.
    fn read_accessor_vec4(&self, index: GltfIndex) -> Result<Vec<[f32; 4]>> {
        self.read_accessor_with(index, Some(&AccessorType::VEC4), 16, |bytes| {
            [
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
                f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            ]
        })
    }

    /// Read an accessor as scalar f32 values.
    fn read_accessor_scalar(&self, index: GltfIndex) -> Result<Vec<f32>> {
        self.read_accessor_with(index, Some(&AccessorType::SCALAR), 4, |bytes| {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        })
    }

    /// Read indices from an accessor (supports u8, u16, u32).
    fn read_accessor_indices(&self, index: GltfIndex) -> Result<Vec<usize>> {
        let accessor = self
            .header
            .accessors
            .get(index)
            .context("Invalid accessor index")?;
        let component_type = accessor.component_type;

        match component_type {
            COMPONENT_U8 => self.read_accessor_with(index, None, 1, |bytes| bytes[0] as usize),
            COMPONENT_U16 => self.read_accessor_with(index, None, 2, |bytes| {
                u16::from_le_bytes([bytes[0], bytes[1]]) as usize
            }),
            COMPONENT_U32 => self.read_accessor_with(index, None, 4, |bytes| {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize
            }),
            _ => bail!("Unsupported index component type: {}", component_type),
        }
    }

    /// Convert a GLTF camera to our Camera type.
    #[allow(clippy::unused_self)]
    fn convert_camera(&self, gltf_cam: &gltf_2::GltfCamera) -> Camera {
        let projection = if gltf_cam.type_ == CameraType::Perspective {
            if let Some(persp) = &gltf_cam.perspective {
                CameraProjection::Perspective {
                    fov_y: persp.yfov,
                    aspect: persp.aspect_ratio,
                    znear: persp.znear,
                    zfar: persp.zfar.unwrap_or(1000.0),
                }
            } else {
                CameraProjection::default()
            }
        } else if let Some(ortho) = &gltf_cam.orthographic {
            CameraProjection::Orthographic {
                xmag: ortho.xmag,
                ymag: ortho.ymag,
                znear: ortho.znear,
                zfar: ortho.zfar,
            }
        } else {
            CameraProjection::default()
        };

        let name = gltf_cam
            .name
            .as_deref()
            .unwrap_or("")
            .to_string();

        Camera { name, projection }
    }

    /// Convert a GLTF light to our Light type.
    #[allow(clippy::unused_self)]
    fn convert_light(&self, gltf_light: &gltf_2::GltfLight) -> Light {
        let light_type = match gltf_light.type_ {
            GltfLightType::Directional => LightType::Directional,
            GltfLightType::Spot => {
                if let Some(spot) = &gltf_light.spot {
                    LightType::Spot {
                        inner: spot.inner_cone_angle,
                        outer: spot.outer_cone_angle,
                    }
                } else {
                    LightType::Spot {
                        inner: 0.0,
                        outer: std::f64::consts::FRAC_PI_4,
                    }
                }
            }
            GltfLightType::Point => LightType::Point,
        };

        let name = gltf_light
            .name
            .as_deref()
            .unwrap_or("")
            .to_string();

        Light {
            name,
            light_type,
            color: gltf_light.color,
            intensity: gltf_light.intensity,
            range: gltf_light.range,
        }
    }

    /// Build the scene graph from GLTF nodes.
    fn build_scene_graph(&self) -> SceneGraph {
        let mut graph = SceneGraph::new();

        // Create nodes
        for gltf_node in self.header.nodes.as_slice() {
            let transform = self.node_transform(gltf_node);

            let (kind, index) = if let Some(mesh_idx) = gltf_node.mesh {
                (SceneNodeKind::Geometry, vec![mesh_idx as usize])
            } else if let Some(cam_idx) = gltf_node.camera {
                (SceneNodeKind::Camera, vec![cam_idx as usize])
            } else if let Some(light_ext) = gltf_node.extensions.get("KHR_lights_punctual") {
                if let Some(light_idx) = light_ext.get("light").and_then(|v| v.as_u64()) {
                    (SceneNodeKind::Light, vec![light_idx as usize])
                } else {
                    (SceneNodeKind::Custom, vec![])
                }
            } else {
                (SceneNodeKind::Custom, vec![])
            };

            let name = gltf_node
                .name
                .as_deref()
                .unwrap_or("")
                .to_string();

            let children: Vec<usize> = gltf_node
                .children
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|&c| c as usize)
                .collect();

            let node = SceneNode {
                name,
                children,
                transform,
                kind,
                index,
            };

            graph.add_node(node);
        }

        // Set root from default scene
        if let Some(scene_idx) = self.header.scene {
            let scenes = self.header.scenes.as_slice();
            if let Some(scene) = scenes.get(scene_idx as usize)
                && let Some(nodes) = &scene.nodes
                && !nodes.is_empty()
            {
                let nodes_usize: Vec<usize> = nodes.iter().map(|&n| n as usize).collect();
                // If multiple roots, create a synthetic root
                if nodes_usize.len() == 1 {
                    graph.root = nodes_usize[0];
                } else {
                    let scene_name = scene
                        .name
                        .as_deref()
                        .unwrap_or("root")
                        .to_string();
                    let root = SceneNode {
                        name: scene_name,
                        children: nodes_usize,
                        transform: None,
                        kind: SceneNodeKind::Custom,
                        index: vec![],
                    };
                    graph.root = graph.add_node(root);
                }
            }
        }

        graph
    }

    /// Compute the transform matrix for a node.
    #[allow(clippy::unused_self)]
    fn node_transform(&self, node: &gltf_2::Node) -> Option<Matrix4<f64>> {
        const IDENTITY: [f64; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        const DEFAULT_TRANSLATION: [f64; 3] = [0.0, 0.0, 0.0];
        const DEFAULT_ROTATION: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
        const DEFAULT_SCALE: [f64; 3] = [1.0, 1.0, 1.0];

        // Check if an explicit matrix was provided (differs from identity)
        if node.matrix != IDENTITY {
            return Some(Matrix4::from_column_slice(&node.matrix));
        }

        // Check if any TRS values differ from defaults
        let has_trs = node.translation != DEFAULT_TRANSLATION
            || node.rotation != DEFAULT_ROTATION
            || node.scale != DEFAULT_SCALE;
        if !has_trs {
            return None;
        }

        let t = Matrix4::new_translation(&Vector3::new(
            node.translation[0],
            node.translation[1],
            node.translation[2],
        ));
        let r = UnitQuaternion::from_quaternion(Quaternion::new(
            node.rotation[3],
            node.rotation[0],
            node.rotation[1],
            node.rotation[2],
        ))
        .to_homogeneous();
        let s = Matrix4::new_nonuniform_scaling(&Vector3::new(
            node.scale[0],
            node.scale[1],
            node.scale[2],
        ));

        Some(t * r * s)
    }

    /// Load all materials from the GLTF file.
    fn load_materials(&self) -> Vec<Material> {
        self.header
            .materials
            .par_iter()
            .map(|m| self.convert_material(m))
            .collect()
    }

    /// Convert a GLTF material to our Material type.
    fn convert_material(&self, mat: &gltf_2::Material) -> Material {
        let pbr = mat.pbr_metallic_roughness.as_ref();

        // Parse base color factor (RGBA) - fixed array with defaults
        let bcf = pbr.map_or([1.0, 1.0, 1.0, 1.0], |p| p.base_color_factor);
        let base_color_factor = Vector4::new(bcf[0], bcf[1], bcf[2], bcf[3]);

        // Parse emissive factor (RGB) - fixed array
        let ef = mat.emissive_factor;
        let emissive_factor = Vector3::new(ef[0], ef[1], ef[2]);

        // Parse alpha mode
        let alpha_mode = match mat.alpha_mode {
            GltfAlphaMode::MASK => AlphaMode::Mask,
            GltfAlphaMode::BLEND => AlphaMode::Blend,
            GltfAlphaMode::OPAQUE => AlphaMode::Opaque,
        };

        // Get material name
        let name = mat
            .name
            .as_deref()
            .unwrap_or("")
            .to_string();

        Material::PBR(Box::new(PBRMaterial {
            name,
            base_color_factor,
            base_color_texture: pbr
                .and_then(|p| p.base_color_texture.as_ref())
                .and_then(|t| self.load_texture(t.index as usize)),
            metallic_factor: pbr.map_or(1.0, |p| p.metallic_factor),
            roughness_factor: pbr.map_or(1.0, |p| p.roughness_factor),
            metallic_roughness_texture: pbr
                .and_then(|p| p.metallic_roughness_texture.as_ref())
                .and_then(|t| self.load_texture(t.index as usize)),
            normal_texture: mat
                .normal_texture
                .as_ref()
                .and_then(|t| self.load_texture(t.index as usize)),
            normal_scale: mat
                .normal_texture
                .as_ref()
                .and_then(|t| t.scale)
                .unwrap_or(1.0),
            occlusion_texture: mat
                .occlusion_texture
                .as_ref()
                .and_then(|t| self.load_texture(t.index as usize)),
            occlusion_strength: mat
                .occlusion_texture
                .as_ref()
                .and_then(|t| t.strength)
                .unwrap_or(1.0),
            emissive_factor,
            emissive_texture: mat
                .emissive_texture
                .as_ref()
                .and_then(|t| self.load_texture(t.index as usize)),
            alpha_mode,
            alpha_cutoff: mat.alpha_cutoff,
            double_sided: mat.double_sided,
        }))
    }

    /// Convert a GLTF animation.
    fn convert_animation(&self, gltf_anim: &gltf_2::GltfAnimation) -> Result<Animation> {
        let mut samplers = Vec::new();
        let mut channels = Vec::new();

        for gltf_sampler in &gltf_anim.samplers {
            let timestamps: Vec<f64> = self
                .read_accessor_scalar(gltf_sampler.input as usize)?
                .into_iter()
                .map(f64::from)
                .collect();

            let output_accessor = self
                .header
                .accessors
                .get(gltf_sampler.output as usize)
                .context("Invalid output accessor")?;

            let output_idx = gltf_sampler.output as usize;
            let (values, components) = match output_accessor.type_ {
                AccessorType::VEC3 => {
                    let v = self.read_accessor_vec3(output_idx)?;
                    let flat: Vec<f64> = v.into_iter().flat_map(|a| a.map(f64::from)).collect();
                    (flat, 3)
                }
                AccessorType::VEC4 => {
                    let v = self.read_accessor_vec4(output_idx)?;
                    let flat: Vec<f64> = v.into_iter().flat_map(|a| a.map(f64::from)).collect();
                    (flat, 4)
                }
                AccessorType::SCALAR => {
                    let v = self.read_accessor_scalar(output_idx)?;
                    let flat: Vec<f64> = v.into_iter().map(f64::from).collect();
                    (flat, 1)
                }
                _ => bail!("Unsupported animation output type: {:?}", output_accessor.type_),
            };

            let interpolation = match gltf_sampler.interpolation {
                GltfInterpolation::STEP => Interpolation::Step,
                GltfInterpolation::CUBICSPLINE => Interpolation::CubicSpline,
                GltfInterpolation::LINEAR => Interpolation::Linear,
            };

            samplers.push(AnimationSampler {
                timestamps,
                values,
                components,
                interpolation,
            });
        }

        for gltf_channel in &gltf_anim.channels {
            let node = gltf_channel
                .target
                .node
                .context("Animation channel missing node")? as usize;

            let path = match gltf_channel.target.path {
                GltfAnimationPath::Translation => AnimationPath::Translation,
                GltfAnimationPath::Rotation => AnimationPath::Rotation,
                GltfAnimationPath::Scale => AnimationPath::Scale,
                GltfAnimationPath::Weights => AnimationPath::Weights,
            };

            channels.push(AnimationChannel {
                sampler: gltf_channel.sampler as usize,
                node,
                path,
            });
        }

        let name = gltf_anim
            .name
            .as_deref()
            .unwrap_or("")
            .to_string();

        Ok(Animation {
            name,
            samplers,
            channels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glb_header_parsing() {
        // Minimal valid GLB
        let mut glb = Vec::new();

        // Header: magic, version, length
        glb.extend_from_slice(&GLB_MAGIC.to_le_bytes());
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&0u32.to_le_bytes()); // placeholder length

        // JSON chunk
        let json = br#"{"asset":{"version":"2.0"}}"#;
        let json_len = json.len() as u32;
        let padded_len = (json_len + 3) & !3; // Pad to 4 bytes
        glb.extend_from_slice(&padded_len.to_le_bytes());
        glb.extend_from_slice(&GLB_JSON.to_le_bytes());
        glb.extend_from_slice(json);
        for _ in 0..(padded_len - json_len) {
            glb.push(b' ');
        }

        // Update length
        let total_len = glb.len() as u32;
        glb[8..12].copy_from_slice(&total_len.to_le_bytes());

        let loader = GltfLoader::from_glb(&glb).unwrap();
        assert_eq!(loader.header.asset.version, "2.0");
    }

    #[test]
    fn test_invalid_magic() {
        let data = [0u8; 12];
        assert!(GltfLoader::from_glb(&data).is_err());
    }

    #[test]
    fn test_load_cube_glb() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data/cube.glb");

        if !path.exists() {
            // Skip if test file doesn't exist
            return;
        }

        let data = std::fs::read(&path).unwrap();
        let loader = GltfLoader::from_glb(&data).unwrap();
        let scene = loader.to_scene().unwrap();

        // Cube should have geometry
        assert!(!scene.geometry.is_empty(), "Scene should have geometry");

        // Check the first geometry is a mesh (access by iteration)
        let (name, geom) = scene.geometry.iter().next().unwrap();
        if let Geometry::Mesh(mesh) = geom {
            assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
            assert!(!mesh.faces.is_empty(), "Mesh should have faces");
            // A cube has 8 vertices and 12 faces (2 triangles per side * 6 sides)
            // (though vertex count may vary based on how normals are handled)
            println!(
                "Cube '{}': {} vertices, {} faces",
                name,
                mesh.vertices.len(),
                mesh.faces.len()
            );
        }
    }

    #[test]
    fn test_load_duck_glb() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data/Duck.glb");

        if !path.exists() {
            return;
        }

        let data = std::fs::read(&path).unwrap();
        let loader = GltfLoader::from_glb(&data).unwrap();
        let scene = loader.to_scene().unwrap();

        assert!(!scene.geometry.is_empty(), "Duck should have geometry");
        let (name, geom) = scene.geometry.iter().next().unwrap();
        if let Geometry::Mesh(mesh) = geom {
            println!(
                "Duck '{}': {} vertices, {} faces",
                name,
                mesh.vertices.len(),
                mesh.faces.len()
            );
            // Duck model has substantial geometry
            assert!(mesh.vertices.len() > 100);
            assert!(mesh.faces.len() > 100);
        }
    }

    #[test]
    fn test_load_textured_glb() {
        // Test loading a textured GLB file (BoxTextured.glb has materials and textures)
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data/BoxTextured.glb");

        if !path.exists() {
            println!("Skipping test: BoxTextured.glb not found");
            return;
        }

        let data = std::fs::read(&path).unwrap();
        let loader = GltfLoader::from_glb(&data).unwrap();
        let scene = loader.to_scene().unwrap();

        assert!(!scene.geometry.is_empty(), "Scene should have geometry");

        // Check that materials are loaded
        let (name, geom) = scene.geometry.iter().next().unwrap();
        if let Geometry::Mesh(mesh) = geom {
            println!(
                "BoxTextured '{}': {} vertices, {} faces, {} materials",
                name,
                mesh.vertices.len(),
                mesh.faces.len(),
                mesh.materials.len()
            );

            // BoxTextured should have at least one material
            assert!(
                !mesh.materials.is_empty(),
                "Mesh should have materials loaded"
            );

            // Check that the material is a PBR material
            if let Material::PBR(pbr) = &mesh.materials[0] {
                println!("Material: {}", pbr.name);
                println!("  base_color_factor: {:?}", pbr.base_color_factor);
                println!(
                    "  has base_color_texture: {}",
                    pbr.base_color_texture.is_some()
                );
                println!("  metallic_factor: {}", pbr.metallic_factor);
                println!("  roughness_factor: {}", pbr.roughness_factor);

                // BoxTextured should have a base color texture
                assert!(
                    pbr.base_color_texture.is_some(),
                    "PBR material should have a base color texture"
                );
            }

            // Check that UV coordinates are loaded
            assert!(
                !mesh.attributes_vertex.uv.is_empty(),
                "Mesh should have UV coordinates"
            );
            println!("UV sets: {}", mesh.attributes_vertex.uv.len());
        }
    }

    #[test]
    fn test_load_materials_cesium_milk_truck() {
        // CesiumMilkTruck.glb has multiple materials
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data/CesiumMilkTruck.glb");

        if !path.exists() {
            println!("Skipping test: CesiumMilkTruck.glb not found");
            return;
        }

        let data = std::fs::read(&path).unwrap();
        let loader = GltfLoader::from_glb(&data).unwrap();
        let scene = loader.to_scene().unwrap();

        assert!(!scene.geometry.is_empty(), "Scene should have geometry");

        // Count total materials across all meshes
        let mut total_materials = 0;
        let mut has_texture = false;

        for (_name, geometry) in &scene.geometry {
            if let Geometry::Mesh(mesh) = geometry {
                total_materials += mesh.materials.len();
                for material in &mesh.materials {
                    if let Material::PBR(pbr) = material {
                        if pbr.base_color_texture.is_some() {
                            has_texture = true;
                        }
                    }
                }
            }
        }

        println!("CesiumMilkTruck: {} total materials", total_materials);
        assert!(total_materials > 0, "Should have loaded materials");
        assert!(has_texture, "At least one material should have a texture");
    }
}
