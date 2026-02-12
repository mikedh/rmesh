//! Bidirectional glTF ↔ Scene conversions and GLB export.

use std::collections::HashMap;

use anyhow::Result;
use nalgebra::{Matrix4, UnitQuaternion, Vector3};

use crate::attributes::{AlphaMode, GroupingKind, Material};
use crate::geometry::Geometry;
use crate::mesh::Trimesh;
use crate::scene::{
    Animation, AnimationPath, Camera, CameraProjection, Interpolation, Light, LightType, Scene,
    SceneGraph, SceneNodeKind,
};

use super::schema::{
    self as gltf_2, AccessorType, COMPONENT_F32, COMPONENT_U16, COMPONENT_U32, CameraType,
    GltfAlphaMode, GltfAnimationPath, GltfInterpolation, GltfLightType,
};

// GLB constants
const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const GLB_VERSION: u32 = 2;
const GLB_JSON: u32 = 0x4E4F_534A; // "JSON"
const GLB_BIN: u32 = 0x004E_4942; // "BIN\0"

// ─── Camera conversion ───────────────────────────────────────────────

pub fn camera_to_scene(gltf_cam: &gltf_2::Camera) -> Camera {
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

    Camera {
        name: gltf_cam.name.as_deref().unwrap_or("").to_string(),
        projection,
    }
}

pub fn camera_from_scene(cam: &Camera) -> gltf_2::Camera {
    let (type_, perspective, orthographic) = match &cam.projection {
        CameraProjection::Perspective {
            fov_y,
            aspect,
            znear,
            zfar,
        } => (
            CameraType::Perspective,
            Some(gltf_2::CameraPerspective {
                yfov: *fov_y,
                aspect_ratio: *aspect,
                znear: *znear,
                zfar: Some(*zfar),
                ..Default::default()
            }),
            None,
        ),
        CameraProjection::Orthographic {
            xmag,
            ymag,
            znear,
            zfar,
        } => (
            CameraType::Orthographic,
            None,
            Some(gltf_2::CameraOrthographic {
                xmag: *xmag,
                ymag: *ymag,
                znear: *znear,
                zfar: *zfar,
                ..Default::default()
            }),
        ),
    };

    gltf_2::Camera {
        name: if cam.name.is_empty() {
            None
        } else {
            Some(cam.name.clone())
        },
        type_,
        perspective,
        orthographic,
        ..Default::default()
    }
}

// ─── Light conversion ────────────────────────────────────────────────

pub fn light_to_scene(gltf_light: &gltf_2::Light) -> Light {
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

    Light {
        name: gltf_light.name.as_deref().unwrap_or("").to_string(),
        light_type,
        color: gltf_light.color,
        intensity: gltf_light.intensity,
        range: gltf_light.range,
    }
}

pub fn light_from_scene(light: &Light) -> gltf_2::Light {
    let (type_, spot) = match &light.light_type {
        LightType::Directional => (GltfLightType::Directional, None),
        LightType::Point | LightType::Ambient => (GltfLightType::Point, None),
        LightType::Spot { inner, outer } => (
            GltfLightType::Spot,
            Some(gltf_2::LightSpot {
                inner_cone_angle: *inner,
                outer_cone_angle: *outer,
                ..Default::default()
            }),
        ),
    };

    gltf_2::Light {
        name: if light.name.is_empty() {
            None
        } else {
            Some(light.name.clone())
        },
        type_,
        color: light.color,
        intensity: light.intensity,
        range: light.range,
        spot,
        ..Default::default()
    }
}

// ─── Material conversion ─────────────────────────────────────────────

pub fn material_from_scene(mat: &Material) -> gltf_2::Material {
    match mat {
        Material::PBR(pbr) => {
            let bcf = pbr.base_color_factor;
            gltf_2::Material {
                name: if pbr.name.is_empty() {
                    None
                } else {
                    Some(pbr.name.clone())
                },
                alpha_cutoff: pbr.alpha_cutoff,
                alpha_mode: match pbr.alpha_mode {
                    AlphaMode::Opaque => GltfAlphaMode::OPAQUE,
                    AlphaMode::Mask => GltfAlphaMode::MASK,
                    AlphaMode::Blend => GltfAlphaMode::BLEND,
                },
                double_sided: pbr.double_sided,
                emissive_factor: [
                    pbr.emissive_factor.x,
                    pbr.emissive_factor.y,
                    pbr.emissive_factor.z,
                ],
                pbr_metallic_roughness: Some(gltf_2::MaterialPbrMetallicRoughness {
                    base_color_factor: [bcf.x, bcf.y, bcf.z, bcf.w],
                    metallic_factor: pbr.metallic_factor,
                    roughness_factor: pbr.roughness_factor,
                    // Textures are handled separately via BufferBuilder
                    ..Default::default()
                }),
                ..Default::default()
            }
        }
        _ => gltf_2::Material::default(),
    }
}

// ─── Animation conversion ────────────────────────────────────────────

#[allow(clippy::cast_possible_truncation)]
pub fn animation_from_scene(anim: &Animation, buf: &mut BufferBuilder) -> gltf_2::Animation {
    let mut gltf_samplers = Vec::new();
    let mut gltf_channels = Vec::new();

    for (sampler_idx, sampler) in anim.samplers.iter().enumerate() {
        let input_idx = buf.add_scalar_f32(
            &sampler
                .timestamps
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>(),
        );
        let (output_idx, output_type) = match sampler.components {
            3 => {
                let vecs: Vec<[f32; 3]> = sampler
                    .values
                    .chunks(3)
                    .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
                    .collect();
                (buf.add_vec3(&vecs), AccessorType::VEC3)
            }
            4 => {
                let vecs: Vec<[f32; 4]> = sampler
                    .values
                    .chunks(4)
                    .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32])
                    .collect();
                (buf.add_vec4(&vecs), AccessorType::VEC4)
            }
            _ => {
                let scalars: Vec<f32> = sampler.values.iter().map(|&v| v as f32).collect();
                (buf.add_scalar_f32(&scalars), AccessorType::SCALAR)
            }
        };

        // Fix the output accessor type
        buf.accessors[output_idx].type_ = output_type;

        let interpolation = match sampler.interpolation {
            Interpolation::Step => GltfInterpolation::STEP,
            Interpolation::CubicSpline => GltfInterpolation::CUBICSPLINE,
            Interpolation::Linear => GltfInterpolation::LINEAR,
        };

        gltf_samplers.push(gltf_2::AnimationSampler {
            input: input_idx as u64,
            output: output_idx as u64,
            interpolation,
            ..Default::default()
        });

        // Find channels that reference this sampler
        for channel in &anim.channels {
            if channel.sampler == sampler_idx {
                gltf_channels.push(gltf_2::AnimationChannel {
                    sampler: sampler_idx as u64,
                    target: gltf_2::AnimationChannelTarget {
                        node: Some(channel.node as u64),
                        path: match channel.path {
                            AnimationPath::Translation => GltfAnimationPath::Translation,
                            AnimationPath::Rotation => GltfAnimationPath::Rotation,
                            AnimationPath::Scale => GltfAnimationPath::Scale,
                            AnimationPath::Weights => GltfAnimationPath::Weights,
                        },
                        ..Default::default()
                    },
                    ..Default::default()
                });
            }
        }
    }

    gltf_2::Animation {
        name: if anim.name.is_empty() {
            None
        } else {
            Some(anim.name.clone())
        },
        channels: gltf_channels,
        samplers: gltf_samplers,
        ..Default::default()
    }
}

// ─── Scene graph conversion ──────────────────────────────────────────

pub fn scene_graph_from_scene<S: std::hash::BuildHasher>(
    graph: &SceneGraph,
    geom_index_map: &HashMap<usize, usize, S>,
) -> Vec<gltf_2::Node> {
    graph
        .nodes
        .iter()
        .map(|node| {
            let (mesh, camera, light_ext) = match node.kind {
                SceneNodeKind::Geometry => {
                    let mesh_idx = node
                        .index
                        .first()
                        .and_then(|&i| geom_index_map.get(&i))
                        .map(|&i| i as u64);
                    (mesh_idx, None, None)
                }
                SceneNodeKind::Camera => {
                    let cam_idx = node.index.first().map(|&i| i as u64);
                    (None, cam_idx, None)
                }
                SceneNodeKind::Light => {
                    let light_idx = node.index.first().map(|&i| {
                        let mut map = serde_json::Map::new();
                        map.insert("light".to_string(), serde_json::json!(i));
                        serde_json::Value::Object(map)
                    });
                    (None, None, light_idx)
                }
                SceneNodeKind::Custom => (None, None, None),
            };

            let (matrix, translation, rotation, scale) = decompose_transform(node.transform);

            let children = if node.children.is_empty() {
                None
            } else {
                Some(node.children.iter().map(|&c| c as u64).collect())
            };

            let mut extensions = serde_json::Map::new();
            if let Some(light_val) = light_ext {
                extensions.insert("KHR_lights_punctual".to_string(), light_val);
            }

            gltf_2::Node {
                name: if node.name.is_empty() {
                    None
                } else {
                    Some(node.name.clone())
                },
                mesh,
                camera,
                children,
                matrix,
                translation,
                rotation,
                scale,
                extensions,
                ..Default::default()
            }
        })
        .collect()
}

fn decompose_transform(
    transform: Option<Matrix4<f64>>,
) -> ([f64; 16], [f64; 3], [f64; 4], [f64; 3]) {
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let default_t = [0.0, 0.0, 0.0];
    let default_r = [0.0, 0.0, 0.0, 1.0];
    let default_s = [1.0, 1.0, 1.0];

    let Some(mat) = transform else {
        return (identity, default_t, default_r, default_s);
    };

    // Extract translation
    let translation = [mat[(0, 3)], mat[(1, 3)], mat[(2, 3)]];

    // Extract scale from column lengths
    let col0 = Vector3::new(mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]);
    let col1 = Vector3::new(mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]);
    let col2 = Vector3::new(mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]);
    let sx = col0.norm();
    let sy = col1.norm();
    let sz = col2.norm();
    let scale = [sx, sy, sz];

    // Extract rotation (normalize columns to remove scale)
    if sx > 1e-10 && sy > 1e-10 && sz > 1e-10 {
        let rot_mat = Matrix4::from_columns(&[
            (col0 / sx).to_homogeneous(),
            (col1 / sy).to_homogeneous(),
            (col2 / sz).to_homogeneous(),
            nalgebra::Vector4::new(0.0, 0.0, 0.0, 1.0),
        ]);
        let rot3 = rot_mat.fixed_view::<3, 3>(0, 0).into_owned();
        let q =
            UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(rot3));
        let rotation = [q.i, q.j, q.k, q.w]; // glTF order: x,y,z,w
        (identity, translation, rotation, scale)
    } else {
        (identity, translation, default_r, scale)
    }
}

// ─── BufferBuilder ───────────────────────────────────────────────────

/// Builds the binary buffer data plus accessor/buffer-view arrays for GLB export.
pub struct BufferBuilder {
    pub data: Vec<u8>,
    pub accessors: Vec<gltf_2::Accessor>,
    pub buffer_views: Vec<gltf_2::BufferView>,
}

impl Default for BufferBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferBuilder {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            accessors: Vec::new(),
            buffer_views: Vec::new(),
        }
    }

    /// Align data to 4-byte boundary.
    fn align(&mut self) {
        while !self.data.len().is_multiple_of(4) {
            self.data.push(0);
        }
    }

    /// Add a buffer view + accessor, returning the accessor index.
    fn add_accessor(
        &mut self,
        bytes: &[u8],
        count: usize,
        component_type: u32,
        accessor_type: AccessorType,
        min: Vec<f64>,
        max: Vec<f64>,
    ) -> usize {
        self.align();
        let byte_offset = self.data.len();
        self.data.extend_from_slice(bytes);
        let byte_length = bytes.len();

        let bv_idx = self.buffer_views.len();
        self.buffer_views.push(gltf_2::BufferView {
            buffer: 0,
            byte_length: byte_length as u64,
            byte_offset: byte_offset as u64,
            ..Default::default()
        });

        let acc_idx = self.accessors.len();
        self.accessors.push(gltf_2::Accessor {
            buffer_view: Some(bv_idx as u64),
            byte_offset: 0,
            component_type,
            count: count as u64,
            type_: accessor_type,
            min,
            max,
            ..Default::default()
        });

        acc_idx
    }

    /// Add VEC3 f32 data (positions, normals). Returns accessor index.
    pub fn add_vec3(&mut self, data: &[[f32; 3]]) -> usize {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        let mut bytes = Vec::with_capacity(data.len() * 12);
        for v in data {
            for i in 0..3 {
                min[i] = min[i].min(v[i]);
                max[i] = max[i].max(v[i]);
            }
            bytes.extend_from_slice(&v[0].to_le_bytes());
            bytes.extend_from_slice(&v[1].to_le_bytes());
            bytes.extend_from_slice(&v[2].to_le_bytes());
        }
        self.add_accessor(
            &bytes,
            data.len(),
            COMPONENT_F32,
            AccessorType::VEC3,
            min.iter().map(|&v| f64::from(v)).collect(),
            max.iter().map(|&v| f64::from(v)).collect(),
        )
    }

    /// Add VEC4 f32 data (colors, tangents). Returns accessor index.
    pub fn add_vec4(&mut self, data: &[[f32; 4]]) -> usize {
        let mut bytes = Vec::with_capacity(data.len() * 16);
        for v in data {
            bytes.extend_from_slice(&v[0].to_le_bytes());
            bytes.extend_from_slice(&v[1].to_le_bytes());
            bytes.extend_from_slice(&v[2].to_le_bytes());
            bytes.extend_from_slice(&v[3].to_le_bytes());
        }
        self.add_accessor(
            &bytes,
            data.len(),
            COMPONENT_F32,
            AccessorType::VEC4,
            vec![],
            vec![],
        )
    }

    /// Add VEC2 f32 data (UVs). Returns accessor index.
    pub fn add_vec2(&mut self, data: &[[f32; 2]]) -> usize {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for v in data {
            bytes.extend_from_slice(&v[0].to_le_bytes());
            bytes.extend_from_slice(&v[1].to_le_bytes());
        }
        self.add_accessor(
            &bytes,
            data.len(),
            COMPONENT_F32,
            AccessorType::VEC2,
            vec![],
            vec![],
        )
    }

    /// Add SCALAR f32 data (timestamps, weights). Returns accessor index.
    pub fn add_scalar_f32(&mut self, data: &[f32]) -> usize {
        let mut min_v = f32::MAX;
        let mut max_v = f32::MIN;
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.add_accessor(
            &bytes,
            data.len(),
            COMPONENT_F32,
            AccessorType::SCALAR,
            vec![f64::from(min_v)],
            vec![f64::from(max_v)],
        )
    }

    /// Add triangle indices. Auto-selects u16 or u32 based on vertex count.
    /// Returns accessor index.
    #[allow(clippy::cast_possible_truncation)]
    pub fn add_indices(&mut self, indices: &[usize], max_vertex: usize) -> usize {
        if u16::try_from(max_vertex).is_ok() {
            let mut bytes = Vec::with_capacity(indices.len() * 2);
            for &i in indices {
                bytes.extend_from_slice(&(i as u16).to_le_bytes());
            }
            self.add_accessor(
                &bytes,
                indices.len(),
                COMPONENT_U16,
                AccessorType::SCALAR,
                vec![],
                vec![],
            )
        } else {
            let mut bytes = Vec::with_capacity(indices.len() * 4);
            for &i in indices {
                bytes.extend_from_slice(&(i as u32).to_le_bytes());
            }
            self.add_accessor(
                &bytes,
                indices.len(),
                COMPONENT_U32,
                AccessorType::SCALAR,
                vec![],
                vec![],
            )
        }
    }
}

// ─── GLB export ──────────────────────────────────────────────────────

/// Export a Scene to GLB binary data.
pub fn from_scene(scene: &Scene) -> Result<Vec<u8>> {
    let mut buf = BufferBuilder::new();
    let mut gltf_meshes = Vec::new();
    let mut gltf_materials: Vec<gltf_2::Material> = Vec::new();
    let mut material_map: HashMap<String, usize> = HashMap::new();

    // Map from scene geometry index → glTF mesh index
    let mut geom_index_map: HashMap<usize, usize> = HashMap::new();

    // Export geometry
    for (geom_idx, (_name, geom)) in scene.geometry.iter().enumerate() {
        let Geometry::Mesh(trimesh) = geom else {
            continue;
        };

        if let Some(mesh) =
            export_trimesh(trimesh, &mut buf, &mut gltf_materials, &mut material_map)
        {
            geom_index_map.insert(geom_idx, gltf_meshes.len());
            gltf_meshes.push(mesh);
        }
    }

    // Export cameras
    let gltf_cameras: Vec<gltf_2::Camera> = scene.cameras.iter().map(camera_from_scene).collect();

    // Export lights → KHR_lights_punctual
    let gltf_lights: Vec<gltf_2::Light> = scene.lights.iter().map(light_from_scene).collect();

    // Export scene graph
    let gltf_nodes = scene_graph_from_scene(&scene.graph, &geom_index_map);

    // Export animations
    let gltf_animations: Vec<gltf_2::Animation> = scene
        .animations
        .iter()
        .map(|a| animation_from_scene(a, &mut buf))
        .collect();

    // Find root nodes for the default scene
    let root_children: Vec<u64> = if scene.graph.nodes.is_empty() {
        vec![]
    } else {
        // The scene graph root's children are the scene root nodes.
        // If there is a root node, use its children. Otherwise use [0].
        let root = &scene.graph.nodes[scene.graph.root];
        if root.children.is_empty() {
            vec![scene.graph.root as u64]
        } else {
            root.children.iter().map(|&c| c as u64).collect()
        }
    };

    // Build extensions
    let mut extensions = serde_json::Map::new();
    let mut extensions_used = Vec::new();
    if !gltf_lights.is_empty() {
        let lights_ext = gltf_2::GlTfKhrLightsPunctual {
            lights: gltf_lights,
            ..Default::default()
        };
        extensions.insert(
            "KHR_lights_punctual".to_string(),
            serde_json::to_value(lights_ext)?,
        );
        extensions_used.push("KHR_lights_punctual".to_string());
    }

    // Build buffer (just one for GLB)
    let buffers = if buf.data.is_empty() {
        vec![]
    } else {
        vec![gltf_2::Buffer {
            byte_length: buf.data.len() as u64,
            ..Default::default()
        }]
    };

    let gltf = gltf_2::GlTf {
        asset: gltf_2::Asset {
            version: "2.0".to_string(),
            generator: Some("rmesh".to_string()),
            ..Default::default()
        },
        accessors: buf.accessors,
        buffer_views: buf.buffer_views,
        buffers,
        meshes: gltf_meshes,
        materials: gltf_materials,
        cameras: gltf_cameras,
        nodes: gltf_nodes,
        scenes: vec![gltf_2::Scene {
            name: Some("Scene".to_string()),
            nodes: Some(root_children),
            ..Default::default()
        }],
        scene: Some(0),
        animations: gltf_animations,
        extensions,
        extensions_used: if extensions_used.is_empty() {
            None
        } else {
            Some(extensions_used)
        },
        ..Default::default()
    };

    // Serialize to JSON
    let json_bytes = serde_json::to_vec(&gltf)?;

    // Pack GLB
    Ok(pack_glb(&json_bytes, &buf.data))
}

/// Export a single Trimesh to a glTF Mesh with primitives.
#[allow(clippy::cast_possible_truncation)]
fn export_trimesh(
    trimesh: &Trimesh,
    buf: &mut BufferBuilder,
    materials: &mut Vec<gltf_2::Material>,
    material_map: &mut HashMap<String, usize>,
) -> Option<gltf_2::Mesh> {
    if trimesh.vertices.is_empty() || trimesh.faces.is_empty() {
        return None;
    }

    // Find material grouping to split into primitives
    let mat_grouping = trimesh
        .attributes_face
        .groupings
        .iter()
        .find(|g| g.kind == GroupingKind::Material);

    // Group faces by material
    let face_groups: Vec<(Option<usize>, Vec<usize>)> = if let Some(grouping) = mat_grouping {
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (face_idx, &mat_idx) in grouping.indices.iter().enumerate() {
            groups.entry(mat_idx).or_default().push(face_idx);
        }
        let mut sorted: Vec<_> = groups.into_iter().collect();
        sorted.sort_by_key(|(k, _)| *k);
        sorted
            .into_iter()
            .map(|(mat_idx, faces)| {
                // Resolve material index
                let scene_mat = trimesh.materials.get(mat_idx);
                let gltf_mat_idx = scene_mat.map(|m| {
                    let name = m.name().to_string();
                    *material_map.entry(name.clone()).or_insert_with(|| {
                        let idx = materials.len();
                        materials.push(material_from_scene(m));
                        idx
                    })
                });
                (gltf_mat_idx, faces)
            })
            .collect()
    } else {
        // Single primitive with all faces
        let all_faces: Vec<usize> = (0..trimesh.faces.len()).collect();
        let mat_idx = if trimesh.materials.is_empty() {
            None
        } else {
            let m = &trimesh.materials[0];
            let name = m.name().to_string();
            Some(*material_map.entry(name.clone()).or_insert_with(|| {
                let idx = materials.len();
                materials.push(material_from_scene(m));
                idx
            }))
        };
        vec![(mat_idx, all_faces)]
    };

    let mut primitives = Vec::new();

    for (mat_idx, face_indices) in &face_groups {
        // Collect unique vertices used by these faces
        let mut vertex_map: HashMap<usize, usize> = HashMap::new();
        let mut local_verts: Vec<usize> = Vec::new();
        let mut local_faces: Vec<[usize; 3]> = Vec::with_capacity(face_indices.len());

        for &fi in face_indices {
            let face = trimesh.faces[fi];
            let mut new_face = [0usize; 3];
            for (i, &vi) in face.iter().enumerate() {
                let new_vi = *vertex_map.entry(vi).or_insert_with(|| {
                    let idx = local_verts.len();
                    local_verts.push(vi);
                    idx
                });
                new_face[i] = new_vi;
            }
            local_faces.push(new_face);
        }

        // Write positions
        let positions: Vec<[f32; 3]> = local_verts
            .iter()
            .map(|&vi| {
                let p = &trimesh.vertices[vi];
                [p.x as f32, p.y as f32, p.z as f32]
            })
            .collect();
        let pos_acc = buf.add_vec3(&positions);

        let mut attributes = HashMap::new();
        attributes.insert("POSITION".to_string(), pos_acc as u64);

        // Write normals if present
        if let Some(normals) = trimesh.attributes_vertex.normals.first() {
            let normal_data: Vec<[f32; 3]> = local_verts
                .iter()
                .map(|&vi| {
                    let n = &normals[vi];
                    [n.x as f32, n.y as f32, n.z as f32]
                })
                .collect();
            let norm_acc = buf.add_vec3(&normal_data);
            attributes.insert("NORMAL".to_string(), norm_acc as u64);
        }

        // Write UVs
        for (uv_idx, uvs) in trimesh.attributes_vertex.uv.iter().enumerate() {
            let uv_data: Vec<[f32; 2]> = local_verts
                .iter()
                .map(|&vi| {
                    let uv = &uvs[vi];
                    [uv.x as f32, uv.y as f32]
                })
                .collect();
            let uv_acc = buf.add_vec2(&uv_data);
            attributes.insert(format!("TEXCOORD_{uv_idx}"), uv_acc as u64);
        }

        // Write vertex colors
        if let Some(colors) = trimesh.attributes_vertex.colors.first() {
            let color_data: Vec<[f32; 4]> = local_verts
                .iter()
                .map(|&vi| {
                    let c = &colors[vi];
                    [
                        f32::from(c.x) / 255.0,
                        f32::from(c.y) / 255.0,
                        f32::from(c.z) / 255.0,
                        f32::from(c.w) / 255.0,
                    ]
                })
                .collect();
            let color_acc = buf.add_vec4(&color_data);
            attributes.insert("COLOR_0".to_string(), color_acc as u64);
        }

        // Write indices
        let flat_indices: Vec<usize> = local_faces.iter().flat_map(|f| f.iter().copied()).collect();
        let max_vertex = local_verts.len();
        let idx_acc = buf.add_indices(&flat_indices, max_vertex);

        primitives.push(gltf_2::MeshPrimitive {
            attributes,
            indices: Some(idx_acc as u64),
            material: mat_idx.map(|i| i as u64),
            mode: 4, // TRIANGLES
            ..Default::default()
        });
    }

    if primitives.is_empty() {
        return None;
    }

    Some(gltf_2::Mesh {
        primitives,
        ..Default::default()
    })
}

/// Pack JSON and binary data into GLB format.
#[allow(clippy::cast_possible_truncation)]
fn pack_glb(json: &[u8], bin: &[u8]) -> Vec<u8> {
    // Pad JSON to 4-byte alignment with spaces
    let json_padding = (4 - (json.len() % 4)) % 4;
    let json_chunk_length = json.len() + json_padding;

    // Pad BIN to 4-byte alignment with zeros
    let bin_padding = (4 - (bin.len() % 4)) % 4;
    let bin_chunk_length = bin.len() + bin_padding;

    let has_bin = !bin.is_empty();
    let total_length = 12 // GLB header
        + 8 + json_chunk_length // JSON chunk header + data
        + if has_bin { 8 + bin_chunk_length } else { 0 }; // BIN chunk header + data

    let mut out = Vec::with_capacity(total_length);

    // GLB header
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&GLB_VERSION.to_le_bytes());
    out.extend_from_slice(&(total_length as u32).to_le_bytes());

    // JSON chunk
    out.extend_from_slice(&(json_chunk_length as u32).to_le_bytes());
    out.extend_from_slice(&GLB_JSON.to_le_bytes());
    out.extend_from_slice(json);
    out.extend(std::iter::repeat_n(b' ', json_padding));

    // BIN chunk
    if has_bin {
        out.extend_from_slice(&(bin_chunk_length as u32).to_le_bytes());
        out.extend_from_slice(&GLB_BIN.to_le_bytes());
        out.extend_from_slice(bin);
        out.extend(std::iter::repeat_n(0u8, bin_padding));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exchange::gltf::GltfLoader;

    fn test_data_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data")
    }

    /// Test roundtrip: load GLB → to_scene → from_scene → re-load → verify counts.
    fn roundtrip_test(filename: &str) {
        let path = test_data_dir().join(filename);
        if !path.exists() {
            eprintln!("Skipping roundtrip test: {} not found", filename);
            return;
        }

        // Load original
        let data = std::fs::read(&path).unwrap();
        let loader = GltfLoader::from_glb(&data).unwrap();
        let scene = loader.to_scene().unwrap();

        let orig_geom_count = scene.geometry.len();
        let orig_vertex_count: usize = scene
            .geometry
            .values()
            .filter_map(|g| match g {
                Geometry::Mesh(m) => Some(m.vertices.len()),
                _ => None,
            })
            .sum();
        let orig_face_count: usize = scene
            .geometry
            .values()
            .filter_map(|g| match g {
                Geometry::Mesh(m) => Some(m.faces.len()),
                _ => None,
            })
            .sum();

        // Export to GLB
        let glb_data = from_scene(&scene).unwrap();

        // Re-import
        let loader2 = GltfLoader::from_glb(&glb_data).unwrap();
        let scene2 = loader2.to_scene().unwrap();

        let rt_geom_count = scene2.geometry.len();
        let rt_vertex_count: usize = scene2
            .geometry
            .values()
            .filter_map(|g| match g {
                Geometry::Mesh(m) => Some(m.vertices.len()),
                _ => None,
            })
            .sum();
        let rt_face_count: usize = scene2
            .geometry
            .values()
            .filter_map(|g| match g {
                Geometry::Mesh(m) => Some(m.faces.len()),
                _ => None,
            })
            .sum();

        println!(
            "[{}] orig: {} geoms, {} verts, {} faces → rt: {} geoms, {} verts, {} faces",
            filename,
            orig_geom_count,
            orig_vertex_count,
            orig_face_count,
            rt_geom_count,
            rt_vertex_count,
            rt_face_count,
        );

        assert_eq!(
            orig_geom_count, rt_geom_count,
            "{}: geometry count mismatch",
            filename
        );
        // Face count must match exactly
        assert_eq!(
            orig_face_count, rt_face_count,
            "{}: face count mismatch",
            filename
        );
        // Vertex count may differ slightly due to primitive splitting,
        // but should be >= original (splitting only adds, never removes)
        assert!(
            rt_vertex_count >= orig_vertex_count,
            "{}: vertex count decreased: {} < {}",
            filename,
            rt_vertex_count,
            orig_vertex_count,
        );
    }

    #[test]
    fn test_roundtrip_cube() {
        roundtrip_test("cube.glb");
    }

    #[test]
    fn test_roundtrip_duck() {
        roundtrip_test("Duck.glb");
    }

    #[test]
    fn test_roundtrip_box_textured() {
        roundtrip_test("BoxTextured.glb");
    }

    #[test]
    fn test_roundtrip_monkey() {
        roundtrip_test("monkey.glb");
    }

    #[test]
    fn test_export_empty_scene() {
        let scene = Scene::new();
        let glb = from_scene(&scene).unwrap();
        // Should produce valid GLB
        assert!(glb.len() >= 12);
        let magic = u32::from_le_bytes([glb[0], glb[1], glb[2], glb[3]]);
        assert_eq!(magic, GLB_MAGIC);
    }
}
