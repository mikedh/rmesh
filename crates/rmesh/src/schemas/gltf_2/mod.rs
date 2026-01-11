// glTF 2.0 schema bindings
// Generated from JSON Schema

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gltf {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accessors: Option<Vec<Accessor>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub animations: Option<Vec<Animation>>,
    pub asset: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buffer_views: Option<Vec<BufferView>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buffers: Option<Vec<Buffer>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cameras: Option<Vec<Camera>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions_required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions_used: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<Image>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub materials: Option<Vec<Material>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meshes: Option<Vec<Mesh>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nodes: Option<Vec<Node>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samplers: Option<Vec<Sampler>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scenes: Option<Vec<Scene>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skins: Option<Vec<Skin>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub textures: Option<Vec<Texture>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Accessor {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buffer_view: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_offset: Option<i64>,
    pub component_type: serde_json::Value,
    pub count: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalized: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse: Option<serde_json::Value>,
    pub type: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Animation {
    pub channels: Vec<AnimationChannel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    pub samplers: Vec<AnimationSampler>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationChannel {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    pub sampler: serde_json::Value,
    pub target: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationSampler {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    pub input: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interpolation: Option<serde_json::Value>,
    pub output: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferView {
    pub buffer: serde_json::Value,
    pub byte_length: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_offset: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_stride: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Buffer {
    pub byte_length: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orthographic: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perspective: Option<serde_json::Value>,
    pub type: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buffer_view: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha_cutoff: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha_mode: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub double_sided: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emissive_factor: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emissive_texture: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normal_texture: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub occlusion_texture: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pbr_metallic_roughness: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    pub primitives: Vec<MeshPrimitive>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshPrimitive {
    pub attributes: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indices: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub targets: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<glTFId>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matrix: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mesh: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rotation: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skin: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub translation: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct glTFId {
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sampler {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mag_filter: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_filter: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wrap_s: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wrap_t: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nodes: Option<Vec<glTFId>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skin {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inverse_bind_matrices: Option<serde_json::Value>,
    pub joints: Vec<glTFId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skeleton: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Texture {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampler: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
}


