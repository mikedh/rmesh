//! glTF 2.0 schema - AUTO-GENERATED from JSON Schema
//! Do not edit manually. Run `cargo run -p gltf-codegen` to regenerate.

#![allow(unused_imports)]

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

pub type GltfIndex = usize;

// Component type codes
pub const COMPONENT_I8: u32 = 5120;
pub const COMPONENT_U8: u32 = 5121;
pub const COMPONENT_I16: u32 = 5122;
pub const COMPONENT_U16: u32 = 5123;
pub const COMPONENT_U32: u32 = 5125;
pub const COMPONENT_F32: u32 = 5126;

// GL primitive modes
pub const GL_POINTS: u32 = 0;
pub const GL_LINES: u32 = 1;
pub const GL_LINE_LOOP: u32 = 2;
pub const GL_LINE_STRIP: u32 = 3;
pub const GL_TRIANGLES: u32 = 4;
pub const GL_TRIANGLE_STRIP: u32 = 5;
pub const GL_TRIANGLE_FAN: u32 = 6;

pub fn component_size(t: u32) -> usize {
    match t {
        COMPONENT_I8 | COMPONENT_U8 => 1,
        COMPONENT_I16 | COMPONENT_U16 => 2,
        _ => 4,
    }
}

pub fn accessor_type_count(t: &str) -> usize {
    match t {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT2" => 4,
        "MAT3" => 9,
        "MAT4" => 16,
        _ => 1,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Accessor {
    #[serde(rename = "bufferView")]
    pub buffer_view: Option<GltfIndex>,
    #[serde(rename = "byteOffset")]
    pub byte_offset: Option<i64>,
    #[serde(rename = "componentType")]
    pub component_type: serde_json::Value,
    pub count: i64,
    pub max: Option<Vec<f64>>,
    pub min: Option<Vec<f64>>,
    pub name: Option<serde_json::Value>,
    pub normalized: Option<bool>,
    pub sparse: Option<AccessorSparse>,
    #[serde(rename = "type")]
    pub type_: serde_json::Value,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for Accessor {
    fn default() -> Self {
        Self {
            buffer_view: Default::default(),
            byte_offset: Some(0),
            component_type: Default::default(),
            count: Default::default(),
            max: Default::default(),
            min: Default::default(),
            name: Default::default(),
            normalized: Some(false),
            sparse: Default::default(),
            type_: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AccessorSparse {
    pub count: i64,
    pub indices: AccessorSparseIndices,
    pub values: AccessorSparseValues,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AccessorSparseIndices {
    #[serde(rename = "bufferView")]
    pub buffer_view: GltfIndex,
    #[serde(rename = "byteOffset")]
    pub byte_offset: Option<i64>,
    #[serde(rename = "componentType")]
    pub component_type: serde_json::Value,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for AccessorSparseIndices {
    fn default() -> Self {
        Self {
            buffer_view: Default::default(),
            byte_offset: Some(0),
            component_type: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AccessorSparseValues {
    #[serde(rename = "bufferView")]
    pub buffer_view: GltfIndex,
    #[serde(rename = "byteOffset")]
    pub byte_offset: Option<i64>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for AccessorSparseValues {
    fn default() -> Self {
        Self {
            buffer_view: Default::default(),
            byte_offset: Some(0),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Animation {
    pub channels: Vec<AnimationChannel>,
    pub name: Option<serde_json::Value>,
    pub samplers: Vec<AnimationSampler>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AnimationChannel {
    pub sampler: GltfIndex,
    pub target: AnimationChannelTarget,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AnimationChannelTarget {
    pub node: Option<GltfIndex>,
    pub path: serde_json::Value,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AnimationChannelTargetKhranimationPointer {
    pub pointer: String,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AnimationSampler {
    pub input: GltfIndex,
    pub interpolation: Option<serde_json::Value>,
    pub output: GltfIndex,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for AnimationSampler {
    fn default() -> Self {
        Self {
            input: Default::default(),
            interpolation: Some(serde_json::Value::String("LINEAR".to_string())),
            output: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Asset {
    pub copyright: Option<String>,
    pub generator: Option<String>,
    #[serde(rename = "minVersion")]
    pub min_version: Option<String>,
    pub version: String,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Buffer {
    #[serde(rename = "byteLength")]
    pub byte_length: i64,
    pub name: Option<serde_json::Value>,
    pub uri: Option<String>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BufferKhrmeshoptCompression {
    pub fallback: Option<bool>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for BufferKhrmeshoptCompression {
    fn default() -> Self {
        Self {
            fallback: Some(false),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BufferView {
    pub buffer: GltfIndex,
    #[serde(rename = "byteLength")]
    pub byte_length: i64,
    #[serde(rename = "byteOffset")]
    pub byte_offset: Option<i64>,
    #[serde(rename = "byteStride")]
    pub byte_stride: Option<i64>,
    pub name: Option<serde_json::Value>,
    pub target: Option<serde_json::Value>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for BufferView {
    fn default() -> Self {
        Self {
            buffer: Default::default(),
            byte_length: Default::default(),
            byte_offset: Some(0),
            byte_stride: Default::default(),
            name: Default::default(),
            target: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BufferViewKhrmeshoptCompression {
    pub buffer: GltfIndex,
    #[serde(rename = "byteLength")]
    pub byte_length: i64,
    #[serde(rename = "byteOffset")]
    pub byte_offset: Option<i64>,
    #[serde(rename = "byteStride")]
    pub byte_stride: i64,
    pub count: i64,
    pub filter: Option<String>,
    pub mode: String,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for BufferViewKhrmeshoptCompression {
    fn default() -> Self {
        Self {
            buffer: Default::default(),
            byte_length: Default::default(),
            byte_offset: Some(0),
            byte_stride: Default::default(),
            count: Default::default(),
            filter: Some("NONE".to_string()),
            mode: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Camera {
    pub name: Option<serde_json::Value>,
    pub orthographic: Option<CameraOrthographic>,
    pub perspective: Option<CameraPerspective>,
    #[serde(rename = "type")]
    pub type_: serde_json::Value,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CameraOrthographic {
    pub xmag: f64,
    pub ymag: f64,
    pub zfar: f64,
    pub znear: f64,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CameraPerspective {
    #[serde(rename = "aspectRatio")]
    pub aspect_ratio: Option<f64>,
    pub yfov: f64,
    pub zfar: Option<f64>,
    pub znear: f64,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Extension {
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Gltf {
    pub accessors: Option<Vec<Accessor>>,
    pub animations: Option<Vec<Animation>>,
    pub asset: Asset,
    #[serde(rename = "bufferViews")]
    pub buffer_views: Option<Vec<BufferView>>,
    pub buffers: Option<Vec<Buffer>>,
    pub cameras: Option<Vec<Camera>>,
    #[serde(rename = "extensionsRequired")]
    pub extensions_required: Option<Vec<String>>,
    #[serde(rename = "extensionsUsed")]
    pub extensions_used: Option<Vec<String>>,
    pub images: Option<Vec<Image>>,
    pub materials: Option<Vec<Material>>,
    pub meshes: Option<Vec<Mesh>>,
    pub nodes: Option<Vec<Node>>,
    pub samplers: Option<Vec<Sampler>>,
    pub scene: Option<GltfIndex>,
    pub scenes: Option<Vec<Scene>>,
    pub skins: Option<Vec<Skin>>,
    pub textures: Option<Vec<Texture>>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GltfChildOfRootProperty {
    pub name: Option<String>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GltfKhrlightsPunctual {
    pub lights: Vec<Light>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GltfKhrmaterialsVariants {
    pub variants: Vec<GltfChildOfRootProperty>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GltfKhrxmpJsonLd {
    pub packets: Vec<serde_json::Value>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GltfProperty {
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Image {
    #[serde(rename = "bufferView")]
    pub buffer_view: Option<GltfIndex>,
    #[serde(rename = "mimeType")]
    pub mime_type: Option<serde_json::Value>,
    pub name: Option<serde_json::Value>,
    pub uri: Option<String>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct KhrxmpJsonLd {
    pub packet: GltfIndex,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Light {
    pub color: Option<Vec<f64>>,
    pub intensity: Option<f64>,
    pub name: Option<serde_json::Value>,
    pub range: Option<f64>,
    pub spot: Option<LightSpot>,
    #[serde(rename = "type")]
    pub type_: String,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            color: Some(vec![1.0, 1.0, 1.0]),
            intensity: Some(1.0),
            name: Default::default(),
            range: Default::default(),
            spot: Default::default(),
            type_: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct LightSpot {
    #[serde(rename = "innerConeAngle")]
    pub inner_cone_angle: Option<f64>,
    #[serde(rename = "outerConeAngle")]
    pub outer_cone_angle: Option<f64>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for LightSpot {
    fn default() -> Self {
        Self {
            inner_cone_angle: Some(0.0),
            outer_cone_angle: Some(0.8),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Material {
    #[serde(rename = "alphaCutoff")]
    pub alpha_cutoff: Option<f64>,
    #[serde(rename = "alphaMode")]
    pub alpha_mode: Option<serde_json::Value>,
    #[serde(rename = "doubleSided")]
    pub double_sided: Option<bool>,
    #[serde(rename = "emissiveFactor")]
    pub emissive_factor: Option<Vec<f64>>,
    #[serde(rename = "emissiveTexture")]
    pub emissive_texture: Option<TextureInfo>,
    pub name: Option<serde_json::Value>,
    #[serde(rename = "normalTexture")]
    pub normal_texture: Option<MaterialNormalTextureInfo>,
    #[serde(rename = "occlusionTexture")]
    pub occlusion_texture: Option<MaterialOcclusionTextureInfo>,
    #[serde(rename = "pbrMetallicRoughness")]
    pub pbr_metallic_roughness: Option<MaterialPbrMetallicRoughness>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            alpha_cutoff: Some(0.5),
            alpha_mode: Some(serde_json::Value::String("OPAQUE".to_string())),
            double_sided: Some(false),
            emissive_factor: Some(vec![0.0, 0.0, 0.0]),
            emissive_texture: Default::default(),
            name: Default::default(),
            normal_texture: Default::default(),
            occlusion_texture: Default::default(),
            pbr_metallic_roughness: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsAnisotropy {
    #[serde(rename = "anisotropyRotation")]
    pub anisotropy_rotation: Option<f64>,
    #[serde(rename = "anisotropyStrength")]
    pub anisotropy_strength: Option<f64>,
    #[serde(rename = "anisotropyTexture")]
    pub anisotropy_texture: Option<TextureInfo>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialKhrmaterialsAnisotropy {
    fn default() -> Self {
        Self {
            anisotropy_rotation: Some(0.0),
            anisotropy_strength: Some(0.0),
            anisotropy_texture: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsClearcoat {
    #[serde(rename = "clearcoatFactor")]
    pub clearcoat_factor: Option<f64>,
    #[serde(rename = "clearcoatNormalTexture")]
    pub clearcoat_normal_texture: Option<MaterialNormalTextureInfo>,
    #[serde(rename = "clearcoatRoughnessFactor")]
    pub clearcoat_roughness_factor: Option<f64>,
    #[serde(rename = "clearcoatRoughnessTexture")]
    pub clearcoat_roughness_texture: Option<TextureInfo>,
    #[serde(rename = "clearcoatTexture")]
    pub clearcoat_texture: Option<TextureInfo>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialKhrmaterialsClearcoat {
    fn default() -> Self {
        Self {
            clearcoat_factor: Some(0.0),
            clearcoat_normal_texture: Default::default(),
            clearcoat_roughness_factor: Some(0.0),
            clearcoat_roughness_texture: Default::default(),
            clearcoat_texture: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsDiffuseTransmission {
    #[serde(rename = "diffuseTransmissionColorFactor")]
    pub diffuse_transmission_color_factor: Option<Vec<f64>>,
    #[serde(rename = "diffuseTransmissionColorTexture")]
    pub diffuse_transmission_color_texture: Option<TextureInfo>,
    #[serde(rename = "diffuseTransmissionFactor")]
    pub diffuse_transmission_factor: Option<f64>,
    #[serde(rename = "diffuseTransmissionTexture")]
    pub diffuse_transmission_texture: Option<TextureInfo>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialKhrmaterialsDiffuseTransmission {
    fn default() -> Self {
        Self {
            diffuse_transmission_color_factor: Some(vec![1.0, 1.0, 1.0]),
            diffuse_transmission_color_texture: Default::default(),
            diffuse_transmission_factor: Some(0.0),
            diffuse_transmission_texture: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsDispersion {
    pub dispersion: Option<f64>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialKhrmaterialsDispersion {
    fn default() -> Self {
        Self {
            dispersion: Some(0.0),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsEmissiveStrength {
    #[serde(rename = "emissiveStrength")]
    pub emissive_strength: Option<f64>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialKhrmaterialsEmissiveStrength {
    fn default() -> Self {
        Self {
            emissive_strength: Some(1.0),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsIor {
    pub ior: Option<f64>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialKhrmaterialsIor {
    fn default() -> Self {
        Self {
            ior: Some(1.5),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsIridescence {
    #[serde(rename = "iridescenceFactor")]
    pub iridescence_factor: Option<f64>,
    #[serde(rename = "iridescenceIor")]
    pub iridescence_ior: Option<f64>,
    #[serde(rename = "iridescenceTexture")]
    pub iridescence_texture: Option<TextureInfo>,
    #[serde(rename = "iridescenceThicknessMaximum")]
    pub iridescence_thickness_maximum: Option<f64>,
    #[serde(rename = "iridescenceThicknessMinimum")]
    pub iridescence_thickness_minimum: Option<f64>,
    #[serde(rename = "iridescenceThicknessTexture")]
    pub iridescence_thickness_texture: Option<TextureInfo>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialKhrmaterialsIridescence {
    fn default() -> Self {
        Self {
            iridescence_factor: Some(0.0),
            iridescence_ior: Some(1.3),
            iridescence_texture: Default::default(),
            iridescence_thickness_maximum: Some(400.0),
            iridescence_thickness_minimum: Some(100.0),
            iridescence_thickness_texture: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsSheen {
    #[serde(rename = "sheenColorFactor")]
    pub sheen_color_factor: Option<Vec<f64>>,
    #[serde(rename = "sheenColorTexture")]
    pub sheen_color_texture: Option<TextureInfo>,
    #[serde(rename = "sheenRoughnessFactor")]
    pub sheen_roughness_factor: Option<f64>,
    #[serde(rename = "sheenRoughnessTexture")]
    pub sheen_roughness_texture: Option<TextureInfo>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialKhrmaterialsSheen {
    fn default() -> Self {
        Self {
            sheen_color_factor: Some(vec![0.0, 0.0, 0.0]),
            sheen_color_texture: Default::default(),
            sheen_roughness_factor: Some(0.0),
            sheen_roughness_texture: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsSpecular {
    #[serde(rename = "specularColorFactor")]
    pub specular_color_factor: Option<Vec<f64>>,
    #[serde(rename = "specularColorTexture")]
    pub specular_color_texture: Option<TextureInfo>,
    #[serde(rename = "specularFactor")]
    pub specular_factor: Option<f64>,
    #[serde(rename = "specularTexture")]
    pub specular_texture: Option<TextureInfo>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialKhrmaterialsSpecular {
    fn default() -> Self {
        Self {
            specular_color_factor: Some(vec![1.0, 1.0, 1.0]),
            specular_color_texture: Default::default(),
            specular_factor: Some(1.0),
            specular_texture: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsTransmission {
    #[serde(rename = "transmissionFactor")]
    pub transmission_factor: Option<f64>,
    #[serde(rename = "transmissionTexture")]
    pub transmission_texture: Option<TextureInfo>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialKhrmaterialsTransmission {
    fn default() -> Self {
        Self {
            transmission_factor: Some(0.0),
            transmission_texture: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsUnlit {
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialKhrmaterialsVolume {
    #[serde(rename = "attenuationColor")]
    pub attenuation_color: Option<Vec<f64>>,
    #[serde(rename = "attenuationDistance")]
    pub attenuation_distance: Option<f64>,
    #[serde(rename = "thicknessFactor")]
    pub thickness_factor: Option<f64>,
    #[serde(rename = "thicknessTexture")]
    pub thickness_texture: Option<TextureInfo>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialKhrmaterialsVolume {
    fn default() -> Self {
        Self {
            attenuation_color: Some(vec![1.0, 1.0, 1.0]),
            attenuation_distance: Default::default(),
            thickness_factor: Some(0.0),
            thickness_texture: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialNormalTextureInfo {
    pub index: Option<serde_json::Value>,
    pub scale: Option<f64>,
    #[serde(rename = "texCoord")]
    pub tex_coord: Option<serde_json::Value>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialNormalTextureInfo {
    fn default() -> Self {
        Self {
            index: Default::default(),
            scale: Some(1.0),
            tex_coord: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialOcclusionTextureInfo {
    pub index: Option<serde_json::Value>,
    pub strength: Option<f64>,
    #[serde(rename = "texCoord")]
    pub tex_coord: Option<serde_json::Value>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialOcclusionTextureInfo {
    fn default() -> Self {
        Self {
            index: Default::default(),
            strength: Some(1.0),
            tex_coord: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MaterialPbrMetallicRoughness {
    #[serde(rename = "baseColorFactor")]
    pub base_color_factor: Option<Vec<f64>>,
    #[serde(rename = "baseColorTexture")]
    pub base_color_texture: Option<TextureInfo>,
    #[serde(rename = "metallicFactor")]
    pub metallic_factor: Option<f64>,
    #[serde(rename = "metallicRoughnessTexture")]
    pub metallic_roughness_texture: Option<TextureInfo>,
    #[serde(rename = "roughnessFactor")]
    pub roughness_factor: Option<f64>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MaterialPbrMetallicRoughness {
    fn default() -> Self {
        Self {
            base_color_factor: Some(vec![1.0, 1.0, 1.0, 1.0]),
            base_color_texture: Default::default(),
            metallic_factor: Some(1.0),
            metallic_roughness_texture: Default::default(),
            roughness_factor: Some(1.0),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Mesh {
    pub name: Option<serde_json::Value>,
    pub primitives: Vec<MeshPrimitive>,
    pub weights: Option<Vec<f64>>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MeshPrimitive {
    pub attributes: HashMap<String, GltfIndex>,
    pub indices: Option<GltfIndex>,
    pub material: Option<GltfIndex>,
    pub mode: Option<serde_json::Value>,
    pub targets: Option<Vec<HashMap<String, GltfIndex>>>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for MeshPrimitive {
    fn default() -> Self {
        Self {
            attributes: Default::default(),
            indices: Default::default(),
            material: Default::default(),
            mode: Some(serde_json::json!(4)),
            targets: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MeshPrimitiveKhrdracoMeshCompression {
    pub attributes: HashMap<String, GltfIndex>,
    #[serde(rename = "bufferView")]
    pub buffer_view: GltfIndex,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MeshPrimitiveKhrmaterialsVariants {
    pub mappings: Vec<GltfProperty>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Node {
    pub camera: Option<GltfIndex>,
    pub children: Option<Vec<GltfIndex>>,
    pub matrix: Option<Vec<f64>>,
    pub mesh: Option<GltfIndex>,
    pub name: Option<serde_json::Value>,
    pub rotation: Option<Vec<f64>>,
    pub scale: Option<Vec<f64>>,
    pub skin: Option<GltfIndex>,
    pub translation: Option<Vec<f64>>,
    pub weights: Option<Vec<f64>>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            camera: Default::default(),
            children: Default::default(),
            matrix: Some(vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]),
            mesh: Default::default(),
            name: Default::default(),
            rotation: Some(vec![0.0, 0.0, 0.0, 1.0]),
            scale: Some(vec![1.0, 1.0, 1.0]),
            skin: Default::default(),
            translation: Some(vec![0.0, 0.0, 0.0]),
            weights: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct NodeKhrlightsPunctual {
    pub light: GltfIndex,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct NodeKhrnodeVisibility {
    pub visible: Option<bool>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for NodeKhrnodeVisibility {
    fn default() -> Self {
        Self {
            visible: Some(true),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Sampler {
    #[serde(rename = "magFilter")]
    pub mag_filter: Option<serde_json::Value>,
    #[serde(rename = "minFilter")]
    pub min_filter: Option<serde_json::Value>,
    pub name: Option<serde_json::Value>,
    #[serde(rename = "wrapS")]
    pub wrap_s: Option<serde_json::Value>,
    #[serde(rename = "wrapT")]
    pub wrap_t: Option<serde_json::Value>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            mag_filter: Default::default(),
            min_filter: Default::default(),
            name: Default::default(),
            wrap_s: Some(serde_json::json!(10497)),
            wrap_t: Some(serde_json::json!(10497)),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Scene {
    pub name: Option<serde_json::Value>,
    pub nodes: Option<Vec<GltfIndex>>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Skin {
    #[serde(rename = "inverseBindMatrices")]
    pub inverse_bind_matrices: Option<GltfIndex>,
    pub joints: Vec<GltfIndex>,
    pub name: Option<serde_json::Value>,
    pub skeleton: Option<GltfIndex>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Texture {
    pub name: Option<serde_json::Value>,
    pub sampler: Option<GltfIndex>,
    pub source: Option<GltfIndex>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TextureInfo {
    pub index: GltfIndex,
    #[serde(rename = "texCoord")]
    pub tex_coord: Option<i64>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for TextureInfo {
    fn default() -> Self {
        Self {
            index: Default::default(),
            tex_coord: Some(0),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TextureInfoKhrtextureTransform {
    pub offset: Option<Vec<f64>>,
    pub rotation: Option<f64>,
    pub scale: Option<Vec<f64>>,
    #[serde(rename = "texCoord")]
    pub tex_coord: Option<i64>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

impl Default for TextureInfoKhrtextureTransform {
    fn default() -> Self {
        Self {
            offset: Some(vec![0.0, 0.0]),
            rotation: Some(0.0),
            scale: Some(vec![1.0, 1.0]),
            tex_coord: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TextureKhrtextureBasisu {
    pub source: Option<GltfIndex>,
    pub extensions: Option<HashMap<String, serde_json::Value>>,
    pub extras: Option<serde_json::Value>,
}

// Type aliases for extension convenience
pub type KhrLightsPunctual = GltfKhrlightsPunctual;
pub type GltfLight = Light;
pub type GltfScene = Scene;
pub type GltfAnimation = Animation;
pub type GltfCamera = Camera;
