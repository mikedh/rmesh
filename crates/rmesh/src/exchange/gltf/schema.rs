//! glTF 2.0 schema - AUTO-GENERATED from JSON Schema
//! Do not edit manually. Run `cargo run -p codegen --bin gltf-codegen` to regenerate.

#![allow(unused_imports)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::doc_lazy_continuation)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Index into a glTF array (accessor, buffer, node, etc).
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

pub fn accessor_type_count(t: &AccessorType) -> usize {
    t.component_count()
}

/// Error types.
pub mod error {
    /// Error from a `TryFrom` or `FromStr` implementation.
    pub struct ConversionError(std::borrow::Cow<'static, str>);
    impl std::error::Error for ConversionError {}
    impl std::fmt::Display for ConversionError {
        fn fmt(
            &self,
            f: &mut std::fmt::Formatter<'_>,
        ) -> Result<(), std::fmt::Error> {
            std::fmt::Display::fmt(&self.0, f)
        }
    }
    impl std::fmt::Debug for ConversionError {
        fn fmt(
            &self,
            f: &mut std::fmt::Formatter<'_>,
        ) -> Result<(), std::fmt::Error> {
            std::fmt::Debug::fmt(&self.0, f)
        }
    }
    impl From<&'static str> for ConversionError {
        fn from(value: &'static str) -> Self {
            Self(value.into())
        }
    }
    impl From<String> for ConversionError {
        fn from(value: String) -> Self {
            Self(value.into())
        }
    }
}
///A typed view into a buffer view that contains raw binary data.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Accessor {
    ///The index of the bufferView.
    #[serde(
        rename = "bufferView",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub buffer_view: Option<u64>,
    ///The offset relative to the start of the buffer view in bytes.
    #[serde(rename = "byteOffset", default)]
    pub byte_offset: u64,
    ///The datatype of the accessor's components.
    #[serde(rename = "componentType")]
    pub component_type: u32,
    ///The number of elements referenced by this accessor.
    pub count: u64,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///Maximum value of each component in this accessor.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub max: Vec<f64>,
    ///Minimum value of each component in this accessor.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub min: Vec<f64>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///Specifies whether integer data values are normalized before usage.
    #[serde(default)]
    pub normalized: bool,
    ///Sparse storage of elements that deviate from their initialization value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse: Option<AccessorSparse>,
    ///Specifies if the accessor's elements are scalars, vectors, or matrices.
    #[serde(rename = "type")]
    pub type_: AccessorType,
}
///Sparse storage of accessor values that deviate from their initialization value.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct AccessorSparse {
    ///Number of deviating accessor values stored in the sparse array.
    pub count: u64,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///An object pointing to a buffer view containing the indices of deviating accessor values. The number of indices is equal to `count`. Indices **MUST** strictly increase.
    pub indices: AccessorSparseIndices,
    ///An object pointing to a buffer view containing the deviating accessor values.
    pub values: AccessorSparseValues,
}
///An object pointing to a buffer view containing the indices of deviating accessor values. The number of indices is equal to `accessor.sparse.count`. Indices **MUST** strictly increase.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct AccessorSparseIndices {
    ///The index of the buffer view with sparse indices. The referenced buffer view **MUST NOT** have its `target` or `byteStride` properties defined. The buffer view and the optional `byteOffset` **MUST** be aligned to the `componentType` byte length.
    #[serde(rename = "bufferView")]
    pub buffer_view: u64,
    ///The offset relative to the start of the buffer view in bytes.
    #[serde(rename = "byteOffset", default)]
    pub byte_offset: u64,
    ///The indices data type.
    #[serde(rename = "componentType")]
    pub component_type: u32,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
}
///An object pointing to a buffer view containing the deviating accessor values. The number of elements is equal to `accessor.sparse.count` times number of components. The elements have the same component type as the base accessor. The elements are tightly packed. Data **MUST** be aligned following the same rules as the base accessor.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct AccessorSparseValues {
    ///The index of the bufferView with sparse values. The referenced buffer view **MUST NOT** have its `target` or `byteStride` properties defined.
    #[serde(rename = "bufferView")]
    pub buffer_view: u64,
    ///The offset relative to the start of the bufferView in bytes.
    #[serde(rename = "byteOffset", default)]
    pub byte_offset: u64,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
}
///A keyframe animation.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Animation {
    ///An array of animation channels. An animation channel combines an animation sampler with a target property being animated. Different channels of the same animation **MUST NOT** have the same targets.
    pub channels: Vec<AnimationChannel>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///An array of animation samplers. An animation sampler combines timestamps with a sequence of output values and defines an interpolation algorithm.
    pub samplers: Vec<AnimationSampler>,
}
///An animation channel combines an animation sampler with a target property being animated.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct AnimationChannel {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The index of a sampler in this animation used to compute the value for the target.
    pub sampler: u64,
    ///The descriptor of the animated property.
    pub target: AnimationChannelTarget,
}
///The descriptor of the animated property.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct AnimationChannelTarget {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The index of the node to animate. When undefined, the animated object **MAY** be defined by an extension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<u64>,
    ///The name of the node's TRS property to animate, or the `"weights"` of the Morph Targets it instantiates. For the `"translation"` property, the values that are provided by the sampler are the translation along the X, Y, and Z axes. For the `"rotation"` property, the values are a quaternion in the order (x, y, z, w), where w is the scalar. For the `"scale"` property, the values are the scaling factors along the X, Y, and Z axes.
    pub path: GltfAnimationPath,
}
///An animation sampler combines timestamps with a sequence of output values and defines an interpolation algorithm.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct AnimationSampler {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The index of an accessor containing keyframe timestamps.
    pub input: u64,
    ///Interpolation algorithm.
    #[serde(default = "defaults::animation_sampler_interpolation")]
    pub interpolation: GltfInterpolation,
    ///The index of an accessor, containing keyframe output values.
    pub output: u64,
}
///Metadata about the glTF asset.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Asset {
    ///A copyright message suitable for display to credit the content creator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copyright: Option<String>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///Tool that generated this glTF model.  Useful for debugging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator: Option<String>,
    ///The minimum glTF version in the form of `<major>.<minor>` that this asset targets. This property **MUST NOT** be greater than the asset version.
    #[serde(
        rename = "minVersion",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub min_version: Option<String>,
    ///The glTF version in the form of `<major>.<minor>` that this asset targets.
    pub version: String,
}
///A buffer points to binary geometry, animation, or skins.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Buffer {
    ///The length of the buffer in bytes.
    #[serde(rename = "byteLength")]
    pub byte_length: u64,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///The URI (or IRI) of the buffer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}
///A view into a buffer generally representing a subset of the buffer.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct BufferView {
    ///The index of the buffer.
    pub buffer: u64,
    ///The length of the bufferView in bytes.
    #[serde(rename = "byteLength")]
    pub byte_length: u64,
    ///The offset into the buffer in bytes.
    #[serde(rename = "byteOffset", default)]
    pub byte_offset: u64,
    ///The stride, in bytes.
    #[serde(
        rename = "byteStride",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub byte_stride: Option<u32>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///The hint representing the intended GPU buffer type to use with this buffer view.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<u32>,
}
///A camera's projection.  A node **MAY** reference a camera to apply a transform to place the camera in the scene.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Camera {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///An orthographic camera containing properties to create an orthographic projection matrix. This property **MUST NOT** be defined when `perspective` is defined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orthographic: Option<CameraOrthographic>,
    ///A perspective camera containing properties to create a perspective projection matrix. This property **MUST NOT** be defined when `orthographic` is defined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perspective: Option<CameraPerspective>,
    ///Specifies if the camera uses a perspective or orthographic projection.
    #[serde(rename = "type")]
    pub type_: CameraType,
}
///An orthographic camera containing properties to create an orthographic projection matrix.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct CameraOrthographic {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    pub xmag: f64,
    pub ymag: f64,
    pub zfar: f64,
    pub znear: f64,
}
///A perspective camera containing properties to create a perspective projection matrix.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct CameraPerspective {
    #[serde(
        rename = "aspectRatio",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub aspect_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    pub yfov: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zfar: Option<f64>,
    pub znear: f64,
}
///The root object for a glTF asset.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct GlTf {
    ///An array of accessors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accessors: Vec<Accessor>,
    ///An array of keyframe animations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub animations: Vec<Animation>,
    ///Metadata about the glTF asset.
    pub asset: Asset,
    ///An array of bufferViews.
    #[serde(
        rename = "bufferViews",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub buffer_views: Vec<BufferView>,
    ///An array of buffers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub buffers: Vec<Buffer>,
    ///An array of cameras.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cameras: Vec<Camera>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    ///Names of glTF extensions required to properly load this asset.
    #[serde(
        rename = "extensionsRequired",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub extensions_required: Option<Vec<String>>,
    ///Names of glTF extensions used in this asset.
    #[serde(
        rename = "extensionsUsed",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub extensions_used: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///An array of images.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Image>,
    ///An array of materials.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<Material>,
    ///An array of meshes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub meshes: Vec<Mesh>,
    ///An array of nodes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<Node>,
    ///An array of samplers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub samplers: Vec<Sampler>,
    ///The index of the default scene.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene: Option<u64>,
    ///An array of scenes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scenes: Vec<Scene>,
    ///An array of skins.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skins: Vec<Skin>,
    ///An array of textures.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub textures: Vec<Texture>,
}
///`GlTfKhrLightsPunctual`
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct GlTfKhrLightsPunctual {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    pub lights: Vec<Light>,
}
///Image data used to create a texture. Image **MAY** be referenced by an URI (or IRI) or a buffer view index.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Image {
    ///The index of the bufferView that contains the image. This field **MUST NOT** be defined when `uri` is defined.
    #[serde(
        rename = "bufferView",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub buffer_view: Option<u64>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The image's media type. This field **MUST** be defined when `bufferView` is defined.
    #[serde(
        rename = "mimeType",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub mime_type: Option<String>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///The URI (or IRI) of the image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}
///A directional, point, or spot light.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Light {
    ///Color of the light source.
    #[serde(default = "defaults::light_color")]
    pub color: [f64; 3usize],
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///Intensity of the light source.
    #[serde(default = "defaults::light_intensity")]
    pub intensity: f64,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spot: Option<LightSpot>,
    ///Specifies the light type.
    #[serde(rename = "type")]
    pub type_: GltfLightType,
}
///`LightSpot`
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct LightSpot {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///Angle in radians from centre of spotlight where falloff begins.
    #[serde(rename = "innerConeAngle", default = "defaults::light_spot_inner_cone_angle")]
    pub inner_cone_angle: f64,
    ///Angle in radians from centre of spotlight where falloff ends.
    #[serde(rename = "outerConeAngle", default = "defaults::light_spot_outer_cone_angle")]
    pub outer_cone_angle: f64,
}
///The material appearance of a primitive.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Material {
    ///The alpha cutoff value of the material.
    #[serde(rename = "alphaCutoff", default = "defaults::material_alpha_cutoff")]
    pub alpha_cutoff: f64,
    ///The alpha rendering mode of the material.
    #[serde(rename = "alphaMode", default = "defaults::material_alpha_mode")]
    pub alpha_mode: GltfAlphaMode,
    ///Specifies whether the material is double sided.
    #[serde(rename = "doubleSided", default)]
    pub double_sided: bool,
    ///The factors for the emissive color of the material.
    #[serde(rename = "emissiveFactor", default = "defaults::material_emissive_factor")]
    pub emissive_factor: [f64; 3usize],
    ///The emissive texture.
    #[serde(
        rename = "emissiveTexture",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub emissive_texture: Option<TextureInfo>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///The tangent space normal texture.
    #[serde(
        rename = "normalTexture",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub normal_texture: Option<MaterialNormalTextureInfo>,
    ///The occlusion texture.
    #[serde(
        rename = "occlusionTexture",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub occlusion_texture: Option<MaterialOcclusionTextureInfo>,
    ///A set of parameter values that are used to define the metallic-roughness material model from Physically Based Rendering (PBR) methodology. When undefined, all the default values of `pbrMetallicRoughness` **MUST** apply.
    #[serde(
        rename = "pbrMetallicRoughness",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub pbr_metallic_roughness: Option<MaterialPbrMetallicRoughness>,
}
///`MaterialNormalTextureInfo`
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct MaterialNormalTextureInfo {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The index of the texture.
    pub index: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    ///The set index of texture's TEXCOORD attribute used for texture coordinate mapping.
    #[serde(rename = "texCoord", default)]
    pub tex_coord: u64,
}
///`MaterialOcclusionTextureInfo`
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct MaterialOcclusionTextureInfo {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The index of the texture.
    pub index: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strength: Option<f64>,
    ///The set index of texture's TEXCOORD attribute used for texture coordinate mapping.
    #[serde(rename = "texCoord", default)]
    pub tex_coord: u64,
}
///A set of parameter values that are used to define the metallic-roughness material model from Physically-Based Rendering (PBR) methodology.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct MaterialPbrMetallicRoughness {
    ///The factors for the base color of the material.
    #[serde(
        rename = "baseColorFactor",
        default = "defaults::material_pbr_metallic_roughness_base_color_factor"
    )]
    pub base_color_factor: [f64; 4usize],
    ///The base color texture.
    #[serde(
        rename = "baseColorTexture",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub base_color_texture: Option<TextureInfo>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The factor for the metalness of the material.
    #[serde(rename = "metallicFactor", default = "defaults::pbr_metallic_factor")]
    pub metallic_factor: f64,
    ///The metallic-roughness texture.
    #[serde(
        rename = "metallicRoughnessTexture",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub metallic_roughness_texture: Option<TextureInfo>,
    ///The factor for the roughness of the material.
    #[serde(rename = "roughnessFactor", default = "defaults::pbr_roughness_factor")]
    pub roughness_factor: f64,
}
///A set of primitives to be rendered.  Its global transform is defined by a node that references it.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Mesh {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///An array of primitives, each defining geometry to be rendered.
    pub primitives: Vec<MeshPrimitive>,
    ///Array of weights to be applied to the morph targets. The number of array elements **MUST** match the number of morph targets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub weights: Vec<f64>,
}
///Geometry to be rendered with the given material.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct MeshPrimitive {
    ///A plain JSON object, where each key corresponds to a mesh attribute semantic and each value is the index of the accessor containing attribute's data.
    pub attributes: HashMap<String, u64>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The index of the accessor that contains the vertex indices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indices: Option<u64>,
    ///The index of the material to apply to this primitive when rendering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<u64>,
    ///The topology type of primitives to render.
    #[serde(default = "defaults::default_u64::<u32, 4>")]
    pub mode: u32,
    ///An array of morph targets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<
        HashMap<String, u64>,
    >,
}
///A node in the node hierarchy.  When the node contains `skin`, all `mesh.primitives` **MUST** contain `JOINTS_0` and `WEIGHTS_0` attributes.  A node **MAY** have either a `matrix` or any combination of `translation`/`rotation`/`scale` (TRS) properties. TRS properties are converted to matrices and postmultiplied in the `T * R * S` order to compose the transformation matrix; first the scale is applied to the vertices, then the rotation, and then the translation. If none are provided, the transform is the identity. When a node is targeted for animation (referenced by an animation.channel.target), `matrix` **MUST NOT** be present.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Node {
    ///The index of the camera referenced by this node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<u64>,
    ///The indices of this node's children.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///A floating-point 4x4 transformation matrix stored in column-major order.
    #[serde(default = "defaults::node_matrix")]
    pub matrix: [f64; 16usize],
    ///The index of the mesh in this node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<u64>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///The node's unit quaternion rotation in the order (x, y, z, w), where w is the scalar.
    #[serde(default = "defaults::node_rotation")]
    pub rotation: [f64; 4usize],
    ///The node's non-uniform scale, given as the scaling factors along the x, y, and z axes.
    #[serde(default = "defaults::node_scale")]
    pub scale: [f64; 3usize],
    ///The index of the skin referenced by this node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skin: Option<u64>,
    ///The node's translation along the x, y, and z axes.
    #[serde(default = "defaults::node_translation")]
    pub translation: [f64; 3usize],
    ///The weights of the instantiated morph target. The number of array elements **MUST** match the number of morph targets of the referenced mesh. When defined, `mesh` **MUST** also be defined.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub weights: Vec<f64>,
}
///`NodeKhrLightsPunctual`
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct NodeKhrLightsPunctual {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The id of the light referenced by this node.
    pub light: u64,
}
///Texture sampler properties for filtering and wrapping modes.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Sampler {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///Magnification filter.
    #[serde(
        rename = "magFilter",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub mag_filter: Option<u32>,
    ///Minification filter.
    #[serde(
        rename = "minFilter",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub min_filter: Option<u32>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///S (U) wrapping mode.
    #[serde(rename = "wrapS", default = "defaults::default_u64::<u32, 10497>")]
    pub wrap_s: u32,
    ///T (V) wrapping mode.
    #[serde(rename = "wrapT", default = "defaults::default_u64::<u32, 10497>")]
    pub wrap_t: u32,
}
///The root nodes of a scene.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Scene {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///The indices of each root node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes: Option<Vec<u64>>,
}
///Joints and matrices defining a skin.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Skin {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The index of the accessor containing the floating-point 4x4 inverse-bind matrices.
    #[serde(
        rename = "inverseBindMatrices",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub inverse_bind_matrices: Option<u64>,
    ///Indices of skeleton nodes, used as joints in this skin.
    pub joints: Vec<u64>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///The index of the node used as a skeleton root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skeleton: Option<u64>,
}
///A texture and its sampler.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Texture {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The user-defined name of this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    ///The index of the sampler used by this texture. When undefined, a sampler with repeat wrapping and auto filtering **SHOULD** be used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<u64>,
    ///The index of the image used by this texture. When undefined, an extension or other mechanism **SHOULD** supply an alternate texture source, otherwise behavior is undefined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<u64>,
}
///Reference to a texture.
///
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct TextureInfo {
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extensions: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
    ///The index of the texture.
    pub index: u64,
    ///The set index of texture's TEXCOORD attribute used for texture coordinate mapping.
    #[serde(rename = "texCoord", default)]
    pub tex_coord: u64,
}
/// Generation of default values for serde.
pub mod defaults {
    pub(super) fn default_u64<T, const V: u64>() -> T
    where
        T: std::convert::TryFrom<u64>,
        <T as std::convert::TryFrom<u64>>::Error: std::fmt::Debug,
    {
        T::try_from(V).unwrap()
    }
    pub(super) fn animation_sampler_interpolation() -> super::GltfInterpolation {
        super::GltfInterpolation::LINEAR
    }
    pub(super) fn light_color() -> [f64; 3usize] {
        [1_f64, 1_f64, 1_f64]
    }
    pub(super) fn material_alpha_mode() -> super::GltfAlphaMode {
        super::GltfAlphaMode::OPAQUE
    }
    pub(super) fn material_emissive_factor() -> [f64; 3usize] {
        [0.0_f64, 0.0_f64, 0.0_f64]
    }
    pub(super) fn material_pbr_metallic_roughness_base_color_factor() -> [f64; 4usize] {
        [1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64]
    }
    pub(super) fn node_matrix() -> [f64; 16usize] {
        [
            1.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            1.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            1.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            1.0_f64,
        ]
    }
    pub(super) fn node_rotation() -> [f64; 4usize] {
        [0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]
    }
    pub(super) fn node_scale() -> [f64; 3usize] {
        [1.0_f64, 1.0_f64, 1.0_f64]
    }
    pub(super) fn node_translation() -> [f64; 3usize] {
        [0.0_f64, 0.0_f64, 0.0_f64]
    }
    pub(super) fn material_alpha_cutoff() -> f64 {
        0.5
    }
    pub(super) fn pbr_metallic_factor() -> f64 {
        1.0
    }
    pub(super) fn pbr_roughness_factor() -> f64 {
        1.0
    }
    pub(super) fn light_intensity() -> f64 {
        1.0
    }
    pub(super) fn light_spot_inner_cone_angle() -> f64 {
        0.0
    }
    pub(super) fn light_spot_outer_cone_angle() -> f64 {
        std::f64::consts::FRAC_PI_4
    }
}


// Type aliases for extension convenience
pub type KhrLightsPunctual = GlTfKhrLightsPunctual;
pub type GltfLight = Light;
pub type GltfScene = Scene;
pub type GltfAnimation = Animation;
pub type GltfCamera = Camera;

/// Accessor element type (SCALAR, VEC2, VEC3, VEC4, MAT2, MAT3, MAT4).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessorType {
    #[default]
    SCALAR,
    VEC2,
    VEC3,
    VEC4,
    MAT2,
    MAT3,
    MAT4,
}

impl AccessorType {
    /// Number of components for this accessor type.
    pub fn component_count(&self) -> usize {
        match self {
            Self::SCALAR => 1,
            Self::VEC2 => 2,
            Self::VEC3 => 3,
            Self::VEC4 => 4,
            Self::MAT2 => 4,
            Self::MAT3 => 9,
            Self::MAT4 => 16,
        }
    }
}

/// Camera projection type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CameraType {
    #[default]
    Perspective,
    Orthographic,
}

/// Light type (KHR_lights_punctual).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GltfLightType {
    #[default]
    Point,
    Directional,
    Spot,
}

/// Material alpha rendering mode.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GltfAlphaMode {
    #[default]
    OPAQUE,
    MASK,
    BLEND,
}

/// Animation sampler interpolation algorithm.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GltfInterpolation {
    #[default]
    LINEAR,
    STEP,
    CUBICSPLINE,
}

/// Animation channel target path.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GltfAnimationPath {
    #[default]
    Translation,
    Rotation,
    Scale,
    Weights,
}
