use nalgebra::{Vector2, Vector3, Vector4};
use serde::{Deserialize, Serialize};

use crate::exchange::FileType;
use crate::image::LazyImage;

pub type UV = Vec<Vector2<f64>>;
pub type MaterialIndices = Vec<usize>;
pub type GroupingIndices = Vec<usize>;
pub type Color = Vec<Vector4<u8>>;
pub type Normal = Vec<Vector3<f64>>;
pub type Tangent = Vec<Vector4<f64>>;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum GroupingKind {
    #[default]
    Unspecified,
    Material,
    Group,
    Smoothing,
    Object,
    /// Per-face surface index; geometry lives in `Trimesh.face_surfaces`.
    Surface,
}

/// Sentinel value meaning "this face has no group assignment".
pub const UNSET: usize = usize::MAX;

/// Per-face grouping attribute with a lookup table of names.
/// Each face has an index into `names` stored in `indices`.
/// Use [`UNSET`] for faces that have no assignment.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Grouping {
    pub kind: GroupingKind,
    pub names: Vec<String>,
    pub indices: Vec<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Attributes {
    pub uv: Vec<UV>,
    pub normals: Vec<Normal>,
    pub colors: Vec<Color>,
    pub tangents: Vec<Tangent>,
    pub groupings: Vec<Grouping>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoadSource {
    // what format was this mesh loaded from?
    pub format: Option<FileType>,

    // many formats have a header which would otherwise be discarded
    pub header: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SimpleMaterial {
    pub name: String,
    pub diffuse: Option<Vector3<f64>>,
    pub specular: Option<Vector3<f64>>,
    pub shininess: Option<f64>,
    pub alpha: Option<f64>,
    pub diffuse_texture: Option<LazyImage>,
}

/// Alpha blending mode for materials.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlphaMode {
    /// Fully opaque, alpha channel is ignored.
    #[default]
    Opaque,
    /// Either fully opaque or fully transparent based on alpha_cutoff.
    Mask,
    /// Alpha blending is enabled.
    Blend,
}

/// PBR (Physically Based Rendering) metallic-roughness material.
/// Based on the glTF 2.0 material specification.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PBRMaterial {
    /// Material name for identification.
    pub name: String,
    /// Base color factor (RGBA), multiplied with base_color_texture if present.
    pub base_color_factor: Vector4<f64>,
    /// Base color texture (RGBA).
    pub base_color_texture: Option<LazyImage>,
    /// Metallic factor (0.0 = dielectric, 1.0 = metal).
    pub metallic_factor: f64,
    /// Roughness factor (0.0 = smooth, 1.0 = rough).
    pub roughness_factor: f64,
    /// Combined metallic-roughness texture (B=metallic, G=roughness).
    pub metallic_roughness_texture: Option<LazyImage>,
    /// Normal map texture (RGB tangent-space normal).
    pub normal_texture: Option<LazyImage>,
    /// Scale for normal map (default 1.0).
    pub normal_scale: f64,
    /// Ambient occlusion texture (R channel).
    pub occlusion_texture: Option<LazyImage>,
    /// Strength of occlusion effect (0.0-1.0, default 1.0).
    pub occlusion_strength: f64,
    /// Emissive color factor (RGB).
    pub emissive_factor: Vector3<f64>,
    /// Emissive texture (RGB).
    pub emissive_texture: Option<LazyImage>,
    /// Alpha blending mode.
    pub alpha_mode: AlphaMode,
    /// Alpha cutoff threshold for Mask mode (default 0.5).
    pub alpha_cutoff: f64,
    /// Whether the material is double-sided.
    pub double_sided: bool,
}

impl SimpleMaterial {
    /// Convert to a PBR material, mapping diffuse → base_color.
    pub fn to_pbr(&self) -> PBRMaterial {
        let d = self.diffuse.unwrap_or(Vector3::new(0.4, 0.4, 0.4));
        let a = self.alpha.unwrap_or(1.0);
        let roughness = self
            .shininess
            .map_or(0.5, |s| 1.0 - (s / 100.0).clamp(0.0, 1.0));
        PBRMaterial {
            name: self.name.clone(),
            base_color_factor: Vector4::new(d.x, d.y, d.z, a),
            base_color_texture: self.diffuse_texture.clone(),
            metallic_factor: 0.0,
            roughness_factor: roughness,
            ..PBRMaterial::new()
        }
    }
}

impl PBRMaterial {
    /// Create a new PBRMaterial with default values matching glTF spec.
    pub fn new() -> Self {
        Self {
            name: String::new(),
            base_color_factor: Vector4::new(1.0, 1.0, 1.0, 1.0),
            base_color_texture: None,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive_factor: Vector3::zeros(),
            emissive_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmptyMaterial {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Material {
    Empty(EmptyMaterial),
    Simple(SimpleMaterial),
    PBR(Box<PBRMaterial>),
}

impl Material {
    /// Get the name of the material.
    pub fn name(&self) -> &str {
        match self {
            Material::Empty(_) => "",
            Material::Simple(m) => &m.name,
            Material::PBR(m) => &m.name,
        }
    }

    /// Convert any material variant to PBR.
    pub fn to_pbr(&self) -> PBRMaterial {
        match self {
            Material::Empty(_) => PBRMaterial::new(),
            Material::Simple(m) => m.to_pbr(),
            Material::PBR(m) => m.as_ref().clone(),
        }
    }
}

pub const DEFAULT_COLOR: Vector4<u8> = Vector4::new(100, 100, 100, 255);
