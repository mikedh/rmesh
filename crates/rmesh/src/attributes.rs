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

impl Grouping {
    /// Extract a subset of this grouping by the given element indices.
    ///
    /// Compacts the name table to only include referenced names.
    /// UNSET indices remain UNSET.
    #[must_use]
    pub fn subset(&self, indices: &[usize]) -> Self {
        if indices.is_empty() {
            return Self {
                kind: self.kind.clone(),
                names: Vec::new(),
                indices: Vec::new(),
            };
        }

        // Extract the raw index values at the requested positions
        let raw: Vec<usize> = indices.iter().map(|&i| self.indices[i]).collect();

        // Find distinct non-UNSET name indices that appear
        let mut seen = vec![false; self.names.len()];
        for &idx in &raw {
            if idx != UNSET && idx < self.names.len() {
                seen[idx] = true;
            }
        }

        // Build old→new remapping and compact names
        let mut remap = vec![UNSET; self.names.len()];
        let mut names = Vec::new();
        for (old, &used) in seen.iter().enumerate() {
            if used {
                remap[old] = names.len();
                names.push(self.names[old].clone());
            }
        }

        // Remap indices
        let new_indices = raw
            .iter()
            .map(|&idx| {
                if idx == UNSET || idx >= self.names.len() {
                    UNSET
                } else {
                    remap[idx]
                }
            })
            .collect();

        Self {
            kind: self.kind.clone(),
            names,
            indices: new_indices,
        }
    }
}

impl Attributes {
    /// Concatenate multiple `Attributes` in order.
    ///
    /// Each `(attributes, count)` pair specifies the attribute data and its
    /// element count (number of vertices or faces). When a channel exists in
    /// some inputs but not others, missing inputs are padded with defaults
    /// (`Vector2::zeros` for UVs, `Vector3::zeros` for normals,
    /// `DEFAULT_COLOR` for colors, `Vector4::zeros` for tangents, `UNSET`
    /// for grouping indices).
    ///
    /// Groupings are merged by kind: groupings of the same kind across
    /// inputs are combined (names unified, indices offset). Inputs without
    /// a grouping that others have are padded with `UNSET`.
    pub fn concatenate(slices: &[(&Attributes, usize)]) -> Self {
        if slices.is_empty() {
            return Self::default();
        }

        macro_rules! concat_channels {
            ($field:ident, $default:expr) => {{
                let max_channels = slices
                    .iter()
                    .map(|(a, _)| a.$field.len())
                    .max()
                    .unwrap_or(0);
                (0..max_channels)
                    .map(|ch| {
                        let mut combined = Vec::new();
                        for &(attrs, count) in slices {
                            if ch < attrs.$field.len() {
                                combined.extend_from_slice(&attrs.$field[ch]);
                            } else {
                                combined.resize(combined.len() + count, $default);
                            }
                        }
                        combined
                    })
                    .collect()
            }};
        }

        // Collect all distinct grouping kinds across all inputs
        let mut all_kinds: Vec<GroupingKind> = Vec::new();
        for (attrs, _) in slices {
            for g in &attrs.groupings {
                if !all_kinds.contains(&g.kind) {
                    all_kinds.push(g.kind.clone());
                }
            }
        }

        // Merge groupings by kind, padding missing inputs with UNSET
        let groupings: Vec<Grouping> = all_kinds
            .into_iter()
            .map(|kind| {
                let mut names: Vec<String> = Vec::new();
                let mut indices: Vec<usize> = Vec::new();

                for &(attrs, count) in slices {
                    if let Some(g) = attrs.groupings.iter().find(|g| g.kind == kind) {
                        let name_offset = names.len();
                        names.extend_from_slice(&g.names);
                        for &idx in &g.indices {
                            if idx == UNSET {
                                indices.push(UNSET);
                            } else {
                                indices.push(idx + name_offset);
                            }
                        }
                    } else {
                        // This input has no grouping of this kind — pad with UNSET
                        indices.resize(indices.len() + count, UNSET);
                    }
                }

                Grouping {
                    kind,
                    names,
                    indices,
                }
            })
            .collect();

        Self {
            uv: concat_channels!(uv, Vector2::zeros()),
            normals: concat_channels!(normals, Vector3::zeros()),
            colors: concat_channels!(colors, DEFAULT_COLOR),
            tangents: concat_channels!(tangents, Vector4::zeros()),
            groupings,
        }
    }

    /// Extract a subset of all attribute channels at the given element indices.
    #[must_use]
    pub fn subset(&self, indices: &[usize]) -> Self {
        macro_rules! subset_channels {
            ($channels:expr) => {
                $channels
                    .iter()
                    .map(|channel| indices.iter().map(|&i| channel[i]).collect())
                    .collect()
            };
        }
        Self {
            uv: subset_channels!(self.uv),
            normals: subset_channels!(self.normals),
            colors: subset_channels!(self.colors),
            tangents: subset_channels!(self.tangents),
            groupings: self.groupings.iter().map(|g| g.subset(indices)).collect(),
        }
    }
}

pub const DEFAULT_COLOR: Vector4<u8> = Vector4::new(100, 100, 100, 255);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grouping_subset_basic() {
        let g = Grouping {
            kind: GroupingKind::Material,
            names: vec!["red".into(), "blue".into(), "green".into()],
            indices: vec![0, 1, 2, 0, 1],
        };

        // Take faces 1, 3 (blue, red) → names compacted to [red, blue]
        let sub = g.subset(&[1, 3]);
        assert_eq!(sub.kind, GroupingKind::Material);
        assert_eq!(sub.names.len(), 2);
        assert!(sub.names.contains(&"red".to_string()));
        assert!(sub.names.contains(&"blue".to_string()));
        assert_eq!(sub.indices.len(), 2);
        // Both indices should be valid
        assert!(sub.indices[0] < sub.names.len());
        assert!(sub.indices[1] < sub.names.len());
    }

    #[test]
    fn test_grouping_subset_with_unset() {
        let g = Grouping {
            kind: GroupingKind::Group,
            names: vec!["a".into(), "b".into()],
            indices: vec![0, UNSET, 1],
        };

        let sub = g.subset(&[0, 1, 2]);
        assert_eq!(sub.indices[1], UNSET);
        assert!(sub.indices[0] != UNSET);
        assert!(sub.indices[2] != UNSET);
    }

    #[test]
    fn test_grouping_subset_empty() {
        let g = Grouping {
            kind: GroupingKind::Material,
            names: vec!["red".into()],
            indices: vec![0, 0],
        };
        let sub = g.subset(&[]);
        assert!(sub.names.is_empty());
        assert!(sub.indices.is_empty());
    }

    #[test]
    fn test_attributes_subset() {
        let attrs = Attributes {
            colors: vec![vec![
                Vector4::new(255, 0, 0, 255),
                Vector4::new(0, 255, 0, 255),
                Vector4::new(0, 0, 255, 255),
            ]],
            normals: vec![vec![
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ]],
            ..Default::default()
        };

        let sub = attrs.subset(&[0, 2]);
        assert_eq!(sub.colors[0].len(), 2);
        assert_eq!(sub.colors[0][0], Vector4::new(255, 0, 0, 255));
        assert_eq!(sub.colors[0][1], Vector4::new(0, 0, 255, 255));
        assert_eq!(sub.normals[0].len(), 2);
        assert_eq!(sub.normals[0][0], Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(sub.normals[0][1], Vector3::new(0.0, 0.0, 1.0));
    }
}
