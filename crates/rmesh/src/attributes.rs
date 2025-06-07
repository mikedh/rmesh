use image::DynamicImage;
use nalgebra::{Vector2, Vector3, Vector4};

use crate::exchange::MeshFormat;

pub type UV = Vec<Vector2<f64>>;
pub type MaterialIndices = Vec<usize>;
pub type GroupingIndices = Vec<usize>;
pub type Color = Vec<Vector4<u8>>;
pub type Normal = Vec<Vector3<f64>>;

#[derive(Debug, Clone, Default)]

pub enum GroupingKind {
    #[default]
    Unspecified,
    MaterialIndex,
    GroupingIndex,
    SmoothingIndex,
}

#[derive(Debug, Clone, Default)]
pub struct Grouping {
    pub name: String,
    pub kind: GroupingKind,
    pub indices: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct Attributes {
    pub uv: Vec<UV>,
    pub normals: Vec<Normal>,
    pub colors: Vec<Color>,
    pub groupings: Vec<Grouping>,
}

#[derive(Debug, Clone, Default)]
pub struct LoadSource {
    // what format was this mesh loaded from?
    pub format: Option<MeshFormat>,

    // many formats have a header which would otherwise be discarded
    pub header: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SimpleMaterial {
    pub name: String,
    pub diffuse: Option<Vector3<f64>>,
    pub specular: Option<Vector3<f64>>,
    pub shininess: Option<f64>,
    pub alpha: Option<f64>,
    pub image: Option<DynamicImage>,
}

#[derive(Debug, Clone)]
pub struct PBRMaterial {}

#[derive(Debug, Clone)]
pub struct EmptyMaterial {}

#[derive(Debug, Clone)]
pub enum Material {
    Empty(EmptyMaterial),
    Simple(SimpleMaterial),
    PBR(PBRMaterial),
}

pub const DEFAULT_COLOR: Vector4<u8> = Vector4::new(100, 100, 100, 255);
