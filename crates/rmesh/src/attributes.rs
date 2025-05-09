use image::DynamicImage;
use nalgebra::{Vector2, Vector3, Vector4};

use crate::exchange::MeshFormat;

#[derive(Debug, Clone, Default)]
pub enum Attribute {
    #[default]
    Unspecified,

    // UV coordinates, typically 0.0 - 1.0
    UV(Vec<Vector2<f64>>),

    // What material was this face or vertex assigned to?
    Material(Vec<usize>),

    // Was this vertex or face part of a group?
    Grouping(Vec<usize>),

    // RGB or RGBA color
    Color(Vec<Vector4<u8>>),

    // A normal vector
    Normal(Vec<Vector3<f64>>),
}

#[derive(Debug, Clone, Default)]
pub struct LoadSource {
    // what format was this mesh loaded from?
    pub format: Option<MeshFormat>,

    // many formats have a header which would otherwise be discarded
    pub header: Option<String>,
}

pub struct Grouping {
    pub name: String,
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
