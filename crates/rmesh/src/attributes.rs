use nalgebra::{Vector2, Vector3, Vector4};

/// Enum representing the kind of attribute for vertices or faces

#[derive(Debug, Clone, Default)]
pub enum Attribute {
    #[default]
    Unspecified,
    // UV coordinates, typically 0.0 - 1.0
    UV(Vec<Vector2<f64>>),
    // What material was this face or vertex assigned to?
    Material(Vec<usize>),
    // Was this vertex or face part of a group?
    Group(Vec<usize>),
    // RGB or RGBA color
    Color(Vec<Vector4<u8>>),
    // A normal vector
    Normal(Vec<Vector3<f64>>),
    // A (key, value) pair for custom attributes
    CustomFloat((String, Vec<Vector3<f64>>)),
    CustomInt((String, Vec<Vector3<i64>>)),
}
