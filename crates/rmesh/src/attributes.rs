use ndarray::Array2;

use nalgebra::{Vector3, Vector2};

/// Enum representing the kind of attribute for vertices or faces
pub enum Attribute {

    // UV coordinates, typically 0.0 - 1.0
    UV(Vec<Vector2<f64>>),
    // RGB or RGBA color
    Color(Array2<u8>),

    // A normal vector
    Normal(Array2<f64>),

    // A (key, value) pair for custom attributes
    CustomFloat((String, Array2<f64>)),
    CustomInt((String, Array2<i64>)),
}
