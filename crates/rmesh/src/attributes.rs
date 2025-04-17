use nalgebra::Vector4;

/// Enum representing the kind of attribute for vertices or faces
pub enum AttributeKind {
    UV,
    Color,
    Normal,
    Custom,
}

/// Struct representing an attribute for vertices or faces
pub struct Attribute {
    pub kind: AttributeKind,
    pub name: String,
    pub data_f64: Option<Vector4<f64>>,
    pub data_u8: Option<Vector4<u8>>,
}
