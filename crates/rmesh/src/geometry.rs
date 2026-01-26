use crate::mesh::Trimesh;
use crate::path::{Path2D, Path3D};

/// Geometry types that can be loaded or created
pub enum Geometry {
    /// A triangle mesh
    Mesh(Box<Trimesh>),
    /// A 2D path (curves in a plane)
    Path2D(Path2D),
    /// A 3D path (curves in space)
    Path3D(Path3D),
}
