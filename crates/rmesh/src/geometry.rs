use crate::creation::feature::FeatureModel;
use crate::mesh::Trimesh;
use crate::path::{Path2D, Path3D};

/// A point cloud with optional per-point colors and normals
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    /// 3D positions of points
    pub points: Vec<nalgebra::Point3<f64>>,
    /// Optional per-point colors (RGBA)
    pub colors: Option<Vec<nalgebra::Vector4<u8>>>,
    /// Optional per-point normals
    pub normals: Option<Vec<nalgebra::Vector3<f64>>>,
}

/// Geometry types that can be loaded or created
#[derive(Debug, Clone)]
pub enum Geometry {
    /// A triangle mesh
    Mesh(Box<Trimesh>),
    /// A 2D path (curves in a plane)
    Path2D(Path2D),
    /// A 3D path (curves in space)
    Path3D(Path3D),
    /// A feature-based CAD model
    Feature(Box<FeatureModel>),
    /// A point cloud
    PointCloud(PointCloud),
}
