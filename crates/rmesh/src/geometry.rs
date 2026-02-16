use nalgebra::Point3;

use crate::boundary::BrepModel;
use crate::bounds::Bounds3;
use crate::creation::feature::FeatureModel;
use crate::mesh::Trimesh;
use crate::path::{Path2D, Path3D};

/// A point cloud with optional per-point colors and normals
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    /// 3D positions of points
    pub points: Vec<Point3<f64>>,
    /// Optional per-point colors (RGBA)
    pub colors: Option<Vec<nalgebra::Vector4<u8>>>,
    /// Optional per-point normals
    pub normals: Option<Vec<nalgebra::Vector3<f64>>>,
}

impl PointCloud {
    /// Compute the axis-aligned bounding box of the point cloud.
    pub fn bounds(&self) -> Option<Bounds3> {
        Bounds3::from_points(&self.points)
    }
}

/// Geometry types that can be loaded or created
#[derive(Debug, Clone)]
pub enum Geometry {
    /// A triangle mesh
    Mesh(Box<Trimesh>),
    /// A 2D path (curves in a plane) - boxed due to size (336 bytes)
    Path2D(Box<Path2D>),
    /// A 3D path (curves in space)
    Path3D(Path3D),
    /// A feature-based CAD model
    Feature(Box<FeatureModel>),
    /// A point cloud
    PointCloud(PointCloud),
    /// A BREP (boundary representation) solid model
    Brep(Box<BrepModel>),
}

impl Geometry {
    /// Compute the 3D axis-aligned bounding box for this geometry.
    ///
    /// Returns `None` if the geometry is empty or the type doesn't support bounds.
    /// For `Path2D`, the z-component is always 0.
    pub fn bounds(&self) -> Option<Bounds3> {
        match self {
            Geometry::Mesh(mesh) => mesh.bounds(),
            Geometry::Path2D(path) => path.as_ref().bounds().map(Bounds3::from_bounds2),
            Geometry::Path3D(path) => path.bounds(),
            Geometry::PointCloud(pc) => pc.bounds(),
            Geometry::Feature(_) => None,
            Geometry::Brep(brep) => brep.bounds(),
        }
    }
}
