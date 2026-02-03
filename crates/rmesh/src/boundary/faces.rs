//! Analytical surface types from BREP faces (ISO 10303-42 elementary surfaces).
//!
//! When a STEP file is loaded via cascadio, each triangle can be tagged with the
//! BREP face it came from. These types capture the analytical surface definition
//! so that projection and buffer operations can preserve exact geometry (e.g.
//! cylinders project as circles, arcs from fillets are preserved through offset).

use nalgebra::{Point2, Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::creation::Plane;

/// An analytical surface from a BREP model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Surface {
    Plane(SurfacePlane),
    Cylinder(Cylinder),
    Cone(Cone),
    Sphere(Sphere),
    Torus(Torus),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfacePlane {
    pub origin: Point3<f64>,
    pub normal: Vector3<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cylinder {
    pub origin: Point3<f64>,
    pub axis: Vector3<f64>,
    pub radius: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cone {
    pub apex: Point3<f64>,
    pub axis: Vector3<f64>,
    pub half_angle: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sphere {
    pub center: Point3<f64>,
    pub radius: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Torus {
    pub center: Point3<f64>,
    pub axis: Vector3<f64>,
    pub major_radius: f64,
    pub minor_radius: f64,
}

impl Surface {
    /// If this surface projects as a circle onto the given plane,
    /// return (center_2d, radius).
    ///
    /// Currently handles cylinders whose axis is approximately parallel to the
    /// projection plane normal (covers drill holes, bores, fillets, rounds).
    pub fn project_as_circle(&self, plane: &Plane) -> Option<(Point2<f64>, f64)> {
        match self {
            Surface::Cylinder(c) => {
                let axis_norm = c.axis.normalize();
                // Cylinder axis must be ~parallel to the plane normal
                if axis_norm.dot(&plane.normal).abs() > 1.0 - 1e-6 {
                    let center_2d = plane.to_2d(&[c.origin]);
                    Some((center_2d[0], c.radius))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Human-readable kind name for grouping labels.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Surface::Plane(_) => "Plane",
            Surface::Cylinder(_) => "Cylinder",
            Surface::Cone(_) => "Cone",
            Surface::Sphere(_) => "Sphere",
            Surface::Torus(_) => "Torus",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Vector3};

    #[test]
    fn test_cylinder_project_as_circle_aligned() {
        let cyl = Surface::Cylinder(Cylinder {
            origin: Point3::new(1.0, 2.0, 0.0),
            axis: Vector3::new(0.0, 0.0, 1.0),
            radius: 5.0,
        });
        let plane = Plane::new(Vector3::new(0.0, 0.0, 1.0), Point3::origin());
        let (center, radius) = cyl.project_as_circle(&plane).unwrap();
        assert_relative_eq!(radius, 5.0, epsilon = 1e-10);
        assert_relative_eq!(center.x, 1.0, epsilon = 1e-6);
        assert_relative_eq!(center.y, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cylinder_project_as_circle_misaligned() {
        let cyl = Surface::Cylinder(Cylinder {
            origin: Point3::new(0.0, 0.0, 0.0),
            axis: Vector3::new(1.0, 0.0, 0.0),
            radius: 5.0,
        });
        let plane = Plane::new(Vector3::new(0.0, 0.0, 1.0), Point3::origin());
        assert!(cyl.project_as_circle(&plane).is_none());
    }

    #[test]
    fn test_plane_surface_no_circle() {
        let prim = Surface::Plane(SurfacePlane {
            origin: Point3::origin(),
            normal: Vector3::z(),
        });
        let plane = Plane::new(Vector3::z(), Point3::origin());
        assert!(prim.project_as_circle(&plane).is_none());
    }

    #[test]
    fn test_serde_roundtrip() {
        let cyl = Surface::Cylinder(Cylinder {
            origin: Point3::new(1.0, 2.0, 3.0),
            axis: Vector3::new(0.0, 0.0, 1.0),
            radius: 5.0,
        });
        let json = serde_json::to_string(&cyl).unwrap();
        let deserialized: Surface = serde_json::from_str(&json).unwrap();
        assert_eq!(cyl, deserialized);
    }

    #[test]
    fn test_kind_name() {
        assert_eq!(
            Surface::Plane(SurfacePlane {
                origin: Point3::origin(),
                normal: Vector3::z(),
            })
            .kind_name(),
            "Plane"
        );
        assert_eq!(
            Surface::Cylinder(Cylinder {
                origin: Point3::origin(),
                axis: Vector3::z(),
                radius: 1.0,
            })
            .kind_name(),
            "Cylinder"
        );
        assert_eq!(
            Surface::Cone(Cone {
                apex: Point3::origin(),
                axis: Vector3::z(),
                half_angle: 0.5,
            })
            .kind_name(),
            "Cone"
        );
        assert_eq!(
            Surface::Sphere(Sphere {
                center: Point3::origin(),
                radius: 1.0,
            })
            .kind_name(),
            "Sphere"
        );
        assert_eq!(
            Surface::Torus(Torus {
                center: Point3::origin(),
                axis: Vector3::z(),
                major_radius: 2.0,
                minor_radius: 0.5,
            })
            .kind_name(),
            "Torus"
        );
    }
}
