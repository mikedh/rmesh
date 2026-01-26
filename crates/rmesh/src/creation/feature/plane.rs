//! Sketch plane definition using quaternion orientation
//!
//! A sketch plane is defined by an origin point and a quaternion orientation.
//! This avoids issues with shear/scale that would be possible with a full matrix.

use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

/// A plane in 3D space for sketching, defined by origin + quaternion orientation.
///
/// The plane's local X and Y axes are determined by rotating the world X and Y axes
/// by the orientation quaternion. The plane normal is the rotated Z axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SketchPlane {
    /// Origin point of the plane (where local (0,0) maps to)
    pub origin: Point3<f64>,
    /// Orientation as a unit quaternion (defines the plane's local X, Y axes)
    #[serde(with = "quaternion_serde")]
    pub orientation: UnitQuaternion<f64>,
}

impl SketchPlane {
    /// Create a new plane from origin and orientation
    pub fn new(origin: Point3<f64>, orientation: UnitQuaternion<f64>) -> Self {
        Self { origin, orientation }
    }

    /// The XY plane (Z=0, normal pointing +Z)
    pub fn xy() -> Self {
        Self {
            origin: Point3::origin(),
            orientation: UnitQuaternion::identity(),
        }
    }

    /// The XZ plane (Y=0, normal pointing +Y)
    pub fn xz() -> Self {
        Self {
            origin: Point3::origin(),
            // Rotate -90 degrees around X axis
            orientation: UnitQuaternion::from_axis_angle(
                &Vector3::x_axis(),
                -std::f64::consts::FRAC_PI_2,
            ),
        }
    }

    /// The YZ plane (X=0, normal pointing +X)
    pub fn yz() -> Self {
        Self {
            origin: Point3::origin(),
            // Rotate 90 degrees around Y axis
            orientation: UnitQuaternion::from_axis_angle(
                &Vector3::y_axis(),
                std::f64::consts::FRAC_PI_2,
            ),
        }
    }

    /// Create a plane at the given Z height, parallel to XY
    pub fn at_z(z: f64) -> Self {
        Self {
            origin: Point3::new(0.0, 0.0, z),
            orientation: UnitQuaternion::identity(),
        }
    }

    /// Create a plane from origin and normal vector
    ///
    /// The X and Y axes in the plane are computed arbitrarily but deterministically.
    pub fn from_normal(origin: Point3<f64>, normal: Vector3<f64>) -> Self {
        let normal = normal.normalize();

        // If normal is close to +Z, use identity rotation
        if normal.dot(&Vector3::z()) > 0.9999 {
            return Self {
                origin,
                orientation: UnitQuaternion::identity(),
            };
        }

        // If normal is close to -Z, rotate 180 degrees around X
        if normal.dot(&Vector3::z()) < -0.9999 {
            return Self {
                origin,
                orientation: UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI),
            };
        }

        // General case: rotate from +Z to the normal
        let axis = Vector3::z().cross(&normal).normalize();
        let angle = Vector3::z().dot(&normal).acos();

        Self {
            origin,
            orientation: UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle),
        }
    }

    /// Get the normal vector of this plane (local +Z direction)
    pub fn normal(&self) -> Vector3<f64> {
        self.orientation.transform_vector(&Vector3::z())
    }

    /// Get the local X axis direction
    pub fn x_axis(&self) -> Vector3<f64> {
        self.orientation.transform_vector(&Vector3::x())
    }

    /// Get the local Y axis direction
    pub fn y_axis(&self) -> Vector3<f64> {
        self.orientation.transform_vector(&Vector3::y())
    }

    /// Transform a 2D point on this plane to 3D world coordinates
    pub fn point_to_world(&self, local: Point2<f64>) -> Point3<f64> {
        // Start with point in local plane coordinates (z=0)
        let local_3d = Point3::new(local.x, local.y, 0.0);
        // Rotate by orientation
        let rotated = self.orientation.transform_point(&local_3d);
        // Translate by origin
        Point3::new(
            rotated.x + self.origin.x,
            rotated.y + self.origin.y,
            rotated.z + self.origin.z,
        )
    }

    /// Transform multiple 2D points to 3D world coordinates
    pub fn points_to_world(&self, points: &[Point2<f64>]) -> Vec<Point3<f64>> {
        points.iter().map(|p| self.point_to_world(*p)).collect()
    }

    /// Project a 3D world point onto this plane, returning 2D local coordinates
    pub fn point_to_local(&self, world: Point3<f64>) -> Point2<f64> {
        // Translate to origin
        let translated = Point3::new(
            world.x - self.origin.x,
            world.y - self.origin.y,
            world.z - self.origin.z,
        );
        // Inverse rotate
        let local = self.orientation.inverse_transform_point(&translated);
        Point2::new(local.x, local.y)
    }

    /// Project multiple 3D points to 2D local coordinates
    pub fn points_to_local(&self, points: &[Point3<f64>]) -> Vec<Point2<f64>> {
        points.iter().map(|p| self.point_to_local(*p)).collect()
    }
}

impl Default for SketchPlane {
    fn default() -> Self {
        Self::xy()
    }
}

/// Custom serde module for UnitQuaternion
mod quaternion_serde {
    use nalgebra::UnitQuaternion;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct QuatComponents {
        w: f64,
        i: f64,
        j: f64,
        k: f64,
    }

    pub fn serialize<S>(q: &UnitQuaternion<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let quat = q.quaternion();
        QuatComponents {
            w: quat.w,
            i: quat.i,
            j: quat.j,
            k: quat.k,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<UnitQuaternion<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let c = QuatComponents::deserialize(deserializer)?;
        Ok(UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            c.w, c.i, c.j, c.k,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_xy_plane() {
        let plane = SketchPlane::xy();
        assert_relative_eq!(plane.normal(), Vector3::z(), epsilon = 1e-10);
        assert_relative_eq!(plane.x_axis(), Vector3::x(), epsilon = 1e-10);
        assert_relative_eq!(plane.y_axis(), Vector3::y(), epsilon = 1e-10);
    }

    #[test]
    fn test_xz_plane() {
        let plane = SketchPlane::xz();
        assert_relative_eq!(plane.normal(), Vector3::y(), epsilon = 1e-10);
    }

    #[test]
    fn test_yz_plane() {
        let plane = SketchPlane::yz();
        assert_relative_eq!(plane.normal(), Vector3::x(), epsilon = 1e-10);
    }

    #[test]
    fn test_point_to_world() {
        let plane = SketchPlane::xy();
        let world = plane.point_to_world(Point2::new(5.0, 3.0));
        assert_relative_eq!(world, Point3::new(5.0, 3.0, 0.0), epsilon = 1e-10);

        // Test with offset origin
        let plane = SketchPlane::at_z(10.0);
        let world = plane.point_to_world(Point2::new(5.0, 3.0));
        assert_relative_eq!(world, Point3::new(5.0, 3.0, 10.0), epsilon = 1e-10);
    }

    #[test]
    fn test_point_roundtrip() {
        let plane = SketchPlane::from_normal(Point3::new(1.0, 2.0, 3.0), Vector3::new(1.0, 1.0, 1.0));
        let local = Point2::new(5.0, 3.0);
        let world = plane.point_to_world(local);
        let back = plane.point_to_local(world);
        assert_relative_eq!(back, local, epsilon = 1e-10);
    }

    #[test]
    fn test_serde_roundtrip() {
        let plane = SketchPlane::xz();
        let json = serde_json::to_string(&plane).unwrap();
        let parsed: SketchPlane = serde_json::from_str(&json).unwrap();

        assert_relative_eq!(plane.normal(), parsed.normal(), epsilon = 1e-10);
        assert_relative_eq!(plane.origin, parsed.origin, epsilon = 1e-10);
    }
}
