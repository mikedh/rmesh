//! Trackball camera controller for interactive 3D navigation.

use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector2, Vector3};

/// A trackball camera controller that orbits around a center point.
#[derive(Debug, Clone)]
pub struct Trackball {
    /// The point the camera orbits around.
    pub center: Point3<f64>,
    /// Distance from the center to the camera.
    pub distance: f64,
    /// Rotation of the camera around the center.
    pub rotation: UnitQuaternion<f64>,
}

impl Default for Trackball {
    fn default() -> Self {
        Trackball {
            center: Point3::origin(),
            distance: 5.0,
            rotation: UnitQuaternion::identity(),
        }
    }
}

impl Trackball {
    /// Create a new trackball with the given center and distance.
    pub fn new(center: Point3<f64>, distance: f64) -> Self {
        Trackball {
            center,
            distance,
            rotation: UnitQuaternion::identity(),
        }
    }

    /// Get the camera position in world space.
    pub fn position(&self) -> Point3<f64> {
        let offset = self.rotation * Vector3::new(0.0, 0.0, self.distance);
        self.center + offset
    }

    /// Compute the view matrix for rendering.
    pub fn view_matrix(&self) -> Matrix4<f64> {
        let eye = self.position();
        let up = self.rotation * Vector3::y();

        // Look-at matrix
        let f = (self.center - eye).normalize();
        let s = f.cross(&up).normalize();
        let u = s.cross(&f);

        Matrix4::new(
            s.x,
            s.y,
            s.z,
            -s.dot(&eye.coords),
            u.x,
            u.y,
            u.z,
            -u.dot(&eye.coords),
            -f.x,
            -f.y,
            -f.z,
            f.dot(&eye.coords),
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Rotate the camera based on mouse delta (in normalized screen coordinates).
    pub fn rotate(&mut self, delta: Vector2<f64>) {
        // Sensitivity factor
        let sensitivity = 2.0;

        // Rotate around world Y axis (yaw)
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -delta.x * sensitivity);

        // Rotate around local X axis (pitch)
        let right = self.rotation * Vector3::x();
        let pitch = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(right),
            -delta.y * sensitivity,
        );

        self.rotation = yaw * pitch * self.rotation;
    }

    /// Pan the camera (move center) based on mouse delta.
    pub fn pan(&mut self, delta: Vector2<f64>) {
        // Scale pan speed by distance
        let scale = self.distance * 0.5;

        let right = self.rotation * Vector3::x();
        let up = self.rotation * Vector3::y();

        self.center += right * (-delta.x * scale) + up * (delta.y * scale);
    }

    /// Zoom by adjusting distance (factor > 1 zooms out, < 1 zooms in).
    pub fn zoom(&mut self, factor: f64) {
        self.distance *= factor;
        self.distance = self.distance.clamp(0.01, 10000.0);
    }

    /// Fit the camera to view the given bounding box.
    pub fn fit(&mut self, min: Point3<f64>, max: Point3<f64>) {
        // Set center to bounding box center
        self.center = Point3::new(
            (min.x + max.x) / 2.0,
            (min.y + max.y) / 2.0,
            (min.z + max.z) / 2.0,
        );

        // Set distance based on bounding box size
        let size = (max - min).norm();
        self.distance = size * 1.5;

        // Reset rotation to default view
        self.rotation = UnitQuaternion::identity();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trackball_default() {
        let tb = Trackball::default();
        assert_eq!(tb.center, Point3::origin());
        assert!((tb.distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_trackball_position() {
        let tb = Trackball::new(Point3::origin(), 10.0);
        let pos = tb.position();
        // Default rotation: camera is at (0, 0, 10)
        assert!((pos.x).abs() < 1e-10);
        assert!((pos.y).abs() < 1e-10);
        assert!((pos.z - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_trackball_view_matrix() {
        let tb = Trackball::new(Point3::origin(), 5.0);
        let view = tb.view_matrix();
        // View matrix should be valid (determinant close to 1 for orthonormal rotation)
        let det = view.fixed_view::<3, 3>(0, 0).determinant();
        assert!((det.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trackball_fit() {
        let mut tb = Trackball::default();
        tb.fit(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        assert_eq!(tb.center, Point3::origin());
        // Size is sqrt(12) ≈ 3.46, distance should be ~5.2
        assert!(tb.distance > 3.0 && tb.distance < 10.0);
    }
}
