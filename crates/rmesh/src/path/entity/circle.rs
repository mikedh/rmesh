//! Circle entities

use nalgebra::{Unit, Vector3};
use serde::{Deserialize, Serialize};

use super::Curve;
use crate::serialize::unit_vector3;

/// A full circle defined by center vertex index and radius (2D)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Circle2 {
    pub center: usize,
    pub radius: f64,
}

impl Circle2 {
    /// Create a new circle
    pub fn new(center: usize, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl Curve for Circle2 {
    fn end_indices(&self) -> Option<[usize; 2]> {
        Some([self.center, self.center])
    }

    fn is_closed(&self) -> bool {
        true
    }
}

/// A full circle in 3D space, requiring a normal to define the plane
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Circle3 {
    pub center: usize,
    pub radius: f64,
    /// Unit normal vector defining the circle's plane
    #[serde(with = "unit_vector3")]
    pub normal: Unit<Vector3<f64>>,
}

impl Circle3 {
    /// Create a new 3D circle
    ///
    /// # Panics
    /// Panics if the normal vector is zero.
    pub fn new(center: usize, radius: f64, normal: Vector3<f64>) -> Self {
        Self {
            center,
            radius,
            normal: Unit::try_new(normal, 1e-10)
                .expect("Circle3 normal vector cannot be zero or near-zero"),
        }
    }
}

impl Curve for Circle3 {
    fn end_indices(&self) -> Option<[usize; 2]> {
        Some([self.center, self.center])
    }

    fn is_closed(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle2() {
        let circle = Circle2::new(0, 10.0);
        assert_eq!(circle.end_indices(), Some([0, 0]));
        assert!(circle.is_closed());
    }

    #[test]
    fn test_circle_3d() {
        let circle = Circle3::new(0, 10.0, Vector3::z());
        assert!(circle.is_closed());
        assert_eq!(*circle.normal, Vector3::z());
        assert_eq!(circle.end_indices(), Some([0, 0]));
    }

    #[test]
    #[should_panic(expected = "Circle3 normal vector cannot be zero")]
    fn test_circle_3d_zero_normal_panics() {
        let _ = Circle3::new(0, 10.0, Vector3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_circle_3d_deserialize_zero_normal_returns_error() {
        // JSON with a zero normal vector - should return Err, not panic
        let json = r#"{"center": 0, "radius": 10.0, "normal": [0.0, 0.0, 0.0]}"#;
        let result: Result<Circle3, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("zero"),
            "Expected error about zero vector, got: {}",
            err
        );
    }

    #[test]
    fn test_circle_3d_serde_roundtrip() {
        let circle = Circle3::new(0, 10.0, Vector3::z());
        let json = serde_json::to_string(&circle).unwrap();
        let deserialized: Circle3 = serde_json::from_str(&json).unwrap();
        assert_eq!(circle, deserialized);
    }
}
