//! Ellipse entity

use serde::{Deserialize, Serialize};

use super::Curve;

/// A 2D ellipse defined by center vertex index, semi-major/minor axes, and rotation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ellipse2 {
    pub center: usize,
    /// Semi-major axis length
    pub major: f64,
    /// Semi-minor axis length
    pub minor: f64,
    /// Rotation angle in radians (rotation of major axis from +X)
    #[serde(default)]
    pub rotation: f64,
}

impl Ellipse2 {
    /// Create a new ellipse
    pub fn new(center: usize, major: f64, minor: f64, rotation: f64) -> Self {
        Self {
            center,
            major,
            minor,
            rotation,
        }
    }

    /// Create an axis-aligned ellipse (no rotation)
    pub fn axis_aligned(center: usize, major: f64, minor: f64) -> Self {
        Self::new(center, major, minor, 0.0)
    }
}

impl Curve for Ellipse2 {
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
    fn test_ellipse() {
        let ellipse = Ellipse2::new(0, 10.0, 5.0, 0.0);
        assert_eq!(ellipse.end_indices(), Some([0, 0]));
        assert!(ellipse.is_closed());
    }

    #[test]
    fn test_ellipse_axis_aligned() {
        let ellipse = Ellipse2::axis_aligned(0, 10.0, 5.0);
        assert_eq!(ellipse.rotation, 0.0);
    }
}
