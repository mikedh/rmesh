//! Bezier curve entities

use serde::{Deserialize, Serialize};

use super::Curve;

/// Cubic Bezier curve with 4 control point indices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CubicBezier {
    pub p0: usize,
    pub p1: usize,
    pub p2: usize,
    pub p3: usize,
}

impl CubicBezier {
    /// Create a new cubic Bezier curve
    pub fn new(p0: usize, p1: usize, p2: usize, p3: usize) -> Self {
        Self { p0, p1, p2, p3 }
    }
}

impl Curve for CubicBezier {
    fn end_indices(&self) -> Option<[usize; 2]> {
        Some([self.p0, self.p3])
    }
}

/// Quadratic Bezier curve with 3 control point indices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuadraticBezier {
    pub p0: usize,
    pub p1: usize,
    pub p2: usize,
}

impl QuadraticBezier {
    /// Create a new quadratic Bezier curve
    pub fn new(p0: usize, p1: usize, p2: usize) -> Self {
        Self { p0, p1, p2 }
    }
}

impl Curve for QuadraticBezier {
    fn end_indices(&self) -> Option<[usize; 2]> {
        Some([self.p0, self.p2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_bezier() {
        let bezier = CubicBezier::new(0, 1, 2, 3);
        assert_eq!(bezier.end_indices(), Some([0, 3]));
        assert!(!bezier.is_closed());
    }

    #[test]
    fn test_quadratic_bezier() {
        let bezier = QuadraticBezier::new(0, 1, 2);
        assert_eq!(bezier.end_indices(), Some([0, 2]));
    }
}
