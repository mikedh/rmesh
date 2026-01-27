//! B-spline curve entity

use serde::{Deserialize, Serialize};

use super::Curve;

/// B-spline curve defined by control point indices, knot vector, and degree
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BSpline {
    pub points: Vec<usize>,
    pub knots: Vec<f64>,
    pub degree: u8,
}

impl BSpline {
    /// Create a new B-spline curve
    pub fn new(points: Vec<usize>, knots: Vec<f64>, degree: u8) -> Self {
        Self {
            points,
            knots,
            degree,
        }
    }
}

impl Curve for BSpline {
    fn end_indices(&self) -> Option<[usize; 2]> {
        Some([*self.points.first()?, *self.points.last()?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bspline() {
        let spline = BSpline::new(vec![0, 1, 2, 3], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2);
        assert_eq!(spline.end_indices(), Some([0, 3]));
        assert!(!spline.is_closed());
    }
}
