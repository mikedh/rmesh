//! Curve primitives for 2D and 3D paths using vertex indices
//!
//! This module provides the fundamental curve types used in paths and sketches.
//! All entities reference vertices by index into a shared vertex array, enabling
//! efficient storage and simplified connectivity detection.

pub mod arc;
mod bezier;
mod bspline;
mod circle;
mod ellipse;
mod line;

pub use arc::{Arc2, Arc3};
pub use bezier::{CubicBezier, QuadraticBezier};
pub use bspline::BSpline;
pub use circle::{Circle2, Circle3};
pub use ellipse::Ellipse2;
pub use line::Line;

use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};

/// Winding direction for arcs and circles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Winding {
    /// Counter-clockwise (positive angle)
    #[default]
    Ccw,
    /// Clockwise (negative angle)
    Cw,
}

impl Winding {
    /// Returns the sign multiplier for angle calculations
    pub fn sign(&self) -> f64 {
        match self {
            Winding::Ccw => 1.0,
            Winding::Cw => -1.0,
        }
    }
}

/// Common operations for all curve types with indexed vertices
pub trait Curve {
    /// Get the start and finish vertex indices as [start, finish].
    ///
    /// Returns `None` if the curve is degenerate or has no points.
    /// For closed curves (circles, ellipses), both indices are the same.
    fn end_indices(&self) -> Option<[usize; 2]>;

    /// Check if this is a closed curve (start == finish, like a full circle)
    fn is_closed(&self) -> bool {
        false
    }
}

/// Helper trait for 2D point operations needed by entities
pub trait Point2Ops: Copy {
    fn x(&self) -> f64;
    fn y(&self) -> f64;
    fn new(x: f64, y: f64) -> Self;
}

impl Point2Ops for Point2<f64> {
    fn x(&self) -> f64 {
        self.x
    }
    fn y(&self) -> f64 {
        self.y
    }
    fn new(x: f64, y: f64) -> Self {
        Point2::new(x, y)
    }
}

/// Helper trait for 3D point operations needed by entities
pub trait Point3Ops: Copy {
    fn x(&self) -> f64;
    fn y(&self) -> f64;
    fn z(&self) -> f64;
    fn new(x: f64, y: f64, z: f64) -> Self;
}

impl Point3Ops for Point3<f64> {
    fn x(&self) -> f64 {
        self.x
    }
    fn y(&self) -> f64 {
        self.y
    }
    fn z(&self) -> f64 {
        self.z
    }
    fn new(x: f64, y: f64, z: f64) -> Self {
        Point3::new(x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winding_sign() {
        assert_eq!(Winding::Ccw.sign(), 1.0);
        assert_eq!(Winding::Cw.sign(), -1.0);
    }
}
