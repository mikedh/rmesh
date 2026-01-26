//! Curve primitives for 2D and 3D paths
//!
//! This module provides the fundamental curve types used in paths and sketches.
//! Types are generic over dimension where possible, with explicit 3D variants
//! where additional fields (like normals) are required.

use nalgebra::{Point2, Point3, Vector3};
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

// =============================================================================
// Line - identical structure in 2D and 3D
// =============================================================================

/// A line segment from start to finish
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Line<P> {
    pub start: P,
    pub finish: P,
}

/// 2D line segment
pub type Line2D = Line<Point2<f64>>;
/// 3D line segment
pub type Line3D = Line<Point3<f64>>;

impl<P: Copy> Line<P> {
    /// Create a new line segment
    pub fn new(start: P, finish: P) -> Self {
        Self { start, finish }
    }
}

// =============================================================================
// Arc - uses OrientedArc design: endpoints + center + winding
// =============================================================================

/// A circular arc defined by endpoints, center, and winding direction.
///
/// The arc travels from `start` to `finish` around `center` in the direction
/// specified by `winding`. Angle and radius are computed properties.
///
/// If `center` is `None`, this represents a degenerate arc (straight line).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arc<P> {
    pub start: P,
    pub finish: P,
    /// Arc center. None indicates a degenerate (straight) arc.
    pub center: Option<P>,
    pub winding: Winding,
}

/// 2D circular arc
pub type Arc2D = Arc<Point2<f64>>;
/// 3D circular arc (lives in the plane defined by start/finish/center)
pub type Arc3D = Arc<Point3<f64>>;

impl Arc2D {
    /// Create a new 2D arc
    pub fn new(start: Point2<f64>, finish: Point2<f64>, center: Point2<f64>, winding: Winding) -> Self {
        Self {
            start,
            finish,
            center: Some(center),
            winding,
        }
    }

    /// Create an arc from endpoints and sweep angle.
    /// Computes the center point from the geometry.
    pub fn from_angle(start: Point2<f64>, finish: Point2<f64>, sweep_angle: f64) -> Self {
        if sweep_angle.abs() < 1e-10 {
            // Degenerate arc (straight line)
            return Self {
                start,
                finish,
                center: None,
                winding: Winding::Ccw,
            };
        }

        // Midpoint of the chord
        let mid = Point2::new((start.x + finish.x) / 2.0, (start.y + finish.y) / 2.0);

        // Half chord length
        let chord_len = ((finish.x - start.x).powi(2) + (finish.y - start.y).powi(2)).sqrt();
        let half_chord = chord_len / 2.0;

        // Radius from sweep angle: r = half_chord / sin(sweep/2)
        let half_sweep = sweep_angle.abs() / 2.0;
        let radius = half_chord / half_sweep.sin();

        // Distance from midpoint to center
        let h = (radius * radius - half_chord * half_chord).sqrt();

        // Perpendicular direction to chord
        let dx = finish.x - start.x;
        let dy = finish.y - start.y;
        let len = (dx * dx + dy * dy).sqrt();
        let px = -dy / len;
        let py = dx / len;

        // Choose center side based on sweep angle sign
        let sign = if sweep_angle > 0.0 { 1.0 } else { -1.0 };
        let center = Point2::new(mid.x + sign * h * px, mid.y + sign * h * py);

        let winding = if sweep_angle > 0.0 {
            Winding::Ccw
        } else {
            Winding::Cw
        };

        Self {
            start,
            finish,
            center: Some(center),
            winding,
        }
    }

    /// Compute the arc's radius (distance from center to start)
    pub fn radius(&self) -> f64 {
        match self.center {
            Some(c) => ((self.start.x - c.x).powi(2) + (self.start.y - c.y).powi(2)).sqrt(),
            None => 0.0,
        }
    }

    /// Compute the arc's sweep angle in radians
    pub fn angle(&self) -> f64 {
        let Some(center) = self.center else {
            return 0.0;
        };

        let start_angle = (self.start.y - center.y).atan2(self.start.x - center.x);
        let end_angle = (self.finish.y - center.y).atan2(self.finish.x - center.x);

        let mut sweep = end_angle - start_angle;

        // Adjust sweep based on winding direction
        match self.winding {
            Winding::Ccw => {
                if sweep < 0.0 {
                    sweep += std::f64::consts::TAU;
                }
            }
            Winding::Cw => {
                if sweep > 0.0 {
                    sweep -= std::f64::consts::TAU;
                }
            }
        }

        sweep
    }

    /// Get the start angle in radians (0 = +X axis)
    pub fn start_angle(&self) -> f64 {
        match self.center {
            Some(c) => (self.start.y - c.y).atan2(self.start.x - c.x),
            None => 0.0,
        }
    }
}

impl Arc3D {
    /// Create a new 3D arc
    pub fn new(start: Point3<f64>, finish: Point3<f64>, center: Point3<f64>, winding: Winding) -> Self {
        Self {
            start,
            finish,
            center: Some(center),
            winding,
        }
    }

    /// Compute the arc's radius
    pub fn radius(&self) -> f64 {
        match self.center {
            Some(c) => (self.start - c).norm(),
            None => 0.0,
        }
    }

    /// Compute the plane normal for this 3D arc
    pub fn normal(&self) -> Option<Vector3<f64>> {
        let center = self.center?;
        let v1 = self.start - center;
        let v2 = self.finish - center;
        let n = v1.cross(&v2);
        if n.norm() < 1e-10 {
            None
        } else {
            Some(n.normalize())
        }
    }
}

// =============================================================================
// Circle - full circles (not arcs)
// =============================================================================

/// A full circle defined by center and radius (2D)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Circle2D {
    pub center: Point2<f64>,
    pub radius: f64,
}

impl Circle2D {
    /// Create a new 2D circle
    pub fn new(center: Point2<f64>, radius: f64) -> Self {
        Self { center, radius }
    }
}

/// A full circle in 3D space, requiring a normal to define the plane
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Circle3D {
    pub center: Point3<f64>,
    pub radius: f64,
    pub normal: Vector3<f64>,
}

impl Circle3D {
    /// Create a new 3D circle
    pub fn new(center: Point3<f64>, radius: f64, normal: Vector3<f64>) -> Self {
        Self {
            center,
            radius,
            normal: normal.normalize(),
        }
    }
}

// =============================================================================
// Ellipse - 2D only for now
// =============================================================================

/// A 2D ellipse defined by center, semi-major/minor axes, and rotation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ellipse2D {
    pub center: Point2<f64>,
    /// Semi-major axis length
    pub major: f64,
    /// Semi-minor axis length
    pub minor: f64,
    /// Rotation angle in radians (rotation of major axis from +X)
    pub rotation: f64,
}

impl Ellipse2D {
    /// Create a new ellipse
    pub fn new(center: Point2<f64>, major: f64, minor: f64, rotation: f64) -> Self {
        Self {
            center,
            major,
            minor,
            rotation,
        }
    }

    /// Create an axis-aligned ellipse (no rotation)
    pub fn axis_aligned(center: Point2<f64>, major: f64, minor: f64) -> Self {
        Self::new(center, major, minor, 0.0)
    }
}

// =============================================================================
// Bezier curves - generic over dimension
// =============================================================================

/// Cubic Bezier curve with 4 control points
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CubicBezier<P> {
    pub p0: P,
    pub p1: P,
    pub p2: P,
    pub p3: P,
}

/// 2D cubic Bezier curve
pub type CubicBezier2D = CubicBezier<Point2<f64>>;
/// 3D cubic Bezier curve
pub type CubicBezier3D = CubicBezier<Point3<f64>>;

impl<P: Copy> CubicBezier<P> {
    /// Create a new cubic Bezier curve
    pub fn new(p0: P, p1: P, p2: P, p3: P) -> Self {
        Self { p0, p1, p2, p3 }
    }
}

/// Quadratic Bezier curve with 3 control points
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuadraticBezier<P> {
    pub p0: P,
    pub p1: P,
    pub p2: P,
}

/// 2D quadratic Bezier curve
pub type QuadraticBezier2D = QuadraticBezier<Point2<f64>>;
/// 3D quadratic Bezier curve
pub type QuadraticBezier3D = QuadraticBezier<Point3<f64>>;

impl<P: Copy> QuadraticBezier<P> {
    /// Create a new quadratic Bezier curve
    pub fn new(p0: P, p1: P, p2: P) -> Self {
        Self { p0, p1, p2 }
    }
}

// =============================================================================
// BSpline - generic over dimension
// =============================================================================

/// B-spline curve defined by control points, knot vector, and degree
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BSpline<P> {
    pub points: Vec<P>,
    pub knots: Vec<f64>,
    pub degree: u8,
}

/// 2D B-spline curve
pub type BSpline2D = BSpline<Point2<f64>>;
/// 3D B-spline curve
pub type BSpline3D = BSpline<Point3<f64>>;

impl<P> BSpline<P> {
    /// Create a new B-spline curve
    pub fn new(points: Vec<P>, knots: Vec<f64>, degree: u8) -> Self {
        Self {
            points,
            knots,
            degree,
        }
    }
}

// =============================================================================
// Curve trait for common operations
// =============================================================================

/// Common operations for all curve types
pub trait Curve {
    /// The point type for this curve (Point2 or Point3)
    type Point: Copy;

    /// Get the start point of the curve
    fn start(&self) -> Self::Point;

    /// Get the end point of the curve
    fn finish(&self) -> Self::Point;

    /// Check if this is a closed curve (start == finish, like a full circle)
    fn is_closed(&self) -> bool {
        false
    }
}

// Implement Curve trait for 2D types
impl Curve for Line2D {
    type Point = Point2<f64>;
    fn start(&self) -> Self::Point {
        self.start
    }
    fn finish(&self) -> Self::Point {
        self.finish
    }
}

impl Curve for Arc2D {
    type Point = Point2<f64>;
    fn start(&self) -> Self::Point {
        self.start
    }
    fn finish(&self) -> Self::Point {
        self.finish
    }
}

impl Curve for Circle2D {
    type Point = Point2<f64>;
    fn start(&self) -> Self::Point {
        Point2::new(self.center.x + self.radius, self.center.y)
    }
    fn finish(&self) -> Self::Point {
        self.start() // Circles are closed
    }
    fn is_closed(&self) -> bool {
        true
    }
}

impl Curve for Ellipse2D {
    type Point = Point2<f64>;
    fn start(&self) -> Self::Point {
        // Start at the rightmost point (major axis endpoint, rotated)
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        Point2::new(
            self.center.x + self.major * cos_r,
            self.center.y + self.major * sin_r,
        )
    }
    fn finish(&self) -> Self::Point {
        self.start() // Ellipses are closed
    }
    fn is_closed(&self) -> bool {
        true
    }
}

impl Curve for CubicBezier2D {
    type Point = Point2<f64>;
    fn start(&self) -> Self::Point {
        self.p0
    }
    fn finish(&self) -> Self::Point {
        self.p3
    }
}

impl Curve for QuadraticBezier2D {
    type Point = Point2<f64>;
    fn start(&self) -> Self::Point {
        self.p0
    }
    fn finish(&self) -> Self::Point {
        self.p2
    }
}

impl Curve for BSpline2D {
    type Point = Point2<f64>;
    fn start(&self) -> Self::Point {
        self.points.first().copied().unwrap_or(Point2::origin())
    }
    fn finish(&self) -> Self::Point {
        self.points.last().copied().unwrap_or(Point2::origin())
    }
}

// Implement Curve trait for 3D types
impl Curve for Line3D {
    type Point = Point3<f64>;
    fn start(&self) -> Self::Point {
        self.start
    }
    fn finish(&self) -> Self::Point {
        self.finish
    }
}

impl Curve for Arc3D {
    type Point = Point3<f64>;
    fn start(&self) -> Self::Point {
        self.start
    }
    fn finish(&self) -> Self::Point {
        self.finish
    }
}

impl Curve for Circle3D {
    type Point = Point3<f64>;
    fn start(&self) -> Self::Point {
        // Need to compute a start point on the circle
        // Use cross product to find a vector in the plane
        let arbitrary = if self.normal.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };
        let in_plane = self.normal.cross(&arbitrary).normalize();
        self.center + in_plane * self.radius
    }
    fn finish(&self) -> Self::Point {
        self.start() // Circles are closed
    }
    fn is_closed(&self) -> bool {
        true
    }
}

impl Curve for CubicBezier3D {
    type Point = Point3<f64>;
    fn start(&self) -> Self::Point {
        self.p0
    }
    fn finish(&self) -> Self::Point {
        self.p3
    }
}

impl Curve for QuadraticBezier3D {
    type Point = Point3<f64>;
    fn start(&self) -> Self::Point {
        self.p0
    }
    fn finish(&self) -> Self::Point {
        self.p2
    }
}

impl Curve for BSpline3D {
    type Point = Point3<f64>;
    fn start(&self) -> Self::Point {
        self.points.first().copied().unwrap_or(Point3::origin())
    }
    fn finish(&self) -> Self::Point {
        self.points.last().copied().unwrap_or(Point3::origin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_line_2d() {
        let line = Line2D::new(Point2::new(0.0, 0.0), Point2::new(10.0, 5.0));
        assert_eq!(line.start(), Point2::new(0.0, 0.0));
        assert_eq!(line.finish(), Point2::new(10.0, 5.0));
        assert!(!line.is_closed());
    }

    #[test]
    fn test_arc_2d_from_angle() {
        let arc = Arc2D::from_angle(
            Point2::new(10.0, 0.0),
            Point2::new(0.0, 10.0),
            std::f64::consts::FRAC_PI_2,
        );
        assert_relative_eq!(arc.radius(), 10.0, epsilon = 1e-6);
        assert_relative_eq!(arc.angle(), std::f64::consts::FRAC_PI_2, epsilon = 1e-6);
    }

    #[test]
    fn test_circle_2d() {
        let circle = Circle2D::new(Point2::new(5.0, 5.0), 10.0);
        assert_eq!(circle.start(), Point2::new(15.0, 5.0));
        assert!(circle.is_closed());
    }

    #[test]
    fn test_winding_sign() {
        assert_eq!(Winding::Ccw.sign(), 1.0);
        assert_eq!(Winding::Cw.sign(), -1.0);
    }

    #[test]
    fn test_cubic_bezier_2d() {
        let bezier = CubicBezier2D::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );
        assert_eq!(bezier.start(), Point2::new(0.0, 0.0));
        assert_eq!(bezier.finish(), Point2::new(4.0, 0.0));
    }
}
