//! Arc entities with computed center

use nalgebra::{Point2, Unit, Vector3};
use serde::{Deserialize, Serialize};

use super::{Curve, Point2Ops, Winding};
use crate::serialize::unit_vector3;

/// A circular arc defined by endpoint indices and sweep angle.
///
/// The center is computed from the endpoints and angle, rather than stored.
/// If `angle` is `None`, this represents a degenerate arc (straight line).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arc2 {
    pub start: usize,
    pub finish: usize,
    /// Sweep angle in radians. None indicates a degenerate (straight) arc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle: Option<f64>,
    #[serde(default)]
    pub winding: Winding,
}

impl Arc2 {
    /// Create a new arc with sweep angle
    pub fn new(start: usize, finish: usize, angle: f64, winding: Winding) -> Self {
        let angle = if angle.abs() < 1e-10 {
            None
        } else {
            Some(angle)
        };
        Self {
            start,
            finish,
            angle,
            winding,
        }
    }

    /// Create a degenerate arc (straight line)
    pub fn degenerate(start: usize, finish: usize) -> Self {
        Self {
            start,
            finish,
            angle: None,
            winding: Winding::Ccw,
        }
    }

    /// Compute center point from start, finish, and angle
    pub fn center<P: Point2Ops>(&self, vertices: &[P]) -> Option<P> {
        let angle = self.angle?;
        let start = vertices.get(self.start)?;
        let finish = vertices.get(self.finish)?;
        arc_center(start, finish, angle)
    }

    /// Compute the arc's radius given vertices
    pub fn radius<P: Point2Ops>(&self, vertices: &[P]) -> Option<f64> {
        let center = self.center(vertices)?;
        let start = vertices.get(self.start)?;
        Some(((start.x() - center.x()).powi(2) + (start.y() - center.y()).powi(2)).sqrt())
    }

    /// Get the start angle in radians (0 = +X axis)
    pub fn start_angle<P: Point2Ops>(&self, vertices: &[P]) -> Option<f64> {
        let center = self.center(vertices)?;
        let start = vertices.get(self.start)?;
        Some((start.y() - center.y()).atan2(start.x() - center.x()))
    }

    /// Get the sweep angle (returns 0 for degenerate arcs)
    pub fn sweep_angle(&self) -> f64 {
        self.angle.unwrap_or(0.0)
    }

    /// Check if this arc is degenerate (straight line)
    pub fn is_degenerate(&self) -> bool {
        self.angle.is_none()
    }
}

impl Curve for Arc2 {
    fn end_indices(&self) -> Option<[usize; 2]> {
        Some([self.start, self.finish])
    }
}

/// A 3D circular arc defined by endpoint indices, sweep angle, and plane normal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arc3 {
    pub start: usize,
    pub finish: usize,
    /// Sweep angle in radians. None indicates a degenerate (straight) arc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle: Option<f64>,
    /// Unit normal vector defining the arc's plane
    #[serde(with = "unit_vector3")]
    pub normal: Unit<Vector3<f64>>,
    #[serde(default)]
    pub winding: Winding,
}

impl Arc3 {
    /// Create a new 3D arc with sweep angle and plane normal
    ///
    /// # Panics
    /// Panics if the normal vector is zero or near-zero.
    pub fn new(
        start: usize,
        finish: usize,
        angle: f64,
        normal: Vector3<f64>,
        winding: Winding,
    ) -> Self {
        let angle = if angle.abs() < 1e-10 {
            None
        } else {
            Some(angle)
        };
        Self {
            start,
            finish,
            angle,
            normal: Unit::try_new(normal, 1e-10)
                .expect("Arc3 normal vector cannot be zero or near-zero"),
            winding,
        }
    }

    /// Create a degenerate arc (straight line)
    pub fn degenerate(start: usize, finish: usize, normal: Vector3<f64>) -> Self {
        let normal =
            Unit::try_new(normal, 1e-10).unwrap_or_else(|| Unit::new_unchecked(Vector3::z())); // Default to Z-up for degenerate case
        Self {
            start,
            finish,
            angle: None,
            normal,
            winding: Winding::Ccw,
        }
    }

    /// Get the sweep angle (returns 0 for degenerate arcs)
    pub fn sweep_angle(&self) -> f64 {
        self.angle.unwrap_or(0.0)
    }

    /// Check if this arc is degenerate (straight line)
    pub fn is_degenerate(&self) -> bool {
        self.angle.is_none()
    }
}

impl Curve for Arc3 {
    fn end_indices(&self) -> Option<[usize; 2]> {
        Some([self.start, self.finish])
    }
}

/// Compute arc center from endpoints and sweep angle
///
/// Returns `None` if:
/// - The sweep angle is too small (effectively a straight line)
/// - The chord length is zero (coincident points)
/// - The geometry is invalid (h_sq significantly negative)
pub fn arc_center<P: Point2Ops>(start: &P, finish: &P, sweep_angle: f64) -> Option<P> {
    // Midpoint of the chord
    let mid_x = f64::midpoint(start.x(), finish.x());
    let mid_y = f64::midpoint(start.y(), finish.y());

    // Half chord length
    let dx = finish.x() - start.x();
    let dy = finish.y() - start.y();
    let chord_len = (dx * dx + dy * dy).sqrt();
    let half_chord = chord_len / 2.0;

    // Perpendicular direction to chord
    if chord_len < 1e-10 {
        // Coincident points - no valid arc center
        return None;
    }

    // Radius from sweep angle: r = half_chord / sin(sweep/2)
    let half_sweep = sweep_angle.abs() / 2.0;
    let sin_half = half_sweep.sin();
    if sin_half.abs() < 1e-10 {
        // Very small angle - degenerate arc (effectively a straight line)
        return None;
    }
    let radius = half_chord / sin_half;

    // Distance from midpoint to center
    let h_sq = radius * radius - half_chord * half_chord;
    if h_sq < -1e-10 {
        // Invalid geometry - this shouldn't happen with valid inputs
        return None;
    }
    // Clamp small negative values from floating point error
    let h = h_sq.max(0.0).sqrt();

    let px = -dy / chord_len;
    let py = dx / chord_len;

    // Choose center side based on sweep angle sign
    let sign = if sweep_angle > 0.0 { 1.0 } else { -1.0 };
    Some(P::new(mid_x + sign * h * px, mid_y + sign * h * py))
}

/// Compute arc center from 3 points (2D)
///
/// Returns the center of the circle passing through all three points,
/// or None if the points are collinear.
pub fn arc_center_from_3_points(
    p1: Point2<f64>,
    p2: Point2<f64>,
    p3: Point2<f64>,
) -> Option<Point2<f64>> {
    // Using the circumcenter formula
    let ax = p1.x;
    let ay = p1.y;
    let bx = p2.x;
    let by = p2.y;
    let cx = p3.x;
    let cy = p3.y;

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-10 {
        return None; // Collinear points
    }

    let ax2_ay2 = ax * ax + ay * ay;
    let bx2_by2 = bx * bx + by * by;
    let cx2_cy2 = cx * cx + cy * cy;

    let ux = (ax2_ay2 * (by - cy) + bx2_by2 * (cy - ay) + cx2_cy2 * (ay - by)) / d;
    let uy = (ax2_ay2 * (cx - bx) + bx2_by2 * (ax - cx) + cx2_cy2 * (bx - ax)) / d;

    Some(Point2::new(ux, uy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_arc_center_calculation() {
        let vertices = vec![
            Point2::new(10.0, 0.0), // index 0
            Point2::new(0.0, 10.0), // index 1
        ];
        let arc = Arc2::new(0, 1, std::f64::consts::FRAC_PI_2, Winding::Ccw);

        let center = arc.center(&vertices).unwrap();
        assert_relative_eq!(center.x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(center.y, 0.0, epsilon = 1e-6);

        let radius = arc.radius(&vertices).unwrap();
        assert_relative_eq!(radius, 10.0, epsilon = 1e-6);

        assert_eq!(arc.end_indices(), Some([0, 1]));
    }

    #[test]
    fn test_degenerate_arc() {
        let arc = Arc2::degenerate(0, 1);
        assert!(arc.is_degenerate());
        assert_eq!(arc.sweep_angle(), 0.0);
    }

    #[test]
    fn test_arc_center_from_3_points() {
        let p1 = Point2::new(10.0, 0.0);
        let p2 = Point2::new(0.0, 10.0);
        let p3 = Point2::new(-10.0, 0.0);

        let center = arc_center_from_3_points(p1, p2, p3).unwrap();
        assert_relative_eq!(center.x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(center.y, 0.0, epsilon = 1e-6);
    }
}
