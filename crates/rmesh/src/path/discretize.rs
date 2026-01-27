//! Discretization algorithms for curve primitives
//!
//! This module provides functions to convert curves into polylines (sequences of points)
//! with configurable tolerance for approximation quality.

use nalgebra::{Point2, Point3, Vector3};

use super::entity::{
    Arc2, Arc3, BSpline, Circle2, Circle3, CubicBezier, Ellipse2, Line, QuadraticBezier,
};
use super::{Segment2D, Segment3D};

// =============================================================================
// Main discretization entry points
// =============================================================================

/// Discretize a 2D segment using vertex array
pub fn discretize_segment_2d(
    segment: &Segment2D,
    vertices: &[Point2<f64>],
    tolerance: f64,
) -> Vec<Point2<f64>> {
    match segment {
        Segment2D::Line(l) => discretize_line_2d(l, vertices),
        Segment2D::Arc(a) => discretize_arc_2d(a, vertices, tolerance),
        Segment2D::Circle(c) => discretize_circle_2d(c, vertices, tolerance),
        Segment2D::Ellipse(e) => discretize_ellipse_2d(e, vertices, tolerance),
        Segment2D::CubicBezier(b) => discretize_cubic_bezier_2d(b, vertices, tolerance),
        Segment2D::QuadraticBezier(b) => discretize_quadratic_bezier_2d(b, vertices, tolerance),
        Segment2D::BSpline(s) => discretize_bspline_2d(s, vertices, tolerance),
    }
}

/// Discretize a 3D segment using vertex array
pub fn discretize_segment_3d(
    segment: &Segment3D,
    vertices: &[Point3<f64>],
    tolerance: f64,
) -> Vec<Point3<f64>> {
    match segment {
        Segment3D::Line(l) => discretize_line_3d(l, vertices),
        Segment3D::Arc(a) => discretize_arc_3d(a, vertices, tolerance),
        Segment3D::Circle(c) => discretize_circle_3d(c, vertices, tolerance),
        Segment3D::CubicBezier(b) => discretize_cubic_bezier_3d(b, vertices, tolerance),
        Segment3D::QuadraticBezier(b) => discretize_quadratic_bezier_3d(b, vertices, tolerance),
        Segment3D::BSpline(s) => discretize_bspline_3d(s, vertices, tolerance),
    }
}

// =============================================================================
// 2D Discretization
// =============================================================================

/// Discretize a 2D line/polyline
///
/// Returns all points except the last (caller adds finish point for chaining).
pub fn discretize_line_2d(line: &Line, vertices: &[Point2<f64>]) -> Vec<Point2<f64>> {
    line.points
        .iter()
        .take(line.points.len().saturating_sub(1))
        .filter_map(|&i| vertices.get(i).copied())
        .collect()
}

/// Discretize a 2D arc into line segments
pub fn discretize_arc_2d(arc: &Arc2, vertices: &[Point2<f64>], tolerance: f64) -> Vec<Point2<f64>> {
    let Some(center) = arc.center(vertices) else {
        // Degenerate arc is just a line - use bounds-checked access
        return vertices.get(arc.start).copied().into_iter().collect();
    };

    let Some(radius) = arc.radius(vertices) else {
        return vertices.get(arc.start).copied().into_iter().collect();
    };
    if radius < 1e-10 {
        return vertices.get(arc.start).copied().into_iter().collect();
    }

    let sweep = arc.sweep_angle();
    let Some(start_angle) = arc.start_angle(vertices) else {
        return vertices.get(arc.start).copied().into_iter().collect();
    };
    discretize_arc_internal(center, radius, start_angle, sweep, tolerance)
}

/// Discretize a 2D circle into line segments
///
/// Returns a closed polygon (first point repeated at end) for proper perimeter calculation.
pub fn discretize_circle_2d(
    circle: &Circle2,
    vertices: &[Point2<f64>],
    tolerance: f64,
) -> Vec<Point2<f64>> {
    let Some(center) = vertices.get(circle.center).copied() else {
        return Vec::new();
    };
    let mut points =
        discretize_arc_internal(center, circle.radius, 0.0, std::f64::consts::TAU, tolerance);
    // Close the circle by adding the first point at the end
    if let Some(&first) = points.first() {
        points.push(first);
    }
    points
}

/// Discretize a 2D ellipse into line segments
///
/// Returns a closed polygon (first point repeated at end) for proper perimeter calculation.
pub fn discretize_ellipse_2d(
    ellipse: &Ellipse2,
    vertices: &[Point2<f64>],
    tolerance: f64,
) -> Vec<Point2<f64>> {
    let Some(center) = vertices.get(ellipse.center).copied() else {
        return Vec::new();
    };

    // Use average radius for segment count calculation
    let avg_radius = (ellipse.major + ellipse.minor) / 2.0;
    if avg_radius < 1e-10 {
        // Degenerate ellipse - return center point
        return vec![center, center];
    }
    let max_angle_step = (8.0 * tolerance / avg_radius)
        .sqrt()
        .min(std::f64::consts::FRAC_PI_4);
    let num_segments = (std::f64::consts::TAU / max_angle_step).ceil() as usize;
    let num_segments = num_segments.max(8);

    let angle_step = std::f64::consts::TAU / num_segments as f64;
    let cos_rot = ellipse.rotation.cos();
    let sin_rot = ellipse.rotation.sin();

    let mut points: Vec<Point2<f64>> = (0..num_segments)
        .map(|i| {
            let t = i as f64 * angle_step;
            // Point on axis-aligned ellipse
            let x = ellipse.major * t.cos();
            let y = ellipse.minor * t.sin();
            // Rotate and translate
            Point2::new(
                center.x + x * cos_rot - y * sin_rot,
                center.y + x * sin_rot + y * cos_rot,
            )
        })
        .collect();
    // Close the ellipse by adding the first point at the end
    if let Some(&first) = points.first() {
        points.push(first);
    }
    points
}

/// Maximum recursion depth for Bezier subdivision to prevent stack overflow
const MAX_BEZIER_DEPTH: u32 = 32;

/// Discretize a 2D cubic Bezier curve using De Casteljau subdivision
pub fn discretize_cubic_bezier_2d(
    bezier: &CubicBezier,
    vertices: &[Point2<f64>],
    tolerance: f64,
) -> Vec<Point2<f64>> {
    let Some(p0) = vertices.get(bezier.p0).copied() else {
        return Vec::new();
    };
    let Some(p1) = vertices.get(bezier.p1).copied() else {
        return vec![p0];
    };
    let Some(p2) = vertices.get(bezier.p2).copied() else {
        return vec![p0];
    };
    let Some(p3) = vertices.get(bezier.p3).copied() else {
        return vec![p0];
    };
    let mut result = Vec::new();
    discretize_cubic_recursive_2d(p0, p1, p2, p3, tolerance, &mut result, 0);
    result
}

/// Discretize a 2D quadratic Bezier curve
pub fn discretize_quadratic_bezier_2d(
    bezier: &QuadraticBezier,
    vertices: &[Point2<f64>],
    tolerance: f64,
) -> Vec<Point2<f64>> {
    let Some(p0) = vertices.get(bezier.p0).copied() else {
        return Vec::new();
    };
    let Some(p1) = vertices.get(bezier.p1).copied() else {
        return vec![p0];
    };
    let Some(p2) = vertices.get(bezier.p2).copied() else {
        return vec![p0];
    };
    let mut result = Vec::new();
    discretize_quadratic_recursive_2d(p0, p1, p2, tolerance, &mut result, 0);
    result
}

/// Discretize a 2D B-spline curve
pub fn discretize_bspline_2d(
    spline: &BSpline,
    vertices: &[Point2<f64>],
    tolerance: f64,
) -> Vec<Point2<f64>> {
    let points: Vec<_> = spline
        .points
        .iter()
        .filter_map(|&i| vertices.get(i).copied())
        .collect();
    if points.len() < 2 {
        return points;
    }

    // For now, use a simple uniform sampling approach
    // Ensure at least 2 samples to avoid division by zero
    let num_samples = estimate_bspline_samples(&points, tolerance).max(2);

    (0..num_samples)
        .map(|i| {
            let t = i as f64 / (num_samples - 1) as f64;
            evaluate_bspline_2d(&points, spline.degree, t)
        })
        .collect()
}

// =============================================================================
// 3D Discretization
// =============================================================================

/// Discretize a 3D line/polyline
pub fn discretize_line_3d(line: &Line, vertices: &[Point3<f64>]) -> Vec<Point3<f64>> {
    line.points
        .iter()
        .take(line.points.len().saturating_sub(1))
        .filter_map(|&i| vertices.get(i).copied())
        .collect()
}

/// Discretize a 3D arc into line segments
pub fn discretize_arc_3d(arc: &Arc3, vertices: &[Point3<f64>], tolerance: f64) -> Vec<Point3<f64>> {
    let Some(start) = vertices.get(arc.start).copied() else {
        return Vec::new();
    };
    let Some(finish) = vertices.get(arc.finish).copied() else {
        return vec![start];
    };

    let sweep = arc.sweep_angle();
    if sweep.abs() < 1e-10 {
        return vec![start];
    }

    // Compute center in 3D using the chord and angle
    let chord = finish - start;
    let chord_len = chord.norm();
    if chord_len < 1e-10 {
        return vec![start];
    }

    // Radius from chord and angle
    let half_chord = chord_len / 2.0;
    let half_sweep = sweep.abs() / 2.0;
    let sin_half = half_sweep.sin();
    if sin_half.abs() < 1e-10 {
        return vec![start];
    }
    let radius = half_chord / sin_half;

    // Midpoint of chord
    let mid = start + chord * 0.5;

    // Use the arc's normal to define the plane
    let normal = arc.normal;
    let chord_dir = chord.normalize();
    let perp = normal.cross(&chord_dir).normalize();

    // Distance from midpoint to center
    let h_sq = radius * radius - half_chord * half_chord;
    let h = if h_sq > 0.0 { h_sq.sqrt() } else { 0.0 };

    // Choose center side based on sweep angle sign
    let sign = if sweep > 0.0 { 1.0 } else { -1.0 };
    let center = mid + perp * (sign * h);

    // Build basis vectors for the arc plane
    let v1 = (start - center).normalize();
    let v2 = normal.cross(&v1).normalize();

    // Number of segments based on tolerance
    let max_angle_step = (8.0 * tolerance / radius)
        .sqrt()
        .min(std::f64::consts::FRAC_PI_4);
    let num_segments = (sweep.abs() / max_angle_step).ceil() as usize;
    let num_segments = num_segments.max(1);
    let angle_step = sweep / num_segments as f64;

    (0..num_segments)
        .map(|i| {
            let angle = i as f64 * angle_step;
            center + (v1 * angle.cos() + v2 * angle.sin()) * radius
        })
        .collect()
}

/// Discretize a 3D circle into line segments
///
/// Returns a closed polygon (first point repeated at end) for proper perimeter calculation.
pub fn discretize_circle_3d(
    circle: &Circle3,
    vertices: &[Point3<f64>],
    tolerance: f64,
) -> Vec<Point3<f64>> {
    let Some(center) = vertices.get(circle.center).copied() else {
        return Vec::new();
    };

    // Find basis vectors in the circle's plane
    let arbitrary = if circle.normal.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let v1 = circle.normal.cross(&arbitrary).normalize();
    let v2 = circle.normal.cross(&v1).normalize();

    let max_angle_step = (8.0 * tolerance / circle.radius)
        .sqrt()
        .min(std::f64::consts::FRAC_PI_4);
    let num_segments = (std::f64::consts::TAU / max_angle_step).ceil() as usize;
    let num_segments = num_segments.max(8);
    let angle_step = std::f64::consts::TAU / num_segments as f64;

    let mut points: Vec<Point3<f64>> = (0..num_segments)
        .map(|i| {
            let angle = i as f64 * angle_step;
            center + (v1 * angle.cos() + v2 * angle.sin()) * circle.radius
        })
        .collect();
    // Close the circle by adding the first point at the end
    if let Some(&first) = points.first() {
        points.push(first);
    }
    points
}

/// Discretize a 3D cubic Bezier curve
pub fn discretize_cubic_bezier_3d(
    bezier: &CubicBezier,
    vertices: &[Point3<f64>],
    tolerance: f64,
) -> Vec<Point3<f64>> {
    let Some(p0) = vertices.get(bezier.p0).copied() else {
        return Vec::new();
    };
    let Some(p1) = vertices.get(bezier.p1).copied() else {
        return vec![p0];
    };
    let Some(p2) = vertices.get(bezier.p2).copied() else {
        return vec![p0];
    };
    let Some(p3) = vertices.get(bezier.p3).copied() else {
        return vec![p0];
    };
    let mut result = Vec::new();
    discretize_cubic_recursive_3d(p0, p1, p2, p3, tolerance, &mut result, 0);
    result
}

/// Discretize a 3D quadratic Bezier curve
pub fn discretize_quadratic_bezier_3d(
    bezier: &QuadraticBezier,
    vertices: &[Point3<f64>],
    tolerance: f64,
) -> Vec<Point3<f64>> {
    let Some(p0) = vertices.get(bezier.p0).copied() else {
        return Vec::new();
    };
    let Some(p1) = vertices.get(bezier.p1).copied() else {
        return vec![p0];
    };
    let Some(p2) = vertices.get(bezier.p2).copied() else {
        return vec![p0];
    };
    let mut result = Vec::new();
    discretize_quadratic_recursive_3d(p0, p1, p2, tolerance, &mut result, 0);
    result
}

/// Discretize a 3D B-spline curve
pub fn discretize_bspline_3d(
    spline: &BSpline,
    vertices: &[Point3<f64>],
    tolerance: f64,
) -> Vec<Point3<f64>> {
    let points: Vec<_> = spline
        .points
        .iter()
        .filter_map(|&i| vertices.get(i).copied())
        .collect();
    if points.len() < 2 {
        return points;
    }

    // Ensure at least 2 samples to avoid division by zero
    let num_samples = estimate_bspline_samples_3d(&points, tolerance).max(2);

    (0..num_samples)
        .map(|i| {
            let t = i as f64 / (num_samples - 1) as f64;
            evaluate_bspline_3d(&points, spline.degree, t)
        })
        .collect()
}

// =============================================================================
// Internal helper functions
// =============================================================================

/// Internal arc discretization for 2D (center-angle representation)
fn discretize_arc_internal(
    center: Point2<f64>,
    radius: f64,
    start_angle: f64,
    sweep: f64,
    tolerance: f64,
) -> Vec<Point2<f64>> {
    // Number of segments based on tolerance
    // For a circle: error ~= r * (1 - cos(theta/2)) ~= r * theta^2/8 for small theta
    // Solving for theta: theta = sqrt(8 * tolerance / radius)
    let max_angle_step = (8.0 * tolerance / radius)
        .sqrt()
        .min(std::f64::consts::FRAC_PI_4);
    let num_segments = (sweep.abs() / max_angle_step).ceil() as usize;
    let num_segments = num_segments.max(1);

    let angle_step = sweep / num_segments as f64;

    (0..num_segments)
        .map(|i| {
            let angle = start_angle + i as f64 * angle_step;
            Point2::new(
                center.x + radius * angle.cos(),
                center.y + radius * angle.sin(),
            )
        })
        .collect()
}

/// Recursive cubic Bezier discretization for 2D
fn discretize_cubic_recursive_2d(
    p0: Point2<f64>,
    p1: Point2<f64>,
    p2: Point2<f64>,
    p3: Point2<f64>,
    tolerance: f64,
    result: &mut Vec<Point2<f64>>,
    depth: u32,
) {
    // Flatness test: check if control points are close to the line p0-p3
    let d1 = point_to_line_distance_2d(p1, p0, p3);
    let d2 = point_to_line_distance_2d(p2, p0, p3);

    if (d1 <= tolerance && d2 <= tolerance) || depth >= MAX_BEZIER_DEPTH {
        result.push(p0);
    } else {
        // Subdivide at t=0.5 using De Casteljau
        let p01 = midpoint_2d(p0, p1);
        let p12 = midpoint_2d(p1, p2);
        let p23 = midpoint_2d(p2, p3);
        let p012 = midpoint_2d(p01, p12);
        let p123 = midpoint_2d(p12, p23);
        let p0123 = midpoint_2d(p012, p123);

        discretize_cubic_recursive_2d(p0, p01, p012, p0123, tolerance, result, depth + 1);
        discretize_cubic_recursive_2d(p0123, p123, p23, p3, tolerance, result, depth + 1);
    }
}

/// Recursive quadratic Bezier discretization for 2D
fn discretize_quadratic_recursive_2d(
    p0: Point2<f64>,
    p1: Point2<f64>,
    p2: Point2<f64>,
    tolerance: f64,
    result: &mut Vec<Point2<f64>>,
    depth: u32,
) {
    let d = point_to_line_distance_2d(p1, p0, p2);

    if d <= tolerance || depth >= MAX_BEZIER_DEPTH {
        result.push(p0);
    } else {
        let p01 = midpoint_2d(p0, p1);
        let p12 = midpoint_2d(p1, p2);
        let p012 = midpoint_2d(p01, p12);

        discretize_quadratic_recursive_2d(p0, p01, p012, tolerance, result, depth + 1);
        discretize_quadratic_recursive_2d(p012, p12, p2, tolerance, result, depth + 1);
    }
}

/// Recursive cubic Bezier discretization for 3D
fn discretize_cubic_recursive_3d(
    p0: Point3<f64>,
    p1: Point3<f64>,
    p2: Point3<f64>,
    p3: Point3<f64>,
    tolerance: f64,
    result: &mut Vec<Point3<f64>>,
    depth: u32,
) {
    let d1 = point_to_line_distance_3d(p1, p0, p3);
    let d2 = point_to_line_distance_3d(p2, p0, p3);

    if (d1 <= tolerance && d2 <= tolerance) || depth >= MAX_BEZIER_DEPTH {
        result.push(p0);
    } else {
        let p01 = midpoint_3d(p0, p1);
        let p12 = midpoint_3d(p1, p2);
        let p23 = midpoint_3d(p2, p3);
        let p012 = midpoint_3d(p01, p12);
        let p123 = midpoint_3d(p12, p23);
        let p0123 = midpoint_3d(p012, p123);

        discretize_cubic_recursive_3d(p0, p01, p012, p0123, tolerance, result, depth + 1);
        discretize_cubic_recursive_3d(p0123, p123, p23, p3, tolerance, result, depth + 1);
    }
}

/// Recursive quadratic Bezier discretization for 3D
fn discretize_quadratic_recursive_3d(
    p0: Point3<f64>,
    p1: Point3<f64>,
    p2: Point3<f64>,
    tolerance: f64,
    result: &mut Vec<Point3<f64>>,
    depth: u32,
) {
    let d = point_to_line_distance_3d(p1, p0, p2);

    if d <= tolerance || depth >= MAX_BEZIER_DEPTH {
        result.push(p0);
    } else {
        let p01 = midpoint_3d(p0, p1);
        let p12 = midpoint_3d(p1, p2);
        let p012 = midpoint_3d(p01, p12);

        discretize_quadratic_recursive_3d(p0, p01, p012, tolerance, result, depth + 1);
        discretize_quadratic_recursive_3d(p012, p12, p2, tolerance, result, depth + 1);
    }
}

// =============================================================================
// Utility functions
// =============================================================================

fn midpoint_2d(a: Point2<f64>, b: Point2<f64>) -> Point2<f64> {
    Point2::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
}

fn midpoint_3d(a: Point3<f64>, b: Point3<f64>) -> Point3<f64> {
    Point3::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0, (a.z + b.z) / 2.0)
}

fn point_to_line_distance_2d(
    point: Point2<f64>,
    line_start: Point2<f64>,
    line_end: Point2<f64>,
) -> f64 {
    let dx = line_end.x - line_start.x;
    let dy = line_end.y - line_start.y;
    let len_sq = dx * dx + dy * dy;

    if len_sq < 1e-10 {
        return ((point.x - line_start.x).powi(2) + (point.y - line_start.y).powi(2)).sqrt();
    }

    let cross = (point.x - line_start.x) * dy - (point.y - line_start.y) * dx;
    cross.abs() / len_sq.sqrt()
}

fn point_to_line_distance_3d(
    point: Point3<f64>,
    line_start: Point3<f64>,
    line_end: Point3<f64>,
) -> f64 {
    let line_vec = line_end - line_start;
    let len_sq = line_vec.norm_squared();

    if len_sq < 1e-10 {
        return (point - line_start).norm();
    }

    let point_vec = point - line_start;
    let cross = line_vec.cross(&point_vec);
    cross.norm() / len_sq.sqrt()
}

/// Estimate number of samples for B-spline based on control polygon length
fn estimate_bspline_samples(points: &[Point2<f64>], tolerance: f64) -> usize {
    let polygon_length: f64 = points
        .windows(2)
        .map(|w| ((w[1].x - w[0].x).powi(2) + (w[1].y - w[0].y).powi(2)).sqrt())
        .sum();

    let samples = (polygon_length / tolerance).ceil() as usize;
    samples.clamp(points.len(), 1000)
}

fn estimate_bspline_samples_3d(points: &[Point3<f64>], tolerance: f64) -> usize {
    let polygon_length: f64 = points.windows(2).map(|w| (w[1] - w[0]).norm()).sum();

    let samples = (polygon_length / tolerance).ceil() as usize;
    samples.clamp(points.len(), 1000)
}

/// Evaluate a 2D B-spline at parameter t using simplified Bezier-like evaluation
fn evaluate_bspline_2d(points: &[Point2<f64>], degree: u8, t: f64) -> Point2<f64> {
    let n = points.len();
    if n == 0 {
        return Point2::origin();
    }
    if n == 1 {
        return points[0];
    }

    // Bernstein polynomial evaluation (treating as Bezier for simplicity)
    let degree = (n - 1).min(degree as usize);
    let mut result = Point2::new(0.0, 0.0);

    for (i, point) in points.iter().take(degree + 1).enumerate() {
        let coeff = bernstein(degree, i, t);
        result.x += point.x * coeff;
        result.y += point.y * coeff;
    }

    result
}

fn evaluate_bspline_3d(points: &[Point3<f64>], degree: u8, t: f64) -> Point3<f64> {
    let n = points.len();
    if n == 0 {
        return Point3::origin();
    }
    if n == 1 {
        return points[0];
    }

    let degree = (n - 1).min(degree as usize);
    let mut result = Point3::new(0.0, 0.0, 0.0);

    for (i, point) in points.iter().take(degree + 1).enumerate() {
        let coeff = bernstein(degree, i, t);
        result.x += point.x * coeff;
        result.y += point.y * coeff;
        result.z += point.z * coeff;
    }

    result
}

/// Bernstein polynomial basis function
fn bernstein(n: usize, i: usize, t: f64) -> f64 {
    binomial(n, i) * t.powi(i as i32) * (1.0 - t).powi((n - i) as i32)
}

/// Binomial coefficient C(n, k)
fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64 / (i + 1) as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::entity::Winding;
    use approx::assert_relative_eq;

    #[test]
    fn test_arc_discretization_2d() {
        let vertices = vec![Point2::new(10.0, 0.0), Point2::new(0.0, 10.0)];
        let arc = Arc2::new(0, 1, std::f64::consts::FRAC_PI_2, Winding::Ccw);
        let points = discretize_arc_2d(&arc, &vertices, 0.1);

        // All points should be on the arc
        for p in &points {
            let dist = (p.x * p.x + p.y * p.y).sqrt();
            assert_relative_eq!(dist, 10.0, epsilon = 0.2);
        }
    }

    #[test]
    fn test_circle_discretization_2d() {
        let vertices = vec![Point2::origin()];
        let circle = Circle2::new(0, 10.0);
        let points = discretize_circle_2d(&circle, &vertices, 0.1);

        assert!(points.len() > 20);
        for p in &points {
            let dist = (p.x * p.x + p.y * p.y).sqrt();
            assert_relative_eq!(dist, 10.0, epsilon = 0.2);
        }
    }

    #[test]
    fn test_cubic_bezier_discretization_2d() {
        let vertices = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        ];
        let bezier = CubicBezier::new(0, 1, 2, 3);
        let points = discretize_cubic_bezier_2d(&bezier, &vertices, 0.01);

        assert!(!points.is_empty());
        assert_relative_eq!(points[0].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(points[0].y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ellipse_discretization() {
        let vertices = vec![Point2::origin()];
        let ellipse = Ellipse2::new(0, 10.0, 5.0, 0.0);
        let points = discretize_ellipse_2d(&ellipse, &vertices, 0.1);

        assert!(!points.is_empty());
        // First point should be on the major axis
        assert_relative_eq!(points[0].x, 10.0, epsilon = 1e-6);
        assert_relative_eq!(points[0].y, 0.0, epsilon = 1e-6);
    }
}
