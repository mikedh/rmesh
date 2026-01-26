//! Tessellation algorithms for curve primitives
//!
//! This module provides functions to convert curves into polylines (sequences of points)
//! with configurable tolerance for approximation quality.

use nalgebra::{Point2, Point3};

use super::entity::{
    Arc2D, Arc3D, BSpline2D, BSpline3D, Circle2D, Circle3D, CubicBezier2D, CubicBezier3D,
    Ellipse2D, Line2D, Line3D, QuadraticBezier2D, QuadraticBezier3D, Winding,
};

// =============================================================================
// 2D Tessellation
// =============================================================================

/// Tessellate a 2D line segment
///
/// Returns the start point only (for chaining with other curves).
pub fn tessellate_line_2d(line: &Line2D) -> Vec<Point2<f64>> {
    vec![line.start]
}

/// Tessellate a 2D arc into line segments
///
/// # Arguments
/// * `arc` - The arc to tessellate
/// * `tolerance` - Maximum allowed deviation from the true curve
pub fn tessellate_arc_2d(arc: &Arc2D, tolerance: f64) -> Vec<Point2<f64>> {
    let Some(center) = arc.center else {
        // Degenerate arc is just a line
        return vec![arc.start];
    };

    let radius = arc.radius();
    if radius < 1e-10 {
        return vec![arc.start];
    }

    let sweep = arc.angle();
    tessellate_arc_internal(center, radius, arc.start_angle(), sweep, tolerance)
}

/// Tessellate a 2D circle into line segments
pub fn tessellate_circle_2d(circle: &Circle2D, tolerance: f64) -> Vec<Point2<f64>> {
    tessellate_arc_internal(
        circle.center,
        circle.radius,
        0.0,
        std::f64::consts::TAU,
        tolerance,
    )
}

/// Tessellate a 2D ellipse into line segments
pub fn tessellate_ellipse_2d(ellipse: &Ellipse2D, tolerance: f64) -> Vec<Point2<f64>> {
    // Use average radius for segment count calculation
    let avg_radius = (ellipse.major + ellipse.minor) / 2.0;
    let max_angle_step = (8.0 * tolerance / avg_radius).sqrt().min(std::f64::consts::FRAC_PI_4);
    let num_segments = (std::f64::consts::TAU / max_angle_step).ceil() as usize;
    let num_segments = num_segments.max(8);

    let angle_step = std::f64::consts::TAU / num_segments as f64;
    let cos_rot = ellipse.rotation.cos();
    let sin_rot = ellipse.rotation.sin();

    (0..num_segments)
        .map(|i| {
            let t = i as f64 * angle_step;
            // Point on axis-aligned ellipse
            let x = ellipse.major * t.cos();
            let y = ellipse.minor * t.sin();
            // Rotate and translate
            Point2::new(
                ellipse.center.x + x * cos_rot - y * sin_rot,
                ellipse.center.y + x * sin_rot + y * cos_rot,
            )
        })
        .collect()
}

/// Tessellate a 2D cubic Bezier curve using De Casteljau subdivision
pub fn tessellate_cubic_bezier_2d(bezier: &CubicBezier2D, tolerance: f64) -> Vec<Point2<f64>> {
    let mut result = Vec::new();
    tessellate_cubic_recursive_2d(bezier.p0, bezier.p1, bezier.p2, bezier.p3, tolerance, &mut result);
    result
}

/// Tessellate a 2D quadratic Bezier curve
pub fn tessellate_quadratic_bezier_2d(bezier: &QuadraticBezier2D, tolerance: f64) -> Vec<Point2<f64>> {
    let mut result = Vec::new();
    tessellate_quadratic_recursive_2d(bezier.p0, bezier.p1, bezier.p2, tolerance, &mut result);
    result
}

/// Tessellate a 2D B-spline curve
pub fn tessellate_bspline_2d(spline: &BSpline2D, tolerance: f64) -> Vec<Point2<f64>> {
    if spline.points.len() < 2 {
        return spline.points.clone();
    }

    // For now, use a simple uniform sampling approach
    // A proper implementation would use knot insertion or Cox-de Boor
    let num_samples = estimate_bspline_samples(&spline.points, tolerance);

    (0..num_samples)
        .map(|i| {
            let t = i as f64 / (num_samples - 1) as f64;
            evaluate_bspline_2d(spline, t)
        })
        .collect()
}

// =============================================================================
// 3D Tessellation
// =============================================================================

/// Tessellate a 3D line segment
pub fn tessellate_line_3d(line: &Line3D) -> Vec<Point3<f64>> {
    vec![line.start]
}

/// Tessellate a 3D arc into line segments
pub fn tessellate_arc_3d(arc: &Arc3D, tolerance: f64) -> Vec<Point3<f64>> {
    let Some(center) = arc.center else {
        return vec![arc.start];
    };

    let radius = arc.radius();
    if radius < 1e-10 {
        return vec![arc.start];
    }

    // Get the plane of the arc
    let v1 = (arc.start - center).normalize();
    let normal = arc.normal().unwrap_or(nalgebra::Vector3::z());
    let v2 = normal.cross(&v1).normalize();

    // Calculate sweep angle using the plane basis
    let end_vec = arc.finish - center;
    let end_angle = end_vec.dot(&v2).atan2(end_vec.dot(&v1));

    let mut sweep = end_angle;
    match arc.winding {
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

    // Number of segments based on tolerance
    let max_angle_step = (8.0 * tolerance / radius).sqrt().min(std::f64::consts::FRAC_PI_4);
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

/// Tessellate a 3D circle into line segments
pub fn tessellate_circle_3d(circle: &Circle3D, tolerance: f64) -> Vec<Point3<f64>> {
    // Find basis vectors in the circle's plane
    let arbitrary = if circle.normal.x.abs() < 0.9 {
        nalgebra::Vector3::x()
    } else {
        nalgebra::Vector3::y()
    };
    let v1 = circle.normal.cross(&arbitrary).normalize();
    let v2 = circle.normal.cross(&v1).normalize();

    let max_angle_step = (8.0 * tolerance / circle.radius)
        .sqrt()
        .min(std::f64::consts::FRAC_PI_4);
    let num_segments = (std::f64::consts::TAU / max_angle_step).ceil() as usize;
    let num_segments = num_segments.max(8);
    let angle_step = std::f64::consts::TAU / num_segments as f64;

    (0..num_segments)
        .map(|i| {
            let angle = i as f64 * angle_step;
            circle.center + (v1 * angle.cos() + v2 * angle.sin()) * circle.radius
        })
        .collect()
}

/// Tessellate a 3D cubic Bezier curve
pub fn tessellate_cubic_bezier_3d(bezier: &CubicBezier3D, tolerance: f64) -> Vec<Point3<f64>> {
    let mut result = Vec::new();
    tessellate_cubic_recursive_3d(bezier.p0, bezier.p1, bezier.p2, bezier.p3, tolerance, &mut result);
    result
}

/// Tessellate a 3D quadratic Bezier curve
pub fn tessellate_quadratic_bezier_3d(bezier: &QuadraticBezier3D, tolerance: f64) -> Vec<Point3<f64>> {
    let mut result = Vec::new();
    tessellate_quadratic_recursive_3d(bezier.p0, bezier.p1, bezier.p2, tolerance, &mut result);
    result
}

/// Tessellate a 3D B-spline curve
pub fn tessellate_bspline_3d(spline: &BSpline3D, tolerance: f64) -> Vec<Point3<f64>> {
    if spline.points.len() < 2 {
        return spline.points.clone();
    }

    let num_samples = estimate_bspline_samples_3d(&spline.points, tolerance);

    (0..num_samples)
        .map(|i| {
            let t = i as f64 / (num_samples - 1) as f64;
            evaluate_bspline_3d(spline, t)
        })
        .collect()
}

// =============================================================================
// Internal helper functions
// =============================================================================

/// Internal arc tessellation for 2D (center-angle representation)
fn tessellate_arc_internal(
    center: Point2<f64>,
    radius: f64,
    start_angle: f64,
    sweep: f64,
    tolerance: f64,
) -> Vec<Point2<f64>> {
    // Number of segments based on tolerance
    // For a circle: error ~= r * (1 - cos(theta/2)) ~= r * theta^2/8 for small theta
    // Solving for theta: theta = sqrt(8 * tolerance / radius)
    let max_angle_step = (8.0 * tolerance / radius).sqrt().min(std::f64::consts::FRAC_PI_4);
    let num_segments = (sweep.abs() / max_angle_step).ceil() as usize;
    let num_segments = num_segments.max(1);

    let angle_step = sweep / num_segments as f64;

    (0..num_segments)
        .map(|i| {
            let angle = start_angle + i as f64 * angle_step;
            Point2::new(center.x + radius * angle.cos(), center.y + radius * angle.sin())
        })
        .collect()
}

/// Recursive cubic Bezier tessellation for 2D
fn tessellate_cubic_recursive_2d(
    p0: Point2<f64>,
    p1: Point2<f64>,
    p2: Point2<f64>,
    p3: Point2<f64>,
    tolerance: f64,
    result: &mut Vec<Point2<f64>>,
) {
    // Flatness test: check if control points are close to the line p0-p3
    let d1 = point_to_line_distance_2d(p1, p0, p3);
    let d2 = point_to_line_distance_2d(p2, p0, p3);

    if d1 <= tolerance && d2 <= tolerance {
        result.push(p0);
    } else {
        // Subdivide at t=0.5 using De Casteljau
        let p01 = midpoint_2d(p0, p1);
        let p12 = midpoint_2d(p1, p2);
        let p23 = midpoint_2d(p2, p3);
        let p012 = midpoint_2d(p01, p12);
        let p123 = midpoint_2d(p12, p23);
        let p0123 = midpoint_2d(p012, p123);

        tessellate_cubic_recursive_2d(p0, p01, p012, p0123, tolerance, result);
        tessellate_cubic_recursive_2d(p0123, p123, p23, p3, tolerance, result);
    }
}

/// Recursive quadratic Bezier tessellation for 2D
fn tessellate_quadratic_recursive_2d(
    p0: Point2<f64>,
    p1: Point2<f64>,
    p2: Point2<f64>,
    tolerance: f64,
    result: &mut Vec<Point2<f64>>,
) {
    let d = point_to_line_distance_2d(p1, p0, p2);

    if d <= tolerance {
        result.push(p0);
    } else {
        let p01 = midpoint_2d(p0, p1);
        let p12 = midpoint_2d(p1, p2);
        let p012 = midpoint_2d(p01, p12);

        tessellate_quadratic_recursive_2d(p0, p01, p012, tolerance, result);
        tessellate_quadratic_recursive_2d(p012, p12, p2, tolerance, result);
    }
}

/// Recursive cubic Bezier tessellation for 3D
fn tessellate_cubic_recursive_3d(
    p0: Point3<f64>,
    p1: Point3<f64>,
    p2: Point3<f64>,
    p3: Point3<f64>,
    tolerance: f64,
    result: &mut Vec<Point3<f64>>,
) {
    let d1 = point_to_line_distance_3d(p1, p0, p3);
    let d2 = point_to_line_distance_3d(p2, p0, p3);

    if d1 <= tolerance && d2 <= tolerance {
        result.push(p0);
    } else {
        let p01 = midpoint_3d(p0, p1);
        let p12 = midpoint_3d(p1, p2);
        let p23 = midpoint_3d(p2, p3);
        let p012 = midpoint_3d(p01, p12);
        let p123 = midpoint_3d(p12, p23);
        let p0123 = midpoint_3d(p012, p123);

        tessellate_cubic_recursive_3d(p0, p01, p012, p0123, tolerance, result);
        tessellate_cubic_recursive_3d(p0123, p123, p23, p3, tolerance, result);
    }
}

/// Recursive quadratic Bezier tessellation for 3D
fn tessellate_quadratic_recursive_3d(
    p0: Point3<f64>,
    p1: Point3<f64>,
    p2: Point3<f64>,
    tolerance: f64,
    result: &mut Vec<Point3<f64>>,
) {
    let d = point_to_line_distance_3d(p1, p0, p2);

    if d <= tolerance {
        result.push(p0);
    } else {
        let p01 = midpoint_3d(p0, p1);
        let p12 = midpoint_3d(p1, p2);
        let p012 = midpoint_3d(p01, p12);

        tessellate_quadratic_recursive_3d(p0, p01, p012, tolerance, result);
        tessellate_quadratic_recursive_3d(p012, p12, p2, tolerance, result);
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

fn point_to_line_distance_2d(point: Point2<f64>, line_start: Point2<f64>, line_end: Point2<f64>) -> f64 {
    let dx = line_end.x - line_start.x;
    let dy = line_end.y - line_start.y;
    let len_sq = dx * dx + dy * dy;

    if len_sq < 1e-10 {
        return ((point.x - line_start.x).powi(2) + (point.y - line_start.y).powi(2)).sqrt();
    }

    let cross = (point.x - line_start.x) * dy - (point.y - line_start.y) * dx;
    cross.abs() / len_sq.sqrt()
}

fn point_to_line_distance_3d(point: Point3<f64>, line_start: Point3<f64>, line_end: Point3<f64>) -> f64 {
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
    let polygon_length: f64 = points
        .windows(2)
        .map(|w| (w[1] - w[0]).norm())
        .sum();

    let samples = (polygon_length / tolerance).ceil() as usize;
    samples.clamp(points.len(), 1000)
}

/// Evaluate a 2D B-spline at parameter t using Cox-de Boor (simplified)
fn evaluate_bspline_2d(spline: &BSpline2D, t: f64) -> Point2<f64> {
    // Simple Bezier-like evaluation for degree 3
    // A proper implementation would use the full Cox-de Boor algorithm
    let n = spline.points.len();
    if n == 0 {
        return Point2::origin();
    }
    if n == 1 {
        return spline.points[0];
    }

    // Bernstein polynomial evaluation (treating as Bezier for simplicity)
    let degree = (n - 1).min(spline.degree as usize);
    let mut result = Point2::new(0.0, 0.0);

    for (i, point) in spline.points.iter().take(degree + 1).enumerate() {
        let coeff = bernstein(degree, i, t);
        result.x += point.x * coeff;
        result.y += point.y * coeff;
    }

    result
}

fn evaluate_bspline_3d(spline: &BSpline3D, t: f64) -> Point3<f64> {
    let n = spline.points.len();
    if n == 0 {
        return Point3::origin();
    }
    if n == 1 {
        return spline.points[0];
    }

    let degree = (n - 1).min(spline.degree as usize);
    let mut result = Point3::new(0.0, 0.0, 0.0);

    for (i, point) in spline.points.iter().take(degree + 1).enumerate() {
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

/// Compute arc center from 3 points (2D)
///
/// Returns the center of the circle passing through all three points,
/// or None if the points are collinear.
pub fn arc_center_from_3_points_2d(
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

/// Compute arc center from 3 points (3D)
///
/// Returns the center of the circle passing through all three points,
/// or None if the points are collinear.
pub fn arc_center_from_3_points_3d(
    p1: Point3<f64>,
    p2: Point3<f64>,
    p3: Point3<f64>,
) -> Option<Point3<f64>> {
    // Vectors from p1
    let v1 = p2 - p1;
    let v2 = p3 - p1;

    // Normal to the plane
    let normal = v1.cross(&v2);
    if normal.norm() < 1e-10 {
        return None; // Collinear points
    }

    // Midpoints
    let m1 = p1 + v1 * 0.5;
    let m2 = p1 + v2 * 0.5;

    // Perpendicular bisector directions (in the plane)
    let d1 = normal.cross(&v1).normalize();
    let d2 = normal.cross(&v2).normalize();

    // Find intersection of perpendicular bisectors
    // m1 + t1 * d1 = m2 + t2 * d2
    // Solve for t1 using cross product
    let diff = m2 - m1;
    let cross_d = d1.cross(&d2);
    let denom = cross_d.norm_squared();

    if denom < 1e-10 {
        return None;
    }

    let t1 = diff.cross(&d2).dot(&cross_d) / denom;
    Some(m1 + d1 * t1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_arc_tessellation_2d() {
        let arc = Arc2D::new(
            Point2::new(10.0, 0.0),
            Point2::new(0.0, 10.0),
            Point2::new(0.0, 0.0),
            Winding::Ccw,
        );
        let points = tessellate_arc_2d(&arc, 0.1);

        // All points should be on the arc
        for p in &points {
            let dist = (p.x * p.x + p.y * p.y).sqrt();
            assert_relative_eq!(dist, 10.0, epsilon = 0.2);
        }
    }

    #[test]
    fn test_circle_tessellation_2d() {
        let circle = Circle2D::new(Point2::new(0.0, 0.0), 10.0);
        let points = tessellate_circle_2d(&circle, 0.1);

        assert!(points.len() > 20);
        for p in &points {
            let dist = (p.x * p.x + p.y * p.y).sqrt();
            assert_relative_eq!(dist, 10.0, epsilon = 0.2);
        }
    }

    #[test]
    fn test_cubic_bezier_tessellation_2d() {
        let bezier = CubicBezier2D::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );
        let points = tessellate_cubic_bezier_2d(&bezier, 0.01);

        assert!(!points.is_empty());
        assert_relative_eq!(points[0].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(points[0].y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_center_from_3_points_2d() {
        let p1 = Point2::new(10.0, 0.0);
        let p2 = Point2::new(0.0, 10.0);
        let p3 = Point2::new(-10.0, 0.0);

        let center = arc_center_from_3_points_2d(p1, p2, p3).unwrap();
        assert_relative_eq!(center.x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(center.y, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ellipse_tessellation() {
        let ellipse = Ellipse2D::new(Point2::new(0.0, 0.0), 10.0, 5.0, 0.0);
        let points = tessellate_ellipse_2d(&ellipse, 0.1);

        assert!(!points.is_empty());
        // First point should be on the major axis
        assert_relative_eq!(points[0].x, 10.0, epsilon = 1e-6);
        assert_relative_eq!(points[0].y, 0.0, epsilon = 1e-6);
    }
}
