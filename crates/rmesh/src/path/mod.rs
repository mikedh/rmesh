//! Path module for 2D and 3D curve operations
//!
//! This module provides path primitives with explicit 2D and 3D types.
//! Sketches use `Path2D`; sweep paths and edge curves use `Path3D`.
//!
//! # Structure
//!
//! - [`entity`] - Curve primitives: Arc, Line, Circle, Ellipse, Bezier, BSpline
//! - [`tessellate`] - Tessellation algorithms (De Casteljau, arc stepping)
//! - [`svg`] - SVG parsing/export for Path2D
//!
//! # Example
//!
//! ```
//! use rmesh::path::{Path2D, Segment2D, Line2D};
//! use nalgebra::Point2;
//!
//! // Create a simple rectangular path
//! let path = Path2D::rectangle(10.0, 5.0);
//!
//! // Or from SVG
//! let path = Path2D::from_svg("M0,0 L10,0 L10,5 L0,5 Z").unwrap();
//!
//! // Tessellate to points
//! let points = path.tessellate(0.1);
//! ```

pub mod entity;
pub mod svg;
pub mod tessellate;

use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};

// Re-export commonly used types
pub use entity::{
    Arc2D, Arc3D, BSpline2D, BSpline3D, Circle2D, Circle3D, CubicBezier2D, CubicBezier3D,
    Curve, Ellipse2D, Line, Line2D, Line3D, QuadraticBezier2D, QuadraticBezier3D, Winding,
};
pub use svg::SvgError;
pub use tessellate::{arc_center_from_3_points_2d, arc_center_from_3_points_3d};

// =============================================================================
// Segment enums - wrapping entity types for storage in paths
// =============================================================================

/// A 2D path segment (one curve element)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Segment2D {
    Line(Line2D),
    Arc(Arc2D),
    Circle(Circle2D),
    Ellipse(Ellipse2D),
    CubicBezier(CubicBezier2D),
    QuadraticBezier(QuadraticBezier2D),
    BSpline(BSpline2D),
}

impl Segment2D {
    /// Get the start point of this segment
    pub fn start(&self) -> Point2<f64> {
        use entity::Curve;
        match self {
            Segment2D::Line(l) => l.start(),
            Segment2D::Arc(a) => a.start(),
            Segment2D::Circle(c) => c.start(),
            Segment2D::Ellipse(e) => e.start(),
            Segment2D::CubicBezier(b) => b.start(),
            Segment2D::QuadraticBezier(b) => b.start(),
            Segment2D::BSpline(s) => s.start(),
        }
    }

    /// Get the end point of this segment
    pub fn finish(&self) -> Point2<f64> {
        use entity::Curve;
        match self {
            Segment2D::Line(l) => l.finish(),
            Segment2D::Arc(a) => a.finish(),
            Segment2D::Circle(c) => c.finish(),
            Segment2D::Ellipse(e) => e.finish(),
            Segment2D::CubicBezier(b) => b.finish(),
            Segment2D::QuadraticBezier(b) => b.finish(),
            Segment2D::BSpline(s) => s.finish(),
        }
    }

    /// Check if this segment is closed (e.g., a full circle)
    pub fn is_closed(&self) -> bool {
        use entity::Curve;
        match self {
            Segment2D::Line(l) => l.is_closed(),
            Segment2D::Arc(a) => a.is_closed(),
            Segment2D::Circle(c) => c.is_closed(),
            Segment2D::Ellipse(e) => e.is_closed(),
            Segment2D::CubicBezier(b) => b.is_closed(),
            Segment2D::QuadraticBezier(b) => b.is_closed(),
            Segment2D::BSpline(s) => s.is_closed(),
        }
    }

    /// Tessellate this segment into a sequence of points
    pub fn tessellate(&self, tolerance: f64) -> Vec<Point2<f64>> {
        match self {
            Segment2D::Line(l) => tessellate::tessellate_line_2d(l),
            Segment2D::Arc(a) => tessellate::tessellate_arc_2d(a, tolerance),
            Segment2D::Circle(c) => tessellate::tessellate_circle_2d(c, tolerance),
            Segment2D::Ellipse(e) => tessellate::tessellate_ellipse_2d(e, tolerance),
            Segment2D::CubicBezier(b) => tessellate::tessellate_cubic_bezier_2d(b, tolerance),
            Segment2D::QuadraticBezier(b) => tessellate::tessellate_quadratic_bezier_2d(b, tolerance),
            Segment2D::BSpline(s) => tessellate::tessellate_bspline_2d(s, tolerance),
        }
    }
}

/// A 3D path segment (one curve element)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Segment3D {
    Line(Line3D),
    Arc(Arc3D),
    Circle(Circle3D),
    CubicBezier(CubicBezier3D),
    QuadraticBezier(QuadraticBezier3D),
    BSpline(BSpline3D),
}

impl Segment3D {
    /// Get the start point of this segment
    pub fn start(&self) -> Point3<f64> {
        use entity::Curve;
        match self {
            Segment3D::Line(l) => l.start(),
            Segment3D::Arc(a) => a.start(),
            Segment3D::Circle(c) => c.start(),
            Segment3D::CubicBezier(b) => b.start(),
            Segment3D::QuadraticBezier(b) => b.start(),
            Segment3D::BSpline(s) => s.start(),
        }
    }

    /// Get the end point of this segment
    pub fn finish(&self) -> Point3<f64> {
        use entity::Curve;
        match self {
            Segment3D::Line(l) => l.finish(),
            Segment3D::Arc(a) => a.finish(),
            Segment3D::Circle(c) => c.finish(),
            Segment3D::CubicBezier(b) => b.finish(),
            Segment3D::QuadraticBezier(b) => b.finish(),
            Segment3D::BSpline(s) => s.finish(),
        }
    }

    /// Check if this segment is closed
    pub fn is_closed(&self) -> bool {
        use entity::Curve;
        match self {
            Segment3D::Line(l) => l.is_closed(),
            Segment3D::Arc(a) => a.is_closed(),
            Segment3D::Circle(c) => c.is_closed(),
            Segment3D::CubicBezier(b) => b.is_closed(),
            Segment3D::QuadraticBezier(b) => b.is_closed(),
            Segment3D::BSpline(s) => s.is_closed(),
        }
    }

    /// Tessellate this segment into a sequence of points
    pub fn tessellate(&self, tolerance: f64) -> Vec<Point3<f64>> {
        match self {
            Segment3D::Line(l) => tessellate::tessellate_line_3d(l),
            Segment3D::Arc(a) => tessellate::tessellate_arc_3d(a, tolerance),
            Segment3D::Circle(c) => tessellate::tessellate_circle_3d(c, tolerance),
            Segment3D::CubicBezier(b) => tessellate::tessellate_cubic_bezier_3d(b, tolerance),
            Segment3D::QuadraticBezier(b) => tessellate::tessellate_quadratic_bezier_3d(b, tolerance),
            Segment3D::BSpline(s) => tessellate::tessellate_bspline_3d(s, tolerance),
        }
    }
}

// =============================================================================
// Path structs
// =============================================================================

/// A 2D path consisting of connected curve segments
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Path2D {
    /// The curve segments that make up this path
    pub segments: Vec<Segment2D>,
    /// Whether this path forms a closed loop
    pub closed: bool,
}

impl Path2D {
    /// Create a new empty path
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a path from segments
    pub fn from_segments(segments: Vec<Segment2D>, closed: bool) -> Self {
        Self { segments, closed }
    }

    /// Create a rectangular path centered at origin
    pub fn rectangle(width: f64, height: f64) -> Self {
        let w = width / 2.0;
        let h = height / 2.0;

        let segments = vec![
            Segment2D::Line(Line2D::new(Point2::new(-w, -h), Point2::new(w, -h))),
            Segment2D::Line(Line2D::new(Point2::new(w, -h), Point2::new(w, h))),
            Segment2D::Line(Line2D::new(Point2::new(w, h), Point2::new(-w, h))),
            Segment2D::Line(Line2D::new(Point2::new(-w, h), Point2::new(-w, -h))),
        ];

        Self {
            segments,
            closed: true,
        }
    }

    /// Create a rectangular path with corner at origin
    pub fn rectangle_corner(width: f64, height: f64) -> Self {
        let segments = vec![
            Segment2D::Line(Line2D::new(Point2::new(0.0, 0.0), Point2::new(width, 0.0))),
            Segment2D::Line(Line2D::new(Point2::new(width, 0.0), Point2::new(width, height))),
            Segment2D::Line(Line2D::new(Point2::new(width, height), Point2::new(0.0, height))),
            Segment2D::Line(Line2D::new(Point2::new(0.0, height), Point2::new(0.0, 0.0))),
        ];

        Self {
            segments,
            closed: true,
        }
    }

    /// Create a circular path centered at origin
    pub fn circle(radius: f64) -> Self {
        Self {
            segments: vec![Segment2D::Circle(Circle2D::new(Point2::origin(), radius))],
            closed: true,
        }
    }

    /// Add a segment to this path
    pub fn push(&mut self, segment: Segment2D) {
        self.segments.push(segment);
    }

    /// Check if this path is empty
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Get the number of segments in this path
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Tessellate this path into a polyline
    ///
    /// # Arguments
    /// * `tolerance` - Maximum deviation from true curve
    ///
    /// # Returns
    /// A vector of points representing the tessellated path.
    /// If the path is closed, the last point connects back to the first.
    pub fn tessellate(&self, tolerance: f64) -> Vec<Point2<f64>> {
        let mut points = Vec::new();

        for segment in &self.segments {
            let segment_points = segment.tessellate(tolerance);
            points.extend(segment_points);
        }

        // Add the final endpoint if there are segments
        if let Some(last) = self.segments.last() {
            points.push(last.finish());
        }

        points
    }

    /// Calculate the total length of this path
    pub fn length(&self) -> f64 {
        // Approximate using tessellation
        let points = self.tessellate(0.01);
        points
            .windows(2)
            .map(|w| ((w[1].x - w[0].x).powi(2) + (w[1].y - w[0].y).powi(2)).sqrt())
            .sum()
    }

    /// Get the area enclosed by this path (if closed)
    ///
    /// Uses the shoelace formula on the tessellated polygon.
    /// Returns 0 for open paths.
    pub fn area(&self) -> f64 {
        if !self.closed {
            return 0.0;
        }

        let points = self.tessellate(0.01);
        if points.len() < 3 {
            return 0.0;
        }

        // Shoelace formula
        let mut sum = 0.0;
        for i in 0..points.len() {
            let j = (i + 1) % points.len();
            sum += points[i].x * points[j].y;
            sum -= points[j].x * points[i].y;
        }

        (sum / 2.0).abs()
    }
}

/// A 3D path consisting of connected curve segments
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Path3D {
    /// The curve segments that make up this path
    pub segments: Vec<Segment3D>,
    /// Whether this path forms a closed loop
    pub closed: bool,
}

impl Path3D {
    /// Create a new empty path
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a path from segments
    pub fn from_segments(segments: Vec<Segment3D>, closed: bool) -> Self {
        Self { segments, closed }
    }

    /// Add a segment to this path
    pub fn push(&mut self, segment: Segment3D) {
        self.segments.push(segment);
    }

    /// Check if this path is empty
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Get the number of segments in this path
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Tessellate this path into a polyline
    pub fn tessellate(&self, tolerance: f64) -> Vec<Point3<f64>> {
        let mut points = Vec::new();

        for segment in &self.segments {
            let segment_points = segment.tessellate(tolerance);
            points.extend(segment_points);
        }

        // Add the final endpoint
        if let Some(last) = self.segments.last() {
            points.push(last.finish());
        }

        points
    }

    /// Calculate the total length of this path
    pub fn length(&self) -> f64 {
        let points = self.tessellate(0.01);
        points.windows(2).map(|w| (w[1] - w[0]).norm()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rectangle() {
        let path = Path2D::rectangle(10.0, 5.0);
        assert_eq!(path.segments.len(), 4);
        assert!(path.closed);

        // Check perimeter (approximation via tessellation)
        assert_relative_eq!(path.length(), 30.0, epsilon = 0.1);
    }

    #[test]
    fn test_circle() {
        let path = Path2D::circle(10.0);
        assert_eq!(path.segments.len(), 1);
        assert!(path.closed);

        // Circumference should be 2*pi*r
        let expected = 2.0 * std::f64::consts::PI * 10.0;
        assert_relative_eq!(path.length(), expected, epsilon = 0.1);
    }

    #[test]
    fn test_area() {
        let path = Path2D::rectangle(10.0, 5.0);
        assert_relative_eq!(path.area(), 50.0, epsilon = 0.1);

        let circle = Path2D::circle(10.0);
        let expected_area = std::f64::consts::PI * 100.0;
        assert_relative_eq!(circle.area(), expected_area, epsilon = 1.0);
    }

    #[test]
    fn test_tessellate() {
        let path = Path2D::rectangle(10.0, 5.0);
        let points = path.tessellate(0.1);

        // Should have 4 corners + final endpoint
        assert_eq!(points.len(), 5);
    }

    #[test]
    fn test_from_svg() {
        let path = Path2D::from_svg("M0,0 L10,0 L10,5 L0,5 Z").unwrap();
        assert_eq!(path.segments.len(), 4);
        assert!(path.closed);
    }

    #[test]
    fn test_to_svg() {
        let path = Path2D::rectangle(10.0, 5.0);
        let svg = path.to_svg();

        // Should contain move and line commands
        assert!(svg.contains('M'));
        assert!(svg.contains('L'));
        assert!(svg.contains('Z'));
    }

    #[test]
    fn test_segment_2d_start_finish() {
        let line = Segment2D::Line(Line2D::new(Point2::new(0.0, 0.0), Point2::new(10.0, 5.0)));
        assert_eq!(line.start(), Point2::new(0.0, 0.0));
        assert_eq!(line.finish(), Point2::new(10.0, 5.0));
    }

    #[test]
    fn test_path_3d() {
        let mut path = Path3D::new();
        path.push(Segment3D::Line(Line3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        )));
        path.push(Segment3D::Line(Line3D::new(
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
        )));

        assert_eq!(path.len(), 2);
        assert_relative_eq!(path.length(), 20.0, epsilon = 0.1);
    }
}
