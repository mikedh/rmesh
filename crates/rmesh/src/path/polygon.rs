//! Polygon2D with hole support using i_overlay
//!
//! This module provides a Polygon2D structure that properly handles
//! exterior rings and interior holes, using i_overlay for enclosure detection.

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;
use i_overlay::mesh::outline::offset::OutlineOffset;
use i_overlay::mesh::style::{LineJoin, OutlineStyle};
use nalgebra::{Matrix4, Point2, Point3};
use serde::{Deserialize, Serialize};

use super::{Line, Path2D, Path3D, Segment3D};

/// Type alias for i_overlay contours
type Contour = Vec<[f64; 2]>;
type Contours = Vec<Contour>;
type Shape = Vec<Contour>;

/// A 2D polygon with exterior boundary and interior holes
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Polygon2D {
    /// Exterior ring (CCW orientation for positive area)
    pub exterior: Vec<Point2<f64>>,
    /// Interior rings / holes (CW orientation)
    pub interiors: Vec<Vec<Point2<f64>>>,
    /// Optional transform from 2D polygon space back to 3D
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_3d: Option<Matrix4<f64>>,
}

impl Polygon2D {
    /// Create a new polygon with just an exterior ring
    pub fn new(exterior: Vec<Point2<f64>>) -> Self {
        Self {
            exterior,
            interiors: Vec::new(),
            to_3d: None,
        }
    }

    /// Create a polygon with exterior and interior rings
    pub fn with_holes(exterior: Vec<Point2<f64>>, interiors: Vec<Vec<Point2<f64>>>) -> Self {
        Self {
            exterior,
            interiors,
            to_3d: None,
        }
    }

    /// Calculate the area of this polygon (exterior minus holes)
    pub fn area(&self) -> f64 {
        let exterior_area = signed_area(&self.exterior).abs();
        let holes_area: f64 = self.interiors.iter().map(|h| signed_area(h).abs()).sum();
        exterior_area - holes_area
    }

    /// Check if a point is inside this polygon
    pub fn contains(&self, point: Point2<f64>) -> bool {
        // Point must be inside exterior
        if !point_in_polygon(point, &self.exterior) {
            return false;
        }

        // Point must not be inside any hole
        for hole in &self.interiors {
            if point_in_polygon(point, hole) {
                return false;
            }
        }

        true
    }

    /// Get the axis-aligned bounding box
    pub fn bounds(&self) -> Option<(Point2<f64>, Point2<f64>)> {
        if self.exterior.is_empty() {
            return None;
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for p in &self.exterior {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        Some((Point2::new(min_x, min_y), Point2::new(max_x, max_y)))
    }

    /// Get the number of vertices in the exterior ring
    pub fn num_exterior_vertices(&self) -> usize {
        self.exterior.len()
    }

    /// Get the number of holes
    pub fn num_holes(&self) -> usize {
        self.interiors.len()
    }

    /// Convert to i_overlay shape format (exterior + holes as contour list).
    fn to_shape(&self) -> Shape {
        let mut contours = Vec::with_capacity(1 + self.interiors.len());
        contours.push(self.exterior.iter().map(|p| [p.x, p.y]).collect());
        for hole in &self.interiors {
            contours.push(hole.iter().map(|p| [p.x, p.y]).collect());
        }
        contours
    }

    /// Perform a boolean operation between this polygon and another.
    fn boolean(&self, other: &Polygon2D, rule: OverlayRule) -> Vec<Polygon2D> {
        let subj = self.to_shape();
        let clip = other.to_shape();
        let result = subj.overlay(&clip, rule, FillRule::EvenOdd);
        let mut polys = shapes_to_polygons(&result, None);
        // Propagate to_3d from self to results
        for p in &mut polys {
            p.to_3d = self.to_3d;
        }
        polys
    }

    /// Compute the union of this polygon with another.
    pub fn union(&self, other: &Polygon2D) -> Vec<Polygon2D> {
        self.boolean(other, OverlayRule::Union)
    }

    /// Compute the difference of this polygon minus another.
    pub fn difference(&self, other: &Polygon2D) -> Vec<Polygon2D> {
        self.boolean(other, OverlayRule::Difference)
    }

    /// Compute the intersection of this polygon with another.
    pub fn intersection(&self, other: &Polygon2D) -> Vec<Polygon2D> {
        self.boolean(other, OverlayRule::Intersect)
    }

    /// Offset the polygon boundary by `distance`.
    ///
    /// Positive = expand outward, negative = shrink inward.
    /// Returns empty vec if polygon shrinks to nothing.
    pub fn buffer(&self, distance: f64) -> Vec<Polygon2D> {
        let shape = self.to_shape();
        let style = OutlineStyle::new(distance).line_join(LineJoin::Round(0.1));
        let result = shape.outline(&style);
        let mut polys = shapes_to_polygons(&result, None);
        // Propagate to_3d from self to results
        for p in &mut polys {
            p.to_3d = self.to_3d;
        }
        polys
    }

    /// Convert this polygon to a `Path2D` with Line segments tracing each ring.
    pub fn to_path2d(&self) -> Path2D {
        let mut path = Path2D::new();

        let mut add_ring = |ring: &[Point2<f64>]| {
            if ring.len() < 2 {
                return;
            }
            let start = path.vertices.len();
            for p in ring {
                path.add_vertex(*p);
            }
            let end = path.vertices.len();
            for i in start..end {
                let j = if i + 1 < end { i + 1 } else { start };
                path.push(super::Segment2D::Line(Line::new(i, j)));
            }
        };

        add_ring(&self.exterior);
        for interior in &self.interiors {
            add_ring(interior);
        }
        path.to_3d = self.to_3d;
        path
    }

    /// Lift this 2D polygon back to 3D using the stored `to_3d` transform.
    ///
    /// Each ring (exterior + interiors) becomes a closed loop of `Line` segments
    /// in the returned `Path3D`. Returns `None` if no `to_3d` transform is set.
    pub fn to_path3d(&self) -> Option<Path3D> {
        let to_3d = self.to_3d?;

        let mut path = Path3D::new();

        // Helper: add a closed ring of Line segments
        let mut add_ring = |ring: &[Point2<f64>]| {
            if ring.len() < 2 {
                return;
            }
            let start = path.vertices.len();
            for p in ring {
                path.add_vertex(to_3d.transform_point(&Point3::new(p.x, p.y, 0.0)));
            }
            let end = path.vertices.len();
            for i in start..end {
                let j = if i + 1 < end { i + 1 } else { start };
                path.push(Segment3D::Line(Line::new(i, j)));
            }
        };

        add_ring(&self.exterior);
        for interior in &self.interiors {
            add_ring(interior);
        }

        Some(path)
    }
}

/// Calculate the signed area of a polygon ring using the shoelace formula
///
/// Returns positive for CCW winding, negative for CW winding.
pub fn signed_area(points: &[Point2<f64>]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut sum = 0.0;
    let n = points.len();
    for i in 0..n {
        let j = (i + 1) % n;
        sum += points[i].x * points[j].y;
        sum -= points[j].x * points[i].y;
    }

    sum / 2.0
}

/// Check if a point is inside a polygon using ray casting
fn point_in_polygon(point: Point2<f64>, polygon: &[Point2<f64>]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let mut inside = false;
    let n = polygon.len();

    let mut j = n - 1;
    for i in 0..n {
        let pi = polygon[i];
        let pj = polygon[j];

        if ((pi.y > point.y) != (pj.y > point.y))
            && (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)
        {
            inside = !inside;
        }
        j = i;
    }

    inside
}

/// Convert discretized rings from a Path2D into Polygon2D structures
///
/// Uses i_overlay to determine enclosure relationships and assign holes to shells.
/// The tolerance parameter is used for ring discretization.
pub fn polygons_from_path(path: &Path2D, _tolerance: f64) -> Vec<Polygon2D> {
    // Note: tolerance is passed but Path2D now uses its internal deviation
    // The caller (Path2D::polygons) sets path.deviation before calling if needed
    let rings = path.rings_discrete();
    if rings.is_empty() {
        return Vec::new();
    }

    // If there's only one ring, it's just a simple polygon
    if rings.len() == 1 {
        let mut exterior = rings.into_iter().next().unwrap();
        // Ensure CCW orientation
        if signed_area(&exterior) < 0.0 {
            exterior.reverse();
        }
        return vec![Polygon2D {
            exterior,
            interiors: Vec::new(),
            to_3d: path.to_3d,
        }];
    }

    // Use i_overlay to handle enclosure detection and hole assignment
    polygons_from_rings_overlay(&rings, path.to_3d)
}

/// Use i_overlay to build polygons with proper hole assignment
fn polygons_from_rings_overlay(
    rings: &[Vec<Point2<f64>>],
    to_3d: Option<Matrix4<f64>>,
) -> Vec<Polygon2D> {
    if rings.is_empty() {
        return Vec::new();
    }

    // Convert rings to i_overlay format: Vec<Vec<[f64; 2]>>
    let contours: Contours = rings
        .iter()
        .filter(|ring| ring.len() >= 3)
        .map(|ring| ring.iter().map(|p| [p.x, p.y]).collect())
        .collect();

    if contours.is_empty() {
        return Vec::new();
    }

    // Use SingleFloatOverlay - overlay with an empty shape to extract normalized polygons
    let empty: Contours = vec![];
    let result = contours.overlay(&empty, OverlayRule::Subject, FillRule::EvenOdd);

    shapes_to_polygons(&result, to_3d)
}

/// Convert i_overlay shapes result to `Vec<Polygon2D>`.
///
/// Each shape is `[exterior, hole0, hole1, ...]`.
fn shapes_to_polygons(shapes: &[Shape], to_3d: Option<Matrix4<f64>>) -> Vec<Polygon2D> {
    shapes
        .iter()
        .filter(|s| !s.is_empty() && !s[0].is_empty())
        .map(|shape| {
            let exterior = shape[0].iter().map(|p| Point2::new(p[0], p[1])).collect();
            let interiors = shape[1..]
                .iter()
                .map(|c| c.iter().map(|p| Point2::new(p[0], p[1])).collect())
                .collect();
            Polygon2D {
                exterior,
                interiors,
                to_3d,
            }
        })
        .collect()
}

/// Create a simple Polygon2D from a list of points (no holes)
pub fn polygon_from_points(points: Vec<Point2<f64>>) -> Option<Polygon2D> {
    if points.len() < 3 {
        return None;
    }

    let mut exterior = points;

    // Ensure CCW orientation
    if signed_area(&exterior) < 0.0 {
        exterior.reverse();
    }

    Some(Polygon2D::new(exterior))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_signed_area_ccw() {
        // CCW square
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        assert_relative_eq!(signed_area(&points), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_signed_area_cw() {
        // CW square
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 0.0),
        ];
        assert_relative_eq!(signed_area(&points), -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polygon_area() {
        let polygon = Polygon2D::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);
        assert_relative_eq!(polygon.area(), 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polygon_with_hole() {
        let exterior = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];

        // Hole (CW orientation)
        let hole = vec![
            Point2::new(2.0, 2.0),
            Point2::new(2.0, 8.0),
            Point2::new(8.0, 8.0),
            Point2::new(8.0, 2.0),
        ];

        let polygon = Polygon2D::with_holes(exterior, vec![hole]);

        // Exterior area = 100, hole area = 36
        assert_relative_eq!(polygon.area(), 64.0, epsilon = 1e-10);
    }

    #[test]
    fn test_point_in_polygon() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];

        assert!(point_in_polygon(Point2::new(5.0, 5.0), &square));
        assert!(!point_in_polygon(Point2::new(15.0, 5.0), &square));
        assert!(!point_in_polygon(Point2::new(-1.0, 5.0), &square));
    }

    #[test]
    fn test_polygon_contains() {
        let exterior = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];

        let hole = vec![
            Point2::new(2.0, 2.0),
            Point2::new(2.0, 8.0),
            Point2::new(8.0, 8.0),
            Point2::new(8.0, 2.0),
        ];

        let polygon = Polygon2D::with_holes(exterior, vec![hole]);

        // Point in exterior but not in hole
        assert!(polygon.contains(Point2::new(1.0, 1.0)));

        // Point in hole
        assert!(!polygon.contains(Point2::new(5.0, 5.0)));

        // Point outside
        assert!(!polygon.contains(Point2::new(15.0, 5.0)));
    }

    #[test]
    fn test_bounds() {
        let polygon = Polygon2D::new(vec![
            Point2::new(1.0, 2.0),
            Point2::new(5.0, 2.0),
            Point2::new(5.0, 8.0),
            Point2::new(1.0, 8.0),
        ]);

        let (min, max) = polygon.bounds().unwrap();
        assert_relative_eq!(min.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(min.y, 2.0, epsilon = 1e-10);
        assert_relative_eq!(max.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(max.y, 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polygons_from_path() {
        let path = Path2D::rectangle(10.0, 10.0);
        let polygons = path.polygons();

        assert_eq!(polygons.len(), 1);
        assert_relative_eq!(polygons[0].area(), 100.0, epsilon = 0.1);
    }

    /// Helper: create a CCW square polygon at (x, y) with given side length.
    fn make_square(x: f64, y: f64, size: f64) -> Polygon2D {
        Polygon2D::new(vec![
            Point2::new(x, y),
            Point2::new(x + size, y),
            Point2::new(x + size, y + size),
            Point2::new(x, y + size),
        ])
    }

    #[test]
    fn test_union_overlapping() {
        // Two overlapping unit squares: [0,1]x[0,1] and [0.5,1.5]x[0,1]
        let a = make_square(0.0, 0.0, 1.0);
        let b = make_square(0.5, 0.0, 1.0);

        let result = a.union(&b);
        assert_eq!(result.len(), 1);

        let area: f64 = result.iter().map(|p| p.area()).sum();
        // Union of two overlapping unit squares = 1.5
        assert_relative_eq!(area, 1.5, epsilon = 1e-6);
        assert!(area > a.area());
        assert!(area > b.area());
    }

    #[test]
    fn test_difference() {
        // Large square minus small centered square → polygon with hole
        let outer = make_square(0.0, 0.0, 10.0);
        let inner = make_square(2.0, 2.0, 6.0);

        let result = outer.difference(&inner);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_holes(), 1);

        let area: f64 = result.iter().map(|p| p.area()).sum();
        // 100 - 36 = 64
        assert_relative_eq!(area, 64.0, epsilon = 1e-6);
    }

    #[test]
    fn test_intersection() {
        // Two overlapping unit squares
        let a = make_square(0.0, 0.0, 1.0);
        let b = make_square(0.5, 0.0, 1.0);

        let result = a.intersection(&b);
        assert_eq!(result.len(), 1);

        let area: f64 = result.iter().map(|p| p.area()).sum();
        // Overlap region = 0.5 x 1.0 = 0.5
        assert_relative_eq!(area, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_buffer_positive() {
        let square = make_square(0.0, 0.0, 10.0);
        let original_area = square.area();

        let result = square.buffer(1.0);
        assert!(!result.is_empty());

        let area: f64 = result.iter().map(|p| p.area()).sum();
        assert!(area > original_area);
    }

    #[test]
    fn test_buffer_negative() {
        let square = make_square(0.0, 0.0, 10.0);
        let original_area = square.area();

        let result = square.buffer(-1.0);
        assert!(!result.is_empty());

        let area: f64 = result.iter().map(|p| p.area()).sum();
        assert!(area < original_area);
    }

    #[test]
    fn test_buffer_collapse() {
        // A small square buffered by a large negative → nothing left
        let square = make_square(0.0, 0.0, 2.0);

        let result = square.buffer(-10.0);
        assert!(result.is_empty());
    }
}
