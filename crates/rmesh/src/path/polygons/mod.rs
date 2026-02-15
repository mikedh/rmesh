//! Polygon2D with hole support using i_overlay
//!
//! This module provides a Polygon2D structure that properly handles
//! exterior rings and interior holes, using i_overlay for enclosure detection.

mod boolean;
pub(crate) mod query;

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;
use nalgebra::{Matrix4, Point2, Point3};
use serde::{Deserialize, Serialize};

use super::{Line, Path2D, Path3D, Segment3D};

pub(crate) use query::point_in_polygon;

/// Type alias for i_overlay contours
pub(crate) type Contour = Vec<[f64; 2]>;
pub(crate) type Contours = Vec<Contour>;
pub(crate) type Shape = Vec<Contour>;

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

    /// Batch check if points are inside this polygon.
    ///
    /// Returns `Vec<bool>` parallel to `points` — true if inside the exterior
    /// and outside all holes.
    pub fn contains(&self, points: &[Point2<f64>]) -> Vec<bool> {
        let holes: Vec<&[Point2<f64>]> = self.interiors.iter().map(|h| h.as_slice()).collect();
        query::point_in_polygon(&self.exterior, &holes, points)
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

    /// Get the extents (width, height) of the bounding box.
    ///
    /// Returns None for empty polygons.
    pub fn extents(&self) -> Option<[f64; 2]> {
        self.bounds()
            .map(|(min, max)| [max.x - min.x, max.y - min.y])
    }

    /// Get the number of vertices in the exterior ring
    pub fn num_exterior_vertices(&self) -> usize {
        self.exterior.len()
    }

    /// Get the number of holes
    pub fn num_holes(&self) -> usize {
        self.interiors.len()
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
pub(crate) fn shapes_to_polygons(shapes: &[Shape], to_3d: Option<Matrix4<f64>>) -> Vec<Polygon2D> {
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

        assert!(point_in_polygon(&square, &[], &[Point2::new(5.0, 5.0)])[0]);
        assert!(!point_in_polygon(&square, &[], &[Point2::new(15.0, 5.0)])[0]);
        assert!(!point_in_polygon(&square, &[], &[Point2::new(-1.0, 5.0)])[0]);
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

        let p1 = Point2::new(1.0, 1.0);
        let p2 = Point2::new(5.0, 5.0);
        let p3 = Point2::new(15.0, 5.0);
        let r = polygon.contains(&[p1, p2, p3]);

        // Point in exterior but not in hole
        assert!(r[0]);
        // Point in hole
        assert!(!r[1]);
        // Point outside
        assert!(!r[2]);
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
}
