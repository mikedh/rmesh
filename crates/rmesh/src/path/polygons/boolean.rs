//! Boolean operations on Polygon2D via i_overlay.

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;
use i_overlay::mesh::outline::offset::OutlineOffset;
use i_overlay::mesh::style::{LineJoin, OutlineStyle};

use super::{Polygon2D, Shape};

impl Polygon2D {
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
        let mut polys = super::shapes_to_polygons(&result, None);
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
        let mut polys = super::shapes_to_polygons(&result, None);
        // Propagate to_3d from self to results
        for p in &mut polys {
            p.to_3d = self.to_3d;
        }
        polys
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Point2;

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
