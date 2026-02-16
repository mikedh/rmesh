//! Axis-aligned bounding boxes for 2D and 3D geometry.

use nalgebra::{Matrix4, Point2, Point3, Vector4};

/// 2D axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds2 {
    pub min: Point2<f64>,
    pub max: Point2<f64>,
}

/// 3D axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds3 {
    pub min: Point3<f64>,
    pub max: Point3<f64>,
}

// ============================================================================
// Bounds2
// ============================================================================

impl Bounds2 {
    /// Create from explicit min/max points (no validation).
    pub fn new(min: Point2<f64>, max: Point2<f64>) -> Self {
        Self { min, max }
    }

    /// Empty bounds (MAX/MIN sentinels) for accumulation patterns.
    pub fn empty() -> Self {
        Self {
            min: Point2::new(f64::MAX, f64::MAX),
            max: Point2::new(f64::MIN, f64::MIN),
        }
    }

    /// Compute bounds from a point slice. Returns `None` if empty.
    pub fn from_points(points: &[Point2<f64>]) -> Option<Self> {
        let (first, rest) = points.split_first()?;
        let mut min = *first;
        let mut max = *first;
        for p in rest {
            min = min.inf(p);
            max = max.sup(p);
        }
        Some(Self { min, max })
    }

    /// Compute bounds from an iterator of points. Returns `None` if empty.
    #[allow(clippy::should_implement_trait)]
    // Returns Option<Self> which FromIterator cannot express.
    pub fn from_iter(iter: impl Iterator<Item = Point2<f64>>) -> Option<Self> {
        let mut bounds = Self::empty();
        let mut found = false;
        for p in iter {
            bounds.include_point(&p);
            found = true;
        }
        if found { Some(bounds) } else { None }
    }

    /// Union of two bounding boxes.
    #[must_use]
    pub fn union(&self, other: &Bounds2) -> Bounds2 {
        Bounds2 {
            min: self.min.inf(&other.min),
            max: self.max.sup(&other.max),
        }
    }

    /// Expand to include a point.
    pub fn include_point(&mut self, p: &Point2<f64>) {
        self.min = self.min.inf(p);
        self.max = self.max.sup(p);
    }

    /// Extents (width, height) as a vector.
    pub fn extents(&self) -> nalgebra::Vector2<f64> {
        self.max - self.min
    }

    /// Center point of the bounding box.
    pub fn center(&self) -> Point2<f64> {
        nalgebra::center(&self.min, &self.max)
    }

    /// Length of the diagonal.
    pub fn diagonal(&self) -> f64 {
        (self.max - self.min).norm()
    }

    /// Whether this box overlaps another.
    pub fn overlaps(&self, other: &Bounds2) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Whether this box contains a point.
    pub fn contains_point(&self, p: &Point2<f64>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    /// The 4 corners of the bounding box.
    pub fn corners(&self) -> [Point2<f64>; 4] {
        [
            Point2::new(self.min.x, self.min.y),
            Point2::new(self.max.x, self.min.y),
            Point2::new(self.min.x, self.max.y),
            Point2::new(self.max.x, self.max.y),
        ]
    }

    /// Whether this is an empty (sentinel) bounding box.
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y
    }

    /// Convert to `Option`, returning `None` if empty.
    pub fn to_option(self) -> Option<Self> {
        if self.is_empty() { None } else { Some(self) }
    }

    /// Area of the bounding box.
    pub fn area(&self) -> f64 {
        let e = self.extents();
        e.x * e.y
    }
}

// ============================================================================
// Bounds3
// ============================================================================

impl Bounds3 {
    /// Create from explicit min/max points (no validation).
    pub fn new(min: Point3<f64>, max: Point3<f64>) -> Self {
        Self { min, max }
    }

    /// Empty bounds (MAX/MIN sentinels) for accumulation patterns.
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f64::MAX, f64::MAX, f64::MAX),
            max: Point3::new(f64::MIN, f64::MIN, f64::MIN),
        }
    }

    /// Compute bounds from a point slice. Returns `None` if empty.
    pub fn from_points(points: &[Point3<f64>]) -> Option<Self> {
        let (first, rest) = points.split_first()?;
        let mut min = *first;
        let mut max = *first;
        for p in rest {
            min = min.inf(p);
            max = max.sup(p);
        }
        Some(Self { min, max })
    }

    /// Compute bounds from an iterator of points. Returns `None` if empty.
    #[allow(clippy::should_implement_trait)]
    // Returns Option<Self> which FromIterator cannot express.
    pub fn from_iter(iter: impl Iterator<Item = Point3<f64>>) -> Option<Self> {
        let mut bounds = Self::empty();
        let mut found = false;
        for p in iter {
            bounds.include_point(&p);
            found = true;
        }
        if found { Some(bounds) } else { None }
    }

    /// Compute bounds from a large point slice using parallel reduction.
    pub fn from_points_par(points: &[Point3<f64>]) -> Option<Self> {
        use rayon::prelude::*;
        if points.is_empty() {
            return None;
        }
        let (min, max) = points.par_iter().map(|v| (*v, *v)).reduce(
            || (points[0], points[0]),
            |(l1, u1), (l2, u2)| (l1.inf(&l2), u1.sup(&u2)),
        );
        Some(Self { min, max })
    }

    /// Union of two bounding boxes.
    #[must_use]
    pub fn union(&self, other: &Bounds3) -> Bounds3 {
        Bounds3 {
            min: self.min.inf(&other.min),
            max: self.max.sup(&other.max),
        }
    }

    /// Expand to include a point.
    pub fn include_point(&mut self, p: &Point3<f64>) {
        self.min = self.min.inf(p);
        self.max = self.max.sup(p);
    }

    /// Extents (width, height, depth) as a vector.
    pub fn extents(&self) -> nalgebra::Vector3<f64> {
        self.max - self.min
    }

    /// Center point of the bounding box.
    pub fn center(&self) -> Point3<f64> {
        nalgebra::center(&self.min, &self.max)
    }

    /// Length of the diagonal.
    pub fn diagonal(&self) -> f64 {
        (self.max - self.min).norm()
    }

    /// Whether this box overlaps another.
    pub fn overlaps(&self, other: &Bounds3) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Whether this box contains a point.
    pub fn contains_point(&self, p: &Point3<f64>) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// The 8 corners of the bounding box.
    pub fn corners(&self) -> [Point3<f64>; 8] {
        [
            Point3::new(self.min.x, self.min.y, self.min.z),
            Point3::new(self.max.x, self.min.y, self.min.z),
            Point3::new(self.min.x, self.max.y, self.min.z),
            Point3::new(self.max.x, self.max.y, self.min.z),
            Point3::new(self.min.x, self.min.y, self.max.z),
            Point3::new(self.max.x, self.min.y, self.max.z),
            Point3::new(self.min.x, self.max.y, self.max.z),
            Point3::new(self.max.x, self.max.y, self.max.z),
        ]
    }

    /// Whether this is an empty (sentinel) bounding box.
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    /// Convert to `Option`, returning `None` if empty.
    pub fn to_option(self) -> Option<Self> {
        if self.is_empty() { None } else { Some(self) }
    }

    /// Volume of the bounding box.
    pub fn volume(&self) -> f64 {
        let e = self.extents();
        e.x * e.y * e.z
    }

    /// Transform all 8 corners by a 4x4 matrix and compute the AABB.
    #[must_use]
    pub fn transformed(&self, m: &Matrix4<f64>) -> Bounds3 {
        let mut result = Bounds3::empty();
        for corner in &self.corners() {
            let v = m * Vector4::new(corner.x, corner.y, corner.z, 1.0);
            result.include_point(&Point3::new(v.x, v.y, v.z));
        }
        result
    }

    /// Intersection of two bounding boxes (may be empty).
    #[must_use]
    pub fn intersection(&self, other: &Bounds3) -> Bounds3 {
        Bounds3 {
            min: self.min.sup(&other.min),
            max: self.max.inf(&other.max),
        }
    }

    /// Convert to pair of `[f64; 3]` arrays.
    pub fn to_arrays(&self) -> ([f64; 3], [f64; 3]) {
        (
            [self.min.x, self.min.y, self.min.z],
            [self.max.x, self.max.y, self.max.z],
        )
    }

    /// Create from pair of `[f64; 3]` arrays.
    pub fn from_arrays(min: [f64; 3], max: [f64; 3]) -> Self {
        Self {
            min: Point3::new(min[0], min[1], min[2]),
            max: Point3::new(max[0], max[1], max[2]),
        }
    }

    /// Create a 3D bounding box from a 2D bounding box (z = 0).
    pub fn from_bounds2(b: Bounds2) -> Self {
        Self {
            min: Point3::new(b.min.x, b.min.y, 0.0),
            max: Point3::new(b.max.x, b.max.y, 0.0),
        }
    }
}

// ============================================================================
// Free functions
// ============================================================================

/// Compute 2D bounds from a point slice. Returns `None` if empty.
pub fn bounds2(points: &[Point2<f64>]) -> Option<Bounds2> {
    Bounds2::from_points(points)
}

/// Compute 3D bounds from a point slice. Returns `None` if empty.
pub fn bounds3(points: &[Point3<f64>]) -> Option<Bounds3> {
    Bounds3::from_points(points)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // --- Bounds2 tests ---

    #[test]
    fn test_bounds2_from_points() {
        let points = vec![
            Point2::new(1.0, 2.0),
            Point2::new(5.0, 3.0),
            Point2::new(3.0, 8.0),
        ];
        let b = Bounds2::from_points(&points).unwrap();
        assert_eq!(b.min, Point2::new(1.0, 2.0));
        assert_eq!(b.max, Point2::new(5.0, 8.0));
    }

    #[test]
    fn test_bounds2_from_points_empty() {
        assert!(Bounds2::from_points(&[]).is_none());
    }

    #[test]
    fn test_bounds2_from_points_single() {
        let b = Bounds2::from_points(&[Point2::new(3.0, 4.0)]).unwrap();
        assert_eq!(b.min, Point2::new(3.0, 4.0));
        assert_eq!(b.max, Point2::new(3.0, 4.0));
    }

    #[test]
    fn test_bounds2_from_iter() {
        let points = vec![Point2::new(1.0, 2.0), Point2::new(5.0, 8.0)];
        let b = Bounds2::from_iter(points.into_iter()).unwrap();
        assert_eq!(b.min, Point2::new(1.0, 2.0));
        assert_eq!(b.max, Point2::new(5.0, 8.0));
    }

    #[test]
    fn test_bounds2_from_iter_empty() {
        assert!(Bounds2::from_iter(std::iter::empty()).is_none());
    }

    #[test]
    fn test_bounds2_empty_sentinel() {
        let b = Bounds2::empty();
        assert!(b.is_empty());
        assert!(b.to_option().is_none());
    }

    #[test]
    fn test_bounds2_include_point() {
        let mut b = Bounds2::empty();
        b.include_point(&Point2::new(1.0, 2.0));
        b.include_point(&Point2::new(5.0, 8.0));
        assert_eq!(b.min, Point2::new(1.0, 2.0));
        assert_eq!(b.max, Point2::new(5.0, 8.0));
        assert!(!b.is_empty());
    }

    #[test]
    fn test_bounds2_union() {
        let a = Bounds2::new(Point2::new(0.0, 0.0), Point2::new(2.0, 2.0));
        let b = Bounds2::new(Point2::new(1.0, 1.0), Point2::new(5.0, 5.0));
        let u = a.union(&b);
        assert_eq!(u.min, Point2::new(0.0, 0.0));
        assert_eq!(u.max, Point2::new(5.0, 5.0));
    }

    #[test]
    fn test_bounds2_extents() {
        let b = Bounds2::new(Point2::new(1.0, 2.0), Point2::new(4.0, 6.0));
        let e = b.extents();
        assert_relative_eq!(e.x, 3.0);
        assert_relative_eq!(e.y, 4.0);
    }

    #[test]
    fn test_bounds2_center() {
        let b = Bounds2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 6.0));
        let c = b.center();
        assert_relative_eq!(c.x, 2.0);
        assert_relative_eq!(c.y, 3.0);
    }

    #[test]
    fn test_bounds2_diagonal() {
        let b = Bounds2::new(Point2::new(0.0, 0.0), Point2::new(3.0, 4.0));
        assert_relative_eq!(b.diagonal(), 5.0);
    }

    #[test]
    fn test_bounds2_overlaps() {
        let a = Bounds2::new(Point2::new(0.0, 0.0), Point2::new(2.0, 2.0));
        let b = Bounds2::new(Point2::new(1.0, 1.0), Point2::new(3.0, 3.0));
        let c = Bounds2::new(Point2::new(5.0, 5.0), Point2::new(6.0, 6.0));
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_bounds2_contains_point() {
        let b = Bounds2::new(Point2::new(0.0, 0.0), Point2::new(2.0, 2.0));
        assert!(b.contains_point(&Point2::new(1.0, 1.0)));
        assert!(b.contains_point(&Point2::new(0.0, 0.0)));
        assert!(!b.contains_point(&Point2::new(3.0, 1.0)));
    }

    #[test]
    fn test_bounds2_corners() {
        let b = Bounds2::new(Point2::new(1.0, 2.0), Point2::new(3.0, 4.0));
        let c = b.corners();
        assert_eq!(c.len(), 4);
        assert!(c.contains(&Point2::new(1.0, 2.0)));
        assert!(c.contains(&Point2::new(3.0, 4.0)));
    }

    #[test]
    fn test_bounds2_area() {
        let b = Bounds2::new(Point2::new(0.0, 0.0), Point2::new(3.0, 4.0));
        assert_relative_eq!(b.area(), 12.0);
    }

    #[test]
    fn test_bounds2_free_function() {
        let points = vec![Point2::new(1.0, 2.0), Point2::new(5.0, 8.0)];
        let b = bounds2(&points).unwrap();
        assert_eq!(b.min, Point2::new(1.0, 2.0));
        assert_eq!(b.max, Point2::new(5.0, 8.0));
    }

    // --- Bounds3 tests ---

    #[test]
    fn test_bounds3_from_points() {
        let points = vec![
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(5.0, 3.0, 1.0),
            Point3::new(3.0, 8.0, 4.0),
        ];
        let b = Bounds3::from_points(&points).unwrap();
        assert_eq!(b.min, Point3::new(1.0, 2.0, 1.0));
        assert_eq!(b.max, Point3::new(5.0, 8.0, 4.0));
    }

    #[test]
    fn test_bounds3_from_points_empty() {
        assert!(Bounds3::from_points(&[]).is_none());
    }

    #[test]
    fn test_bounds3_from_points_single() {
        let b = Bounds3::from_points(&[Point3::new(3.0, 4.0, 5.0)]).unwrap();
        assert_eq!(b.min, Point3::new(3.0, 4.0, 5.0));
        assert_eq!(b.max, Point3::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn test_bounds3_from_iter() {
        let points = vec![Point3::new(1.0, 2.0, 3.0), Point3::new(5.0, 8.0, 9.0)];
        let b = Bounds3::from_iter(points.into_iter()).unwrap();
        assert_eq!(b.min, Point3::new(1.0, 2.0, 3.0));
        assert_eq!(b.max, Point3::new(5.0, 8.0, 9.0));
    }

    #[test]
    fn test_bounds3_from_iter_empty() {
        assert!(Bounds3::from_iter(std::iter::empty()).is_none());
    }

    #[test]
    fn test_bounds3_from_points_par() {
        let points: Vec<Point3<f64>> = (0..100)
            .map(|i| Point3::new(i as f64, (i * 2) as f64, (i * 3) as f64))
            .collect();
        let b = Bounds3::from_points_par(&points).unwrap();
        assert_eq!(b.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(b.max, Point3::new(99.0, 198.0, 297.0));
    }

    #[test]
    fn test_bounds3_from_points_par_empty() {
        assert!(Bounds3::from_points_par(&[]).is_none());
    }

    #[test]
    fn test_bounds3_empty_sentinel() {
        let b = Bounds3::empty();
        assert!(b.is_empty());
        assert!(b.to_option().is_none());
    }

    #[test]
    fn test_bounds3_include_point() {
        let mut b = Bounds3::empty();
        b.include_point(&Point3::new(1.0, 2.0, 3.0));
        b.include_point(&Point3::new(5.0, 8.0, 9.0));
        assert_eq!(b.min, Point3::new(1.0, 2.0, 3.0));
        assert_eq!(b.max, Point3::new(5.0, 8.0, 9.0));
        assert!(!b.is_empty());
    }

    #[test]
    fn test_bounds3_union() {
        let a = Bounds3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 2.0, 2.0));
        let b = Bounds3::new(Point3::new(1.0, 1.0, 1.0), Point3::new(5.0, 5.0, 5.0));
        let u = a.union(&b);
        assert_eq!(u.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(u.max, Point3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn test_bounds3_extents() {
        let b = Bounds3::new(Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 6.0, 10.0));
        let e = b.extents();
        assert_relative_eq!(e.x, 3.0);
        assert_relative_eq!(e.y, 4.0);
        assert_relative_eq!(e.z, 7.0);
    }

    #[test]
    fn test_bounds3_center() {
        let b = Bounds3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(4.0, 6.0, 8.0));
        let c = b.center();
        assert_relative_eq!(c.x, 2.0);
        assert_relative_eq!(c.y, 3.0);
        assert_relative_eq!(c.z, 4.0);
    }

    #[test]
    fn test_bounds3_diagonal() {
        let b = Bounds3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 2.0, 2.0));
        assert_relative_eq!(b.diagonal(), 3.0);
    }

    #[test]
    fn test_bounds3_overlaps() {
        let a = Bounds3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 2.0, 2.0));
        let b = Bounds3::new(Point3::new(1.0, 1.0, 1.0), Point3::new(3.0, 3.0, 3.0));
        let c = Bounds3::new(Point3::new(5.0, 5.0, 5.0), Point3::new(6.0, 6.0, 6.0));
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_bounds3_contains_point() {
        let b = Bounds3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 2.0, 2.0));
        assert!(b.contains_point(&Point3::new(1.0, 1.0, 1.0)));
        assert!(b.contains_point(&Point3::new(0.0, 0.0, 0.0)));
        assert!(!b.contains_point(&Point3::new(3.0, 1.0, 1.0)));
    }

    #[test]
    fn test_bounds3_corners() {
        let b = Bounds3::new(Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0));
        let c = b.corners();
        assert_eq!(c.len(), 8);
        assert!(c.contains(&Point3::new(1.0, 2.0, 3.0)));
        assert!(c.contains(&Point3::new(4.0, 5.0, 6.0)));
    }

    #[test]
    fn test_bounds3_volume() {
        let b = Bounds3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 4.0, 5.0));
        assert_relative_eq!(b.volume(), 60.0);
    }

    #[test]
    fn test_bounds3_transformed_identity() {
        let b = Bounds3::new(Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0));
        let t = b.transformed(&Matrix4::identity());
        assert_relative_eq!(t.min.x, b.min.x, epsilon = 1e-10);
        assert_relative_eq!(t.min.y, b.min.y, epsilon = 1e-10);
        assert_relative_eq!(t.min.z, b.min.z, epsilon = 1e-10);
        assert_relative_eq!(t.max.x, b.max.x, epsilon = 1e-10);
        assert_relative_eq!(t.max.y, b.max.y, epsilon = 1e-10);
        assert_relative_eq!(t.max.z, b.max.z, epsilon = 1e-10);
    }

    #[test]
    fn test_bounds3_transformed_translation() {
        let b = Bounds3::new(Point3::origin(), Point3::new(1.0, 1.0, 1.0));
        let mut m = Matrix4::identity();
        m[(0, 3)] = 10.0;
        m[(1, 3)] = 20.0;
        m[(2, 3)] = 30.0;
        let t = b.transformed(&m);
        assert_relative_eq!(t.min.x, 10.0, epsilon = 1e-10);
        assert_relative_eq!(t.min.y, 20.0, epsilon = 1e-10);
        assert_relative_eq!(t.min.z, 30.0, epsilon = 1e-10);
        assert_relative_eq!(t.max.x, 11.0, epsilon = 1e-10);
        assert_relative_eq!(t.max.y, 21.0, epsilon = 1e-10);
        assert_relative_eq!(t.max.z, 31.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bounds3_intersection() {
        let a = Bounds3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 3.0, 3.0));
        let b = Bounds3::new(Point3::new(1.0, 1.0, 1.0), Point3::new(5.0, 5.0, 5.0));
        let i = a.intersection(&b);
        assert_eq!(i.min, Point3::new(1.0, 1.0, 1.0));
        assert_eq!(i.max, Point3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_bounds3_intersection_empty() {
        let a = Bounds3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let b = Bounds3::new(Point3::new(5.0, 5.0, 5.0), Point3::new(6.0, 6.0, 6.0));
        let i = a.intersection(&b);
        assert!(i.is_empty());
    }

    #[test]
    fn test_bounds3_to_from_arrays() {
        let b = Bounds3::new(Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0));
        let (min_arr, max_arr) = b.to_arrays();
        assert_eq!(min_arr, [1.0, 2.0, 3.0]);
        assert_eq!(max_arr, [4.0, 5.0, 6.0]);

        let b2 = Bounds3::from_arrays(min_arr, max_arr);
        assert_eq!(b, b2);
    }

    #[test]
    fn test_bounds3_from_bounds2() {
        let b2 = Bounds2::new(Point2::new(1.0, 2.0), Point2::new(3.0, 4.0));
        let b3 = Bounds3::from_bounds2(b2);
        assert_eq!(b3.min, Point3::new(1.0, 2.0, 0.0));
        assert_eq!(b3.max, Point3::new(3.0, 4.0, 0.0));
    }

    #[test]
    fn test_bounds3_free_function() {
        let points = vec![Point3::new(1.0, 2.0, 3.0), Point3::new(5.0, 8.0, 9.0)];
        let b = bounds3(&points).unwrap();
        assert_eq!(b.min, Point3::new(1.0, 2.0, 3.0));
        assert_eq!(b.max, Point3::new(5.0, 8.0, 9.0));
    }
}
