//! Path module for 2D and 3D curve operations
//!
//! This module provides path primitives with explicit 2D and 3D types.
//! Sketches use `Path2D`; sweep paths and edge curves use `Path3D`.
//!
//! # Structure
//!
//! - [`entity`] - Curve primitives: Arc, Line, Circle, Ellipse, Bezier, BSpline
//! - [`discretize`] - Discretization algorithms (De Casteljau, arc stepping)
//! - [`svg`] - SVG parsing/export for Path2D
//! - [`graph`] - Entity connectivity graph and ring detection
//! - [`polygons`] - Polygon2D with hole support using i_overlay
//!
//! # Indexed Vertex Storage
//!
//! Paths store vertices in a shared `vertices: Vec<P>` array, and segments
//! reference vertices by index. This enables:
//! - Efficient deduplication of shared vertices
//! - Direct index-based connectivity detection
//! - Simplified graph operations
//!
//! # Deviation (Tolerance)
//!
//! Discretization uses a **maximum chord height error** - the perpendicular
//! distance from the true curve to the approximating line segment. Smaller
//! values = more points, higher fidelity.
//!
//! By default, deviation is computed automatically from path extents:
//! `max_extent * DEVIATION_RATIO` (0.01% of the largest dimension).
//!
//! You can override this by setting the `deviation` field on Path2D/Path3D.
//!
//! # Example
//!
//! ```
//! use rmesh::path::{Path2D, Segment2D};
//! use nalgebra::Point2;
//!
//! // Create a simple rectangular path
//! let path = Path2D::rectangle(10.0, 5.0);
//!
//! // Or from SVG
//! let path = Path2D::from_svg("M0,0 L10,0 L10,5 L0,5 Z").unwrap();
//!
//! // Discretize to a single polyline
//! let points = path.discretize();
//!
//! // Or get per-segment polylines (for rendering)
//! let segments = path.to_segments();
//! ```

pub mod discretize;
pub mod entity;
pub mod graph;
pub mod polygons;
pub mod raster;
pub mod svg;

/// Default deviation ratio for curve discretization.
///
/// When tolerance is `None`, actual tolerance = `path.extent_max() * DEVIATION_RATIO`.
/// This ensures tolerance scales with geometry size.
///
/// Value of 1e-4 means max chord error is 0.01% of the largest path dimension.
pub const DEVIATION_RATIO: f64 = 1e-4;

/// Minimum tolerance floor to avoid infinite subdivision on tiny geometry.
const MIN_TOLERANCE: f64 = 1e-12;

/// Fallback tolerance for segments without extent information (1 micron).
///
/// Used when a segment is discretized standalone without path context.
/// Paths use scale-relative tolerance via `DEVIATION_RATIO` instead.
pub const DEFAULT_TOLERANCE: f64 = 0.001;

/// Resolve deviation: user override or computed from extents * DEVIATION_RATIO
macro_rules! resolve_deviation {
    ($self:expr) => {
        $self.deviation.unwrap_or_else(|| {
            ($self
                .extents()
                .map(|e| e.iter().copied().reduce(f64::max).unwrap_or(0.0))
                .unwrap_or(0.0)
                * DEVIATION_RATIO)
                .max(MIN_TOLERANCE)
        })
    };
}

use nalgebra::{Matrix4, Point2, Point3};
use serde::{Deserialize, Serialize};

use crate::bounds::{Bounds2, Bounds3};
use crate::cache::Cache;
use crate::creation::Triangulator;
use crate::project::EPSILON_RELATIVE;

/// Snap ring vertices radially onto the nearest expected circle.
///
/// For each vertex, finds the circle whose center is closest and projects
/// the vertex radially onto that circle if the radial error is below a
/// sagitta-based tolerance. This ensures `ring_to_segments` can detect
/// circle membership within `EPSILON_RELATIVE` after i_overlay buffering
/// introduces discretization error.
fn snap_ring_to_circles(ring: &mut [Point2<f64>], circles: &[(Point2<f64>, f64)]) {
    if circles.is_empty() {
        return;
    }
    for pt in ring.iter_mut() {
        // Find the circle whose center is nearest
        let mut best_ci = 0;
        let mut best_dist = f64::INFINITY;
        for (ci, (center, _)) in circles.iter().enumerate() {
            let d = (pt.x - center.x).powi(2) + (pt.y - center.y).powi(2);
            if d < best_dist {
                best_dist = d;
                best_ci = ci;
            }
        }
        let (center, radius) = circles[best_ci];
        let offset_x = pt.x - center.x;
        let offset_y = pt.y - center.y;
        let d = (offset_x * offset_x + offset_y * offset_y).sqrt();
        if d < 1e-15 {
            continue;
        }
        let radial_error = (d - radius).abs();
        // Snap if within 1% of radius (generous enough for polygon buffering)
        if radial_error < radius * 0.01 {
            let scale = radius / d;
            pt.x = center.x + offset_x * scale;
            pt.y = center.y + offset_y * scale;
        }
    }
}

// Re-export commonly used types
pub use entity::arc::{arc_center, arc_center_from_3_points};
pub use entity::{
    Arc2, Arc3, BSpline, Circle2, Circle3, CubicBezier, Curve, Ellipse2, Line, QuadraticBezier,
    Winding,
};
pub use graph::EntityGraph;
pub use polygons::Polygon2D;
pub use svg::SvgError;

// =============================================================================
// Segment enums - wrapping entity types for storage in paths
// =============================================================================

/// A 2D path segment (one curve element with vertex indices)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Segment2D {
    Line(Line),
    Arc(Arc2),
    Circle(Circle2),
    Ellipse(Ellipse2),
    CubicBezier(CubicBezier),
    QuadraticBezier(QuadraticBezier),
    BSpline(BSpline),
}

impl Segment2D {
    /// Get the start and finish vertex indices as [start, finish].
    ///
    /// Returns `None` if the segment is degenerate.
    pub fn end_indices(&self) -> Option<[usize; 2]> {
        use entity::Curve;
        match self {
            Segment2D::Line(l) => l.end_indices(),
            Segment2D::Arc(a) => a.end_indices(),
            Segment2D::Circle(c) => c.end_indices(),
            Segment2D::Ellipse(e) => e.end_indices(),
            Segment2D::CubicBezier(b) => b.end_indices(),
            Segment2D::QuadraticBezier(b) => b.end_indices(),
            Segment2D::BSpline(s) => s.end_indices(),
        }
    }

    /// Get the start vertex index
    pub fn start_index(&self) -> Option<usize> {
        self.end_indices().map(|[s, _]| s)
    }

    /// Get the end vertex index
    pub fn finish_index(&self) -> Option<usize> {
        self.end_indices().map(|[_, f]| f)
    }

    /// Get the start point using vertices array
    pub fn start(&self, vertices: &[Point2<f64>]) -> Option<Point2<f64>> {
        self.start_index().and_then(|i| vertices.get(i).copied())
    }

    /// Get the end point using vertices array
    pub fn finish(&self, vertices: &[Point2<f64>]) -> Option<Point2<f64>> {
        self.finish_index().and_then(|i| vertices.get(i).copied())
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

    /// Discretize this segment into a sequence of points
    ///
    /// # Arguments
    /// * `vertices` - The vertex array to look up indices
    /// * `tolerance` - Max chord height error
    pub fn discretize(&self, vertices: &[Point2<f64>], tolerance: f64) -> Vec<Point2<f64>> {
        discretize::discretize_segment_2d(self, vertices, tolerance)
    }
}

/// A 3D path segment (one curve element with vertex indices)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Segment3D {
    Line(Line),
    Arc(Arc3),
    Circle(Circle3),
    CubicBezier(CubicBezier),
    QuadraticBezier(QuadraticBezier),
    BSpline(BSpline),
}

impl Segment3D {
    /// Get the start and finish vertex indices as [start, finish].
    ///
    /// Returns `None` if the segment is degenerate.
    pub fn end_indices(&self) -> Option<[usize; 2]> {
        use entity::Curve;
        match self {
            Segment3D::Line(l) => l.end_indices(),
            Segment3D::Arc(a) => a.end_indices(),
            Segment3D::Circle(c) => c.end_indices(),
            Segment3D::CubicBezier(b) => b.end_indices(),
            Segment3D::QuadraticBezier(b) => b.end_indices(),
            Segment3D::BSpline(s) => s.end_indices(),
        }
    }

    /// Get the start vertex index
    pub fn start_index(&self) -> Option<usize> {
        self.end_indices().map(|[s, _]| s)
    }

    /// Get the end vertex index
    pub fn finish_index(&self) -> Option<usize> {
        self.end_indices().map(|[_, f]| f)
    }

    /// Get the start point using vertices array
    pub fn start(&self, vertices: &[Point3<f64>]) -> Option<Point3<f64>> {
        self.start_index().and_then(|i| vertices.get(i).copied())
    }

    /// Get the end point using vertices array
    pub fn finish(&self, vertices: &[Point3<f64>]) -> Option<Point3<f64>> {
        self.finish_index().and_then(|i| vertices.get(i).copied())
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

    /// Discretize this segment into a sequence of points
    ///
    /// # Arguments
    /// * `vertices` - The vertex array to look up indices
    /// * `tolerance` - Max chord height error
    pub fn discretize(&self, vertices: &[Point3<f64>], tolerance: f64) -> Vec<Point3<f64>> {
        discretize::discretize_segment_3d(self, vertices, tolerance)
    }
}

// =============================================================================
// Path structs
// =============================================================================

/// A 2D path consisting of vertices and curve segments
///
/// Vertices are stored in a shared array, and segments reference them by index.
///
/// # Caching
///
/// Derived properties like `bounds()` and `extents()` are lazily computed
/// and cached on first access. The cache is thread-safe and uses `OnceLock`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Path2D {
    /// Shared vertex array
    pub vertices: Vec<Point2<f64>>,

    /// The curve segments that make up this path (using vertex indices)
    pub segments: Vec<Segment2D>,

    /// User override for deviation. If None, computed from extents * DEVIATION_RATIO.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deviation: Option<f64>,

    /// Optional transform from 2D path space back to 3D
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_3d: Option<Matrix4<f64>>,

    // Cached - computed lazily on first access (skip in serde)
    #[serde(skip)]
    cache_bounds: Cache<Option<Bounds2>>,
    #[serde(skip)]
    cache_polygons: Cache<Vec<Polygon2D>>,
    #[serde(skip)]
    #[allow(clippy::type_complexity)]
    cache_triangulation: Cache<(Vec<Point2<f64>>, Vec<[usize; 3]>)>,
}

impl Clone for Path2D {
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            segments: self.segments.clone(),
            deviation: self.deviation,
            to_3d: self.to_3d,
            // Fresh caches - will recompute on demand
            cache_bounds: Cache::new(),
            cache_polygons: Cache::new(),
            cache_triangulation: Cache::new(),
        }
    }
}

impl PartialEq for Path2D {
    fn eq(&self, other: &Self) -> bool {
        self.vertices == other.vertices
            && self.segments == other.segments
            && self.deviation == other.deviation
            && self.to_3d == other.to_3d
    }
}

impl Path2D {
    /// Create a new empty path
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a path from vertices and segments
    pub fn from_vertices_and_segments(
        vertices: Vec<Point2<f64>>,
        segments: Vec<Segment2D>,
    ) -> Self {
        Self {
            vertices,
            segments,
            deviation: None,
            to_3d: None,
            cache_bounds: Cache::new(),
            cache_polygons: Cache::new(),
            cache_triangulation: Cache::new(),
        }
    }

    /// Create a path from point arrays (one Line entity per input array)
    ///
    /// Each input `Vec<Point2<f64>>` becomes a single `Line` entity.
    /// Points are deduplicated into the shared vertex array.
    ///
    /// # Example
    /// ```
    /// use rmesh::path::Path2D;
    /// use nalgebra::Point2;
    ///
    /// let path = Path2D::from_segments(vec![
    ///     vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0), Point2::new(10.0, 10.0)],
    ///     vec![Point2::new(10.0, 10.0), Point2::new(0.0, 10.0), Point2::new(0.0, 0.0)],
    /// ]);
    /// assert_eq!(path.segments.len(), 2);
    /// ```
    pub fn from_segments(point_arrays: Vec<Vec<Point2<f64>>>) -> Self {
        let mut path = Self::new();
        for points in point_arrays {
            if points.len() < 2 {
                continue;
            }
            let indices: Vec<usize> = points
                .into_iter()
                .map(|p| path.add_or_get_vertex(p, 1e-10))
                .collect();
            path.segments
                .push(Segment2D::Line(Line::from_points(indices)));
        }
        path
    }

    /// Add a vertex and return its index
    pub fn add_vertex(&mut self, point: Point2<f64>) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(point);
        idx
    }

    /// Add or get vertex index (deduplicates within tolerance)
    pub fn add_or_get_vertex(&mut self, point: Point2<f64>, tolerance: f64) -> usize {
        // Check if vertex already exists
        for (i, v) in self.vertices.iter().enumerate() {
            let dx = v.x - point.x;
            let dy = v.y - point.y;
            if dx * dx + dy * dy < tolerance * tolerance {
                return i;
            }
        }
        self.add_vertex(point)
    }

    /// Create a rectangular path centered at origin
    pub fn rectangle(width: f64, height: f64) -> Self {
        let w = width / 2.0;
        let h = height / 2.0;

        let vertices = vec![
            Point2::new(-w, -h),
            Point2::new(w, -h),
            Point2::new(w, h),
            Point2::new(-w, h),
        ];

        let segments = vec![
            Segment2D::Line(Line::new(0, 1)),
            Segment2D::Line(Line::new(1, 2)),
            Segment2D::Line(Line::new(2, 3)),
            Segment2D::Line(Line::new(3, 0)),
        ];

        Self::from_vertices_and_segments(vertices, segments)
    }

    /// Create a rectangular path with corner at origin
    pub fn rectangle_corner(width: f64, height: f64) -> Self {
        let vertices = vec![
            Point2::new(0.0, 0.0),
            Point2::new(width, 0.0),
            Point2::new(width, height),
            Point2::new(0.0, height),
        ];

        let segments = vec![
            Segment2D::Line(Line::new(0, 1)),
            Segment2D::Line(Line::new(1, 2)),
            Segment2D::Line(Line::new(2, 3)),
            Segment2D::Line(Line::new(3, 0)),
        ];

        Self::from_vertices_and_segments(vertices, segments)
    }

    /// Create a circular path centered at origin
    pub fn circle(radius: f64) -> Self {
        let vertices = vec![Point2::origin()];
        let segments = vec![Segment2D::Circle(Circle2::new(0, radius))];
        Self::from_vertices_and_segments(vertices, segments)
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

    // === Point access helpers ===

    /// Get start point of segment at index
    pub fn segment_start(&self, idx: usize) -> Option<Point2<f64>> {
        self.segments.get(idx)?.start(&self.vertices)
    }

    /// Get finish point of segment at index
    pub fn segment_finish(&self, idx: usize) -> Option<Point2<f64>> {
        self.segments.get(idx)?.finish(&self.vertices)
    }

    // === Bounding box (cached) ===

    /// Get the axis-aligned bounding box of this path
    ///
    /// Returns `None` if the path is empty. Cached on first access.
    pub fn bounds(&self) -> Option<Bounds2> {
        *self.cache_bounds.get_or_init(|| self.compute_bounds())
    }

    /// Compute bounds (internal, called by cache)
    fn compute_bounds(&self) -> Option<Bounds2> {
        let mut b = Bounds2::from_points(&self.vertices)?;

        // Expand for circles/ellipses
        for segment in &self.segments {
            match segment {
                Segment2D::Circle(circle) => {
                    let center = self.vertices[circle.center];
                    let r = circle.radius;
                    b.include_point(&Point2::new(center.x - r, center.y - r));
                    b.include_point(&Point2::new(center.x + r, center.y + r));
                }
                Segment2D::Ellipse(ellipse) => {
                    let center = self.vertices[ellipse.center];
                    let r = ellipse.major.max(ellipse.minor);
                    b.include_point(&Point2::new(center.x - r, center.y - r));
                    b.include_point(&Point2::new(center.x + r, center.y + r));
                }
                Segment2D::Arc(arc) => {
                    if let (Some(center), Some(r)) =
                        (arc.center(&self.vertices), arc.radius(&self.vertices))
                    {
                        b.include_point(&Point2::new(center.x - r, center.y - r));
                        b.include_point(&Point2::new(center.x + r, center.y + r));
                    }
                }
                _ => {}
            }
        }

        Some(b)
    }

    /// Get the extents (dimensions) of this path as [width, height]
    ///
    /// Returns None for empty paths.
    pub fn extents(&self) -> Option<[f64; 2]> {
        self.bounds().map(|b| {
            let e = b.extents();
            [e.x, e.y]
        })
    }

    // === Discretization ===

    /// Discretize all segments to per-segment point sequences.
    ///
    /// Uses `deviation` field if set, otherwise computes from extents * DEVIATION_RATIO.
    ///
    /// # Returns
    /// A vector of point sequences, one per segment. Useful for rendering
    /// where each segment needs to be drawn independently.
    pub fn to_segments(&self) -> Vec<Vec<Point2<f64>>> {
        let tol = resolve_deviation!(self);
        self.segments
            .iter()
            .filter_map(|s| {
                let mut points = s.discretize(&self.vertices, tol);
                // Add the finish point for each segment
                points.push(s.finish(&self.vertices)?);
                Some(points)
            })
            .collect()
    }

    /// Discretize this path into a single polyline.
    ///
    /// Concatenates all discretized segments into one contiguous point sequence.
    /// For per-segment polylines (e.g. rendering), use `to_segments()` instead.
    pub fn discretize(&self) -> Vec<Point2<f64>> {
        let tol = resolve_deviation!(self);
        let mut points = Vec::new();
        for segment in &self.segments {
            points.extend(segment.discretize(&self.vertices, tol));
        }
        // Add the final endpoint (but not for closed shapes like circles)
        if let Some(last) = self.segments.last()
            && !last.is_closed()
            && let Some(finish) = last.finish(&self.vertices)
        {
            points.push(finish);
        }
        points
    }

    // === Connectivity ===

    /// Build graph of segment connectivity from endpoints
    pub fn entity_graph(&self) -> EntityGraph {
        EntityGraph::from_path_2d(self)
    }

    /// Group segments into connected components
    pub fn connected_entities(&self) -> Vec<Vec<usize>> {
        self.entity_graph().connected_components()
    }

    // === Rings ===

    /// Find closed rings (cycles in the entity graph)
    ///
    /// Returns vectors of segment indices that form closed loops.
    pub fn rings_entity(&self) -> Vec<Vec<usize>> {
        self.entity_graph().find_rings(&self.segments)
    }

    /// Discretize closed rings to point sequences
    ///
    /// Returns only the closed rings as discretized point sequences.
    pub fn rings_discrete(&self) -> Vec<Vec<Point2<f64>>> {
        let tol = resolve_deviation!(self);
        let rings = self.rings_entity();
        rings
            .into_iter()
            .filter_map(|ring_indices| {
                let mut points = Vec::new();
                for &idx in &ring_indices {
                    let segment = self.segments.get(idx)?;
                    points.extend(segment.discretize(&self.vertices, tol));
                }
                // Close the ring only if not already closed
                if let Some(&first_idx) = ring_indices.first() {
                    let first_point = self.segments.get(first_idx)?.start(&self.vertices)?;
                    // Check if already closed
                    let already_closed = points.last().is_some_and(|last| {
                        (first_point.x - last.x).abs() < 1e-10
                            && (first_point.y - last.y).abs() < 1e-10
                    });
                    if !already_closed {
                        points.push(first_point);
                    }
                }
                Some(points)
            })
            .collect()
    }

    // === Polygons (uses i_overlay) ===

    /// Get polygons with proper hole assignment using i_overlay
    ///
    /// This method:
    /// 1. Finds all closed rings in the path
    /// 2. Uses i_overlay to determine enclosure relationships
    /// 3. Returns properly constructed polygons with holes
    ///
    /// Results are cached on first access.
    pub fn polygons(&self) -> &[Polygon2D] {
        self.cache_polygons
            .get_or_init(|| polygons::polygons_from_path(self, resolve_deviation!(self)))
    }

    /// Triangulate the path's polygons into a flat triangle mesh.
    ///
    /// Returns `(vertices, triangles)` where triangles index into vertices.
    /// Handles non-convex polygons and holes via earcut. Results are cached.
    pub fn triangulate(&self) -> &(Vec<Point2<f64>>, Vec<[usize; 3]>) {
        self.cache_triangulation
            .get_or_init(|| self.compute_triangulation())
    }

    fn compute_triangulation(&self) -> (Vec<Point2<f64>>, Vec<[usize; 3]>) {
        let polygons = self.polygons();
        let mut all_vertices: Vec<Point2<f64>> = Vec::new();
        let mut all_triangles: Vec<[usize; 3]> = Vec::new();
        let mut triangulator = Triangulator::new();

        for poly in polygons {
            let base = all_vertices.len();

            // Add exterior vertices, build index list
            let ext_indices: Vec<usize> = (0..poly.exterior.len())
                .map(|i| {
                    all_vertices.push(poly.exterior[i]);
                    base + i
                })
                .collect();

            // Add hole vertices, build hole index lists
            let hole_indices: Vec<Vec<usize>> = poly
                .interiors
                .iter()
                .map(|hole| {
                    hole.iter()
                        .map(|pt| {
                            let idx = all_vertices.len();
                            all_vertices.push(*pt);
                            idx
                        })
                        .collect()
                })
                .collect();

            let tris = triangulator
                .triangulate_2d(&ext_indices, &hole_indices, &all_vertices, false)
                .unwrap_or_default();
            all_triangles.extend(tris);
        }

        (all_vertices, all_triangles)
    }

    /// Calculate the total length of this path
    pub fn length(&self) -> f64 {
        let points = self.discretize();
        points
            .windows(2)
            .map(|w| ((w[1].x - w[0].x).powi(2) + (w[1].y - w[0].y).powi(2)).sqrt())
            .sum()
    }

    /// Lift this 2D path back to 3D using the stored `to_3d` transform.
    ///
    /// Transforms all vertices through the 4x4 matrix and converts
    /// segments that have direct 3D equivalents (Line, CubicBezier,
    /// QuadraticBezier, BSpline). Arc, Circle, and Ellipse segments
    /// are skipped since their 3D representations differ structurally.
    ///
    /// Returns `None` if no `to_3d` transform is set.
    pub fn to_path3d(&self) -> Option<Path3D> {
        let to_3d = self.to_3d?;
        let vertices: Vec<Point3<f64>> = self
            .vertices
            .iter()
            .map(|p| to_3d.transform_point(&Point3::new(p.x, p.y, 0.0)))
            .collect();

        let segments: Vec<Segment3D> = self
            .segments
            .iter()
            .filter_map(|s| match s {
                Segment2D::Line(l) => Some(Segment3D::Line(l.clone())),
                Segment2D::CubicBezier(b) => Some(Segment3D::CubicBezier(*b)),
                Segment2D::QuadraticBezier(b) => Some(Segment3D::QuadraticBezier(*b)),
                Segment2D::BSpline(b) => Some(Segment3D::BSpline(b.clone())),
                _ => None,
            })
            .collect();

        Some(Path3D::from_vertices_and_segments(vertices, segments))
    }

    /// Offset the path boundary by `distance`, preserving analytical curves.
    ///
    /// Circles and arcs from the original path are recognized in the
    /// buffered result and emitted with adjusted radii rather than being
    /// discretized into line segments.
    ///
    /// Positive distance = expand outward, negative = shrink inward.
    /// Returns empty vec if the path shrinks to nothing.
    pub fn buffer(&self, distance: f64) -> Vec<Path2D> {
        use crate::project::polygon_to_path;

        // 1. Collect original analytical circles from our segments
        let mut known_circles: Vec<(Point2<f64>, f64)> = Vec::new();
        for seg in &self.segments {
            match seg {
                Segment2D::Circle(c) => {
                    let center = self.vertices[c.center];
                    known_circles.push((center, c.radius));
                }
                Segment2D::Arc(a) => {
                    if let (Some(center), Some(r)) =
                        (a.center(&self.vertices), a.radius(&self.vertices))
                    {
                        // Only add unique circles
                        let already = known_circles.iter().any(|(kc, kr)| {
                            let tol = (r * EPSILON_RELATIVE).max(f64::EPSILON * 100.0);
                            (kc.x - center.x).powi(2) + (kc.y - center.y).powi(2) < tol * tol
                                && (kr - r).abs() < tol
                        });
                        if !already {
                            known_circles.push((center, r));
                        }
                    }
                }
                _ => {}
            }
        }

        // 2. Build adjusted expected circles for the buffered geometry.
        //    Exterior arcs grow by +distance, hole arcs shrink by -distance;
        //    provide both candidates and let ring_to_segments match whichever fits.
        let adjusted: Vec<(Point2<f64>, f64)> = known_circles
            .iter()
            .flat_map(|&(center, radius)| {
                [radius + distance, radius - distance]
                    .into_iter()
                    .filter(|&r| r > 0.0)
                    .map(move |r| (center, r))
            })
            .collect();

        // 3. Discretize to polygons and buffer them
        let polygons = self.polygons();
        let mut results: Vec<Path2D> = Vec::new();

        for poly in polygons {
            let buffered = poly.buffer(distance);
            for bp in &buffered {
                let mut path = if adjusted.is_empty() {
                    bp.to_path2d()
                } else {
                    // Snap buffered polygon vertices onto expected circles
                    // so that ring_to_segments can detect them within
                    // EPSILON_RELATIVE tolerance.
                    let mut snapped = bp.clone();
                    snap_ring_to_circles(&mut snapped.exterior, &adjusted);
                    for interior in &mut snapped.interiors {
                        snap_ring_to_circles(interior, &adjusted);
                    }
                    polygon_to_path(&snapped, &adjusted, None)
                };
                path.to_3d = self.to_3d;
                results.push(path);
            }
        }

        results
    }

    /// Clip this path against a boolean image mask, returning visible fragments as polylines.
    ///
    /// Discretizes the path, resamples at pixel-level density, batch-checks
    /// against the mask, inflates transitions to close small gaps, links
    /// nearby blocks, and returns contiguous visible fragments in metric coordinates.
    ///
    /// # Arguments
    /// * `mask` - Boolean image mask (nonzero = inside)
    /// * `to_raster` - Affine transform from metric → pixel coordinates
    /// * `inflate_radius` - 1D dilation radius in samples for gap closing along curve
    /// * `link_distance` - Metric distance threshold for merging nearby visible blocks
    pub fn raster_clip(
        &self,
        mask: &raster::BooleanImage,
        to_raster: &nalgebra::Matrix3<f64>,
        inflate_radius: usize,
        link_distance: f64,
    ) -> Vec<Vec<Point2<f64>>> {
        let polyline = self.discretize();
        raster::raster_clip(&polyline, mask, to_raster, inflate_radius, link_distance)
    }

    /// Get the area of the first ring
    ///
    /// Uses the shoelace formula. Returns 0 if no rings are found.
    pub fn area(&self) -> f64 {
        let rings = self.rings_discrete();
        if rings.is_empty() {
            return 0.0;
        }
        polygons::signed_area(&rings[0]).abs()
    }
}

/// A 3D path consisting of vertices and curve segments
///
/// # Caching
///
/// Derived properties like `bounds()` and `extents()` are lazily computed
/// and cached on first access. The cache is thread-safe and uses `OnceLock`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Path3D {
    /// Shared vertex array
    pub vertices: Vec<Point3<f64>>,

    /// The curve segments that make up this path (using vertex indices)
    pub segments: Vec<Segment3D>,

    /// User override for deviation. If None, computed from extents * DEVIATION_RATIO.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deviation: Option<f64>,

    // Cached - computed lazily on first access (skip in serde)
    #[serde(skip)]
    cache_bounds: Cache<Option<Bounds3>>,
}

impl Clone for Path3D {
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            segments: self.segments.clone(),
            deviation: self.deviation,
            // Fresh caches - will recompute on demand
            cache_bounds: Cache::new(),
        }
    }
}

impl PartialEq for Path3D {
    fn eq(&self, other: &Self) -> bool {
        self.vertices == other.vertices
            && self.segments == other.segments
            && self.deviation == other.deviation
    }
}

impl Path3D {
    /// Create a new empty path
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a path from vertices and segments
    pub fn from_vertices_and_segments(
        vertices: Vec<Point3<f64>>,
        segments: Vec<Segment3D>,
    ) -> Self {
        Self {
            vertices,
            segments,
            deviation: None,
            cache_bounds: Cache::new(),
        }
    }

    /// Add a vertex and return its index
    pub fn add_vertex(&mut self, point: Point3<f64>) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(point);
        idx
    }

    /// Add or get vertex index (deduplicates within tolerance)
    pub fn add_or_get_vertex(&mut self, point: Point3<f64>, tolerance: f64) -> usize {
        for (i, v) in self.vertices.iter().enumerate() {
            if (*v - point).norm() < tolerance {
                return i;
            }
        }
        self.add_vertex(point)
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

    // === Point access helpers ===

    /// Get start point of segment at index
    pub fn segment_start(&self, idx: usize) -> Option<Point3<f64>> {
        self.segments.get(idx)?.start(&self.vertices)
    }

    /// Get finish point of segment at index
    pub fn segment_finish(&self, idx: usize) -> Option<Point3<f64>> {
        self.segments.get(idx)?.finish(&self.vertices)
    }

    // === Bounding box (cached) ===

    /// Get the axis-aligned bounding box of this path
    ///
    /// Returns `None` if the path is empty. Cached on first access.
    pub fn bounds(&self) -> Option<Bounds3> {
        *self.cache_bounds.get_or_init(|| self.compute_bounds())
    }

    /// Compute bounds (internal, called by cache)
    fn compute_bounds(&self) -> Option<Bounds3> {
        let mut b = Bounds3::from_points(&self.vertices)?;

        // Expand for circles
        for segment in &self.segments {
            if let Segment3D::Circle(circle) = segment {
                let center = self.vertices[circle.center];
                let r = circle.radius;
                b.include_point(&Point3::new(center.x - r, center.y - r, center.z - r));
                b.include_point(&Point3::new(center.x + r, center.y + r, center.z + r));
            }
        }

        Some(b)
    }

    /// Get the extents (dimensions) of this path as [width, height, depth]
    ///
    /// Returns None for empty paths.
    pub fn extents(&self) -> Option<[f64; 3]> {
        self.bounds().map(|b| {
            let e = b.extents();
            [e.x, e.y, e.z]
        })
    }

    // === Discretization ===

    /// Discretize all segments to per-segment point sequences.
    ///
    /// Uses `deviation` field if set, otherwise computes from extents * DEVIATION_RATIO.
    pub fn to_segments(&self) -> Vec<Vec<Point3<f64>>> {
        let tol = resolve_deviation!(self);
        self.segments
            .iter()
            .filter_map(|s| {
                let mut points = s.discretize(&self.vertices, tol);
                points.push(s.finish(&self.vertices)?);
                Some(points)
            })
            .collect()
    }

    /// Discretize this path into a single polyline.
    pub fn discretize(&self) -> Vec<Point3<f64>> {
        let tol = resolve_deviation!(self);
        let mut points = Vec::new();
        for segment in &self.segments {
            points.extend(segment.discretize(&self.vertices, tol));
        }
        // Add the final endpoint (but not for closed shapes like circles)
        if let Some(last) = self.segments.last()
            && !last.is_closed()
            && let Some(finish) = last.finish(&self.vertices)
        {
            points.push(finish);
        }
        points
    }

    // === Connectivity ===

    /// Build graph of segment connectivity from endpoints
    pub fn entity_graph(&self) -> EntityGraph {
        EntityGraph::from_path_3d(self)
    }

    /// Group segments into connected components
    pub fn connected_entities(&self) -> Vec<Vec<usize>> {
        self.entity_graph().connected_components()
    }

    // === Rings ===

    /// Find closed rings (cycles in the entity graph)
    pub fn rings_entity(&self) -> Vec<Vec<usize>> {
        self.entity_graph().find_rings_3d(&self.segments)
    }

    /// Discretize closed rings to point sequences
    pub fn rings_discrete(&self) -> Vec<Vec<Point3<f64>>> {
        let tol = resolve_deviation!(self);
        let rings = self.rings_entity();
        rings
            .into_iter()
            .filter_map(|ring_indices| {
                let mut points = Vec::new();
                for &idx in &ring_indices {
                    let segment = self.segments.get(idx)?;
                    points.extend(segment.discretize(&self.vertices, tol));
                }
                // Close the ring only if not already closed
                if let Some(&first_idx) = ring_indices.first() {
                    let first_point = self.segments.get(first_idx)?.start(&self.vertices)?;
                    let already_closed = points
                        .last()
                        .is_some_and(|last| (first_point - last).norm() < 1e-10);
                    if !already_closed {
                        points.push(first_point);
                    }
                }
                Some(points)
            })
            .collect()
    }

    /// Calculate the total length of this path
    pub fn length(&self) -> f64 {
        let points = self.discretize();
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
        assert_eq!(path.vertices.len(), 4);

        // Check perimeter (approximation via discretization)
        assert_relative_eq!(path.length(), 30.0, epsilon = 0.1);
    }

    #[test]
    fn test_circle() {
        let path = Path2D::circle(10.0);
        assert_eq!(path.segments.len(), 1);
        assert_eq!(path.vertices.len(), 1); // Just center

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
    fn test_to_segments() {
        let path = Path2D::rectangle(10.0, 5.0);
        let segments = path.to_segments();

        // Should have 4 segments
        assert_eq!(segments.len(), 4);

        // Each line segment should have 2 points (start + finish)
        for seg in &segments {
            assert_eq!(seg.len(), 2);
        }
    }

    #[test]
    fn test_discretize() {
        let path = Path2D::rectangle(10.0, 5.0);
        let points = path.discretize();

        // Should have 4 corners + final endpoint = 5 points
        assert_eq!(points.len(), 5);
    }

    #[test]
    fn test_from_svg() {
        let path = Path2D::from_svg("M0,0 L10,0 L10,5 L0,5 Z").unwrap();
        assert_eq!(path.segments.len(), 4);
    }

    #[test]
    fn test_to_svg() {
        let path = Path2D::rectangle(10.0, 5.0);
        let svg = path.to_svg();

        // Should contain move and line commands
        assert!(svg.contains('M'));
        assert!(svg.contains('L'));
    }

    #[test]
    fn test_segment_2d_start_finish() {
        let path = Path2D::from_vertices_and_segments(
            vec![Point2::new(0.0, 0.0), Point2::new(10.0, 5.0)],
            vec![Segment2D::Line(Line::new(0, 1))],
        );
        assert_eq!(path.segment_start(0), Some(Point2::new(0.0, 0.0)));
        assert_eq!(path.segment_finish(0), Some(Point2::new(10.0, 5.0)));
    }

    #[test]
    fn test_path_3d() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
        ];
        let segments = vec![
            Segment3D::Line(Line::new(0, 1)),
            Segment3D::Line(Line::new(1, 2)),
        ];
        let path = Path3D::from_vertices_and_segments(vertices, segments);

        assert_eq!(path.len(), 2);
        assert_relative_eq!(path.length(), 20.0, epsilon = 0.1);
    }

    #[test]
    fn test_connected_entities() {
        // Two disconnected line segments
        let path = Path2D::from_vertices_and_segments(
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(10.0, 0.0),
                Point2::new(11.0, 0.0),
            ],
            vec![
                Segment2D::Line(Line::new(0, 1)),
                Segment2D::Line(Line::new(2, 3)),
            ],
        );

        let components = path.connected_entities();
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_rings_entity() {
        let path = Path2D::rectangle(10.0, 5.0);
        let rings = path.rings_entity();

        // Rectangle should form one ring
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0].len(), 4);
    }

    #[test]
    fn test_custom_deviation() {
        let path = Path2D::circle(100.0);

        // Coarse deviation = fewer points
        let mut coarse_path = path.clone();
        coarse_path.deviation = Some(1.0);
        let coarse = coarse_path.to_segments();

        // Fine deviation = more points
        let mut fine_path = path.clone();
        fine_path.deviation = Some(0.001);
        let fine = fine_path.to_segments();

        let coarse_count: usize = coarse.iter().map(|s| s.len()).sum();
        let fine_count: usize = fine.iter().map(|s| s.len()).sum();

        assert!(fine_count > coarse_count);
    }

    #[test]
    fn test_extents() {
        let path = Path2D::rectangle(10.0, 5.0);
        let extents = path.extents().unwrap();
        assert_relative_eq!(extents[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(extents[1], 5.0, epsilon = 1e-10);

        // Empty path should return None
        let empty = Path2D::new();
        assert!(empty.extents().is_none());
    }

    #[test]
    fn test_extents_3d() {
        let path = Path3D::from_vertices_and_segments(
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 5.0, 3.0)],
            vec![Segment3D::Line(Line::new(0, 1))],
        );
        let extents = path.extents().unwrap();
        assert_relative_eq!(extents[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(extents[1], 5.0, epsilon = 1e-10);
        assert_relative_eq!(extents[2], 3.0, epsilon = 1e-10);

        // Empty path should return None
        let empty = Path3D::new();
        assert!(empty.extents().is_none());
    }

    #[test]
    fn test_vertex_deduplication() {
        let mut path = Path2D::new();
        let idx1 = path.add_or_get_vertex(Point2::new(0.0, 0.0), 0.01);
        let idx2 = path.add_or_get_vertex(Point2::new(0.001, 0.001), 0.01); // Within tolerance
        let idx3 = path.add_or_get_vertex(Point2::new(10.0, 0.0), 0.01); // New vertex

        assert_eq!(idx1, idx2); // Should be same index
        assert_ne!(idx1, idx3); // Should be different
        assert_eq!(path.vertices.len(), 2);
    }

    #[test]
    fn test_triangulate() {
        // Rectangle should produce exactly 2 triangles
        let mut rect = Path2D::rectangle(10.0, 5.0);
        rect.deviation = Some(0.001);
        let (vertices, triangles) = rect.triangulate();

        assert_eq!(
            triangles.len(),
            2,
            "Rectangle should triangulate to 2 triangles"
        );
        assert!(
            !vertices.is_empty(),
            "Triangulation should produce vertices"
        );

        // All triangle indices must be in-bounds
        for tri in triangles {
            for &idx in tri {
                assert!(
                    idx < vertices.len(),
                    "Triangle index {} out of bounds (vertices: {})",
                    idx,
                    vertices.len()
                );
            }
        }

        // Circle should triangulate to many triangles with valid indices
        let mut circle = Path2D::circle(5.0);
        circle.deviation = Some(0.001);
        let (c_verts, c_tris) = circle.triangulate();

        assert!(
            c_tris.len() > 4,
            "Circle should triangulate to many triangles, got {}",
            c_tris.len()
        );

        for tri in c_tris {
            for &idx in tri {
                assert!(
                    idx < c_verts.len(),
                    "Circle triangle index {} out of bounds (vertices: {})",
                    idx,
                    c_verts.len()
                );
            }
        }
    }

    // =========================================================================
    // Buffer arc preservation tests
    // =========================================================================

    /// Helper: find all Circle2 segments and return their radii
    fn circle_radii(paths: &[Path2D]) -> Vec<f64> {
        paths
            .iter()
            .flat_map(|p| {
                p.segments.iter().filter_map(|s| {
                    if let Segment2D::Circle(c) = s {
                        Some(c.radius)
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Helper: find all Arc2 segments and return their radii
    fn arc_radii(paths: &[Path2D]) -> Vec<f64> {
        paths
            .iter()
            .flat_map(|p| {
                p.segments.iter().filter_map(|s| {
                    if let Segment2D::Arc(a) = s {
                        a.radius(&p.vertices)
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Helper: find all circular radii (Circle2 or Arc2)
    fn all_circular_radii(paths: &[Path2D]) -> Vec<f64> {
        let mut r = circle_radii(paths);
        r.extend(arc_radii(paths));
        r
    }

    #[test]
    fn test_buffer_circle_expand_geometry() {
        let path = Path2D::circle(10.0);
        let buffered = path.buffer(2.0);
        assert_eq!(buffered.len(), 1, "buffer should produce 1 path");

        // Verify geometry: circumference should be ~2*pi*12
        let expected_length = 2.0 * std::f64::consts::PI * 12.0;
        assert_relative_eq!(buffered[0].length(), expected_length, epsilon = 1.0);

        // Verify area: pi*12^2
        let expected_area = std::f64::consts::PI * 144.0;
        assert_relative_eq!(buffered[0].area(), expected_area, epsilon = 5.0);
    }

    #[test]
    fn test_buffer_circle_shrink_geometry() {
        let path = Path2D::circle(10.0);
        let buffered = path.buffer(-3.0);
        assert_eq!(buffered.len(), 1);

        // Shrunk circle should be smaller than original
        assert!(
            buffered[0].length() < path.length(),
            "shrunk circle should have smaller circumference"
        );
        assert!(
            buffered[0].area() < path.area(),
            "shrunk circle should have smaller area"
        );
    }

    #[test]
    fn test_buffer_circle_to_zero() {
        let path = Path2D::circle(5.0);
        let buffered = path.buffer(-5.0);
        // Should collapse to nothing
        assert!(
            buffered.is_empty() || buffered.iter().all(|p| p.segments.is_empty()),
            "circle buffered by -radius should vanish"
        );
    }

    #[test]
    fn test_buffer_circle_shrink_past_zero() {
        let path = Path2D::circle(5.0);
        let buffered = path.buffer(-6.0);
        assert!(
            buffered.is_empty() || buffered.iter().all(|p| p.segments.is_empty()),
            "circle buffered past zero should vanish"
        );
    }

    #[test]
    fn test_buffer_preserves_to_3d() {
        let mut path = Path2D::circle(10.0);
        path.to_3d = Some(Matrix4::identity());
        let buffered = path.buffer(2.0);
        assert_eq!(buffered.len(), 1);
        assert!(
            buffered[0].to_3d.is_some(),
            "to_3d transform should be preserved through buffer"
        );
    }

    #[test]
    fn test_buffer_rectangle_expand() {
        let path = Path2D::rectangle(10.0, 5.0);
        let buffered = path.buffer(1.0);
        assert!(!buffered.is_empty());

        // Expanded rectangle should be larger
        let original_area = path.area();
        let buffered_area: f64 = buffered.iter().map(|p| p.area()).sum();
        assert!(
            buffered_area > original_area,
            "buffered rectangle should have larger area"
        );
    }

    #[test]
    fn test_buffer_cylinder_project_roundtrip() {
        use crate::creation::create_cylinder;

        let mesh = create_cylinder(10.0, 20.0, 64);

        // Project along Z axis at z=0 (should intersect the barrel)
        let normal = nalgebra::Vector3::new(0.0, 0.0, 1.0);
        let origin = nalgebra::Point3::origin();
        let projections = mesh.project(&normal, &origin, &[0.0]);

        // project returns Vec<Option<Vec<Path2D>>>, one per level
        assert!(!projections.is_empty());
        let paths = projections[0].as_ref().expect("should have paths at z=0");

        // Find the path that contains circles
        let path_with_circle = paths
            .iter()
            .find(|p| p.segments.iter().any(|s| matches!(s, Segment2D::Circle(_))));

        if let Some(path) = path_with_circle {
            let radii = circle_radii(&[path.clone()]);
            assert!(
                !radii.is_empty(),
                "projected cylinder should have Circle2 segments"
            );
            // Original radius should be ~10
            assert_relative_eq!(radii[0], 10.0, epsilon = 0.5);

            // Buffer outward by 3 - check geometry
            let expanded = path.buffer(3.0);
            assert!(!expanded.is_empty());
            let expected_length = 2.0 * std::f64::consts::PI * 13.0;
            assert_relative_eq!(expanded[0].length(), expected_length, epsilon = 2.0);

            // Buffer inward by 2 - check it produces something smaller
            let shrunk = path.buffer(-2.0);
            assert!(!shrunk.is_empty());
            assert!(
                shrunk[0].length() < expanded[0].length(),
                "shrunk should be shorter than expanded"
            );
        }
    }
}
