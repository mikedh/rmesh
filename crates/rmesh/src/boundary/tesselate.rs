//! BREP surface tessellation.
//!
//! Converts BREP faces (analytical surfaces + edge loops) into triangle meshes.
//! The [`ShellTessellator`] produces watertight output through a 5-phase pipeline:
//!
//! 1. **Edge discretization** — Each BREP edge is discretized once and assigned pool
//!    indices. Adjacent faces sharing an edge get the same vertices by construction
//!    (not by post-hoc merging), guaranteeing watertight seams.
//! 2. **Contour refinement** (Phase 1.5) — For faces with holes, outer contour edges
//!    are refined until all inner loop vertices fall inside the outer polygon in UV space.
//! 3. **Face triangulation** — Each face is triangulated via constrained Delaunay
//!    triangulation (CDT) in UV space. Fallback chain: UV CDT → plane projection CDT
//!    (multiple candidate planes scored by projection quality) → fan triangulation.
//! 4. **Adaptive refinement** — Triangles exceeding the chord tolerance are subdivided
//!    by splitting the worst edge. Boundary edges are split globally across all adjacent
//!    faces to maintain watertightness.
//! 5. **Final assembly** — Per-face triangulations are assembled into a single `Trimesh`
//!    with per-vertex normals and face-to-surface grouping attributes.
//!
//! Known limitations: ~85/17,500 faces fall back to plane projection CDT, and
//! 4/153 Rosetta benchmark bodies produce non-watertight output.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::manual_midpoint)]
#![allow(clippy::cloned_instead_of_copied)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::{Point2, Point3, Vector3};

use super::Surface;
use super::cdt;
use super::faces::{CURVATURE_TOL, Cone, Cylinder, GEOMETRY_TOL, Sphere, SurfacePlane, Torus};
use super::topology::{BrepEdge, BrepModel, Curve, EdgeUse, OrientedEdge};
use crate::attributes::{Attributes, Grouping, GroupingKind};
use crate::creation::{Plane, perpendicular};
use crate::mesh::Trimesh;

/// Parameters controlling tesselation quality.
#[derive(Debug, Clone)]
pub struct TesselationParams {
    /// Absolute chord error in meters.
    pub tolerance: f64,
    /// Chord error as fraction of bounding-box diagonal.
    /// Effective tolerance = max(tolerance / length_scale, tolerance_relative * char_length).
    pub tolerance_relative: f64,
    /// Minimum segments per curved edge
    pub min_segments: usize,
    /// Maximum segments per curved edge
    pub max_segments: usize,
    /// Vertex merge tolerance in UV space for snapping nearly-identical vertices.
    /// This prevents CDT failures on near-collinear points.
    pub merge_tolerance: f64,
}

impl Default for TesselationParams {
    fn default() -> Self {
        Self {
            tolerance: 0.0005,     // 0.5mm absolute chord error
            tolerance_relative: 0.001, // 0.1% of bounding-box diagonal
            min_segments: 16,      // ensures circles always look circular
            max_segments: 256,
            merge_tolerance: 1e-8, // Small enough to not affect geometry
        }
    }
}

/// UV aspect ratio above which angular surfaces skip UV-space CDT and fall back
/// to plane projection. Empirically derived: cone/sphere faces spanning nearly the
/// full circle produce aspect ratios of 100–215:1 that always cause CDT failure.
/// The value 50 catches these extreme cases while allowing moderate angular spans
/// (e.g. 3–4 radians) to still attempt UV CDT, which produces better results.
const UV_ASPECT_SKIP_THRESHOLD: f64 = 50.0;

/// Error analysis for a single triangle during adaptive subdivision.
///
/// Multi-point error checking examines chord error at the centroid and all three
/// edge midpoints. This ensures tolerance is maintained everywhere, not just
/// at the centroid.
#[derive(Debug, Clone)]
struct TriangleError {
    /// Chord error at triangle centroid
    centroid_error: f64,
    /// Maximum chord error across all three edge midpoints
    max_edge_midpoint_error: f64,
    /// Index of the edge with maximum error (0, 1, or 2)
    max_error_edge: usize,
    /// Whether this triangle needs subdivision
    needs_subdivision: bool,
}

/// State for a face during the global refinement process.
///
/// Holds all the triangulation state needed to track which vertices belong to this face,
/// how they map to the global vertex pool, and which edges are on the BREP boundary.
#[derive(Debug, Clone)]
struct FaceTriangulation {
    /// UV coordinates of all vertices in this face's local space
    vertices_uv: Vec<Point2<f64>>,
    /// Triangles using local vertex indices
    triangles: Vec<[usize; 3]>,
    /// Maps local vertex index -> pool (global) vertex index
    local_to_pool: Vec<usize>,
    /// Maps pool (global) vertex index -> local vertex index
    pool_to_local: HashMap<usize, usize>,
    /// Set of local edge pairs that lie on BREP boundaries (edges shared between faces)
    /// Stored as (min_local_idx, max_local_idx) for canonical ordering
    boundary_edges: HashSet<(usize, usize)>,
    /// Edge-to-triangle adjacency: canonical edge (min,max) → triangle indices.
    /// Built at Phase 3 start and maintained incrementally during splits.
    edge_tris: HashMap<(usize, usize), Vec<usize>>,
}

impl FaceTriangulation {
    fn new() -> Self {
        Self {
            vertices_uv: Vec::new(),
            triangles: Vec::new(),
            local_to_pool: Vec::new(),
            pool_to_local: HashMap::new(),
            boundary_edges: HashSet::new(),
            edge_tris: HashMap::new(),
        }
    }

    /// Add a vertex to this face's local coordinate system.
    /// Returns the local index.
    fn add_vertex(&mut self, uv: Point2<f64>, pool_idx: usize) -> usize {
        let local_idx = self.vertices_uv.len();
        self.vertices_uv.push(uv);
        self.local_to_pool.push(pool_idx);
        self.pool_to_local.insert(pool_idx, local_idx);
        local_idx
    }

    /// Get local index for a pool index, if this face has that vertex.
    fn get_local(&self, pool_idx: usize) -> Option<usize> {
        self.pool_to_local.get(&pool_idx).copied()
    }

    /// Mark an edge as being on the BREP boundary.
    fn mark_boundary_edge(&mut self, local_a: usize, local_b: usize) {
        let edge = if local_a < local_b {
            (local_a, local_b)
        } else {
            (local_b, local_a)
        };
        self.boundary_edges.insert(edge);
    }

    /// Check if an edge is on the BREP boundary.
    fn is_boundary_edge(&self, local_a: usize, local_b: usize) -> bool {
        let edge = if local_a < local_b {
            (local_a, local_b)
        } else {
            (local_b, local_a)
        };
        self.boundary_edges.contains(&edge)
    }

    /// Build the edge→triangle adjacency map from current triangles.
    fn build_edge_tris(&mut self) {
        self.edge_tris.clear();
        for (tri_idx, tri) in self.triangles.iter().enumerate() {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                let key = (a.min(b), a.max(b));
                self.edge_tris.entry(key).or_default().push(tri_idx);
            }
        }
    }

    /// Split all triangles sharing edge (local_a, local_b) by inserting local_mid.
    /// Uses edge_tris for O(1) lookup instead of scanning all triangles.
    /// Returns true if any triangles were split.
    #[must_use]
    fn split_triangles_at_edge(
        &mut self,
        local_a: usize,
        local_b: usize,
        local_mid: usize,
    ) -> bool {
        let key = (local_a.min(local_b), local_a.max(local_b));
        let tri_indices = match self.edge_tris.remove(&key) {
            Some(v) if !v.is_empty() => v,
            _ => return false,
        };

        for &tri_idx in &tri_indices {
            let tri = self.triangles[tri_idx];
            // Find the opposite vertex
            let v_opp = if (tri[0] == local_a && tri[1] == local_b)
                || (tri[0] == local_b && tri[1] == local_a)
            {
                tri[2]
            } else if (tri[1] == local_a && tri[2] == local_b)
                || (tri[1] == local_b && tri[2] == local_a)
            {
                tri[0]
            } else {
                tri[1]
            };

            // Split: replace original with [local_a, local_mid, v_opp],
            //         append [local_mid, local_b, v_opp]
            let half_a = [local_a, local_mid, v_opp];
            let half_b = [local_mid, local_b, v_opp];

            // Remove old triangle's edges from the map
            for i in 0..3 {
                let ea = tri[i];
                let eb = tri[(i + 1) % 3];
                let ekey = (ea.min(eb), ea.max(eb));
                if ekey != key
                    && let Some(list) = self.edge_tris.get_mut(&ekey)
                {
                    list.retain(|&idx| idx != tri_idx);
                }
            }

            // Replace in-place
            self.triangles[tri_idx] = half_a;
            let new_idx = self.triangles.len();
            self.triangles.push(half_b);

            // Add new edges to the map
            for (idx, half) in [(tri_idx, &half_a), (new_idx, &half_b)] {
                for i in 0..3 {
                    let ea = half[i];
                    let eb = half[(i + 1) % 3];
                    let ekey = (ea.min(eb), ea.max(eb));
                    self.edge_tris.entry(ekey).or_default().push(idx);
                }
            }
        }

        true
    }
}

// ============================================================================
// Surface parametric evaluation
// ============================================================================

impl SurfacePlane {
    /// Get orthonormal basis vectors for the plane.
    fn basis(&self) -> (Vector3<f64>, Vector3<f64>) {
        let n = self.normal.normalize();
        let u = perpendicular(&n).normalize();
        let v = n.cross(&u);
        (u, v)
    }

    /// Map 3D point to (u, v) parameters.
    pub fn to_parametric(&self, point: &Point3<f64>) -> Point2<f64> {
        let (u_axis, v_axis) = self.basis();
        let d = point - self.origin;
        Point2::new(d.dot(&u_axis), d.dot(&v_axis))
    }

    /// Evaluate (u, v) to 3D point.
    pub fn evaluate(&self, u: f64, v: f64) -> Point3<f64> {
        let (u_axis, v_axis) = self.basis();
        self.origin + u * u_axis + v * v_axis
    }

    /// Surface normal (constant for planes).
    pub fn normal_at(&self, _u: f64, _v: f64) -> Vector3<f64> {
        self.normal.normalize()
    }
}

impl Cylinder {
    /// Map 3D point to (theta, h) parameters.
    pub fn to_parametric(&self, point: &Point3<f64>) -> Point2<f64> {
        let (x_axis, y_axis) = self.basis();
        let axis = self.axis_unit();
        let d = point - self.origin;

        let h = d.dot(&axis);
        let radial = d - h * axis;
        let theta = radial.dot(&y_axis).atan2(radial.dot(&x_axis));

        Point2::new(theta, h)
    }

    /// Evaluate (theta, h) to 3D point.
    pub fn evaluate(&self, theta: f64, h: f64) -> Point3<f64> {
        let (x_axis, y_axis) = self.basis();
        let axis = self.axis_unit();
        let radial = theta.cos() * x_axis + theta.sin() * y_axis;
        self.origin + self.radius * radial + h * axis
    }

    /// Surface normal at (theta, h).
    pub fn normal_at(&self, theta: f64, _h: f64) -> Vector3<f64> {
        let (x_axis, y_axis) = self.basis();
        theta.cos() * x_axis + theta.sin() * y_axis
    }
}

impl Cone {
    /// Map 3D point to (theta, d) parameters where d is distance from apex.
    pub fn to_parametric(&self, point: &Point3<f64>) -> Point2<f64> {
        let (x_axis, y_axis) = self.basis();
        let axis = self.axis_unit();
        let from_apex = point - self.apex;

        let d = from_apex.dot(&axis);
        let radial = from_apex - d * axis;
        let theta = radial.dot(&y_axis).atan2(radial.dot(&x_axis));

        Point2::new(theta, d)
    }

    /// Evaluate (theta, d) to 3D point.
    pub fn evaluate(&self, theta: f64, d: f64) -> Point3<f64> {
        let (x_axis, y_axis) = self.basis();
        let axis = self.axis_unit();
        let r = d * self.half_angle.tan();
        let radial = theta.cos() * x_axis + theta.sin() * y_axis;
        self.apex + d * axis + r * radial
    }

    /// Surface normal at (theta, d).
    ///
    /// Note: the normal is independent of `d` (distance from apex). At the apex
    /// itself (d=0), the surface normal is undefined; the returned value is the
    /// limit direction approaching the apex from angle `theta`.
    pub fn normal_at(&self, theta: f64, _d: f64) -> Vector3<f64> {
        let (x_axis, y_axis) = self.basis();
        let axis = self.axis_unit();
        let radial = theta.cos() * x_axis + theta.sin() * y_axis;
        // Normal is perpendicular to surface, pointing outward
        let cos_a = self.half_angle.cos();
        let sin_a = self.half_angle.sin();
        (cos_a * radial - sin_a * axis).normalize()
    }
}

impl Sphere {
    /// Map 3D point to (longitude, latitude) parameters.
    ///
    /// At the poles (lat = ±π/2), longitude is undefined mathematically.
    /// We return lon=0 at poles for consistency. This means UV midpoint
    /// computation across poles is incorrect in UV space — the adaptive
    /// refinement loop (Phase 3) uses 3D midpoints for angular surfaces
    /// to avoid this issue.
    pub fn to_parametric(&self, point: &Point3<f64>) -> Point2<f64> {
        let d = (point - self.center).normalize();
        let lat = d.z.asin();

        // At poles (|lat| ≈ π/2), the x-y projection is near zero, making
        // atan2 numerically unstable. Return lon=0 for consistency.
        let xy_len_sq = d.x * d.x + d.y * d.y;
        let lon = if xy_len_sq < GEOMETRY_TOL {
            0.0 // At pole - longitude is arbitrary, use 0
        } else {
            d.y.atan2(d.x)
        };

        Point2::new(lon, lat)
    }

    /// Evaluate (lon, lat) to 3D point.
    pub fn evaluate(&self, lon: f64, lat: f64) -> Point3<f64> {
        let cos_lat = lat.cos();
        let dir = Vector3::new(cos_lat * lon.cos(), cos_lat * lon.sin(), lat.sin());
        self.center + self.radius * dir
    }

    /// Surface normal at (lon, lat).
    pub fn normal_at(&self, lon: f64, lat: f64) -> Vector3<f64> {
        let cos_lat = lat.cos();
        Vector3::new(cos_lat * lon.cos(), cos_lat * lon.sin(), lat.sin())
    }
}

impl Torus {
    /// Map 3D point to (major_angle, minor_angle) parameters.
    pub fn to_parametric(&self, point: &Point3<f64>) -> Point2<f64> {
        let (x_axis, y_axis) = self.basis();
        let axis = self.axis_unit();
        let d = point - self.center;

        // Project onto the torus plane to find major angle
        let in_plane = d - d.dot(&axis) * axis;
        let major_angle = in_plane.dot(&y_axis).atan2(in_plane.dot(&x_axis));

        // Find the tube center for this major angle
        let tube_center_dir = major_angle.cos() * x_axis + major_angle.sin() * y_axis;
        let tube_center = self.center + self.major_radius * tube_center_dir;

        // Vector from tube center to point
        let to_point = point - tube_center;
        let in_tube_plane = to_point.dot(&tube_center_dir);
        let along_axis = to_point.dot(&axis);
        let minor_angle = along_axis.atan2(in_tube_plane);

        Point2::new(major_angle, minor_angle)
    }

    /// Evaluate (major_angle, minor_angle) to 3D point.
    pub fn evaluate(&self, major: f64, minor: f64) -> Point3<f64> {
        let (x_axis, y_axis) = self.basis();
        let axis = self.axis_unit();

        let tube_center_dir = major.cos() * x_axis + major.sin() * y_axis;
        let tube_center = self.center + self.major_radius * tube_center_dir;

        let outward = minor.cos() * tube_center_dir + minor.sin() * axis;
        tube_center + self.minor_radius * outward
    }

    /// Surface normal at (major, minor).
    pub fn normal_at(&self, major: f64, minor: f64) -> Vector3<f64> {
        let (x_axis, y_axis) = self.basis();
        let axis = self.axis_unit();
        let tube_center_dir = major.cos() * x_axis + major.sin() * y_axis;
        (minor.cos() * tube_center_dir + minor.sin() * axis).normalize()
    }
}

impl Surface {
    /// Map 3D point to parametric (u, v) coordinates.
    pub fn to_parametric(&self, point: &Point3<f64>) -> Point2<f64> {
        match self {
            Surface::Plane(p) => p.to_parametric(point),
            Surface::Cylinder(c) => c.to_parametric(point),
            Surface::Cone(c) => c.to_parametric(point),
            Surface::Sphere(s) => s.to_parametric(point),
            Surface::Torus(t) => t.to_parametric(point),
            Surface::BSpline(b) => b.parameter_at(point),
        }
    }

    /// Evaluate parametric (u, v) to 3D point.
    pub fn evaluate(&self, u: f64, v: f64) -> Point3<f64> {
        match self {
            Surface::Plane(p) => p.evaluate(u, v),
            Surface::Cylinder(c) => c.evaluate(u, v),
            Surface::Cone(c) => c.evaluate(u, v),
            Surface::Sphere(s) => s.evaluate(u, v),
            Surface::Torus(t) => t.evaluate(u, v),
            Surface::BSpline(b) => b.evaluate(u, v),
        }
    }

    /// Surface normal at parametric (u, v).
    pub fn normal_at(&self, u: f64, v: f64) -> Vector3<f64> {
        match self {
            Surface::Plane(p) => p.normal_at(u, v),
            Surface::Cylinder(c) => c.normal_at(u, v),
            Surface::Cone(c) => c.normal_at(u, v),
            Surface::Sphere(s) => s.normal_at(u, v),
            Surface::Torus(t) => t.normal_at(u, v),
            Surface::BSpline(b) => b.normal_at(u, v),
        }
    }

    /// Return a projection hint normal for plane projection fallback.
    ///
    /// For elementary surfaces with a known axis (cylinder, cone, torus),
    /// projecting along the axis gives a clean annular/circular layout.
    /// For spheres, use the normal at the UV centroid of the boundary.
    /// For BSplines, use the normal at the UV centroid.
    /// Returns `None` for planes (which shouldn't need projection fallback).
    fn projection_hint(&self, u_mid: f64, v_mid: f64) -> Option<Vector3<f64>> {
        match self {
            Surface::Plane(_) => None,
            Surface::Cylinder(c) => Some(c.axis_unit()),
            Surface::Cone(c) => Some(c.axis_unit()),
            Surface::Torus(t) => Some(t.axis_unit()),
            Surface::Sphere(_) | Surface::BSpline(_) => {
                let n = self.normal_at(u_mid, v_mid);
                if n.norm() > 1e-12 { Some(n.normalize()) } else { None }
            }
        }
    }
}

// ============================================================================
// 2D geometry utilities
// ============================================================================

/// Check if a 2D point is inside a polygon using the ray-casting algorithm.
/// The polygon is given as a slice of UV points (not closed — last != first).
fn point_in_polygon(point: &Point2<f64>, polygon: &[Point2<f64>]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let px = point.x;
    let py = point.y;
    let mut j = n - 1;
    for i in 0..n {
        let yi = polygon[i].y;
        let yj = polygon[j].y;
        let xi = polygon[i].x;
        let xj = polygon[j].x;
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Find the closest edge segment in a polygon to a query point.
/// Returns (edge_index, squared_distance) where edge_index is the starting vertex index.
fn closest_polygon_edge(point: &Point2<f64>, polygon: &[Point2<f64>]) -> (usize, f64) {
    let n = polygon.len();
    let mut best_idx = 0;
    let mut best_dist_sq = f64::MAX;
    for i in 0..n {
        let j = (i + 1) % n;
        let a = &polygon[i];
        let b = &polygon[j];
        let ab = b - a;
        let ap = point - a;
        let len_sq = ab.norm_squared();
        let t = if len_sq > 1e-30 { ap.dot(&ab) / len_sq } else { 0.0 };
        let t_clamped = t.clamp(0.0, 1.0);
        let closest = a + t_clamped * ab;
        let d_sq = (point - closest).norm_squared();
        if d_sq < best_dist_sq {
            best_dist_sq = d_sq;
            best_idx = i;
        }
    }
    (best_idx, best_dist_sq)
}

// ============================================================================
// UV coordinate utilities
// ============================================================================

/// Unwrap angular coordinates in a polygon to avoid discontinuities at ±π.
///
/// For surfaces with angular parametrization (cylinder, cone, sphere, torus),
/// the u coordinate (theta/longitude) can jump from π to -π or vice versa.
/// This creates self-intersecting polygons in UV space that triangulators can't handle.
///
/// This function adjusts the u coordinate to be continuous by adding/subtracting 2π
/// when consecutive vertices jump across the ±π boundary.
/// Unwrap a single angular coordinate sequence to remove ±π discontinuities.
fn unwrap_angular_sequence(values: impl Iterator<Item = f64>, out: &mut [f64]) {
    use std::f64::consts::{PI, TAU};

    let mut offset = 0.0;
    let mut prev = None;

    for (i, val) in values.enumerate() {
        let curr = val + offset;
        if let Some(p) = prev {
            let diff = curr - p;
            if diff > PI {
                offset -= TAU;
            } else if diff < -PI {
                offset += TAU;
            }
        }
        out[i] = val + offset;
        prev = Some(out[i]);
    }
}

fn unwrap_angular_coords(uvs: &mut [Point2<f64>]) {
    if uvs.len() < 2 {
        return;
    }

    // Unwrap the u coordinate (first angular coordinate)
    let mut u_vals: Vec<f64> = vec![0.0; uvs.len()];
    unwrap_angular_sequence(uvs.iter().map(|p| p.x), &mut u_vals);
    for (i, uv) in uvs.iter_mut().enumerate() {
        uv.x = u_vals[i];
    }
}

/// Unwrap both u and v angular coordinates for surfaces where both are angular
/// (e.g. torus where both major and minor angles can have ±π discontinuities).
fn unwrap_angular_coords_both(uvs: &mut [Point2<f64>]) {
    if uvs.len() < 2 {
        return;
    }

    let mut u_vals: Vec<f64> = vec![0.0; uvs.len()];
    let mut v_vals: Vec<f64> = vec![0.0; uvs.len()];
    unwrap_angular_sequence(uvs.iter().map(|p| p.x), &mut u_vals);
    unwrap_angular_sequence(uvs.iter().map(|p| p.y), &mut v_vals);
    for (i, uv) in uvs.iter_mut().enumerate() {
        uv.x = u_vals[i];
        uv.y = v_vals[i];
    }
}

// ============================================================================
// Vertex merging
// ============================================================================

/// Result of vertex merging operation.
struct MergeResult {
    /// Merged UV coordinates (deduplicated)
    merged_uvs: Vec<Point2<f64>>,
    /// Remapped contour indices referencing merged_uvs
    merged_contours: Vec<Vec<usize>>,
    /// Maps original index -> merged index
    original_to_merged: Vec<usize>,
    /// Maps merged index -> first original index that mapped to it
    merged_to_original: Vec<usize>,
}

/// Merge near-duplicate vertices in UV space to prevent CDT failures.
///
/// When contours have nearly-collinear or nearly-coincident points, CDT can fail
/// to find a valid seed triangle. This function merges vertices that are within
/// the given tolerance, ensuring the triangulator sees a cleaner input.
///
/// Uses O(n log n) KD-tree + union-find algorithm instead of O(n²) grid search.
fn merge_near_vertices(
    uvs: &[Point2<f64>],
    contours: &[Vec<usize>],
    tolerance: f64,
) -> MergeResult {
    if tolerance <= 0.0 || uvs.is_empty() {
        // No merging needed
        let n = uvs.len();
        return MergeResult {
            merged_uvs: uvs.to_vec(),
            merged_contours: contours.to_vec(),
            original_to_merged: (0..n).collect(),
            merged_to_original: (0..n).collect(),
        };
    }

    let n = uvs.len();
    let tolerance_sq = tolerance * tolerance;

    // Build KD-tree once from all input points - O(n log n)
    let entries: Vec<[f64; 2]> = uvs.iter().map(|p| [p.x, p.y]).collect();
    let tree: ImmutableKdTree<f64, 2> = ImmutableKdTree::new_from_slice(&entries);

    // Union-find data structure to track merged clusters
    let mut parent: Vec<usize> = (0..n).collect();

    // Find with path compression
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]]; // path compression
            i = parent[i];
        }
        i
    }

    // For each point, find all neighbors within tolerance and union them - O(n log n) average
    for (i, uv) in uvs.iter().enumerate() {
        // Use within_unsorted for speed (we don't need sorted results)
        for neighbor in tree.within_unsorted::<SquaredEuclidean>(&[uv.x, uv.y], tolerance_sq) {
            let j = neighbor.item as usize;
            if j > i {
                // Only union with later points to avoid double-processing
                let pi = find(&mut parent, i);
                let pj = find(&mut parent, j);
                if pi != pj {
                    // Always merge to lower index (canonical representative)
                    if pi < pj {
                        parent[pj] = pi;
                    } else {
                        parent[pi] = pj;
                    }
                }
            }
        }
    }

    // Build merged vertex list from representatives
    let mut repr_to_merged: HashMap<usize, usize> = HashMap::new();
    let mut merged_uvs = Vec::new();
    let mut merged_to_original = Vec::new();

    let original_to_merged: Vec<usize> = (0..n)
        .map(|i| {
            let repr = find(&mut parent, i);
            *repr_to_merged.entry(repr).or_insert_with(|| {
                let idx = merged_uvs.len();
                merged_uvs.push(uvs[repr]);
                merged_to_original.push(repr);
                idx
            })
        })
        .collect();

    // Remap contour indices, removing consecutive duplicates
    let merged_contours = contours
        .iter()
        .map(|contour| {
            let remapped: Vec<usize> = contour.iter().map(|&i| original_to_merged[i]).collect();
            // Remove consecutive duplicates (important for closed contours)
            let mut deduped = Vec::with_capacity(remapped.len());
            for &idx in &remapped {
                if deduped.last() != Some(&idx) {
                    deduped.push(idx);
                }
            }
            // If contour became too small (< 3 unique points), keep it anyway
            // CDT will handle the error properly
            deduped
        })
        .collect();

    MergeResult {
        merged_uvs,
        merged_contours,
        original_to_merged,
        merged_to_original,
    }
}

// ============================================================================
// Fallbacks for self-intersecting UV boundaries
// ============================================================================

/// Log a CDT failure case to `/tmp/cdt_failures.jsonl` for diagnosis.
///
/// Each line is a JSON object with the face index, surface type, hole count,
/// error description, 2D points, and contour indices. These can be loaded into
/// unit tests for fast CDT iteration without needing STEP files.
#[cfg(test)]
fn log_cdt_failure(
    face_idx: usize,
    surface_type: &str,
    holes: usize,
    error: &str,
    pts: &[(f64, f64)],
    contours: &[Vec<usize>],
) {
    use std::io::Write;
    use std::sync::Mutex;

    static LOG_FILE: Mutex<()> = Mutex::new(());

    let pts_vec: Vec<[f64; 2]> = pts.iter().map(|&(x, y)| [x, y]).collect();
    let contours_vec: Vec<Vec<usize>> = contours.to_vec();
    let case = serde_json::json!({
        "face": format!("face_{}", face_idx),
        "surface_type": surface_type,
        "holes": holes,
        "error": error,
        "pts": pts_vec,
        "contours": contours_vec,
    });
    let line = format!("{}\n", case);
    let _guard = LOG_FILE.lock().unwrap();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/cdt_failures.jsonl")
    {
        let _ = f.write_all(line.as_bytes());
    }
}

/// Check if all edges from all contours (outer + holes) are present in the triangulation.
///
/// This validates that the CDT result properly preserves all boundary constraints,
/// including inner loops (holes). When `original_to_merged` is provided, contour edges
/// where both vertices map to the same merged index are skipped — these edges were
/// collapsed by vertex merging and cannot appear in the triangulation.
fn check_all_contour_edges_present(
    triangles: &[(usize, usize, usize)],
    contours: &[Vec<usize>],
    original_to_merged: Option<&[usize]>,
) -> bool {
    // Build set of all triangle edges
    let mut tri_edges: HashSet<(usize, usize)> = HashSet::new();
    for &(a, b, c) in triangles {
        tri_edges.insert((a.min(b), a.max(b)));
        tri_edges.insert((b.min(c), b.max(c)));
        tri_edges.insert((c.min(a), c.max(a)));
    }

    // Check all contour edges (outer and all holes)
    for contour in contours {
        if contour.is_empty() {
            continue;
        }

        // Closed contours have first == last, so iterate windows
        for window in contour.windows(2) {
            let a = window[0];
            let b = window[1];
            // Skip edges collapsed by vertex merging (both endpoints merged to same vertex)
            if let Some(mapping) = original_to_merged {
                if mapping[a] == mapping[b] {
                    continue;
                }
            }
            let edge = (a.min(b), a.max(b));
            if !tri_edges.contains(&edge) {
                return false;
            }
        }
    }

    true
}

/// Triangulate by projecting 3D points onto a best-fit plane.
///
/// This is a fallback for when UV parametrization creates a self-intersecting
/// boundary (common with certain B-spline surfaces). We project the 3D points
/// onto a plane and triangulate in that 2D space instead.
///
/// `hint_normals` provides surface-specific projection directions
/// (e.g., the axis of a cone/sphere/cylinder, surface normals at boundary vertices)
/// that are likely to produce good results.
fn triangulate_with_plane_projection(
    pts_3d: &[Point3<f64>],
    contours: &[Vec<usize>],
    merge_tolerance: f64,
    hint_normals: &[Vector3<f64>],
) -> Result<Vec<(usize, usize, usize)>, cdt::Error> {
    // Try multiple plane orientations and pick the best result by projection quality.
    // Quality is measured by min_dim / max_dim aspect ratio of the projected bounding box —
    // a more square projection means less distortion and better triangulation.
    let mut planes: Vec<Plane> = Vec::with_capacity(20);

    // Surface-specific hint normals from caller
    for n in hint_normals {
        planes.push(Plane::new(*n, pts_3d[0]));
    }

    // PCA best-fit planes
    if let Ok(p) = Plane::from_points(pts_3d, false) { planes.push(p); }
    if let Ok(p) = Plane::from_points(pts_3d, true) { planes.push(p); }

    // Axis-aligned planes
    planes.push(Plane::new(Vector3::new(0.0, 0.0, 1.0), pts_3d[0]));
    planes.push(Plane::new(Vector3::new(0.0, 1.0, 0.0), pts_3d[0]));
    planes.push(Plane::new(Vector3::new(1.0, 0.0, 0.0), pts_3d[0]));

    // Diagonal planes (45-degree between axis pairs)
    let s = std::f64::consts::FRAC_1_SQRT_2;
    planes.push(Plane::new(Vector3::new(s, s, 0.0), pts_3d[0]));
    planes.push(Plane::new(Vector3::new(s, 0.0, s), pts_3d[0]));
    planes.push(Plane::new(Vector3::new(0.0, s, s), pts_3d[0]));

    // Edge cross-product planes (normals perpendicular to boundary edge pairs)
    if let Some(contour) = contours.first() {
        let cn = contour.len();
        if cn >= 4 {
            for &(i, j) in &[(0, cn / 4), (0, cn / 2), (cn / 4, cn / 2)] {
                let d1 = pts_3d[contour[(i + 1) % cn]] - pts_3d[contour[i]];
                let d2 = pts_3d[contour[(j + 1) % cn]] - pts_3d[contour[j]];
                let cross = d1.cross(&d2);
                if cross.norm() > 1e-10 {
                    planes.push(Plane::new(cross.normalize(), pts_3d[0]));
                }
            }
        }
    }

    // Rotated hint planes (45-degree tilts of each hint)
    for n in hint_normals {
        let perp = perpendicular(n);
        let cos45 = std::f64::consts::FRAC_1_SQRT_2;
        let rotated = *n * cos45 + perp * cos45;
        if rotated.norm() > 1e-10 {
            planes.push(Plane::new(rotated.normalize(), pts_3d[0]));
        }
        let perp2 = n.cross(&perp);
        if perp2.norm() > 1e-10 {
            let rotated2 = *n * cos45 + perp2.normalize() * cos45;
            planes.push(Plane::new(rotated2.normalize(), pts_3d[0]));
        }
    }

    let mut best_tris: Option<Vec<(usize, usize, usize)>> = None;
    let mut best_score: f64 = -1.0;

    for plane_opt in &planes {
        let pts_2d = plane_opt.to_2d(pts_3d);
        let pts_tuples: Vec<(f64, f64)> = pts_2d.iter().map(|p| (p.x, p.y)).collect();

        // Check if projection is degenerate (very thin bounding box)
        let (min_x, max_x) = pts_tuples
            .iter()
            .fold((f64::MAX, f64::MIN), |(mn, mx), &(x, _)| {
                (mn.min(x), mx.max(x))
            });
        let (min_y, max_y) = pts_tuples
            .iter()
            .fold((f64::MAX, f64::MIN), |(mn, mx), &(_, y)| {
                (mn.min(y), mx.max(y))
            });
        let dx = max_x - min_x;
        let dy = max_y - min_y;

        let min_dim = dx.min(dy);
        let max_dim = dx.max(dy);

        if min_dim < 1e-10 || (max_dim / min_dim > 100.0) {
            continue; // Skip degenerate projections
        }

        // Aspect ratio component of the projection score (1.0 = perfectly square)
        let aspect_score = min_dim / max_dim;

        // Compute composite score from aspect ratio + triangle quality.
        // Triangle quality (min/max area ratio) penalizes sliver triangles that
        // cause manifold issues, breaking ties when aspect ratios are similar.
        let composite_score = |mapped_tris: &[(usize, usize, usize)]| -> f64 {
            if mapped_tris.is_empty() {
                return aspect_score * 0.6;
            }
            let mut max_area = 0.0_f64;
            let mut min_area = f64::MAX;
            for &(a, b, c) in mapped_tris {
                let (ax, ay) = pts_tuples[a];
                let (bx, by) = pts_tuples[b];
                let (cx, cy) = pts_tuples[c];
                let area = ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)).abs();
                max_area = max_area.max(area);
                min_area = min_area.min(area);
            }
            let area_ratio = if max_area > 1e-20 { min_area / max_area } else { 0.0 };
            aspect_score * 0.6 + area_ratio * 0.4
        };

        // Apply vertex merging
        let pts_as_point2: Vec<Point2<f64>> =
            pts_tuples.iter().map(|&(x, y)| Point2::new(x, y)).collect();
        let merge_result = merge_near_vertices(&pts_as_point2, contours, merge_tolerance);
        let merged_pts: Vec<(f64, f64)> =
            merge_result.merged_uvs.iter().map(|p| (p.x, p.y)).collect();

        // Try CDT with contours first (catch_unwind guards against CDT assertions
        // on degenerate projections that pass the aspect-ratio filter)
        let contours_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cdt::triangulate_contours(&merged_pts, &merge_result.merged_contours)
        }));
        if let Ok(Ok(tris)) = contours_result {
            let mapped_tris: Vec<(usize, usize, usize)> = tris
                .iter()
                .filter_map(|&(a, b, c)| {
                    let (oa, ob, oc) = (
                        merge_result.merged_to_original[a],
                        merge_result.merged_to_original[b],
                        merge_result.merged_to_original[c],
                    );
                    if oa == ob || ob == oc || oc == oa { None } else { Some((oa, ob, oc)) }
                })
                .collect();

            let score = composite_score(&mapped_tris);
            if check_all_contour_edges_present(&mapped_tris, contours, Some(&merge_result.original_to_merged)) && score > best_score {
                best_score = score;
                best_tris = Some(mapped_tris);
                continue;
            }
        }

        // CDT with contours failed — try triangulate_with_edges which takes
        // explicit edge constraints and guarantees they're preserved
        let boundary_edges: Vec<(usize, usize)> = merge_result
            .merged_contours
            .iter()
            .flat_map(|contour| contour.windows(2).map(|w| (w[0], w[1])))
            .collect();

        let edges_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cdt::triangulate_with_edges(&merged_pts, &boundary_edges)
        }));
        if let Ok(Ok(tris)) = edges_result {
            let mapped_tris: Vec<(usize, usize, usize)> = tris
                .iter()
                .filter_map(|&(a, b, c)| {
                    let (oa, ob, oc) = (
                        merge_result.merged_to_original[a],
                        merge_result.merged_to_original[b],
                        merge_result.merged_to_original[c],
                    );
                    if oa == ob || ob == oc || oc == oa { None } else { Some((oa, ob, oc)) }
                })
                .collect();

            let score = composite_score(&mapped_tris);
            if check_all_contour_edges_present(&mapped_tris, contours, Some(&merge_result.original_to_merged)) && score > best_score {
                best_score = score;
                best_tris = Some(mapped_tris);
            }
        }
    }

    if let Some(tris) = best_tris {
        return Ok(tris);
    }

    // All CDT approaches failed - use fan triangulation as last resort.
    // This guarantees all boundary edges are included but may produce poor quality triangles.
    // NOTE: We only triangulate the outer contour using fan; holes are not handled properly.
    // This is a known limitation when CDT fails.
    #[cfg(test)]
    if contours.len() > 1 {
        eprintln!(
            "WARNING: fan triangulation fallback on face with {} holes — holes will be filled in",
            contours.len() - 1
        );
    }
    if !contours.is_empty() && !contours[0].is_empty() {
        let outer_contour = &contours[0];
        // Closed contour has first == last, so actual vertex count is len - 1
        // If not closed (first != last), use full length
        let is_closed = outer_contour.first() == outer_contour.last();
        let vertex_count = if is_closed && outer_contour.len() > 1 {
            outer_contour.len() - 1
        } else {
            outer_contour.len()
        };
        if vertex_count >= 3 {
            // Create fan triangles using actual contour indices (not synthetic coordinates)
            // Fan triangulation: vertex 0 connects to all other edges
            let mut tris = Vec::with_capacity(vertex_count - 2);
            for i in 1..(vertex_count - 1) {
                tris.push((outer_contour[0], outer_contour[i], outer_contour[i + 1]));
            }
            if !tris.is_empty() {
                return Ok(tris);
            }
        }
    }

    Err(cdt::Error::CannotInitialize)
}

// ============================================================================
// Model tesselation
// ============================================================================

/// Discretization of a single BREP edge with pool indices.
struct EdgeDiscretization {
    /// Pool indices for all points along the edge (including start/end vertices).
    /// In forward direction (from start_vertex to end_vertex).
    pool_indices: Vec<usize>,
}

/// Tessellator that ensures watertight output through edge-centric refinement.
///
/// The key insight: edges are discretized ONCE with pool indices assigned immediately.
/// Adjacent faces sharing an edge get the SAME pool indices by lookup, not by matching.
/// When refinement is needed, edges are refined GLOBALLY so all faces sharing that edge
/// receive the same midpoint.
///
/// Phase 1: Global edge discretization
///   - Add all BREP vertices to pool (by vertex_idx)
///   - Discretize each edge and add interior points to pool (by edge_idx)
///   - Store edge_idx → [pool_indices] mapping
///
/// Phase 2: Initial face triangulation (NO subdivision)
///   - Look up boundary vertices from edge pool (already have pool IDs)
///   - Project to UV and triangulate
///   - Build FaceTriangulation state for each face
///
/// Phase 3: Global refinement loop
///   - Collect edges needing refinement from ANY face
///   - For BOUNDARY edges: refine in BOTH adjacent faces simultaneously
///   - For INTERIOR edges: refine within single face
///   - Single midpoint added to pool, shared by all faces using that edge
///
/// Phase 4: Final assembly
///   - Convert local indices to pool indices
///   - Assign normals
struct ShellTessellator<'a> {
    model: &'a BrepModel,
    params: &'a TesselationParams,
    /// Effective tolerance in model units, computed from absolute + relative tolerances.
    effective_tolerance: f64,

    // Phase 1 results
    /// Global vertex pool for the entire shell
    vertices: Vec<Point3<f64>>,
    /// BREP vertex index → pool index
    brep_vertex_to_pool: HashMap<usize, usize>,
    /// Edge discretizations: edge_idx → pool indices
    edge_discretization: HashMap<usize, EdgeDiscretization>,
    /// Edge adjacency: edge_idx → faces using that edge
    edge_adjacency: HashMap<usize, Vec<EdgeUse>>,
    /// Maps pool edge (min, max) → BREP edge index (for boundary edge lookup)
    pool_edge_to_brep: HashMap<(usize, usize), usize>,

    // Phase 2/3 results
    /// Per-face triangulation state
    face_states: Vec<FaceTriangulation>,

    // Phase 4 results
    /// Triangles output
    triangles: Vec<[usize; 3]>,
    /// Normals at each vertex
    normals: Vec<Vector3<f64>>,
    /// Face index for each triangle
    face_indices: Vec<usize>,
}

impl<'a> ShellTessellator<'a> {
    fn new(model: &'a BrepModel, params: &'a TesselationParams) -> Self {
        // Convert absolute tolerance from meters to model units, then take the
        // max with the relative tolerance scaled by bounding-box diagonal.
        let scale = if model.length_scale > 0.0 {
            model.length_scale
        } else {
            1.0
        };
        let char_length = model.characteristic_length();
        let effective_tolerance = (params.tolerance / scale)
            .max(params.tolerance_relative * char_length);

        Self {
            model,
            params,
            effective_tolerance,
            vertices: Vec::new(),
            brep_vertex_to_pool: HashMap::new(),
            edge_discretization: HashMap::new(),
            edge_adjacency: HashMap::new(),
            pool_edge_to_brep: HashMap::new(),
            face_states: Vec::new(),
            triangles: Vec::new(),
            normals: Vec::new(),
            face_indices: Vec::new(),
        }
    }

    /// Add a vertex to the pool.
    fn add_vertex(&mut self, p: Point3<f64>) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(p);
        idx
    }

    // =========================================================================
    // Phase 1: Global Edge Discretization
    // =========================================================================

    /// Phase 1: Discretize all edges globally BEFORE any face processing.
    /// This ensures edge vertices are shared BY CONSTRUCTION.
    fn phase1_discretize_all_edges(&mut self) {
        // Build edge adjacency map
        self.edge_adjacency = self.model.build_edge_adjacency();

        // Step 1: Add all BREP vertices to the pool first
        for (vertex_idx, vertex) in self.model.vertices.iter().enumerate() {
            let pool_idx = self.add_vertex(vertex.point);
            self.brep_vertex_to_pool.insert(vertex_idx, pool_idx);
        }

        // Step 2: Discretize each edge and add interior points to pool
        for (edge_idx, edge) in self.model.edges.iter().enumerate() {
            let curve = &self.model.curves[edge.curve];
            let n = self.compute_segment_count(curve, edge, edge_idx);

            let mut pool_indices = Vec::with_capacity(n + 1);

            // Start vertex (already in pool from step 1)
            let start_pool_idx = self.brep_vertex_to_pool[&edge.start_vertex];
            pool_indices.push(start_pool_idx);

            // Interior points (add to pool now - these are new vertices)
            for i in 1..n {
                let t = edge.t_start + (edge.t_end - edge.t_start) * (i as f64 / n as f64);
                let p = curve.evaluate(t);
                let pool_idx = self.add_vertex(p);
                pool_indices.push(pool_idx);
            }

            // End vertex (already in pool from step 1)
            let end_pool_idx = self.brep_vertex_to_pool[&edge.end_vertex];
            pool_indices.push(end_pool_idx);

            // Build pool edge to BREP edge mapping for boundary lookup
            for window in pool_indices.windows(2) {
                let key = (window[0].min(window[1]), window[0].max(window[1]));
                self.pool_edge_to_brep.insert(key, edge_idx);
            }

            self.edge_discretization
                .insert(edge_idx, EdgeDiscretization { pool_indices });
        }
    }

    /// Compute the number of segments for discretizing an edge.
    fn compute_segment_count(&self, curve: &Curve, edge: &BrepEdge, edge_idx: usize) -> usize {
        // Segment count from curve geometry
        let curve_count = match curve {
            Curve::Line(_) => 1, // Lines just need start and end
            Curve::Circle(c) => {
                // Guard against degenerate circle (zero or near-zero radius)
                if c.radius <= f64::EPSILON {
                    return self.params.min_segments;
                }
                let angle = (edge.t_end - edge.t_start).abs();
                // Guard against degenerate arc (zero or near-zero angle)
                if angle <= f64::EPSILON {
                    return self.params.min_segments;
                }
                let max_step = (8.0 * self.effective_tolerance / c.radius).sqrt();
                let n = (angle / max_step).ceil() as usize;
                n.clamp(self.params.min_segments, self.params.max_segments)
            }
            Curve::Ellipse(e) => {
                let avg_r = (e.semi_major + e.semi_minor) / 2.0;
                // Guard against degenerate ellipse (zero or near-zero average radius)
                if avg_r <= f64::EPSILON {
                    return self.params.min_segments;
                }
                let angle = (edge.t_end - edge.t_start).abs();
                // Guard against degenerate arc (zero or near-zero angle)
                if angle <= f64::EPSILON {
                    return self.params.min_segments;
                }
                let max_step = (8.0 * self.effective_tolerance / avg_r).sqrt();
                let n = (angle / max_step).ceil() as usize;
                n.clamp(self.params.min_segments, self.params.max_segments)
            }
            Curve::BSpline(bs) => {
                let base_segments =
                    (bs.control_points.len() * (bs.degree + 1)).max(self.params.min_segments);
                base_segments.clamp(self.params.min_segments, self.params.max_segments)
            }
        };

        // Segment count from surface curvature
        let surface_count = self.compute_surface_segment_count(curve, edge, edge_idx);

        curve_count.max(surface_count)
    }

    /// Compute segment count based on surface curvature along the edge.
    fn compute_surface_segment_count(
        &self,
        curve: &Curve,
        edge: &BrepEdge,
        edge_idx: usize,
    ) -> usize {
        let mut max_kappa = 0.0_f64;

        // Use edge_adjacency HashMap for O(1) lookup instead of scanning all faces.
        if let Some(uses) = self.edge_adjacency.get(&edge_idx) {
            for eu in uses {
                let face = &self.model.faces[eu.face_idx];
                let surface = &self.model.face_surfaces[face.surface];
                let t_mid = (edge.t_start + edge.t_end) / 2.0;
                let p_mid = curve.evaluate(t_mid);
                let uv_mid = surface.to_parametric(&p_mid);
                let curvature = surface.curvature_at(uv_mid.x, uv_mid.y);

                max_kappa = max_kappa.max(curvature.kappa_max());
            }
        }

        if max_kappa < CURVATURE_TOL {
            return 1;
        }

        let p_start = curve.evaluate(edge.t_start);
        let p_end = curve.evaluate(edge.t_end);
        let edge_length = (p_end - p_start).norm();
        let l_max = (8.0 * self.effective_tolerance / max_kappa).sqrt();
        let n = (edge_length / l_max).ceil() as usize;
        n.clamp(1, self.params.max_segments)
    }

    /// Get ALL pool indices for an edge, properly ordered for the face.
    /// Returns the COMPLETE sequence including both endpoints.
    /// Deduplication happens at the contour assembly level to ensure faces sharing
    /// an edge with opposite orientations have matching vertex sets.
    fn get_edge_pool_indices(&self, oe: &OrientedEdge) -> Vec<usize> {
        let disc = self
            .edge_discretization
            .get(&oe.edge)
            .expect("edge not pre-discretized");

        if oe.same_sense {
            disc.pool_indices.clone()
        } else {
            disc.pool_indices.iter().rev().cloned().collect()
        }
    }

    // =========================================================================
    // Phase 1.5: Adaptive Outer Contour Refinement
    // =========================================================================

    /// Refine outer contours of faces with holes so inner vertices don't escape
    /// the coarsely-discretized outer polygon.
    ///
    /// Problem: an outer circle of radius R discretized to N segments creates an
    /// inscribed polygon whose edges are at distance R·cos(π/N) from center.
    /// Inner loops discretized at higher resolution can have vertices at radius > R·cos(π/N),
    /// placing them OUTSIDE the outer polygon. This causes CDT "Fixed edges cross" errors.
    ///
    /// Solution: for each face with inner loops, check if all inner UV vertices are
    /// inside the outer UV polygon. If not, refine the nearest outer edge by inserting
    /// a midpoint on the BREP curve. This is a global edge operation, so all faces
    /// sharing that edge see the same new vertices.
    fn phase1_5_refine_outer_contours(&mut self) {
        const MAX_REFINEMENT_ITERATIONS: usize = 5;

        let mut prev_refine_count = usize::MAX;
        for _iteration in 0..MAX_REFINEMENT_ITERATIONS {
            // Collect edges that need refinement across all faces.
            // Key: (pool_a, pool_b) canonical pair, Value: BREP edge index
            let mut edges_to_refine: BTreeSet<(usize, usize)> = BTreeSet::new();

            for (_face_idx, face) in self.model.faces.iter().enumerate() {
                if face.inner_loops.is_empty() {
                    continue;
                }

                let surface = &self.model.face_surfaces[face.surface];
                let outer_loop = &self.model.loops[face.outer_loop];

                // Get outer loop pool indices
                let mut outer_pool_indices = Vec::new();
                for oe in &outer_loop.edges {
                    let edge_indices = self.get_edge_pool_indices(oe);
                    for &idx in &edge_indices {
                        if outer_pool_indices.last() != Some(&idx) {
                            outer_pool_indices.push(idx);
                        }
                    }
                }
                if outer_pool_indices.len() > 1
                    && outer_pool_indices.first() == outer_pool_indices.last()
                {
                    outer_pool_indices.pop();
                }

                if outer_pool_indices.len() < 3 {
                    continue;
                }

                // Compute outer UV coordinates
                let mut outer_uvs: Vec<Point2<f64>> = outer_pool_indices
                    .iter()
                    .map(|&pi| surface.to_parametric(&self.vertices[pi]))
                    .collect();
                if surface.is_doubly_angular() {
                    unwrap_angular_coords_both(&mut outer_uvs);
                } else if surface.is_angular() {
                    unwrap_angular_coords(&mut outer_uvs);
                }

                // Collect all inner loop UV coordinates
                let mut any_outside = false;
                for &inner_loop_idx in &face.inner_loops {
                    let inner_loop = &self.model.loops[inner_loop_idx];
                    let mut inner_pool_indices = Vec::new();
                    for oe in &inner_loop.edges {
                        let edge_indices = self.get_edge_pool_indices(oe);
                        for &idx in &edge_indices {
                            if inner_pool_indices.last() != Some(&idx) {
                                inner_pool_indices.push(idx);
                            }
                        }
                    }
                    if inner_pool_indices.len() > 1
                        && inner_pool_indices.first() == inner_pool_indices.last()
                    {
                        inner_pool_indices.pop();
                    }

                    let mut inner_uvs: Vec<Point2<f64>> = inner_pool_indices
                        .iter()
                        .map(|&pi| surface.to_parametric(&self.vertices[pi]))
                        .collect();
                    if surface.is_doubly_angular() {
                        unwrap_angular_coords_both(&mut inner_uvs);
                    } else if surface.is_angular() {
                        unwrap_angular_coords(&mut inner_uvs);
                    }

                    // Check each inner vertex against the outer polygon
                    for inner_uv in &inner_uvs {
                        if !point_in_polygon(inner_uv, &outer_uvs) {
                            any_outside = true;
                            // Find closest outer edge
                            let (edge_local_idx, _dist_sq) =
                                closest_polygon_edge(inner_uv, &outer_uvs);
                            let next_idx = (edge_local_idx + 1) % outer_pool_indices.len();
                            let pool_a = outer_pool_indices[edge_local_idx];
                            let pool_b = outer_pool_indices[next_idx];
                            let key = (pool_a.min(pool_b), pool_a.max(pool_b));
                            edges_to_refine.insert(key);
                        }
                    }
                }

                if !any_outside {
                    continue;
                }

                // Also log this for debugging
                #[cfg(test)]
                if _iteration == 0 {
                    eprintln!(
                        "    face {_face_idx}: inner vertices outside outer polygon, refining {} edges",
                        edges_to_refine.len()
                    );
                }
            }

            if edges_to_refine.is_empty() {
                break;
            }

            // Stagnation detection: if the number of edges to refine did not
            // decrease since the last iteration, further refinement won't help.
            let refine_count = edges_to_refine.len();
            if refine_count >= prev_refine_count {
                break;
            }
            prev_refine_count = refine_count;

            // Refine each edge by inserting a midpoint on the BREP curve
            for (pool_a, pool_b) in &edges_to_refine {
                let Some(&brep_edge_idx) = self.pool_edge_to_brep.get(&(*pool_a, *pool_b)) else {
                    continue;
                };

                let edge = &self.model.edges[brep_edge_idx];
                let curve = &self.model.curves[edge.curve];
                let disc = self.edge_discretization.get(&brep_edge_idx).unwrap();

                // Find the positions of pool_a, pool_b within the edge's pool_indices
                let pos_a = disc.pool_indices.iter().position(|&p| p == *pool_a);
                let pos_b = disc.pool_indices.iter().position(|&p| p == *pool_b);
                let (Some(pos_a), Some(pos_b)) = (pos_a, pos_b) else {
                    continue;
                };

                // Compute the curve parameter at the midpoint between the two positions
                let n = disc.pool_indices.len() - 1; // number of segments
                if n == 0 {
                    continue;
                }
                let t_a = edge.t_start + (edge.t_end - edge.t_start) * (pos_a as f64 / n as f64);
                let t_b = edge.t_start + (edge.t_end - edge.t_start) * (pos_b as f64 / n as f64);
                let t_mid = (t_a + t_b) / 2.0;

                // Evaluate the curve at the midpoint
                let p_mid = curve.evaluate(t_mid);
                let pool_mid = self.add_vertex(p_mid);

                // Insert into pool_indices between pos_a and pos_b
                let insert_pos = pos_a.max(pos_b); // insert before the later position
                let disc = self.edge_discretization.get_mut(&brep_edge_idx).unwrap();
                disc.pool_indices.insert(insert_pos, pool_mid);

                // Update pool_edge_to_brep: remove old edge, add two new edges
                self.pool_edge_to_brep.remove(&(*pool_a, *pool_b));
                let key_a = ((*pool_a).min(pool_mid), (*pool_a).max(pool_mid));
                let key_b = (pool_mid.min(*pool_b), pool_mid.max(*pool_b));
                self.pool_edge_to_brep.insert(key_a, brep_edge_idx);
                self.pool_edge_to_brep.insert(key_b, brep_edge_idx);
            }

            #[cfg(test)]
            eprintln!(
                "    phase1.5 iteration {}: refined {} outer edges",
                _iteration,
                edges_to_refine.len()
            );
        }
    }

    // =========================================================================
    // Phase 2: Initial Face Triangulation (NO subdivision)
    // =========================================================================

    /// Phase 2: Triangulate a single face without subdivision.
    /// Returns a FaceTriangulation struct with boundary edges marked.
    fn phase2_initial_triangulation(&mut self, face_idx: usize) -> FaceTriangulation {
        let face = &self.model.faces[face_idx];
        let surface = &self.model.face_surfaces[face.surface];
        let outer_loop = &self.model.loops[face.outer_loop];

        let mut state = FaceTriangulation::new();

        // Collect boundary pool indices from pre-discretized edges
        // Deduplicate at contour level to ensure consistent vertex sets across adjacent faces
        let mut outer_pool_indices = Vec::new();
        for oe in &outer_loop.edges {
            let edge_indices = self.get_edge_pool_indices(oe);
            for &idx in &edge_indices {
                // Skip if this is a duplicate of the last vertex (edge chaining)
                if outer_pool_indices.last() != Some(&idx) {
                    outer_pool_indices.push(idx);
                }
            }
        }
        // Remove final vertex if it duplicates the first (closed loop)
        if outer_pool_indices.len() > 1 && outer_pool_indices.first() == outer_pool_indices.last() {
            outer_pool_indices.pop();
        }

        // Add outer boundary vertices to local state
        // First, collect raw UV coordinates
        let mut raw_uvs: Vec<Point2<f64>> = outer_pool_indices
            .iter()
            .map(|&pool_idx| surface.to_parametric(&self.vertices[pool_idx]))
            .collect();

        // Unwrap angular coordinates to avoid discontinuities at ±π.
        // Torus has two angular coordinates (major, minor) that both need unwrapping.
        // Cylinder, cone, sphere only have one angular coordinate (u/theta/longitude).
        if surface.is_doubly_angular() {
            unwrap_angular_coords_both(&mut raw_uvs);
        } else {
            unwrap_angular_coords(&mut raw_uvs);
        }

        for (i, &pool_idx) in outer_pool_indices.iter().enumerate() {
            state.add_vertex(raw_uvs[i], pool_idx);
        }

        // Mark outer boundary edges
        let outer_len = outer_pool_indices.len();
        for i in 0..outer_len {
            let local_a = i;
            let local_b = (i + 1) % outer_len;
            state.mark_boundary_edge(local_a, local_b);
        }

        // Collect inner loop pool indices
        // Apply same deduplication logic as outer loop for consistency
        // Track both hole starts and their lengths for contour building
        let mut hole_info: Vec<(usize, usize)> = Vec::new(); // (start, length)
        for &inner_loop_idx in &face.inner_loops {
            let inner_loop = &self.model.loops[inner_loop_idx];
            let mut loop_indices = Vec::new();
            for oe in &inner_loop.edges {
                let edge_indices = self.get_edge_pool_indices(oe);
                for &idx in &edge_indices {
                    if loop_indices.last() != Some(&idx) {
                        loop_indices.push(idx);
                    }
                }
            }
            // Remove final vertex if it duplicates the first (closed loop)
            if loop_indices.len() > 1 && loop_indices.first() == loop_indices.last() {
                loop_indices.pop();
            }

            let hole_start = state.vertices_uv.len();
            let hole_len = loop_indices.len();
            hole_info.push((hole_start, hole_len));

            // Add inner boundary vertices with angular unwrapping
            let mut inner_uvs: Vec<Point2<f64>> = loop_indices
                .iter()
                .map(|&pool_idx| surface.to_parametric(&self.vertices[pool_idx]))
                .collect();
            if surface.is_doubly_angular() {
                unwrap_angular_coords_both(&mut inner_uvs);
            } else {
                unwrap_angular_coords(&mut inner_uvs);
            }

            for (i, &pool_idx) in loop_indices.iter().enumerate() {
                state.add_vertex(inner_uvs[i], pool_idx);
            }

            // Mark inner boundary edges
            for i in 0..hole_len {
                let local_a = hole_start + i;
                let local_b = hole_start + (i + 1) % hole_len;
                state.mark_boundary_edge(local_a, local_b);
            }
        }

        // Build contours for CDT triangulation (references only boundary points)
        // Each contour must be closed (last index == first index)
        let outer_len = outer_pool_indices.len();
        let boundary_len = state.vertices_uv.len();

        // Build outer contour: [0, 1, 2, ..., n-1, 0] (closed)
        let mut outer_contour: Vec<usize> = (0..outer_len).collect();
        outer_contour.push(0); // Close the contour

        // Build inner contours (holes) using stored hole_info
        let mut contours: Vec<Vec<usize>> = vec![outer_contour];
        for &(hole_start, hole_len) in &hole_info {
            let mut inner_contour: Vec<usize> = (hole_start..hole_start + hole_len).collect();
            inner_contour.push(hole_start); // Close the contour
            contours.push(inner_contour);
        }

        // Compute UV bounds from boundary vertices
        let (u_min, u_max) = state
            .vertices_uv
            .iter()
            .fold((f64::MAX, f64::MIN), |(mn, mx), p| {
                (mn.min(p.x), mx.max(p.x))
            });
        let (v_min, v_max) = state
            .vertices_uv
            .iter()
            .fold((f64::MAX, f64::MIN), |(mn, mx), p| {
                (mn.min(p.y), mx.max(p.y))
            });

        // Generate interior samples for non-planar surfaces
        let interior_uvs =
            surface.generate_interior_samples(u_min, u_max, v_min, v_max, self.effective_tolerance);

        // Add interior points to vertex lists
        // Interior points are unconstrained - CDT will connect them appropriately
        for uv in &interior_uvs {
            let p_3d = surface.evaluate(uv.x, uv.y);
            // Interior points get their own pool index (not shared with other faces)
            // This is fine because they're inside the face, not on boundaries
            let pool_idx = self.add_vertex(p_3d);
            state.add_vertex(*uv, pool_idx);
        }

        // Check if UV CDT is likely to fail for angular surfaces with extreme distortion.
        // Cone/Sphere faces spanning nearly the full circle produce UV aspect ratios
        // of 100-215:1 that always cause CDT failure. Only skip for truly extreme cases
        // (nearly full circle or very thin UV ribbon) — moderate spans (e.g. 3-4 radians)
        // may still succeed and produce better results than plane projection.
        let u_span = u_max - u_min;
        let v_span = v_max - v_min;
        let aspect = if v_span > 1e-12 { u_span / v_span } else { f64::MAX };
        let skip_uv_cdt = surface.is_angular() && aspect > UV_ASPECT_SKIP_THRESHOLD;

        if !skip_uv_cdt {
            // Convert UV points for CDT
            let pts: Vec<(f64, f64)> = state.vertices_uv.iter().map(|p| (p.x, p.y)).collect();

            // Apply vertex merging to prevent CDT failures on near-collinear points
            let pts_as_point2: Vec<Point2<f64>> =
                pts.iter().map(|&(x, y)| Point2::new(x, y)).collect();
            let merge_result =
                merge_near_vertices(&pts_as_point2, &contours, self.params.merge_tolerance);
            let merged_pts_tuples: Vec<(f64, f64)> =
                merge_result.merged_uvs.iter().map(|p| (p.x, p.y)).collect();

            // Use CDT to triangulate with merged vertices and contours
            state.triangles =
                match cdt::triangulate_contours(&merged_pts_tuples, &merge_result.merged_contours) {
                    Ok(tris) => {
                        // Convert CDT output (tuples) to arrays
                        // Map triangles back to original indices using merged_to_original
                        // Use filter_map with bounds checking to avoid panics on invalid indices
                        // Also skip degenerate triangles where merging collapsed two vertices
                        let filtered: Vec<[usize; 3]> = tris.iter()
                            .filter_map(|&(a, b, c)| {
                                let orig_a = *merge_result.merged_to_original.get(a)?;
                                let orig_b = *merge_result.merged_to_original.get(b)?;
                                let orig_c = *merge_result.merged_to_original.get(c)?;
                                // Skip degenerate triangles (two or more vertices collapsed)
                                if orig_a == orig_b || orig_b == orig_c || orig_c == orig_a {
                                    return None;
                                }
                                Some([orig_a, orig_b, orig_c])
                            })
                            .collect();

                        // Validate all contour edges are present in the triangulation.
                        // Merging may have collapsed boundary vertices, creating edges
                        // that don't match their neighbors.
                        let as_tuples: Vec<(usize, usize, usize)> = filtered.iter()
                            .map(|t| (t[0], t[1], t[2])).collect();
                        if check_all_contour_edges_present(&as_tuples, &contours, Some(&merge_result.original_to_merged)) {
                            filtered
                        } else {
                            #[cfg(test)]
                            log_cdt_failure(face_idx, surface.kind_name(), face.inner_loops.len(),
                                "boundary_check_failed", &merged_pts_tuples, &merge_result.merged_contours);
                            // CDT succeeded but boundary not preserved — fall through to fallback
                            Vec::new()
                        }
                    }
                    Err(_cdt_err) => {
                        #[cfg(test)]
                        log_cdt_failure(face_idx, surface.kind_name(), face.inner_loops.len(),
                            &format!("{_cdt_err}"), &merged_pts_tuples, &merge_result.merged_contours);
                        // CDT failed — fall through to recovery below
                        Vec::new()
                    }
                };
        }

        // Recovery: if CDT produced no triangles (either CDT failed, or boundary
        // validation failed), try plane projection then fan triangulation.
        if state.triangles.is_empty() && boundary_len >= 3 {
            // Try plane projection with all points (boundary + interior)
            let pts_3d: Vec<Point3<f64>> = state
                .local_to_pool
                .iter()
                .map(|&pool_idx| self.vertices[pool_idx])
                .collect();

            // Build multiple hint normals for plane projection candidates.
            let u_mid = (u_min + u_max) / 2.0;
            let v_mid = (v_min + v_max) / 2.0;
            let mut hint_normals: Vec<Vector3<f64>> = Vec::new();

            // Primary: surface projection hint (axis for cylinder/cone/torus,
            // surface normal at centroid for sphere/bspline)
            if let Some(n) = surface.projection_hint(u_mid, v_mid) {
                hint_normals.push(n);
            }

            // Secondary: surface normals at evenly-spaced boundary vertices
            let n_samples = state.vertices_uv.len().min(8);
            if n_samples >= 4 {
                let step = n_samples / 4;
                for i in (0..n_samples).step_by(step.max(1)) {
                    let uv = state.vertices_uv[i];
                    let n = surface.normal_at(uv.x, uv.y);
                    if n.norm() > 1e-12 {
                        hint_normals.push(n.normalize());
                    }
                }
            }

            if let Ok(tris) = triangulate_with_plane_projection(
                &pts_3d,
                &contours,
                self.params.merge_tolerance,
                &hint_normals,
            ) {
                state.triangles = tris.iter().map(|&(a, b, c)| [a, b, c]).collect();
            }
        }

        state
    }

    // =========================================================================
    // Phase 3: Global Refinement Loop
    // =========================================================================

    /// Phase 3: Refine edges globally until all triangles meet tolerance.
    ///
    /// The key insight: when an edge needs refinement, ALL faces using that edge
    /// must be updated with the SAME midpoint vertex. This ensures watertightness.
    ///
    /// Uses incremental tracking: only checks faces that were modified in the previous
    /// iteration, reducing complexity from O(F × T × I) to O(M × T_avg × I) where M << F.
    fn phase3_global_refinement(&mut self) {
        const MAX_ITERATIONS: usize = 5;
        const MAX_VERTICES: usize = 200_000;

        // Initially, all non-planar faces need checking
        let mut faces_to_check: BTreeSet<usize> = self
            .face_states
            .iter()
            .enumerate()
            .filter_map(|(idx, _)| {
                let face = &self.model.faces[idx];
                let surface = &self.model.face_surfaces[face.surface];
                if surface.is_planar() { None } else { Some(idx) }
            })
            .collect();

        // Build edge→triangle adjacency maps for Phase 3 refinement.
        // Build for ALL faces, not just non-planar, because planar neighbors
        // of curved faces also get split when sharing BREP boundary edges.
        for state in &mut self.face_states {
            state.build_edge_tris();
        }

        let mut prev_max_error = f64::MAX;

        for _iteration in 0..MAX_ITERATIONS {
            // Safety: bail if we've exceeded the vertex budget
            if self.vertices.len() > MAX_VERTICES {
                break;
            }

            // 3a. Collect all edges needing refinement from faces in the work set
            // Map: (pool_a, pool_b) -> first face_idx that requested it (for UV computation)
            let mut edges_to_refine: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
            let mut iteration_max_error = 0.0_f64;

            for &face_idx in &faces_to_check {
                let state = &self.face_states[face_idx];
                let face = &self.model.faces[face_idx];
                let surface = &self.model.face_surfaces[face.surface];

                // Pre-evaluate all vertex positions for this face (read-only phase).
                // Each vertex is shared by ~5-6 triangles, so this avoids redundant
                // surface.evaluate() calls.
                let positions: Vec<Point3<f64>> = state
                    .vertices_uv
                    .iter()
                    .map(|uv| surface.evaluate(uv.x, uv.y))
                    .collect();

                for tri in &state.triangles {
                    let error = self.check_triangle_error_state(face_idx, state, tri, &positions);

                    if error.needs_subdivision {
                        iteration_max_error =
                            iteration_max_error.max(error.max_edge_midpoint_error);

                        // Find the edge with maximum error
                        let (local_a, local_b) = match error.max_error_edge {
                            0 => (tri[0], tri[1]),
                            1 => (tri[1], tri[2]),
                            _ => (tri[2], tri[0]),
                        };

                        let pool_a = state.local_to_pool[local_a];
                        let pool_b = state.local_to_pool[local_b];
                        let pool_edge = (pool_a.min(pool_b), pool_a.max(pool_b));

                        // Record all faces that request this edge
                        edges_to_refine.entry(pool_edge).or_default().push(face_idx);
                    }
                }
            }

            if edges_to_refine.is_empty() {
                break;
            }

            // Error-based bail: stop if max error didn't decrease by at least 50%
            if iteration_max_error > prev_max_error * 0.5 {
                break;
            }
            prev_max_error = iteration_max_error;

            // Track which faces get modified for next iteration
            let mut modified_faces: BTreeSet<usize> = BTreeSet::new();

            // 3b. Refine each edge ONCE, update ALL faces that use it
            for ((pool_a, pool_b), requesting_faces) in edges_to_refine {
                // Pick the face with highest curvature at the edge midpoint
                // so that curved surfaces evaluate the midpoint rather than planar neighbors.
                let best_face_idx = *requesting_faces
                    .iter()
                    .max_by(|&&fi_a, &&fi_b| {
                        let ka = self.edge_midpoint_curvature(fi_a, pool_a, pool_b);
                        let kb = self.edge_midpoint_curvature(fi_b, pool_a, pool_b);
                        ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();

                // Look up the local indices from the best face's current state
                let best_state = &self.face_states[best_face_idx];

                // Get local indices for this edge in the best face
                let Some(local_a) = best_state.get_local(pool_a) else {
                    continue; // Edge not in this face anymore (shouldn't happen)
                };
                let Some(local_b) = best_state.get_local(pool_b) else {
                    continue;
                };

                let best_face = &self.model.faces[best_face_idx];
                let best_surface = &self.model.face_surfaces[best_face.surface];

                let uv_a = best_state.vertices_uv[local_a];
                let uv_b = best_state.vertices_uv[local_b];

                // Compute midpoint differently for angular vs non-angular surfaces
                let (_uv_mid, p_mid) = if best_surface.is_angular() {
                    // For angular surfaces (cylinder, cone, sphere, torus), compute
                    // midpoint in 3D to avoid ±π discontinuity issues in UV space
                    let p_a = best_surface.evaluate(uv_a.x, uv_a.y);
                    let p_b = best_surface.evaluate(uv_b.x, uv_b.y);
                    let p_mid_3d = Point3::from((p_a.coords + p_b.coords) / 2.0);
                    // Project back to UV (handles angular coordinates correctly)
                    let uv_mid = best_surface.to_parametric(&p_mid_3d);
                    // Re-evaluate to ensure point is exactly on surface
                    let p_on_surface = best_surface.evaluate(uv_mid.x, uv_mid.y);
                    (uv_mid, p_on_surface)
                } else {
                    // Standard UV averaging for non-angular surfaces
                    let uv_mid = Point2::from((uv_a.coords + uv_b.coords) / 2.0);
                    let p_mid = best_surface.evaluate(uv_mid.x, uv_mid.y);
                    (uv_mid, p_mid)
                };

                let pool_mid = self.add_vertex(p_mid);

                // Check if this is a BREP boundary edge
                let brep_edge_idx = self.pool_edge_to_brep.get(&(pool_a, pool_b)).copied();

                // Collect ALL face indices that need to be updated
                let mut faces_to_update: HashSet<usize> = HashSet::new();
                // Include all faces that requested this edge
                for &fi in &requesting_faces {
                    faces_to_update.insert(fi);
                }

                // If this is a BREP boundary edge, add all adjacent faces
                if let Some(edge_idx) = brep_edge_idx
                    && let Some(edge_uses) = self.edge_adjacency.get(&edge_idx)
                {
                    for edge_use in edge_uses {
                        faces_to_update.insert(edge_use.face_idx);
                    }
                }

                // Verify all faces have both endpoints before splitting.
                // If any face is missing an endpoint, skip this edge entirely
                // to prevent T-junctions from partial splits.
                let all_faces_have_edge = faces_to_update.iter().all(|&fi| {
                    let s = &self.face_states[fi];
                    s.get_local(pool_a).is_some() && s.get_local(pool_b).is_some()
                });
                if !all_faces_have_edge {
                    continue;
                }

                // Split the edge in ALL faces that use it
                for &face_idx in &faces_to_update {
                    let success = self.split_edge_in_face(face_idx, pool_a, pool_b, pool_mid);
                    debug_assert!(success, "split_edge_in_face failed for face {face_idx}");

                    // Track modified faces for next iteration (only non-planar ones)
                    let face = &self.model.faces[face_idx];
                    let surface = &self.model.face_surfaces[face.surface];
                    if !surface.is_planar() {
                        modified_faces.insert(face_idx);
                    }
                }

                // Update pool_edge_to_brep for the new edges
                if let Some(edge_idx) = brep_edge_idx {
                    let key_a = (pool_a.min(pool_mid), pool_a.max(pool_mid));
                    let key_b = (pool_mid.min(pool_b), pool_mid.max(pool_b));
                    self.pool_edge_to_brep.insert(key_a, edge_idx);
                    self.pool_edge_to_brep.insert(key_b, edge_idx);
                }
            }

            // Next iteration only checks faces that were modified
            faces_to_check = modified_faces;
        }
    }

    /// Compute maximum curvature at the midpoint of an edge for a given face.
    /// Used to select the highest-curvature face when multiple faces share an edge,
    /// so that the curved surface evaluates the midpoint rather than a planar neighbor.
    fn edge_midpoint_curvature(&self, face_idx: usize, pool_a: usize, pool_b: usize) -> f64 {
        let state = &self.face_states[face_idx];
        let Some(local_a) = state.get_local(pool_a) else { return 0.0 };
        let Some(local_b) = state.get_local(pool_b) else { return 0.0 };
        let face = &self.model.faces[face_idx];
        let surface = &self.model.face_surfaces[face.surface];
        let uv_a = state.vertices_uv[local_a];
        let uv_b = state.vertices_uv[local_b];
        let uv_mid = if surface.is_angular() {
            let p_a = surface.evaluate(uv_a.x, uv_a.y);
            let p_b = surface.evaluate(uv_b.x, uv_b.y);
            let p_mid_3d = Point3::from((p_a.coords + p_b.coords) / 2.0);
            surface.to_parametric(&p_mid_3d)
        } else {
            Point2::from((uv_a.coords + uv_b.coords) / 2.0)
        };
        let kappa = surface.curvature_at(uv_mid.x, uv_mid.y).kappa_max();
        if kappa.is_finite() { kappa } else { 1e6 }
    }

    // =========================================================================
    // Phase 3.5: Merge Duplicate 3D Vertices
    // =========================================================================

    /// Check triangle error using face state and pre-evaluated vertex positions.
    fn check_triangle_error_state(
        &self,
        face_idx: usize,
        state: &FaceTriangulation,
        tri: &[usize; 3],
        positions: &[Point3<f64>],
    ) -> TriangleError {
        let face = &self.model.faces[face_idx];
        let surface = &self.model.face_surfaces[face.surface];

        let uv0 = state.vertices_uv[tri[0]];
        let uv1 = state.vertices_uv[tri[1]];
        let uv2 = state.vertices_uv[tri[2]];

        let p0 = positions[tri[0]];
        let p1 = positions[tri[1]];
        let p2 = positions[tri[2]];

        // Check centroid error
        let uv_center = Point2::from((uv0.coords + uv1.coords + uv2.coords) / 3.0);
        let p_center_surface = surface.evaluate(uv_center.x, uv_center.y);
        let p_center_linear = Point3::from((p0.coords + p1.coords + p2.coords) / 3.0);
        let centroid_error = (p_center_surface - p_center_linear).norm();

        // Check edge midpoint errors
        let edges = [(uv0, uv1, p0, p1), (uv1, uv2, p1, p2), (uv2, uv0, p2, p0)];

        let mut max_edge_error = 0.0;
        let mut max_error_edge = 0;
        let is_angular = surface.is_angular();

        for (edge_idx, (uv_a, uv_b, p_a, p_b)) in edges.iter().enumerate() {
            // For angular surfaces, compute midpoint UV by projecting 3D midpoint
            let p_mid_surface = if is_angular {
                let p_mid_3d = Point3::from((p_a.coords + p_b.coords) / 2.0);
                let uv_mid = surface.to_parametric(&p_mid_3d);
                surface.evaluate(uv_mid.x, uv_mid.y)
            } else {
                let uv_mid = Point2::from((uv_a.coords + uv_b.coords) / 2.0);
                surface.evaluate(uv_mid.x, uv_mid.y)
            };
            let p_mid_linear = Point3::from((p_a.coords + p_b.coords) / 2.0);
            let edge_error = (p_mid_surface - p_mid_linear).norm();

            if edge_error > max_edge_error {
                max_edge_error = edge_error;
                max_error_edge = edge_idx;
            }
        }

        let max_error = centroid_error.max(max_edge_error);
        let needs_subdivision = max_error > self.effective_tolerance;

        TriangleError {
            centroid_error,
            max_edge_midpoint_error: max_edge_error,
            max_error_edge,
            needs_subdivision,
        }
    }

    /// Split an edge in a face's triangulation.
    /// Updates the face state with the new vertex and re-triangulates affected triangles.
    ///
    /// Returns true if the edge was successfully split, false if the edge wasn't found
    /// in this face (which may indicate an inconsistent edge adjacency).
    #[must_use]
    fn split_edge_in_face(
        &mut self,
        face_idx: usize,
        pool_a: usize,
        pool_b: usize,
        pool_mid: usize,
    ) -> bool {
        let state = &mut self.face_states[face_idx];

        // Check if this face has both endpoints
        let Some(local_a) = state.get_local(pool_a) else {
            // This face doesn't have vertex pool_a - edge not in this face
            return false;
        };
        let Some(local_b) = state.get_local(pool_b) else {
            // Face has pool_a but not pool_b - this indicates inconsistent edge adjacency
            return false;
        };

        let face = &self.model.faces[face_idx];
        let surface = &self.model.face_surfaces[face.surface];

        // Compute UV for midpoint (same logic as in phase3_global_refinement)
        let uv_a = state.vertices_uv[local_a];
        let uv_b = state.vertices_uv[local_b];
        let uv_mid = if surface.is_angular() {
            // For angular surfaces, get 3D midpoint and project back to UV
            let p_a = surface.evaluate(uv_a.x, uv_a.y);
            let p_b = surface.evaluate(uv_b.x, uv_b.y);
            let p_mid_3d = Point3::from((p_a.coords + p_b.coords) / 2.0);
            surface.to_parametric(&p_mid_3d)
        } else {
            Point2::from((uv_a.coords + uv_b.coords) / 2.0)
        };

        // Add the midpoint to local state
        let local_mid = state.add_vertex(uv_mid, pool_mid);

        // Update boundary edge tracking if this was a boundary edge
        if state.is_boundary_edge(local_a, local_b) {
            state
                .boundary_edges
                .remove(&(local_a.min(local_b), local_a.max(local_b)));
            state.mark_boundary_edge(local_a, local_mid);
            state.mark_boundary_edge(local_mid, local_b);
        }

        // Split triangles using edge_tris adjacency map (O(1) lookup, not O(T) scan).
        // Return value intentionally ignored: edge may not appear in any triangle
        // if the face's CDT failed and used a fallback triangulation.
        let _ = state.split_triangles_at_edge(local_a, local_b, local_mid);

        true
    }

    // =========================================================================
    // Phase 4: Final Assembly
    // =========================================================================

    /// Phase 4: Convert face states to final triangles and compute normals.
    fn phase4_final_assembly(&mut self) {
        // Initialize normals for all vertices
        self.normals = vec![Vector3::zeros(); self.vertices.len()];

        for (face_idx, state) in self.face_states.iter().enumerate() {
            let face = &self.model.faces[face_idx];
            let surface = &self.model.face_surfaces[face.surface];

            // Accumulate normals from all contributing faces.
            // Shared vertices get averaged normals for smooth shading across face boundaries.
            for (local_idx, uv) in state.vertices_uv.iter().enumerate() {
                let pool_idx = state.local_to_pool[local_idx];
                let n = surface.normal_at(uv.x, uv.y);
                let signed_n = if face.same_sense { n } else { -n };
                self.normals[pool_idx] += signed_n;
            }

            // Convert local triangles to pool indices
            for tri in &state.triangles {
                let pool_tri = if face.same_sense {
                    [
                        state.local_to_pool[tri[0]],
                        state.local_to_pool[tri[1]],
                        state.local_to_pool[tri[2]],
                    ]
                } else {
                    // Flip winding for reversed faces
                    [
                        state.local_to_pool[tri[0]],
                        state.local_to_pool[tri[2]],
                        state.local_to_pool[tri[1]],
                    ]
                };
                self.triangles.push(pool_tri);
                self.face_indices.push(face_idx);
            }
        }

        // Normalize accumulated normals
        for n in &mut self.normals {
            let len = n.norm();
            if len > 1e-12 {
                *n /= len;
            }
        }
    }

    /// Tessellate all faces and return a Trimesh.
    fn tessellate(mut self) -> Trimesh {
        let t = std::time::Instant::now();

        // Phase 1: Discretize all edges globally
        self.phase1_discretize_all_edges();

        // Phase 1.5: Refine outer contours where inner loops escape the outer polygon
        self.phase1_5_refine_outer_contours();
        let p1 = t.elapsed();

        // Phase 2: Initial triangulation for each face (no subdivision)
        for face_idx in 0..self.model.faces.len() {
            let state = self.phase2_initial_triangulation(face_idx);
            self.face_states.push(state);
        }
        let p2 = t.elapsed() - p1;

        // Phase 3: Global refinement loop
        self.phase3_global_refinement();
        let p3 = t.elapsed() - p1 - p2;

        // Phase 4: Final assembly
        self.phase4_final_assembly();
        let p4 = t.elapsed() - p1 - p2 - p3;

        #[cfg(test)]
        eprintln!(
            "    tess phases: edge={:.1}ms CDT={:.1}ms refine={:.1}ms assemble={:.1}ms",
            p1.as_secs_f64() * 1e3,
            p2.as_secs_f64() * 1e3,
            p3.as_secs_f64() * 1e3,
            p4.as_secs_f64() * 1e3,
        );

        // Build Trimesh with attributes
        let mut attrs_vertex = Attributes::default();
        attrs_vertex.normals.push(self.normals);

        let mut attrs_face = Attributes::default();
        attrs_face.groupings.push(Grouping {
            kind: GroupingKind::Surface,
            names: vec![], // Could add face names if desired
            indices: self.face_indices,
        });

        Trimesh::new(
            self.vertices,
            self.triangles,
            Some(attrs_vertex),
            Some(attrs_face),
        )
        .expect("tessellation produced valid mesh")
    }
}

impl BrepModel {
    /// Tesselate the entire BREP model into a triangle mesh.
    ///
    /// This method ensures watertight output by deduplicating edge vertices:
    /// adjacent faces sharing an edge will use the same vertex positions.
    ///
    /// Returns a `Trimesh` with:
    /// - `vertices`: 3D vertex positions
    /// - `faces`: triangle indices
    /// - `attributes_vertex.normals[0]`: per-vertex normals
    /// - `attributes_face.groupings`: face-to-surface mapping with `GroupingKind::Surface`
    pub fn tesselate(&self, params: &TesselationParams) -> Trimesh {
        ShellTessellator::new(self, params).tessellate()
    }

}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, PI};

    use super::super::topology::{CurveCircle, CurveLine};

    #[test]
    fn test_plane_parametric_roundtrip() {
        let plane = SurfacePlane {
            origin: Point3::new(1.0, 2.0, 3.0),
            normal: Vector3::new(0.0, 0.0, 1.0),
        };

        let test_points = [
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(5.0, 7.0, 3.0),
            Point3::new(-2.0, 0.0, 3.0),
        ];

        for p in &test_points {
            let uv = plane.to_parametric(p);
            let back = plane.evaluate(uv.x, uv.y);
            assert_relative_eq!(back.x, p.x, epsilon = 1e-10);
            assert_relative_eq!(back.y, p.y, epsilon = 1e-10);
            assert_relative_eq!(back.z, p.z, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cylinder_parametric_roundtrip() {
        let cyl = Cylinder::new(Point3::origin(), Vector3::z(), 2.0);

        // Test points on the cylinder
        for theta in [0.0, FRAC_PI_2, PI, -FRAC_PI_2] {
            for h in [-1.0, 0.0, 1.0, 5.0] {
                let p = cyl.evaluate(theta, h);
                let uv = cyl.to_parametric(&p);
                let back = cyl.evaluate(uv.x, uv.y);
                assert_relative_eq!(back.x, p.x, epsilon = 1e-10);
                assert_relative_eq!(back.y, p.y, epsilon = 1e-10);
                assert_relative_eq!(back.z, p.z, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sphere_parametric_roundtrip() {
        let sphere = Sphere {
            center: Point3::new(1.0, 2.0, 3.0),
            radius: 5.0,
        };

        // Test points on the sphere (avoid poles for atan2 stability)
        for lon in [-PI, -FRAC_PI_2, 0.0, FRAC_PI_2, PI - 0.01] {
            for lat in [-FRAC_PI_2 + 0.1, 0.0, FRAC_PI_2 - 0.1] {
                let p = sphere.evaluate(lon, lat);
                let uv = sphere.to_parametric(&p);
                let back = sphere.evaluate(uv.x, uv.y);
                assert_relative_eq!(back.x, p.x, epsilon = 1e-9);
                assert_relative_eq!(back.y, p.y, epsilon = 1e-9);
                assert_relative_eq!(back.z, p.z, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_tesselate_triangle_face() {
        let mut model = BrepModel::new();

        // Create a triangular face
        let v0 = model.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = model.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = model.add_vertex(Point3::new(0.5, 1.0, 0.0));

        let c0 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(1.0, 0.0, 0.0),
        }));
        let c1 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(1.0, 0.0, 0.0),
            direction: Vector3::new(-0.5, 1.0, 0.0),
        }));
        let c2 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.5, 1.0, 0.0),
            direction: Vector3::new(-0.5, -1.0, 0.0),
        }));

        let e0 = model.add_edge(c0, v0, v1, 0.0, 1.0);
        let e1 = model.add_edge(c1, v1, v2, 0.0, 1.0);
        let e2 = model.add_edge(c2, v2, v0, 0.0, 1.0);

        let loop_idx = model.add_loop(vec![
            OrientedEdge {
                edge: e0,
                same_sense: true,
            },
            OrientedEdge {
                edge: e1,
                same_sense: true,
            },
            OrientedEdge {
                edge: e2,
                same_sense: true,
            },
        ]);

        let surf_idx = model.add_surface(Surface::Plane(SurfacePlane {
            origin: Point3::origin(),
            normal: Vector3::z(),
        }));

        model.add_face(surf_idx, loop_idx, vec![], true);

        let params = TesselationParams::default();
        let result = model.tesselate(&params);

        assert!(!result.vertices.is_empty());
        assert!(!result.faces.is_empty());

        // All normals should point in +Z
        let normals = &result.attributes_vertex.normals[0];
        for n in normals {
            assert_relative_eq!(n.z, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tesselate_square_with_hole() {
        let mut model = BrepModel::new();

        // Outer square: 4 vertices
        let v0 = model.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = model.add_vertex(Point3::new(2.0, 0.0, 0.0));
        let v2 = model.add_vertex(Point3::new(2.0, 2.0, 0.0));
        let v3 = model.add_vertex(Point3::new(0.0, 2.0, 0.0));

        // Inner square (hole): 4 vertices
        let h0 = model.add_vertex(Point3::new(0.5, 0.5, 0.0));
        let h1 = model.add_vertex(Point3::new(1.5, 0.5, 0.0));
        let h2 = model.add_vertex(Point3::new(1.5, 1.5, 0.0));
        let h3 = model.add_vertex(Point3::new(0.5, 1.5, 0.0));

        // Line curves for outer edges - curve eval at t must give vertex positions
        // v0->v1: (0,0,0) to (2,0,0)
        let c0 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(2.0, 0.0, 0.0),
        }));
        // v1->v2: (2,0,0) to (2,2,0)
        let c1 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(2.0, 0.0, 0.0),
            direction: Vector3::new(0.0, 2.0, 0.0),
        }));
        // v2->v3: (2,2,0) to (0,2,0)
        let c2 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(2.0, 2.0, 0.0),
            direction: Vector3::new(-2.0, 0.0, 0.0),
        }));
        // v3->v0: (0,2,0) to (0,0,0)
        let c3 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.0, 2.0, 0.0),
            direction: Vector3::new(0.0, -2.0, 0.0),
        }));

        // Line curves for inner edges (CW winding for hole)
        // h0->h3: (0.5,0.5,0) to (0.5,1.5,0) - up
        let hc0 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.5, 0.5, 0.0),
            direction: Vector3::new(0.0, 1.0, 0.0),
        }));
        // h3->h2: (0.5,1.5,0) to (1.5,1.5,0) - right
        let hc1 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.5, 1.5, 0.0),
            direction: Vector3::new(1.0, 0.0, 0.0),
        }));
        // h2->h1: (1.5,1.5,0) to (1.5,0.5,0) - down
        let hc2 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(1.5, 1.5, 0.0),
            direction: Vector3::new(0.0, -1.0, 0.0),
        }));
        // h1->h0: (1.5,0.5,0) to (0.5,0.5,0) - left
        let hc3 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(1.5, 0.5, 0.0),
            direction: Vector3::new(-1.0, 0.0, 0.0),
        }));

        // Outer edges (t from 0 to 1)
        let e0 = model.add_edge(c0, v0, v1, 0.0, 1.0);
        let e1 = model.add_edge(c1, v1, v2, 0.0, 1.0);
        let e2 = model.add_edge(c2, v2, v3, 0.0, 1.0);
        let e3 = model.add_edge(c3, v3, v0, 0.0, 1.0);

        // Inner edges (CW for hole: h0->h3->h2->h1->h0)
        let h_e0 = model.add_edge(hc0, h0, h3, 0.0, 1.0);
        let h_e1 = model.add_edge(hc1, h3, h2, 0.0, 1.0);
        let h_e2 = model.add_edge(hc2, h2, h1, 0.0, 1.0);
        let h_e3 = model.add_edge(hc3, h1, h0, 0.0, 1.0);

        let outer_loop = model.add_loop(vec![
            OrientedEdge {
                edge: e0,
                same_sense: true,
            },
            OrientedEdge {
                edge: e1,
                same_sense: true,
            },
            OrientedEdge {
                edge: e2,
                same_sense: true,
            },
            OrientedEdge {
                edge: e3,
                same_sense: true,
            },
        ]);

        let inner_loop = model.add_loop(vec![
            OrientedEdge {
                edge: h_e0,
                same_sense: true,
            },
            OrientedEdge {
                edge: h_e1,
                same_sense: true,
            },
            OrientedEdge {
                edge: h_e2,
                same_sense: true,
            },
            OrientedEdge {
                edge: h_e3,
                same_sense: true,
            },
        ]);

        let surf_idx = model.add_surface(Surface::Plane(SurfacePlane {
            origin: Point3::origin(),
            normal: Vector3::z(),
        }));

        model.add_face(surf_idx, outer_loop, vec![inner_loop], true);

        let params = TesselationParams::default();
        let result = model.tesselate(&params);

        assert!(!result.vertices.is_empty());
        assert!(!result.faces.is_empty());

        // Should have more triangles than a simple square (because of the hole)
        assert!(result.faces.len() >= 4);
    }

    #[test]
    fn test_watertight_shared_edge() {
        // Create two triangular faces sharing an edge.
        // Verify that the shared edge vertices are deduplicated.
        //
        //     v2
        //    /|\
        //   / | \
        //  /  |  \
        // v0--+--v1
        //  \  |  /
        //   \ | /
        //    \|/
        //     v3
        //
        // Face 1: v0 -> v1 -> v2 (upper triangle)
        // Face 2: v0 -> v3 -> v1 (lower triangle)
        // Shared edge: v0 -> v1 (used forward in face1, reversed in face2)

        let mut model = BrepModel::new();

        let v0 = model.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = model.add_vertex(Point3::new(2.0, 0.0, 0.0));
        let v2 = model.add_vertex(Point3::new(1.0, 1.0, 0.0));
        let v3 = model.add_vertex(Point3::new(1.0, -1.0, 0.0));

        // Curves for edges
        let c_v0_v1 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(2.0, 0.0, 0.0),
        }));
        let c_v1_v2 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(2.0, 0.0, 0.0),
            direction: Vector3::new(-1.0, 1.0, 0.0),
        }));
        let c_v2_v0 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(1.0, 1.0, 0.0),
            direction: Vector3::new(-1.0, -1.0, 0.0),
        }));
        let c_v0_v3 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(1.0, -1.0, 0.0),
        }));
        let c_v3_v1 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(1.0, -1.0, 0.0),
            direction: Vector3::new(1.0, 1.0, 0.0),
        }));

        // SHARED edge v0->v1
        let e_shared = model.add_edge(c_v0_v1, v0, v1, 0.0, 1.0);
        // Upper triangle edges
        let e_v1_v2 = model.add_edge(c_v1_v2, v1, v2, 0.0, 1.0);
        let e_v2_v0 = model.add_edge(c_v2_v0, v2, v0, 0.0, 1.0);
        // Lower triangle edges
        let e_v0_v3 = model.add_edge(c_v0_v3, v0, v3, 0.0, 1.0);
        let e_v3_v1 = model.add_edge(c_v3_v1, v3, v1, 0.0, 1.0);

        // Upper face: v0 -> v1 -> v2 -> v0 (CCW when looking from +Z)
        let loop1 = model.add_loop(vec![
            OrientedEdge {
                edge: e_shared,
                same_sense: true,
            }, // v0 -> v1
            OrientedEdge {
                edge: e_v1_v2,
                same_sense: true,
            }, // v1 -> v2
            OrientedEdge {
                edge: e_v2_v0,
                same_sense: true,
            }, // v2 -> v0
        ]);

        // Lower face: v1 -> v0 -> v3 -> v1 (CCW when looking from -Z)
        // Uses shared edge REVERSED: v1 -> v0
        let loop2 = model.add_loop(vec![
            OrientedEdge {
                edge: e_shared,
                same_sense: false,
            }, // v1 -> v0 (reversed)
            OrientedEdge {
                edge: e_v0_v3,
                same_sense: true,
            }, // v0 -> v3
            OrientedEdge {
                edge: e_v3_v1,
                same_sense: true,
            }, // v3 -> v1
        ]);

        let surf_idx = model.add_surface(Surface::Plane(SurfacePlane {
            origin: Point3::origin(),
            normal: Vector3::z(),
        }));

        model.add_face(surf_idx, loop1, vec![], true);
        model.add_face(surf_idx, loop2, vec![], false); // Opposite face normal

        let params = TesselationParams::default();

        // Test watertight tessellation
        let result = model.tesselate(&params);

        // With deduplication, we should have exactly 4 vertices (v0, v1, v2, v3)
        // Without deduplication, we'd have 6 vertices (3 per face)
        assert_eq!(
            result.vertices.len(),
            4,
            "Expected 4 deduplicated vertices, got {}",
            result.vertices.len()
        );

        // Should have 2 triangles
        assert_eq!(result.faces.len(), 2);
    }

    #[test]
    fn test_watertight_cylinder_edges() {
        // Create two faces on a cylinder sharing a circular edge.
        // This tests that curved edge discretization is shared.
        use std::f64::consts::FRAC_PI_2;

        let mut model = BrepModel::new();

        // Four vertices forming a "tube segment"
        let v0 = model.add_vertex(Point3::new(1.0, 0.0, 0.0)); // bottom, theta=0
        let v1 = model.add_vertex(Point3::new(0.0, 1.0, 0.0)); // bottom, theta=pi/2
        let v2 = model.add_vertex(Point3::new(1.0, 0.0, 1.0)); // top, theta=0
        let v3 = model.add_vertex(Point3::new(0.0, 1.0, 1.0)); // top, theta=pi/2

        // Bottom arc (circular)
        let c_bottom = model.add_curve(Curve::Circle(CurveCircle {
            center: Point3::origin(),
            axis: Vector3::z(),
            x_axis: Vector3::x(),
            radius: 1.0,
        }));

        // Top arc (circular)
        let c_top = model.add_curve(Curve::Circle(CurveCircle {
            center: Point3::new(0.0, 0.0, 1.0),
            axis: Vector3::z(),
            x_axis: Vector3::x(),
            radius: 1.0,
        }));

        // Vertical lines
        let c_line1 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(1.0, 0.0, 0.0),
            direction: Vector3::z(),
        }));
        let c_line2 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.0, 1.0, 0.0),
            direction: Vector3::z(),
        }));

        // Edges
        let e_bottom = model.add_edge(c_bottom, v0, v1, 0.0, FRAC_PI_2);
        let e_top = model.add_edge(c_top, v2, v3, 0.0, FRAC_PI_2);
        let e_line1 = model.add_edge(c_line1, v0, v2, 0.0, 1.0);
        let e_line2 = model.add_edge(c_line2, v1, v3, 0.0, 1.0);

        // Cylindrical face: bottom arc -> line2 -> top arc (reversed) -> line1 (reversed)
        let loop1 = model.add_loop(vec![
            OrientedEdge {
                edge: e_bottom,
                same_sense: true,
            },
            OrientedEdge {
                edge: e_line2,
                same_sense: true,
            },
            OrientedEdge {
                edge: e_top,
                same_sense: false,
            },
            OrientedEdge {
                edge: e_line1,
                same_sense: false,
            },
        ]);

        let cyl_surf = model.add_surface(Surface::Cylinder(Cylinder::new(
            Point3::origin(),
            Vector3::z(),
            1.0,
        )));

        model.add_face(cyl_surf, loop1, vec![], true);

        let params = TesselationParams {
            tolerance: 0.01,
            min_segments: 4,
            max_segments: 256,
            ..Default::default()
        };

        let result = model.tesselate(&params);

        // Verify we have some triangles
        assert!(!result.faces.is_empty());

        // Verify all vertices are on or near the cylinder surface
        for v in &result.vertices {
            let r = (v.x * v.x + v.y * v.y).sqrt();
            assert_relative_eq!(r, 1.0, epsilon = 0.01);
            assert!(v.z >= -0.01 && v.z <= 1.01);
        }
    }

    #[test]
    fn test_mesh_validation_watertight() {
        // Create a closed tetrahedron (4 faces, 4 vertices)
        // This should be a watertight mesh
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 0.866, 0.0),
            Point3::new(0.5, 0.289, 0.816),
        ];

        // All faces must have consistent winding (outward normals)
        let triangles = vec![
            [0, 2, 1], // bottom (outward = -Z)
            [0, 1, 3], // front
            [1, 2, 3], // right
            [2, 0, 3], // left
        ];

        let mesh = Trimesh::new(vertices, triangles, None, None).expect("valid mesh");

        // A tetrahedron should be watertight
        assert!(mesh.is_watertight());
    }

    #[test]
    fn test_mesh_validation_with_hole() {
        // Single triangle - has 3 boundary edges
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];

        let triangles = vec![[0, 1, 2]];

        let mesh = Trimesh::new(vertices, triangles, None, None).expect("valid mesh");

        // A single triangle is not watertight (has boundary edges)
        assert!(!mesh.is_watertight());
    }

    #[test]
    fn test_mesh_validation_non_manifold() {
        // Three triangles sharing the same edge - non-manifold
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, -1.0, 0.0),
            Point3::new(0.5, 0.0, 1.0),
        ];

        // All three triangles share edge (0, 1)
        let triangles = vec![
            [0, 1, 2], // top
            [0, 1, 3], // bottom (note: same edge direction as top - non-manifold)
            [0, 1, 4], // front
        ];

        let mesh = Trimesh::new(vertices, triangles, None, None).expect("valid mesh");

        // Edge (0, 1) is used 3 times - not watertight
        assert!(!mesh.is_watertight());
    }

    #[test]
    fn test_watertight_shared_edge_validation() {
        // Build the same model from test_watertight_shared_edge and validate it
        let mut model = BrepModel::new();

        let v0 = model.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = model.add_vertex(Point3::new(2.0, 0.0, 0.0));
        let v2 = model.add_vertex(Point3::new(1.0, 1.0, 0.0));
        let v3 = model.add_vertex(Point3::new(1.0, -1.0, 0.0));

        let c_v0_v1 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(2.0, 0.0, 0.0),
        }));
        let c_v1_v2 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(2.0, 0.0, 0.0),
            direction: Vector3::new(-1.0, 1.0, 0.0),
        }));
        let c_v2_v0 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(1.0, 1.0, 0.0),
            direction: Vector3::new(-1.0, -1.0, 0.0),
        }));
        let c_v0_v3 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(1.0, -1.0, 0.0),
        }));
        let c_v3_v1 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(1.0, -1.0, 0.0),
            direction: Vector3::new(1.0, 1.0, 0.0),
        }));

        let e_shared = model.add_edge(c_v0_v1, v0, v1, 0.0, 1.0);
        let e_v1_v2 = model.add_edge(c_v1_v2, v1, v2, 0.0, 1.0);
        let e_v2_v0 = model.add_edge(c_v2_v0, v2, v0, 0.0, 1.0);
        let e_v0_v3 = model.add_edge(c_v0_v3, v0, v3, 0.0, 1.0);
        let e_v3_v1 = model.add_edge(c_v3_v1, v3, v1, 0.0, 1.0);

        let loop1 = model.add_loop(vec![
            OrientedEdge {
                edge: e_shared,
                same_sense: true,
            },
            OrientedEdge {
                edge: e_v1_v2,
                same_sense: true,
            },
            OrientedEdge {
                edge: e_v2_v0,
                same_sense: true,
            },
        ]);

        let loop2 = model.add_loop(vec![
            OrientedEdge {
                edge: e_shared,
                same_sense: false,
            },
            OrientedEdge {
                edge: e_v0_v3,
                same_sense: true,
            },
            OrientedEdge {
                edge: e_v3_v1,
                same_sense: true,
            },
        ]);

        let surf_idx = model.add_surface(Surface::Plane(SurfacePlane {
            origin: Point3::origin(),
            normal: Vector3::z(),
        }));

        model.add_face(surf_idx, loop1, vec![], true);
        model.add_face(surf_idx, loop2, vec![], false);

        let params = TesselationParams::default();
        let result = model.tesselate(&params);

        // 2 triangles form a bow-tie shape - not watertight since it's not a closed surface
        // The shared edge should be properly deduplicated (vertices shared)
        assert_eq!(
            result.vertices.len(),
            4,
            "Should have 4 deduplicated vertices"
        );

        // Verify we have the expected number of triangles
        assert_eq!(result.faces.len(), 2, "Expected 2 triangles");
    }

    /// Helper to create a unit cube BREP model with proper edge sharing.
    /// The cube has:
    /// - 8 vertices at (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)
    /// - 12 edges (each shared by exactly 2 faces with opposite orientations)
    /// - 6 faces (planar quadrilaterals, each triangulated into 2 triangles)
    fn create_watertight_cube() -> BrepModel {
        let mut model = BrepModel::new();

        // 8 vertex positions for a unit cube
        let pts = [
            Point3::new(0.0, 0.0, 0.0), // 0: bottom-back-left
            Point3::new(1.0, 0.0, 0.0), // 1: bottom-back-right
            Point3::new(1.0, 1.0, 0.0), // 2: bottom-front-right
            Point3::new(0.0, 1.0, 0.0), // 3: bottom-front-left
            Point3::new(0.0, 0.0, 1.0), // 4: top-back-left
            Point3::new(1.0, 0.0, 1.0), // 5: top-back-right
            Point3::new(1.0, 1.0, 1.0), // 6: top-front-right
            Point3::new(0.0, 1.0, 1.0), // 7: top-front-left
        ];

        let v0 = model.add_vertex(pts[0]);
        let v1 = model.add_vertex(pts[1]);
        let v2 = model.add_vertex(pts[2]);
        let v3 = model.add_vertex(pts[3]);
        let v4 = model.add_vertex(pts[4]);
        let v5 = model.add_vertex(pts[5]);
        let v6 = model.add_vertex(pts[6]);
        let v7 = model.add_vertex(pts[7]);

        // Helper to create a line curve between two points
        let line = |from: Point3<f64>, to: Point3<f64>| -> Curve {
            Curve::Line(CurveLine {
                origin: from,
                direction: to - from,
            })
        };

        // 12 curves for the cube edges
        // Bottom face edges (z=0)
        let c_01 = model.add_curve(line(pts[0], pts[1]));
        let c_12 = model.add_curve(line(pts[1], pts[2]));
        let c_23 = model.add_curve(line(pts[2], pts[3]));
        let c_30 = model.add_curve(line(pts[3], pts[0]));

        // Top face edges (z=1)
        let c_45 = model.add_curve(line(pts[4], pts[5]));
        let c_56 = model.add_curve(line(pts[5], pts[6]));
        let c_67 = model.add_curve(line(pts[6], pts[7]));
        let c_74 = model.add_curve(line(pts[7], pts[4]));

        // Vertical edges
        let c_04 = model.add_curve(line(pts[0], pts[4]));
        let c_15 = model.add_curve(line(pts[1], pts[5]));
        let c_26 = model.add_curve(line(pts[2], pts[6]));
        let c_37 = model.add_curve(line(pts[3], pts[7]));

        // Create edges (curve, start_vertex, end_vertex, t_start, t_end)
        let e_01 = model.add_edge(c_01, v0, v1, 0.0, 1.0);
        let e_12 = model.add_edge(c_12, v1, v2, 0.0, 1.0);
        let e_23 = model.add_edge(c_23, v2, v3, 0.0, 1.0);
        let e_30 = model.add_edge(c_30, v3, v0, 0.0, 1.0);

        let e_45 = model.add_edge(c_45, v4, v5, 0.0, 1.0);
        let e_56 = model.add_edge(c_56, v5, v6, 0.0, 1.0);
        let e_67 = model.add_edge(c_67, v6, v7, 0.0, 1.0);
        let e_74 = model.add_edge(c_74, v7, v4, 0.0, 1.0);

        let e_04 = model.add_edge(c_04, v0, v4, 0.0, 1.0);
        let e_15 = model.add_edge(c_15, v1, v5, 0.0, 1.0);
        let e_26 = model.add_edge(c_26, v2, v6, 0.0, 1.0);
        let e_37 = model.add_edge(c_37, v3, v7, 0.0, 1.0);

        // Create 6 faces with proper edge sharing
        // Each edge is used exactly twice: once forward, once reversed

        // Face 0: Bottom (z=0), normal -Z, vertices 0-1-2-3 (CW from outside)
        // For outward normal (-Z), we go CCW when looking from outside: 0->3->2->1->0
        let loop_bottom = model.add_loop(vec![
            OrientedEdge {
                edge: e_30,
                same_sense: false,
            }, // 0->3
            OrientedEdge {
                edge: e_23,
                same_sense: false,
            }, // 3->2
            OrientedEdge {
                edge: e_12,
                same_sense: false,
            }, // 2->1
            OrientedEdge {
                edge: e_01,
                same_sense: false,
            }, // 1->0
        ]);
        let surf_bottom = model.add_surface(Surface::Plane(SurfacePlane {
            origin: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::new(0.0, 0.0, -1.0),
        }));
        model.add_face(surf_bottom, loop_bottom, vec![], true);

        // Face 1: Top (z=1), normal +Z, vertices 4-5-6-7 (CCW from outside)
        let loop_top = model.add_loop(vec![
            OrientedEdge {
                edge: e_45,
                same_sense: true,
            }, // 4->5
            OrientedEdge {
                edge: e_56,
                same_sense: true,
            }, // 5->6
            OrientedEdge {
                edge: e_67,
                same_sense: true,
            }, // 6->7
            OrientedEdge {
                edge: e_74,
                same_sense: true,
            }, // 7->4
        ]);
        let surf_top = model.add_surface(Surface::Plane(SurfacePlane {
            origin: Point3::new(0.0, 0.0, 1.0),
            normal: Vector3::new(0.0, 0.0, 1.0),
        }));
        model.add_face(surf_top, loop_top, vec![], true);

        // Face 2: Front (y=1), normal +Y, vertices 3-2-6-7 (CCW from outside)
        let loop_front = model.add_loop(vec![
            OrientedEdge {
                edge: e_23,
                same_sense: true,
            }, // 2->3
            OrientedEdge {
                edge: e_37,
                same_sense: true,
            }, // 3->7
            OrientedEdge {
                edge: e_67,
                same_sense: false,
            }, // 7->6
            OrientedEdge {
                edge: e_26,
                same_sense: false,
            }, // 6->2
        ]);
        let surf_front = model.add_surface(Surface::Plane(SurfacePlane {
            origin: Point3::new(0.0, 1.0, 0.0),
            normal: Vector3::new(0.0, 1.0, 0.0),
        }));
        model.add_face(surf_front, loop_front, vec![], true);

        // Face 3: Back (y=0), normal -Y, vertices 0-1-5-4 (CCW from outside)
        let loop_back = model.add_loop(vec![
            OrientedEdge {
                edge: e_01,
                same_sense: true,
            }, // 0->1
            OrientedEdge {
                edge: e_15,
                same_sense: true,
            }, // 1->5
            OrientedEdge {
                edge: e_45,
                same_sense: false,
            }, // 5->4
            OrientedEdge {
                edge: e_04,
                same_sense: false,
            }, // 4->0
        ]);
        let surf_back = model.add_surface(Surface::Plane(SurfacePlane {
            origin: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::new(0.0, -1.0, 0.0),
        }));
        model.add_face(surf_back, loop_back, vec![], true);

        // Face 4: Right (x=1), normal +X, vertices 1-2-6-5 (CCW from outside)
        let loop_right = model.add_loop(vec![
            OrientedEdge {
                edge: e_12,
                same_sense: true,
            }, // 1->2
            OrientedEdge {
                edge: e_26,
                same_sense: true,
            }, // 2->6
            OrientedEdge {
                edge: e_56,
                same_sense: false,
            }, // 6->5
            OrientedEdge {
                edge: e_15,
                same_sense: false,
            }, // 5->1
        ]);
        let surf_right = model.add_surface(Surface::Plane(SurfacePlane {
            origin: Point3::new(1.0, 0.0, 0.0),
            normal: Vector3::new(1.0, 0.0, 0.0),
        }));
        model.add_face(surf_right, loop_right, vec![], true);

        // Face 5: Left (x=0), normal -X, vertices 0-3-7-4 (CCW from outside)
        let loop_left = model.add_loop(vec![
            OrientedEdge {
                edge: e_30,
                same_sense: true,
            }, // 3->0
            OrientedEdge {
                edge: e_04,
                same_sense: true,
            }, // 0->4
            OrientedEdge {
                edge: e_74,
                same_sense: false,
            }, // 4->7
            OrientedEdge {
                edge: e_37,
                same_sense: false,
            }, // 7->3
        ]);
        let surf_left = model.add_surface(Surface::Plane(SurfacePlane {
            origin: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::new(-1.0, 0.0, 0.0),
        }));
        model.add_face(surf_left, loop_left, vec![], true);

        model
    }

    #[test]
    fn test_watertight_cube() {
        let model = create_watertight_cube();

        // Verify the BREP model is valid
        let brep_errors = model.validate();
        assert!(
            brep_errors.is_empty(),
            "BREP validation failed: {:?}",
            brep_errors
        );

        // Verify edge sharing is correct for watertight mesh
        let edge_errors = model.validate_edge_sharing();
        assert!(
            edge_errors.is_empty(),
            "Edge sharing validation failed: {:?}",
            edge_errors
        );

        // Tessellate the cube
        let params = TesselationParams::default();
        let mesh = model.tesselate(&params);

        // A cube has 6 faces, each quad becomes 2 triangles = 12 triangles
        assert_eq!(
            mesh.faces.len(),
            12,
            "Expected 12 triangles for cube, got {}",
            mesh.faces.len()
        );

        // Validate mesh is watertight using Trimesh's is_watertight method
        assert!(mesh.is_watertight(), "Cube mesh should be watertight");

        // Also check winding consistency
        assert!(
            mesh.is_winding_consistent(),
            "Cube mesh should have consistent winding"
        );
    }

    #[test]
    fn test_cube_vertex_sharing() {
        // Verify that adjacent faces share the same vertex pool indices
        let model = create_watertight_cube();
        let params = TesselationParams::default();
        let mesh = model.tesselate(&params);

        // A unit cube should have exactly 8 vertices (no duplicates)
        assert_eq!(
            mesh.vertices.len(),
            8,
            "Expected 8 vertices for cube with shared edges, got {}",
            mesh.vertices.len()
        );

        // Verify vertices are at expected positions
        let mut found_corners = vec![false; 8];
        let expected_corners = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ];

        for v in &mesh.vertices {
            for (i, expected) in expected_corners.iter().enumerate() {
                if (v - expected).norm() < 1e-9 {
                    found_corners[i] = true;
                    break;
                }
            }
        }

        for (i, &found) in found_corners.iter().enumerate() {
            assert!(
                found,
                "Missing cube corner vertex at {:?}",
                expected_corners[i]
            );
        }
    }

    #[test]
    fn test_merge_near_vertices_no_duplicates() {
        // When no points are within tolerance, nothing should be merged
        let uvs = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let contours = vec![vec![0, 1, 2, 3, 0]];
        let result = merge_near_vertices(&uvs, &contours, 1e-8);

        assert_eq!(result.merged_uvs.len(), 4);
        assert_eq!(result.merged_contours.len(), 1);
        assert_eq!(result.merged_contours[0], vec![0, 1, 2, 3, 0]);
    }

    #[test]
    fn test_merge_near_vertices_with_duplicates() {
        // Points within tolerance should be merged
        let uvs = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0 + 1e-10, 0.0 + 1e-10), // Near duplicate of index 1
            Point2::new(0.0, 1.0),
        ];
        let contours = vec![vec![0, 1, 2, 3, 0]];
        let result = merge_near_vertices(&uvs, &contours, 1e-8);

        // Index 2 should merge with index 1
        assert_eq!(result.merged_uvs.len(), 3);
        // Contour should have consecutive duplicates removed
        assert_eq!(result.merged_contours[0].len(), 4); // [0, 1, 2, 0] with 2 mapped to 1 -> [0, 1, 2, 0]
    }

    #[test]
    fn test_surface_is_angular() {
        // Test that is_angular correctly identifies angular surfaces
        let plane = Surface::Plane(SurfacePlane {
            origin: Point3::origin(),
            normal: Vector3::z(),
        });
        assert!(!plane.is_angular());

        let cylinder = Surface::Cylinder(Cylinder::new(Point3::origin(), Vector3::z(), 1.0));
        assert!(cylinder.is_angular());

        let sphere = Surface::Sphere(Sphere {
            center: Point3::origin(),
            radius: 1.0,
        });
        assert!(sphere.is_angular());

        let cone = Surface::Cone(Cone::new(Point3::origin(), Vector3::z(), 0.5));
        assert!(cone.is_angular());

        let torus = Surface::Torus(Torus::new(Point3::origin(), Vector3::z(), 2.0, 0.5));
        assert!(torus.is_angular());
    }

    #[test]
    fn test_tesselation_params_default_has_merge_tolerance() {
        let params = TesselationParams::default();
        assert!(params.merge_tolerance > 0.0);
        assert!(params.merge_tolerance < 1e-6); // Should be small
    }

    #[test]
    fn test_point_in_polygon_convex() {
        // Unit square: (0,0), (1,0), (1,1), (0,1)
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        // Interior
        assert!(point_in_polygon(&Point2::new(0.5, 0.5), &square));
        // Exterior
        assert!(!point_in_polygon(&Point2::new(2.0, 0.5), &square));
        assert!(!point_in_polygon(&Point2::new(-0.1, 0.5), &square));
        assert!(!point_in_polygon(&Point2::new(0.5, -0.1), &square));
    }

    #[test]
    fn test_point_in_polygon_concave() {
        // L-shape: concave polygon
        let l_shape = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 2.0),
            Point2::new(0.0, 2.0),
        ];
        // Inside the L
        assert!(point_in_polygon(&Point2::new(0.5, 0.5), &l_shape));
        assert!(point_in_polygon(&Point2::new(0.5, 1.5), &l_shape));
        // Inside bounding box but outside the L (the concave notch)
        assert!(!point_in_polygon(&Point2::new(1.5, 1.5), &l_shape));
        // Fully outside
        assert!(!point_in_polygon(&Point2::new(3.0, 0.5), &l_shape));
    }

    #[test]
    fn test_point_in_polygon_degenerate() {
        // Fewer than 3 points
        let line = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        assert!(!point_in_polygon(&Point2::new(0.5, 0.0), &line));

        let empty: Vec<Point2<f64>> = vec![];
        assert!(!point_in_polygon(&Point2::new(0.0, 0.0), &empty));
    }

    #[test]
    fn test_closest_polygon_edge_square() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];

        // Point below bottom edge — closest to edge 0 (vertex 0→1)
        let (idx, _dist_sq) = closest_polygon_edge(&Point2::new(0.5, -1.0), &square);
        assert_eq!(idx, 0);

        // Point to the right of right edge — closest to edge 1 (vertex 1→2)
        let (idx, _dist_sq) = closest_polygon_edge(&Point2::new(2.0, 0.5), &square);
        assert_eq!(idx, 1);

        // Point above top edge — closest to edge 2 (vertex 2→3)
        let (idx, _dist_sq) = closest_polygon_edge(&Point2::new(0.5, 2.0), &square);
        assert_eq!(idx, 2);

        // Point to the left — closest to edge 3 (vertex 3→0)
        let (idx, _dist_sq) = closest_polygon_edge(&Point2::new(-1.0, 0.5), &square);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_closest_polygon_edge_on_vertex() {
        let triangle = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];

        // Point exactly at vertex 0 — distance should be ~0
        let (_idx, dist_sq) = closest_polygon_edge(&Point2::new(0.0, 0.0), &triangle);
        assert!(dist_sq < 1e-20);
    }
}
