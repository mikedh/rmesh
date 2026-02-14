//! BREP surface tessellation.
//!
//! Converts BREP faces (analytical surfaces + edge loops) into triangle meshes.
//! The [`ShellTessellator`] produces watertight output through a 3-phase pipeline:
//!
//! 1. **Edge discretization** — Each BREP edge is discretized once and assigned pool
//!    indices. Adjacent faces sharing an edge get the same vertices by construction
//!    (not by post-hoc merging), guaranteeing watertight seams.
//!    Phase 1.5 refines outer contours where inner loop vertices escape the polygon.
//! 2. **Bubble pack + CDT** — For each face: bubble packing (Shimada & Gossard)
//!    generates well-distributed interior points in UV space, then CDT triangulates
//!    the boundary + interior points. A single-pass chord error check splits any
//!    edge still exceeding tolerance.
//! 3. **Final assembly** — Per-face triangulations are assembled into a single `Trimesh`
//!    with per-vertex normals and face-to-surface grouping attributes.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::manual_midpoint)]

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};

use rayon::prelude::*;

use nalgebra::{Point2, Point3, Vector3};

use super::Surface;
use super::cdt;
use super::faces::{CURVATURE_TOL, Cone, Cylinder, GEOMETRY_TOL, Sphere, SurfacePlane, Torus};
use super::topology::{BrepEdge, BrepModel, Curve, EdgeUse, OrientedEdge};
use crate::attributes::{Attributes, Grouping, GroupingKind};
use crate::creation::{Plane, Triangulator};
use crate::mesh::Trimesh;

#[cfg(test)]
fn validation_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("RMESH_VALIDATE").is_ok())
}

/// Parameters controlling tessellation quality.
///
/// Tolerance is purely relative: the effective chord-height threshold is
/// `tolerance_relative * bounding_box_diagonal`, making tessellation quality
/// independent of model units (mm, inches, meters). A 1mm screw and a 10m
/// beam both get the same visual fidelity at the same `tolerance_relative`.
///
/// The per-face chord error pass subdivides any triangle edge whose
/// midpoint deviates from the true surface by more than the effective
/// tolerance. Boundary edges are split globally across adjacent faces to
/// maintain watertightness.
///
/// # Defaults
///
/// | Parameter | Default | Effect |
/// |-----------|---------|--------|
/// | `tolerance_relative` | 0.001 | 0.1% of bbox diagonal — visually smooth |
/// | `min_segments` | 16 | Full circles always get >= 16 segments |
/// | `max_segments` | 256 | Caps subdivision on very tight tolerances |
#[derive(Debug, Clone)]
pub struct TesselationParams {
    /// Chord error as fraction of bounding-box diagonal.
    pub tolerance_relative: f64,
    /// Minimum segments per full circle (ensures cylinders look circular).
    pub min_segments: usize,
    /// Maximum segments per curved edge.
    pub max_segments: usize,
}

impl Default for TesselationParams {
    fn default() -> Self {
        Self {
            tolerance_relative: 0.001, // 0.1% of bounding-box diagonal
            min_segments: 16,          // ensures circles always look circular
            max_segments: 256,
        }
    }
}

/// Threshold for considering a surface normal vector as near-zero (degenerate).
/// Used when normalizing normals and checking projection validity.
const NEAR_ZERO_NORMAL: f64 = 1e-12;

/// Return an ordered (min, max) edge key for use in hash/set lookups.
#[inline]
fn canonical_edge(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// Validate a face triangulation: no degenerate triangles, correct edge
/// multiplicities, and all boundary edges present in the triangulation.
/// Returns the number of validation warnings (0 = perfect).
#[cfg(test)]
fn validate_triangulation(
    triangles: &[[usize; 3]],
    local_to_pool: &[usize],
    boundary_edges: &HashSet<(usize, usize)>,
    face_idx: usize,
    phase: &str,
    skip_self_loops: bool,
) -> usize {
    let mut warnings = 0;

    // No degenerate triangles (all 3 pool indices must be distinct)
    for tri in triangles {
        let pa = local_to_pool[tri[0]];
        let pb = local_to_pool[tri[1]];
        let pc = local_to_pool[tri[2]];
        if pa == pb || pb == pc || pa == pc {
            warnings += 1;
        }
    }

    // Edge multiplicity checks
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
    for tri in triangles {
        for &[a, b] in &[[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]] {
            let key = canonical_edge(local_to_pool[a], local_to_pool[b]);
            *edge_count.entry(key).or_default() += 1;
        }
    }

    let pool_boundary: HashSet<(usize, usize)> = boundary_edges
        .iter()
        .map(|&(a, b)| canonical_edge(local_to_pool[a], local_to_pool[b]))
        .collect();

    for (&key, &count) in &edge_count {
        let is_boundary = pool_boundary.contains(&key);
        if is_boundary {
            if count != 1 {
                warnings += 1;
            }
        } else if count != 2 {
            warnings += 1;
        }
    }

    // All boundary edges must appear in the triangulation
    for &pool_edge in &pool_boundary {
        if skip_self_loops && pool_edge.0 == pool_edge.1 {
            continue;
        }
        if !edge_count.contains_key(&pool_edge) {
            warnings += 1;
        }
    }

    warnings
}

/// Build the set of expected contour edges from closed contour index lists.
fn contour_edge_set(contours: &[Vec<usize>]) -> HashSet<(usize, usize)> {
    let cap: usize = contours.iter().map(|c| c.len().saturating_sub(1)).sum();
    let mut expected = HashSet::with_capacity(cap);
    for contour in contours {
        for w in contour.windows(2) {
            expected.insert(canonical_edge(w[0], w[1]));
        }
    }
    expected
}

/// Build edge-count map from triangles (how many times each edge appears).
fn build_edge_counts(triangles: &[[usize; 3]]) -> HashMap<(usize, usize), usize> {
    let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
    for tri in triangles {
        *counts.entry(canonical_edge(tri[0], tri[1])).or_default() += 1;
        *counts.entry(canonical_edge(tri[1], tri[2])).or_default() += 1;
        *counts.entry(canonical_edge(tri[2], tri[0])).or_default() += 1;
    }
    counts
}

/// Check that all expected contour edges appear in the triangulation exactly once.
///
/// A boundary edge should appear in exactly 1 triangle (it's on the boundary).
/// If a boundary edge appears in 2+ triangles, the CDT has over-counted it, which
/// means the triangulation is incorrect even though the edge is "present".
fn contours_complete_with(triangles: &[[usize; 3]], expected: &HashSet<(usize, usize)>) -> bool {
    let counts = build_edge_counts(triangles);
    expected
        .iter()
        .all(|e| counts.get(e).copied().unwrap_or(0) == 1)
}

/// Count how many expected contour edges appear in the triangulation exactly once.
fn count_boundary_edges(triangles: &[[usize; 3]], expected: &HashSet<(usize, usize)>) -> usize {
    let counts = build_edge_counts(triangles);
    expected
        .iter()
        .filter(|e| counts.get(e).copied().unwrap_or(0) == 1)
        .count()
}

/// Fan triangulation: connect every contour edge to a central vertex.
/// Works for any contour shape but produces poor element quality.
/// Used as a last resort when CDT fails in both UV and 3D.
fn fan_triangulate(contours: &[Vec<usize>]) -> Vec<[usize; 3]> {
    let mut triangles = Vec::new();
    // Use vertex 0 as the fan center for the outer contour
    if let Some(outer) = contours.first()
        && outer.len() >= 4
    {
        // outer contour is [0, 1, 2, ..., n-1, 0], fan from vertex 0
        for w in outer.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a == 0 || b == 0 {
                continue; // skip edges incident on the fan center
            }
            triangles.push([0, a, b]);
        }
    }
    triangles
}

/// Deduplicate local vertices that share the same pool index.
///
/// Returns (deduplicated UV points, remapped contours, mapping from dedup index → original local index).
/// This prevents CDT from producing degenerate triangles when two local vertices
/// have the same 3D position.
#[allow(clippy::type_complexity)]
fn dedup_by_pool(
    state: &FaceTriangulation,
    contours: &[Vec<usize>],
) -> (Vec<(f64, f64)>, Vec<Vec<usize>>, Vec<usize>) {
    let n = state.local_to_pool.len();

    // Map each pool index to the first local index that uses it
    let mut pool_to_dedup: HashMap<usize, usize> = HashMap::new();
    let mut local_to_dedup: Vec<usize> = vec![0; n];
    let mut dedup_pts: Vec<(f64, f64)> = Vec::new();
    let mut dedup_to_local: Vec<usize> = Vec::new();

    for (local_idx, &pool_idx) in state.local_to_pool.iter().enumerate() {
        if let Some(&dedup_idx) = pool_to_dedup.get(&pool_idx) {
            local_to_dedup[local_idx] = dedup_idx;
        } else {
            let dedup_idx = dedup_pts.len();
            pool_to_dedup.insert(pool_idx, dedup_idx);
            local_to_dedup[local_idx] = dedup_idx;
            let uv = state.vertices_uv[local_idx];
            dedup_pts.push((uv.x, uv.y));
            dedup_to_local.push(local_idx);
        }
    }

    // Remap contours and remove consecutive duplicates
    let dedup_contours: Vec<Vec<usize>> = contours
        .iter()
        .map(|contour| {
            let mut remapped: Vec<usize> = contour.iter().map(|&idx| local_to_dedup[idx]).collect();
            // Remove consecutive duplicates
            remapped.dedup();
            // Ensure closure
            if remapped.len() > 1 && remapped.first() != remapped.last() {
                remapped.push(remapped[0]);
            }
            remapped
        })
        .collect();

    (dedup_pts, dedup_contours, dedup_to_local)
}

/// State for a face during the global refinement process.
///
/// Holds all the triangulation state needed to track which vertices belong to this face,
/// how they map to the global vertex pool, and which edges are on the BREP boundary.
#[derive(Debug, Clone, Default)]
struct FaceTriangulation {
    /// UV coordinates of all vertices in this face's local space
    vertices_uv: Vec<Point2<f64>>,
    /// Triangles using local vertex indices
    triangles: Vec<[usize; 3]>,
    /// Maps local vertex index -> pool (global) vertex index
    local_to_pool: Vec<usize>,
    /// Set of local edge pairs that lie on BREP boundaries (edges shared between faces)
    /// Stored as (min_local_idx, max_local_idx) for canonical ordering
    boundary_edges: HashSet<(usize, usize)>,
    /// Edge-to-triangle adjacency: canonical edge (min,max) → triangle indices.
    /// Built at Phase 3 start and maintained incrementally during splits.
    edge_tris: HashMap<(usize, usize), Vec<usize>>,
    /// Which triangulation strategy produced the result.
    /// 0=uv_cdt, 1=plane_cdt, 2=earcut, 3=best_incomplete, 4=fan
    #[cfg(test)]
    diag_strategy: u8,
}

impl FaceTriangulation {
    /// Add a vertex to this face's local coordinate system.
    /// Returns the local index.
    fn add_vertex(&mut self, uv: Point2<f64>, pool_idx: usize) -> usize {
        let local_idx = self.vertices_uv.len();
        self.vertices_uv.push(uv);
        self.local_to_pool.push(pool_idx);
        local_idx
    }

    /// Mark an edge as being on the BREP boundary.
    fn mark_boundary_edge(&mut self, local_a: usize, local_b: usize) {
        self.boundary_edges.insert(canonical_edge(local_a, local_b));
    }

    /// Check if an edge is on the BREP boundary.
    fn is_boundary_edge(&self, local_a: usize, local_b: usize) -> bool {
        self.boundary_edges
            .contains(&canonical_edge(local_a, local_b))
    }

    /// Build the edge→triangle adjacency map from current triangles.
    fn build_edge_tris(&mut self) {
        self.edge_tris.clear();
        for (tri_idx, tri) in self.triangles.iter().enumerate() {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                let key = canonical_edge(a, b);
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
        let key = canonical_edge(local_a, local_b);
        let tri_indices = match self.edge_tris.remove(&key) {
            Some(v) if !v.is_empty() => v,
            _ => return false,
        };

        for &tri_idx in &tri_indices {
            let tri = self.triangles[tri_idx];
            // Find the opposite vertex
            let v_opp = *tri.iter().find(|&&v| v != local_a && v != local_b).unwrap();

            // Split: replace original with [local_a, local_mid, v_opp],
            //         append [local_mid, local_b, v_opp]
            let half_a = [local_a, local_mid, v_opp];
            let half_b = [local_mid, local_b, v_opp];

            // Remove old triangle's edges from the map
            for i in 0..3 {
                let ea = tri[i];
                let eb = tri[(i + 1) % 3];
                let ekey = canonical_edge(ea, eb);
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
                    let ekey = canonical_edge(ea, eb);
                    self.edge_tris.entry(ekey).or_default().push(idx);
                }
            }
        }

        true
    }

    /// Refine interior edges whose chord error exceeds tolerance.
    fn refine_chord_errors(
        &mut self,
        surface: &Surface,
        pool_vertices: &[Point3<f64>],
        new_vertices: &mut Vec<Point3<f64>>,
        sentinel_base: usize,
        tolerance: f64,
    ) {
        const MAX_PASS_ITERS: usize = 2;

        self.build_edge_tris();

        let mut edges_to_split: HashMap<(usize, usize), (Point2<f64>, Point3<f64>)> =
            HashMap::new();

        for _ in 0..MAX_PASS_ITERS {
            edges_to_split.clear();

            for tri in &self.triangles {
                let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];

                for (la, lb) in edges {
                    if self.is_boundary_edge(la, lb) {
                        continue;
                    }

                    let key = canonical_edge(la, lb);
                    if edges_to_split.contains_key(&key) {
                        continue;
                    }

                    let uv_a = self.vertices_uv[la];
                    let uv_b = self.vertices_uv[lb];

                    let (uv_mid, p_mid_surface) = surface.midpoint_uv(&uv_a, &uv_b);

                    let pool_a = self.local_to_pool[la];
                    let pool_b = self.local_to_pool[lb];
                    let p_a = lookup_vertex(pool_a, pool_vertices, new_vertices, sentinel_base);
                    let p_b = lookup_vertex(pool_b, pool_vertices, new_vertices, sentinel_base);
                    let p_mid_linear = p_a.lerp(&p_b, 0.5);

                    let error = (p_mid_surface - p_mid_linear).norm();
                    if error > tolerance {
                        edges_to_split.insert(key, (uv_mid, p_mid_surface));
                    }
                }
            }

            if edges_to_split.is_empty() {
                break;
            }

            for ((la, lb), (uv_mid, p_mid)) in &edges_to_split {
                let pool_mid = sentinel_base + new_vertices.len();
                new_vertices.push(*p_mid);
                let local_mid = self.add_vertex(*uv_mid, pool_mid);
                let _ = self.split_triangles_at_edge(*la, *lb, local_mid);
            }
        }
    }
}

// ============================================================================
// Surface parametric evaluation
// ============================================================================

impl SurfacePlane {
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
            Surface::Offset(o) => o.to_parametric(point),
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
            Surface::Offset(o) => o.evaluate(u, v),
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
            Surface::Offset(o) => o.normal_at(u, v),
        }
    }
}

// ============================================================================
// 2D geometry utilities
// ============================================================================

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
        let t = if len_sq > 1e-30 {
            ap.dot(&ab) / len_sq
        } else {
            0.0
        };
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

impl Surface {
    /// Unwrap UV coordinates to remove angular discontinuities at ±π.
    /// Dispatches to the appropriate unwrapping function based on surface type.
    fn unwrap_uvs(&self, uvs: &mut [Point2<f64>]) {
        if self.is_doubly_angular() {
            unwrap_angular_coords(uvs, true);
        } else if self.is_angular() {
            unwrap_angular_coords(uvs, false);
        }
    }

    /// Compute the UV midpoint of an edge, handling angular surfaces correctly.
    /// For angular surfaces, averages in 3D then projects back to UV.
    /// Returns (uv_mid, p_mid_on_surface).
    fn midpoint_uv(&self, uv_a: &Point2<f64>, uv_b: &Point2<f64>) -> (Point2<f64>, Point3<f64>) {
        if self.is_angular() {
            let p_a = self.evaluate(uv_a.x, uv_a.y);
            let p_b = self.evaluate(uv_b.x, uv_b.y);
            let p_mid_3d = p_a.lerp(&p_b, 0.5);
            let uv_mid = self.to_parametric(&p_mid_3d);
            let p_on_surface = self.evaluate(uv_mid.x, uv_mid.y);
            (uv_mid, p_on_surface)
        } else {
            let uv_mid = uv_a.lerp(uv_b, 0.5);
            let p_mid = self.evaluate(uv_mid.x, uv_mid.y);
            (uv_mid, p_mid)
        }
    }
}

fn unwrap_angular_coords(uvs: &mut [Point2<f64>], both: bool) {
    if uvs.len() < 2 {
        return;
    }

    let mut buf: Vec<f64> = vec![0.0; uvs.len()];
    unwrap_angular_sequence(uvs.iter().map(|p| p.x), &mut buf);
    for (i, uv) in uvs.iter_mut().enumerate() {
        uv.x = buf[i];
    }
    if both {
        unwrap_angular_sequence(uvs.iter().map(|p| p.y), &mut buf);
        for (i, uv) in uvs.iter_mut().enumerate() {
            uv.y = buf[i];
        }
    }
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

// ============================================================================
// Free functions for parallel Phase 2
// ============================================================================

/// Resolve a vertex index to its 3D position, dispatching between
/// the global pool and a thread-local buffer for newly created vertices.
#[inline]
fn lookup_vertex(
    idx: usize,
    pool: &[Point3<f64>],
    local: &[Point3<f64>],
    base: usize,
) -> Point3<f64> {
    if idx >= base {
        local[idx - base]
    } else {
        pool[idx]
    }
}

/// Get ALL pool indices for an edge, properly ordered for the face.
/// Returns the COMPLETE sequence including both endpoints.
fn get_edge_pool_indices<'a>(
    edge_discretization: &'a HashMap<usize, EdgeDiscretization>,
    oe: &OrientedEdge,
) -> Cow<'a, [usize]> {
    let disc = edge_discretization
        .get(&oe.edge)
        .expect("edge not pre-discretized");

    if oe.same_sense {
        Cow::Borrowed(&disc.pool_indices)
    } else {
        Cow::Owned(disc.pool_indices.iter().rev().copied().collect())
    }
}

/// Collect deduplicated pool indices for a loop's edges.
/// Skips consecutive duplicates from edge chaining and removes the
/// closing duplicate if the loop is closed.
fn collect_loop_indices(
    edge_discretization: &HashMap<usize, EdgeDiscretization>,
    loop_: &super::topology::BrepLoop,
) -> Vec<usize> {
    let mut indices = Vec::new();
    for oe in &loop_.edges {
        for &idx in get_edge_pool_indices(edge_discretization, oe).as_ref() {
            if indices.last() != Some(&idx) {
                indices.push(idx);
            }
        }
    }
    if indices.len() > 1 && indices.first() == indices.last() {
        indices.pop();
    }
    indices
}

/// Triangulate a face with a multi-strategy fallback chain.
/// Returns (triangles, strategy_code) where strategy codes are:
/// 0=uv_cdt, 1=plane_cdt, 2=earcut_uv, 3=best_incomplete, 4=fan,
/// 5=earcut_3d, 6=multi_axis_cdt
fn triangulate_face_robust_pts(
    pts: &[(f64, f64)],
    contours: &[Vec<usize>],
    state: &FaceTriangulation,
    pool_vertices: &[Point3<f64>],
    new_vertices: &[Point3<f64>],
    sentinel_base: usize,
) -> (Vec<[usize; 3]>, u8, bool) {
    // Build contour edge set once — invariant across all CDT attempts
    let expected = contour_edge_set(contours);

    // Track the best incomplete result (most boundary edges preserved)
    let mut best_incomplete: Option<(Vec<[usize; 3]>, usize, u8)> = None;
    let mut track_best =
        |result: &[[usize; 3]], strategy: u8, expected: &HashSet<(usize, usize)>| {
            if result.is_empty() {
                return;
            }
            let count = count_boundary_edges(result, expected);
            if best_incomplete
                .as_ref()
                .is_none_or(|(_, best_count, _)| count > *best_count)
            {
                best_incomplete = Some((result.to_vec(), count, strategy));
            }
        };

    // Track whether CDT diverged — if so, all projections will hit the same
    // O(n²) behavior (same constraint topology), so skip remaining CDT attempts.
    let mut cdt_diverged = false;

    /// Check if a CDT error is a divergence error (Diverged or TimeBudgetExceeded).
    fn is_diverged(e: &cdt::Error) -> bool {
        matches!(e, cdt::Error::Diverged | cdt::Error::TimeBudgetExceeded)
    }

    // Try 1: CDT in UV space
    match cdt::triangulate_contours(pts, contours) {
        Ok(tris) => {
            let result: Vec<[usize; 3]> = tris.iter().map(|&(a, b, c)| [a, b, c]).collect();
            if contours_complete_with(&result, &expected) {
                return (result, 0, false);
            }
            track_best(&result, 0, &expected);
        }
        Err(e) if is_diverged(&e) => {
            cdt_diverged = true;
        }
        Err(_) => {}
    }

    // Build 3D positions once for plane-based fallbacks
    let positions: Vec<Point3<f64>> = state
        .local_to_pool
        .iter()
        .map(|&pool_idx| lookup_vertex(pool_idx, pool_vertices, new_vertices, sentinel_base))
        .collect();

    // Try 2: CDT in best-fit 3D plane projection (skip if CDT diverged)
    if !cdt_diverged && positions.len() >= 3 {
        if let Ok(plane) = Plane::from_points(&positions, true) {
            let pts_2d: Vec<(f64, f64)> = plane
                .to_2d(&positions)
                .into_iter()
                .map(|p| (p.x, p.y))
                .collect();

            match cdt::triangulate_contours(&pts_2d, contours) {
                Ok(tris) => {
                    let result: Vec<[usize; 3]> = tris.iter().map(|&(a, b, c)| [a, b, c]).collect();
                    if contours_complete_with(&result, &expected) {
                        return (result, 1, false);
                    }
                    track_best(&result, 1, &expected);
                }
                Err(e) if is_diverged(&e) => {
                    cdt_diverged = true;
                }
                Err(_) => {}
            }
        }
    }

    // Try 2b: CDT in axis-aligned plane projections (XY, XZ, YZ) — skip if CDT diverged
    if !cdt_diverged && positions.len() >= 3 {
        let projections: [(fn(&Point3<f64>) -> (f64, f64), &str); 3] = [
            (|p| (p.x, p.y), "XY"),
            (|p| (p.x, p.z), "XZ"),
            (|p| (p.y, p.z), "YZ"),
        ];
        for (proj_fn, _) in &projections {
            let pts_2d: Vec<(f64, f64)> = positions.iter().map(proj_fn).collect();
            match cdt::triangulate_contours(&pts_2d, contours) {
                Ok(tris) => {
                    let result: Vec<[usize; 3]> = tris.iter().map(|&(a, b, c)| [a, b, c]).collect();
                    if contours_complete_with(&result, &expected) {
                        return (result, 6, false);
                    }
                    track_best(&result, 6, &expected);
                }
                Err(e) if is_diverged(&e) => {
                    cdt_diverged = true;
                    break;
                }
                Err(_) => {}
            }
        }
    }

    // Try 3: Earcut in UV space
    {
        let exterior: Vec<usize> = contours[0][..contours[0].len() - 1].to_vec();
        let interiors: Vec<Vec<usize>> = contours[1..]
            .iter()
            .map(|c| c[..c.len() - 1].to_vec())
            .collect();
        let vertices: Vec<Point2<f64>> = pts.iter().map(|&(x, y)| Point2::new(x, y)).collect();
        let mut tri = Triangulator::new();
        let result = tri.triangulate_2d(&exterior, &interiors, &vertices, false);
        if !result.is_empty() && contours_complete_with(&result, &expected) {
            return (result, 2, cdt_diverged);
        }
        track_best(&result, 2, &expected);
    }

    // Try 3b: Earcut in 3D plane projection
    if positions.len() >= 3 {
        let exterior: Vec<usize> = contours[0][..contours[0].len() - 1].to_vec();
        let interiors: Vec<Vec<usize>> = contours[1..]
            .iter()
            .map(|c| c[..c.len() - 1].to_vec())
            .collect();
        let mut tri = Triangulator::new();
        if let Ok(result) = tri.triangulate_3d(&exterior, &interiors, &positions, false, false) {
            if !result.is_empty() && contours_complete_with(&result, &expected) {
                return (result, 5, cdt_diverged);
            }
            track_best(&result, 5, &expected);
        }
    }

    // Try 4: Fan triangulation
    let fan = fan_triangulate(contours);
    let fan_count = count_boundary_edges(&fan, &expected);

    // Return best incomplete if it preserves more boundary edges than fan
    if let Some((best_tris, best_count, _)) = best_incomplete {
        if best_count > fan_count {
            return (best_tris, 3, cdt_diverged);
        }
    }

    (fan, 4, cdt_diverged)
}

/// Tessellate a single face independently. New vertices are stored in a local
/// buffer with sentinel pool indices (`sentinel_base + i`), to be rewritten
/// during the serial merge phase.
fn tessellate_face(
    face_idx: usize,
    model: &BrepModel,
    pool_vertices: &[Point3<f64>],
    edge_discretization: &HashMap<usize, EdgeDiscretization>,
    brep_vertex_to_pool: &HashMap<usize, usize>,
    effective_tolerance: f64,
    sentinel_base: usize,
) -> (FaceTriangulation, Vec<Point3<f64>>) {
    let face = &model.faces[face_idx];
    let surface = &model.face_surfaces[face.surface];

    let mut state = FaceTriangulation::default();
    let mut new_vertices: Vec<Point3<f64>> = Vec::new();

    // Determine which loop is the real outer contour (may swap with inner if outer is a vertex loop)
    let outer_loop_data = &model.loops[face.outer_loop];
    let outer_is_vertex_loop = outer_loop_data.vertex.is_some();

    // If the outer loop is a vertex loop, find the first inner edge loop to use as outer contour
    let (effective_outer_idx, vertex_loop_vertex) = if outer_is_vertex_loop {
        let pole_vertex = outer_loop_data.vertex.unwrap();
        // Find first inner loop that has edges
        let mut found_inner = None;
        for (i, &il_idx) in face.inner_loops.iter().enumerate() {
            if model.loops[il_idx].vertex.is_none() && !model.loops[il_idx].edges.is_empty() {
                found_inner = Some((i, il_idx));
                break;
            }
        }
        if let Some((_, il_idx)) = found_inner {
            (il_idx, Some(pole_vertex))
        } else {
            // No edge loops at all — degenerate face, return empty
            return (state, new_vertices);
        }
    } else {
        (face.outer_loop, None)
    };

    let effective_outer_loop = &model.loops[effective_outer_idx];

    // 1. Collect boundary vertices
    let outer_pool_indices = collect_loop_indices(edge_discretization, effective_outer_loop);

    let mut raw_uvs: Vec<Point2<f64>> = outer_pool_indices
        .iter()
        .map(|&pool_idx| surface.to_parametric(&pool_vertices[pool_idx]))
        .collect();
    surface.unwrap_uvs(&mut raw_uvs);

    for (i, &pool_idx) in outer_pool_indices.iter().enumerate() {
        state.add_vertex(raw_uvs[i], pool_idx);
    }

    let outer_len = outer_pool_indices.len();
    for i in 0..outer_len {
        state.mark_boundary_edge(i, (i + 1) % outer_len);
    }

    // Collect inner loop (hole) vertices — skip vertex loops and the swapped-in outer
    let mut hole_info: Vec<(usize, usize)> = Vec::new();
    for &inner_loop_idx in &face.inner_loops {
        // Skip this loop if it was promoted to outer
        if inner_loop_idx == effective_outer_idx {
            continue;
        }
        let inner_loop = &model.loops[inner_loop_idx];
        // Skip vertex loops (degenerate, zero-area — no geometric effect as holes)
        if inner_loop.vertex.is_some() {
            continue;
        }
        let loop_indices = collect_loop_indices(edge_discretization, inner_loop);

        let hole_start = state.vertices_uv.len();
        let hole_len = loop_indices.len();
        hole_info.push((hole_start, hole_len));

        let mut inner_uvs: Vec<Point2<f64>> = loop_indices
            .iter()
            .map(|&pool_idx| surface.to_parametric(&pool_vertices[pool_idx]))
            .collect();
        surface.unwrap_uvs(&mut inner_uvs);

        for (i, &pool_idx) in loop_indices.iter().enumerate() {
            state.add_vertex(inner_uvs[i], pool_idx);
        }

        for i in 0..hole_len {
            state.mark_boundary_edge(hole_start + i, hole_start + (i + 1) % hole_len);
        }
    }

    // Build contours for CDT
    let mut outer_contour: Vec<usize> = (0..outer_len).collect();
    outer_contour.push(0);

    let mut contours: Vec<Vec<usize>> = vec![outer_contour];
    for &(hole_start, hole_len) in &hole_info {
        let mut inner_contour: Vec<usize> = (hole_start..hole_start + hole_len).collect();
        inner_contour.push(hole_start);
        contours.push(inner_contour);
    }

    // 1.5. Add pole vertex from vertex loop as an interior CDT point
    if let Some(pole_brep_idx) = vertex_loop_vertex {
        if let Some(&pole_pool_idx) = brep_vertex_to_pool.get(&pole_brep_idx) {
            let pole_point = &pool_vertices[pole_pool_idx];
            let pole_uv = surface.to_parametric(pole_point);
            state.add_vertex(pole_uv, pole_pool_idx);
        }
    }

    // Record boundary vertex count before adding interior points
    let n_boundary = state.vertices_uv.len();
    let n_new_verts_before = new_vertices.len();

    // 2. Bubble packing for non-planar faces
    if !surface.is_planar() {
        let hole_uv_slices: Vec<&[Point2<f64>]> = hole_info
            .iter()
            .map(|&(start, len)| &state.vertices_uv[start..start + len])
            .collect();

        let interior_uvs = super::hex_grid::generate_interior_points(
            surface,
            &state.vertices_uv[..outer_len],
            &hole_uv_slices,
            effective_tolerance,
        );

        for uv in &interior_uvs {
            let p_3d = surface.evaluate(uv.x, uv.y);
            let pool_idx = sentinel_base + new_vertices.len();
            new_vertices.push(p_3d);
            state.add_vertex(*uv, pool_idx);
        }
    }

    // 2.5. Deduplicate vertices that share the same pool index
    let (dedup_pts, dedup_contours, dedup_map) = dedup_by_pool(&state, &contours);

    // 3. CDT triangulation
    let (dedup_tris, _strategy, cdt_diverged) = triangulate_face_robust_pts(
        &dedup_pts,
        &dedup_contours,
        &state,
        pool_vertices,
        &new_vertices,
        sentinel_base,
    );

    // Map dedup triangles back to original local indices
    let mut mapped_tris: Vec<[usize; 3]> = dedup_tris
        .into_iter()
        .map(|[a, b, c]| [dedup_map[a], dedup_map[b], dedup_map[c]])
        .collect();

    // 3.5. If CDT with hex-grid interior points failed boundary check, retry without them.
    // Skip this retry if CDT diverged — same constraints will hit the same O(n²) behavior.
    let expected_edges = contour_edge_set(&contours);
    let mut final_strategy = _strategy;
    if !cdt_diverged
        && !contours_complete_with(&mapped_tris, &expected_edges)
        && n_boundary < state.vertices_uv.len()
    {
        // Truncate back to boundary-only vertices (remove hex-grid interior points)
        state.vertices_uv.truncate(n_boundary);
        state.local_to_pool.truncate(n_boundary);
        new_vertices.truncate(n_new_verts_before);

        let (dedup_pts2, dedup_contours2, dedup_map2) = dedup_by_pool(&state, &contours);
        let (dedup_tris2, strategy2, _) = triangulate_face_robust_pts(
            &dedup_pts2,
            &dedup_contours2,
            &state,
            pool_vertices,
            &new_vertices,
            sentinel_base,
        );
        mapped_tris = dedup_tris2
            .into_iter()
            .map(|[a, b, c]| [dedup_map2[a], dedup_map2[b], dedup_map2[c]])
            .collect();
        final_strategy = strategy2;
    }

    #[cfg(test)]
    {
        state.diag_strategy = final_strategy;
    }
    let _ = final_strategy;

    state.triangles = mapped_tris;

    // CDT validation (test-only, warnings instead of panics)
    #[cfg(test)]
    if validation_enabled() {
        let w = validate_triangulation(
            &state.triangles,
            &state.local_to_pool,
            &state.boundary_edges,
            face_idx,
            "CDT",
            false,
        );
        if w > 0 && w < 100 {
            eprintln!("face {face_idx}: {w} CDT validation warnings");
        }
    }

    // 4. Chord error pass
    if !surface.is_planar() && !state.triangles.is_empty() {
        state.refine_chord_errors(
            surface,
            pool_vertices,
            &mut new_vertices,
            sentinel_base,
            effective_tolerance,
        );
    }

    // Post-chord-refinement validation (test-only, warnings instead of panics)
    #[cfg(test)]
    if validation_enabled() {
        let w = validate_triangulation(
            &state.triangles,
            &state.local_to_pool,
            &state.boundary_edges,
            face_idx,
            "POST-CHORD",
            true,
        );
        if w > 0 && w < 100 {
            eprintln!("face {face_idx}: {w} post-chord validation warnings");
        }
    }

    (state, new_vertices)
}

/// Tessellator that ensures watertight output through edge-centric refinement.
///
/// The key insight: edges are discretized ONCE with pool indices assigned immediately.
/// Adjacent faces sharing an edge get the SAME pool indices by lookup, not by matching.
///
/// Phase 1: Global edge discretization + contour refinement
/// Phase 2: Bubble pack + CDT per face (with inline chord error pass)
/// Phase 3: Final assembly (normals, Trimesh construction)
struct ShellTessellator<'a> {
    model: &'a BrepModel,
    params: &'a TesselationParams,
    /// Effective tolerance in model units (tolerance_relative * bounding-box diagonal).
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

    // Phase 2 results
    /// Per-face triangulation state
    face_states: Vec<FaceTriangulation>,

    // Phase 3 results
    /// Triangles output
    triangles: Vec<[usize; 3]>,
    /// Normals at each vertex
    normals: Vec<Vector3<f64>>,
    /// Face index for each triangle
    face_indices: Vec<usize>,
}

impl<'a> ShellTessellator<'a> {
    fn new(model: &'a BrepModel, params: &'a TesselationParams) -> Self {
        let char_length = model.characteristic_length();
        let effective_tolerance = if char_length > f64::EPSILON {
            params.tolerance_relative * char_length
        } else {
            params.tolerance_relative // unit-scale fallback for degenerate models
        };

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

            for face in self.model.faces.iter() {
                if face.inner_loops.is_empty() {
                    continue;
                }

                let surface = &self.model.face_surfaces[face.surface];
                let outer_loop = &self.model.loops[face.outer_loop];

                let outer_pool_indices =
                    collect_loop_indices(&self.edge_discretization, outer_loop);

                if outer_pool_indices.len() < 3 {
                    continue;
                }

                // Compute outer UV coordinates
                let mut outer_uvs: Vec<Point2<f64>> = outer_pool_indices
                    .iter()
                    .map(|&pi| surface.to_parametric(&self.vertices[pi]))
                    .collect();
                surface.unwrap_uvs(&mut outer_uvs);

                // Collect all inner loop UV coordinates
                for &inner_loop_idx in &face.inner_loops {
                    let inner_loop = &self.model.loops[inner_loop_idx];
                    let inner_pool_indices =
                        collect_loop_indices(&self.edge_discretization, inner_loop);

                    let mut inner_uvs: Vec<Point2<f64>> = inner_pool_indices
                        .iter()
                        .map(|&pi| surface.to_parametric(&self.vertices[pi]))
                        .collect();
                    surface.unwrap_uvs(&mut inner_uvs);

                    // Check each inner vertex against the outer polygon
                    let inside =
                        super::polygon_query::point_in_polygon(&outer_uvs, &[], &inner_uvs);
                    for (i, inner_uv) in inner_uvs.iter().enumerate() {
                        if !inside[i] {
                            // Find closest outer edge
                            let (edge_local_idx, _dist_sq) =
                                closest_polygon_edge(inner_uv, &outer_uvs);
                            let next_idx = (edge_local_idx + 1) % outer_pool_indices.len();
                            let pool_a = outer_pool_indices[edge_local_idx];
                            let pool_b = outer_pool_indices[next_idx];
                            let key = canonical_edge(pool_a, pool_b);
                            edges_to_refine.insert(key);
                        }
                    }
                }

                // (per-face message suppressed to reduce noise in corpus tests)
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
                let key_a = canonical_edge(*pool_a, pool_mid);
                let key_b = (pool_mid.min(*pool_b), pool_mid.max(*pool_b));
                self.pool_edge_to_brep.insert(key_a, brep_edge_idx);
                self.pool_edge_to_brep.insert(key_b, brep_edge_idx);
            }
        }
    }

    // =========================================================================
    // Phase 3: Final Assembly
    // =========================================================================

    /// Phase 3: Convert face states to final triangles and compute normals.
    fn phase3_final_assembly(&mut self) {
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
                // Skip degenerate triangles (can arise from CDT fallback)
                if pool_tri[0] == pool_tri[1]
                    || pool_tri[1] == pool_tri[2]
                    || pool_tri[0] == pool_tri[2]
                {
                    continue;
                }
                self.triangles.push(pool_tri);
                self.face_indices.push(face_idx);
            }
        }

        // Normalize accumulated normals
        for n in &mut self.normals {
            n.try_normalize_mut(NEAR_ZERO_NORMAL);
        }
    }

    /// Tessellate all faces and return a Trimesh.
    fn tessellate(mut self) -> Trimesh {
        let _t1 = std::time::Instant::now();

        // Phase 1: Discretize all edges globally
        self.phase1_discretize_all_edges();

        // Phase 1 validation (test-only)
        #[cfg(test)]
        if validation_enabled() {
            // Every BREP edge shared by 2 faces: both faces reference it
            for (&edge_idx, disc) in &self.edge_discretization {
                assert!(
                    disc.pool_indices.len() >= 2,
                    "edge {edge_idx}: discretization has {} vertices (need >= 2)",
                    disc.pool_indices.len()
                );
                // Check for duplicate consecutive pool indices
                for w in disc.pool_indices.windows(2) {
                    assert!(
                        w[0] != w[1],
                        "edge {edge_idx}: consecutive duplicate pool index {}",
                        w[0]
                    );
                }
            }

            // Every shared BREP edge should have at most 2 face uses (manifold)
            for (&edge_idx, uses) in &self.edge_adjacency {
                if uses.len() > 2 {
                    eprintln!(
                        "    WARN: edge {edge_idx}: has {} face uses (max 2 for manifold)",
                        uses.len()
                    );
                }
            }

            // collect_loop_indices must produce consistent loops
            for (face_idx, face) in self.model.faces.iter().enumerate() {
                let outer_loop = &self.model.loops[face.outer_loop];
                if outer_loop.vertex.is_some() {
                    continue; // vertex loop — tessellate_face() swaps to inner edge loop
                }
                let indices = collect_loop_indices(&self.edge_discretization, outer_loop);
                assert!(
                    indices.len() >= 3,
                    "face {face_idx}: outer loop has {} indices (need >= 3)",
                    indices.len()
                );
                // No consecutive duplicate pool indices
                for w in indices.windows(2) {
                    assert!(
                        w[0] != w[1],
                        "face {face_idx}: outer loop consecutive dup pool {}",
                        w[0]
                    );
                }
                // First != last (not closed)
                assert!(
                    indices.first() != indices.last(),
                    "face {face_idx}: outer loop is still closed after dedup ({} == {})",
                    indices.first().unwrap(),
                    indices.last().unwrap()
                );
            }
        }

        // Phase 1.5: Refine outer contours where inner loops escape the outer polygon
        self.phase1_5_refine_outer_contours();
        let _t2 = std::time::Instant::now();

        // Phase 2: Bubble pack + CDT triangulation for each face (parallel)
        // Sentinel well above any real pool index; leaves room for per-face new vertices
        // without risk of overlap or overflow during the `actual_base + (idx - sentinel_base)` rewrite.
        let sentinel_base = usize::MAX / 2;
        let face_results: Vec<_> = (0..self.model.faces.len())
            .into_par_iter()
            .map(|face_idx| {
                tessellate_face(
                    face_idx,
                    self.model,
                    &self.vertices,
                    &self.edge_discretization,
                    &self.brep_vertex_to_pool,
                    self.effective_tolerance,
                    sentinel_base,
                )
            })
            .collect();

        // Serial merge: rewrite sentinel indices to real pool indices
        for (_face_idx, (mut state, new_verts)) in face_results.into_iter().enumerate() {
            let actual_base = self.vertices.len();
            for idx in &mut state.local_to_pool {
                if *idx >= sentinel_base {
                    *idx = actual_base + (*idx - sentinel_base);
                }
            }

            // Post-merge validation: all pool indices should be valid
            #[cfg(test)]
            if validation_enabled() {
                let new_pool_size = self.vertices.len() + new_verts.len();
                for (local_idx, &pool_idx) in state.local_to_pool.iter().enumerate() {
                    assert!(
                        pool_idx < new_pool_size,
                        "face {_face_idx}: local {local_idx} has pool {pool_idx} >= pool size {new_pool_size}"
                    );
                }
                // Check no sentinel indices remain
                for &pool_idx in &state.local_to_pool {
                    assert!(
                        pool_idx < sentinel_base,
                        "face {_face_idx}: unrewritten sentinel pool index {pool_idx}"
                    );
                }
            }

            self.vertices.extend(new_verts);
            self.face_states.push(state);
        }
        // Cross-face boundary edge validation (test-only)
        #[cfg(test)]
        if validation_enabled() {
            // For each face, collect pool edge → count within that face
            let face_pool_edge_counts: Vec<HashMap<(usize, usize), usize>> = self
                .face_states
                .iter()
                .map(|state| {
                    let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
                    for tri in &state.triangles {
                        for &[a, b] in &[[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]] {
                            let key =
                                canonical_edge(state.local_to_pool[a], state.local_to_pool[b]);
                            *counts.entry(key).or_default() += 1;
                        }
                    }
                    counts
                })
                .collect();

            // For each BREP edge shared between two faces, check that all
            // pool edge segments appear in BOTH face triangulations
            let mut mismatch_count = 0usize;
            for (&edge_idx, disc) in &self.edge_discretization {
                let Some(uses) = self.edge_adjacency.get(&edge_idx) else {
                    continue;
                };
                if uses.len() != 2 {
                    continue; // skip non-manifold or boundary BREP edges
                }
                let face_a = uses[0].face_idx;
                let face_b = uses[1].face_idx;
                for w in disc.pool_indices.windows(2) {
                    let pool_edge = canonical_edge(w[0], w[1]);
                    let in_a = face_pool_edge_counts[face_a].contains_key(&pool_edge);
                    let in_b = face_pool_edge_counts[face_b].contains_key(&pool_edge);
                    if in_a != in_b {
                        mismatch_count += 1;
                    }
                }
            }
            if mismatch_count > 0 {
                eprintln!("    WARN: {mismatch_count} cross-face edge mismatches detected");
            }
        }

        // Phase 3: Final assembly
        self.phase3_final_assembly();

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

    /// Tessellate, find non-watertight faces, return a reduced BrepModel
    /// containing only the broken faces plus their edge-adjacent neighbors.
    ///
    /// Returns `None` if the mesh is already watertight.
    pub fn debug_reduce(&self, params: &TesselationParams) -> Option<BrepModel> {
        let mesh = self.tesselate(params);
        if mesh.is_watertight() {
            return None;
        }

        // Get BREP face indices that border non-manifold mesh edges
        let broken_faces = mesh.non_watertight_face_indices();
        if broken_faces.is_empty() {
            return None;
        }

        // Expand to edge-adjacent BREP faces
        let adjacency = self.build_edge_adjacency();
        let mut expanded = broken_faces.clone();
        for uses in adjacency.values() {
            let touches_broken = uses.iter().any(|u| broken_faces.contains(&u.face_idx));
            if touches_broken {
                for u in uses {
                    expanded.insert(u.face_idx);
                }
            }
        }

        let face_vec: Vec<usize> = expanded.into_iter().collect();
        Some(self.subset(&face_vec))
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
        let plane = SurfacePlane::new(Point3::new(1.0, 2.0, 3.0), Vector3::new(0.0, 0.0, 1.0));

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

        let surf_idx = model.add_surface(Surface::Plane(SurfacePlane::new(
            Point3::origin(),
            Vector3::z(),
        )));

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

        let surf_idx = model.add_surface(Surface::Plane(SurfacePlane::new(
            Point3::origin(),
            Vector3::z(),
        )));

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

        let surf_idx = model.add_surface(Surface::Plane(SurfacePlane::new(
            Point3::origin(),
            Vector3::z(),
        )));

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

        let surf_idx = model.add_surface(Surface::Plane(SurfacePlane::new(
            Point3::origin(),
            Vector3::z(),
        )));

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
        let surf_bottom = model.add_surface(Surface::Plane(SurfacePlane::new(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        )));
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
        let surf_top = model.add_surface(Surface::Plane(SurfacePlane::new(
            Point3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        )));
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
        let surf_front = model.add_surface(Surface::Plane(SurfacePlane::new(
            Point3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        )));
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
        let surf_back = model.add_surface(Surface::Plane(SurfacePlane::new(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
        )));
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
        let surf_right = model.add_surface(Surface::Plane(SurfacePlane::new(
            Point3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        )));
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
        let surf_left = model.add_surface(Surface::Plane(SurfacePlane::new(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
        )));
        model.add_face(surf_left, loop_left, vec![], true);

        model
    }

    #[test]
    fn test_watertight_cube() {
        let model = create_watertight_cube();

        // Verify the BREP model is valid
        let brep_errors = model.errors_index();
        assert!(
            brep_errors.is_empty(),
            "BREP validation failed: {:?}",
            brep_errors
        );

        // Verify edge sharing is correct for watertight mesh
        let edge_errors = model.errors_edge_sharing();
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
    fn test_surface_is_angular() {
        // Test that is_angular correctly identifies angular surfaces
        let plane = Surface::Plane(SurfacePlane::new(Point3::origin(), Vector3::z()));
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
    fn test_point_in_polygon_convex() {
        use crate::boundary::polygon_query::point_in_polygon;
        // Unit square: (0,0), (1,0), (1,1), (0,1)
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        // Interior
        assert!(point_in_polygon(&square, &[], &[Point2::new(0.5, 0.5)])[0]);
        // Exterior
        assert!(!point_in_polygon(&square, &[], &[Point2::new(2.0, 0.5)])[0]);
        assert!(!point_in_polygon(&square, &[], &[Point2::new(-0.1, 0.5)])[0]);
        assert!(!point_in_polygon(&square, &[], &[Point2::new(0.5, -0.1)])[0]);
    }

    #[test]
    fn test_point_in_polygon_concave() {
        use crate::boundary::polygon_query::point_in_polygon;
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
        assert!(point_in_polygon(&l_shape, &[], &[Point2::new(0.5, 0.5)])[0]);
        assert!(point_in_polygon(&l_shape, &[], &[Point2::new(0.5, 1.5)])[0]);
        // Inside bounding box but outside the L (the concave notch)
        assert!(!point_in_polygon(&l_shape, &[], &[Point2::new(1.5, 1.5)])[0]);
        // Fully outside
        assert!(!point_in_polygon(&l_shape, &[], &[Point2::new(3.0, 0.5)])[0]);
    }

    #[test]
    fn test_point_in_polygon_degenerate() {
        use crate::boundary::polygon_query::point_in_polygon;
        // Fewer than 3 points
        let line = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        assert!(!point_in_polygon(&line, &[], &[Point2::new(0.5, 0.0)])[0]);

        let empty: Vec<Point2<f64>> = vec![];
        assert!(!point_in_polygon(&empty, &[], &[Point2::new(0.0, 0.0)])[0]);
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

    #[test]
    fn test_watertight_regressions() {
        let dir = std::path::Path::new("/home/mikedh/dev/rmesh/feat_obj/test/regression/brep");
        if !dir.is_dir() {
            eprintln!("test/regression/brep directory not found, skipping");
            return;
        }
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("wt_regression_") && n.ends_with(".json"))
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            eprintln!("No wt_regression files found, skipping");
            return;
        }
        use crate::serialize::RmeshSerializable;
        let params = TesselationParams::default();
        let mut pass = 0;
        let mut brep_wt = 0;
        let total = paths.len();
        for path in &paths {
            let bytes = std::fs::read(path).unwrap();
            let model = BrepModel::from_bytes(&bytes).unwrap();
            let brep_watertight = model.is_watertight();
            if brep_watertight {
                brep_wt += 1;
            }
            let mesh = model.tesselate(&params);
            let name = path.file_stem().unwrap().to_string_lossy();
            if mesh.is_watertight() || !brep_watertight {
                pass += 1;
            } else {
                // Real tessellation bug: BREP is watertight but mesh is broken
                eprintln!(
                    "  TESS BUG: {} ({} faces, {} tris)",
                    name,
                    model.faces.len(),
                    mesh.faces.len(),
                );
            }
        }
        eprintln!("  watertight regressions: {pass}/{total} (brep_wt: {brep_wt}/{total})");
        assert_eq!(pass, total, "watertight regressions: {pass}/{total}");
    }
}
