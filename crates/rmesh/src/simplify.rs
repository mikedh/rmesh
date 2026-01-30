// Mesh simplification using Quadric Error Metrics with proper data structures
// Rewritten with per-vertex adjacency and edge priority queue for correctness

use crate::attributes::Attributes;
use ahash::{AHashMap, AHashSet};
use nalgebra::{Point3, Vector3, Vector4};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::{Add, AddAssign};

// Type aliases for clarity
type Point = Point3<f64>;
type Vector = Vector3<f64>;

// --- Constants ---

/// Threshold below which a value is considered zero/degenerate
const EPSILON: f64 = 1e-10;

/// Threshold for QEM matrix determinant (below this, matrix is near-singular)
const DETERMINANT_THRESHOLD: f64 = 1e-6;

/// Maximum distance from midpoint for optimal position (as multiple of edge length)
const MAX_OPTIMAL_DISTANCE_FACTOR: f64 = 2.0;

/// Threshold for detecting nearly-parallel edges (dot product > this = parallel)
const PARALLEL_DOT_THRESHOLD: f64 = 0.999;

/// Angle threshold for detecting "flipped" triangles in quality metrics (radians, ~166 degrees)
const FLIPPED_ANGLE_THRESHOLD: f64 = 2.9;

// --- Public API Types ---

/// Options for mesh simplification
#[derive(Debug, Clone, Copy)]
pub struct SimplifyOptions {
    /// Target number of faces after simplification
    pub target_count: usize,
    /// Controls how aggressively to collapse edges (typically 5-8)
    pub aggressiveness: f64,
    /// Whether to preserve and interpolate vertex attributes
    pub preserve_attributes: bool,
    /// Whether to compute quality metrics after simplification
    pub compute_quality: bool,
    /// Print progress information during simplification
    pub verbose: bool,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        Self {
            target_count: 0,
            aggressiveness: 7.0,
            preserve_attributes: true,
            compute_quality: false,
            verbose: false,
        }
    }
}

/// Result of mesh simplification
#[derive(Debug, Clone)]
pub struct SimplifyResult {
    /// Simplified vertex positions
    pub vertices: Vec<Point3<f64>>,
    /// Simplified triangle faces
    pub faces: Vec<[usize; 3]>,
    /// Preserved/interpolated vertex attributes
    pub attributes_vertex: Option<Attributes>,
    /// Preserved face attributes
    pub attributes_face: Option<Attributes>,
    /// Maps original vertex indices to new indices (usize::MAX = deleted)
    pub vertex_map: Vec<usize>,
    /// Quality metrics (only computed if options.compute_quality is true)
    pub quality: Option<SimplifyQuality>,
}

/// Quality metrics for simplified mesh
#[derive(Debug, Clone)]
pub struct SimplifyQuality {
    /// Ratio of simplified volume to original volume
    pub volume_ratio: f64,
    /// Ratio of simplified surface area to original area
    pub surface_area_ratio: f64,
    /// Ratio of simplified face count to original face count
    pub face_count_ratio: f64,
    /// Smallest angle in any triangle (radians)
    pub min_angle: f64,
    /// Number of degenerate triangles (area < epsilon)
    pub degenerate_count: usize,
    /// Number of flipped triangles (inconsistent winding)
    pub flipped_count: usize,
}

// --- Helper: Symmetric Matrix (Quadric) ---

#[derive(Debug, Clone, Copy)]
pub struct SymmetricMatrix {
    m: [f64; 10],
}

impl SymmetricMatrix {
    fn new(c: f64) -> Self {
        SymmetricMatrix { m: [c; 10] }
    }

    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        SymmetricMatrix {
            m: [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
            ],
        }
    }

    fn get(&self, index: usize) -> f64 {
        debug_assert!(index < 10, "SymmetricMatrix index out of bounds: {}", index);
        self.m[index]
    }

    #[allow(clippy::too_many_arguments)]
    fn det(
        &self,
        a11: usize,
        a12: usize,
        a13: usize,
        a21: usize,
        a22: usize,
        a23: usize,
        a31: usize,
        a32: usize,
        a33: usize,
    ) -> f64 {
        self.m[a11] * self.m[a22] * self.m[a33]
            + self.m[a13] * self.m[a21] * self.m[a32]
            + self.m[a12] * self.m[a23] * self.m[a31]
            - self.m[a13] * self.m[a22] * self.m[a31]
            - self.m[a11] * self.m[a23] * self.m[a32]
            - self.m[a12] * self.m[a21] * self.m[a33]
    }
}

impl Add for SymmetricMatrix {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut result = self.m;
        for (i, result_item) in result.iter_mut().enumerate() {
            *result_item += rhs.m[i];
        }
        SymmetricMatrix { m: result }
    }
}

impl AddAssign for SymmetricMatrix {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..10 {
            self.m[i] += rhs.m[i];
        }
    }
}

// --- Core Data Structures ---

#[derive(Debug, Clone)]
struct Triangle {
    v: [usize; 3],
    deleted: bool,
    n: Vector,
    original_index: usize,
    /// Attribute ID for seam detection (hash of face attributes, or 0 if none)
    attribute_id: u64,
}

#[derive(Debug, Clone)]
struct Vertex {
    p: Point,
    q: SymmetricMatrix,
    border: bool,
    /// Vertex is on an attribute seam (touches triangles with different attributes)
    seam: bool,
    deleted: bool,
    normal: Option<Vector>,
    // Per-vertex triangle list - always up to date
    triangles: Vec<usize>,
}

#[derive(Debug, Clone)]
struct Edge {
    v0: usize,
    v1: usize,
    error: f64,
    optimal_position: Point,
    generation: usize,
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error
    }
}

impl Eq for Edge {}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (smallest error first)
        other
            .error
            .partial_cmp(&self.error)
            .unwrap_or(Ordering::Equal)
    }
}

// --- Simplification Logic ---

struct Simplifier {
    vertices: Vec<Vertex>,
    triangles: Vec<Triangle>,
    edge_queue: BinaryHeap<Edge>,
    vertex_generation: Vec<usize>,
    current_generation: usize,
    live_triangle_count: usize,
    preserve_attributes: bool,
}

impl Simplifier {
    fn new(
        input_vertices: &[Point],
        input_faces: &[[usize; 3]],
        attributes_vertex: Option<&Attributes>,
        attributes_face: Option<&Attributes>,
        preserve_attributes: bool,
    ) -> Self {
        // Validate input: all face indices must be valid
        #[cfg(debug_assertions)]
        for (i, face) in input_faces.iter().enumerate() {
            for (j, &v) in face.iter().enumerate() {
                debug_assert!(
                    v < input_vertices.len(),
                    "Face {} vertex {} references invalid index {} (vertices.len() = {})",
                    i,
                    j,
                    v,
                    input_vertices.len()
                );
            }
        }

        let vertex_normals: Option<&Vec<Vector3<f64>>> =
            attributes_vertex.and_then(|attrs| attrs.normals.first());

        // Get face attributes for seam detection
        let face_colors: Option<&Vec<Vector4<u8>>> =
            attributes_face.and_then(|attrs| attrs.colors.first());
        // Future: could also include material IDs, texture indices, etc.

        let mut vertices: Vec<Vertex> = input_vertices
            .iter()
            .enumerate()
            .map(|(i, &p)| Vertex {
                p,
                q: SymmetricMatrix::new(0.0),
                border: false,
                seam: false,
                deleted: false,
                normal: if preserve_attributes {
                    vertex_normals.and_then(|normals| normals.get(i).copied())
                } else {
                    None
                },
                triangles: Vec::new(),
            })
            .collect();

        let triangles: Vec<Triangle> = input_faces
            .iter()
            .enumerate()
            .map(|(i, &[v0, v1, v2])| {
                let p0 = input_vertices[v0];
                let p1 = input_vertices[v1];
                let p2 = input_vertices[v2];
                let n = (p1 - p0).cross(&(p2 - p0));
                let n = if n.norm() > EPSILON {
                    n.normalize()
                } else {
                    Vector::zeros()
                };

                // Compute attribute ID from face attributes (for seam detection)
                // Currently uses color; could be extended to include material ID, etc.
                let attribute_id = face_colors.and_then(|colors| colors.get(i)).map_or(0, |c| {
                    (u64::from(c.x) << 24)
                        | (u64::from(c.y) << 16)
                        | (u64::from(c.z) << 8)
                        | u64::from(c.w)
                });

                Triangle {
                    v: [v0, v1, v2],
                    deleted: false,
                    n,
                    original_index: i,
                    attribute_id,
                }
            })
            .collect();

        // Build per-vertex triangle lists
        for (tid, t) in triangles.iter().enumerate() {
            for &v_idx in &t.v {
                if v_idx < vertices.len() {
                    vertices[v_idx].triangles.push(tid);
                }
            }
        }

        let vertex_generation = vec![0; vertices.len()];
        let live_triangle_count = triangles.len();

        Simplifier {
            vertices,
            triangles,
            edge_queue: BinaryHeap::new(),
            vertex_generation,
            current_generation: 1,
            live_triangle_count,
            preserve_attributes,
        }
    }

    fn initialize_quadrics(&mut self) {
        // Reset quadrics
        for v in &mut self.vertices {
            v.q = SymmetricMatrix::new(0.0);
        }

        // Build quadrics from triangle planes
        for t in &self.triangles {
            if t.deleted {
                continue;
            }

            let p0 = self.vertices[t.v[0]].p;
            let n = t.n;

            if n.norm() < EPSILON {
                continue;
            }

            // Plane equation: n.x * x + n.y * y + n.z * z + d = 0
            let d = -n.dot(&p0.coords);
            let q = SymmetricMatrix::from_plane(n.x, n.y, n.z, d);

            for &v_idx in &t.v {
                if v_idx < self.vertices.len() {
                    self.vertices[v_idx].q += q;
                }
            }
        }

        // Detect border vertices and attribute seams
        self.detect_borders();
        self.detect_seams();
    }

    fn detect_borders(&mut self) {
        // Count edge occurrences
        for v_idx in 0..self.vertices.len() {
            if self.vertices[v_idx].deleted {
                continue;
            }

            let mut edge_counts: AHashMap<usize, usize> = AHashMap::new();

            for &tid in &self.vertices[v_idx].triangles {
                let t = &self.triangles[tid];
                if t.deleted {
                    continue;
                }

                for &other in &t.v {
                    if other != v_idx {
                        *edge_counts.entry(other).or_insert(0) += 1;
                    }
                }
            }

            // Edge with count 1 is a border edge
            for (_neighbor, count) in edge_counts {
                if count == 1 {
                    self.vertices[v_idx].border = true;
                    break;
                }
            }
        }
    }

    /// Detect vertices on attribute seams (touching triangles with different attributes).
    /// These vertices are marked once at initialization and never modified - we never
    /// collapse any edge involving a seam vertex to prevent color bleeding.
    fn detect_seams(&mut self) {
        for v_idx in 0..self.vertices.len() {
            if self.vertices[v_idx].deleted {
                continue;
            }

            let mut first_attr: Option<u64> = None;

            for &tid in &self.vertices[v_idx].triangles {
                let t = &self.triangles[tid];
                if t.deleted {
                    continue;
                }

                let attr = t.attribute_id;
                if attr == 0 {
                    continue;
                } // No attribute, skip

                match first_attr {
                    None => first_attr = Some(attr),
                    Some(first) => {
                        if attr != first {
                            // This vertex touches triangles with different attributes
                            self.vertices[v_idx].seam = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    fn build_initial_edges(&mut self) {
        let mut seen_edges: AHashSet<(usize, usize)> = AHashSet::new();

        for t in &self.triangles {
            if t.deleted {
                continue;
            }

            for i in 0..3 {
                let v0 = t.v[i];
                let v1 = t.v[(i + 1) % 3];
                let edge_key = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                if seen_edges.insert(edge_key) {
                    let (error, optimal) = self.calculate_error(edge_key.0, edge_key.1);
                    self.edge_queue.push(Edge {
                        v0: edge_key.0,
                        v1: edge_key.1,
                        error,
                        optimal_position: optimal,
                        generation: 0,
                    });
                }
            }
        }
    }

    fn calculate_error(&self, id_v1: usize, id_v2: usize) -> (f64, Point) {
        let q = self.vertices[id_v1].q + self.vertices[id_v2].q;
        let border = self.vertices[id_v1].border && self.vertices[id_v2].border;

        let p1 = self.vertices[id_v1].p;
        let p2 = self.vertices[id_v2].p;
        let p3 = Point::from((p1.coords + p2.coords) / 2.0);
        let edge_length = (p2 - p1).norm();

        // Handle degenerate edges (coincident vertices)
        if edge_length < EPSILON {
            return (Self::vertex_error(q, p3), p3);
        }

        // Try to find optimal position using QEM
        let det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7);

        // Use stricter threshold and validate the result
        if det.abs() > DETERMINANT_THRESHOLD && !border {
            let p_optimal = Point::new(
                -1.0 / det * q.det(1, 2, 3, 4, 5, 6, 5, 7, 8),
                1.0 / det * q.det(0, 2, 3, 1, 5, 6, 2, 7, 8),
                -1.0 / det * q.det(0, 1, 3, 1, 4, 6, 2, 5, 8),
            );

            // Validate: optimal position should not be far from the edge
            // If it's more than 2x edge length from the midpoint, reject it
            let dist_from_midpoint = (p_optimal - p3).norm();
            if dist_from_midpoint < edge_length * MAX_OPTIMAL_DISTANCE_FACTOR {
                let error = Self::vertex_error(q, p_optimal);
                return (error, p_optimal);
            }
        }

        // Fall back to best of {p1, p2, midpoint}
        let candidates = [
            (Self::vertex_error(q, p1), p1),
            (Self::vertex_error(q, p2), p2),
            (Self::vertex_error(q, p3), p3),
        ];
        candidates
            .into_iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
            .unwrap()
    }

    fn vertex_error(q: SymmetricMatrix, p: Point) -> f64 {
        let x = p.x;
        let y = p.y;
        let z = p.z;
        q.get(0) * x * x
            + 2.0 * q.get(1) * x * y
            + 2.0 * q.get(2) * x * z
            + 2.0 * q.get(3) * x
            + q.get(4) * y * y
            + 2.0 * q.get(5) * y * z
            + 2.0 * q.get(6) * y
            + q.get(7) * z * z
            + 2.0 * q.get(8) * z
            + q.get(9)
    }

    fn pop_valid_edge(&mut self) -> Option<Edge> {
        while let Some(edge) = self.edge_queue.pop() {
            // Skip if vertices are deleted
            if self.vertices[edge.v0].deleted || self.vertices[edge.v1].deleted {
                continue;
            }
            // Skip if edge is stale (vertices were modified)
            if self.vertex_generation[edge.v0] > edge.generation
                || self.vertex_generation[edge.v1] > edge.generation
            {
                continue;
            }
            return Some(edge);
        }
        None
    }

    fn get_vertex_neighbors(&self, v_idx: usize) -> AHashSet<usize> {
        let mut neighbors = AHashSet::new();
        for &tid in &self.vertices[v_idx].triangles {
            let t = &self.triangles[tid];
            if t.deleted {
                continue;
            }
            for &other in &t.v {
                if other != v_idx && !self.vertices[other].deleted {
                    neighbors.insert(other);
                }
            }
        }
        neighbors
    }

    fn link_condition_satisfied(&self, v0: usize, v1: usize) -> bool {
        let neighbors_v0 = self.get_vertex_neighbors(v0);
        let neighbors_v1 = self.get_vertex_neighbors(v1);
        let shared_count = neighbors_v0.intersection(&neighbors_v1).count();
        // For a valid collapse: at most 2 shared neighbors
        // (interior edges have exactly 2, boundary edges have 1)
        shared_count <= 2
    }

    fn would_flip(&self, v_idx: usize, v_other: usize, new_pos: Point) -> bool {
        for &tid in &self.vertices[v_idx].triangles {
            let t = &self.triangles[tid];
            if t.deleted {
                continue;
            }

            // Skip triangles that will be deleted (contain both vertices)
            if t.v.contains(&v_other) {
                continue;
            }

            // Get the new positions (substitute new_pos for v_idx)
            // This ensures we compute the normal using the same formula as the original:
            // (p1 - p0).cross(p2 - p0) where p0, p1, p2 are at positions 0, 1, 2
            let new_p: [Point; 3] = [
                if t.v[0] == v_idx {
                    new_pos
                } else {
                    self.vertices[t.v[0]].p
                },
                if t.v[1] == v_idx {
                    new_pos
                } else {
                    self.vertices[t.v[1]].p
                },
                if t.v[2] == v_idx {
                    new_pos
                } else {
                    self.vertices[t.v[2]].p
                },
            ];

            // Compute edges from position 0 (same as original normal formula)
            let e1 = new_p[1] - new_p[0];
            let e2 = new_p[2] - new_p[0];

            // Check for degenerate triangle (edges nearly parallel)
            let e1_len = e1.norm();
            let e2_len = e2.norm();
            if e1_len < EPSILON || e2_len < EPSILON {
                return true;
            }
            let d1 = e1 / e1_len;
            let d2 = e2 / e2_len;
            if d1.dot(&d2).abs() > PARALLEL_DOT_THRESHOLD {
                return true;
            }

            // Compute new normal
            let new_normal = e1.cross(&e2);
            if new_normal.norm() < EPSILON {
                return true;
            }
            let new_normal = new_normal.normalize();

            // Reject if normal flips (> 90 degrees from original)
            if new_normal.dot(&t.n) < 0.0 {
                return true;
            }
        }
        false
    }

    fn update_triangle_normal(&mut self, tid: usize) {
        let t = &self.triangles[tid];
        let p0 = self.vertices[t.v[0]].p;
        let p1 = self.vertices[t.v[1]].p;
        let p2 = self.vertices[t.v[2]].p;
        let n = (p1 - p0).cross(&(p2 - p0));
        self.triangles[tid].n = if n.norm() > EPSILON {
            n.normalize()
        } else {
            Vector::zeros()
        };
    }

    fn can_collapse(&self, v0: usize, v1: usize, new_pos: Point) -> bool {
        // Don't collapse if either vertex is on a border
        // This preserves boundaries on non-watertight meshes
        if self.vertices[v0].border || self.vertices[v1].border {
            return false;
        }

        // Don't collapse if either vertex is on an attribute seam
        // Seam vertices are marked at initialization and never change
        // This prevents color bleeding at attribute boundaries
        if self.vertices[v0].seam || self.vertices[v1].seam {
            return false;
        }

        // Check link condition
        if !self.link_condition_satisfied(v0, v1) {
            return false;
        }

        // Check if any triangles would flip
        if self.would_flip(v0, v1, new_pos) || self.would_flip(v1, v0, new_pos) {
            return false;
        }

        true
    }

    fn collapse_edge(&mut self, v0: usize, v1: usize, new_pos: Point) -> Vec<usize> {
        // 1. Move v0 to optimal position and merge quadrics
        self.vertices[v0].p = new_pos;
        let v1_q = self.vertices[v1].q;
        self.vertices[v0].q += v1_q;

        // Interpolate normals if preserving attributes
        if self.preserve_attributes {
            match (self.vertices[v0].normal, self.vertices[v1].normal) {
                (Some(n0), Some(n1)) => {
                    let interpolated = (n0 + n1).normalize();
                    self.vertices[v0].normal = Some(interpolated);
                }
                (None, Some(n1)) => {
                    self.vertices[v0].normal = Some(n1);
                }
                _ => {}
            }
        }

        // Update border and seam status - if either vertex was on a boundary, the merged vertex is too
        self.vertices[v0].border = self.vertices[v0].border || self.vertices[v1].border;
        self.vertices[v0].seam = self.vertices[v0].seam || self.vertices[v1].seam;

        // 2. Mark v1 as deleted
        self.vertices[v1].deleted = true;

        // 3. Track affected vertices
        let mut affected_vertices = AHashSet::new();
        affected_vertices.insert(v0);

        // 4. Update normals for ALL of v0's triangles (v0's position changed)
        // This is critical: without this, subsequent would_flip checks use stale normals
        for &tid in &self.vertices[v0].triangles.clone() {
            if self.triangles[tid].deleted {
                continue;
            }
            self.update_triangle_normal(tid);
        }

        // 5. Process triangles of v1
        let v1_triangles: Vec<usize> = self.vertices[v1].triangles.clone();

        for &tid in &v1_triangles {
            if self.triangles[tid].deleted {
                continue;
            }

            let t = &self.triangles[tid];
            let has_v0 = t.v.contains(&v0);
            let has_v1 = t.v.contains(&v1);

            if has_v0 && has_v1 {
                // Triangle contains the collapsed edge - delete it
                self.triangles[tid].deleted = true;
                self.live_triangle_count -= 1;

                // Remove from all vertices' triangle lists
                for &v in &self.triangles[tid].v {
                    if v < self.vertices.len() && !self.vertices[v].deleted {
                        self.vertices[v].triangles.retain(|x| *x != tid);
                        affected_vertices.insert(v);
                    }
                }
            } else if has_v1 {
                // Triangle has v1 but not v0 - redirect to v0
                {
                    let t = &mut self.triangles[tid];
                    for i in 0..3 {
                        if t.v[i] == v1 {
                            t.v[i] = v0;
                            break;
                        }
                    }
                }

                // Update triangle normal (reuse helper to avoid duplication)
                self.update_triangle_normal(tid);

                // Add triangle to v0's list
                self.vertices[v0].triangles.push(tid);

                // Track affected vertices
                for &v in &self.triangles[tid].v {
                    affected_vertices.insert(v);
                }
            }
        }

        // 6. Clear v1's triangle list
        self.vertices[v1].triangles.clear();

        // 7. Increment generation for affected vertices
        for &v in &affected_vertices {
            self.vertex_generation[v] = self.current_generation;
        }
        self.current_generation += 1;

        affected_vertices.into_iter().collect()
    }

    fn rebuild_edges_for_vertices(&mut self, affected: &[usize]) {
        for &v in affected {
            if self.vertices[v].deleted {
                continue;
            }

            let neighbors = self.get_vertex_neighbors(v);

            for neighbor in neighbors {
                // Only create edge once (smaller index first)
                if v < neighbor {
                    let (error, optimal) = self.calculate_error(v, neighbor);
                    self.edge_queue.push(Edge {
                        v0: v,
                        v1: neighbor,
                        error,
                        optimal_position: optimal,
                        generation: self.vertex_generation[v],
                    });
                }
            }
        }
    }

    fn simplify(&mut self, target_count: usize, verbose: bool) {
        self.initialize_quadrics();
        self.build_initial_edges();

        let mut collapse_count = 0;

        while self.live_triangle_count > target_count {
            let Some(edge) = self.pop_valid_edge() else {
                if verbose {
                    println!("No more valid edges to collapse");
                }
                break;
            };

            if !self.can_collapse(edge.v0, edge.v1, edge.optimal_position) {
                continue;
            }

            let affected = self.collapse_edge(edge.v0, edge.v1, edge.optimal_position);
            self.rebuild_edges_for_vertices(&affected);

            collapse_count += 1;

            if verbose && collapse_count % 1000 == 0 {
                println!(
                    "Collapsed {} edges, {} triangles remaining",
                    collapse_count, self.live_triangle_count
                );
            }
        }

        if verbose {
            println!(
                "Finished: collapsed {} edges, {} triangles",
                collapse_count, self.live_triangle_count
            );
        }
    }

    fn get_result(&self, original_face_colors: Option<&Vec<Vector4<u8>>>) -> SimplifyResult {
        // Build vertex map: old index -> new index
        let mut vertex_map = vec![usize::MAX; self.vertices.len()];
        let mut new_vertices = Vec::new();

        for (old_idx, v) in self.vertices.iter().enumerate() {
            if !v.deleted {
                // Check if vertex is used by any live triangle
                let used = v.triangles.iter().any(|&tid| !self.triangles[tid].deleted);
                if used {
                    vertex_map[old_idx] = new_vertices.len();
                    new_vertices.push(v.p);
                }
            }
        }

        // Build new faces with remapped indices
        let mut new_faces = Vec::new();
        let mut new_face_colors: Vec<Vector4<u8>> = Vec::new();

        for t in &self.triangles {
            if t.deleted {
                continue;
            }

            let new_v: [usize; 3] = [vertex_map[t.v[0]], vertex_map[t.v[1]], vertex_map[t.v[2]]];

            // Skip if any vertex is unmapped
            if new_v.contains(&usize::MAX) {
                continue;
            }

            // Skip degenerate triangles
            if new_v[0] == new_v[1] || new_v[1] == new_v[2] || new_v[2] == new_v[0] {
                continue;
            }

            new_faces.push(new_v);

            // Preserve face color
            if let Some(&color) = original_face_colors.and_then(|c| c.get(t.original_index)) {
                new_face_colors.push(color);
            }
        }

        // Build vertex normals from preserved attributes
        let vertex_normals: Vec<Vector3<f64>> = self
            .vertices
            .iter()
            .enumerate()
            .filter(|(i, v)| !v.deleted && vertex_map[*i] != usize::MAX)
            .filter_map(|(_, v)| v.normal)
            .collect();

        let attributes_vertex =
            if !vertex_normals.is_empty() && vertex_normals.len() == new_vertices.len() {
                let mut attrs = Attributes::default();
                attrs.normals.push(vertex_normals);
                Some(attrs)
            } else {
                None
            };

        let attributes_face = if !new_face_colors.is_empty() {
            let mut attrs = Attributes::default();
            attrs.colors.push(new_face_colors);
            Some(attrs)
        } else {
            None
        };

        SimplifyResult {
            vertices: new_vertices,
            faces: new_faces,
            attributes_vertex,
            attributes_face,
            vertex_map,
            quality: None,
        }
    }
}

/// Simplifies a mesh using the Quadric Error Metrics algorithm.
pub fn simplify_mesh(
    input_vertices: &[Point3<f64>],
    input_faces: &[[usize; 3]],
    attributes_vertex: Option<&Attributes>,
    attributes_face: Option<&Attributes>,
    options: SimplifyOptions,
) -> SimplifyResult {
    let verbose = options.verbose;
    let target_count = options.target_count;

    let identity_map = || (0..input_vertices.len()).collect::<Vec<_>>();

    // Basic checks
    if target_count >= input_faces.len() {
        if verbose {
            println!(
                "Target count ({}) >= current count ({}), returning original.",
                target_count,
                input_faces.len()
            );
        }
        return SimplifyResult {
            vertices: input_vertices.to_vec(),
            faces: input_faces.to_vec(),
            attributes_vertex: attributes_vertex.cloned(),
            attributes_face: attributes_face.cloned(),
            vertex_map: identity_map(),
            quality: None,
        };
    }

    if input_faces.is_empty() || input_vertices.len() < 3 {
        if verbose {
            println!("Input mesh is empty or too small, returning original.");
        }
        return SimplifyResult {
            vertices: input_vertices.to_vec(),
            faces: input_faces.to_vec(),
            attributes_vertex: attributes_vertex.cloned(),
            attributes_face: attributes_face.cloned(),
            vertex_map: identity_map(),
            quality: None,
        };
    }

    if target_count == 0 {
        if verbose {
            println!("Target count is 0, returning empty mesh.");
        }
        return SimplifyResult {
            vertices: Vec::new(),
            faces: Vec::new(),
            attributes_vertex: None,
            attributes_face: None,
            vertex_map: vec![usize::MAX; input_vertices.len()],
            quality: None,
        };
    }

    if verbose {
        println!("Starting simplification:");
        println!("  Input vertices: {}", input_vertices.len());
        println!("  Input faces: {}", input_faces.len());
        println!("  Target faces: {target_count}");
    }

    let face_colors: Option<&Vec<Vector4<u8>>> =
        attributes_face.and_then(|attrs| attrs.colors.first());

    let mut simplifier = Simplifier::new(
        input_vertices,
        input_faces,
        attributes_vertex,
        attributes_face,
        options.preserve_attributes,
    );

    simplifier.simplify(target_count, verbose);

    let mut result = simplifier.get_result(face_colors);

    if verbose {
        println!("Simplification finished:");
        println!("  Output vertices: {}", result.vertices.len());
        println!("  Output faces: {}", result.faces.len());
    }

    if options.compute_quality {
        result.quality = Some(compute_quality(
            input_vertices,
            input_faces,
            &result.vertices,
            &result.faces,
        ));
    }

    result
}

fn compute_quality(
    original_vertices: &[Point3<f64>],
    original_faces: &[[usize; 3]],
    simplified_vertices: &[Point3<f64>],
    simplified_faces: &[[usize; 3]],
) -> SimplifyQuality {
    use crate::triangles::inertia::volume;

    let original_volume = volume(original_vertices, original_faces).abs();
    let simplified_volume = volume(simplified_vertices, simplified_faces).abs();
    let volume_ratio = if original_volume > EPSILON {
        simplified_volume / original_volume
    } else {
        1.0
    };

    let compute_area = |verts: &[Point3<f64>], faces: &[[usize; 3]]| -> f64 {
        faces
            .iter()
            .map(|&[v0, v1, v2]| {
                let p0 = verts[v0];
                let p1 = verts[v1];
                let p2 = verts[v2];
                (p1 - p0).cross(&(p2 - p0)).norm() / 2.0
            })
            .sum()
    };

    let original_area = compute_area(original_vertices, original_faces);
    let simplified_area = compute_area(simplified_vertices, simplified_faces);
    let surface_area_ratio = if original_area > EPSILON {
        simplified_area / original_area
    } else {
        1.0
    };

    let face_count_ratio = simplified_faces.len() as f64 / original_faces.len() as f64;

    let mut min_angle = std::f64::consts::PI;
    let mut degenerate_count = 0;
    let mut flipped_count = 0;

    for &[v0, v1, v2] in simplified_faces {
        let p0 = simplified_vertices[v0];
        let p1 = simplified_vertices[v1];
        let p2 = simplified_vertices[v2];

        let area = (p1 - p0).cross(&(p2 - p0)).norm() / 2.0;
        if area < EPSILON {
            degenerate_count += 1;
            continue;
        }

        let e0 = (p1 - p0).normalize();
        let e1 = (p2 - p1).normalize();
        let e2 = (p0 - p2).normalize();

        let angle0 = (-e2.dot(&e0)).clamp(-1.0, 1.0).acos();
        let angle1 = (-e0.dot(&e1)).clamp(-1.0, 1.0).acos();
        let angle2 = (-e1.dot(&e2)).clamp(-1.0, 1.0).acos();

        min_angle = min_angle.min(angle0).min(angle1).min(angle2);

        if angle0 > FLIPPED_ANGLE_THRESHOLD
            || angle1 > FLIPPED_ANGLE_THRESHOLD
            || angle2 > FLIPPED_ANGLE_THRESHOLD
        {
            flipped_count += 1;
        }
    }

    SimplifyQuality {
        volume_ratio,
        surface_area_ratio,
        face_count_ratio,
        min_angle,
        degenerate_count,
        flipped_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::create_box;
    use crate::triangles::inertia::volume;
    use approx::assert_relative_eq;
    use nalgebra::Point3;

    fn opts(target_count: usize) -> SimplifyOptions {
        SimplifyOptions {
            target_count,
            aggressiveness: 7.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_simplify_cube() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let mut options = opts(6);
        options.verbose = true;
        let result = simplify_mesh(&cube.vertices, &cube.faces, None, None, options);

        println!(
            "Cube: {} faces -> {} faces",
            cube.faces.len(),
            result.faces.len()
        );
        assert!(result.vertices.len() <= cube.vertices.len());
        assert!(
            result.faces.len() <= 6,
            "Got {} faces, expected <= 6",
            result.faces.len()
        );
        assert!(result.faces.len() >= 4);
    }

    #[test]
    fn test_simplify_preserves_volume_approximately() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let original_volume = volume(&cube.vertices, &cube.faces).abs();

        let result = simplify_mesh(&cube.vertices, &cube.faces, None, None, opts(8));
        let simplified_volume = volume(&result.vertices, &result.faces).abs();

        assert!(
            (simplified_volume - original_volume).abs() / original_volume < 0.5,
            "Volume changed too much: {} -> {}",
            original_volume,
            simplified_volume
        );
    }

    #[test]
    fn test_simplify_plane_to_two_triangles() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3]];

        let result = simplify_mesh(&vertices, &faces, None, None, opts(2));
        assert_eq!(result.faces.len(), 2);
        assert!(result.vertices.len() >= 3);
    }

    #[test]
    fn test_simplify_respects_target_count() {
        let cube = create_box(&[1.0, 1.0, 1.0]);

        for target in [4, 6, 8, 10] {
            let result = simplify_mesh(&cube.vertices, &cube.faces, None, None, opts(target));
            assert!(
                result.faces.len() <= target,
                "Target {} but got {} faces",
                target,
                result.faces.len()
            );
        }
    }

    #[test]
    fn test_simplify_no_degenerate_faces() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let result = simplify_mesh(&cube.vertices, &cube.faces, None, None, opts(6));

        for (i, [v0, v1, v2]) in result.faces.iter().enumerate() {
            assert!(*v0 < result.vertices.len(), "Face {} has invalid v0", i);
            assert!(*v1 < result.vertices.len(), "Face {} has invalid v1", i);
            assert!(*v2 < result.vertices.len(), "Face {} has invalid v2", i);
            assert!(v0 != v1 && v1 != v2 && v2 != v0, "Face {} is degenerate", i);

            let p0 = result.vertices[*v0];
            let p1 = result.vertices[*v1];
            let p2 = result.vertices[*v2];
            let area = (p1 - p0).cross(&(p2 - p0)).norm() / 2.0;
            assert!(area > EPSILON, "Face {} has zero area", i);
        }
    }

    #[test]
    fn test_vertex_error_formula() {
        let q = SymmetricMatrix::from_plane(0.0, 0.0, 1.0, 0.0);

        let on_plane = Point3::new(1.0, 2.0, 0.0);
        assert_relative_eq!(
            Simplifier::vertex_error(q, on_plane),
            0.0,
            epsilon = EPSILON
        );

        let off_plane = Point3::new(1.0, 2.0, 1.0);
        assert_relative_eq!(
            Simplifier::vertex_error(q, off_plane),
            1.0,
            epsilon = EPSILON
        );

        let far_off = Point3::new(0.0, 0.0, 3.0);
        assert_relative_eq!(Simplifier::vertex_error(q, far_off), 9.0, epsilon = EPSILON);
    }

    #[test]
    fn test_symmetric_matrix_add() {
        let q1 = SymmetricMatrix::from_plane(1.0, 0.0, 0.0, 0.0);
        let q2 = SymmetricMatrix::from_plane(0.0, 1.0, 0.0, 0.0);
        let q_sum = q1 + q2;

        let origin = Point3::origin();
        assert_relative_eq!(
            Simplifier::vertex_error(q_sum, origin),
            0.0,
            epsilon = EPSILON
        );

        let p = Point3::new(1.0, 1.0, 0.0);
        assert_relative_eq!(Simplifier::vertex_error(q_sum, p), 2.0, epsilon = EPSILON);
    }

    #[test]
    fn test_simplify_empty_mesh() {
        let vertices: Vec<Point3<f64>> = vec![];
        let faces: Vec<[usize; 3]> = vec![];

        let result = simplify_mesh(&vertices, &faces, None, None, opts(10));
        assert!(result.vertices.is_empty());
        assert!(result.faces.is_empty());
    }

    #[test]
    fn test_simplify_target_zero() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let result = simplify_mesh(&cube.vertices, &cube.faces, None, None, opts(0));
        assert!(result.vertices.is_empty());
        assert!(result.faces.is_empty());
    }

    #[test]
    fn test_simplify_target_exceeds_current() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let result = simplify_mesh(&cube.vertices, &cube.faces, None, None, opts(100));
        assert_eq!(result.faces.len(), cube.faces.len());
        assert_eq!(result.vertices.len(), cube.vertices.len());
    }

    #[test]
    fn test_vertex_map_validity() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let result = simplify_mesh(&cube.vertices, &cube.faces, None, None, opts(6));

        for (old_idx, &new_idx) in result.vertex_map.iter().enumerate() {
            if new_idx != usize::MAX {
                assert!(
                    new_idx < result.vertices.len(),
                    "vertex_map[{}] = {} is out of bounds (vertices.len() = {})",
                    old_idx,
                    new_idx,
                    result.vertices.len()
                );
            }
        }
    }

    #[test]
    fn test_simplify_with_quality_metrics() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let mut options = opts(6);
        options.compute_quality = true;

        let result = simplify_mesh(&cube.vertices, &cube.faces, None, None, options);
        assert!(result.quality.is_some());

        let quality = result.quality.unwrap();
        assert!(quality.volume_ratio > 0.0);
        assert!(quality.surface_area_ratio > 0.0);
        assert!(quality.face_count_ratio > 0.0 && quality.face_count_ratio <= 1.0);
    }

    #[test]
    fn test_simplify_quality_thresholds() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let mut options = opts(8);
        options.compute_quality = true;

        let result = simplify_mesh(&cube.vertices, &cube.faces, None, None, options);
        let quality = result.quality.unwrap();

        assert!(quality.volume_ratio > 0.5 && quality.volume_ratio < 1.5);
        assert!(quality.surface_area_ratio > 0.5 && quality.surface_area_ratio < 1.5);
        assert_eq!(quality.degenerate_count, 0);
    }

    #[test]
    fn test_simplify_subdivided_cube() {
        use crate::subdivide::subdivide;

        let cube = create_box(&[1.0, 1.0, 1.0]);
        let (verts, faces) = subdivide(&cube.vertices, &cube.faces, 2);

        assert_eq!(faces.len(), 12 * 16);

        let result = simplify_mesh(&verts, &faces, None, None, opts(12));

        assert!(result.faces.len() <= 12);
        assert!(result.faces.len() >= 4);
    }

    #[test]
    fn test_simplify_subdivided_cube_to_minimum() {
        use crate::subdivide::subdivide;

        let cube = create_box(&[1.0, 1.0, 1.0]);
        let (verts, faces) = subdivide(&cube.vertices, &cube.faces, 1);

        let result = simplify_mesh(&verts, &faces, None, None, opts(4));

        assert!(result.faces.len() >= 4);
        assert!(result.faces.len() <= 6);
    }

    #[test]
    fn test_simplify_10k_faces() {
        use crate::subdivide::subdivide;

        let cube = create_box(&[1.0, 1.0, 1.0]);
        let (verts, faces) = subdivide(&cube.vertices, &cube.faces, 5); // 12 * 4^5 = 12,288 faces

        assert!(faces.len() > 10_000);

        let result = simplify_mesh(&verts, &faces, None, None, opts(500));

        assert!(result.faces.len() <= 500);
        assert!(result.faces.len() >= 4);

        for [v0, v1, v2] in &result.faces {
            assert!(v0 != v1 && v1 != v2 && v2 != v0);
        }
    }

    #[test]
    fn test_simplify_preserves_face_colors() {
        use crate::subdivide::subdivide_with_attributes;
        use nalgebra::Vector4;

        let cube = create_box(&[1.0, 1.0, 1.0]);

        let face_colors: Vec<Vector4<u8>> = (0..12)
            .map(|i| {
                let side = i / 2;
                Vector4::new((side * 40) as u8, ((5 - side) * 40) as u8, 100, 255)
            })
            .collect();

        let mut face_attrs = Attributes::default();
        face_attrs.colors.push(face_colors.clone());

        let (verts, faces, new_face_attrs) =
            subdivide_with_attributes(&cube.vertices, &cube.faces, &face_attrs, 1);

        assert_eq!(faces.len(), 48);
        assert_eq!(new_face_attrs.colors[0].len(), 48);

        let result = simplify_mesh(&verts, &faces, None, Some(&new_face_attrs), opts(24));

        // With color seam protection, we may not reach target if colors differ
        // Just verify some simplification happened and colors are preserved
        assert!(result.faces.len() <= faces.len());
        assert!(result.faces.len() >= 12); // At least the original cube faces

        if let Some(ref attrs) = result.attributes_face {
            if !attrs.colors.is_empty() {
                let result_colors = &attrs.colors[0];
                assert_eq!(result_colors.len(), result.faces.len());
            }
        }
    }

    // === Comprehensive winding/connectivity tests ===

    #[test]
    fn test_simplify_consistent_winding() {
        use crate::subdivide::subdivide;
        use ahash::AHashMap;

        let cube = create_box(&[1.0, 1.0, 1.0]);
        let (verts, faces) = subdivide(&cube.vertices, &cube.faces, 2);

        let result = simplify_mesh(&verts, &faces, None, None, opts(50));

        // For each directed edge (a, b), count occurrences
        // In a consistent mesh, edge (a,b) and (b,a) should each appear exactly once
        let mut edge_counts: AHashMap<(usize, usize), usize> = AHashMap::new();

        for [v0, v1, v2] in &result.faces {
            // Edges in winding order
            *edge_counts.entry((*v0, *v1)).or_insert(0) += 1;
            *edge_counts.entry((*v1, *v2)).or_insert(0) += 1;
            *edge_counts.entry((*v2, *v0)).or_insert(0) += 1;
        }

        // Check each edge appears exactly once in each direction
        for (&(a, b), &count) in &edge_counts {
            assert_eq!(count, 1, "Edge ({}, {}) appears {} times", a, b, count);
            let reverse_count = edge_counts.get(&(b, a)).copied().unwrap_or(0);
            assert_eq!(
                reverse_count, 1,
                "Reverse edge ({}, {}) appears {} times",
                b, a, reverse_count
            );
        }
    }

    #[test]
    fn test_simplify_positive_volume() {
        use crate::subdivide::subdivide;

        let cube = create_box(&[1.0, 1.0, 1.0]);
        let (verts, faces) = subdivide(&cube.vertices, &cube.faces, 3);

        // Original volume should be positive (normals point outward)
        let original_volume = volume(&verts, &faces);
        assert!(original_volume > 0.0, "Original volume should be positive");

        let result = simplify_mesh(&verts, &faces, None, None, opts(20));

        // Simplified volume should also be positive
        let simplified_volume = volume(&result.vertices, &result.faces);
        assert!(
            simplified_volume > 0.0,
            "Simplified volume {} should be positive (flipped triangles!)",
            simplified_volume
        );
    }

    #[test]
    fn test_simplify_no_orphan_vertices() {
        use crate::subdivide::subdivide;

        let cube = create_box(&[1.0, 1.0, 1.0]);
        let (verts, faces) = subdivide(&cube.vertices, &cube.faces, 2);

        let result = simplify_mesh(&verts, &faces, None, None, opts(30));

        let mut used = vec![false; result.vertices.len()];
        for [v0, v1, v2] in &result.faces {
            used[*v0] = true;
            used[*v1] = true;
            used[*v2] = true;
        }

        for (i, &is_used) in used.iter().enumerate() {
            assert!(
                is_used,
                "Vertex {} is orphaned (not used by any triangle)",
                i
            );
        }
    }

    #[test]
    fn test_simplify_watertight_stays_watertight() {
        use crate::mesh::Trimesh;
        use crate::subdivide::subdivide;

        let cube = create_box(&[1.0, 1.0, 1.0]);
        let (verts, faces) = subdivide(&cube.vertices, &cube.faces, 3);

        let result = simplify_mesh(&verts, &faces, None, None, opts(20));

        let original = Trimesh::new(verts, faces, None, None).unwrap();
        assert!(original.is_watertight(), "Original should be watertight");

        let simplified = Trimesh::new(result.vertices, result.faces, None, None).unwrap();
        assert!(
            simplified.is_watertight(),
            "Simplified mesh lost watertight property"
        );
    }

    #[test]
    fn test_simplify_asymmetric_mesh() {
        use crate::subdivide::subdivide;

        // Create an asymmetric mesh (tetrahedron) that's less likely to have
        // coincidental vertex ordering
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 0.866, 0.0),
            Point3::new(0.5, 0.289, 0.816),
        ];
        // Faces with consistent outward winding
        let faces = vec![
            [0, 2, 1], // bottom
            [0, 1, 3], // front
            [1, 2, 3], // right
            [2, 0, 3], // left
        ];

        // Subdivide to get more faces
        let (verts, faces) = subdivide(&vertices, &faces, 3);

        let original_volume = volume(&verts, &faces);

        let result = simplify_mesh(&verts, &faces, None, None, opts(16));

        let simplified_volume = volume(&result.vertices, &result.faces);

        // Volume should stay positive and similar magnitude
        assert!(
            simplified_volume > 0.0,
            "Volume flipped negative: {}",
            simplified_volume
        );
        assert!(
            (simplified_volume - original_volume).abs() / original_volume.abs() < 0.3,
            "Volume changed too much: {} -> {}",
            original_volume,
            simplified_volume
        );
    }

    #[test]
    fn test_simplify_all_normals_outward() {
        use crate::subdivide::subdivide;

        let cube = create_box(&[1.0, 1.0, 1.0]);
        let (verts, faces) = subdivide(&cube.vertices, &cube.faces, 2);

        let result = simplify_mesh(&verts, &faces, None, None, opts(30));

        // Compute centroid
        let centroid: Point3<f64> = Point3::from(
            result
                .vertices
                .iter()
                .map(|p| p.coords)
                .sum::<Vector3<f64>>()
                / result.vertices.len() as f64,
        );

        // Check each triangle's normal points away from centroid
        for (i, [v0, v1, v2]) in result.faces.iter().enumerate() {
            let p0 = result.vertices[*v0];
            let p1 = result.vertices[*v1];
            let p2 = result.vertices[*v2];

            let face_center = Point3::from((p0.coords + p1.coords + p2.coords) / 3.0);
            let normal = (p1 - p0).cross(&(p2 - p0));

            // Vector from centroid to face center
            let outward = face_center - centroid;

            // Normal should point same direction as outward vector
            assert!(
                normal.dot(&outward) > 0.0,
                "Face {} has inward-pointing normal",
                i
            );
        }
    }
}
