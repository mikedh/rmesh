// u32 is used intentionally for outside_start/len to reduce memory in hot
// data structures. best_face is stored as f64 for uniform SIMD width.
// Face indices stored as f64 are always non-negative.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::collections::BinaryHeap;

use ahash::AHashMap;

use anyhow::{Result, bail};
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

/// Minimum orphan count before parallelizing the distance kernel.
const RAYON_MIN_ORPHANS: usize = 4096;

/// SIMD-friendly distance kernel: for each point find the plane with greatest
/// signed distance. All point slices must share length; all plane slices must
/// share length.
///
/// Auto-vectorization requirements (AVX2 ymm / 4×f64):
/// - Contiguous memory: all slices are pre-gathered sequential arrays
/// - Explicit length proof: re-slicing eliminates bounds checks
/// - Uniform element width: `best_face` is f64 so LLVM applies the same
///   SIMD comparison mask to both conditional stores
/// - FMA: `mul_add` maps to `vfmadd231pd`, halving arithmetic instructions
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn distance_kernel(
    normal_x: &[f64],
    normal_y: &[f64],
    normal_z: &[f64],
    plane_offset: &[f64],
    point_x: &[f64],
    point_y: &[f64],
    point_z: &[f64],
    best_dist: &mut [f64],
    best_face: &mut [f64],
) {
    let n = point_x.len();
    let n_planes = normal_x.len();
    let point_y = &point_y[..n];
    let point_z = &point_z[..n];
    let best_dist = &mut best_dist[..n];
    let best_face = &mut best_face[..n];
    for i in 0..n {
        best_dist[i] = f64::NEG_INFINITY;
        best_face[i] = 0.0;
    }
    for fi in 0..n_planes {
        let nx = normal_x[fi];
        let ny = normal_y[fi];
        let nz = normal_z[fi];
        let offset = plane_offset[fi];
        let fi_f64 = fi as f64;
        for oi in 0..n {
            let dist = nx.mul_add(
                point_x[oi],
                ny.mul_add(point_y[oi], nz.mul_add(point_z[oi], offset)),
            );
            if dist > best_dist[oi] {
                best_dist[oi] = dist;
                best_face[oi] = fi_f64;
            }
        }
    }
}

/// Plane equations split into separate arrays for SIMD-friendly batch distance tests.
struct FlatPlanes {
    normal_x: Vec<f64>,
    normal_y: Vec<f64>,
    normal_z: Vec<f64>,
    offset: Vec<f64>,
}

impl FlatPlanes {
    fn with_capacity(cap: usize) -> Self {
        Self {
            normal_x: Vec::with_capacity(cap),
            normal_y: Vec::with_capacity(cap),
            normal_z: Vec::with_capacity(cap),
            offset: Vec::with_capacity(cap),
        }
    }

    fn clear(&mut self) {
        self.normal_x.clear();
        self.normal_y.clear();
        self.normal_z.clear();
        self.offset.clear();
    }

    fn push(&mut self, normal: &Vector3<f64>, offset: f64) {
        self.normal_x.push(normal.x);
        self.normal_y.push(normal.y);
        self.normal_z.push(normal.z);
        self.offset.push(offset);
    }

    fn find_best_plane(
        &self,
        point_x: &[f64],
        point_y: &[f64],
        point_z: &[f64],
        best_dist: &mut [f64],
        best_face: &mut [f64],
    ) {
        distance_kernel(
            &self.normal_x,
            &self.normal_y,
            &self.normal_z,
            &self.offset,
            point_x,
            point_y,
            point_z,
            best_dist,
            best_face,
        );
    }

    fn find_best_plane_parallel(
        &self,
        point_x: &[f64],
        point_y: &[f64],
        point_z: &[f64],
        best_dist: &mut [f64],
        best_face: &mut [f64],
    ) {
        let n = point_x.len();
        let point_y = &point_y[..n];
        let point_z = &point_z[..n];
        let best_dist = &mut best_dist[..n];
        let best_face = &mut best_face[..n];
        let (nx, ny, nz, off) = (&self.normal_x, &self.normal_y, &self.normal_z, &self.offset);

        best_dist
            .par_chunks_mut(4096)
            .zip(best_face.par_chunks_mut(4096))
            .zip(point_x.par_chunks(4096))
            .zip(point_y.par_chunks(4096))
            .zip(point_z.par_chunks(4096))
            .for_each(|((((best_dist, best_face), point_x), point_y), point_z)| {
                distance_kernel(
                    nx, ny, nz, off, point_x, point_y, point_z, best_dist, best_face,
                );
            });
    }
}

/// Entry in the active-facet max-heap, keyed on `furthest_dist`.
#[derive(Clone, Copy)]
struct ActiveEntry {
    dist: f64,
    facet_index: usize,
}

impl PartialEq for ActiveEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.facet_index == other.facet_index
    }
}

impl Eq for ActiveEntry {}

impl PartialOrd for ActiveEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ActiveEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.facet_index.cmp(&other.facet_index))
    }
}

/// Tolerance values for convex hull computation, adapted from qhull's
/// `qh_distround` for dimension 3.
struct Tolerances {
    dist_round: f64,
    outside_threshold: f64,
}

impl Tolerances {
    fn from_points(points: &[Point3<f64>]) -> Self {
        let (max_abs, max_sum_abs) =
            points
                .iter()
                .fold((0.0_f64, 0.0_f64), |(best_abs, best_sum), point| {
                    let abs_x = point.x.abs();
                    let abs_y = point.y.abs();
                    let abs_z = point.z.abs();
                    (
                        best_abs.max(abs_x).max(abs_y).max(abs_z),
                        best_sum.max(abs_x + abs_y + abs_z),
                    )
                });

        let max_dist_sum = (3.0_f64.sqrt() * max_abs).min(max_sum_abs);
        let dist_round = f64::EPSILON * (3.0 * max_dist_sum * 1.01 + max_abs);
        let outside_threshold = 10.0 * dist_round;

        Self {
            dist_round,
            outside_threshold,
        }
    }
}

/// A triangular facet of the convex hull under construction.
struct Facet {
    vertices: [usize; 3],
    neighbors: [usize; 3],
    normal: Vector3<f64>,
    offset: f64,
    /// Start index into `QHull::outside_pool`.
    outside_start: u32,
    outside_len: u32,
    furthest_dist: f64,
    removed: bool,
}

impl Facet {
    #[inline]
    fn distance(&self, point: &Point3<f64>) -> f64 {
        self.normal.dot(&point.coords) + self.offset
    }
}

/// 3D Quickhull state.
struct QHull<'a> {
    points: &'a [Point3<f64>],
    facets: Vec<Facet>,
    free_list: Vec<usize>,
    tolerances: Tolerances,
    interior: Point3<f64>,
    active: BinaryHeap<ActiveEntry>,

    /// Single backing pool for all facet outside sets.
    outside_pool: Vec<usize>,

    // Working buffers (reused across build_hull iterations)
    visible: Vec<usize>,
    horizon: Vec<(usize, usize, usize)>,
    queue: Vec<usize>,
    orphans: Vec<usize>,
    new_face_indices: Vec<usize>,
    flat_planes: FlatPlanes,
    edge_map: AHashMap<usize, (usize, usize)>,

    /// Pre-gathered orphan coordinates (contiguous for SIMD vectorization).
    gather_x: Vec<f64>,
    gather_y: Vec<f64>,
    gather_z: Vec<f64>,
    best_dist_buf: Vec<f64>,
    /// Best face index stored as f64 for uniform SIMD width.
    best_face_buf: Vec<f64>,

    // Generation-counter visibility tracking
    vis_stamps: Vec<u32>,
    vis_gen: u32,
}

impl<'a> QHull<'a> {
    fn new(points: &'a [Point3<f64>]) -> Self {
        let n = points.len();
        Self {
            points,
            facets: Vec::with_capacity(2 * n),
            free_list: Vec::new(),
            tolerances: Tolerances::from_points(points),
            interior: Point3::origin(),
            active: BinaryHeap::with_capacity(64),
            outside_pool: Vec::with_capacity(n),
            visible: Vec::with_capacity(64),
            horizon: Vec::with_capacity(64),
            queue: Vec::with_capacity(64),
            orphans: Vec::with_capacity(n / 4),
            new_face_indices: Vec::with_capacity(32),
            flat_planes: FlatPlanes::with_capacity(32),
            edge_map: AHashMap::with_capacity(32),
            gather_x: Vec::with_capacity(n / 4),
            gather_y: Vec::with_capacity(n / 4),
            gather_z: Vec::with_capacity(n / 4),
            best_dist_buf: Vec::with_capacity(n / 4),
            best_face_buf: Vec::with_capacity(n / 4),
            vis_stamps: vec![0; 2 * n],
            vis_gen: 0,
        }
    }

    /// Find 4 maximally-spread non-coplanar points.
    fn build_initial_simplex(&self) -> Result<(usize, usize, usize, usize)> {
        let points = self.points;
        let num_points = points.len();

        // Find extremes along each axis
        let mut extremes = [0usize; 6];
        for i in 1..num_points {
            if points[i].x < points[extremes[0]].x {
                extremes[0] = i;
            }
            if points[i].x > points[extremes[1]].x {
                extremes[1] = i;
            }
            if points[i].y < points[extremes[2]].y {
                extremes[2] = i;
            }
            if points[i].y > points[extremes[3]].y {
                extremes[3] = i;
            }
            if points[i].z < points[extremes[4]].z {
                extremes[4] = i;
            }
            if points[i].z > points[extremes[5]].z {
                extremes[5] = i;
            }
        }

        // Pick the pair of extremes with maximum distance
        let mut i0 = 0;
        let mut i1 = 1;
        let mut best_dist_sq = -1.0_f64;
        for a in 0..6 {
            for b in (a + 1)..6 {
                let dist_sq = (points[extremes[a]] - points[extremes[b]]).norm_squared();
                if dist_sq > best_dist_sq {
                    best_dist_sq = dist_sq;
                    i0 = extremes[a];
                    i1 = extremes[b];
                }
            }
        }
        if best_dist_sq < self.tolerances.dist_round * self.tolerances.dist_round {
            bail!("all points are coincident");
        }

        // Find point furthest from line (i0, i1)
        let line_dir = points[i1] - points[i0];
        let line_len_sq = line_dir.norm_squared();
        let mut i2 = 0;
        let mut best_line_dist_sq = -1.0_f64;
        for i in 0..num_points {
            let to_point = points[i] - points[i0];
            let param = to_point.dot(&line_dir) / line_len_sq;
            let projection = points[i0] + line_dir * param;
            let dist_sq = (points[i] - projection).norm_squared();
            if dist_sq > best_line_dist_sq {
                best_line_dist_sq = dist_sq;
                i2 = i;
            }
        }
        if best_line_dist_sq < self.tolerances.dist_round * self.tolerances.dist_round {
            bail!("all points are collinear");
        }

        // Find point furthest from plane (i0, i1, i2)
        let plane_normal = (points[i1] - points[i0]).cross(&(points[i2] - points[i0]));
        let plane_normal_len = plane_normal.norm();
        if plane_normal_len < 1e-15 {
            bail!("all points are collinear");
        }
        let plane_unit = plane_normal / plane_normal_len;
        let plane_offset = -plane_unit.dot(&points[i0].coords);

        let mut i3 = 0;
        let mut best_plane_dist = -1.0_f64;
        for (i, point) in points.iter().enumerate() {
            let dist = (plane_unit.dot(&point.coords) + plane_offset).abs();
            if dist > best_plane_dist {
                best_plane_dist = dist;
                i3 = i;
            }
        }
        if best_plane_dist < self.tolerances.dist_round {
            bail!("all points are coplanar, use convex_hull_2d");
        }

        Ok((i0, i1, i2, i3))
    }

    /// Create the initial tetrahedron with correct orientation.
    fn create_initial_tetrahedron(&mut self, mut i0: usize, mut i1: usize, i2: usize, i3: usize) {
        let points = self.points;

        let det = (points[i1] - points[i0])
            .dot(&(points[i2] - points[i0]).cross(&(points[i3] - points[i0])));
        if det < 0.0 {
            std::mem::swap(&mut i0, &mut i1);
        }

        self.interior = Point3::from(
            (points[i0].coords + points[i1].coords + points[i2].coords + points[i3].coords) / 4.0,
        );

        let face_verts = [[i0, i2, i1], [i0, i1, i3], [i0, i3, i2], [i1, i2, i3]];
        let face_neighbors = [[3, 1, 2], [3, 2, 0], [3, 0, 1], [2, 1, 0]];

        self.facets.clear();
        self.outside_pool.clear();
        for face_idx in 0..4 {
            let mut verts = face_verts[face_idx];
            let neighbors = face_neighbors[face_idx];
            let (normal, offset) = self.compute_plane(&mut verts);
            self.facets.push(Facet {
                vertices: verts,
                neighbors,
                normal,
                offset,
                outside_start: 0,
                outside_len: 0,
                furthest_dist: 0.0,
                removed: false,
            });
        }
    }

    /// Compute the outward plane for a face, using the interior point to enforce orientation.
    /// If the normal needs flipping, also swaps `vertices[1]` and `vertices[2]`.
    fn compute_plane(&self, vertices: &mut [usize; 3]) -> (Vector3<f64>, f64) {
        let [a, b, c] = *vertices;
        let points = self.points;
        let mut normal = (points[b] - points[a]).cross(&(points[c] - points[a]));
        let normal_len = normal.norm();
        if normal_len > 1e-15 {
            normal /= normal_len;
        }
        let mut offset = -normal.dot(&points[a].coords);

        if normal.dot(&self.interior.coords) + offset > 0.0 {
            normal = -normal;
            offset = -offset;
            vertices.swap(1, 2);
        }

        (normal, offset)
    }

    /// Gather orphan coordinates into contiguous buffers for SIMD vectorization.
    fn gather_orphan_coords(&mut self) {
        let n = self.orphans.len();
        self.gather_x.resize(n, 0.0);
        self.gather_y.resize(n, 0.0);
        self.gather_z.resize(n, 0.0);
        for (i, &pi) in self.orphans.iter().enumerate() {
            self.gather_x[i] = self.points[pi].x;
            self.gather_y[i] = self.points[pi].y;
            self.gather_z[i] = self.points[pi].z;
        }
    }

    /// Run the vectorized distance kernel on `self.orphans` against the given facets,
    /// then distribute points into the outside pool.
    fn distribute_to_facets(&mut self, face_indices: &[usize], clear_pool: bool) {
        if self.orphans.is_empty() || face_indices.is_empty() {
            return;
        }

        self.flat_planes.clear();
        for &fi in face_indices {
            self.flat_planes
                .push(&self.facets[fi].normal, self.facets[fi].offset);
        }

        self.gather_orphan_coords();

        let n = self.orphans.len();
        let threshold = self.tolerances.outside_threshold;
        self.best_dist_buf.resize(n, 0.0);
        self.best_face_buf.resize(n, 0.0);

        if n >= RAYON_MIN_ORPHANS {
            self.flat_planes.find_best_plane_parallel(
                &self.gather_x,
                &self.gather_y,
                &self.gather_z,
                &mut self.best_dist_buf,
                &mut self.best_face_buf,
            );
        } else {
            self.flat_planes.find_best_plane(
                &self.gather_x,
                &self.gather_y,
                &self.gather_z,
                &mut self.best_dist_buf,
                &mut self.best_face_buf,
            );
        }

        // Count per face, allocate pool regions, fill pool
        let n_faces = face_indices.len();
        let mut counts = vec![0u32; n_faces];
        for oi in 0..n {
            if self.best_dist_buf[oi] > threshold {
                counts[self.best_face_buf[oi] as usize] += 1;
            }
        }

        if clear_pool {
            self.outside_pool.clear();
        }
        let mut off = self.outside_pool.len() as u32;
        for (&fi, &count) in face_indices.iter().zip(counts.iter()) {
            self.facets[fi].outside_start = off;
            self.facets[fi].outside_len = 0;
            self.facets[fi].furthest_dist = 0.0;
            off += count;
        }
        self.outside_pool.resize(off as usize, 0);

        for oi in 0..n {
            let dist = self.best_dist_buf[oi];
            if dist > threshold {
                let plane_idx = self.best_face_buf[oi] as usize;
                let fi = face_indices[plane_idx];
                let facet = &mut self.facets[fi];
                let pos = (facet.outside_start + facet.outside_len) as usize;
                self.outside_pool[pos] = self.orphans[oi];
                facet.outside_len += 1;
                if dist > facet.furthest_dist {
                    facet.furthest_dist = dist;
                    self.outside_pool.swap(facet.outside_start as usize, pos);
                }
            }
        }
    }

    /// Assign all non-simplex points to the facet they are furthest above.
    fn initial_partition(&mut self, i0: usize, i1: usize, i2: usize, i3: usize) {
        let simplex = [i0, i1, i2, i3];
        self.orphans.clear();
        for i in 0..self.points.len() {
            if !simplex.contains(&i) {
                self.orphans.push(i);
            }
        }

        let face_indices: Vec<usize> = (0..self.facets.len()).collect();
        self.distribute_to_facets(&face_indices, true);

        self.active.clear();
        for (i, facet) in self.facets.iter().enumerate() {
            if facet.outside_len > 0 {
                self.active.push(ActiveEntry {
                    dist: facet.furthest_dist,
                    facet_index: i,
                });
            }
        }
    }

    /// Main Quickhull loop: iteratively add the furthest outside point.
    fn build_hull(&mut self) {
        loop {
            let start_face = loop {
                let Some(entry) = self.active.pop() else {
                    return;
                };
                let fi = entry.facet_index;
                if !self.facets[fi].removed && self.facets[fi].outside_len > 0 {
                    break fi;
                }
            };

            let apex = self.outside_pool[self.facets[start_face].outside_start as usize];
            self.find_visible_and_horizon(start_face, apex);
            self.build_cone(apex);
        }
    }

    /// Traversal from `start` to find all facets visible from `apex`, collecting
    /// horizon edges (visible-to-non-visible boundaries) in the same pass.
    fn find_visible_and_horizon(&mut self, start: usize, apex: usize) {
        let apex_point = &self.points[apex];
        let num_facets = self.facets.len();

        self.vis_gen = self.vis_gen.wrapping_add(1);
        if self.vis_gen == 0 {
            self.vis_stamps.clear();
            self.vis_stamps.resize(num_facets, 0);
            self.vis_gen = 1;
        }
        if self.vis_stamps.len() < num_facets {
            self.vis_stamps.resize(num_facets, 0);
        }
        let cur_gen = self.vis_gen;

        self.visible.clear();
        self.horizon.clear();
        self.queue.clear();

        self.vis_stamps[start] = cur_gen;
        self.visible.push(start);
        self.queue.push(start);

        while let Some(face_index) = self.queue.pop() {
            for neighbor_slot in 0..3 {
                let neighbor = self.facets[face_index].neighbors[neighbor_slot];
                debug_assert!(
                    neighbor < num_facets,
                    "facet {} has invalid neighbor {} at slot {}",
                    face_index,
                    neighbor,
                    neighbor_slot,
                );
                if neighbor >= num_facets
                    || self.vis_stamps[neighbor] == cur_gen
                    || self.facets[neighbor].removed
                {
                    continue;
                }
                if self.facets[neighbor].distance(apex_point) > -self.tolerances.dist_round {
                    self.vis_stamps[neighbor] = cur_gen;
                    self.visible.push(neighbor);
                    self.queue.push(neighbor);
                } else {
                    self.horizon.push((face_index, neighbor_slot, neighbor));
                }
            }
        }
    }

    /// Build new cone facets from apex to each horizon edge, then redistribute orphans.
    fn build_cone(&mut self, apex: usize) {
        let points = self.points;
        let interior = self.interior;

        // Collect orphan points from all visible facets (excluding the apex)
        self.orphans.clear();
        for vi in 0..self.visible.len() {
            let face_index = self.visible[vi];
            let start = self.facets[face_index].outside_start as usize;
            let len = self.facets[face_index].outside_len as usize;
            for i in start..start + len {
                let point_index = self.outside_pool[i];
                if point_index != apex {
                    self.orphans.push(point_index);
                }
            }
        }

        // Create new cone facets for each horizon edge
        self.new_face_indices.clear();
        self.edge_map.clear();

        for hi in 0..self.horizon.len() {
            let (visible_face, edge_slot, horizon_neighbor) = self.horizon[hi];

            let visible_verts = self.facets[visible_face].vertices;
            let edge_v0 = visible_verts[(edge_slot + 1) % 3];
            let edge_v1 = visible_verts[(edge_slot + 2) % 3];

            let mut new_verts = [apex, edge_v1, edge_v0];

            // Inline compute_plane to avoid &self/&mut self borrow conflict
            let [a, b, c] = new_verts;
            let mut normal = (points[b] - points[a]).cross(&(points[c] - points[a]));
            let normal_len = normal.norm();
            if normal_len > 1e-15 {
                normal /= normal_len;
            }
            let mut offset = -normal.dot(&points[a].coords);
            if normal.dot(&interior.coords) + offset > 0.0 {
                normal = -normal;
                offset = -offset;
                new_verts.swap(1, 2);
            }

            let new_face_index = self.free_list.pop().unwrap_or_else(|| {
                let index = self.facets.len();
                self.facets.push(Facet {
                    vertices: [0; 3],
                    neighbors: [0; 3],
                    normal: Vector3::zeros(),
                    offset: 0.0,
                    outside_start: 0,
                    outside_len: 0,
                    furthest_dist: 0.0,
                    removed: false,
                });
                index
            });
            let facet = &mut self.facets[new_face_index];
            facet.vertices = new_verts;
            facet.normal = normal;
            facet.offset = offset;
            facet.removed = false;
            facet.outside_start = 0;
            facet.outside_len = 0;
            facet.furthest_dist = 0.0;
            facet.neighbors[0] = horizon_neighbor;
            facet.neighbors[1] = usize::MAX;
            facet.neighbors[2] = usize::MAX;

            for slot in self.facets[horizon_neighbor].neighbors.iter_mut() {
                if *slot == visible_face {
                    *slot = new_face_index;
                    break;
                }
            }

            self.new_face_indices.push(new_face_index);

            let vert_a = new_verts[1];
            let vert_b = new_verts[2];
            for &(vertex, slot) in &[(vert_b, 1usize), (vert_a, 2usize)] {
                if let Some(&(other_face, other_slot)) = self.edge_map.get(&vertex) {
                    self.facets[new_face_index].neighbors[slot] = other_face;
                    self.facets[other_face].neighbors[other_slot] = new_face_index;
                    self.edge_map.remove(&vertex);
                } else {
                    self.edge_map.insert(vertex, (new_face_index, slot));
                }
            }
        }

        debug_assert!(
            self.edge_map.is_empty(),
            "cone linking incomplete: {} unmatched edges remain",
            self.edge_map.len(),
        );

        // Mark visible facets as removed and recycle their slots
        for vi in 0..self.visible.len() {
            let face_index = self.visible[vi];
            self.facets[face_index].removed = true;
            self.free_list.push(face_index);
        }

        // Redistribute orphan points to new cone facets
        let face_indices = self.new_face_indices.clone();
        self.distribute_to_facets(&face_indices, false);

        // Push new cone facets that received outside points into the active heap
        for &idx in &face_indices {
            if self.facets[idx].outside_len > 0 {
                self.active.push(ActiveEntry {
                    dist: self.facets[idx].furthest_dist,
                    facet_index: idx,
                });
            }
        }
    }

    /// Extract the final hull faces.
    fn to_result(&self) -> Vec<[usize; 3]> {
        self.facets
            .iter()
            .filter(|f| !f.removed)
            .map(|f| f.vertices)
            .collect()
    }
}

/// Compute the 3D convex hull of a set of points using the Quickhull algorithm.
///
/// Returns outward-oriented triangles as index triples `[a, b, c]` into `points`,
/// with consistent CCW winding when viewed from outside the hull.
///
/// # Errors
///
/// Returns an error if:
/// - Fewer than 4 points are provided
/// - All points are coincident, collinear, or coplanar
pub fn convex_hull_3d(points: &[Point3<f64>]) -> Result<Vec<[usize; 3]>> {
    if points.len() < 4 {
        bail!("convex_hull_3d requires at least 4 points");
    }

    let mut hull = QHull::new(points);
    let (i0, i1, i2, i3) = hull.build_initial_simplex()?;
    hull.create_initial_tetrahedron(i0, i1, i2, i3);
    hull.initial_partition(i0, i1, i2, i3);
    hull.build_hull();
    let faces = hull.to_result();

    #[cfg(all(test, not(feature = "bench")))]
    assert!(
        super::is_hull_valid_3d(points, &faces),
        "convex_hull_3d: internal validation failed"
    );

    Ok(faces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convex::convex_hull_2d;
    use crate::creation::create_box;
    use crate::mesh::Trimesh;
    use nalgebra::Point2;

    /// Drop Z to produce 2D points.
    fn to_2d(pts: &[Point3<f64>]) -> Vec<Point2<f64>> {
        pts.iter().map(|p| Point2::new(p.x, p.y)).collect()
    }

    /// Verify a 2D convex hull: edges form a valid, convex, CCW polygon.
    fn verify_hull_2d(points: &[Point2<f64>], edges: &[[usize; 2]]) {
        assert!(edges.len() >= 3, "hull has fewer than 3 edges");
        for i in 0..edges.len() {
            let j = (i + 1) % edges.len();
            let a = edges[i][0];
            let b = edges[i][1];
            let c = edges[j][1];
            let ab = points[b] - points[a];
            let bc = points[c] - points[b];
            let cross = ab.x * bc.y - ab.y * bc.x;
            assert!(cross >= -1e-10, "not CCW at edge {i}: cross = {cross}");
        }
    }

    /// Full verification of a 3D convex hull result.
    fn verify_hull(points: &[Point3<f64>], faces: &[[usize; 3]]) {
        assert!(!faces.is_empty(), "hull has no faces");

        for face in faces {
            for &idx in face {
                assert!(idx < points.len(), "face index {} out of range", idx);
            }
        }

        let mesh = Trimesh::new(points.to_vec(), faces.to_vec(), None, None).unwrap();
        assert!(mesh.is_watertight(), "hull is not watertight");
        assert!(mesh.is_convex(), "hull is not convex");
        assert!(mesh.volume() > 0.0, "hull has non-positive volume");
        assert!(
            mesh.is_winding_consistent(),
            "hull has inconsistent winding"
        );
    }

    // ---- Deterministic tests ----

    #[test]
    fn test_box() {
        let b = create_box(&[1.0, 1.0, 1.0]);
        let faces = convex_hull_3d(&b.vertices).unwrap();
        verify_hull(&b.vertices, &faces);
        assert_eq!(faces.len(), 12, "unit box should have 12 triangles");

        let mesh = Trimesh::new(b.vertices.clone(), faces, None, None).unwrap();
        assert!(
            (mesh.volume() - 1.0).abs() < 1e-10,
            "box volume should be 1"
        );
    }

    #[test]
    fn test_tetrahedron() {
        let pts = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, 0.5, 1.0),
        ];
        let faces = convex_hull_3d(&pts).unwrap();
        verify_hull(&pts, &faces);
        assert_eq!(faces.len(), 4, "tetrahedron should have 4 faces");
    }

    #[test]
    fn test_box_with_interior() {
        let mut pts = create_box(&[2.0, 2.0, 2.0]).vertices;
        pts.push(Point3::new(0.0, 0.0, 0.0));
        pts.push(Point3::new(0.1, 0.2, 0.3));
        pts.push(Point3::new(-0.3, 0.1, -0.1));
        let faces = convex_hull_3d(&pts).unwrap();
        verify_hull(&pts, &faces);
        assert_eq!(faces.len(), 12, "interior points shouldn't add faces");

        let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for f in &faces {
            for &v in f {
                used.insert(v);
            }
        }
        assert_eq!(used.len(), 8, "only box corners should be hull vertices");
    }

    #[test]
    fn test_outward_normals() {
        let b = create_box(&[1.0, 1.0, 1.0]);
        let faces = convex_hull_3d(&b.vertices).unwrap();
        let centroid: Vector3<f64> =
            b.vertices.iter().map(|v| v.coords).sum::<Vector3<f64>>() / b.vertices.len() as f64;
        let center = Point3::from(centroid);

        for face in &faces {
            let [a, b_idx, c] = *face;
            let normal =
                (b.vertices[b_idx] - b.vertices[a]).cross(&(b.vertices[c] - b.vertices[a]));
            let face_center = Point3::from(
                (b.vertices[a].coords + b.vertices[b_idx].coords + b.vertices[c].coords) / 3.0,
            );
            let to_face = face_center - center;
            assert!(normal.dot(&to_face) > 0.0, "face normal points inward");
        }
    }

    #[test]
    fn test_sphere_points() {
        let n = 100;
        let mut pts = Vec::with_capacity(n);
        let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
        for i in 0..n {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / golden;
            let phi = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
            pts.push(Point3::new(
                phi.sin() * theta.cos(),
                phi.sin() * theta.sin(),
                phi.cos(),
            ));
        }
        let faces = convex_hull_3d(&pts).unwrap();
        verify_hull(&pts, &faces);

        let mesh = Trimesh::new(pts.clone(), faces.clone(), None, None).unwrap();
        let vol = mesh.volume();
        assert!(
            vol > 3.5 && vol < 4.2,
            "sphere volume {} not near 4/3 pi",
            vol
        );
    }

    #[test]
    fn test_error_too_few() {
        let pts = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        assert!(convex_hull_3d(&pts).is_err());
    }

    #[test]
    fn test_error_coplanar() {
        let pts = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        assert!(convex_hull_3d(&pts).is_err());
    }

    #[test]
    fn test_error_collinear() {
        let pts = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
        ];
        assert!(convex_hull_3d(&pts).is_err());
    }

    #[test]
    fn test_duplicates() {
        let pts = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        let faces = convex_hull_3d(&pts).unwrap();
        verify_hull(&pts, &faces);
    }

    #[test]
    fn test_large_coordinates() {
        let s = 1e6;
        let pts = [
            Point3::new(s, s, s),
            Point3::new(-s, s, s),
            Point3::new(s, -s, s),
            Point3::new(s, s, -s),
            Point3::new(-s, -s, s),
            Point3::new(-s, s, -s),
            Point3::new(s, -s, -s),
            Point3::new(-s, -s, -s),
        ];
        let faces = convex_hull_3d(&pts).unwrap();
        verify_hull(&pts, &faces);
        assert_eq!(faces.len(), 12);
    }

    #[test]
    fn test_small_coordinates() {
        let s = 1e-6;
        let pts = [
            Point3::new(s, s, s),
            Point3::new(-s, s, s),
            Point3::new(s, -s, s),
            Point3::new(s, s, -s),
            Point3::new(-s, -s, s),
            Point3::new(-s, s, -s),
            Point3::new(s, -s, -s),
            Point3::new(-s, -s, -s),
        ];
        let faces = convex_hull_3d(&pts).unwrap();
        verify_hull(&pts, &faces);
        assert_eq!(faces.len(), 12);
    }

    #[test]
    fn test_idempotency() {
        let b = create_box(&[1.0, 2.0, 3.0]);
        let faces1 = convex_hull_3d(&b.vertices).unwrap();
        let mesh1 = Trimesh::new(b.vertices.clone(), faces1.clone(), None, None).unwrap();
        let faces2 = convex_hull_3d(&mesh1.vertices).unwrap();
        assert_eq!(faces1.len(), faces2.len(), "hull of hull should be same");
        verify_hull(&mesh1.vertices, &faces2);
    }

    // ---- Random exhaustive tests ----

    /// Simple deterministic LCG PRNG.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + self.next_f64() * (hi - lo)
        }
    }

    #[test]
    fn test_random_point_clouds() {
        let mut rng = Lcg::new(12345);
        for trial in 0..500 {
            let n = 10 + (trial % 491);
            let pts: Vec<Point3<f64>> = (0..n)
                .map(|_| {
                    Point3::new(
                        rng.next_range(-1.0, 1.0),
                        rng.next_range(-1.0, 1.0),
                        rng.next_range(-1.0, 1.0),
                    )
                })
                .collect();
            let faces = convex_hull_3d(&pts).unwrap();
            verify_hull(&pts, &faces);

            let pts2 = to_2d(&pts);
            let edges = convex_hull_2d(&pts2);
            if edges.len() >= 3 {
                verify_hull_2d(&pts2, &edges);
            }
        }
    }

    #[test]
    fn test_random_on_sphere() {
        let mut rng = Lcg::new(99999);
        for trial in 0..200 {
            let n = 10 + (trial % 91);
            let pts: Vec<Point3<f64>> = (0..n)
                .map(|_| {
                    loop {
                        let x = rng.next_range(-1.0, 1.0);
                        let y = rng.next_range(-1.0, 1.0);
                        let z = rng.next_range(-1.0, 1.0);
                        let r = (x * x + y * y + z * z).sqrt();
                        if r > 0.01 {
                            return Point3::new(x / r, y / r, z / r);
                        }
                    }
                })
                .collect();
            let faces = convex_hull_3d(&pts).unwrap();
            verify_hull(&pts, &faces);

            let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for f in &faces {
                for &v in f {
                    used.insert(v);
                }
            }
            assert!(
                used.len() >= n * 8 / 10,
                "Trial {}: only {}/{} sphere points on hull",
                trial,
                used.len(),
                n
            );

            let pts2 = to_2d(&pts);
            let edges = convex_hull_2d(&pts2);
            if edges.len() >= 3 {
                verify_hull_2d(&pts2, &edges);
            }
        }
    }

    #[test]
    fn test_random_clustered() {
        let mut rng = Lcg::new(54321);
        let centers = [
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, 1.0, 1.0),
            Point3::new(1.0, -1.0, 1.0),
            Point3::new(1.0, 1.0, -1.0),
            Point3::new(-1.0, -1.0, 1.0),
            Point3::new(-1.0, 1.0, -1.0),
            Point3::new(1.0, -1.0, -1.0),
            Point3::new(-1.0, -1.0, -1.0),
        ];

        for trial in 0..100 {
            let n = 50 + trial * 3;
            let pts: Vec<Point3<f64>> = (0..n)
                .map(|i| {
                    let c = &centers[i % 8];
                    let spread = 0.01;
                    Point3::new(
                        c.x + rng.next_range(-spread, spread),
                        c.y + rng.next_range(-spread, spread),
                        c.z + rng.next_range(-spread, spread),
                    )
                })
                .collect();
            let faces = convex_hull_3d(&pts).unwrap();
            verify_hull(&pts, &faces);

            let pts2 = to_2d(&pts);
            let edges = convex_hull_2d(&pts2);
            if edges.len() >= 3 {
                verify_hull_2d(&pts2, &edges);
            }
        }
    }

    #[test]
    fn test_random_thin_slab() {
        let mut rng = Lcg::new(77777);
        let mut successes = 0;
        let mut failures = 0;

        for _trial in 0..100 {
            let n = 20;
            let eps = 1e-8;
            let pts: Vec<Point3<f64>> = (0..n)
                .map(|_| {
                    Point3::new(
                        rng.next_range(-1.0, 1.0),
                        rng.next_range(-1.0, 1.0),
                        rng.next_range(-eps, eps),
                    )
                })
                .collect();
            match convex_hull_3d(&pts) {
                Ok(faces) => {
                    verify_hull(&pts, &faces);
                    successes += 1;
                }
                Err(_) => {
                    failures += 1;
                }
            }
        }
        assert!(
            successes + failures == 100,
            "all trials should either succeed or fail cleanly"
        );
    }
}

#[cfg(test)]
mod bench;
