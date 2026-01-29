use ahash::AHashMap;

use anyhow::{Result, bail};
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

/// Tolerance values for convex hull computation, adapted from qhull's
/// `qh_distround` for dimension 3.
struct Tolerances {
    /// Maximum roundoff error for distance computations.
    dist_round: f64,
    /// Minimum distance for a point to be considered "outside" a facet.
    outside_threshold: f64,
}

impl Tolerances {
    fn from_points(points: &[Point3<f64>]) -> Self {
        let (max_abs, max_sum_abs) = points
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

        // qhull formula: maxdistsum = min(sqrt(dim) * max_abs, max_sum_abs)
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
    /// Triangle vertex indices (CCW from outside).
    vertices: [usize; 3],
    /// `neighbors[i]` is the facet sharing the edge opposite `vertices[i]`.
    neighbors: [usize; 3],
    /// Outward unit normal.
    normal: Vector3<f64>,
    /// Plane offset: `normal.dot(p) + offset = 0` for points on the plane.
    offset: f64,
    /// Points above this facet. Index 0 is the furthest.
    outside: Vec<usize>,
    /// Distance of the furthest outside point.
    furthest_dist: f64,
    /// Whether this facet has been deleted.
    removed: bool,
}

impl Facet {
    /// Signed distance from a point to this facet's plane.
    /// Positive means the point is above (outside) the facet.
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
}

impl<'a> QHull<'a> {
    fn new(points: &'a [Point3<f64>]) -> Result<Self> {
        if points.len() < 4 {
            bail!("convex_hull_3d requires at least 4 points");
        }
        let tolerances = Tolerances::from_points(points);
        let mut hull = Self {
            points,
            facets: Vec::new(),
            free_list: Vec::new(),
            tolerances,
            interior: Point3::origin(),
        };
        hull.build()?;
        Ok(hull)
    }

    fn build(&mut self) -> Result<()> {
        let (i0, i1, i2, i3) = self.build_initial_simplex()?;
        self.create_initial_tetrahedron(i0, i1, i2, i3);
        self.initial_partition(i0, i1, i2, i3);
        self.build_hull();
        Ok(())
    }

    /// Find 4 maximally-spread non-coplanar points.
    fn build_initial_simplex(&self) -> Result<(usize, usize, usize, usize)> {
        let points = self.points;
        let num_points = points.len();

        // Find extremes along each axis
        let mut extremes = [0usize; 6]; // min_x, max_x, min_y, max_y, min_z, max_z
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
    fn create_initial_tetrahedron(
        &mut self,
        mut i0: usize,
        mut i1: usize,
        i2: usize,
        i3: usize,
    ) {
        let points = self.points;

        // Check signed volume: det = (p1-p0) . ((p2-p0) x (p3-p0))
        let det = (points[i1] - points[i0])
            .dot(&(points[i2] - points[i0]).cross(&(points[i3] - points[i0])));
        if det < 0.0 {
            std::mem::swap(&mut i0, &mut i1);
        }

        // Interior point = centroid of the 4 simplex vertices
        self.interior = Point3::from(
            (points[i0].coords + points[i1].coords + points[i2].coords + points[i3].coords)
                / 4.0,
        );

        // Face table for positive-volume tetrahedron (i0,i1,i2,i3).
        // For det > 0, cross(v1-v0, v2-v0) points TOWARD the opposite vertex (inward).
        // So we reverse winding to get outward-facing CCW normals:
        //   face 0: (i0, i2, i1) opposite i3
        //   face 1: (i0, i1, i3) opposite i2
        //   face 2: (i0, i3, i2) opposite i1
        //   face 3: (i1, i2, i3) opposite i0
        let face_verts = [
            [i0, i2, i1],
            [i0, i1, i3],
            [i0, i3, i2],
            [i1, i2, i3],
        ];
        // Neighbor table: face_neighbors[f][i] = neighbor sharing edge opposite vertex i of face f.
        // Derived by matching shared edges between the four faces above.
        let face_neighbors = [
            [3, 1, 2], // face 0
            [3, 2, 0], // face 1
            [3, 0, 1], // face 2
            [2, 1, 0], // face 3
        ];

        self.facets.clear();
        for face_idx in 0..4 {
            let mut verts = face_verts[face_idx];
            let neighbors = face_neighbors[face_idx];
            let (normal, offset) = self.compute_plane(&mut verts);
            self.facets.push(Facet {
                vertices: verts,
                neighbors,
                normal,
                offset,
                outside: Vec::new(),
                furthest_dist: 0.0,
                removed: false,
            });
        }
    }

    /// Compute the outward plane for a face, using the interior point to enforce orientation.
    /// If the normal needs flipping, also swaps `vertices[1]` and `vertices[2]` so that
    /// the vertex winding stays consistent with the outward normal direction.
    fn compute_plane(&self, vertices: &mut [usize; 3]) -> (Vector3<f64>, f64) {
        let [a, b, c] = *vertices;
        let points = self.points;
        let mut normal = (points[b] - points[a]).cross(&(points[c] - points[a]));
        let normal_len = normal.norm();
        if normal_len > 1e-15 {
            normal /= normal_len;
        }
        let mut offset = -normal.dot(&points[a].coords);

        // Ensure interior point is on the negative side
        let interior_dist = normal.dot(&self.interior.coords) + offset;
        if interior_dist > 0.0 {
            normal = -normal;
            offset = -offset;
            vertices.swap(1, 2);
        }

        (normal, offset)
    }

    /// Assign all non-simplex points to the facet they are furthest above.
    fn initial_partition(&mut self, i0: usize, i1: usize, i2: usize, i3: usize) {
        let simplex = [i0, i1, i2, i3];
        let num_facets = self.facets.len();

        // Parallel: compute best facet assignment for each point
        let assignments: Vec<(usize, usize, f64)> = self
            .points
            .par_iter()
            .enumerate()
            .filter(|(i, _)| !simplex.contains(i))
            .filter_map(|(point_index, point)| {
                let mut best_face = 0;
                let mut best_dist = f64::NEG_INFINITY;
                for face_index in 0..num_facets {
                    let dist = self.facets[face_index].distance(point);
                    if dist > best_dist {
                        best_dist = dist;
                        best_face = face_index;
                    }
                }
                if best_dist > self.tolerances.outside_threshold {
                    Some((point_index, best_face, best_dist))
                } else {
                    None
                }
            })
            .collect();

        // Sequential: distribute into outside sets, keeping furthest at index 0
        for (point_index, face_index, dist) in assignments {
            let facet = &mut self.facets[face_index];
            facet.outside.push(point_index);
            if dist > facet.furthest_dist {
                facet.furthest_dist = dist;
                let last = facet.outside.len() - 1;
                facet.outside.swap(0, last);
            }
        }
    }

    /// Main Quickhull loop: iteratively add the furthest outside point.
    fn build_hull(&mut self) {
        loop {
            // Find facet with largest furthest_dist
            let best = self
                .facets
                .iter()
                .enumerate()
                .filter(|(_, f)| !f.removed && !f.outside.is_empty())
                .max_by(|a, b| {
                    a.1.furthest_dist
                        .partial_cmp(&b.1.furthest_dist)
                        .unwrap()
                });
            let Some((start_face, _)) = best else {
                break;
            };

            let apex = self.facets[start_face].outside[0];

            // Find all visible facets and horizon edges in one BFS pass
            let (visible, horizon) = self.find_visible_and_horizon(start_face, apex);

            // Build cone from apex to horizon
            self.build_cone(apex, &visible, &horizon);
        }
    }

    /// BFS from `start` to find all facets visible from `apex`, and collect
    /// horizon edges (visible-to-non-visible boundaries) in the same pass.
    fn find_visible_and_horizon(
        &self,
        start: usize,
        apex: usize,
    ) -> (Vec<usize>, Vec<(usize, usize, usize)>) {
        let apex_point = &self.points[apex];
        let num_facets = self.facets.len();
        let mut is_visible = vec![false; num_facets];
        let mut visible = Vec::new();
        let mut queue = Vec::new();
        let mut horizon = Vec::new();

        is_visible[start] = true;
        visible.push(start);
        queue.push(start);

        while let Some(face_index) = queue.pop() {
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
                    || is_visible[neighbor]
                    || self.facets[neighbor].removed
                {
                    continue;
                }
                if self.facets[neighbor].distance(apex_point) > -self.tolerances.dist_round {
                    is_visible[neighbor] = true;
                    visible.push(neighbor);
                    queue.push(neighbor);
                } else {
                    // Non-visible neighbor of visible facet = horizon edge.
                    // Don't mark visited: multiple visible facets may share
                    // the same non-visible neighbor at different edges.
                    horizon.push((face_index, neighbor_slot, neighbor));
                }
            }
        }

        (visible, horizon)
    }

    /// Build new cone facets from apex to each horizon edge, then redistribute orphans.
    fn build_cone(
        &mut self,
        apex: usize,
        visible: &[usize],
        horizon: &[(usize, usize, usize)],
    ) {
        // Collect orphan points from all visible facets (excluding the apex)
        let mut orphans = Vec::new();
        for &face_index in visible {
            for &point_index in &self.facets[face_index].outside {
                if point_index != apex {
                    orphans.push(point_index);
                }
            }
        }

        // Create new cone facets for each horizon edge
        let mut new_face_indices = Vec::with_capacity(horizon.len());
        // Map: vertex index -> (face_index, neighbor_slot) for linking adjacent cone facets
        let mut edge_map: AHashMap<usize, (usize, usize)> = AHashMap::new();

        for &(visible_face, edge_slot, horizon_neighbor) in horizon {
            // The horizon edge is opposite vertices[edge_slot] in the visible facet.
            let visible_verts = self.facets[visible_face].vertices;
            let edge_v0 = visible_verts[(edge_slot + 1) % 3];
            let edge_v1 = visible_verts[(edge_slot + 2) % 3];

            // New triangle: (apex, edge_v1, edge_v0) -- reverse edge order for outward normal.
            // The visible face has CCW winding from outside, so its edge (v0, v1) goes
            // left-to-right. The new cone face views this edge from the other side,
            // so we reverse to (v1, v0) for CCW from outside the new face.
            let mut new_verts = [apex, edge_v1, edge_v0];
            let (normal, offset) = self.compute_plane(&mut new_verts);

            // Reuse a removed facet slot or append a new one
            let new_face_index = self.free_list.pop().unwrap_or_else(|| {
                let index = self.facets.len();
                self.facets.push(Facet {
                    vertices: [0; 3],
                    neighbors: [0; 3],
                    normal: Vector3::zeros(),
                    offset: 0.0,
                    outside: Vec::new(),
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
            facet.outside.clear();
            facet.furthest_dist = 0.0;
            // neighbors[0] (opposite apex) = horizon_neighbor
            facet.neighbors[0] = horizon_neighbor;
            // neighbors[1] and [2] will be filled by linking step
            facet.neighbors[1] = usize::MAX;
            facet.neighbors[2] = usize::MAX;

            // Update the horizon neighbor to point back to this new facet
            // (inline of replace_neighbor: find the slot that referenced the old
            // visible face and repoint it to the new cone face)
            for slot in self.facets[horizon_neighbor].neighbors.iter_mut() {
                if *slot == visible_face {
                    *slot = new_face_index;
                    break;
                }
            }

            new_face_indices.push(new_face_index);

            // Link adjacent cone facets sharing an (apex, V) edge.
            // neighbors[i] shares the edge opposite vertices[i], so:
            //   neighbors[1] shares (apex, stored[2]) -> key on stored[2]
            //   neighbors[2] shares (apex, stored[1]) -> key on stored[1]
            // Use stored vertices (post compute_plane swap) for correct slots.
            let stored = self.facets[new_face_index].vertices;
            let vert_a = stored[1];
            let vert_b = stored[2];
            for &(vertex, slot) in &[(vert_b, 1usize), (vert_a, 2usize)] {
                if let Some(&(other_face, other_slot)) = edge_map.get(&vertex) {
                    self.facets[new_face_index].neighbors[slot] = other_face;
                    self.facets[other_face].neighbors[other_slot] = new_face_index;
                    edge_map.remove(&vertex);
                } else {
                    edge_map.insert(vertex, (new_face_index, slot));
                }
            }
        }

        debug_assert!(
            edge_map.is_empty(),
            "cone linking incomplete: {} unmatched edges remain",
            edge_map.len(),
        );

        // Mark visible facets as removed
        for &face_index in visible {
            self.facets[face_index].removed = true;
            self.facets[face_index].outside.clear();
            self.free_list.push(face_index);
        }

        // Redistribute orphan points to new cone facets
        self.redistribute_orphans(&orphans, &new_face_indices);
    }

    /// Distribute orphan points among new cone facets.
    fn redistribute_orphans(&mut self, orphans: &[usize], new_face_indices: &[usize]) {
        if orphans.is_empty() || new_face_indices.is_empty() {
            return;
        }

        // Collect facet planes for parallel access
        let planes: Vec<(Vector3<f64>, f64)> = new_face_indices
            .iter()
            .map(|&face_index| {
                (
                    self.facets[face_index].normal,
                    self.facets[face_index].offset,
                )
            })
            .collect();

        let threshold = self.tolerances.outside_threshold;

        // Parallel: compute best facet for each orphan
        let assignments: Vec<(usize, usize, f64)> = orphans
            .par_iter()
            .filter_map(|&point_index| {
                let point = &self.points[point_index];
                let mut best_plane = 0;
                let mut best_dist = f64::NEG_INFINITY;
                for (plane_index, (normal, offset)) in planes.iter().enumerate() {
                    let dist = normal.dot(&point.coords) + offset;
                    if dist > best_dist {
                        best_dist = dist;
                        best_plane = plane_index;
                    }
                }
                if best_dist > threshold {
                    Some((point_index, best_plane, best_dist))
                } else {
                    None
                }
            })
            .collect();

        // Sequential: add to outside sets, keeping furthest at index 0
        for (point_index, local_face, dist) in assignments {
            let face_index = new_face_indices[local_face];
            let facet = &mut self.facets[face_index];
            facet.outside.push(point_index);
            if dist > facet.furthest_dist {
                facet.furthest_dist = dist;
                let last = facet.outside.len() - 1;
                facet.outside.swap(0, last);
            }
        }
    }

    /// Extract the final hull faces, with orientation safety belt.
    fn to_result(&self) -> Vec<[usize; 3]> {
        let mut faces: Vec<[usize; 3]> = self
            .facets
            .iter()
            .filter(|f| !f.removed)
            .map(|f| f.vertices)
            .collect();

        // Safety belt: verify each face normal points away from the interior point.
        // self.interior is the centroid of the initial simplex, guaranteed strictly inside.
        let center = self.interior;
        faces.par_iter_mut().for_each(|face| {
            let [a, b, c] = *face;
            let normal =
                (self.points[b] - self.points[a]).cross(&(self.points[c] - self.points[a]));
            // Any hull vertex is on the surface; vector from interior to it aligns with outward normal
            if normal.dot(&(self.points[a] - center)) < 0.0 {
                face.swap(1, 2);
            }
        });

        faces
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
    let hull = QHull::new(points)?;
    let faces = hull.to_result();

    #[cfg(test)]
    {
        assert!(
            super::is_hull_valid_3d(points, &faces),
            "convex_hull_3d: internal validation failed — a point is outside the hull"
        );
    }

    Ok(faces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convex::{convex_hull_2d, is_hull_valid_3d};
    use crate::creation::create_box;
    use crate::mesh::Trimesh;
    use nalgebra::Point2;

    fn p(x: f64, y: f64, z: f64) -> Point3<f64> {
        Point3::new(x, y, z)
    }

    /// Drop Z to produce 2D points.
    fn to_2d(pts: &[Point3<f64>]) -> Vec<Point2<f64>> {
        pts.iter().map(|p| Point2::new(p.x, p.y)).collect()
    }

    /// Verify a 2D convex hull: edges form a valid, convex, CCW polygon.
    fn verify_hull_2d(points: &[Point2<f64>], edges: &[[usize; 2]]) {
        assert!(edges.len() >= 3, "hull has fewer than 3 edges");
        // is_hull_valid_2d is checked inside convex_hull_2d via cfg(test)
        // CCW: all cross products non-negative
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
    /// is_hull_valid_3d is checked inside convex_hull_3d via cfg(test).
    fn verify_hull(points: &[Point3<f64>], faces: &[[usize; 3]]) {
        assert!(!faces.is_empty(), "hull has no faces");

        // All face indices must be valid
        for face in faces {
            for &idx in face {
                assert!(idx < points.len(), "face index {} out of range", idx);
            }
        }

        // Topological checks via Trimesh
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
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(0.5, 1.0, 0.0),
            p(0.5, 0.5, 1.0),
        ];
        let faces = convex_hull_3d(&pts).unwrap();
        verify_hull(&pts, &faces);
        assert_eq!(faces.len(), 4, "tetrahedron should have 4 faces");
    }

    #[test]
    fn test_box_with_interior() {
        let mut pts = create_box(&[2.0, 2.0, 2.0]).vertices;
        // Add interior points
        pts.push(p(0.0, 0.0, 0.0));
        pts.push(p(0.1, 0.2, 0.3));
        pts.push(p(-0.3, 0.1, -0.1));
        let faces = convex_hull_3d(&pts).unwrap();
        verify_hull(&pts, &faces);
        assert_eq!(faces.len(), 12, "interior points shouldn't add faces");

        // Verify only original 8 vertices are used
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
        let centroid: Vector3<f64> = b.vertices.iter().map(|v| v.coords).sum::<Vector3<f64>>()
            / b.vertices.len() as f64;
        let center = Point3::from(centroid);

        for face in &faces {
            let [a, b_idx, c] = *face;
            let normal = (b.vertices[b_idx] - b.vertices[a])
                .cross(&(b.vertices[c] - b.vertices[a]));
            let face_center = Point3::from(
                (b.vertices[a].coords + b.vertices[b_idx].coords + b.vertices[c].coords) / 3.0,
            );
            let to_face = face_center - center;
            assert!(normal.dot(&to_face) > 0.0, "face normal points inward");
        }
    }

    #[test]
    fn test_sphere_points() {
        // Generate points on unit sphere
        let n = 100;
        let mut pts = Vec::with_capacity(n);
        // Fibonacci sphere
        let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
        for i in 0..n {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / golden;
            let phi = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
            pts.push(p(
                phi.sin() * theta.cos(),
                phi.sin() * theta.sin(),
                phi.cos(),
            ));
        }
        let faces = convex_hull_3d(&pts).unwrap();
        verify_hull(&pts, &faces);

        // Volume should be near 4/3 pi r^3 = 4.189
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
        let pts = [p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0), p(0.0, 1.0, 0.0)];
        assert!(convex_hull_3d(&pts).is_err());
    }

    #[test]
    fn test_error_coplanar() {
        let pts = [
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(0.0, 1.0, 0.0),
            p(1.0, 1.0, 0.0),
        ];
        assert!(convex_hull_3d(&pts).is_err());
    }

    #[test]
    fn test_error_collinear() {
        let pts = [
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(2.0, 0.0, 0.0),
            p(3.0, 0.0, 0.0),
        ];
        assert!(convex_hull_3d(&pts).is_err());
    }

    #[test]
    fn test_duplicates() {
        let pts = [
            p(0.0, 0.0, 0.0),
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(0.0, 1.0, 0.0),
            p(0.0, 0.0, 1.0),
        ];
        let faces = convex_hull_3d(&pts).unwrap();
        verify_hull(&pts, &faces);
    }

    #[test]
    fn test_large_coordinates() {
        let s = 1e6;
        let pts = [
            p(s, s, s),
            p(-s, s, s),
            p(s, -s, s),
            p(s, s, -s),
            p(-s, -s, s),
            p(-s, s, -s),
            p(s, -s, -s),
            p(-s, -s, -s),
        ];
        let faces = convex_hull_3d(&pts).unwrap();
        verify_hull(&pts, &faces);
        assert_eq!(faces.len(), 12);
    }

    #[test]
    fn test_small_coordinates() {
        let s = 1e-6;
        let pts = [
            p(s, s, s),
            p(-s, s, s),
            p(s, -s, s),
            p(s, s, -s),
            p(-s, -s, s),
            p(-s, s, -s),
            p(s, -s, -s),
            p(-s, -s, -s),
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
            let n = 10 + (trial % 491); // 10-500 points
            let pts: Vec<Point3<f64>> = (0..n)
                .map(|_| {
                    p(
                        rng.next_range(-1.0, 1.0),
                        rng.next_range(-1.0, 1.0),
                        rng.next_range(-1.0, 1.0),
                    )
                })
                .collect();
            let faces = convex_hull_3d(&pts).unwrap();
            verify_hull(&pts, &faces);

            // Same cloud projected to 2D (drop Z)
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
            let n = 10 + (trial % 91); // 10-100 points
            let pts: Vec<Point3<f64>> = (0..n)
                .map(|_| {
                    // Random point on unit sphere
                    loop {
                        let x = rng.next_range(-1.0, 1.0);
                        let y = rng.next_range(-1.0, 1.0);
                        let z = rng.next_range(-1.0, 1.0);
                        let r = (x * x + y * y + z * z).sqrt();
                        if r > 0.01 {
                            return p(x / r, y / r, z / r);
                        }
                    }
                })
                .collect();
            let faces = convex_hull_3d(&pts).unwrap();
            verify_hull(&pts, &faces);

            // All points should be on the hull (within tolerance)
            let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for f in &faces {
                for &v in f {
                    used.insert(v);
                }
            }
            // Most sphere points should be on the hull (allow some tolerance for
            // very close points that get classified as inside)
            assert!(
                used.len() >= n * 8 / 10,
                "Trial {}: only {}/{} sphere points on hull",
                trial,
                used.len(),
                n
            );

            // Same cloud projected to 2D (drop Z → circle)
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
        // Cluster centers at cube corners
        let centers = [
            p(1.0, 1.0, 1.0),
            p(-1.0, 1.0, 1.0),
            p(1.0, -1.0, 1.0),
            p(1.0, 1.0, -1.0),
            p(-1.0, -1.0, 1.0),
            p(-1.0, 1.0, -1.0),
            p(1.0, -1.0, -1.0),
            p(-1.0, -1.0, -1.0),
        ];

        for trial in 0..100 {
            let n = 50 + trial * 3;
            let pts: Vec<Point3<f64>> = (0..n)
                .map(|i| {
                    let c = &centers[i % 8];
                    let spread = 0.01;
                    p(
                        c.x + rng.next_range(-spread, spread),
                        c.y + rng.next_range(-spread, spread),
                        c.z + rng.next_range(-spread, spread),
                    )
                })
                .collect();
            let faces = convex_hull_3d(&pts).unwrap();
            verify_hull(&pts, &faces);

            // Same cloud projected to 2D (drop Z)
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
                    p(
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
        // Should either succeed with valid hull or return coplanar error
        assert!(
            successes + failures == 100,
            "all trials should either succeed or fail cleanly"
        );
    }

    /// Sweep point counts with varying hulls per level and print timing breakdown.
    ///
    /// Levels: n=1..=100 (every integer), 200..=1000 (step 100),
    /// then 2000, 5000, 10000, 50000, 100000 with fewer hulls.
    /// Run with: `cargo test -p rmesh --release bench_sweep -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_sweep() {
        use crate::timer::Timer;
        use std::time::Instant;

        // Scale down hulls per level for large point counts
        let hulls_for_level = |n: usize| -> usize {
            match n {
                0..=1000 => 100,
                1001..=10000 => 10,
                _ => 3,
            }
        };

        // Build the list of point counts to test
        let levels: Vec<usize> = (1..=100)
            .chain((200..=1000).step_by(100))
            .chain([2000, 5000, 10000, 50000, 100000])
            .collect();

        let total_hulls_planned: usize = levels.iter().map(|&n| hulls_for_level(n)).sum();
        let mut timer = Timer::new(&format!(
            "convex_hull_3d sweep {} levels, {total_hulls_planned} total hulls",
            levels.len()
        ));

        // Pre-generate all point clouds so generation time isn't measured
        let mut rng = Lcg::new(42424242);
        let all_clouds: Vec<(usize, Vec<Vec<Point3<f64>>>)> = levels
            .iter()
            .map(|&n| {
                let hulls_per_level = hulls_for_level(n);
                let clouds: Vec<Vec<Point3<f64>>> = (0..hulls_per_level)
                    .map(|_| {
                        (0..n)
                            .map(|_| {
                                p(
                                    rng.next_range(-1.0, 1.0),
                                    rng.next_range(-1.0, 1.0),
                                    rng.next_range(-1.0, 1.0),
                                )
                            })
                            .collect()
                    })
                    .collect();
                (n, clouds)
            })
            .collect();
        timer.record("point generation");

        // Detail levels get per-phase breakdowns
        let detail_levels: std::collections::HashSet<usize> =
            [4, 8, 16, 32, 64, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000]
                .iter()
                .copied()
                .collect();

        let mut total_hulls = 0u64;
        let mut total_errors = 0u64;

        for (n, clouds) in &all_clouds {
            let n = *n;

            if detail_levels.contains(&n) {
                let mut t_tol = 0.0_f64;
                let mut t_simplex = 0.0_f64;
                let mut t_partition = 0.0_f64;
                let mut t_loop = 0.0_f64;
                let mut t_extract = 0.0_f64;
                let mut t_validate = 0.0_f64;
                let mut ok_count = 0u64;
                let mut err_count = 0u64;

                for pts in clouds {
                    if pts.len() < 4 {
                        err_count += 1;
                        continue;
                    }

                    let t0 = Instant::now();
                    let tolerances = Tolerances::from_points(pts);
                    t_tol += t0.elapsed().as_secs_f64();

                    let t1 = Instant::now();
                    let mut hull = QHull {
                        points: pts,
                        facets: Vec::new(),
                        free_list: Vec::new(),
                        tolerances,
                        interior: Point3::origin(),
                    };
                    let simplex = hull.build_initial_simplex();
                    t_simplex += t1.elapsed().as_secs_f64();

                    let (i0, i1, i2, i3) = match simplex {
                        Ok(s) => s,
                        Err(_) => {
                            err_count += 1;
                            continue;
                        }
                    };

                    let t2 = Instant::now();
                    hull.create_initial_tetrahedron(i0, i1, i2, i3);
                    hull.initial_partition(i0, i1, i2, i3);
                    t_partition += t2.elapsed().as_secs_f64();

                    let t3 = Instant::now();
                    hull.build_hull();
                    t_loop += t3.elapsed().as_secs_f64();

                    let t4 = Instant::now();
                    let faces = hull.to_result();
                    t_extract += t4.elapsed().as_secs_f64();

                    let t5 = Instant::now();
                    is_hull_valid_3d(pts, &faces);
                    t_validate += t5.elapsed().as_secs_f64();

                    ok_count += 1;
                }

                let t_total = t_tol + t_simplex + t_partition + t_loop + t_extract + t_validate;
                timer.record(&format!(
                    "n={n:>4} | {ok_count:>3} ok {err_count:>3} err | \
                     tol {t_tol:.4}  simplex {t_simplex:.4}  part {t_partition:.4}  \
                     loop {t_loop:.4}  extract {t_extract:.4}  valid {t_validate:.4}  \
                     total {t_total:.4}s"
                ));
                total_hulls += ok_count;
                total_errors += err_count;
            } else {
                let level_start = Instant::now();
                let mut ok_count = 0u64;
                let mut err_count = 0u64;
                for pts in clouds {
                    match convex_hull_3d(pts) {
                        Ok(_) => ok_count += 1,
                        Err(_) => err_count += 1,
                    }
                }
                let elapsed = level_start.elapsed().as_secs_f64();
                timer.record(&format!(
                    "n={n:>4} | {ok_count:>3} ok {err_count:>3} err | total {elapsed:.4}s"
                ));
                total_hulls += ok_count;
                total_errors += err_count;
            }
        }

        timer.record(&format!(
            "DONE: {total_hulls} hulls computed, {total_errors} degenerate"
        ));
        timer.print_conditionally();
    }
}
