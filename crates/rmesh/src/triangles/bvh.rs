//! Triangle-specific bounding volume hierarchy for spatial queries.
//!
//! Provides fast ray intersection and closest-point queries over triangle meshes.
//! The BVH is built top-down using median splits on triangle centroids along
//! the longest AABB axis. Leaf nodes contain small ranges of triangles.
//!
//! Also supports GPU export via `GpuBvhNode` for use in compute shaders.

use nalgebra::{Point3, Vector3};

use super::closest::closest_point_on_triangle;

/// Maximum triangles per leaf node before splitting.
const MAX_LEAF_SIZE: usize = 4;

/// Result of a ray intersection query.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// Distance along the ray to the hit point.
    pub t: f64,
    /// Index of the hit triangle in the original face array.
    pub face_index: u32,
    /// Hit point in world space.
    pub point: Point3<f64>,
}

/// Result of a closest-point query.
#[derive(Debug, Clone, Copy)]
pub struct ClosestHit {
    /// Squared distance from query point to closest point.
    pub distance_squared: f64,
    /// Index of the closest triangle in the original face array.
    pub face_index: u32,
    /// Closest point on the triangle surface.
    pub point: Point3<f64>,
}

/// Internal BVH node. Either an inner node with two children or a leaf with triangles.
#[derive(Debug, Clone)]
struct BvhNode {
    aabb_min: Point3<f64>,
    aabb_max: Point3<f64>,
    data: NodeData,
}

#[derive(Debug, Clone)]
enum NodeData {
    Inner { left: usize, right: usize },
    Leaf { first_tri: usize, tri_count: usize },
}

/// A bounding volume hierarchy over triangles for fast spatial queries.
///
/// Built once from mesh geometry, then used for repeated ray and closest-point queries.
/// The tree is stored as a flat array in depth-first order for cache efficiency.
#[derive(Debug, Clone)]
pub struct TriangleBvh {
    nodes: Vec<BvhNode>,
    /// Triangle indices reordered so leaf ranges are contiguous.
    tri_indices: Vec<u32>,
}

impl TriangleBvh {
    /// Build a BVH from mesh vertices and faces.
    ///
    /// Uses top-down construction with median splits on the longest AABB axis
    /// of triangle centroids.
    pub fn build(vertices: &[Point3<f64>], faces: &[[usize; 3]]) -> Self {
        if faces.is_empty() {
            return Self {
                nodes: Vec::new(),
                tri_indices: Vec::new(),
            };
        }

        let centroids: Vec<Point3<f64>> = faces
            .iter()
            .map(|[i0, i1, i2]| {
                let v0 = &vertices[*i0];
                let v1 = &vertices[*i1];
                let v2 = &vertices[*i2];
                Point3::from((v0.coords + v1.coords + v2.coords) / 3.0)
            })
            .collect();

        #[allow(clippy::cast_possible_truncation)]
        let mut tri_indices: Vec<u32> = (0..faces.len() as u32).collect();
        let mut nodes = Vec::with_capacity(2 * faces.len());

        Self::build_recursive(
            vertices,
            faces,
            &centroids,
            &mut tri_indices,
            &mut nodes,
            0,
            faces.len(),
        );

        Self { nodes, tri_indices }
    }

    fn build_recursive(
        vertices: &[Point3<f64>],
        faces: &[[usize; 3]],
        centroids: &[Point3<f64>],
        tri_indices: &mut [u32],
        nodes: &mut Vec<BvhNode>,
        start: usize,
        end: usize,
    ) -> usize {
        let count = end - start;
        let aabb = Self::compute_aabb(vertices, faces, &tri_indices[start..end]);

        // Leaf node
        if count <= MAX_LEAF_SIZE {
            let node_idx = nodes.len();
            nodes.push(BvhNode {
                aabb_min: aabb.0,
                aabb_max: aabb.1,
                data: NodeData::Leaf {
                    first_tri: start,
                    tri_count: count,
                },
            });
            return node_idx;
        }

        // Find longest axis of centroid AABB
        let (c_min, c_max) = Self::centroid_bounds(centroids, &tri_indices[start..end]);
        let extent = c_max - c_min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };

        // Sort by centroid along the chosen axis and split at median
        let mid = start + count / 2;
        tri_indices[start..end].select_nth_unstable_by(count / 2, |&a, &b| {
            let ca = centroids[a as usize][axis];
            let cb = centroids[b as usize][axis];
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Reserve slot for this inner node
        let node_idx = nodes.len();
        nodes.push(BvhNode {
            aabb_min: aabb.0,
            aabb_max: aabb.1,
            data: NodeData::Inner { left: 0, right: 0 },
        });

        let left =
            Self::build_recursive(vertices, faces, centroids, tri_indices, nodes, start, mid);
        let right = Self::build_recursive(vertices, faces, centroids, tri_indices, nodes, mid, end);

        nodes[node_idx].data = NodeData::Inner { left, right };
        node_idx
    }

    fn compute_aabb(
        vertices: &[Point3<f64>],
        faces: &[[usize; 3]],
        indices: &[u32],
    ) -> (Point3<f64>, Point3<f64>) {
        let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);

        for &idx in indices {
            let [i0, i1, i2] = faces[idx as usize];
            for &vi in &[i0, i1, i2] {
                let v = &vertices[vi];
                min = min.inf(v);
                max = max.sup(v);
            }
        }
        (min, max)
    }

    fn centroid_bounds(centroids: &[Point3<f64>], indices: &[u32]) -> (Point3<f64>, Point3<f64>) {
        let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);

        for &idx in indices {
            let c = &centroids[idx as usize];
            min = min.inf(c);
            max = max.sup(c);
        }
        (min, max)
    }

    /// Cast a ray and find the nearest triangle intersection.
    ///
    /// Uses Moller-Trumbore ray-triangle intersection. Returns `None` if no hit.
    pub fn trace_ray(
        &self,
        origin: &Point3<f64>,
        direction: &Vector3<f64>,
        vertices: &[Point3<f64>],
        faces: &[[usize; 3]],
    ) -> Option<RayHit> {
        if self.nodes.is_empty() {
            return None;
        }

        let inv_dir = Vector3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z);
        let mut best: Option<RayHit> = None;
        let mut stack = vec![0usize];

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];

            if !Self::ray_aabb(
                origin,
                &inv_dir,
                &node.aabb_min,
                &node.aabb_max,
                best.as_ref().map_or(f64::MAX, |h| h.t),
            ) {
                continue;
            }

            match &node.data {
                NodeData::Leaf {
                    first_tri,
                    tri_count,
                } => {
                    for i in *first_tri..(*first_tri + *tri_count) {
                        let fi = self.tri_indices[i] as usize;
                        let [i0, i1, i2] = faces[fi];
                        if let Some(t) = Self::moller_trumbore(
                            origin,
                            direction,
                            &vertices[i0],
                            &vertices[i1],
                            &vertices[i2],
                        ) && t > 0.0
                            && best.as_ref().is_none_or(|b| t < b.t)
                        {
                            #[allow(clippy::cast_possible_truncation)]
                            let face_idx = fi as u32;
                            best = Some(RayHit {
                                t,
                                face_index: face_idx,
                                point: Point3::from(origin.coords + direction * t),
                            });
                        }
                    }
                }
                NodeData::Inner { left, right } => {
                    stack.push(*right);
                    stack.push(*left);
                }
            }
        }
        best
    }

    /// Find the closest point on the mesh surface to a query point.
    ///
    /// Only considers triangles within `max_dist` of the query point.
    /// Returns `None` if no triangle is within range.
    pub fn closest_point(
        &self,
        query: &Point3<f64>,
        max_dist: f64,
        vertices: &[Point3<f64>],
        faces: &[[usize; 3]],
    ) -> Option<ClosestHit> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut best_dist_sq = max_dist * max_dist;
        let mut best: Option<ClosestHit> = None;
        let mut stack = vec![0usize];

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];

            if Self::point_aabb_dist_sq(query, &node.aabb_min, &node.aabb_max) > best_dist_sq {
                continue;
            }

            match &node.data {
                NodeData::Leaf {
                    first_tri,
                    tri_count,
                } => {
                    for i in *first_tri..(*first_tri + *tri_count) {
                        let fi = self.tri_indices[i] as usize;
                        let [i0, i1, i2] = faces[fi];
                        let cp = closest_point_on_triangle(
                            &vertices[i0],
                            &vertices[i1],
                            &vertices[i2],
                            query,
                        );
                        let d2 = (query - cp).norm_squared();
                        if d2 < best_dist_sq {
                            best_dist_sq = d2;
                            #[allow(clippy::cast_possible_truncation)]
                            let face_idx = fi as u32;
                            best = Some(ClosestHit {
                                distance_squared: d2,
                                face_index: face_idx,
                                point: cp,
                            });
                        }
                    }
                }
                NodeData::Inner { left, right } => {
                    // Visit closer child first
                    let dl = Self::point_aabb_dist_sq(
                        query,
                        &self.nodes[*left].aabb_min,
                        &self.nodes[*left].aabb_max,
                    );
                    let dr = Self::point_aabb_dist_sq(
                        query,
                        &self.nodes[*right].aabb_min,
                        &self.nodes[*right].aabb_max,
                    );
                    if dl < dr {
                        stack.push(*right);
                        stack.push(*left);
                    } else {
                        stack.push(*left);
                        stack.push(*right);
                    }
                }
            }
        }
        best
    }

    /// Moller-Trumbore ray-triangle intersection.
    /// Returns parameter `t` along ray if hit (may be negative for behind-origin hits).
    fn moller_trumbore(
        origin: &Point3<f64>,
        direction: &Vector3<f64>,
        v0: &Point3<f64>,
        v1: &Point3<f64>,
        v2: &Point3<f64>,
    ) -> Option<f64> {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = direction.cross(&edge2);
        let a = edge1.dot(&h);

        if a.abs() < 1e-12 {
            return None;
        }

        let f = 1.0 / a;
        let s = origin - v0;
        let u = f * s.dot(&h);
        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(&edge1);
        let v = f * direction.dot(&q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        Some(f * edge2.dot(&q))
    }

    /// Slab-based ray-AABB intersection test.
    fn ray_aabb(
        origin: &Point3<f64>,
        inv_dir: &Vector3<f64>,
        aabb_min: &Point3<f64>,
        aabb_max: &Point3<f64>,
        max_t: f64,
    ) -> bool {
        let t1 = (aabb_min.x - origin.x) * inv_dir.x;
        let t2 = (aabb_max.x - origin.x) * inv_dir.x;
        let tmin = t1.min(t2);
        let tmax = t1.max(t2);

        let t1 = (aabb_min.y - origin.y) * inv_dir.y;
        let t2 = (aabb_max.y - origin.y) * inv_dir.y;
        let tmin = tmin.max(t1.min(t2));
        let tmax = tmax.min(t1.max(t2));

        let t1 = (aabb_min.z - origin.z) * inv_dir.z;
        let t2 = (aabb_max.z - origin.z) * inv_dir.z;
        let tmin = tmin.max(t1.min(t2));
        let tmax = tmax.min(t1.max(t2));

        tmax >= tmin.max(0.0) && tmin < max_t
    }

    /// Squared distance from a point to an AABB.
    fn point_aabb_dist_sq(p: &Point3<f64>, aabb_min: &Point3<f64>, aabb_max: &Point3<f64>) -> f64 {
        let mut d2 = 0.0;
        for i in 0..3 {
            let v = p[i];
            if v < aabb_min[i] {
                let d = aabb_min[i] - v;
                d2 += d * d;
            } else if v > aabb_max[i] {
                let d = v - aabb_max[i];
                d2 += d * d;
            }
        }
        d2
    }

    /// Export the BVH to GPU-ready format (f32, 32 bytes per node).
    ///
    /// Returns `(nodes, tri_indices)` suitable for uploading to a WGPU storage buffer.
    /// Node layout uses the MSB of `right_child_or_tri_count` as a leaf flag.
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_gpu(&self) -> (Vec<GpuBvhNode>, Vec<u32>) {
        let gpu_nodes: Vec<GpuBvhNode> = self
            .nodes
            .iter()
            .map(|node| match &node.data {
                NodeData::Inner { left, right } => GpuBvhNode {
                    aabb_min: [
                        node.aabb_min.x as f32,
                        node.aabb_min.y as f32,
                        node.aabb_min.z as f32,
                    ],
                    left_child_or_first_tri: *left as u32,
                    aabb_max: [
                        node.aabb_max.x as f32,
                        node.aabb_max.y as f32,
                        node.aabb_max.z as f32,
                    ],
                    right_child_or_tri_count: *right as u32,
                },
                NodeData::Leaf {
                    first_tri,
                    tri_count,
                } => GpuBvhNode {
                    aabb_min: [
                        node.aabb_min.x as f32,
                        node.aabb_min.y as f32,
                        node.aabb_min.z as f32,
                    ],
                    left_child_or_first_tri: *first_tri as u32,
                    aabb_max: [
                        node.aabb_max.x as f32,
                        node.aabb_max.y as f32,
                        node.aabb_max.z as f32,
                    ],
                    // MSB set = leaf node
                    right_child_or_tri_count: (*tri_count as u32) | 0x8000_0000,
                },
            })
            .collect();
        (gpu_nodes, self.tri_indices.clone())
    }
}

/// GPU-ready BVH node, 32 bytes, maps directly to a WGPU storage buffer.
///
/// For inner nodes: `left_child_or_first_tri` = left child index,
/// `right_child_or_tri_count` = right child index (MSB clear).
///
/// For leaf nodes: `left_child_or_first_tri` = first triangle offset,
/// `right_child_or_tri_count` = triangle count with MSB set.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBvhNode {
    pub aabb_min: [f32; 3],
    pub left_child_or_first_tri: u32,
    pub aabb_max: [f32; 3],
    pub right_child_or_tri_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn unit_triangle() -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        (vertices, faces)
    }

    fn unit_box() -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [2, 3, 7],
            [2, 7, 6], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ];
        (vertices, faces)
    }

    #[test]
    fn test_build_single_triangle() {
        let (v, f) = unit_triangle();
        let bvh = TriangleBvh::build(&v, &f);
        assert_eq!(bvh.nodes.len(), 1);
        assert_eq!(bvh.tri_indices.len(), 1);
    }

    #[test]
    fn test_build_empty() {
        let bvh = TriangleBvh::build(&[], &[]);
        assert!(bvh.nodes.is_empty());
        assert!(bvh.tri_indices.is_empty());
    }

    #[test]
    fn test_ray_hit() {
        let (v, f) = unit_triangle();
        let bvh = TriangleBvh::build(&v, &f);

        // Ray pointing down at the triangle
        let origin = Point3::new(0.2, 0.2, 5.0);
        let dir = Vector3::new(0.0, 0.0, -1.0);
        let hit = bvh.trace_ray(&origin, &dir, &v, &f).unwrap();

        assert_relative_eq!(hit.t, 5.0, epsilon = 1e-10);
        assert_eq!(hit.face_index, 0);
        assert_relative_eq!(hit.point.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ray_miss() {
        let (v, f) = unit_triangle();
        let bvh = TriangleBvh::build(&v, &f);

        // Ray pointing away from the triangle
        let origin = Point3::new(0.2, 0.2, 5.0);
        let dir = Vector3::new(0.0, 0.0, 1.0);
        let hit = bvh.trace_ray(&origin, &dir, &v, &f);
        assert!(hit.is_none());
    }

    #[test]
    fn test_closest_point_on_surface() {
        let (v, f) = unit_triangle();
        let bvh = TriangleBvh::build(&v, &f);

        let query = Point3::new(0.2, 0.2, 3.0);
        let hit = bvh.closest_point(&query, 10.0, &v, &f).unwrap();

        assert_relative_eq!(hit.point.x, 0.2, epsilon = 1e-10);
        assert_relative_eq!(hit.point.y, 0.2, epsilon = 1e-10);
        assert_relative_eq!(hit.point.z, 0.0, epsilon = 1e-10);
        assert_relative_eq!(hit.distance_squared, 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_closest_point_beyond_threshold() {
        let (v, f) = unit_triangle();
        let bvh = TriangleBvh::build(&v, &f);

        let query = Point3::new(0.2, 0.2, 100.0);
        let hit = bvh.closest_point(&query, 1.0, &v, &f);
        assert!(hit.is_none());
    }

    #[test]
    fn test_ray_box() {
        let (v, f) = unit_box();
        let bvh = TriangleBvh::build(&v, &f);

        // Ray through center of box
        let origin = Point3::new(0.5, 0.5, -1.0);
        let dir = Vector3::new(0.0, 0.0, 1.0);
        let hit = bvh.trace_ray(&origin, &dir, &v, &f).unwrap();

        assert_relative_eq!(hit.t, 1.0, epsilon = 1e-10);
        assert_relative_eq!(hit.point.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_to_gpu_roundtrip() {
        let (v, f) = unit_box();
        let bvh = TriangleBvh::build(&v, &f);
        let (gpu_nodes, gpu_indices) = bvh.to_gpu();

        assert_eq!(gpu_nodes.len(), bvh.nodes.len());
        assert_eq!(gpu_indices.len(), bvh.tri_indices.len());

        // Each GPU node is 32 bytes
        assert_eq!(std::mem::size_of::<GpuBvhNode>(), 32);

        // Check leaf flag encoding
        for (cpu, gpu) in bvh.nodes.iter().zip(gpu_nodes.iter()) {
            match &cpu.data {
                NodeData::Leaf { tri_count, .. } => {
                    assert!(gpu.right_child_or_tri_count & 0x8000_0000 != 0);
                    let decoded_count = gpu.right_child_or_tri_count & 0x7FFF_FFFF;
                    assert_eq!(decoded_count, *tri_count as u32);
                }
                NodeData::Inner { .. } => {
                    assert!(gpu.right_child_or_tri_count & 0x8000_0000 == 0);
                }
            }
        }
    }
}
