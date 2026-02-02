use nalgebra::{Matrix4, Point2, Point3, Vector2, Vector3};
use rayon::prelude::*;

use crate::convex::convex_hull_2d;
use crate::creation::perpendicular;

/// An oriented bounding box defined by a 4x4 transform and axis-aligned extents.
///
/// The `transform` matrix encodes the rotation and translation of the box.
/// Its upper-left 3x3 submatrix contains the OBB axes as columns, and the
/// translation column is the box center in world space.
/// The `extents` are the full sizes along each local axis.
#[derive(Debug, Clone)]
pub struct OrientedBoundingBox {
    /// Rotation + translation: columns of the upper-left 3x3 are the OBB axes,
    /// the translation column is the box center in world space.
    pub transform: Matrix4<f64>,
    /// Full size along each local axis.
    pub extents: [f64; 3],
}

impl OrientedBoundingBox {
    /// Volume of the oriented bounding box.
    pub fn volume(&self) -> f64 {
        self.extents[0] * self.extents[1] * self.extents[2]
    }

    /// Create a `Trimesh` representing this OBB as a box mesh in world space.
    pub fn to_trimesh(&self) -> crate::mesh::Trimesh {
        let mut mesh = crate::creation::create_box(&self.extents);
        // Transform each vertex from local OBB space to world space
        for v in &mut mesh.vertices {
            let h = self.transform * v.to_homogeneous();
            *v = Point3::from_homogeneous(h).unwrap();
        }
        mesh
    }
}

/// Internal result from 2D rotating calipers.
struct MinRect2D {
    /// Rotation angle of the rectangle (radians).
    angle: f64,
    /// Center of the rectangle in 2D.
    center: Point2<f64>,
    /// Width and height of the rectangle.
    extents: [f64; 2],
}

/// Extract ordered vertices from a convex hull 2D edge loop.
///
/// `convex_hull_2d` returns CCW-ordered edges `[a, b]`. This extracts the
/// unique vertex sequence in order.
fn hull_edge_loop_to_vertices(points: &[Point2<f64>], edges: &[[usize; 2]]) -> Vec<Point2<f64>> {
    edges.iter().map(|e| points[e[0]]).collect()
}

/// Compute the minimum-area bounding rectangle of a CCW convex polygon
/// using the O(h) rotating calipers algorithm.
///
/// `hull_verts` must be a CCW-ordered convex polygon with at least 3 vertices.
fn min_area_rectangle_2d(hull_verts: &[Point2<f64>]) -> MinRect2D {
    let h = hull_verts.len();
    assert!(h >= 3, "need at least 3 hull vertices");

    // Find initial extreme indices: bottom (min y), right (max x), top (max y), left (min x).
    let mut bottom = 0usize;
    let mut right = 0usize;
    let mut top = 0usize;
    let mut left = 0usize;
    for i in 1..h {
        if hull_verts[i].y < hull_verts[bottom].y
            || (hull_verts[i].y == hull_verts[bottom].y && hull_verts[i].x < hull_verts[bottom].x)
        {
            bottom = i;
        }
        if hull_verts[i].x > hull_verts[right].x
            || (hull_verts[i].x == hull_verts[right].x && hull_verts[i].y < hull_verts[right].y)
        {
            right = i;
        }
        if hull_verts[i].y > hull_verts[top].y
            || (hull_verts[i].y == hull_verts[top].y && hull_verts[i].x > hull_verts[top].x)
        {
            top = i;
        }
        if hull_verts[i].x < hull_verts[left].x
            || (hull_verts[i].x == hull_verts[left].x && hull_verts[i].y > hull_verts[left].y)
        {
            left = i;
        }
    }

    let next = |i: usize| (i + 1) % h;

    let mut best_area = f64::INFINITY;
    let mut best_angle = 0.0;
    let mut best_center = Point2::new(0.0, 0.0);
    let mut best_extents = [0.0, 0.0];

    // Iterate over each edge starting from bottom, using the edge vertex as bottom support.
    let mut r = right;
    let mut t = top;
    let mut l = left;

    for step in 0..h {
        let i = (bottom + step) % h;
        let j = next(i);
        let edge = hull_verts[j] - hull_verts[i];
        let edge_len = edge.norm();
        if edge_len < 1e-15 {
            continue;
        }
        let dir = edge / edge_len;
        // perp = 90° CCW rotation of dir (points "up" relative to edge)
        let perp = Vector2::new(-dir.y, dir.x);

        // Advance right: maximize projection along dir
        while (hull_verts[next(r)] - hull_verts[r]).dot(&dir) > 0.0 {
            r = next(r);
        }
        // Advance top: maximize projection along perp
        while (hull_verts[next(t)] - hull_verts[t]).dot(&perp) > 0.0 {
            t = next(t);
        }
        // Advance left: maximize projection along -dir (minimize along dir)
        while (hull_verts[next(l)] - hull_verts[l]).dot(&dir) < 0.0 {
            l = next(l);
        }

        // Compute rectangle from four support projections
        let proj_bottom = hull_verts[i].coords.dot(&perp);
        let proj_top = hull_verts[t].coords.dot(&perp);
        let proj_right = hull_verts[r].coords.dot(&dir);
        let proj_left = hull_verts[l].coords.dot(&dir);

        let width = proj_right - proj_left;
        let height = proj_top - proj_bottom;
        let area = width * height;

        if area < best_area {
            best_area = area;
            best_angle = dir.y.atan2(dir.x);
            let center_d = (proj_left + proj_right) / 2.0;
            let center_p = (proj_bottom + proj_top) / 2.0;
            best_center = Point2::new(
                dir.x * center_d + perp.x * center_p,
                dir.y * center_d + perp.y * center_p,
            );
            best_extents = [width, height];
        }
    }

    MinRect2D {
        angle: best_angle,
        center: best_center,
        extents: best_extents,
    }
}

/// Internal candidate evaluation result.
struct CandidateResult {
    volume: f64,
    transform: Matrix4<f64>,
    extents: [f64; 3],
}

/// Evaluate a single candidate projection direction for the 3D OBB.
fn evaluate_candidate(vertices: &[Point3<f64>], w: Vector3<f64>) -> CandidateResult {
    // Build orthonormal basis: u, v in the plane perpendicular to w
    let u = perpendicular(&w).normalize();
    let v = w.cross(&u);

    // Project all vertices to 2D (u, v plane) and track w-extent
    let mut w_min = f64::INFINITY;
    let mut w_max = f64::NEG_INFINITY;
    let projected: Vec<Point2<f64>> = vertices
        .iter()
        .map(|p| {
            let pw = p.coords.dot(&w);
            if pw < w_min {
                w_min = pw;
            }
            if pw > w_max {
                w_max = pw;
            }
            Point2::new(p.coords.dot(&u), p.coords.dot(&v))
        })
        .collect();

    let height = w_max - w_min;
    let w_center = (w_min + w_max) / 2.0;

    // 2D convex hull of projected points
    let edges = convex_hull_2d(&projected);
    if edges.len() < 3 {
        // Degenerate projection (all points collinear along this direction)
        return CandidateResult {
            volume: f64::INFINITY,
            transform: Matrix4::identity(),
            extents: [0.0; 3],
        };
    }

    let hull_verts = hull_edge_loop_to_vertices(&projected, &edges);
    let rect = min_area_rectangle_2d(&hull_verts);

    let volume = rect.extents[0] * rect.extents[1] * height;

    // Build the OBB transform: rotate the rectangle's local axes back to 3D
    let cos_a = rect.angle.cos();
    let sin_a = rect.angle.sin();
    // Rectangle's local X axis in the (u, v) plane
    let rect_u = u * cos_a + v * sin_a;
    // Rectangle's local Y axis in the (u, v) plane
    let rect_v = -u * sin_a + v * cos_a;

    // Center in 3D
    let center = u * rect.center.x + v * rect.center.y + w * w_center;

    // Build 4x4 transform with OBB axes as columns
    let transform = Matrix4::new(
        rect_u.x, rect_v.x, w.x, center.x, rect_u.y, rect_v.y, w.y, center.y, rect_u.z, rect_v.z,
        w.z, center.z, 0.0, 0.0, 0.0, 1.0,
    );

    CandidateResult {
        volume,
        transform,
        extents: [rect.extents[0], rect.extents[1], height],
    }
}

/// Compute the minimum-volume oriented bounding box for a set of 3D points.
///
/// Collects unique face normals as candidate projection directions (plus
/// edge-pair cross products for small hulls). For large hulls the candidate
/// set is capped at `MAX_CANDIDATES` via uniform subsampling. Each candidate
/// is evaluated in parallel with rayon using O(V log V) rotating calipers.
///
/// Parameters
/// ----------
/// vertices
///   The vertices of the convex hull.
/// faces
///   The triangular faces of the convex hull (indices into `vertices`).
///
/// Returns
/// -------
/// The minimum-volume oriented bounding box.
pub fn oriented_bounding_box(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
) -> OrientedBoundingBox {
    const MAX_CANDIDATES: usize = 500;
    const EDGE_CROSS_LIMIT: usize = 100;

    let eps = 1e-10;

    let discretize = |v: Vector3<f64>| -> (i64, i64, i64) {
        let scale = 1_000_000.0;
        let c = canonicalize_direction(v);
        (
            (c.x * scale).round() as i64,
            (c.y * scale).round() as i64,
            (c.z * scale).round() as i64,
        )
    };

    let mut seen = std::collections::HashSet::new();
    let mut candidates: Vec<Vector3<f64>> = Vec::new();

    // Collect unique face normals
    for &[a, b, c] in faces {
        let ab = vertices[b] - vertices[a];
        let ac = vertices[c] - vertices[a];
        let n = ab.cross(&ac);
        let len = n.norm();
        if len < eps {
            continue;
        }
        let n = n / len;
        let key = discretize(n);
        if seen.insert(key) {
            candidates.push(canonicalize_direction(n));
        }
    }

    // Collect unique edge directions
    let mut edge_dirs: Vec<Vector3<f64>> = Vec::new();
    let mut seen_edges = std::collections::HashSet::new();
    let mut seen_dirs = std::collections::HashSet::new();
    for &[a, b, c] in faces {
        for &(i, j) in &[(a, b), (b, c), (c, a)] {
            let key = if i < j { (i, j) } else { (j, i) };
            if !seen_edges.insert(key) {
                continue;
            }
            let d = vertices[j] - vertices[i];
            let len = d.norm();
            if len < eps {
                continue;
            }
            let d = d / len;
            let dir_key = discretize(d);
            if seen_dirs.insert(dir_key) {
                edge_dirs.push(canonicalize_direction(d));
            }
        }
    }

    // For small hulls, also add edge-pair cross products for exact results
    // (needed for cases like tetrahedra where the optimal OBB is edge-flush)
    if edge_dirs.len() <= EDGE_CROSS_LIMIT {
        for i in 0..edge_dirs.len() {
            for j in (i + 1)..edge_dirs.len() {
                let cross = edge_dirs[i].cross(&edge_dirs[j]);
                let len = cross.norm();
                if len < eps {
                    continue;
                }
                let c = cross / len;
                let key = discretize(c);
                if seen.insert(key) {
                    candidates.push(canonicalize_direction(c));
                }
            }
        }
    }

    // If we have too many candidates, subsample uniformly
    if candidates.len() > MAX_CANDIDATES {
        let step = candidates.len() as f64 / MAX_CANDIDATES as f64;
        candidates = (0..MAX_CANDIDATES)
            .map(|i| candidates[(i as f64 * step) as usize])
            .collect();
    }

    // Evaluate all candidates in parallel
    let best = candidates
        .par_iter()
        .map(|&w| evaluate_candidate(vertices, w))
        .min_by(|a, b| a.volume.partial_cmp(&b.volume).unwrap())
        .unwrap();

    OrientedBoundingBox {
        transform: best.transform,
        extents: best.extents,
    }
}

/// Canonicalize a unit direction so that parallel and antiparallel vectors
/// map to the same representative. We pick the direction where the first
/// significant component is positive.
fn canonicalize_direction(d: Vector3<f64>) -> Vector3<f64> {
    let eps = 1e-12;
    for &val in &[d.x, d.y, d.z] {
        if val > eps {
            return d;
        }
        if val < -eps {
            return -d;
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{create_box, create_tetrahedron};
    use approx::assert_relative_eq;
    use nalgebra::Rotation3;

    // ---- 2D Rotating Calipers Tests ----

    #[test]
    fn test_min_rect_unit_square() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let edges = convex_hull_2d(&pts);
        let hull = hull_edge_loop_to_vertices(&pts, &edges);
        let rect = min_area_rectangle_2d(&hull);
        assert_relative_eq!(rect.extents[0] * rect.extents[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_min_rect_rotated_rectangle() {
        // 1x3 rectangle rotated 30 degrees
        let angle = 30.0_f64.to_radians();
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let hw = 0.5;
        let hh = 1.5;
        let corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)];
        let pts: Vec<Point2<f64>> = corners
            .iter()
            .map(|&(x, y)| Point2::new(x * cos_a - y * sin_a, x * sin_a + y * cos_a))
            .collect();
        let edges = convex_hull_2d(&pts);
        let hull = hull_edge_loop_to_vertices(&pts, &edges);
        let rect = min_area_rectangle_2d(&hull);
        assert_relative_eq!(rect.extents[0] * rect.extents[1], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_min_rect_equilateral_triangle() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 3.0_f64.sqrt() / 2.0),
        ];
        let edges = convex_hull_2d(&pts);
        let hull = hull_edge_loop_to_vertices(&pts, &edges);
        let rect = min_area_rectangle_2d(&hull);
        let area = rect.extents[0] * rect.extents[1];
        // AABB area would be 1.0 * sqrt(3)/2 = 0.866
        // Min rect area for equilateral triangle should be <= AABB area
        let aabb_area = 1.0 * (3.0_f64.sqrt() / 2.0);
        assert!(area <= aabb_area + 1e-10);
    }

    // ---- 3D OBB Tests ----

    #[test]
    fn test_obb_axis_aligned_box() {
        let mesh = create_box(&[1.0, 2.0, 3.0]);
        let hull_faces = crate::convex::convex_hull_3d(&mesh.vertices).unwrap();
        let obb = oriented_bounding_box(&mesh.vertices, &hull_faces);
        assert_relative_eq!(obb.volume(), 6.0, epsilon = 1e-6);
        let mut ext = obb.extents;
        ext.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_relative_eq!(ext[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(ext[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(ext[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_obb_rotated_box() {
        let mesh = create_box(&[1.0, 2.0, 3.0]);
        let rot = Rotation3::from_euler_angles(0.3, 0.7, 1.1);
        let rotated_verts: Vec<Point3<f64>> = mesh.vertices.iter().map(|v| rot * v).collect();
        let hull_faces = crate::convex::convex_hull_3d(&rotated_verts).unwrap();
        let obb = oriented_bounding_box(&rotated_verts, &hull_faces);
        assert_relative_eq!(obb.volume(), 6.0, epsilon = 1e-4);
        let mut ext = obb.extents;
        ext.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_relative_eq!(ext[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(ext[1], 2.0, epsilon = 1e-4);
        assert_relative_eq!(ext[2], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_obb_tetrahedron() {
        let edge = 1.0;
        let tet = create_tetrahedron(edge);
        let hull_faces = crate::convex::convex_hull_3d(&tet.vertices).unwrap();
        let obb = oriented_bounding_box(&tet.vertices, &hull_faces);

        // Expected OBB volume for regular tetrahedron: edge^3 * sqrt(2) / 4
        let expected_volume = edge.powi(3) * std::f64::consts::SQRT_2 / 4.0;
        assert_relative_eq!(obb.volume(), expected_volume, epsilon = 1e-6);

        // Verify OBB axes are NOT parallel to any face normal
        // (the optimal box is edge-flush, not face-flush)
        let face_normals: Vec<Vector3<f64>> = hull_faces
            .iter()
            .filter_map(|&[a, b, c]| {
                let n =
                    (tet.vertices[b] - tet.vertices[a]).cross(&(tet.vertices[c] - tet.vertices[a]));
                let len = n.norm();
                if len > 1e-15 { Some(n / len) } else { None }
            })
            .collect();

        // Extract OBB axes from transform columns
        for col in 0..3 {
            let axis = Vector3::new(
                obb.transform[(0, col)],
                obb.transform[(1, col)],
                obb.transform[(2, col)],
            );
            for fn_vec in &face_normals {
                let dot = axis.dot(fn_vec).abs();
                // Should NOT be parallel (dot ~= 1.0)
                assert!(dot < 0.99, "OBB axis should not be parallel to face normal");
            }
        }
    }

    #[test]
    fn test_obb_volume_bounds() {
        // OBB volume should be between hull volume and AABB volume
        let mesh = create_box(&[1.0, 2.0, 3.0]);
        let rot = Rotation3::from_euler_angles(0.5, 0.8, 1.2);
        let rotated_verts: Vec<Point3<f64>> = mesh.vertices.iter().map(|v| rot * v).collect();
        let hull_faces = crate::convex::convex_hull_3d(&rotated_verts).unwrap();
        let hull_mesh =
            crate::mesh::Trimesh::new(rotated_verts.clone(), hull_faces.clone(), None, None)
                .unwrap();
        let hull_volume = hull_mesh.volume();

        let obb = oriented_bounding_box(&rotated_verts, &hull_faces);

        // AABB
        let (mut min_p, mut max_p) = (rotated_verts[0], rotated_verts[0]);
        for v in &rotated_verts {
            min_p = min_p.inf(v);
            max_p = max_p.sup(v);
        }
        let aabb_volume = (max_p.x - min_p.x) * (max_p.y - min_p.y) * (max_p.z - min_p.z);

        assert!(
            hull_volume <= obb.volume() + 1e-10,
            "hull volume {} should be <= OBB volume {}",
            hull_volume,
            obb.volume()
        );
        assert!(
            obb.volume() <= aabb_volume + 1e-10,
            "OBB volume {} should be <= AABB volume {}",
            obb.volume(),
            aabb_volume
        );
    }

    #[test]
    fn test_obb_sphere_approximately_cubic() {
        // Points on a sphere should give an approximately cubic OBB
        let n = 200;
        let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let pts: Vec<Point3<f64>> = (0..n)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / golden;
                let phi = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
                Point3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos())
            })
            .collect();

        let hull_faces = crate::convex::convex_hull_3d(&pts).unwrap();
        let obb = oriented_bounding_box(&pts, &hull_faces);

        let mut ext = obb.extents;
        ext.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ratio = ext[2] / ext[0];
        assert!(
            ratio < 1.15,
            "sphere OBB extent ratio {} should be < 1.15",
            ratio
        );
    }

    #[test]
    fn test_obb_to_trimesh() {
        let mesh = create_box(&[1.0, 2.0, 3.0]);
        let hull_faces = crate::convex::convex_hull_3d(&mesh.vertices).unwrap();
        let obb = oriented_bounding_box(&mesh.vertices, &hull_faces);
        let obb_mesh = obb.to_trimesh();
        assert_eq!(obb_mesh.vertices.len(), 8);
        assert_eq!(obb_mesh.faces.len(), 12);
        assert_relative_eq!(obb_mesh.volume(), 6.0, epsilon = 1e-6);
    }

    /// Benchmark OBB on icospheres of increasing face count.
    ///
    /// Run with: `cargo test -p rmesh --release bench_obb -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_obb_icosphere() {
        use crate::creation::create_icosphere;
        use crate::timer::Timer;

        for subdivisions in 1..=6 {
            let mut timer = Timer::new(&format!("OBB icosphere sub={subdivisions}"));

            let sphere = create_icosphere(1.0, subdivisions);
            timer.record(&format!("create icosphere ({} faces)", sphere.faces.len()));

            let hull = sphere.convex_hull();
            timer.record(&format!(
                "convex hull ({} verts, {} faces)",
                hull.vertices.len(),
                hull.faces.len()
            ));

            let obb = oriented_bounding_box(&hull.vertices, &hull.faces);
            timer.record(&format!(
                "OBB (vol={:.4}, extents=[{:.3}, {:.3}, {:.3}])",
                obb.volume(),
                obb.extents[0],
                obb.extents[1],
                obb.extents[2],
            ));

            timer.print_conditionally();
        }
    }
}
