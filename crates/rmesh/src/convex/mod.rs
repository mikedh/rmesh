mod hull2d;
mod oriented;
mod qhull;

pub use hull2d::convex_hull_2d;
pub use oriented::{OrientedBoundingBox, oriented_bounding_box};
pub use qhull::convex_hull_3d;

/// Check that every input point is on or inside all hull edges.
///
/// For each edge `[a, b]` in a CCW polygon, the outward normal is
/// `(b - a)` rotated 90 degrees clockwise. Every point must have
/// non-positive dot product with that outward normal (relative to `a`).
#[cfg(test)]
pub(crate) fn is_hull_valid_2d(points: &[nalgebra::Point2<f64>], edges: &[[usize; 2]]) -> bool {
    for &[a, b] in edges {
        let edge = points[b] - points[a];
        let nx = edge.y;
        let ny = -edge.x;
        for p in points {
            let dx = p.x - points[a].x;
            let dy = p.y - points[a].y;
            let d = nx * dx + ny * dy;
            if d > 1e-10 {
                return false;
            }
        }
    }
    true
}

/// Check that every input point is on or inside all hull faces.
///
/// For each face `[a, b, c]` with outward CCW winding, the outward normal is
/// `(b - a) x (c - a)`. Every point must satisfy `n · p + offset <= tol`.
///
/// Uses rayon to parallelize across faces and precomputes plane offsets
/// to avoid per-point subtraction in the inner loop.
#[cfg(test)]
pub(crate) fn is_hull_valid_3d(points: &[nalgebra::Point3<f64>], faces: &[[usize; 3]]) -> bool {
    use rayon::prelude::*;

    // Precompute plane normals and offsets for all faces in parallel
    let planes: Vec<(nalgebra::Vector3<f64>, f64, f64)> = faces
        .par_iter()
        .filter_map(|&[a, b, c]| {
            let ab = points[b] - points[a];
            let ac = points[c] - points[a];
            let n = ab.cross(&ac);
            let n_norm = n.norm();
            if n_norm < 1e-15 {
                return None;
            }
            // plane equation: n · p + offset = 0
            let offset = -n.dot(&points[a].coords);
            let threshold = 1e-10 * n_norm;
            Some((n, offset, threshold))
        })
        .collect();

    // Check all faces in parallel: every point must be behind every face
    planes.par_iter().all(|(n, offset, threshold)| {
        points
            .iter()
            .all(|p| n.dot(&p.coords) + offset <= *threshold)
    })
}
