//! Barycentric coordinate conversions.

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

/// Compute barycentric coordinates for a point relative to a triangle.
#[inline]
fn compute_barycentric(
    v0: &Point3<f64>,
    v1: &Point3<f64>,
    v2: &Point3<f64>,
    p: &Point3<f64>,
) -> Vector3<f64> {
    let e0 = v1 - v0;
    let e1 = v2 - v0;
    let e2 = p - v0;

    let d00 = e0.dot(&e0);
    let d01 = e0.dot(&e1);
    let d11 = e1.dot(&e1);
    let d20 = e2.dot(&e0);
    let d21 = e2.dot(&e1);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < f64::EPSILON {
        return Vector3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    Vector3::new(1.0 - v - w, v, w)
}

/// Convert cartesian points to barycentric coordinates.
pub fn points_to_barycentric(
    triangles: &[(Point3<f64>, Point3<f64>, Point3<f64>)],
    points: &[Point3<f64>],
) -> Vec<Vector3<f64>> {
    assert_eq!(triangles.len(), points.len());
    triangles
        .par_iter()
        .zip(points.par_iter())
        .map(|((v0, v1, v2), p)| compute_barycentric(v0, v1, v2, p))
        .collect()
}

/// Convert barycentric coordinates back to cartesian points.
pub fn barycentric_to_points(
    triangles: &[(Point3<f64>, Point3<f64>, Point3<f64>)],
    barycentric: &[Vector3<f64>],
) -> Vec<Point3<f64>> {
    assert_eq!(triangles.len(), barycentric.len());
    triangles
        .par_iter()
        .zip(barycentric.par_iter())
        .map(|((v0, v1, v2), bary)| {
            Point3::from(v0.coords * bary.x + v1.coords * bary.y + v2.coords * bary.z)
        })
        .collect()
}

/// Compute barycentric coordinates using indexed triangles.
pub fn points_to_barycentric_indexed(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    points: &[Point3<f64>],
    face_indices: &[usize],
) -> Vec<Vector3<f64>> {
    assert_eq!(points.len(), face_indices.len());
    points
        .par_iter()
        .zip(face_indices.par_iter())
        .map(|(p, &fi)| {
            let [i0, i1, i2] = faces[fi];
            compute_barycentric(&vertices[i0], &vertices[i1], &vertices[i2], p)
        })
        .collect()
}

/// Check if barycentric coordinates represent a point inside the triangle.
/// If sum == 1 and all components >= 0, then all must be <= 1.
#[inline]
pub fn is_inside(bary: &Vector3<f64>) -> bool {
    bary.x >= 0.0
        && bary.y >= 0.0
        && bary.z >= 0.0
        && (bary.x + bary.y + bary.z - 1.0).abs() < 1e-10
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vertex_barycentric() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);
        let triangles = vec![(v0, v1, v2), (v0, v1, v2), (v0, v1, v2)];
        let points = vec![v0, v1, v2];

        let bary = points_to_barycentric(&triangles, &points);
        assert_relative_eq!(bary[0], Vector3::new(1.0, 0.0, 0.0), epsilon = 1e-10);
        assert_relative_eq!(bary[1], Vector3::new(0.0, 1.0, 0.0), epsilon = 1e-10);
        assert_relative_eq!(bary[2], Vector3::new(0.0, 0.0, 1.0), epsilon = 1e-10);
    }

    #[test]
    fn test_roundtrip() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);
        let triangles = vec![(v0, v1, v2)];
        let original = vec![Point3::new(0.2, 0.3, 0.0)];

        let bary = points_to_barycentric(&triangles, &original);
        let recovered = barycentric_to_points(&triangles, &bary);
        assert_relative_eq!(recovered[0], original[0], epsilon = 1e-10);
    }

    #[test]
    fn test_is_inside() {
        assert!(is_inside(&Vector3::new(0.3, 0.3, 0.4)));
        assert!(is_inside(&Vector3::new(1.0, 0.0, 0.0)));
        assert!(!is_inside(&Vector3::new(-0.1, 0.5, 0.6)));
        assert!(!is_inside(&Vector3::new(0.5, 0.5, 0.5))); // sum > 1
    }
}
