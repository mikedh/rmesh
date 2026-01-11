//! Closest point on triangle computations.
//!
//! Reference: Real-Time Collision Detection (Ericson), Chapter 5

use nalgebra::Point3;
use rayon::prelude::*;

/// Find the closest point on a triangle to a query point.
pub fn closest_point_on_triangle(
    v0: &Point3<f64>,
    v1: &Point3<f64>,
    v2: &Point3<f64>,
    p: &Point3<f64>,
) -> Point3<f64> {
    let ab = v1 - v0;
    let ac = v2 - v0;
    let ap = p - v0;

    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return *v0;
    }

    let bp = p - v1;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return *v1;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return Point3::from(v0.coords + ab * v);
    }

    let cp = p - v2;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return *v2;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return Point3::from(v0.coords + ac * w);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return Point3::from(v1.coords + (v2 - v1) * w);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    Point3::from(v0.coords + ab * v + ac * w)
}

/// Find closest points on triangles for multiple query points.
pub fn closest_points(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    points: &[Point3<f64>],
    face_indices: &[usize],
) -> Vec<Point3<f64>> {
    assert_eq!(points.len(), face_indices.len());
    points
        .par_iter()
        .zip(face_indices.par_iter())
        .map(|(p, &fi)| {
            let [i0, i1, i2] = faces[fi];
            closest_point_on_triangle(&vertices[i0], &vertices[i1], &vertices[i2], p)
        })
        .collect()
}

/// Compute squared distance from point to triangle.
#[inline]
pub fn squared_distance_to_triangle(
    v0: &Point3<f64>,
    v1: &Point3<f64>,
    v2: &Point3<f64>,
    p: &Point3<f64>,
) -> f64 {
    (p - closest_point_on_triangle(v0, v1, v2, p)).norm_squared()
}

/// Compute distance from point to triangle.
#[inline]
pub fn distance_to_triangle(
    v0: &Point3<f64>,
    v1: &Point3<f64>,
    v2: &Point3<f64>,
    p: &Point3<f64>,
) -> f64 {
    squared_distance_to_triangle(v0, v1, v2, p).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_closest_at_vertices() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        assert_relative_eq!(
            closest_point_on_triangle(&v0, &v1, &v2, &Point3::new(-1.0, -1.0, 0.0)),
            v0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            closest_point_on_triangle(&v0, &v1, &v2, &Point3::new(2.0, -1.0, 0.0)),
            v1,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_closest_inside() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);
        let centroid = Point3::new(1.0 / 3.0, 1.0 / 3.0, 0.0);
        let p = Point3::new(1.0 / 3.0, 1.0 / 3.0, 5.0);

        assert_relative_eq!(
            closest_point_on_triangle(&v0, &v1, &v2, &p),
            centroid,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_distance() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);
        let p = Point3::new(1.0 / 3.0, 1.0 / 3.0, 3.0);

        assert_relative_eq!(
            distance_to_triangle(&v0, &v1, &v2, &p),
            3.0,
            epsilon = 1e-10
        );
    }
}
