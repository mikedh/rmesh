//! Mesh projection onto a plane at multiple levels.
//!
//! Projects a triangle mesh onto a 2D plane using an incremental
//! top-to-bottom polygon union strategy. Faces are sorted by their
//! maximum dot product with the plane normal, then swept from
//! highest to lowest level, accumulating projected polygons.

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::solver::{Precision, Solver};
use i_overlay::float::overlay::OverlayOptions;
use i_overlay::float::simplify::SimplifyShape;
use nalgebra::{Matrix4, Point2, Point3, Vector3};
use rayon::prelude::*;

use crate::path::Polygon2D;

type Contour = Vec<[f64; 2]>;
type Shape = Vec<Contour>;

/// Tolerance for classifying a vertex as on/above the clipping plane.
const PLANE_TOL: f64 = 1e-12;

/// 2D cross product of triangle vertices. Positive = CCW.
#[inline]
fn cross_2d(a: &Point2<f64>, b: &Point2<f64>, c: &Point2<f64>) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Compute signed distance of each vertex to a plane defined by `normal` and `origin`.
///
/// `dots[i] = vertices[i] . normal - origin . normal`
///
/// Positive values are on the normal side, negative on the opposite.
/// The inner loop is 3 mul + 2 add + 1 sub and auto-vectorizes with LLVM.
pub fn vertex_dots(
    vertices: &[Point3<f64>],
    normal: &Vector3<f64>,
    origin: &Point3<f64>,
) -> Vec<f64> {
    let origin_dot = origin.coords.dot(normal);
    vertices
        .par_iter()
        .map(|v| v.coords.dot(normal) - origin_dot)
        .collect()
}

/// Project a mesh onto a plane at multiple levels.
///
/// Returns one `Option<Vec<Polygon2D>>` per level (same order as input).
/// Levels where the plane doesn't intersect any geometry return `None`.
///
/// # Arguments
/// * `faces` - Triangle face indices (into `dots` and `vertices`)
/// * `dots` - Pre-computed signed distances from `section::vertex_dots()`
/// * `vertices` - Pre-computed 2D projections from `Plane::to_2d()`
/// * `levels` - Height offsets along the normal to project at
pub fn project(
    faces: &[[usize; 3]],
    dots: &[f64],
    vertices: &[Point2<f64>],
    levels: &[f64],
    to_3d: Option<Matrix4<f64>>,
) -> Vec<Option<Vec<Polygon2D>>> {
    if faces.is_empty() || levels.is_empty() {
        return levels.iter().map(|_| None).collect();
    }

    // Per-face dot ranges (min, max)
    let face_ranges: Vec<(f64, f64)> = faces
        .par_iter()
        .map(|f| {
            let (d0, d1, d2) = (dots[f[0]], dots[f[1]], dots[f[2]]);
            (d0.min(d1).min(d2), d0.max(d1).max(d2))
        })
        .collect();

    // Faces sorted by max_dot descending
    let mut face_order: Vec<usize> = (0..faces.len()).collect();
    face_order.sort_unstable_by(|&a, &b| {
        face_ranges[b]
            .1
            .partial_cmp(&face_ranges[a].1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Levels sorted descending, tracking original indices
    let mut level_order: Vec<(usize, f64)> = levels.iter().copied().enumerate().collect();
    level_order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let solver = Solver::with_precision(Precision::MEDIUM_HIGH);
    let options = OverlayOptions {
        min_output_area: 1e-20,
        ..Default::default()
    };

    let mut results: Vec<Option<Vec<Polygon2D>>> = vec![None; levels.len()];
    let mut cursor: usize = 0;
    let mut partial_faces: Vec<usize> = Vec::new();
    let mut accumulated: Vec<Shape> = Vec::new();

    for &(orig_idx, level) in &level_order {
        let mut new_contours: Vec<Contour> = Vec::new();

        // 1. Add newly entering faces (max_dot >= level, not yet seen)
        while cursor < face_order.len() && face_ranges[face_order[cursor]].1 >= level {
            let fi = face_order[cursor];
            cursor += 1;

            if face_ranges[fi].0 >= level {
                new_contours.extend(face_contour(faces[fi], vertices));
            } else {
                new_contours.extend(clip_face(faces[fi], dots, vertices, level));
                partial_faces.push(fi);
            }
        }

        // 2. Re-clip partial faces from higher levels
        let mut still_partial = Vec::new();
        for fi in partial_faces.drain(..) {
            if face_ranges[fi].0 >= level {
                new_contours.extend(face_contour(faces[fi], vertices));
            } else {
                new_contours.extend(clip_face(faces[fi], dots, vertices, level));
                still_partial.push(fi);
            }
        }
        partial_faces = still_partial;

        // 3. Union new contours into accumulated polygon
        if !new_contours.is_empty() {
            let slab: Vec<Shape> =
                new_contours.simplify_shape_custom(FillRule::Positive, options, solver);
            if !slab.is_empty() {
                if accumulated.is_empty() {
                    accumulated = slab;
                } else {
                    accumulated.extend(slab);
                    let flat: Vec<Contour> = accumulated.into_iter().flatten().collect();
                    accumulated = flat.simplify_shape_custom(FillRule::Positive, options, solver);
                }
            }
        }

        // 4. Convert accumulated shapes to Polygon2D
        //    Adjust to_3d so (x, y, 0) maps to the plane at this level,
        //    i.e. shift the translation by level * normal (column 2 of to_3d).
        if !accumulated.is_empty() {
            let level_to_3d = to_3d.map(|base| {
                let mut m = base;
                m[(0, 3)] += level * m[(0, 2)];
                m[(1, 3)] += level * m[(1, 2)];
                m[(2, 3)] += level * m[(2, 2)];
                m
            });
            let polygons = shapes_to_polygons(&accumulated, level_to_3d);
            if !polygons.is_empty() {
                results[orig_idx] = Some(polygons);
            }
        }
    }

    results
}

/// Full 2D triangle as a CCW contour. None if degenerate.
fn face_contour(face: [usize; 3], vertices: &[Point2<f64>]) -> Option<Contour> {
    let (a, b, c) = (vertices[face[0]], vertices[face[1]], vertices[face[2]]);
    let cross = cross_2d(&a, &b, &c);
    if cross.abs() < 1e-30 {
        return None;
    }
    let mut tri = vec![[a.x, a.y], [b.x, b.y], [c.x, c.y]];
    if cross < 0.0 {
        tri.reverse();
    }
    Some(tri)
}

/// Clip a triangle to the half-space above `level`, returned as a CCW contour.
fn clip_face(
    face: [usize; 3],
    dots: &[f64],
    vertices: &[Point2<f64>],
    level: f64,
) -> Option<Contour> {
    let d = [
        dots[face[0]] - level,
        dots[face[1]] - level,
        dots[face[2]] - level,
    ];
    let inside = [d[0] > -PLANE_TOL, d[1] > -PLANE_TOL, d[2] > -PLANE_TOL];
    let count = inside.iter().filter(|&&v| v).count();

    if count == 0 {
        return None;
    }
    if count == 3 {
        return face_contour(face, vertices);
    }

    // Rotate so the odd-one-out vertex is at index 0.
    let r = if count == 1 {
        inside.iter().position(|&v| v).unwrap()
    } else {
        inside.iter().position(|&v| !v).unwrap()
    };
    let p = [
        vertices[face[r]],
        vertices[face[(r + 1) % 3]],
        vertices[face[(r + 2) % 3]],
    ];
    let d = [d[r], d[(r + 1) % 3], d[(r + 2) % 3]];
    let cross = cross_2d(&p[0], &p[1], &p[2]);
    if cross.abs() < 1e-30 {
        return None;
    }

    let lerp = |a: usize, b: usize| -> [f64; 2] {
        let t = -d[a] / (d[b] - d[a]);
        [
            p[a].x + t * (p[b].x - p[a].x),
            p[a].y + t * (p[b].y - p[a].y),
        ]
    };
    let pt = |i: usize| -> [f64; 2] { [p[i].x, p[i].y] };

    // count==1: vertex 0 inside, 1+2 outside → clipped triangle
    // count==2: vertex 0 outside, 1+2 inside → clipped quad
    let mut pts = if count == 1 {
        vec![pt(0), lerp(0, 1), lerp(0, 2)]
    } else {
        vec![lerp(0, 1), pt(1), pt(2), lerp(0, 2)]
    };
    if cross < 0.0 {
        pts.reverse();
    }
    Some(pts)
}

/// Convert i_overlay shapes to `Vec<Polygon2D>`.
fn shapes_to_polygons(shapes: &[Shape], to_3d: Option<Matrix4<f64>>) -> Vec<Polygon2D> {
    shapes
        .iter()
        .filter(|s| !s.is_empty() && !s[0].is_empty())
        .map(|shape| {
            let exterior = shape[0].iter().map(|p| Point2::new(p[0], p[1])).collect();
            let interiors = shape[1..]
                .iter()
                .map(|c| c.iter().map(|p| Point2::new(p[0], p[1])).collect())
                .collect();
            Polygon2D {
                exterior,
                interiors,
                to_3d,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{Plane, create_box};
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Vector3};

    fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
        if count <= 1 {
            return vec![start];
        }
        let step = (end - start) / (count as f64 - 1.0);
        (0..count).map(|i| start + i as f64 * step).collect()
    }

    #[test]
    fn test_project_cube_z() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let origin = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::new(normal, origin);

        let dots = vertex_dots(&cube.vertices, &normal, &origin);
        let projected = plane.to_2d(&cube.vertices);
        let levels = linspace(-1.0, 1.0, 21);

        let results = project(&cube.faces, &dots, &projected, &levels, None);
        assert_eq!(results.len(), levels.len());

        for (i, level) in levels.iter().enumerate() {
            if *level > -0.5 + 1e-10 && *level < 0.5 - 1e-10 {
                assert!(
                    results[i].is_some(),
                    "Level {level} should have a projection"
                );
            }
            if *level > 0.5 + 1e-6 {
                assert!(
                    results[i].is_none(),
                    "Level {level} should be None (above cube)"
                );
            }
        }
    }

    #[test]
    fn test_project_cube_diagonal() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let normal = Vector3::new(1.0, 1.0, 1.0).normalize();
        let origin = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::new(normal, origin);

        let dots = vertex_dots(&cube.vertices, &normal, &origin);
        let projected = plane.to_2d(&cube.vertices);
        let levels = linspace(-1.0, 1.5, 21);

        let results = project(&cube.faces, &dots, &projected, &levels, None);
        assert_eq!(results.len(), levels.len());

        let max_dot = 3.0_f64.sqrt() / 2.0;
        for (i, level) in levels.iter().enumerate() {
            if *level > max_dot + 1e-6 {
                assert!(
                    results[i].is_none(),
                    "Level {level} should be None (above max dot {max_dot})"
                );
            }
        }
    }

    #[test]
    fn test_project_monotone_areas() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let origin = Point3::origin();
        let plane = Plane::new(normal, origin);

        let dots = vertex_dots(&cube.vertices, &normal, &origin);
        let projected = plane.to_2d(&cube.vertices);
        let levels = linspace(-0.4, 0.4, 9);

        let results = project(&cube.faces, &dots, &projected, &levels, None);

        for (i, _level) in levels.iter().enumerate() {
            let polys = results[i].as_ref().expect("Should have projection");
            let area: f64 = polys.iter().map(|p| p.area()).sum();
            assert_relative_eq!(area, 1.0, epsilon = 0.1);
        }
    }

    #[test]
    fn test_project_empty() {
        let faces: Vec<[usize; 3]> = vec![];
        let dots: Vec<f64> = vec![];
        let vertices: Vec<Point2<f64>> = vec![];
        let levels = vec![0.0, 1.0];

        let results = project(&faces, &dots, &vertices, &levels, None);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_none());
        assert!(results[1].is_none());
    }
}
