//! Mesh projection onto a plane at multiple levels.
//!
//! Projects a triangle mesh onto a 2D plane using an incremental
//! top-to-bottom polygon union strategy. Faces are sorted by their
//! maximum dot product with the plane normal, then swept from
//! highest to lowest level, accumulating projected polygons.
//!
//! When BREP data is available, discrete polygon rings are converted
//! to analytical `Path2D` entities (circles, arcs) via `ring_to_segments`.

use crate::path::polygons::{Contour, Shape, shapes_to_polygons, signed_area};
use crate::path::{Arc2, Circle2, Line, Path2D, Polygon2D, Segment2D, Winding};
use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::solver::{Precision, Solver};
use i_overlay::float::overlay::OverlayOptions;
use i_overlay::float::simplify::SimplifyShape;
use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;
use nalgebra::{Matrix4, Point2, Point3, Vector3};
use rayon::prelude::*;

/// Absolute tolerance for vertex snapping (tied to i_overlay MEDIUM_HIGH precision).
const EPSILON_MERGE: f64 = 1e-8;

/// Squared merge tolerance, used for distance² comparisons.
const EPSILON_MERGE_SQ: f64 = EPSILON_MERGE * EPSILON_MERGE;

/// Relative tolerance for comparing values like radii and circle positions.
pub(crate) const EPSILON_RELATIVE: f64 = 1e-8;

/// Minimum consecutive points on a circle required to emit an arc.
/// Shorter runs are demoted to line segments to avoid spurious micro-arcs
/// from subdivided meshes where midpoint averaging moves vertices off-surface.
const MIN_ARC_POINTS: usize = 4;

/// Tolerance for classifying a vertex as on/above the clipping plane.
const PLANE_TOL: f64 = 1e-12;

/// Strict interior containment test using barycentric coordinates.
/// Returns true only if `p` is strictly inside the triangle `a-b-c`.
fn triangle_contains_point(
    p: &Point2<f64>,
    a: &Point2<f64>,
    b: &Point2<f64>,
    c: &Point2<f64>,
) -> bool {
    const EPS: f64 = 1e-20;
    let denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if denom.abs() < EPS {
        return false;
    }
    let inv = 1.0 / denom;
    let u = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) * inv;
    let v = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) * inv;
    let w = 1.0 - u - v;
    u > EPS && v > EPS && w > EPS
}

/// Returns true if all 3 vertices of `inner` are strictly inside `outer`.
fn triangle_in_triangle(outer: &[Point2<f64>; 3], inner: &[Point2<f64>; 3]) -> bool {
    triangle_contains_point(&inner[0], &outer[0], &outer[1], &outer[2])
        && triangle_contains_point(&inner[1], &outer[0], &outer[1], &outer[2])
        && triangle_contains_point(&inner[2], &outer[0], &outer[1], &outer[2])
}

/// Pre-compute which triangles are strictly contained within other triangles.
///
/// Returns `contained_by[i] = Some(j)` where j is the container with the
/// maximum `face_ranges[j].0` (min_dot).
///
/// Uses a flat 2D grid for broad-phase, an area pre-filter to limit
/// potential containers, and rayon to parallelize the per-triangle queries.
fn compute_containment(
    faces: &[[usize; 3]],
    vertices: &[Point2<f64>],
    face_ranges: &[(f64, f64)],
) -> Vec<Option<usize>> {
    let n = faces.len();
    if n == 0 {
        return Vec::new();
    }

    // Phase 1: compute triangle geometry (area, vertices, AABB) in parallel
    let geom: Vec<(f64, [Point2<f64>; 3], [f64; 4])> = faces
        .par_iter()
        .map(|f| {
            let (a, b, c) = (vertices[f[0]], vertices[f[1]], vertices[f[2]]);
            let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            (
                cross.abs() * 0.5,
                [a, b, c],
                [
                    a.x.min(b.x).min(c.x),
                    a.y.min(b.y).min(c.y),
                    a.x.max(b.x).max(c.x),
                    a.y.max(b.y).max(c.y),
                ],
            )
        })
        .collect();

    // Split into SoA and compute global bounding box in one pass
    let mut areas: Vec<f64> = Vec::with_capacity(n);
    let mut tris: Vec<[Point2<f64>; 3]> = Vec::with_capacity(n);
    let mut aabbs: Vec<[f64; 4]> = Vec::with_capacity(n);
    let (mut bb_min_x, mut bb_min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut bb_max_x, mut bb_max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(area, tri, aabb) in &geom {
        areas.push(area);
        tris.push(tri);
        aabbs.push(aabb);
        bb_min_x = bb_min_x.min(aabb[0]);
        bb_min_y = bb_min_y.min(aabb[1]);
        bb_max_x = bb_max_x.max(aabb[2]);
        bb_max_y = bb_max_y.max(aabb[3]);
    }
    drop(geom);

    // Phase 2: build adaptive grid with all non-degenerate triangles.
    // Grid resolution targets ~1 triangle per cell on average.
    // Each triangle is inserted into every cell its AABB overlaps.
    // Containees query a single cell (their centroid). This is correct
    // because if j is fully inside i, j's centroid is inside i's AABB,
    // so i is guaranteed to be in that cell. The narrow phase skips
    // candidates where areas[i] <= areas[j].
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let grid_side = (n as f64).sqrt().ceil().max(1.0) as usize;
    let dx = (bb_max_x - bb_min_x) + 1e-10;
    let dy = (bb_max_y - bb_min_y) + 1e-10;
    let inv_cx = grid_side as f64 / dx;
    let inv_cy = grid_side as f64 / dy;

    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); grid_side * grid_side];
    for i in 0..n {
        if areas[i] < 1e-30 {
            continue;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (gx0, gy0, gx1, gy1) = (
            ((aabbs[i][0] - bb_min_x) * inv_cx) as usize,
            ((aabbs[i][1] - bb_min_y) * inv_cy) as usize,
            ((aabbs[i][2] - bb_min_x) * inv_cx) as usize,
            ((aabbs[i][3] - bb_min_y) * inv_cy) as usize,
        );
        for gy in gy0..=gy1.min(grid_side - 1) {
            for gx in gx0..=gx1.min(grid_side - 1) {
                grid[gy * grid_side + gx].push(i);
            }
        }
    }

    // Phase 3: for each triangle, look up centroid cell and check containers (parallel)
    let contained_by: Vec<Option<usize>> = (0..n)
        .into_par_iter()
        .map(|j| {
            if areas[j] < 1e-30 {
                return None;
            }
            let tj = &tris[j];
            let cx = (tj[0].x + tj[1].x + tj[2].x) / 3.0;
            let cy = (tj[0].y + tj[1].y + tj[2].y) / 3.0;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let (gx, gy) = (
                ((cx - bb_min_x) * inv_cx) as usize,
                ((cy - bb_min_y) * inv_cy) as usize,
            );
            let gx = gx.min(grid_side - 1);
            let gy = gy.min(grid_side - 1);

            let mut best: Option<usize> = None;
            for &i in &grid[gy * grid_side + gx] {
                if i == j || areas[i] <= areas[j] {
                    continue;
                }
                // i's AABB must fully contain j's AABB for triangle containment
                if aabbs[i][0] > aabbs[j][0]
                    || aabbs[i][1] > aabbs[j][1]
                    || aabbs[i][2] < aabbs[j][2]
                    || aabbs[i][3] < aabbs[j][3]
                {
                    continue;
                }
                if triangle_in_triangle(&tris[i], &tris[j]) {
                    match best {
                        None => best = Some(i),
                        Some(prev) if face_ranges[i].0 > face_ranges[prev].0 => {
                            best = Some(i);
                        }
                        _ => {}
                    }
                }
            }
            best
        })
        .collect();

    contained_by
}

/// Projected cylinder edges grouped by their circle.
pub struct CircleWithEdges {
    pub center: Point2<f64>,
    pub radius: f64,
    pub edges: Vec<(Point2<f64>, Point2<f64>)>,
    /// Maximum sagitta (midpoint deviation) of the longest chord,
    /// computed from the actual mesh edge lengths on this circle.
    pub max_sagitta: f64,
}

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

/// Snap each vertex in output shapes to the nearest original projected vertex,
/// restoring f64 precision lost during i_overlay's f64→i32→f64 round-trip.
///
/// Only snaps if the nearest original vertex is within `threshold` distance.
fn snap_to_original(
    shapes: &mut [Shape],
    tree: &ImmutableKdTree<f64, 2>,
    vertices: &[Point2<f64>],
    threshold_sq: f64,
) {
    for shape in shapes.iter_mut() {
        for contour in shape.iter_mut() {
            for pt in contour.iter_mut() {
                let nearest = tree.nearest_one::<SquaredEuclidean>(&[pt[0], pt[1]]);
                if nearest.distance < threshold_sq {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let v = &vertices[nearest.item as usize];
                    pt[0] = v.x;
                    pt[1] = v.y;
                }
            }
        }
    }
}

/// Snap ring vertices onto cylinder circles using per-circle sagitta tolerance.
///
/// For each vertex, find the nearest circle center via KD-tree. If the vertex's
/// radial distance from the circle differs from the radius by less than the
/// circle's `max_sagitta` (computed from its longest mesh chord), project the
/// vertex radially onto the analytical circle.
///
/// This replaces the previous edge-provenance narrow-phase (`point_segment_dist_sq`)
/// which relied on a fragile absolute tolerance that broke at different model scales.
fn snap_to_cylinder_edges(
    ring: &mut [Point2<f64>],
    circles: &[CircleWithEdges],
    circle_tree: &ImmutableKdTree<f64, 2>,
) {
    if circles.is_empty() {
        return;
    }
    for pt in ring.iter_mut() {
        let nearest = circle_tree.nearest_one::<SquaredEuclidean>(&[pt.x, pt.y]);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let circle = &circles[nearest.item as usize];
        let offset = pt.coords - circle.center.coords;
        let d = offset.norm();
        if d < 1e-15 {
            continue;
        }
        let radial_error = (d - circle.radius).abs();
        if radial_error > circle.max_sagitta {
            continue;
        }
        *pt = circle.center + offset * (circle.radius / d);
    }
}

/// Project a mesh onto a plane at multiple levels, returning discrete polygons.
///
/// Returns one `Option<Vec<Polygon2D>>` per level (same order as input).
/// Levels where the plane doesn't intersect any geometry return `None`.
///
/// # Arguments
/// * `faces` - Triangle face indices (into `dots` and `vertices`)
/// * `dots` - Pre-computed signed distances from `section::vertex_dots()`
/// * `vertices` - Pre-computed 2D projections from `Plane::to_2d()`
/// * `levels` - Height offsets along the normal to project at
pub fn project_polygons(
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
        min_output_area: EPSILON_MERGE_SQ,
        ..Default::default()
    };

    // Build a KD-tree from the projected vertices so we can snap i_overlay's
    // quantized output back to exact f64 positions.
    let entries: Vec<[f64; 2]> = vertices.iter().map(|v| [v.x, v.y]).collect();
    let tree: ImmutableKdTree<f64, 2> = ImmutableKdTree::new_from_slice(&entries);
    let snap_threshold_sq = EPSILON_MERGE_SQ;

    // Pre-compute triangle-in-triangle containment for culling
    let contained_by = compute_containment(faces, vertices, &face_ranges);

    let mut results: Vec<Option<Vec<Polygon2D>>> = vec![None; levels.len()];
    let mut cursor: usize = 0;
    let mut partial_faces: Vec<usize> = Vec::new();
    let mut accumulated: Vec<Shape> = Vec::new();

    // Check if face fi is culled at this level: its container is fully above.
    let is_culled = |fi: usize, level: f64| -> bool {
        matches!(contained_by[fi], Some(c) if face_ranges[c].0 >= level)
    };

    for &(orig_idx, level) in &level_order {
        let mut new_contours: Vec<Contour> = Vec::new();

        // 1. Add newly entering faces (max_dot >= level, not yet seen)
        while cursor < face_order.len() && face_ranges[face_order[cursor]].1 >= level {
            let fi = face_order[cursor];
            cursor += 1;

            if is_culled(fi, level) {
                continue;
            }

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
            if is_culled(fi, level) {
                continue;
            }

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
            let mut slab: Vec<Shape> =
                new_contours.simplify_shape_custom(FillRule::Positive, options, solver);
            snap_to_original(&mut slab, &tree, vertices, snap_threshold_sq);
            if !slab.is_empty() {
                if accumulated.is_empty() {
                    accumulated = slab;
                } else {
                    accumulated.extend(slab);
                    let flat: Vec<Contour> = accumulated.into_iter().flatten().collect();
                    accumulated = flat.simplify_shape_custom(FillRule::Positive, options, solver);
                    snap_to_original(&mut accumulated, &tree, vertices, snap_threshold_sq);
                }
            }
        }

        // 4. Convert accumulated shapes to Polygon2D
        //    Compose to_3d with a local-frame Z translation so that
        //    (x, y, 0) maps to the plane at this level.
        if !accumulated.is_empty() {
            let level_to_3d =
                to_3d.map(|base| base * Matrix4::new_translation(&Vector3::new(0.0, 0.0, level)));
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
    if cross.abs() < EPSILON_MERGE_SQ {
        return None;
    }
    if (b - a).norm_squared() < EPSILON_MERGE_SQ
        || (c - b).norm_squared() < EPSILON_MERGE_SQ
        || (a - c).norm_squared() < EPSILON_MERGE_SQ
    {
        return None;
    }
    Some(if cross > 0.0 {
        vec![[a.x, a.y], [b.x, b.y], [c.x, c.y]]
    } else {
        vec![[c.x, c.y], [b.x, b.y], [a.x, a.y]]
    })
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
    if cross.abs() < EPSILON_MERGE_SQ {
        return None;
    }

    let lerp = |a: usize, b: usize| -> Point2<f64> {
        let t = -d[a] / (d[b] - d[a]);
        p[a] + t * (p[b] - p[a])
    };

    // count==1: vertex 0 inside, 1+2 outside → clipped triangle (3 pts)
    // count==2: vertex 0 outside, 1+2 inside → clipped quad (4 pts)
    let (pts, n) = if count == 1 {
        ([p[0], lerp(0, 1), lerp(0, 2), Point2::origin()], 3)
    } else {
        ([lerp(0, 1), p[1], p[2], lerp(0, 2)], 4)
    };

    // Reject polygons with any near-zero-length edge (grazing clips, slivers).
    if (0..n).any(|i| (pts[(i + 1) % n] - pts[i]).norm_squared() < EPSILON_MERGE_SQ) {
        return None;
    }

    Some(if cross > 0.0 {
        (0..n).map(|i| [pts[i].x, pts[i].y]).collect()
    } else {
        (0..n).rev().map(|i| [pts[i].x, pts[i].y]).collect()
    })
}

/// Convert a discrete ring of points into a sequence of `Segment2D`,
/// recognizing circles and arcs from a set of expected circles.
///
/// # Arguments
/// * `ring` - Closed ring of 2D points (first != last assumed, will wrap)
/// * `expected_circles` - `(center, radius)` pairs from BREP projection
///
/// # Algorithm
/// 1. For each point, check which expected circle (if any) it lies on
///    (within 1e-8 relative radius tolerance).
/// 2. Group consecutive points with the same assignment into runs.
/// 3. Convert each run:
///    - `None` (no circle): Line segments
///    - `Some(idx)` spanning entire ring: `Circle2`
///    - `Some(idx)` partial: `Arc2` with computed sweep angle
pub fn ring_to_segments(
    ring: &[Point2<f64>],
    expected_circles: &[(Point2<f64>, f64)],
) -> (Vec<Point2<f64>>, Vec<Segment2D>) {
    if ring.len() < 2 {
        return (ring.to_vec(), Vec::new());
    }

    // Helper to add a vertex (with dedup within tolerance)
    let tol_sq = EPSILON_MERGE_SQ;
    let add_vertex = |pt: Point2<f64>, verts: &mut Vec<Point2<f64>>| -> usize {
        for (i, v) in verts.iter().enumerate() {
            if (v.x - pt.x).powi(2) + (v.y - pt.y).powi(2) < tol_sq {
                return i;
            }
        }
        let idx = verts.len();
        verts.push(pt);
        idx
    };

    let mut vertices: Vec<Point2<f64>> = Vec::new();
    let mut segments: Vec<Segment2D> = Vec::new();

    // Step 1: assign each point to a circle (or None).
    // After snap_to_original recovers original mesh vertices and
    // snap_to_cylinder_edges projects intersection vertices onto
    // analytical circles, all circle-boundary points are within
    // f64 precision of the exact circle. 1e-8 relative tolerance
    // provides margin without false positives.
    let circle_tol = EPSILON_RELATIVE;
    let assignments: Vec<Option<usize>> = ring
        .iter()
        .map(|pt| {
            for (ci, (center, radius)) in expected_circles.iter().enumerate() {
                let dist_sq = (pt.x - center.x).powi(2) + (pt.y - center.y).powi(2);
                let lo = radius * (1.0 - circle_tol);
                let hi = radius * (1.0 + circle_tol);
                if dist_sq > lo * lo && dist_sq < hi * hi {
                    return Some(ci);
                }
            }
            None
        })
        .collect();

    // Step 2: group consecutive points with the same assignment
    // Each run is (assignment, start_index, end_index_exclusive)
    let mut runs: Vec<(Option<usize>, usize, usize)> = Vec::new();
    let mut run_start = 0;
    let mut current = assignments[0];
    for (i, &assign) in assignments.iter().enumerate().skip(1) {
        if assign != current {
            runs.push((current, run_start, i));
            current = assign;
            run_start = i;
        }
    }
    runs.push((current, run_start, ring.len()));

    // Merge first and last run if they have the same assignment and we wrap
    if runs.len() > 1 && runs.first().unwrap().0 == runs.last().unwrap().0 {
        let last = runs.pop().unwrap();
        runs[0].1 = last.1; // extend first run backwards (we handle wrapping below)
    }

    // Check if entire ring is one circle (all vertices assigned to same circle)
    if runs.len() == 1 && runs[0].0.is_some() {
        let ci = runs[0].0.unwrap();
        let (center, radius) = expected_circles[ci];
        let center_idx = add_vertex(center, &mut vertices);
        segments.push(Segment2D::Circle(Circle2::new(center_idx, radius)));
        return (vertices, segments);
    }

    // Determine ring winding for arc direction
    let ring_area = signed_area(ring);
    let winding = if ring_area >= 0.0 {
        Winding::Ccw
    } else {
        Winding::Cw
    };

    // Collect the last point of each run (needed for inter-run Line connections).
    let run_last_point = |run: &(Option<usize>, usize, usize)| -> Point2<f64> {
        if run.1 > run.2 {
            // Wrapped run: last point is ring[run.2 - 1] (or last element if run.2 == 0)
            if run.2 == 0 {
                ring[ring.len() - 1]
            } else {
                ring[run.2 - 1]
            }
        } else {
            ring[run.2 - 1]
        }
    };

    let run_first_point = |run: &(Option<usize>, usize, usize)| -> Point2<f64> { ring[run.1] };

    for (ri, (assignment, run_start_orig, run_end_orig)) in runs.iter().enumerate() {
        // Insert a connecting Line when transitioning between different circles.
        // Tangent points at fillet junctions lie exactly on the adjacent circle,
        // so without this, short straight sides between fillets disappear entirely.
        let prev_run = if ri == 0 {
            runs.last().unwrap()
        } else {
            &runs[ri - 1]
        };
        if let (Some(prev_ci), Some(cur_ci)) = (prev_run.0, *assignment)
            && prev_ci != cur_ci
        {
            let prev_last = run_last_point(prev_run);
            let cur_first = run_first_point(&(*assignment, *run_start_orig, *run_end_orig));
            let s = add_vertex(prev_last, &mut vertices);
            let e = add_vertex(cur_first, &mut vertices);
            if s != e {
                segments.push(Segment2D::Line(Line::from_points(vec![s, e])));
            }
        }

        // Collect points for this run, handling the wrapped case
        let run_points: Vec<Point2<f64>> = if run_start_orig > run_end_orig {
            // Wrapped: from run_start to end, then 0 to run_end
            ring[*run_start_orig..]
                .iter()
                .chain(ring[..*run_end_orig].iter())
                .copied()
                .collect()
        } else {
            ring[*run_start_orig..*run_end_orig].to_vec()
        };

        if run_points.is_empty() {
            continue;
        }

        match assignment {
            None => {
                // Line segments between consecutive points
                let indices: Vec<usize> = run_points
                    .iter()
                    .map(|p| add_vertex(*p, &mut vertices))
                    .collect();
                // Close connection to next run's first point if needed
                if indices.len() >= 2 {
                    // Add line from the polyline indices
                    segments.push(Segment2D::Line(Line::from_points(indices)));
                }
            }
            Some(ci) => {
                // Require minimum consecutive points for a reliable arc
                if run_points.len() < MIN_ARC_POINTS {
                    // Too few points — emit as line segments
                    let indices: Vec<usize> = run_points
                        .iter()
                        .map(|p| add_vertex(*p, &mut vertices))
                        .collect();
                    if indices.len() >= 2 {
                        segments.push(Segment2D::Line(Line::from_points(indices)));
                    }
                } else {
                    let (center, _radius) = expected_circles[*ci];
                    let first = run_points.first().unwrap();
                    let last = run_points.last().unwrap();

                    let start_idx = add_vertex(*first, &mut vertices);
                    let finish_idx = add_vertex(*last, &mut vertices);

                    if start_idx == finish_idx {
                        // Degenerate zero-sweep — skip.
                        // Closed primitives only from whole-ring check.
                    } else {
                        // Compute sweep angle
                        let start_angle = (first.y - center.y).atan2(first.x - center.x);
                        let end_angle = (last.y - center.y).atan2(last.x - center.x);

                        let mut sweep = end_angle - start_angle;

                        // Adjust sweep based on winding
                        match winding {
                            Winding::Ccw => {
                                if sweep <= 0.0 {
                                    sweep += std::f64::consts::TAU;
                                }
                            }
                            Winding::Cw => {
                                if sweep >= 0.0 {
                                    sweep -= std::f64::consts::TAU;
                                }
                            }
                        }

                        segments.push(Segment2D::Arc(Arc2::new(
                            start_idx, finish_idx, sweep, winding,
                        )));
                    }
                }
            }
        }
    }

    (vertices, segments)
}

/// Convert a `Polygon2D` to a `Path2D`, using expected circles from BREP data
/// to produce analytical `Circle2`/`Arc2` segments where possible.
///
/// When `expected_circles` is empty, all segments become `Line` (same as before).
///
/// `cylinder_edges` provides cylinder edge provenance for snapping
/// union-created intersection vertices onto their circles before
/// `ring_to_segments` attempts assignment. Pass `None` when no cylinder
/// edge data is available (e.g. in the buffer path).
pub fn polygon_to_path(
    polygon: &Polygon2D,
    expected_circles: &[(Point2<f64>, f64)],
    cylinder_edges: Option<(&[CircleWithEdges], &ImmutableKdTree<f64, 2>)>,
) -> Path2D {
    if expected_circles.is_empty() {
        let mut path = polygon.to_path2d();
        path.to_3d = polygon.to_3d;
        return path;
    }

    let mut all_vertices: Vec<Point2<f64>> = Vec::new();
    let mut all_segments: Vec<Segment2D> = Vec::new();

    // Process exterior ring
    let mut exterior = polygon.exterior.clone();
    if let Some((circles, tree)) = cylinder_edges {
        snap_to_cylinder_edges(&mut exterior, circles, tree);
    }
    let (mut verts, segs) = ring_to_segments(&exterior, expected_circles);
    let base = all_vertices.len();
    for seg in segs {
        all_segments.push(remap_segment(&seg, base));
    }
    all_vertices.append(&mut verts);

    // Process interior rings
    for interior in &polygon.interiors {
        if interior.len() < 3 {
            continue;
        }
        let mut ring = interior.clone();
        if let Some((circles, tree)) = cylinder_edges {
            snap_to_cylinder_edges(&mut ring, circles, tree);
        }
        let (mut verts, segs) = ring_to_segments(&ring, expected_circles);
        let base = all_vertices.len();
        for seg in segs {
            all_segments.push(remap_segment(&seg, base));
        }
        all_vertices.append(&mut verts);
    }

    let mut path = Path2D::from_vertices_and_segments(all_vertices, all_segments);
    path.to_3d = polygon.to_3d;
    path
}

/// Remap vertex indices in a segment by adding a base offset.
fn remap_segment(seg: &Segment2D, base: usize) -> Segment2D {
    match seg {
        Segment2D::Line(l) => Segment2D::Line(Line::from_points(
            l.points.iter().map(|i| i + base).collect(),
        )),
        Segment2D::Arc(a) => Segment2D::Arc(Arc2::new(
            a.start + base,
            a.finish + base,
            a.sweep_angle(),
            a.winding,
        )),
        Segment2D::Circle(c) => Segment2D::Circle(Circle2::new(c.center + base, c.radius)),
        other => other.clone(), // Ellipse, Bezier, BSpline pass through unchanged
    }
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

        let results = project_polygons(&cube.faces, &dots, &projected, &levels, None);
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

        let results = project_polygons(&cube.faces, &dots, &projected, &levels, None);
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

        let results = project_polygons(&cube.faces, &dots, &projected, &levels, None);

        for (i, _level) in levels.iter().enumerate() {
            let polys = results[i].as_ref().expect("Should have projection");
            let area: f64 = polys.iter().map(|p| p.area()).sum();
            assert_relative_eq!(area, 1.0, epsilon = 0.1);
        }
    }

    #[test]
    fn test_project_to_3d_offset_origin() {
        // A cube placed far from the world origin: verify that
        // reconstructed 3D vertices land inside the cube's AABB,
        // not 1000 miles away due to a broken transform.
        let offset = Vector3::new(100.0, -200.0, 50.0);
        let extents = [2.0, 2.0, 2.0];
        let base = create_box(&extents);
        let verts_3d: Vec<Point3<f64>> = base.vertices.iter().map(|v| v + offset).collect();

        let center = Point3::from(offset);
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let plane = Plane::new(normal, center);

        let dots = vertex_dots(&verts_3d, &normal, &center);
        let projected = plane.to_2d(&verts_3d);
        let to_3d = plane.transform_to_2d().try_inverse();
        // levels are negative (below the origin plane)
        let levels = vec![-0.8, -0.4, 0.0];

        let results = project_polygons(&base.faces, &dots, &projected, &levels, to_3d);

        // Cube AABB
        let half = extents[0] / 2.0;
        let aabb_min = center - Vector3::new(half, half, half);
        let aabb_max = center + Vector3::new(half, half, half);

        for (i, level) in levels.iter().enumerate() {
            let polys = results[i].as_ref().expect("should have projection");
            for poly in polys {
                let to_3d = poly.to_3d.expect("should have to_3d");
                for p in &poly.exterior {
                    let p3 = to_3d.transform_point(&Point3::new(p.x, p.y, 0.0));
                    // Height relative to origin should equal the level
                    let height = (p3 - center).dot(&normal);
                    assert_relative_eq!(height, *level, epsilon = 1e-6);
                    // Must be inside the cube's AABB (with tolerance)
                    let tol = 1e-6;
                    assert!(
                        p3.x >= aabb_min.x - tol
                            && p3.x <= aabb_max.x + tol
                            && p3.y >= aabb_min.y - tol
                            && p3.y <= aabb_max.y + tol
                            && p3.z >= aabb_min.z - tol
                            && p3.z <= aabb_max.z + tol,
                        "Vertex {p3:?} outside cube AABB [{aabb_min:?}, {aabb_max:?}]"
                    );
                }
            }
        }
    }

    #[test]
    fn test_project_empty() {
        let faces: Vec<[usize; 3]> = vec![];
        let dots: Vec<f64> = vec![];
        let vertices: Vec<Point2<f64>> = vec![];
        let levels = vec![0.0, 1.0];

        let results = project_polygons(&faces, &dots, &vertices, &levels, None);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_none());
        assert!(results[1].is_none());
    }

    /// Generate a CCW circle ring of `n` points.
    fn circle_ring(center: Point2<f64>, radius: f64, n: usize) -> Vec<Point2<f64>> {
        (0..n)
            .map(|i| {
                let angle = std::f64::consts::TAU * i as f64 / n as f64;
                Point2::new(
                    center.x + radius * angle.cos(),
                    center.y + radius * angle.sin(),
                )
            })
            .collect()
    }

    #[test]
    fn test_ring_to_segments_full_circle() {
        let center = Point2::new(1.0, 2.0);
        let radius = 5.0;
        let ring = circle_ring(center, radius, 64);
        let expected = vec![(center, radius)];

        let (verts, segs) = ring_to_segments(&ring, &expected);

        // Should produce a single Circle2
        assert_eq!(segs.len(), 1);
        match &segs[0] {
            Segment2D::Circle(c) => {
                assert_relative_eq!(c.radius, radius, epsilon = 1e-10);
                let cv = verts[c.center];
                assert_relative_eq!(cv.x, center.x, epsilon = 1e-6);
                assert_relative_eq!(cv.y, center.y, epsilon = 1e-6);
            }
            other => panic!("Expected Circle2, got {:?}", other),
        }
    }

    #[test]
    fn test_ring_to_segments_no_circles() {
        // Rectangle ring, no expected circles → all Line
        let ring = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        let (_, segs) = ring_to_segments(&ring, &[]);

        // All segments should be Line
        for s in &segs {
            assert!(
                matches!(s, Segment2D::Line(_)),
                "Expected Line, got {:?}",
                s
            );
        }
    }

    #[test]
    fn test_ring_to_segments_rectangle_with_fillets() {
        // Approximate a rounded rectangle: 4 straight sides + 4 quarter-circle corners
        let r = 1.0;
        let w = 10.0;
        let h = 5.0;

        let mut ring = Vec::new();

        // Bottom side (from lower-left to lower-right minus fillet)
        ring.push(Point2::new(r, 0.0));
        ring.push(Point2::new(w - r, 0.0));

        // Bottom-right fillet (center at w-r, r)
        let cr = Point2::new(w - r, r);
        for i in 0..=8 {
            let angle = -std::f64::consts::FRAC_PI_2 + std::f64::consts::FRAC_PI_2 * i as f64 / 8.0;
            ring.push(Point2::new(cr.x + r * angle.cos(), cr.y + r * angle.sin()));
        }

        // Right side
        ring.push(Point2::new(w, h - r));

        // Top-right fillet (center at w-r, h-r)
        let cr = Point2::new(w - r, h - r);
        for i in 0..=8 {
            let angle = 0.0 + std::f64::consts::FRAC_PI_2 * i as f64 / 8.0;
            ring.push(Point2::new(cr.x + r * angle.cos(), cr.y + r * angle.sin()));
        }

        // Top side
        ring.push(Point2::new(r, h));

        // Top-left fillet (center at r, h-r)
        let cr = Point2::new(r, h - r);
        for i in 0..=8 {
            let angle = std::f64::consts::FRAC_PI_2 + std::f64::consts::FRAC_PI_2 * i as f64 / 8.0;
            ring.push(Point2::new(cr.x + r * angle.cos(), cr.y + r * angle.sin()));
        }

        // Left side
        ring.push(Point2::new(0.0, r));

        // Bottom-left fillet (center at r, r)
        let cr = Point2::new(r, r);
        for i in 0..=8 {
            let angle = std::f64::consts::PI + std::f64::consts::FRAC_PI_2 * i as f64 / 8.0;
            ring.push(Point2::new(cr.x + r * angle.cos(), cr.y + r * angle.sin()));
        }

        let expected_circles = vec![
            (Point2::new(w - r, r), r),
            (Point2::new(w - r, h - r), r),
            (Point2::new(r, h - r), r),
            (Point2::new(r, r), r),
        ];

        let (_, segs) = ring_to_segments(&ring, &expected_circles);

        let n_lines = segs
            .iter()
            .filter(|s| matches!(s, Segment2D::Line(_)))
            .count();
        let n_arcs = segs
            .iter()
            .filter(|s| matches!(s, Segment2D::Arc(_)))
            .count();

        assert!(n_arcs >= 4, "Expected at least 4 arcs, got {}", n_arcs);
        assert!(n_lines >= 4, "Expected at least 4 lines, got {}", n_lines);
    }

    #[test]
    fn test_containment_basic() {
        // Large outer triangle
        let vertices = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(5.0, 10.0),
            // Small inner triangle
            Point2::new(4.0, 2.0),
            Point2::new(6.0, 2.0),
            Point2::new(5.0, 4.0),
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        let face_ranges = vec![(0.0, 1.0), (0.0, 0.5)];

        let result = compute_containment(&faces, &vertices, &face_ranges);
        // Inner triangle (face 1) should be contained by outer (face 0)
        assert_eq!(result[1], Some(0));
        // Outer triangle should not be contained
        assert_eq!(result[0], None);
    }

    #[test]
    fn test_containment_degenerate() {
        // Collinear points form a degenerate triangle
        let vertices = vec![
            Point2::new(0.0, 0.0),
            Point2::new(5.0, 0.0),
            Point2::new(10.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let face_ranges = vec![(0.0, 1.0)];

        let result = compute_containment(&faces, &vertices, &face_ranges);
        // Degenerate faces are excluded from the tree and get None;
        // they already produce no contours from face_contour/clip_face.
        assert_eq!(result[0], None);
    }

    #[cfg(test)]
    #[test]
    #[ignore]
    fn bench_project_featuretype() {
        use crate::attributes::GroupingKind;
        use crate::exchange::{FileType, load};
        use crate::formatting::Table;
        use crate::geometry::Geometry;
        use crate::timer::Profiler;
        use nalgebra::{Point3, Vector3};
        use std::time::Instant;

        // ── Load featuretype.glb ──
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data/featuretype.glb");
        let data = std::fs::read(&path).expect("featuretype.glb not found");
        let scene = load(&data, Some(FileType::GLB), None).unwrap();
        let mesh = scene
            .geometry
            .values()
            .find_map(|g| match g {
                Geometry::Mesh(m) => Some(m.as_ref()),
                _ => None,
            })
            .expect("no mesh in featuretype.glb");

        let original_faces = mesh.faces.len();

        // ── Project at each subdivision level, verify attributes + circles ──
        let axes = [
            ("X", Vector3::x()),
            ("Y", Vector3::y()),
            ("Z", Vector3::z()),
        ];
        let origin = Point3::origin();
        let levels = &[0.0];

        // Expected circle counts per axis: X=0, Y=1, Z=8
        let expected_circles: [usize; 3] = [0, 1, 8];

        let mut table = Table::new(&["subdiv", "axis", "faces", "circles", "arcs", "ms"]);
        let mut profiler = Profiler::start();

        for subdiv in 0..=3 {
            let current = if subdiv == 0 {
                mesh.clone()
            } else {
                mesh.subdivide(subdiv)
            };

            // Verify attributes at every level
            assert_eq!(
                current.faces.len(),
                original_faces * 4_usize.pow(subdiv as u32),
                "subdiv {subdiv}: face count wrong"
            );
            assert_eq!(
                current.face_surfaces.len(),
                mesh.face_surfaces.len(),
                "subdiv {subdiv}: face_surfaces should be cloned, not expanded"
            );
            assert_eq!(
                current.materials.len(),
                mesh.materials.len(),
                "subdiv {subdiv}: materials should be preserved"
            );

            // Verify surface grouping survived
            let orig_sg = mesh
                .attributes_face
                .groupings
                .iter()
                .find(|g| matches!(g.kind, GroupingKind::Surface));
            let sub_sg = current
                .attributes_face
                .groupings
                .iter()
                .find(|g| matches!(g.kind, GroupingKind::Surface));
            if let (Some(og), Some(sg)) = (orig_sg, sub_sg) {
                assert_eq!(
                    sg.indices.len(),
                    current.faces.len(),
                    "subdiv {subdiv}: surface grouping length mismatch"
                );
                let factor = 4_usize.pow(subdiv as u32);
                for (i, &orig_idx) in og.indices.iter().enumerate() {
                    for child in 0..factor {
                        assert_eq!(
                            sg.indices[i * factor + child],
                            orig_idx,
                            "subdiv {subdiv}: surface index mismatch at face {}",
                            i * factor + child
                        );
                    }
                }
            }

            // Project on each axis and verify circle/arc counts match baseline
            for (ai, (name, normal)) in axes.iter().enumerate() {
                let t0 = Instant::now();
                let results = current.project(normal, &origin, levels);
                let ms = t0.elapsed().as_secs_f64() * 1000.0;

                let (n_circles, n_arcs) = results[0]
                    .as_ref()
                    .map(|paths| {
                        let segs: Vec<_> = paths.iter().flat_map(|p| p.segments.iter()).collect();
                        (
                            segs.iter()
                                .filter(|s| matches!(s, crate::path::Segment2D::Circle(_)))
                                .count(),
                            segs.iter()
                                .filter(|s| matches!(s, crate::path::Segment2D::Arc(_)))
                                .count(),
                        )
                    })
                    .unwrap_or((0, 0));

                // At subdiv 0 the original BREP tessellation lies on
                // the analytical cylinders, so circle counts must be exact.
                // Higher subdivisions move vertices off-surface (midpoint
                // averaging doesn't preserve cylinder geometry).
                if subdiv == 0 {
                    assert_eq!(
                        n_circles, expected_circles[ai],
                        "subdiv {subdiv} axis {name}: expected {} circles, got {n_circles}",
                        expected_circles[ai]
                    );

                    // All Z-axis circles should have the same radius
                    if ai == 2 {
                        let radii: Vec<f64> = results[0]
                            .as_ref()
                            .into_iter()
                            .flat_map(|paths| paths.iter())
                            .flat_map(|p| p.segments.iter())
                            .filter_map(|s| match s {
                                crate::path::Segment2D::Circle(c) => Some(c.radius),
                                _ => None,
                            })
                            .collect();
                        if let Some(&first) = radii.first() {
                            for (i, &r) in radii.iter().enumerate() {
                                assert!(
                                    (r - first).abs() < first * 1e-6,
                                    "subdiv {subdiv}: Z circle {i} radius {r} != first {first}"
                                );
                            }
                        }
                    }
                }

                table.row(vec![
                    subdiv.to_string(),
                    name.to_string(),
                    current.faces.len().to_string(),
                    n_circles.to_string(),
                    n_arcs.to_string(),
                    format!("{ms:.2}"),
                ]);
            }
        }
        profiler.stop();

        println!("\n{table}");
        println!("{profiler}");
    }

    #[test]
    fn test_containment_picks_max_min_dot() {
        // Three nested triangles: outer, middle, inner
        let vertices = vec![
            // Outer (face 0)
            Point2::new(0.0, 0.0),
            Point2::new(20.0, 0.0),
            Point2::new(10.0, 20.0),
            // Middle (face 1)
            Point2::new(4.0, 2.0),
            Point2::new(16.0, 2.0),
            Point2::new(10.0, 14.0),
            // Inner (face 2)
            Point2::new(7.0, 4.0),
            Point2::new(13.0, 4.0),
            Point2::new(10.0, 8.0),
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]];
        // Outer has min_dot=0.0, middle has min_dot=2.0, inner has min_dot=1.0
        let face_ranges = vec![(0.0, 5.0), (2.0, 4.0), (1.0, 3.0)];

        let result = compute_containment(&faces, &vertices, &face_ranges);
        // Inner (face 2) is inside both outer and middle.
        // Middle has higher min_dot (2.0 > 0.0), so it should be picked.
        assert_eq!(result[2], Some(1));
        // Middle (face 1) is inside outer only
        assert_eq!(result[1], Some(0));
        // Outer is not contained
        assert_eq!(result[0], None);
    }

    /// Projection of a real part should not produce tiny "speckle" rings.
    ///
    /// These are degenerate zero-area rings from triangle slivers at the
    /// cut plane surviving through the polygon boolean union.
    #[test]
    fn test_project_no_speckles() {
        use crate::exchange::{FileType, load};
        use crate::geometry::Geometry;

        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/project/project");
        let data = std::fs::read(base.with_extension("glb")).unwrap();
        let scene = load(&data, Some(FileType::GLB), None).unwrap();
        let mesh = scene
            .geometry
            .values()
            .find_map(|g| match g {
                Geometry::Mesh(m) => Some(m.as_ref()),
                _ => None,
            })
            .unwrap();

        let json: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(base.with_extension("json")).unwrap())
                .unwrap();
        let arr = |key: &str| -> Vec<f64> {
            json[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        };
        let o = arr("origin");
        let n = arr("normal");
        let levels = arr("levels");
        let origin = Point3::new(o[0], o[1], o[2]);
        let normal = Vector3::new(n[0], n[1], n[2]);

        let results = mesh.project(&normal, &origin, &levels);

        let mut speckle_count = 0usize;
        for (li, result) in results.iter().enumerate() {
            if let Some(paths) = result {
                // Compute AABB area across all paths at this level.
                let mut bounds = crate::bounds::Bounds2::empty();
                for path in paths {
                    if let Some(b) = path.bounds() {
                        bounds = bounds.union(&b);
                    }
                }
                let e = bounds.extents();
                let threshold = e.x * e.y * 0.01;

                for path in paths {
                    for ring in &path.rings_discrete() {
                        let area = crate::path::polygons::signed_area(ring).abs();
                        if area < threshold {
                            speckle_count += 1;
                        }
                    }
                }
                if speckle_count > 0 {
                    eprintln!("level[{li}]={:.6}: {speckle_count} speckle(s)", levels[li],);
                }
            }
        }

        assert_eq!(speckle_count, 0, "found {speckle_count} speckle ring(s)");
    }
}
