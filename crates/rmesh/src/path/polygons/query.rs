//! Batch point-in-polygon via AABB + slab decomposition.
//!
//! For large batches of query points against a single polygon, the slab
//! approach is O(M·√N + N) vs the naive O(M×N) per-point ray casting.

use nalgebra::Point2;
use smallvec::SmallVec;

/// Minimum query count before slab build is amortized.
/// Build cost is ~11ns × N; per-query savings ~0.2ns × N for N≥50.
/// Break-even at M ≈ 55; we round up for safety.
const SLAB_MIN_QUERIES: usize = 64;

/// Batch point-in-polygon for an exterior polygon with optional holes.
///
/// Returns `Vec<bool>` parallel to `points` — true if inside `exterior`
/// and outside all `interiors` (holes).
///
/// Uses AABB early-out + slab decomposition for the fast path.
pub(crate) fn point_in_polygon(
    exterior: &[Point2<f64>],
    interiors: &[&[Point2<f64>]],
    points: &[Point2<f64>],
) -> Vec<bool> {
    if exterior.len() < 3 || points.is_empty() {
        return vec![false; points.len()];
    }

    let m = points.len();

    // Step 1: query exterior
    let mut result: Vec<bool> = if m >= SLAB_MIN_QUERIES {
        let slab = SlabIndex::build(exterior);
        points.iter().map(|p| slab.contains(p.x, p.y)).collect()
    } else {
        let (x_min, y_min, x_max, y_max) = aabb(exterior);
        points
            .iter()
            .map(|p| {
                p.x >= x_min
                    && p.x <= x_max
                    && p.y >= y_min
                    && p.y <= y_max
                    && point_in_polygon_naive(p, exterior)
            })
            .collect()
    };

    // Step 2: for each hole, flip inside points that fall in the hole
    for &hole in interiors {
        if hole.len() < 3 {
            continue;
        }

        let active_count = result.iter().filter(|&&v| v).count();
        if active_count == 0 {
            break;
        }

        if active_count >= SLAB_MIN_QUERIES {
            let hole_slab = SlabIndex::build(hole);
            for (i, r) in result.iter_mut().enumerate() {
                if *r && hole_slab.contains(points[i].x, points[i].y) {
                    *r = false;
                }
            }
        } else {
            let (hx_min, hy_min, hx_max, hy_max) = aabb(hole);
            for (i, r) in result.iter_mut().enumerate() {
                if *r {
                    let p = &points[i];
                    if p.x >= hx_min
                        && p.x <= hx_max
                        && p.y >= hy_min
                        && p.y <= hy_max
                        && point_in_polygon_naive(p, hole)
                    {
                        *r = false;
                    }
                }
            }
        }
    }

    result
}

/// Compute the axis-aligned bounding box of a polygon.
fn aabb(polygon: &[Point2<f64>]) -> (f64, f64, f64, f64) {
    let mut x_min = f64::MAX;
    let mut y_min = f64::MAX;
    let mut x_max = f64::MIN;
    let mut y_max = f64::MIN;
    for p in polygon {
        x_min = x_min.min(p.x);
        y_min = y_min.min(p.y);
        x_max = x_max.max(p.x);
        y_max = y_max.max(p.y);
    }
    (x_min, y_min, x_max, y_max)
}

/// Ray-casting point-in-polygon for a single point.
fn point_in_polygon_naive(point: &Point2<f64>, polygon: &[Point2<f64>]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let px = point.x;
    let py = point.y;
    let mut j = n - 1;
    for i in 0..n {
        let yi = polygon[i].y;
        let yj = polygon[j].y;
        let xi = polygon[i].x;
        let xj = polygon[j].x;
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Slab decomposition index for fast point-in-polygon queries.
///
/// Divides the polygon's y-range into sqrt(N) equal-width horizontal slabs.
/// Each slab stores precomputed `[slope, x_base, yi, yj]` per edge for
/// FMA-based ray casting without per-query division.
struct SlabIndex {
    /// AABB: [x_min, y_min, x_max, y_max]
    bbox: [f64; 4],
    /// `num_slabs / y_range` for O(1) slab lookup
    inv_slab_height: f64,
    /// Per-slab edge data: `[slope, x_base, yi, yj, ...]`.
    /// Inline buffer holds up to 32 edges (128 f64s = 1KB) without heap alloc.
    slabs: Vec<SmallVec<[f64; 128]>>,
}

impl SlabIndex {
    /// Build a slab index from a polygon.
    ///
    /// Uses sqrt(N) slabs for O(N) build with low constant factor.
    fn build(polygon: &[Point2<f64>]) -> Self {
        let n = polygon.len();

        let (x_min, y_min, x_max, y_max) = aabb(polygon);

        let y_range = y_max - y_min;
        if y_range == 0.0 {
            return SlabIndex {
                bbox: [x_min, y_min, x_max, y_max],
                inv_slab_height: 0.0,
                slabs: Vec::new(),
            };
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let num_slabs = ((n as f64).sqrt().ceil() as usize).min(n).max(1);
        let inv_slab_height = num_slabs as f64 / y_range;

        let mut slabs: Vec<SmallVec<[f64; 128]>> =
            (0..num_slabs).map(|_| SmallVec::new()).collect();

        for i in 0..n {
            let j = if i + 1 < n { i + 1 } else { 0 };
            let (xi, yi, xj, yj) = (polygon[i].x, polygon[i].y, polygon[j].x, polygon[j].y);
            #[allow(clippy::float_cmp)] // Exact comparison intentional: skip horizontal edges
            if yi == yj {
                continue;
            }
            let slope = (xj - xi) / (yj - yi);
            let x_base = xi - slope * yi;
            let (ey_min, ey_max) = if yi < yj { (yi, yj) } else { (yj, yi) };
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let lo = ((ey_min - y_min) * inv_slab_height) as usize;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let hi = ((ey_max - y_min) * inv_slab_height).ceil() as usize;
            for slab in &mut slabs[lo.min(num_slabs)..hi.min(num_slabs)] {
                slab.extend_from_slice(&[slope, x_base, yi, yj]);
            }
        }

        SlabIndex {
            bbox: [x_min, y_min, x_max, y_max],
            inv_slab_height,
            slabs,
        }
    }

    /// Test if a point is inside the polygon using slab lookup.
    #[inline]
    fn contains(&self, px: f64, py: f64) -> bool {
        if px < self.bbox[0] || px > self.bbox[2] || py < self.bbox[1] || py > self.bbox[3] {
            return false;
        }

        let num_slabs = self.slabs.len();
        if num_slabs == 0 {
            return false;
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let slab_idx = ((py - self.bbox[1]) * self.inv_slab_height) as usize;
        let slab_idx = slab_idx.min(num_slabs - 1);

        let slab = &self.slabs[slab_idx];
        let mut inside = false;
        for chunk in slab.chunks_exact(4) {
            let (slope, x_base, yi, yj) = (chunk[0], chunk[1], chunk[2], chunk[3]);
            if ((yi > py) != (yj > py)) && (px < slope.mul_add(py, x_base)) {
                inside = !inside;
            }
        }
        inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Deterministic xorshift64 PRNG (no `rand` dep) ----

    struct Xorshift64(u64);

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 { 1 } else { seed })
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        /// Uniform f64 in [lo, hi)
        fn next_f64(&mut self, lo: f64, hi: f64) -> f64 {
            let t = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            lo + t * (hi - lo)
        }
    }

    // ---- Random polygon generators ----

    fn random_convex_polygon(rng: &mut Xorshift64, n: usize, radius: f64) -> Vec<Point2<f64>> {
        let mut angles: Vec<f64> = (0..n)
            .map(|_| rng.next_f64(0.0, std::f64::consts::TAU))
            .collect();
        angles.sort_by(|a, b| a.total_cmp(b));
        angles
            .iter()
            .map(|&a| {
                let r = radius * rng.next_f64(0.7, 1.0);
                Point2::new(r * a.cos(), r * a.sin())
            })
            .collect()
    }

    fn random_star_polygon(rng: &mut Xorshift64, n: usize, radius: f64) -> Vec<Point2<f64>> {
        let mut angles: Vec<f64> = (0..n)
            .map(|_| rng.next_f64(0.0, std::f64::consts::TAU))
            .collect();
        angles.sort_by(|a, b| a.total_cmp(b));
        angles
            .iter()
            .map(|&a| {
                let r = radius * rng.next_f64(0.1, 1.0);
                Point2::new(r * a.cos(), r * a.sin())
            })
            .collect()
    }

    fn random_points(rng: &mut Xorshift64, n: usize, extent: f64) -> Vec<Point2<f64>> {
        (0..n)
            .map(|_| Point2::new(rng.next_f64(-extent, extent), rng.next_f64(-extent, extent)))
            .collect()
    }

    fn point_to_segment_dist(p: &Point2<f64>, a: &Point2<f64>, b: &Point2<f64>) -> f64 {
        let ab = b - a;
        let ap = p - a;
        let t = (ap.dot(&ab) / ab.dot(&ab)).clamp(0.0, 1.0);
        let proj = a + ab * t;
        (p - proj).norm()
    }

    fn point_to_polygon_dist(p: &Point2<f64>, polygon: &[Point2<f64>]) -> f64 {
        let n = polygon.len();
        (0..n)
            .map(|i| point_to_segment_dist(p, &polygon[i], &polygon[(i + 1) % n]))
            .fold(f64::MAX, f64::min)
    }

    fn random_polygon_with_holes(
        rng: &mut Xorshift64,
        n_exterior: usize,
        n_holes: usize,
        radius: f64,
    ) -> (Vec<Point2<f64>>, Vec<Vec<Point2<f64>>>) {
        let exterior = random_star_polygon(rng, n_exterior, radius);
        let mut holes = Vec::new();

        for _ in 0..n_holes {
            let mut placed = false;
            for _ in 0..20 {
                let cx = rng.next_f64(-radius * 0.6, radius * 0.6);
                let cy = rng.next_f64(-radius * 0.6, radius * 0.6);
                let center = Point2::new(cx, cy);

                if !point_in_polygon_naive(&center, &exterior) {
                    continue;
                }
                if holes
                    .iter()
                    .any(|h: &Vec<Point2<f64>>| point_in_polygon_naive(&center, h))
                {
                    continue;
                }

                let dist_to_ext = point_to_polygon_dist(&center, &exterior);
                let dist_to_holes: f64 = holes
                    .iter()
                    .map(|h: &Vec<Point2<f64>>| point_to_polygon_dist(&center, h))
                    .fold(f64::MAX, f64::min);
                let safe_r = (dist_to_ext.min(dist_to_holes) / 3.0).min(radius * 0.3);

                if safe_r < radius * 0.01 {
                    continue;
                }

                let n_hole_verts = (rng.next_u64() % 18 + 3) as usize;
                let hole: Vec<Point2<f64>> = {
                    let mut angles: Vec<f64> = (0..n_hole_verts)
                        .map(|_| rng.next_f64(0.0, std::f64::consts::TAU))
                        .collect();
                    angles.sort_by(|a, b| a.total_cmp(b));
                    angles
                        .iter()
                        .map(|&a| {
                            let r = safe_r * rng.next_f64(0.3, 0.9);
                            Point2::new(cx + r * a.cos(), cy + r * a.sin())
                        })
                        .collect()
                };

                let valid = hole.iter().all(|v| {
                    point_in_polygon_naive(v, &exterior)
                        && !holes
                            .iter()
                            .any(|h: &Vec<Point2<f64>>| point_in_polygon_naive(v, h))
                });

                if valid {
                    holes.push(hole);
                    placed = true;
                    break;
                }
            }
            if !placed {
                continue;
            }
        }

        (exterior, holes)
    }

    // ---- Basic tests ----

    #[test]
    fn test_square() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let points = vec![
            Point2::new(0.5, 0.5),  // center — inside
            Point2::new(2.0, 0.5),  // right — outside
            Point2::new(-0.1, 0.5), // left — outside
            Point2::new(0.5, -0.1), // below — outside
            Point2::new(0.5, 1.1),  // above — outside
            Point2::new(5.0, 5.0),  // far away — outside
        ];
        let result = point_in_polygon(&square, &[], &points);
        assert!(result[0], "center should be inside");
        assert!(!result[1], "right should be outside");
        assert!(!result[2], "left should be outside");
        assert!(!result[3], "below should be outside");
        assert!(!result[4], "above should be outside");
        assert!(!result[5], "far away should be outside");
    }

    #[test]
    fn test_concave_l_shape() {
        let l_shape = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 2.0),
            Point2::new(0.0, 2.0),
        ];
        let points = vec![
            Point2::new(0.5, 0.5), // inside bottom arm
            Point2::new(0.5, 1.5), // inside left arm
            Point2::new(1.5, 1.5), // concavity — outside
            Point2::new(3.0, 0.5), // fully outside
        ];
        let result = point_in_polygon(&l_shape, &[], &points);
        assert!(result[0], "bottom arm inside");
        assert!(result[1], "left arm inside");
        assert!(!result[2], "concavity should be outside");
        assert!(!result[3], "fully outside");
    }

    #[test]
    fn test_degenerate() {
        let line = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        let result = point_in_polygon(&line, &[], &[Point2::new(0.5, 0.0)]);
        assert!(!result[0]);

        let empty: Vec<Point2<f64>> = vec![];
        let result = point_in_polygon(&empty, &[], &[Point2::new(0.0, 0.0)]);
        assert!(!result[0]);

        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let result = point_in_polygon(&square, &[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_point_in_polygon_with_holes() {
        let exterior = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        let hole = vec![
            Point2::new(3.0, 3.0),
            Point2::new(7.0, 3.0),
            Point2::new(7.0, 7.0),
            Point2::new(3.0, 7.0),
        ];
        let holes: Vec<&[Point2<f64>]> = vec![&hole];

        let points = vec![
            Point2::new(5.0, 5.0),  // in hole — false
            Point2::new(1.0, 1.0),  // between exterior and hole — true
            Point2::new(15.0, 5.0), // outside — false
            Point2::new(9.0, 9.0),  // corner region — true
        ];
        let result = point_in_polygon(&exterior, &holes, &points);
        assert!(!result[0], "in hole should be false");
        assert!(result[1], "between boundary and hole should be true");
        assert!(!result[2], "outside should be false");
        assert!(result[3], "corner should be true");
    }

    // ---- Fuzz: slab matches naive ----

    #[test]
    fn test_slab_matches_naive_fuzz() {
        // Convex polygons
        let mut rng = Xorshift64::new(42);
        for trial in 0..5000 {
            let n_verts = (rng.next_u64() % 200 + 5) as usize;
            let polygon = random_convex_polygon(&mut rng, n_verts, 10.0);
            let points = random_points(&mut rng, 500, 15.0);

            let slab = SlabIndex::build(&polygon);
            for (i, pt) in points.iter().enumerate() {
                let naive = point_in_polygon_naive(pt, &polygon);
                let slab_result = slab.contains(pt.x, pt.y);
                assert_eq!(
                    slab_result, naive,
                    "convex trial {trial}, point {i} ({}, {}): slab={slab_result} naive={naive}, {n_verts}-gon",
                    pt.x, pt.y,
                );
            }
        }

        // Star-shaped (concave) polygons
        let mut rng = Xorshift64::new(123);
        for trial in 0..5000 {
            let n_verts = (rng.next_u64() % 200 + 5) as usize;
            let polygon = random_star_polygon(&mut rng, n_verts, 10.0);
            let points = random_points(&mut rng, 500, 15.0);

            let slab = SlabIndex::build(&polygon);
            for (i, pt) in points.iter().enumerate() {
                let naive = point_in_polygon_naive(pt, &polygon);
                let slab_result = slab.contains(pt.x, pt.y);
                assert_eq!(
                    slab_result, naive,
                    "star trial {trial}, point {i} ({}, {}): slab={slab_result} naive={naive}, {n_verts}-gon",
                    pt.x, pt.y,
                );
            }
        }

        // Polygons with holes
        let mut rng = Xorshift64::new(999);
        for trial in 0..2000 {
            let n_ext = (rng.next_u64() % 80 + 10) as usize;
            let n_holes = (rng.next_u64() % 8 + 1) as usize;
            let (exterior, holes_owned) = random_polygon_with_holes(&mut rng, n_ext, n_holes, 10.0);
            let holes_ref: Vec<&[Point2<f64>]> = holes_owned.iter().map(|h| h.as_slice()).collect();
            let points = random_points(&mut rng, 500, 15.0);

            let batch = point_in_polygon(&exterior, &holes_ref, &points);

            for (i, pt) in points.iter().enumerate() {
                let mut naive = point_in_polygon_naive(pt, &exterior);
                if naive {
                    for hole in &holes_owned {
                        if point_in_polygon_naive(pt, hole) {
                            naive = false;
                            break;
                        }
                    }
                }
                assert_eq!(
                    batch[i], naive,
                    "holes trial {trial}, point {i} ({}, {}): batch={} naive={naive}",
                    pt.x, pt.y, batch[i],
                );
            }
        }

        // Regular circle polygons at various vertex counts (from old test_batch_matches_naive)
        for n_verts in [10, 100, 1000] {
            let polygon: Vec<Point2<f64>> = (0..n_verts)
                .map(|i| {
                    let angle = 2.0 * std::f64::consts::PI * i as f64 / n_verts as f64;
                    Point2::new(angle.cos(), angle.sin())
                })
                .collect();

            let points: Vec<Point2<f64>> = (0..1000)
                .map(|i| {
                    let x = ((i * 7919 + 104729) % 40000) as f64 / 10000.0 - 2.0;
                    let y = ((i * 6271 + 87641) % 40000) as f64 / 10000.0 - 2.0;
                    Point2::new(x, y)
                })
                .collect();

            let batch_result = point_in_polygon(&polygon, &[], &points);
            for (i, pt) in points.iter().enumerate() {
                let naive = point_in_polygon_naive(pt, &polygon);
                assert_eq!(
                    batch_result[i], naive,
                    "circle mismatch at point {i} ({}, {}) with {n_verts}-gon",
                    pt.x, pt.y
                );
            }
        }
    }

    // ---- Fuzz: small-M exercises AABB+naive path (M < 64) ----

    #[test]
    fn test_small_m_fuzz() {
        let mut rng = Xorshift64::new(314159);
        for trial in 0..5000 {
            let n_verts = (rng.next_u64() % 200 + 5) as usize;
            let polygon = random_star_polygon(&mut rng, n_verts, 10.0);
            let m = (rng.next_u64() % 28 + 3) as usize; // M in 3..30

            let points = random_points(&mut rng, m, 15.0);
            let batch = point_in_polygon(&polygon, &[], &points);

            for (i, pt) in points.iter().enumerate() {
                let naive = point_in_polygon_naive(pt, &polygon);
                assert_eq!(
                    batch[i], naive,
                    "small-M trial {trial}, point {i} ({}, {}): batch={} naive={naive}, M={m}, N={n_verts}",
                    pt.x, pt.y, batch[i],
                );
            }
        }
    }

    // ---- Cross-check against Polygon2D::contains() oracle ----

    #[test]
    fn test_cross_check_polygon2d_oracle() {
        use crate::path::Polygon2D;

        let mut rng = Xorshift64::new(7777);
        for trial in 0..1000 {
            let n_ext = (rng.next_u64() % 80 + 10) as usize;
            let n_holes = (rng.next_u64() % 6) as usize;
            let (exterior, holes_owned) = random_polygon_with_holes(&mut rng, n_ext, n_holes, 10.0);
            let holes_ref: Vec<&[Point2<f64>]> = holes_owned.iter().map(|h| h.as_slice()).collect();
            let points = random_points(&mut rng, 200, 15.0);

            let batch = point_in_polygon(&exterior, &holes_ref, &points);

            let poly2d = Polygon2D::with_holes(exterior.clone(), holes_owned.clone());

            let oracle = poly2d.contains(&points);
            for (i, pt) in points.iter().enumerate() {
                assert_eq!(
                    batch[i], oracle[i],
                    "trial {trial}, point {i} ({}, {}): batch={} oracle={}",
                    pt.x, pt.y, batch[i], oracle[i],
                );
            }
        }
    }

    // ---- Adversarial edge cases ----

    #[test]
    fn test_shared_y_coordinates() {
        let polygon = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(3.0, 1.0),
            Point2::new(2.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let points = vec![
            Point2::new(0.5, 0.5),
            Point2::new(1.5, 0.5),
            Point2::new(2.5, 0.5),
            Point2::new(-0.5, 0.5),
            Point2::new(3.5, 0.5),
        ];
        let slab = SlabIndex::build(&polygon);
        for (i, pt) in points.iter().enumerate() {
            let naive = point_in_polygon_naive(pt, &polygon);
            let slab_r = slab.contains(pt.x, pt.y);
            assert_eq!(
                slab_r, naive,
                "shared-y point {i}: slab={slab_r} naive={naive}"
            );
        }
    }

    #[test]
    fn test_thin_polygon() {
        let polygon = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 0.001),
            Point2::new(0.0, 0.001),
        ];
        let points = vec![
            Point2::new(5.0, 0.0005),  // inside
            Point2::new(5.0, 0.01),    // outside
            Point2::new(-1.0, 0.0005), // outside
        ];
        let result = point_in_polygon(&polygon, &[], &points);
        let naive: Vec<bool> = points
            .iter()
            .map(|p| point_in_polygon_naive(p, &polygon))
            .collect();
        assert_eq!(result, naive);
    }

    #[test]
    fn test_donut() {
        let n = 64;
        let outer: Vec<Point2<f64>> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                Point2::new(10.0 * a.cos(), 10.0 * a.sin())
            })
            .collect();
        let inner: Vec<Point2<f64>> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                Point2::new(5.0 * a.cos(), 5.0 * a.sin())
            })
            .collect();
        let holes: Vec<&[Point2<f64>]> = vec![&inner];

        let mut rng = Xorshift64::new(555);
        let points = random_points(&mut rng, 500, 12.0);

        let batch = point_in_polygon(&outer, &holes, &points);
        for (i, pt) in points.iter().enumerate() {
            let mut naive = point_in_polygon_naive(pt, &outer);
            if naive && point_in_polygon_naive(pt, &inner) {
                naive = false;
            }
            assert_eq!(batch[i], naive, "donut point {i}");
        }
    }

    #[test]
    fn test_large_hole() {
        let outer = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        let hole = vec![
            Point2::new(0.5, 0.5),
            Point2::new(9.5, 0.5),
            Point2::new(9.5, 9.5),
            Point2::new(0.5, 9.5),
        ];
        let holes: Vec<&[Point2<f64>]> = vec![&hole];

        let mut rng = Xorshift64::new(888);
        let points = random_points(&mut rng, 500, 12.0);

        let points: Vec<Point2<f64>> = points
            .iter()
            .map(|p| Point2::new(p.x.abs() % 12.0 - 1.0, p.y.abs() % 12.0 - 1.0))
            .collect();

        let batch = point_in_polygon(&outer, &holes, &points);
        for (i, pt) in points.iter().enumerate() {
            let mut naive = point_in_polygon_naive(pt, &outer);
            if naive && point_in_polygon_naive(pt, &hole) {
                naive = false;
            }
            assert_eq!(batch[i], naive, "large-hole point {i}");
        }
    }

    // ---- Benchmarks (run manually with --ignored) ----

    #[test]
    #[ignore]
    fn test_large_polygon_perf() {
        use std::f64::consts::PI;

        let n = 50_000;
        let polygon: Vec<Point2<f64>> = (0..n)
            .map(|i| {
                let angle = 2.0 * PI * i as f64 / n as f64;
                Point2::new(angle.cos(), angle.sin())
            })
            .collect();

        let m = 50_000;
        let points: Vec<Point2<f64>> = (0..m)
            .map(|i| {
                let x = ((i * 7919 + 104729) % 30000) as f64 / 10000.0 - 1.5;
                let y = ((i * 6271 + 87641) % 30000) as f64 / 10000.0 - 1.5;
                Point2::new(x, y)
            })
            .collect();

        let t = std::time::Instant::now();
        let batch_result = point_in_polygon(&polygon, &[], &points);
        let slab_time = t.elapsed();
        eprintln!("slab: {slab_time:?} for {n}x{m}");

        for i in (0..m).step_by(500) {
            let naive = point_in_polygon_naive(&points[i], &polygon);
            assert_eq!(batch_result[i], naive, "mismatch at sample point {i}");
        }

        assert!(
            slab_time.as_millis() < 2000,
            "slab took {slab_time:?}, expected < 2s"
        );
    }

    #[test]
    #[ignore]
    fn test_threshold_determination() {
        use std::f64::consts::PI;

        // Sweep polygon sizes representative of real use (hex_grid: N=30-150)
        // and stress test (N=1000+), with point counts from 1 to 10000.
        let polygon_sizes = [10, 50, 100, 500, 1000, 5000];
        let point_counts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];

        eprintln!(
            "\n{:>6} {:>6} {:>10} {:>12} {:>12} {:>7}",
            "N", "M", "product", "naive", "slab", "winner"
        );
        eprintln!("{}", "-".repeat(62));

        for &n in &polygon_sizes {
            let polygon: Vec<Point2<f64>> = (0..n)
                .map(|i| {
                    let angle = 2.0 * PI * i as f64 / n as f64;
                    Point2::new(angle.cos(), angle.sin())
                })
                .collect();

            let mut crossover = None;
            let mut prev_winner_naive = true;

            for &m in &point_counts {
                let points: Vec<Point2<f64>> = (0..m)
                    .map(|i| {
                        let x = ((i * 7919 + 104729) % 30000) as f64 / 10000.0 - 1.5;
                        let y = ((i * 6271 + 87641) % 30000) as f64 / 10000.0 - 1.5;
                        Point2::new(x, y)
                    })
                    .collect();

                // 3 runs, take min to reduce noise
                let mut best_naive = std::time::Duration::MAX;
                let mut best_slab = std::time::Duration::MAX;
                for _ in 0..3 {
                    let t = std::time::Instant::now();
                    let _: Vec<bool> = points
                        .iter()
                        .map(|p| point_in_polygon_naive(p, &polygon))
                        .collect();
                    best_naive = best_naive.min(t.elapsed());

                    // Build + query (what point_in_polygon actually does)
                    let t = std::time::Instant::now();
                    let s = SlabIndex::build(&polygon);
                    let _: Vec<bool> = points.iter().map(|p| s.contains(p.x, p.y)).collect();
                    best_slab = best_slab.min(t.elapsed());
                }

                let naive_wins = best_naive < best_slab;
                let winner = if naive_wins { "NAIVE" } else { "SLAB" };
                eprintln!(
                    "{n:>6} {m:>6} {:>10} {:>12?} {:>12?} {:>7}",
                    n * m,
                    best_naive,
                    best_slab,
                    winner,
                );

                if prev_winner_naive && !naive_wins && crossover.is_none() {
                    crossover = Some(n * m);
                }
                prev_winner_naive = naive_wins;
            }

            if let Some(c) = crossover {
                let m_approx = c / n;
                eprintln!("  -> crossover at m*n ~ {c} (M ~ {m_approx})");
            }
            eprintln!();
        }
    }
}
