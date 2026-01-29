use nalgebra::Point2;

/// Compute the 2D convex hull using Andrew's monotone chain algorithm.
///
/// Returns CCW-ordered edges as index pairs `[a, b]` into `points`,
/// forming a closed polygon edge loop. Returns an empty vec for
/// fewer than 2 distinct points.
pub fn convex_hull_2d(points: &[Point2<f64>]) -> Vec<[usize; 2]> {
    let n = points.len();
    if n < 2 {
        return Vec::new();
    }

    // Sort indices by x, then y
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| {
        points[a]
            .x
            .partial_cmp(&points[b].x)
            .unwrap()
            .then_with(|| points[a].y.partial_cmp(&points[b].y).unwrap())
    });

    // Remove duplicates from sorted order
    idx.dedup_by(|a, b| {
        (points[*a].x - points[*b].x).abs() < 1e-14
            && (points[*a].y - points[*b].y).abs() < 1e-14
    });
    let m = idx.len();
    if m < 2 {
        return Vec::new();
    }

    // 2D cross product of vectors OA and OB where O = points[o], A = points[a], B = points[b]
    let cross = |o: usize, a: usize, b: usize| -> f64 {
        let ox = points[a].x - points[o].x;
        let oy = points[a].y - points[o].y;
        let bx = points[b].x - points[o].x;
        let by = points[b].y - points[o].y;
        ox * by - oy * bx
    };

    let mut hull: Vec<usize> = Vec::with_capacity(2 * m);

    // Lower hull
    for &i in &idx {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], i) <= 0.0 {
            hull.pop();
        }
        hull.push(i);
    }

    // Upper hull
    let lower_len = hull.len();
    for &i in idx.iter().rev().skip(1) {
        while hull.len() > lower_len
            && cross(hull[hull.len() - 2], hull[hull.len() - 1], i) <= 0.0
        {
            hull.pop();
        }
        hull.push(i);
    }

    // Remove the last point (same as first)
    hull.pop();

    // Handle collinear case: all points on a line
    if hull.len() < 3 {
        // Return the two endpoints as edges in both directions
        if hull.len() == 2 {
            return vec![[hull[0], hull[1]], [hull[1], hull[0]]];
        }
        return Vec::new();
    }

    // Build CCW edge loop
    let h = hull.len();
    let edges: Vec<[usize; 2]> = (0..h).map(|i| [hull[i], hull[(i + 1) % h]]).collect();

    #[cfg(test)]
    {
        assert!(
            super::is_hull_valid_2d(points, &edges),
            "convex_hull_2d: internal validation failed — a point is outside the hull"
        );
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point2;

    fn p(x: f64, y: f64) -> Point2<f64> {
        Point2::new(x, y)
    }

    #[test]
    fn test_empty() {
        let edges = convex_hull_2d(&[]);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_single_point() {
        let edges = convex_hull_2d(&[p(0.0, 0.0)]);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_two_points() {
        let pts = [p(0.0, 0.0), p(1.0, 0.0)];
        let edges = convex_hull_2d(&pts);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_triangle() {
        let pts = [p(0.0, 0.0), p(1.0, 0.0), p(0.5, 1.0)];
        let edges = convex_hull_2d(&pts);
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_square_with_interior() {
        let pts = [
            p(0.0, 0.0),
            p(1.0, 0.0),
            p(1.0, 1.0),
            p(0.0, 1.0),
            p(0.5, 0.5), // interior
        ];
        let edges = convex_hull_2d(&pts);
        assert_eq!(edges.len(), 4);

        // Verify interior point is not on hull
        let hull_verts: std::collections::HashSet<usize> =
            edges.iter().map(|e| e[0]).collect();
        assert!(!hull_verts.contains(&4));
    }

    #[test]
    fn test_collinear() {
        let pts = [p(0.0, 0.0), p(1.0, 0.0), p(2.0, 0.0), p(3.0, 0.0)];
        let edges = convex_hull_2d(&pts);
        // Collinear points produce a degenerate hull (line segment)
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_ccw_orientation() {
        let pts = [p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)];
        let edges = convex_hull_2d(&pts);
        // Verify CCW: all cross products should be positive
        for i in 0..edges.len() {
            let j = (i + 1) % edges.len();
            let a = edges[i][0];
            let b = edges[i][1];
            let c = edges[j][1];
            let ab = pts[b] - pts[a];
            let bc = pts[c] - pts[b];
            let cross = ab.x * bc.y - ab.y * bc.x;
            assert!(cross >= -1e-10, "Not CCW at edge {}: cross = {}", i, cross);
        }
    }

    #[test]
    fn test_duplicates() {
        let pts = [
            p(0.0, 0.0),
            p(0.0, 0.0),
            p(1.0, 0.0),
            p(1.0, 0.0),
            p(0.5, 1.0),
        ];
        let edges = convex_hull_2d(&pts);
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_circle_points() {
        let n = 32;
        let pts: Vec<Point2<f64>> = (0..n)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                p(a.cos(), a.sin())
            })
            .collect();
        let edges = convex_hull_2d(&pts);
        assert_eq!(edges.len(), n);
    }

    #[test]
    fn test_random_exhaustive() {
        // Simple LCG for deterministic pseudo-random numbers
        let mut seed: u64 = 42;
        let mut next_f64 = || -> f64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };

        for trial in 0..1000 {
            let n = 10 + (trial % 191); // 10-200 points
            let pts: Vec<Point2<f64>> =
                (0..n).map(|_| p(next_f64(), next_f64())).collect();
            let edges = convex_hull_2d(&pts);

            if edges.len() < 3 {
                continue; // degenerate (collinear)
            }

            // is_hull_valid_2d is checked inside convex_hull_2d via cfg(test)

            // Edge count should be consistent (closed loop)
            let hull_verts: std::collections::HashSet<usize> =
                edges.iter().map(|e| e[0]).collect();
            assert_eq!(hull_verts.len(), edges.len());

            // Verify convexity: all cross products same sign
            for i in 0..edges.len() {
                let j = (i + 1) % edges.len();
                let a = edges[i][0];
                let b = edges[i][1];
                let c = edges[j][1];
                let ab = pts[b] - pts[a];
                let bc = pts[c] - pts[b];
                let cross = ab.x * bc.y - ab.y * bc.x;
                assert!(
                    cross >= -1e-10,
                    "Trial {}: not convex at edge {}, cross = {}",
                    trial,
                    i,
                    cross
                );
            }
        }
    }
}
