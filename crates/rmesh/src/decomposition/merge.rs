//! Priority-queue greedy merge of convex hulls.
//!
//! Merges pairs of hulls with lowest concavity cost until the target
//! hull count is reached. Uses AABB fast-path for non-overlapping pairs
//! and lazy deletion for efficient priority queue management.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ahash::AHashMap;
use nalgebra::Point3;
use rayon::prelude::*;

use super::ConvexHull;
use crate::bounds::Bounds3;

/// A pair of hulls with their merge cost, for the priority queue.
#[derive(Debug, Clone)]
struct HullPair {
    cost: f64,
    id_a: u32,
    id_b: u32,
}

impl PartialEq for HullPair {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for HullPair {}

impl PartialOrd for HullPair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HullPair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost
            .partial_cmp(&other.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Merge cost between two hulls.
///
/// `concavity = |vol_A + vol_B - vol_combined| / vol_total`
///
/// If AABBs don't overlap, uses AABB union volume as a fast upper bound
/// (which is always >= true combined hull volume).
fn merge_cost(
    a: &ConvexHull,
    b: &ConvexHull,
    aabb_a: &Bounds3,
    aabb_b: &Bounds3,
    total_volume: f64,
) -> f64 {
    if total_volume < 1e-15 {
        return 0.0;
    }

    let combined_vol = if aabb_a.overlaps(aabb_b) {
        // Slow path: compute actual combined hull volume
        let mut combined_points = Vec::with_capacity(a.vertices.len() + b.vertices.len());
        combined_points.extend_from_slice(&a.vertices);
        combined_points.extend_from_slice(&b.vertices);
        match crate::convex::convex_hull_3d(&combined_points) {
            Ok(faces) => crate::triangles::inertia::volume(&combined_points, &faces).abs(),
            Err(_) => {
                // Fallback to AABB
                aabb_a.union(aabb_b).volume()
            }
        }
    } else {
        // Fast path: AABB union volume (no hull computation needed)
        aabb_a.union(aabb_b).volume()
    };

    (a.volume + b.volume - combined_vol).abs() / total_volume
}

/// Merge a combined set of points into a single ConvexHull.
fn merge_hulls(a: &ConvexHull, b: &ConvexHull) -> Option<ConvexHull> {
    let mut combined = Vec::with_capacity(a.vertices.len() + b.vertices.len());
    combined.extend_from_slice(&a.vertices);
    combined.extend_from_slice(&b.vertices);

    let Ok(faces) = crate::convex::convex_hull_3d(&combined) else {
        return None;
    };

    // Re-index to only include hull vertices
    let mut used = vec![false; combined.len()];
    for [a, b, c] in &faces {
        used[*a] = true;
        used[*b] = true;
        used[*c] = true;
    }

    let mut index_map = vec![0usize; combined.len()];
    let mut vertices = Vec::new();
    for (i, &u) in used.iter().enumerate() {
        if u {
            index_map[i] = vertices.len();
            vertices.push(combined[i]);
        }
    }

    let faces: Vec<[usize; 3]> = faces
        .iter()
        .map(|[a, b, c]| [index_map[*a], index_map[*b], index_map[*c]])
        .collect();

    let volume = crate::triangles::inertia::volume(&vertices, &faces).abs();
    let center = {
        let sum = vertices
            .iter()
            .fold(nalgebra::Vector3::zeros(), |acc, p| acc + p.coords);
        Point3::from(sum / vertices.len() as f64)
    };

    Some(ConvexHull {
        vertices,
        faces,
        volume,
        center,
    })
}

/// Greedily merge hulls until the target count is reached.
///
/// Uses a min-heap priority queue with lazy deletion for efficiency.
/// Non-overlapping AABB pairs use a fast volume approximation.
pub fn greedy_merge(mut hulls: Vec<ConvexHull>, max_hulls: u32) -> Vec<ConvexHull> {
    if hulls.len() <= max_hulls as usize {
        return hulls;
    }

    // Cap fragments to prevent O(n^2) explosion
    const MAX_FRAGMENTS: usize = 8192;
    if hulls.len() > MAX_FRAGMENTS {
        // Sort by volume descending, keep the largest
        hulls.sort_by(|a, b| {
            b.volume
                .partial_cmp(&a.volume)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hulls.truncate(MAX_FRAGMENTS);
    }

    let total_volume: f64 = hulls.iter().map(|h| h.volume).sum();

    // Build hull map and AABB cache
    let mut hull_map: AHashMap<u32, ConvexHull> = AHashMap::new();
    let mut aabb_map: AHashMap<u32, Bounds3> = AHashMap::new();
    let mut next_id: u32 = 0;

    for hull in hulls {
        let id = next_id;
        next_id += 1;
        aabb_map.insert(id, Bounds3::from_points(&hull.vertices).unwrap());
        hull_map.insert(id, hull);
    }

    // Build initial priority queue
    let mut heap: BinaryHeap<Reverse<HullPair>> = BinaryHeap::new();
    let ids: Vec<u32> = hull_map.keys().copied().collect();

    // Compute all pairwise costs in parallel
    let pairs: Vec<(u32, u32)> = ids
        .iter()
        .enumerate()
        .flat_map(|(i, &a)| ids[i + 1..].iter().map(move |&b| (a, b)))
        .collect();

    let costs: Vec<(u32, u32, f64)> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let cost = merge_cost(
                &hull_map[&a],
                &hull_map[&b],
                &aabb_map[&a],
                &aabb_map[&b],
                total_volume,
            );
            (a, b, cost)
        })
        .collect();

    for (a, b, cost) in costs {
        heap.push(Reverse(HullPair {
            cost,
            id_a: a,
            id_b: b,
        }));
    }

    // Greedy merge loop
    while hull_map.len() > max_hulls as usize {
        let Some(Reverse(pair)) = heap.pop() else {
            break;
        };

        // Lazy deletion: skip if either hull no longer exists
        if !hull_map.contains_key(&pair.id_a) || !hull_map.contains_key(&pair.id_b) {
            continue;
        }

        let hull_a = hull_map.remove(&pair.id_a).expect("checked above");
        let hull_b = hull_map.remove(&pair.id_b).expect("checked above");
        aabb_map.remove(&pair.id_a);
        aabb_map.remove(&pair.id_b);

        let Some(merged) = merge_hulls(&hull_a, &hull_b) else {
            // Can't merge, keep the larger one
            if hull_a.volume >= hull_b.volume {
                hull_map.insert(pair.id_a, hull_a);
                aabb_map.insert(
                    pair.id_a,
                    Bounds3::from_points(&hull_map[&pair.id_a].vertices).unwrap(),
                );
            } else {
                hull_map.insert(pair.id_b, hull_b);
                aabb_map.insert(
                    pair.id_b,
                    Bounds3::from_points(&hull_map[&pair.id_b].vertices).unwrap(),
                );
            }
            continue;
        };

        let new_id = next_id;
        next_id += 1;
        let new_aabb = Bounds3::from_points(&merged.vertices).unwrap();
        aabb_map.insert(new_id, new_aabb);
        hull_map.insert(new_id, merged);

        // Recompute costs to new hull in parallel
        let remaining_ids: Vec<u32> = hull_map.keys().filter(|&&k| k != new_id).copied().collect();

        let new_costs: Vec<(u32, f64)> = remaining_ids
            .par_iter()
            .map(|&other_id| {
                let cost = merge_cost(
                    &hull_map[&new_id],
                    &hull_map[&other_id],
                    &aabb_map[&new_id],
                    &aabb_map[&other_id],
                    total_volume,
                );
                (other_id, cost)
            })
            .collect();

        for (other_id, cost) in new_costs {
            heap.push(Reverse(HullPair {
                cost,
                id_a: new_id,
                id_b: other_id,
            }));
        }
    }

    hull_map.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decomposition::ConvexHull;

    fn make_box_hull(min: Point3<f64>, max: Point3<f64>) -> ConvexHull {
        let vertices = vec![
            Point3::new(min.x, min.y, min.z),
            Point3::new(max.x, min.y, min.z),
            Point3::new(max.x, max.y, min.z),
            Point3::new(min.x, max.y, min.z),
            Point3::new(min.x, min.y, max.z),
            Point3::new(max.x, min.y, max.z),
            Point3::new(max.x, max.y, max.z),
            Point3::new(min.x, max.y, max.z),
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let e = max - min;
        let volume = e.x * e.y * e.z;
        let center = Point3::from((min.coords + max.coords) * 0.5);
        ConvexHull {
            vertices,
            faces,
            volume,
            center,
        }
    }

    #[test]
    fn test_merge_cost_identical() {
        let hull = make_box_hull(Point3::origin(), Point3::new(1.0, 1.0, 1.0));
        let aabb = Bounds3::from_points(&hull.vertices).unwrap();
        let cost = merge_cost(&hull, &hull, &aabb, &aabb, 2.0);
        // Merging identical hulls: combined vol == each vol, cost should be ~vol/total
        assert!(cost < 1.0);
    }

    #[test]
    fn test_merge_cost_distant() {
        let a = make_box_hull(Point3::origin(), Point3::new(1.0, 1.0, 1.0));
        let b = make_box_hull(Point3::new(10.0, 10.0, 10.0), Point3::new(11.0, 11.0, 11.0));
        let aabb_a = Bounds3::from_points(&a.vertices).unwrap();
        let aabb_b = Bounds3::from_points(&b.vertices).unwrap();
        let cost = merge_cost(&a, &b, &aabb_a, &aabb_b, 2.0);
        // Distant hulls should have high merge cost
        assert!(cost > 0.1);
    }

    #[test]
    fn test_merge_reduces_count() {
        let hulls: Vec<ConvexHull> = (0..10)
            .map(|i| {
                let x = f64::from(i) * 2.0;
                make_box_hull(Point3::new(x, 0.0, 0.0), Point3::new(x + 1.0, 1.0, 1.0))
            })
            .collect();

        let result = greedy_merge(hulls, 5);
        assert!(
            result.len() <= 5,
            "Expected <= 5 hulls, got {}",
            result.len()
        );
        assert!(!result.is_empty());
    }

    #[test]
    fn test_merge_already_under_limit() {
        let hulls: Vec<ConvexHull> = (0..3)
            .map(|i| {
                let x = f64::from(i) * 2.0;
                make_box_hull(Point3::new(x, 0.0, 0.0), Point3::new(x + 1.0, 1.0, 1.0))
            })
            .collect();

        let result = greedy_merge(hulls, 5);
        assert_eq!(result.len(), 3);
    }
}
