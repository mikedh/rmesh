//! Shrink-wrap hull vertices to the original mesh surface.
//!
//! For each hull vertex, queries the BVH for the closest point on the
//! original mesh surface within a threshold distance. If found, the
//! vertex is snapped to the surface. The hull is then recomputed to
//! maintain convexity.

use nalgebra::Point3;
use rayon::prelude::*;

use crate::triangles::bvh::TriangleBvh;

use super::ConvexHull;

/// Project hull vertices to the nearest points on the source mesh surface,
/// then recompute the convex hull to maintain convexity.
///
/// `max_dist` is the maximum distance to search for a surface point
/// (typically one or two voxel widths).
pub fn shrink_wrap(
    hull: &ConvexHull,
    bvh: &TriangleBvh,
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    max_dist: f64,
) -> ConvexHull {
    // Project each hull vertex to the nearest surface point
    let projected: Vec<Point3<f64>> = hull
        .vertices
        .par_iter()
        .map(|v| {
            match bvh.closest_point(v, max_dist, vertices, faces) {
                Some(hit) => hit.point,
                None => *v, // Keep original if no surface point found
            }
        })
        .collect();

    // Recompute convex hull from projected points
    match crate::convex::convex_hull_3d(&projected) {
        Ok(new_faces) => {
            // Re-index to only hull vertices
            let mut used = vec![false; projected.len()];
            for [a, b, c] in &new_faces {
                used[*a] = true;
                used[*b] = true;
                used[*c] = true;
            }

            let mut index_map = vec![0usize; projected.len()];
            let mut final_vertices = Vec::new();
            for (i, &u) in used.iter().enumerate() {
                if u {
                    index_map[i] = final_vertices.len();
                    final_vertices.push(projected[i]);
                }
            }

            let final_faces: Vec<[usize; 3]> = new_faces
                .iter()
                .map(|[a, b, c]| [index_map[*a], index_map[*b], index_map[*c]])
                .collect();

            let volume = crate::triangles::inertia::volume(&final_vertices, &final_faces).abs();
            let center = {
                let sum = final_vertices
                    .iter()
                    .fold(nalgebra::Vector3::zeros(), |acc, p| acc + p.coords);
                Point3::from(sum / final_vertices.len() as f64)
            };

            ConvexHull {
                vertices: final_vertices,
                faces: final_faces,
                volume,
                center,
            }
        }
        Err(_) => {
            // If recompute fails, return original hull unchanged
            hull.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shrink_wrap_identity() {
        // When hull vertices ARE the mesh surface, shrink_wrap should barely change them
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

        let bvh = TriangleBvh::build(&vertices, &faces);

        let hull = ConvexHull {
            vertices: vertices.clone(),
            faces: faces.clone(),
            volume: 1.0,
            center: Point3::new(0.5, 0.5, 0.5),
        };

        let result = shrink_wrap(&hull, &bvh, &vertices, &faces, 0.5);
        assert!(!result.vertices.is_empty());
        assert!(!result.faces.is_empty());
        assert!(result.volume > 0.0);
    }

    #[test]
    fn test_shrink_wrap_no_nearby_surface() {
        // Hull far from mesh - vertices should not change
        let mesh_vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let mesh_faces = vec![[0, 1, 2]];
        let bvh = TriangleBvh::build(&mesh_vertices, &mesh_faces);

        let hull = ConvexHull {
            vertices: vec![
                Point3::new(100.0, 100.0, 100.0),
                Point3::new(101.0, 100.0, 100.0),
                Point3::new(100.0, 101.0, 100.0),
                Point3::new(100.0, 100.0, 101.0),
            ],
            faces: vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
            volume: 1.0 / 6.0,
            center: Point3::new(100.25, 100.25, 100.25),
        };

        let result = shrink_wrap(&hull, &bvh, &mesh_vertices, &mesh_faces, 0.5);
        // Should return roughly the same hull since surface is far away
        assert!(!result.vertices.is_empty());
    }
}
