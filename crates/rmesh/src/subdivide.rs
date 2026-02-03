//! Mesh subdivision algorithms.

use nalgebra::Point3;
use rayon::prelude::*;

use crate::attributes::{Attributes, Grouping};

/// Subdivide a mesh by splitting each triangle into 4 triangles.
///
/// Each edge is split at its midpoint, creating 4 smaller triangles
/// from each original triangle. This is sometimes called "mid-edge" subdivision.
///
/// If `face_attributes` is provided, all attribute fields (colors, UVs,
/// normals, tangents, groupings) are propagated: each child face inherits
/// its parent face's attributes.
///
/// # Arguments
/// * `vertices` - Original vertex positions
/// * `faces` - Original triangle faces
/// * `face_attributes` - Optional face attributes to propagate
/// * `iterations` - Number of subdivision iterations (each multiplies faces by 4)
///
/// # Returns
/// New vertices, faces, and face attributes after subdivision.
pub fn subdivide(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    face_attributes: Option<&Attributes>,
    iterations: usize,
) -> (Vec<Point3<f64>>, Vec<[usize; 3]>, Attributes) {
    if iterations == 0 || faces.is_empty() {
        return (
            vertices.to_vec(),
            faces.to_vec(),
            face_attributes.cloned().unwrap_or_default(),
        );
    }

    let mut current_vertices = vertices.to_vec();
    let mut current_faces = faces.to_vec();
    let mut current_attrs = face_attributes.cloned().unwrap_or_default();

    for _ in 0..iterations {
        let (new_verts, new_faces) = subdivide_once(&current_vertices, &current_faces);

        // Propagate face attributes: each parent face becomes 4 child faces
        // Child faces are in order: [top, right, left, center] for each parent
        let mut new_attrs = Attributes::default();

        for colors in &current_attrs.colors {
            new_attrs
                .colors
                .push(colors.iter().flat_map(|&v| [v, v, v, v]).collect());
        }
        for uv in &current_attrs.uv {
            new_attrs
                .uv
                .push(uv.iter().flat_map(|&v| [v, v, v, v]).collect());
        }
        for normals in &current_attrs.normals {
            new_attrs
                .normals
                .push(normals.iter().flat_map(|&v| [v, v, v, v]).collect());
        }
        for tangents in &current_attrs.tangents {
            new_attrs
                .tangents
                .push(tangents.iter().flat_map(|&v| [v, v, v, v]).collect());
        }
        for grouping in &current_attrs.groupings {
            new_attrs.groupings.push(Grouping {
                kind: grouping.kind.clone(),
                names: grouping.names.clone(),
                indices: grouping
                    .indices
                    .iter()
                    .flat_map(|&idx| [idx, idx, idx, idx])
                    .collect(),
            });
        }

        current_vertices = new_verts;
        current_faces = new_faces;
        current_attrs = new_attrs;
    }

    (current_vertices, current_faces, current_attrs)
}

/// Single iteration of mid-edge subdivision using sort-based unique edges.
fn subdivide_once(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
    let num_faces = faces.len();

    // Step 1: Collect all edges (sorted) with their position in face order.
    // Each face contributes 3 edges, so edge at position i corresponds to
    // face i/3, edge i%3 within that face.
    // Edge format: (sorted_edge, original_position)
    let mut edges_with_pos: Vec<([usize; 2], usize)> = faces
        .par_iter()
        .enumerate()
        .flat_map_iter(|(face_idx, &[v0, v1, v2])| {
            let base = face_idx * 3;
            [
                ([v0.min(v1), v0.max(v1)], base),     // edge 0: v0-v1
                ([v1.min(v2), v1.max(v2)], base + 1), // edge 1: v1-v2
                ([v2.min(v0), v2.max(v0)], base + 2), // edge 2: v2-v0
            ]
            .into_iter()
        })
        .collect();

    // Step 2: Sort by edge to group duplicates
    edges_with_pos.par_sort_unstable_by_key(|(edge, _)| *edge);

    // Step 3: Assign midpoint indices to unique edges and build inverse mapping.
    // inverse[original_position] = midpoint_vertex_index
    let mut inverse = vec![0usize; num_faces * 3];
    let mut unique_edges: Vec<[usize; 2]> = Vec::with_capacity(num_faces * 3 / 2); // rough estimate

    let mut i = 0;
    while i < edges_with_pos.len() {
        let current_edge = edges_with_pos[i].0;
        let midpoint_idx = vertices.len() + unique_edges.len();
        unique_edges.push(current_edge);

        // Assign this midpoint index to all edges in this group
        while i < edges_with_pos.len() && edges_with_pos[i].0 == current_edge {
            inverse[edges_with_pos[i].1] = midpoint_idx;
            i += 1;
        }
    }

    // Step 4: Compute midpoint positions in parallel
    let midpoints: Vec<Point3<f64>> = unique_edges
        .par_iter()
        .map(|&[v0, v1]| {
            let p0 = vertices[v0];
            let p1 = vertices[v1];
            Point3::from((p0.coords + p1.coords) * 0.5)
        })
        .collect();

    // Step 5: Build new vertex array
    let mut new_vertices = Vec::with_capacity(vertices.len() + midpoints.len());
    new_vertices.extend_from_slice(vertices);
    new_vertices.extend(midpoints);

    // Step 6: Build new faces in parallel
    // Each original triangle becomes 4 triangles:
    //
    //       v0
    //      /  \
    //     m2--m0
    //    /  \/  \
    //   v2--m1--v1
    //
    // New triangles:
    //   [v0, m0, m2]  - top
    //   [m0, v1, m1]  - right
    //   [m2, m1, v2]  - left
    //   [m0, m1, m2]  - center

    let new_faces: Vec<[usize; 3]> = faces
        .par_iter()
        .enumerate()
        .flat_map_iter(|(face_idx, &[v0, v1, v2])| {
            let base = face_idx * 3;
            let m0 = inverse[base]; // midpoint of v0-v1
            let m1 = inverse[base + 1]; // midpoint of v1-v2
            let m2 = inverse[base + 2]; // midpoint of v2-v0

            [
                [v0, m0, m2], // top
                [m0, v1, m1], // right
                [m2, m1, v2], // left
                [m0, m1, m2], // center
            ]
            .into_iter()
        })
        .collect();

    (new_vertices, new_faces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::{Attributes, GroupingKind};
    use crate::creation::create_box;
    use crate::mesh::Trimesh;
    use crate::triangles::inertia::volume;
    use approx::assert_relative_eq;
    use nalgebra::Vector4;

    #[test]
    fn test_subdivide_face_count() {
        let cube = create_box(&[1.0, 1.0, 1.0]);

        // Each iteration multiplies face count by 4
        let (_, faces1, _) = subdivide(&cube.vertices, &cube.faces, None, 1);
        assert_eq!(faces1.len(), 12 * 4);

        let (_, faces2, _) = subdivide(&cube.vertices, &cube.faces, None, 2);
        assert_eq!(faces2.len(), 12 * 16);

        let (_, faces3, _) = subdivide(&cube.vertices, &cube.faces, None, 3);
        assert_eq!(faces3.len(), 12 * 64);
    }

    #[test]
    fn test_subdivide_preserves_volume() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let original_volume = volume(&cube.vertices, &cube.faces).abs();

        for iterations in 1..=3 {
            let (verts, faces, _) = subdivide(&cube.vertices, &cube.faces, None, iterations);
            let new_volume = volume(&verts, &faces).abs();
            assert_relative_eq!(new_volume, original_volume, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_subdivide_preserves_watertight() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        assert!(cube.is_watertight());

        for iterations in 1..=3 {
            let (verts, faces, _) = subdivide(&cube.vertices, &cube.faces, None, iterations);
            let mesh = Trimesh::new(verts, faces, None, None).unwrap();
            assert!(
                mesh.is_watertight(),
                "Lost watertight at iteration {}",
                iterations
            );
        }
    }

    #[test]
    fn test_subdivide_zero_iterations() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let (verts, faces, _) = subdivide(&cube.vertices, &cube.faces, None, 0);
        assert_eq!(verts.len(), cube.vertices.len());
        assert_eq!(faces.len(), cube.faces.len());
    }

    #[test]
    fn test_subdivide_no_degenerate_faces() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let (verts, faces, _) = subdivide(&cube.vertices, &cube.faces, None, 2);

        for (i, [v0, v1, v2]) in faces.iter().enumerate() {
            assert!(v0 != v1 && v1 != v2 && v2 != v0, "Degenerate face {}", i);
            assert!(*v0 < verts.len() && *v1 < verts.len() && *v2 < verts.len());
        }
    }

    #[test]
    fn test_subdivide_large_mesh() {
        // Create a large mesh to test performance
        let cube = create_box(&[1.0, 1.0, 1.0]);

        // 5 iterations: 12 * 4^5 = 12,288 faces
        let (verts, faces, _) = subdivide(&cube.vertices, &cube.faces, None, 5);
        assert_eq!(faces.len(), 12 * 1024);

        // Verify it's still valid
        let vol = volume(&verts, &faces).abs();
        assert_relative_eq!(vol, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_subdivide_propagates_face_colors() {
        let cube = create_box(&[1.0, 1.0, 1.0]);

        let face_colors: Vec<Vector4<u8>> = (0..12)
            .map(|i| {
                let side = i / 2;
                Vector4::new((side * 40) as u8, ((5 - side) * 40) as u8, 100, 255)
            })
            .collect();

        let mut face_attrs = Attributes::default();
        face_attrs.colors.push(face_colors.clone());

        // Subdivide once: 12 faces -> 48 faces
        let (_, new_faces, new_attrs) =
            subdivide(&cube.vertices, &cube.faces, Some(&face_attrs), 1);

        assert_eq!(new_faces.len(), 48);
        assert_eq!(new_attrs.colors.len(), 1);
        assert_eq!(new_attrs.colors[0].len(), 48);

        for (parent_idx, parent_color) in face_colors.iter().enumerate() {
            for child_offset in 0..4 {
                let child_idx = parent_idx * 4 + child_offset;
                assert_eq!(
                    new_attrs.colors[0][child_idx], *parent_color,
                    "Child face {} should have same color as parent face {}",
                    child_idx, parent_idx
                );
            }
        }

        // Subdivide twice: 12 faces -> 192 faces
        let (_, new_faces2, new_attrs2) =
            subdivide(&cube.vertices, &cube.faces, Some(&face_attrs), 2);

        assert_eq!(new_faces2.len(), 192);
        assert_eq!(new_attrs2.colors[0].len(), 192);

        for (parent_idx, parent_color) in face_colors.iter().enumerate() {
            for child_offset in 0..16 {
                let child_idx = parent_idx * 16 + child_offset;
                assert_eq!(
                    new_attrs2.colors[0][child_idx], *parent_color,
                    "Child face {} should have same color as parent face {} after 2 iterations",
                    child_idx, parent_idx
                );
            }
        }
    }

    #[test]
    fn test_subdivide_propagates_groupings() {
        let cube = create_box(&[1.0, 1.0, 1.0]);

        // Assign each pair of faces to a named group (one per cube side)
        let grouping = Grouping {
            kind: GroupingKind::Group,
            names: vec![
                "side0".into(),
                "side1".into(),
                "side2".into(),
                "side3".into(),
                "side4".into(),
                "side5".into(),
            ],
            indices: (0..12).map(|i| i / 2).collect(),
        };

        let mut face_attrs = Attributes::default();
        face_attrs.groupings.push(grouping);

        let (_, new_faces, new_attrs) =
            subdivide(&cube.vertices, &cube.faces, Some(&face_attrs), 1);

        assert_eq!(new_faces.len(), 48);
        assert_eq!(new_attrs.groupings.len(), 1);
        let g = &new_attrs.groupings[0];
        assert_eq!(g.indices.len(), 48);
        assert_eq!(g.names.len(), 6);

        // Each parent face's group index should appear 4 times consecutively
        for parent_idx in 0..12 {
            let expected_group = parent_idx / 2;
            for child_offset in 0..4 {
                let child_idx = parent_idx * 4 + child_offset;
                assert_eq!(
                    g.indices[child_idx], expected_group,
                    "Child face {} should have group {} from parent face {}",
                    child_idx, expected_group, parent_idx
                );
            }
        }
    }
}
