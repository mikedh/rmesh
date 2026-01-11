//! Face and edge adjacency computations using sort-based algorithms.
//!
//! Uses sorting instead of hash maps for better cache locality on large meshes.

use rayon::prelude::*;

/// Get the 3 edges of a triangle as sorted pairs [min, max].
#[inline]
fn face_edges([i0, i1, i2]: [usize; 3]) -> [[usize; 2]; 3] {
    [
        [i0.min(i1), i0.max(i1)],
        [i1.min(i2), i1.max(i2)],
        [i2.min(i0), i2.max(i0)],
    ]
}

/// Get the 3 directed edges of a triangle (preserving winding order).
#[inline]
fn face_edges_directed([i0, i1, i2]: [usize; 3]) -> [[usize; 2]; 3] {
    [[i0, i1], [i1, i2], [i2, i0]]
}

/// Find pairs of faces that share an edge.
///
/// Uses sorting to find duplicates, which is more cache-friendly than hash maps.
pub fn face_adjacency(faces: &[[usize; 3]]) -> Vec<(usize, usize)> {
    // Build list of (sorted_edge, face_index)
    let mut edge_face: Vec<([usize; 2], usize)> = faces
        .par_iter()
        .enumerate()
        .flat_map_iter(|(face_idx, &face)| {
            face_edges(face)
                .into_iter()
                .map(move |edge| (edge, face_idx))
        })
        .collect();

    // Sort by edge - duplicates will be adjacent
    edge_face.par_sort_unstable_by_key(|(edge, _)| *edge);

    // Find adjacent pairs with same edge
    let mut adjacency = Vec::new();
    let mut i = 0;
    while i + 1 < edge_face.len() {
        if edge_face[i].0 == edge_face[i + 1].0 {
            // These faces share an edge
            adjacency.push((edge_face[i].1, edge_face[i + 1].1));
            i += 2; // Skip both (assuming manifold mesh with 2 faces per edge)
        } else {
            i += 1;
        }
    }

    adjacency
}

/// Edge with direction info for topology checking.
/// Packed as (sorted_edge, is_forward) where is_forward = (original_a < original_b)
#[derive(Clone, Copy)]
struct DirectedEdge {
    edge: [usize; 2], // sorted: [min, max]
    forward: bool,    // true if original direction was min→max
}

/// Check mesh edge topology using sort-based algorithm.
///
/// Returns (is_watertight, is_winding_consistent):
/// - `is_watertight`: every edge is shared by exactly 2 faces
/// - `is_winding_consistent`: adjacent faces traverse shared edges in opposite directions
pub fn check_topology(faces: &[[usize; 3]]) -> (bool, bool) {
    // Build list of directed edges
    let mut edges: Vec<DirectedEdge> = faces
        .par_iter()
        .flat_map_iter(|&face| {
            face_edges_directed(face).into_iter().map(|[a, b]| DirectedEdge {
                edge: [a.min(b), a.max(b)],
                forward: a < b,
            })
        })
        .collect();

    // Sort by edge
    edges.par_sort_unstable_by_key(|e| e.edge);

    // Process groups of same edge
    let mut watertight = true;
    let mut consistent = true;
    let mut i = 0;

    while i < edges.len() {
        let current_edge = edges[i].edge;
        let mut forward_count = 0u32;
        let mut reverse_count = 0u32;

        // Count all edges in this group
        while i < edges.len() && edges[i].edge == current_edge {
            if edges[i].forward {
                forward_count += 1;
            } else {
                reverse_count += 1;
            }
            i += 1;
        }

        // Check watertight: exactly 2 faces per edge
        if forward_count + reverse_count != 2 {
            watertight = false;
        }

        // Check consistent winding: each direction appears at most once
        if forward_count > 1 || reverse_count > 1 {
            consistent = false;
        }
    }

    (watertight, consistent)
}

/// Check if a mesh is watertight (all edges shared by exactly 2 faces).
pub fn is_watertight(faces: &[[usize; 3]]) -> bool {
    check_topology(faces).0
}

/// Check if face winding is consistent across the mesh.
///
/// For consistent winding, adjacent faces must traverse their shared edge
/// in opposite directions. If face A has edge (v0→v1), face B must have (v1→v0).
pub fn is_winding_consistent(faces: &[[usize; 3]]) -> bool {
    check_topology(faces).1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::create_box;

    #[test]
    fn test_face_adjacency() {
        let faces = vec![[0, 1, 2], [1, 3, 2]]; // Two triangles sharing edge 1-2
        let adj = face_adjacency(&faces);
        assert_eq!(adj.len(), 1);
        assert_eq!(adj[0], (0, 1));
    }

    #[test]
    fn test_is_watertight() {
        // Open mesh
        assert!(!is_watertight(&[[0, 1, 2], [1, 3, 2]]));

        // Closed cube
        let cube = create_box(&[1.0, 1.0, 1.0]);
        assert!(is_watertight(&cube.faces));
    }

    #[test]
    fn test_is_winding_consistent() {
        // Consistent winding: adjacent faces traverse shared edge in opposite directions
        // Face 0: 0→1→2 has edge 1→2
        // Face 1: 1→3→2 has edge 3→2 and 2→1 (opposite of 1→2) ✓
        let consistent = vec![[0, 1, 2], [1, 3, 2]];
        assert!(is_winding_consistent(&consistent));

        // Inconsistent winding: both faces traverse edge 1→2 in same direction
        // Face 0: 0→1→2 has edge 1→2
        // Face 1: 3→1→2 has edge 1→2 (same direction!) ✗
        let inconsistent = vec![[0, 1, 2], [3, 1, 2]];
        assert!(!is_winding_consistent(&inconsistent));

        // create_box should have consistent winding
        let cube = create_box(&[1.0, 1.0, 1.0]);
        assert!(is_winding_consistent(&cube.faces));
    }
}
