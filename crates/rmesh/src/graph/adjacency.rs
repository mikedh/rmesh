//! Face and edge adjacency computations using sort-based algorithms.
//!
//! Uses sorting instead of hash maps for better cache locality on large meshes.

/// An edge with vertices sorted [min, max], direction info, and source face.
///
/// - `edge`: Vertex indices sorted as `[min, max]` for consistent comparison
/// - `forward`: `true` if original winding was min→max, `false` if max→min
/// - `face`: Index of the face this edge belongs to
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SortedEdge {
    pub edge: [usize; 2],
    pub forward: bool,
    pub face: usize,
}

/// Pre-computed edge grouping: an index order plus group boundary positions.
///
/// Built once by `edges_grouped()` and shared by all consumers that need
/// to iterate edges grouped by their `[min, max]` value.
#[derive(Clone, Debug)]
pub struct EdgeGroups {
    /// Index array that sorts edges by edge value.
    pub order: Vec<usize>,
    /// Group boundary positions into `order`. Group `i` spans
    /// `order[starts[i]..starts[i + 1]]`. Length = num_groups + 1.
    pub starts: Vec<usize>,
}

/// Mesh edge topology information.
///
/// - `is_watertight`: every edge is shared by exactly 2 faces
/// - `is_winding_consistent`: adjacent faces traverse shared edges in opposite directions
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManifoldStatus {
    pub is_watertight: bool,
    pub is_winding_consistent: bool,
}

impl ManifoldStatus {
    /// Compute topology from edges and pre-computed edge groups.
    pub(crate) fn from_grouped(edges: &[SortedEdge], groups: &EdgeGroups) -> Self {
        let mut is_watertight = true;
        let mut is_winding_consistent = true;

        for w in groups.starts.windows(2) {
            let group = &groups.order[w[0]..w[1]];
            let count = group.len();
            if count != 2 {
                is_watertight = false;
            }
            let forward = group.iter().filter(|&&j| edges[j].forward).count();
            let reverse = count - forward;
            if forward > 1 || reverse > 1 {
                is_winding_consistent = false;
            }
            if !is_watertight && !is_winding_consistent {
                break;
            }
        }

        Self {
            is_watertight,
            is_winding_consistent,
        }
    }
}

/// Find pairs of faces that share an edge using pre-computed edge groups.
///
/// # Arguments
/// * `edges` - Sorted edges with face indices.
/// * `groups` - Pre-computed edge grouping.
///
/// # Returns
/// Vector of `(face_a, face_b)` pairs that share an edge.
///
/// # Assumptions
/// - **Manifold meshes**: Assumes at most 2 faces share any edge. For non-manifold
///   meshes (3+ faces sharing an edge), only the first pair is returned.
/// - **Boundary edges**: Edges with only one adjacent face are silently ignored.
pub(crate) fn face_adjacency(edges: &[SortedEdge], groups: &EdgeGroups) -> Vec<(usize, usize)> {
    let mut result = Vec::with_capacity(groups.starts.len().saturating_sub(1));
    for w in groups.starts.windows(2) {
        let group = &groups.order[w[0]..w[1]];
        if group.len() >= 2 {
            result.push((edges[group[0]].face, edges[group[1]].face));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use crate::creation::create_box;
    use crate::mesh::Trimesh;

    /// A flat quad split into two triangles, reused by several tests.
    fn quad_mesh() -> Trimesh {
        Trimesh::new(
            vec![
                nalgebra::Point3::new(0.0, 0.0, 0.0),
                nalgebra::Point3::new(1.0, 0.0, 0.0),
                nalgebra::Point3::new(0.0, 1.0, 0.0),
                nalgebra::Point3::new(1.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [1, 3, 2]],
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_face_adjacency() {
        let mesh = quad_mesh();
        let adj = mesh.face_adjacency();
        assert_eq!(adj.len(), 1);
        assert_eq!(adj[0], (0, 1));
    }

    #[test]
    fn test_is_watertight() {
        // Open mesh
        assert!(!quad_mesh().is_watertight());

        // Closed cube
        let cube = create_box(&[1.0, 1.0, 1.0]);
        assert!(cube.is_watertight());
    }

    #[test]
    fn test_is_winding_consistent() {
        // Consistent winding: adjacent faces traverse shared edge in opposite directions
        assert!(quad_mesh().is_winding_consistent());

        // Inconsistent winding: both faces traverse edge 1→2 in same direction
        let inconsistent = Trimesh::new(
            vec![
                nalgebra::Point3::new(0.0, 0.0, 0.0),
                nalgebra::Point3::new(1.0, 0.0, 0.0),
                nalgebra::Point3::new(0.0, 1.0, 0.0),
                nalgebra::Point3::new(1.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [3, 1, 2]],
            None,
            None,
        )
        .unwrap();
        assert!(!inconsistent.is_winding_consistent());

        // create_box should have consistent winding
        let cube = create_box(&[1.0, 1.0, 1.0]);
        assert!(cube.is_winding_consistent());
    }

    #[test]
    fn test_manifold_status() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let status = cube.manifold_status();
        assert!(status.is_watertight);
        assert!(status.is_winding_consistent);
    }
}
