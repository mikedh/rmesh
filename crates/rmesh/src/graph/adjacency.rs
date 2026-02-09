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

/// Pre-computed edge grouping: boundary positions into the sorted edges array.
///
/// Built once by `edges_grouped()` and shared by all consumers that need
/// to iterate edges grouped by their `[min, max]` value.
/// The edges array returned by `edges_sorted()` is sorted by `.edge`,
/// so group `i` spans `edges[starts[i]..starts[i + 1]]` directly.
#[derive(Clone, Debug)]
pub struct EdgeGroups {
    /// Group boundary positions. Group `i` spans `starts[i]..starts[i + 1]`
    /// directly into the sorted edges array. Length = num_groups + 1.
    pub starts: Vec<usize>,
}

/// Mesh edge topology information.
///
/// - `is_watertight`: every edge is shared by exactly 2 faces
/// - `is_winding_consistent`: adjacent faces traverse shared edges in opposite directions.
///   Only meaningful when `is_watertight` is true; on non-manifold edges the value is
///   technically correct but semantically ambiguous.
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
            let group = &edges[w[0]..w[1]];
            let count = group.len();
            if count != 2 {
                is_watertight = false;
            }
            let forward = group.iter().filter(|e| e.forward).count();
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
/// Vector of `(face_a, face_b)` pairs that share an edge, normalized to `(min, max)`.
///
/// For non-manifold edges (3+ faces sharing an edge), emits a spanning star:
/// the first face connects to all others, yielding N-1 pairs.
/// Boundary edges (only one adjacent face) are silently ignored.
pub(crate) fn face_adjacency(edges: &[SortedEdge], groups: &EdgeGroups) -> Vec<(usize, usize)> {
    let mut result = Vec::with_capacity(groups.starts.len().saturating_sub(1));
    for w in groups.starts.windows(2) {
        let group = &edges[w[0]..w[1]];
        if group.len() >= 2 {
            // Emit spanning star: first face connects to all others.
            // For manifold meshes (group.len() == 2) this is one pair.
            // For non-manifold edges (3+), this preserves full connectivity.
            let first = group[0].face;
            for e in &group[1..] {
                result.push((first.min(e.face), first.max(e.face)));
            }
        }
    }
    result
}

/// Find connected components of nodes using Union-Find on edge pairs.
///
/// # Arguments
/// * `n_nodes` - Total number of nodes (indices 0..n_nodes).
/// * `edges` - Pairs of adjacent node indices (e.g. from `face_adjacency()`).
///
/// # Returns
/// Groups of node indices, sorted by smallest index in each group.
/// Each group is internally sorted. Empty input returns empty vec.
pub fn connected_components(n_nodes: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    if n_nodes == 0 {
        return vec![];
    }

    // Union-Find with path compression and union by rank
    let mut parent: Vec<usize> = (0..n_nodes).collect();
    let mut rank: Vec<u8> = vec![0; n_nodes];

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        x
    }

    for &(a, b) in edges {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            match rank[ra].cmp(&rank[rb]) {
                std::cmp::Ordering::Less => parent[ra] = rb,
                std::cmp::Ordering::Greater => parent[rb] = ra,
                std::cmp::Ordering::Equal => {
                    parent[rb] = ra;
                    rank[ra] += 1;
                }
            }
        }
    }

    // Group nodes by root
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut root_to_group: Vec<usize> = vec![usize::MAX; n_nodes];
    for node in 0..n_nodes {
        let root = find(&mut parent, node);
        if root_to_group[root] == usize::MAX {
            root_to_group[root] = groups.len();
            groups.push(Vec::new());
        }
        groups[root_to_group[root]].push(node);
    }

    // Components are already sorted internally (iterated 0..n_nodes).
    // Sort by smallest element for deterministic output.
    groups.sort_unstable_by_key(|g| g[0]);
    groups
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

    #[test]
    fn test_connected_components_single() {
        // Quad mesh: 2 faces sharing an edge → 1 component
        let mesh = quad_mesh();
        let components = super::connected_components(mesh.faces.len(), mesh.face_adjacency());
        assert_eq!(components.len(), 1);
        assert_eq!(components[0], vec![0, 1]);
    }

    #[test]
    fn test_connected_components_box() {
        // Box: all 12 faces connected → 1 component
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let components = super::connected_components(cube.faces.len(), cube.face_adjacency());
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 12);
    }

    #[test]
    fn test_connected_components_disjoint() {
        // Two separate triangles with no shared edge → 2 components
        let mesh = Trimesh::new(
            vec![
                nalgebra::Point3::new(0.0, 0.0, 0.0),
                nalgebra::Point3::new(1.0, 0.0, 0.0),
                nalgebra::Point3::new(0.0, 1.0, 0.0),
                nalgebra::Point3::new(5.0, 5.0, 0.0),
                nalgebra::Point3::new(6.0, 5.0, 0.0),
                nalgebra::Point3::new(5.0, 6.0, 0.0),
            ],
            vec![[0, 1, 2], [3, 4, 5]],
            None,
            None,
        )
        .unwrap();
        let components = super::connected_components(mesh.faces.len(), mesh.face_adjacency());
        assert_eq!(components.len(), 2);
        assert_eq!(components[0], vec![0]);
        assert_eq!(components[1], vec![1]);
    }

    #[test]
    fn test_connected_components_no_adjacency() {
        // 3 faces with no adjacency → each is its own component
        let components = super::connected_components(3, &[]);
        assert_eq!(components.len(), 3);
    }

    #[test]
    fn test_connected_components_empty() {
        let components = super::connected_components(0, &[]);
        assert!(components.is_empty());
    }

    #[test]
    fn test_face_adjacency_non_manifold() {
        // 3 faces sharing edge (0, 1) only — each face has a unique third vertex
        let mesh = Trimesh::new(
            vec![
                nalgebra::Point3::new(0.0, 0.0, 0.0),
                nalgebra::Point3::new(1.0, 0.0, 0.0),
                nalgebra::Point3::new(0.0, 1.0, 0.0),
                nalgebra::Point3::new(0.0, 0.0, 1.0),
                nalgebra::Point3::new(0.0, -1.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]],
            None,
            None,
        )
        .unwrap();

        let adj = mesh.face_adjacency();
        // Spanning star from first face to all others → 2 pairs from the shared edge
        assert_eq!(adj.len(), 2);
        // All pairs should be normalized (min, max)
        for &(a, b) in adj {
            assert!(a <= b, "pair ({a}, {b}) not normalized");
        }

        // All 3 faces should be in one connected component
        let components = super::connected_components(mesh.faces.len(), adj);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 3);
    }

    #[test]
    fn test_connected_components_deterministic() {
        // Verify connected_components returns identical results on repeated calls
        let edges = vec![(0, 1), (2, 3), (4, 5), (1, 2)];
        let result1 = super::connected_components(6, &edges);
        let result2 = super::connected_components(6, &edges);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_face_adjacency_deterministic() {
        // Verify face_adjacency returns identical results on repeated calls
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let adj1: Vec<(usize, usize)> = cube.face_adjacency().to_vec();
        // Create a fresh cube to avoid hitting the cache
        let cube2 = create_box(&[1.0, 1.0, 1.0]);
        let adj2: Vec<(usize, usize)> = cube2.face_adjacency().to_vec();
        assert_eq!(adj1, adj2);
    }
}
