use std::borrow::Cow;
use std::collections::BTreeSet;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{
    attributes::{Attributes, GroupingKind, LoadSource, Material, UNSET},
    boundary::Surface,
    cache::Cache,
    creation::Plane,
    graph::{EdgeGroups, ManifoldStatus, SortedEdge, adjacency},
    path::Path2D,
    project::{CircleWithEdges, EPSILON_RELATIVE},
    simplify::{SimplifyOptions, SimplifyResult, simplify_mesh},
    triangles::{
        bvh::TriangleBvh,
        inertia::{self, MassProperties},
    },
};
use kiddo::ImmutableKdTree;
use nalgebra::{Matrix3, Matrix4, Point2, Point3, Vector3};
use rayon::prelude::*;

/// A triangle mesh with vertices and face indices.
///
/// # Caching
///
/// Derived properties like `face_normals()`, `edges()`, etc. are lazily computed
/// and cached on first access. The cache is thread-safe and uses `OnceLock` for
/// lock-free reads after initialization.
///
/// ## Adding New Cached Properties
///
/// 1. Add a `cache_foo: OnceLock<T>` field to the struct
/// 2. Initialize it in `Default` and `Clone` impls
/// 3. Create accessor: `fn foo(&self) -> &T { self.cache_foo.get_or_init(|| ...) }`
/// 4. **Important**: Expensive compute functions (e.g., `ManifoldStatus::new`)
///    should ONLY be called inside `get_or_init` closures, never directly.
///
/// # Mutation Warning
///
/// If you mutate `vertices` or `faces` after calling any cached method,
/// the cached values will be stale. Create a new `Trimesh` instead of mutating.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Trimesh {
    /// Vertex positions
    pub vertices: Vec<Point3<f64>>,
    /// Triangle face indices into the vertices array
    pub faces: Vec<[usize; 3]>,

    /// Vertex attributes (UV coordinates, colors, normals, etc.)
    pub attributes_vertex: Attributes,
    /// Face attributes
    pub attributes_face: Attributes,

    /// Information about where the mesh came from
    pub source: LoadSource,

    /// Materials loaded from the mesh file (e.g., from OBJ's MTL reference)
    pub materials: Vec<Material>,

    /// Analytical surface definitions from BREP data (e.g. via cascadio).
    /// Per-face index mapping lives in `attributes_face.groupings` with
    /// `GroupingKind::Surface`.
    pub face_surfaces: Vec<Surface>,

    /// Optional density for mass property calculations (default 1.0)
    pub density: Option<f64>,

    // Cached derived values - computed lazily on first access
    #[serde(skip)]
    cache_faces_cross: Cache<Vec<Vector3<f64>>>,
    #[serde(skip)]
    cache_face_normals: Cache<Vec<Vector3<f64>>>,
    #[serde(skip)]
    cache_faces_area: Cache<Vec<f64>>,
    #[serde(skip)]
    cache_area: Cache<f64>,
    #[serde(skip)]
    cache_edges: Cache<Vec<[usize; 2]>>,
    #[serde(skip)]
    cache_face_adjacency: Cache<Vec<(usize, usize)>>,
    #[serde(skip)]
    cache_edges_unique: Cache<Vec<[usize; 2]>>,
    #[serde(skip)]
    cache_edges_sorted: Cache<Vec<SortedEdge>>,
    #[serde(skip)]
    cache_edges_grouped: Cache<EdgeGroups>,
    #[serde(skip)]
    cache_edges_unique_inverse: Cache<Vec<usize>>,
    #[serde(skip)]
    cache_mass_properties: Cache<MassProperties>,
    #[serde(skip)]
    cache_manifold_status: Cache<ManifoldStatus>,
    #[serde(skip)]
    cache_vertex_mask: Cache<Vec<bool>>,
    #[serde(skip)]
    cache_bvh: Cache<TriangleBvh>,
    #[serde(skip)]
    cache_convex_hull: Cache<Box<Trimesh>>,
    #[serde(skip)]
    cache_obb: Cache<crate::convex::OrientedBoundingBox>,
}

impl Clone for Trimesh {
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            faces: self.faces.clone(),
            attributes_vertex: self.attributes_vertex.clone(),
            attributes_face: self.attributes_face.clone(),
            source: self.source.clone(),
            materials: self.materials.clone(),
            face_surfaces: self.face_surfaces.clone(),
            density: self.density,

            // Fresh cache - will recompute on demand
            cache_faces_cross: Cache::new(),
            cache_face_normals: Cache::new(),
            cache_faces_area: Cache::new(),
            cache_area: Cache::new(),
            cache_edges: Cache::new(),
            cache_face_adjacency: Cache::new(),
            cache_edges_unique: Cache::new(),
            cache_edges_sorted: Cache::new(),
            cache_edges_grouped: Cache::new(),
            cache_edges_unique_inverse: Cache::new(),
            cache_mass_properties: Cache::new(),
            cache_manifold_status: Cache::new(),
            cache_vertex_mask: Cache::new(),
            cache_bvh: Cache::new(),
            cache_convex_hull: Cache::new(),
            cache_obb: Cache::new(),
        }
    }
}

impl PartialEq for Trimesh {
    fn eq(&self, other: &Self) -> bool {
        self.vertices == other.vertices
            && self.faces == other.faces
            && self.attributes_vertex == other.attributes_vertex
            && self.attributes_face == other.attributes_face
            && self.source == other.source
            && self.materials == other.materials
            && self.face_surfaces == other.face_surfaces
            && self.density == other.density
    }
}

impl Trimesh {
    /// Create a new trimesh from vertices and faces.
    pub fn new(
        vertices: Vec<Point3<f64>>,
        faces: Vec<[usize; 3]>,
        attributes_vertex: Option<Attributes>,
        attributes_face: Option<Attributes>,
    ) -> Result<Self> {
        Ok(Self {
            vertices,
            faces,
            attributes_vertex: attributes_vertex.unwrap_or_default(),
            attributes_face: attributes_face.unwrap_or_default(),
            ..Default::default()
        })
    }

    /// Create a Trimesh from flat slices of vertices and faces.
    pub fn from_slice(vertices: &[f64], faces: &[usize]) -> Result<Self> {
        let vertices: Vec<Point3<f64>> = vertices
            .chunks_exact(3)
            .map(|chunk| Point3::new(chunk[0], chunk[1], chunk[2]))
            .collect();

        let faces: Vec<[usize; 3]> = faces
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();

        Ok(Self {
            vertices,
            faces,
            ..Default::default()
        })
    }

    /// Concatenate multiple meshes into a single mesh.
    ///
    /// Vertices are concatenated in order and face indices are offset accordingly.
    /// Vertex and face attributes are concatenated channel-wise (see `Attributes::concatenate`).
    /// Materials and face surfaces are concatenated with grouping indices remapped.
    ///
    /// Returns an empty mesh if the input slice is empty.
    pub fn concatenate(meshes: &[&Trimesh]) -> Self {
        if meshes.is_empty() {
            return Self::default();
        }
        if meshes.len() == 1 {
            return meshes[0].clone();
        }

        let total_verts: usize = meshes.iter().map(|m| m.vertices.len()).sum();
        let total_faces: usize = meshes.iter().map(|m| m.faces.len()).sum();

        let mut vertices = Vec::with_capacity(total_verts);
        let mut faces = Vec::with_capacity(total_faces);

        for m in meshes {
            let offset = vertices.len();
            vertices.extend_from_slice(&m.vertices);
            faces.extend(
                m.faces
                    .iter()
                    .map(|f| [f[0] + offset, f[1] + offset, f[2] + offset]),
            );
        }

        let attr_vertex_slices: Vec<(&Attributes, usize)> = meshes
            .iter()
            .map(|m| (&m.attributes_vertex, m.vertices.len()))
            .collect();
        let attr_face_slices: Vec<(&Attributes, usize)> = meshes
            .iter()
            .map(|m| (&m.attributes_face, m.faces.len()))
            .collect();

        let mut materials = Vec::new();
        let mut face_surfaces = Vec::new();
        for m in meshes {
            materials.extend_from_slice(&m.materials);
            face_surfaces.extend_from_slice(&m.face_surfaces);
        }

        Self {
            vertices,
            faces,
            attributes_vertex: Attributes::concatenate(&attr_vertex_slices),
            attributes_face: Attributes::concatenate(&attr_face_slices),
            materials,
            face_surfaces,
            ..Default::default()
        }
    }

    /// Simplify the mesh to a target face count.
    ///
    /// Preserves vertex attributes (normals, UVs, colors) through interpolation.
    /// Returns a `SimplifyResult` containing the simplified mesh and optional quality metrics.
    ///
    /// # Example
    ///
    /// ```
    /// use rmesh::simplify::SimplifyOptions;
    /// use rmesh::mesh::Trimesh;
    ///
    /// let mesh = Trimesh::default();
    /// let result = mesh.simplify(&SimplifyOptions {
    ///     target_count: Some(1000),
    ///     aggressiveness: 7.0,
    ///     ..Default::default()
    /// });
    /// // Get the simplified mesh
    /// let simplified: Trimesh = result.into();
    /// ```
    #[must_use]
    pub fn simplify(&self, options: &SimplifyOptions) -> SimplifyResult {
        simplify_mesh(
            &self.vertices,
            &self.faces,
            Some(&self.attributes_vertex),
            Some(&self.attributes_face),
            *options,
        )
    }

    /// Subdivide the mesh by splitting each triangle into 4 triangles.
    ///
    /// Each edge is split at its midpoint. After N iterations,
    /// the face count is multiplied by 4^N.
    /// Face attributes (colors, UVs, normals, tangents, groupings)
    /// are propagated to child faces.
    #[must_use]
    pub fn subdivide(&self, iterations: usize) -> Self {
        let (vertices, faces, attributes_face) = crate::subdivide::subdivide(
            &self.vertices,
            &self.faces,
            Some(&self.attributes_face),
            iterations,
        );
        Self {
            vertices,
            faces,
            attributes_face,
            face_surfaces: self.face_surfaces.clone(),
            materials: self.materials.clone(),
            ..Default::default()
        }
    }

    /// Clean up the mesh by merging vertices, removing degenerate faces, etc.
    ///
    /// Unlike trimesh which uses global tolerances, all options are passed explicitly.
    /// Returns a `CleanupResult` containing the new mesh data and statistics about
    /// what was changed (vertices merged, faces removed, etc.).
    ///
    /// # Example
    ///
    /// ```
    /// use rmesh::cleanup::CleanupOptions;
    /// use rmesh::mesh::Trimesh;
    ///
    /// let mesh = Trimesh::default();
    /// // Merge duplicate vertices and remove degenerate faces
    /// let result = mesh.cleanup(&CleanupOptions {
    ///     merge_vertices: Some(8),
    ///     remove_degenerate: Some(12),
    ///     ..Default::default()
    /// });
    /// // Get the cleaned mesh
    /// let cleaned: Trimesh = result.into();
    /// ```
    #[must_use]
    pub fn cleanup(
        &self,
        options: &crate::cleanup::CleanupOptions,
    ) -> crate::cleanup::CleanupResult {
        crate::cleanup::cleanup(
            &self.vertices,
            &self.faces,
            Some(&self.attributes_vertex),
            Some(&self.attributes_face),
            options,
        )
    }

    /// The non-normalized cross product of every face.
    ///
    /// Cached on first access.
    pub fn faces_cross(&self) -> &[Vector3<f64>] {
        self.cache_faces_cross.get_or_init(|| {
            self.faces
                .par_iter()
                .map(|[i0, i1, i2]| {
                    let v0 = self.vertices[*i0];
                    let v1 = self.vertices[*i1];
                    let v2 = self.vertices[*i2];
                    (v1 - v0).cross(&(v2 - v0))
                })
                .collect()
        })
    }

    /// Calculate the normals for each face of the mesh.
    ///
    /// Cached on first access.
    pub fn face_normals(&self) -> &[Vector3<f64>] {
        self.cache_face_normals.get_or_init(|| {
            self.faces_cross()
                .par_iter()
                .map(|cross| {
                    let n = cross.norm();
                    if n > 1e-30 {
                        cross / n
                    } else {
                        Vector3::zeros()
                    }
                })
                .collect()
        })
    }

    /// Get the edges calculated from the faces.
    ///
    /// Returns 3 edges per face (may contain duplicates).
    /// Cached on first access.
    pub fn edges(&self) -> &[[usize; 2]] {
        self.cache_edges.get_or_init(|| {
            self.faces
                .par_iter()
                .flat_map(|[i0, i1, i2]| [[*i0, *i1], [*i1, *i2], [*i2, *i0]])
                .collect()
        })
    }

    /// The area for each triangle in the mesh.
    ///
    /// Cached on first access.
    pub fn faces_area(&self) -> &[f64] {
        self.cache_faces_area.get_or_init(|| {
            self.faces_cross()
                .par_iter()
                .map(|cross| cross.norm() / 2.0)
                .collect()
        })
    }

    /// The summed area of every triangle in the mesh.
    ///
    /// Cached on first access.
    pub fn area(&self) -> f64 {
        *self
            .cache_area
            .get_or_init(|| self.faces_area().iter().sum())
    }

    /// What are the pairs of face indices that share an edge?
    ///
    /// Cached on first access. Uses sort-based algorithm for cache efficiency.
    /// Pairs are normalized to `(min, max)` for deterministic ordering.
    ///
    /// For non-manifold edges (3+ faces sharing an edge), emits a spanning star
    /// from the first face to all others. Boundary edges are silently ignored.
    pub fn face_adjacency(&self) -> &[(usize, usize)] {
        self.cache_face_adjacency
            .get_or_init(|| adjacency::face_adjacency(self.edges_sorted(), self.edges_grouped()))
    }

    /// Calculate the angles between adjacent faces.
    pub fn face_adjacency_angles(&self) -> Vec<f64> {
        let adjacency = self.face_adjacency();
        let normals = self.face_normals();
        adjacency
            .par_iter()
            .map(|adj| normals[adj.0].angle(&normals[adj.1]))
            .collect()
    }

    /// Compute smooth vertex normals (one per face-corner).
    ///
    /// Returns `Vec<Vector3<f64>>` of length `faces.len() * 3`.
    /// Index `[fi * 3 + ci]` is the normal for face `fi`, vertex `ci`.
    ///
    /// If `groups` is `Some`, uses the provided per-face group IDs directly
    /// (e.g. from OBJ `s N` smooth groups). If `None`, computes groups from
    /// face adjacency: faces sharing an edge whose face normals differ by
    /// less than `angle_threshold` radians are grouped together.
    ///
    /// Within a group, vertex normals are angle-weighted averages of
    /// adjacent face normals. Pass `PI` for fully smooth.
    pub fn smooth_vertex_normals(
        &self,
        angle_threshold: f64,
        groups: Option<&[usize]>,
    ) -> Vec<Vector3<f64>> {
        let n_faces = self.faces.len();
        let face_normals = self.face_normals();

        // Resolve face→group mapping: either from caller or from angle threshold
        let computed_groups;
        let face_group: &[usize] = if let Some(g) = groups {
            g
        } else {
            let adj = self.face_adjacency();
            let angles = self.face_adjacency_angles();

            let smooth_edges: Vec<(usize, usize)> = adj
                .iter()
                .zip(angles.iter())
                .filter(|&(_, angle)| *angle <= angle_threshold)
                .map(|(&pair, _)| pair)
                .collect();

            let components = adjacency::connected_components(n_faces, &smooth_edges);
            computed_groups = {
                let mut fg = vec![0usize; n_faces];
                for (gi, group) in components.iter().enumerate() {
                    for &fi in group {
                        fg[fi] = gi;
                    }
                }
                fg
            };
            &computed_groups
        };

        // Precompute corner angles: corner_angles[fi * 3 + ci] = interior angle
        // at vertex ci of face fi.
        let corner_angles: Vec<f64> = self
            .faces
            .iter()
            .flat_map(|[i0, i1, i2]| {
                let v0 = self.vertices[*i0].coords;
                let v1 = self.vertices[*i1].coords;
                let v2 = self.vertices[*i2].coords;
                let angle_at = |a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>| {
                    let e1 = b - a;
                    let e2 = c - a;
                    let n1 = e1.norm();
                    let n2 = e2.norm();
                    if n1 < 1e-30 || n2 < 1e-30 {
                        return 0.0;
                    }
                    (e1.dot(&e2) / (n1 * n2)).clamp(-1.0, 1.0).acos()
                };
                [
                    angle_at(v0, v1, v2),
                    angle_at(v1, v2, v0),
                    angle_at(v2, v0, v1),
                ]
            })
            .collect();

        // Build vertex→faces map
        let n_verts = self.vertices.len();
        let mut vert_faces: Vec<Vec<usize>> = vec![Vec::new(); n_verts];
        for (fi, face) in self.faces.iter().enumerate() {
            for &vi in face {
                vert_faces[vi].push(fi);
            }
        }

        // For each face corner, compute angle-weighted average of face normals
        // from faces sharing the same vertex AND same smooth group.
        let mut corner_normals = vec![Vector3::zeros(); n_faces * 3];
        for (fi, face) in self.faces.iter().enumerate() {
            let group = face_group[fi];
            for (ci, &vi) in face.iter().enumerate() {
                let mut normal = Vector3::zeros();
                for &neighbor_fi in &vert_faces[vi] {
                    if face_group[neighbor_fi] == group {
                        let neighbor_ci = self.faces[neighbor_fi]
                            .iter()
                            .position(|&v| v == vi)
                            .unwrap();
                        let weight = corner_angles[neighbor_fi * 3 + neighbor_ci];
                        normal += face_normals[neighbor_fi] * weight;
                    }
                }
                let norm = normal.norm();
                corner_normals[fi * 3 + ci] = if norm > 1e-30 {
                    normal / norm
                } else {
                    Vector3::z_axis().into_inner()
                };
            }
        }

        corner_normals
    }

    /// Compute mass properties (volume, mass, center of mass, inertia tensor).
    ///
    /// Uses the polyhedral mass properties algorithm from:
    /// http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
    ///
    /// Cached on first access.
    pub fn mass_properties(&self) -> &MassProperties {
        self.cache_mass_properties.get_or_init(|| {
            inertia::mass_properties(
                &self.vertices,
                &self.faces,
                self.density.unwrap_or(1.0),
                false,
            )
        })
    }

    /// Get the volume of the mesh (signed, based on face winding).
    ///
    /// For a closed mesh with outward-pointing normals, this is positive.
    pub fn volume(&self) -> f64 {
        self.mass_properties().volume
    }

    /// Get the mass of the mesh (volume * density).
    pub fn mass(&self) -> f64 {
        self.mass_properties().mass
    }

    /// Get the center of mass of the mesh.
    pub fn center_mass(&self) -> Vector3<f64> {
        self.mass_properties().center_mass
    }

    /// Get the 3x3 inertia tensor of the mesh.
    ///
    /// Returns None if mass properties calculation was skipped.
    pub fn moment_inertia(&self) -> Option<&Matrix3<f64>> {
        self.mass_properties().inertia.as_ref()
    }

    /// Get principal inertia components (eigenvalues of inertia tensor, sorted descending).
    ///
    /// Returns None if inertia tensor was not computed.
    pub fn principal_inertia_components(&self) -> Option<Vector3<f64>> {
        self.mass_properties().principal_inertia().map(|(v, _)| v)
    }

    /// Get principal inertia vectors (eigenvectors of inertia tensor as matrix columns).
    ///
    /// The columns correspond to the principal axes, ordered by eigenvalue (descending).
    /// Returns None if inertia tensor was not computed.
    pub fn principal_inertia_vectors(&self) -> Option<Matrix3<f64>> {
        self.mass_properties().principal_inertia().map(|(_, m)| m)
    }

    /// Get cached manifold status (computed once from edges_sorted).
    pub fn manifold_status(&self) -> &ManifoldStatus {
        self.cache_manifold_status
            .get_or_init(|| ManifoldStatus::from_grouped(self.edges_sorted(), self.edges_grouped()))
    }

    /// Check if the mesh is watertight (all edges shared by exactly 2 faces).
    pub fn is_watertight(&self) -> bool {
        self.manifold_status().is_watertight
    }

    /// Return BREP face indices that border non-manifold mesh edges.
    ///
    /// Finds mesh edges with count != 2 (boundary or non-manifold), maps them
    /// to incident triangles, then maps those triangles to BREP face indices
    /// via the `GroupingKind::Surface` grouping. Returns an empty set if
    /// no surface grouping exists or the mesh is watertight.
    pub fn non_watertight_face_indices(&self) -> BTreeSet<usize> {
        let edges = self.edges_sorted();
        let groups = self.edges_grouped();

        // Find triangles incident to non-manifold edges
        let mut bad_tris = BTreeSet::new();
        for w in groups.starts.windows(2) {
            let group = &edges[w[0]..w[1]];
            if group.len() != 2 {
                for e in group {
                    bad_tris.insert(e.face);
                }
            }
        }

        if bad_tris.is_empty() {
            return BTreeSet::new();
        }

        // Map triangle indices → BREP face indices via Surface grouping
        let surface_grouping = self
            .attributes_face
            .groupings
            .iter()
            .find(|g| g.kind == GroupingKind::Surface);

        match surface_grouping {
            Some(g) => bad_tris
                .into_iter()
                .filter_map(|tri_idx| g.indices.get(tri_idx).copied())
                .filter(|&idx| idx != crate::attributes::UNSET)
                .collect(),
            None => BTreeSet::new(),
        }
    }

    /// Check if face winding is consistent across the mesh.
    ///
    /// For consistent winding, adjacent faces must traverse their shared edge
    /// in opposite directions. If face A has edge (v0→v1), face B must have (v1→v0).
    pub fn is_winding_consistent(&self) -> bool {
        self.manifold_status().is_winding_consistent
    }

    /// Check if the mesh represents a valid volume.
    ///
    /// A mesh is a volume if it is watertight, has consistent winding,
    /// and has positive volume (outward-facing normals).
    pub fn is_volume(&self) -> bool {
        let status = self.manifold_status();
        status.is_watertight && status.is_winding_consistent && self.volume() > 0.0
    }

    /// Get unique edges (each undirected edge once, sorted as [min, max]).
    ///
    /// Uses sort-based deduplication for cache efficiency. The order is deterministic
    /// (lexicographically sorted by edge). Cached on first access.
    pub fn edges_unique(&self) -> &[[usize; 2]] {
        self.cache_edges_unique.get_or_init(|| {
            let edges = self.edges_sorted();
            let groups = self.edges_grouped();
            let mut unique = Vec::with_capacity(groups.starts.len().saturating_sub(1));
            for w in groups.starts.windows(2) {
                unique.push(edges[w[0]].edge);
            }
            unique
        })
    }

    /// Get all edges with direction info and source face, 3 per face.
    ///
    /// Each edge stores `[min_vertex, max_vertex]`, winding direction, and face index.
    /// Returns 3 edges per face, unlike `edges_unique` which deduplicates.
    /// Cached on first access.
    pub fn edges_sorted(&self) -> &[SortedEdge] {
        self.cache_edges_sorted.get_or_init(|| {
            let mut edges = Vec::with_capacity(self.faces.len() * 3);
            for (face, &[i0, i1, i2]) in self.faces.iter().enumerate() {
                edges.push(SortedEdge {
                    edge: [i0.min(i1), i0.max(i1)],
                    forward: i0 < i1,
                    face,
                });
                edges.push(SortedEdge {
                    edge: [i1.min(i2), i1.max(i2)],
                    forward: i1 < i2,
                    face,
                });
                edges.push(SortedEdge {
                    edge: [i2.min(i0), i2.max(i0)],
                    forward: i2 < i0,
                    face,
                });
            }
            edges.par_sort_unstable_by_key(|e| e.edge);
            edges
        })
    }

    /// Get the index order and group boundaries that group `edges_sorted()` by edge value.
    ///
    /// Sorted once and shared by `face_adjacency`, `manifold_status`,
    /// `edges_unique`, and `edges_unique_inverse`.
    /// Cached on first access.
    fn edges_grouped(&self) -> &EdgeGroups {
        self.cache_edges_grouped.get_or_init(|| {
            let edges = self.edges_sorted();
            let mut starts = vec![0];
            for i in 1..edges.len() {
                if edges[i].edge != edges[i - 1].edge {
                    starts.push(i);
                }
            }
            starts.push(edges.len());
            EdgeGroups { starts }
        })
    }

    /// Get the geometric length of each unique edge.
    ///
    /// Not cached - cheap to recompute and would be stale if vertices change.
    pub fn edges_unique_length(&self) -> Vec<f64> {
        self.edges_unique()
            .par_iter()
            .map(|&[i, j]| (self.vertices[j] - self.vertices[i]).norm())
            .collect()
    }

    /// Get the inverse mapping from edges_sorted to edges_unique indices.
    ///
    /// For each edge in edges_sorted, returns the index of that edge in edges_unique.
    /// Cached on first access.
    pub fn edges_unique_inverse(&self) -> &[usize] {
        self.cache_edges_unique_inverse.get_or_init(|| {
            let edges = self.edges_sorted();
            let groups = self.edges_grouped();
            let mut inverse = vec![0usize; edges.len()];
            for (unique_idx, w) in groups.starts.windows(2).enumerate() {
                for slot in &mut inverse[w[0]..w[1]] {
                    *slot = unique_idx;
                }
            }
            inverse
        })
    }

    /// Get directed boundary edges (edges shared by exactly one face).
    ///
    /// Each returned edge preserves the winding direction from the original face.
    /// Non-manifold edges (shared by 3+ faces) are excluded.
    /// Not cached - derived from `edges_sorted()` and `edges_grouped()`.
    pub fn edges_boundary(&self) -> Vec<[usize; 2]> {
        let edges = self.edges_sorted();
        let groups = self.edges_grouped();
        let mut boundary = Vec::new();
        for w in groups.starts.windows(2) {
            if w[1] - w[0] == 1 {
                let e = &edges[w[0]];
                if e.forward {
                    boundary.push(e.edge);
                } else {
                    boundary.push([e.edge[1], e.edge[0]]);
                }
            }
        }
        boundary
    }

    /// Fill simple holes in the mesh by fan-triangulating boundary loops.
    ///
    /// Returns a new mesh with additional faces closing each hole. The new
    /// faces reverse the boundary edge winding so normals point outward.
    /// Fan triangulation works best on roughly convex holes; non-convex
    /// boundary loops may produce self-intersecting fill faces.
    ///
    /// Vertex attributes are preserved; face attributes are reset since
    /// fill faces have no source attributes. Returns the mesh unchanged
    /// if the boundary contains non-manifold (bowtie) vertices.
    ///
    /// Check `result.is_watertight()` to determine if all holes were filled.
    #[must_use]
    pub fn fill_holes(&self) -> Self {
        let boundary = self.edges_boundary();
        if boundary.is_empty() {
            return self.clone();
        }

        // Build adjacency: vertex -> next vertex in directed boundary
        let mut next_map = std::collections::HashMap::with_capacity(boundary.len());
        for &[a, b] in &boundary {
            if next_map.insert(a, b).is_some() {
                // Non-manifold boundary: a vertex has multiple outgoing
                // boundary edges (bowtie vertex). We can't reliably chain
                // loops, so bail out and return the mesh unchanged.
                return self.clone();
            }
        }

        // Walk chains to find closed loops
        let mut visited = std::collections::HashSet::with_capacity(boundary.len());
        let mut loops: Vec<Vec<usize>> = Vec::new();

        for &[start, _] in &boundary {
            if visited.contains(&start) {
                continue;
            }
            let mut chain = vec![start];
            visited.insert(start);
            let mut current = start;

            loop {
                let Some(&next) = next_map.get(&current) else {
                    break; // open chain, not a loop
                };
                if next == start {
                    // closed loop
                    loops.push(chain);
                    break;
                }
                if visited.contains(&next) {
                    break; // already visited, skip
                }
                visited.insert(next);
                chain.push(next);
                current = next;
            }
        }

        if loops.is_empty() {
            return self.clone();
        }

        // Fan-triangulate each loop with reversed winding
        let mut new_faces = self.faces.clone();
        for loop_verts in &loops {
            if loop_verts.len() < 3 {
                continue;
            }
            let v0 = loop_verts[0];
            for i in 1..loop_verts.len() - 1 {
                // Reverse winding: [v0, v(i+1), vi]
                new_faces.push([v0, loop_verts[i + 1], loop_verts[i]]);
            }
        }

        Self {
            vertices: self.vertices.clone(),
            faces: new_faces,
            attributes_vertex: self.attributes_vertex.clone(),
            materials: self.materials.clone(),
            face_surfaces: self.face_surfaces.clone(),
            source: self.source.clone(),
            density: self.density,
            ..Default::default()
        }
    }

    /// Calculate the Euler characteristic (V - E + F).
    ///
    /// For a closed manifold surface: χ = 2 - 2g where g is the genus.
    /// - Sphere: χ = 2
    /// - Torus: χ = 0
    /// - Double torus: χ = -2
    pub fn euler_number(&self) -> i64 {
        #[allow(clippy::cast_possible_wrap)]
        let v = self.vertex_mask().iter().filter(|&&m| m).count() as i64;
        #[allow(clippy::cast_possible_wrap)]
        let f = self.faces.len() as i64;
        #[allow(clippy::cast_possible_wrap)]
        let e = self.edges_unique().len() as i64;
        v - e + f
    }

    /// Get the extents of the bounding box (max - min for each axis).
    ///
    /// Returns None if the mesh is empty.
    pub fn extents(&self) -> Option<[f64; 3]> {
        self.bounds()
            .map(|(min, max)| [max.x - min.x, max.y - min.y, max.z - min.z])
    }

    /// Get the geometric center of the vertices (mean position).
    ///
    /// This is different from center_mass which is weighted by volume.
    pub fn centroid(&self) -> Option<Point3<f64>> {
        if self.vertices.is_empty() {
            return None;
        }
        let sum: Vector3<f64> = self
            .vertices
            .par_iter()
            .map(|v| v.coords)
            .reduce(Vector3::zeros, |a, b| a + b);
        Some(Point3::from(sum / self.vertices.len() as f64))
    }

    /// Get the actual vertex positions for each face as (n_faces, 3, 3) data.
    ///
    /// Returns a flat Vec where every 9 elements represent one triangle:
    /// [v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, ...]
    pub fn triangles_flat(&self) -> Vec<f64> {
        self.faces
            .par_iter()
            .flat_map(|[i0, i1, i2]| {
                let v0 = &self.vertices[*i0];
                let v1 = &self.vertices[*i1];
                let v2 = &self.vertices[*i2];
                [v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z]
            })
            .collect()
    }

    /// Get the center of each triangle.
    pub fn triangles_center(&self) -> Vec<Point3<f64>> {
        self.faces
            .par_iter()
            .map(|[i0, i1, i2]| {
                let v0 = &self.vertices[*i0];
                let v1 = &self.vertices[*i1];
                let v2 = &self.vertices[*i2];
                Point3::from((v0.coords + v1.coords + v2.coords) / 3.0)
            })
            .collect()
    }

    /// For each adjacent face pair, get the indices of the two vertices not on the shared edge.
    ///
    /// Returns (n_adjacency, 2) array where each row is [unshared_from_face_a, unshared_from_face_b].
    pub fn face_adjacency_unshared(&self) -> Vec<[usize; 2]> {
        let adjacency = self.face_adjacency();
        adjacency
            .par_iter()
            .map(|(face_a, face_b)| {
                let fa = &self.faces[*face_a];
                let fb = &self.faces[*face_b];
                // Find vertex in face_a not in face_b
                let unshared_a = fa
                    .iter()
                    .find(|v| !fb.contains(v))
                    .copied()
                    .unwrap_or(fa[0]);
                // Find vertex in face_b not in face_a
                let unshared_b = fb
                    .iter()
                    .find(|v| !fa.contains(v))
                    .copied()
                    .unwrap_or(fb[0]);
                [unshared_a, unshared_b]
            })
            .collect()
    }

    /// Project the unshared vertex of each adjacent face onto the plane of the other face.
    ///
    /// For each adjacent pair (A, B), computes the signed distance from B's unshared vertex
    /// to the plane of face A. Negative means locally convex, positive means locally concave.
    pub fn face_adjacency_projections(&self) -> Vec<f64> {
        let adjacency = self.face_adjacency();
        let normals = self.face_normals();
        let unshared = self.face_adjacency_unshared();
        let triangles_center = self.triangles_center();

        adjacency
            .par_iter()
            .zip(unshared.par_iter())
            .map(|((face_a, _face_b), [_unshared_a, unshared_b])| {
                // Project unshared vertex from face B onto plane of face A
                let normal_a = &normals[*face_a];
                let center_a = &triangles_center[*face_a];
                let vertex_b = &self.vertices[*unshared_b];

                // Signed distance from vertex_b to plane of face_a
                // plane equation: normal . (p - center) = 0
                // signed distance = normal . (vertex_b - center_a)
                normal_a.dot(&(vertex_b - center_a))
            })
            .collect()
    }

    /// Boolean array indicating whether each adjacent face pair is locally convex.
    ///
    /// A pair is locally convex if the projection of B's unshared vertex onto A's plane
    /// is below the plane (negative projection value within tolerance).
    pub fn face_adjacency_convex(&self) -> Vec<bool> {
        let projections = self.face_adjacency_projections();
        // Use a small tolerance for numerical stability
        let tolerance = 1e-8;
        projections.par_iter().map(|p| *p < tolerance).collect()
    }

    /// Check if the mesh is convex.
    ///
    /// A mesh is convex if all adjacent face pairs are locally convex.
    pub fn is_convex(&self) -> bool {
        if self.faces.is_empty() {
            return false;
        }
        self.face_adjacency_convex().par_iter().all(|&c| c)
    }

    /// Get the triangle BVH for fast ray and closest-point queries.
    ///
    /// Cached on first access.
    pub fn bvh(&self) -> &TriangleBvh {
        self.cache_bvh
            .get_or_init(|| TriangleBvh::build(&self.vertices, &self.faces))
    }

    /// Perform convex decomposition of this mesh.
    ///
    /// Returns a set of approximate convex hulls that together cover the mesh.
    /// Requires the `wgpu` feature and a GPU.
    #[cfg(feature = "wgpu")]
    pub fn convex_decomposition(
        &self,
        params: &crate::decomposition::DecompositionParams,
    ) -> crate::decomposition::DecompositionResult {
        let (device, queue) =
            crate::voxel::request_device().expect("GPU required for convex decomposition");
        crate::decomposition::convex_decomposition(
            &device,
            &queue,
            &self.vertices,
            &self.faces,
            params,
        )
    }

    /// Boolean mask of vertices referenced by at least one face.
    /// `vertex_mask()[i]` is true if vertex `i` appears in any face.
    pub fn vertex_mask(&self) -> &[bool] {
        self.cache_vertex_mask.get_or_init(|| {
            let mut mask = vec![false; self.vertices.len()];
            for f in &self.faces {
                mask[f[0]] = true;
                mask[f[1]] = true;
                mask[f[2]] = true;
            }
            mask
        })
    }

    /// Compute the convex hull of the mesh vertices.
    ///
    /// # Parameters
    /// - `compact`: When `true`, only hull-surface vertices are kept (cached).
    ///   When `false`, all *referenced* source vertices are preserved.
    ///   Both modes filter unreferenced vertices.
    ///
    /// The resulting mesh will be watertight and convex with outward-facing
    /// CCW normals and consistent winding.
    ///
    /// Returns `Cow::Borrowed` for compact=true (references cached hull),
    /// `Cow::Owned` for compact=false (fresh computation).
    pub fn convex_hull(&self, compact: bool) -> Cow<'_, Trimesh> {
        if compact {
            Cow::Borrowed(self.convex_hull_cached())
        } else {
            let (points, faces) = self.convex_hull_raw();
            Cow::Owned(
                Trimesh::new(points, faces, None, None).expect("convex hull mesh creation failed"),
            )
        }
    }

    /// Raw convex hull: returns all referenced vertices and unremapped faces.
    fn convex_hull_raw(&self) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
        let mask = self.vertex_mask();
        let points: Vec<Point3<f64>> = self
            .vertices
            .iter()
            .enumerate()
            .filter(|(i, _)| mask[*i])
            .map(|(_, v)| *v)
            .collect();
        let faces = crate::convex::convex_hull_3d(&points).expect("convex hull computation failed");
        (points, faces)
    }

    /// Cached compact convex hull used by internal callers (e.g. OBB).
    fn convex_hull_cached(&self) -> &Trimesh {
        self.cache_convex_hull.get_or_init(|| {
            let (points, mut faces) = self.convex_hull_raw();

            // Remap faces to reference only hull vertices, discarding interior points.
            // Capacity based on Euler's formula: V = F/2 + 2 for convex polyhedra.
            let mut map = vec![usize::MAX; points.len()];
            let mut new_points = Vec::with_capacity(faces.len() * 2 / 3 + 2);
            for face in faces.iter_mut() {
                for idx in face.iter_mut() {
                    if map[*idx] == usize::MAX {
                        map[*idx] = new_points.len();
                        new_points.push(points[*idx]);
                    }
                    *idx = map[*idx];
                }
            }

            Box::new(
                Trimesh::new(new_points, faces, None, None)
                    .expect("convex hull mesh creation failed"),
            )
        })
    }

    /// Compute the minimum-volume oriented bounding box (OBB) of the mesh.
    ///
    /// Uses the convex hull's face normals and edge-pair cross products as
    /// candidate projection directions, with 2D rotating calipers for each.
    ///
    /// Cached on first access.
    pub fn oriented_bounding_box(&self) -> &crate::convex::OrientedBoundingBox {
        self.cache_obb.get_or_init(|| {
            let hull = self.convex_hull_cached();
            crate::convex::oriented_bounding_box(&hull.vertices, &hull.faces)
        })
    }

    /// Voxelize the mesh at the given resolution.
    ///
    /// Requires the `wgpu` feature and a GPU.
    #[cfg(feature = "wgpu")]
    pub fn voxelize(
        &self,
        resolution: u32,
        fill_mode: crate::voxel::FillMode,
    ) -> crate::voxel::VoxelGrid {
        let (device, queue) =
            crate::voxel::request_device().expect("GPU required for voxelization");
        crate::voxel::VoxelGrid::from_mesh(
            device,
            queue,
            &self.vertices,
            &self.faces,
            resolution,
            fill_mode,
        )
    }

    /// Decompose the mesh into approximate convex parts.
    ///
    /// Returns a `Vec<Trimesh>` of convex hulls covering the mesh.
    /// Requires the `wgpu` feature and a GPU.
    #[cfg(feature = "wgpu")]
    pub fn decompose(&self, params: &crate::decomposition::DecompositionParams) -> Vec<Trimesh> {
        let result = self.convex_decomposition(params);
        result
            .hulls
            .into_iter()
            .filter_map(|hull| Trimesh::new(hull.vertices, hull.faces, None, None).ok())
            .collect()
    }

    /// Project the mesh onto a plane at multiple levels.
    ///
    /// Returns one `Option<Vec<Path2D>>` per level. When BREP data is
    /// available (via `face_surfaces`), circles and arcs from
    /// cylinders aligned with the projection normal are preserved as
    /// analytical `Circle2`/`Arc2` segments. Without BREP data, all
    /// segments are `Line` (equivalent to the previous polygon output).
    ///
    /// # Arguments
    /// * `normal` - Projection direction (will be normalized)
    /// * `origin` - A point on the projection plane
    /// * `levels` - Height offsets along the normal
    pub fn project(
        &self,
        normal: &Vector3<f64>,
        origin: &Point3<f64>,
        levels: &[f64],
    ) -> Vec<Option<Vec<Path2D>>> {
        let normal = normal.normalize();
        let plane = Plane::new(normal, *origin);
        let dots = crate::project::vertex_dots(&self.vertices, &normal, origin);
        let vertices_2d = plane.to_2d(&self.vertices);
        let to_3d: Option<Matrix4<f64>> = plane.transform_to_2d().try_inverse();

        // Get polygons from the existing projection algorithm
        let polygon_results =
            crate::project::project_polygons(&self.faces, &dots, &vertices_2d, levels, to_3d);

        // Collect expected circles from face surface data and deduplicate
        let mut expected_circles: Vec<(Point2<f64>, f64)> = self
            .face_surfaces
            .iter()
            .filter_map(|s| s.project_as_circle(&plane))
            .collect();
        expected_circles.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    a.0.x
                        .partial_cmp(&b.0.x)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(
                    a.0.y
                        .partial_cmp(&b.0.y)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });
        expected_circles.dedup_by(|a, b| {
            let tol = (b.1.abs() * EPSILON_RELATIVE).max(f64::EPSILON * 100.0);
            (a.1 - b.1).abs() < tol && (a.0.x - b.0.x).abs() < tol && (a.0.y - b.0.y).abs() < tol
        });

        // Group cylinder edges by circle for snap_to_cylinder_edges
        let circles_with_edges: Vec<CircleWithEdges> = {
            let mut result: Vec<CircleWithEdges> = expected_circles
                .iter()
                .map(|&(center, radius)| CircleWithEdges {
                    center,
                    radius,
                    edges: Vec::new(),
                    max_sagitta: 0.0,
                })
                .collect();

            // Find Surface grouping for per-face lookup
            if let Some(grouping) = self
                .attributes_face
                .groupings
                .iter()
                .find(|g| matches!(g.kind, GroupingKind::Surface))
            {
                for (fi, &si) in grouping.indices.iter().enumerate() {
                    if si == UNSET {
                        continue;
                    }
                    if let Some((center, radius)) = self.face_surfaces[si].project_as_circle(&plane)
                        && let Some(ci) = result.iter().position(|c| {
                            let tol = (c.radius.abs() * EPSILON_RELATIVE).max(f64::EPSILON * 100.0);
                            (c.center.x - center.x).abs() < tol
                                && (c.center.y - center.y).abs() < tol
                                && (c.radius - radius).abs() < tol
                        })
                    {
                        let face = self.faces[fi];
                        for &(a, b) in &[(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
                        {
                            result[ci].edges.push((vertices_2d[a], vertices_2d[b]));
                        }
                    }
                }
            }

            // Compute max_sagitta from the longest chord on each circle.
            // The sagitta is the maximum distance between a chord and the
            // arc it subtends: s = r - sqrt(r² - (chord/2)²).
            for c in &mut result {
                let max_chord = c
                    .edges
                    .iter()
                    .map(|(a, b)| (b - a).norm())
                    .fold(0.0_f64, f64::max);
                let half = (max_chord / 2.0).min(c.radius);
                c.max_sagitta = c.radius - (c.radius * c.radius - half * half).sqrt();
            }

            result
        };

        // Build kiddo tree from circle centers for broad-phase
        let cylinder_edges = if circles_with_edges.is_empty() {
            None
        } else {
            let circle_centers: Vec<[f64; 2]> = circles_with_edges
                .iter()
                .map(|c| [c.center.x, c.center.y])
                .collect();
            Some((
                circles_with_edges,
                ImmutableKdTree::new_from_slice(&circle_centers),
            ))
        };

        // Convert each polygon to Path2D with arc/circle detection
        polygon_results
            .into_par_iter()
            .map(|opt| {
                opt.map(|polys| {
                    polys
                        .into_par_iter()
                        .map(|p| {
                            crate::project::polygon_to_path(
                                &p,
                                &expected_circles,
                                cylinder_edges
                                    .as_ref()
                                    .map(|(edges, tree)| (edges.as_slice(), tree)),
                            )
                        })
                        .collect()
                })
            })
            .collect()
    }

    /// Create a new mesh from a subset of faces, compacting vertices.
    ///
    /// Face and vertex attributes are carried through. Materials and
    /// face surfaces referenced by the selected faces are compacted.
    #[must_use]
    pub fn submesh(&self, face_indices: &[usize]) -> Self {
        if face_indices.is_empty() {
            return Self::default();
        }

        // Extract selected faces and build vertex remapping
        let mut vertex_map = vec![usize::MAX; self.vertices.len()];
        let mut new_vertices = Vec::new();
        let mut new_faces = Vec::with_capacity(face_indices.len());

        for &fi in face_indices {
            let old_face = self.faces[fi];
            let mut new_face = [0usize; 3];
            for (j, &vi) in old_face.iter().enumerate() {
                if vertex_map[vi] == usize::MAX {
                    vertex_map[vi] = new_vertices.len();
                    new_vertices.push(self.vertices[vi]);
                }
                new_face[j] = vertex_map[vi];
            }
            new_faces.push(new_face);
        }

        // Collect which original vertices were used (in remapped order)
        let mut vertex_indices: Vec<(usize, usize)> = self
            .vertices
            .iter()
            .enumerate()
            .filter(|(i, _)| vertex_map[*i] != usize::MAX)
            .map(|(i, _)| (vertex_map[i], i))
            .collect();
        vertex_indices.sort_unstable_by_key(|(new_idx, _)| *new_idx);
        let vertex_indices: Vec<usize> = vertex_indices.into_iter().map(|(_, old)| old).collect();

        // Subset attributes
        let attributes_face = self.attributes_face.subset(face_indices);
        let attributes_vertex = self.attributes_vertex.subset(&vertex_indices);

        // Compact materials: find referenced material indices from grouping
        let mut materials = self.materials.clone();
        if let Some(mat_grouping) = attributes_face
            .groupings
            .iter()
            .find(|g| matches!(g.kind, GroupingKind::Material))
        {
            // Only keep referenced materials
            let mut mat_used = vec![false; self.materials.len()];
            for &idx in &mat_grouping.indices {
                if idx != UNSET && idx < self.materials.len() {
                    mat_used[idx] = true;
                }
            }
            // If some materials are unreferenced, compact them
            if mat_used.iter().any(|&u| !u) && !self.materials.is_empty() {
                let mut mat_remap = vec![UNSET; self.materials.len()];
                let mut new_mats = Vec::new();
                for (old, &used) in mat_used.iter().enumerate() {
                    if used {
                        mat_remap[old] = new_mats.len();
                        new_mats.push(self.materials[old].clone());
                    }
                }
                materials = new_mats;
            }
        }

        // Compact face_surfaces similarly
        let mut face_surfaces = self.face_surfaces.clone();
        if let Some(surf_grouping) = attributes_face
            .groupings
            .iter()
            .find(|g| matches!(g.kind, GroupingKind::Surface))
        {
            let mut surf_used = vec![false; self.face_surfaces.len()];
            for &idx in &surf_grouping.indices {
                if idx != UNSET && idx < self.face_surfaces.len() {
                    surf_used[idx] = true;
                }
            }
            if surf_used.iter().any(|&u| !u) && !self.face_surfaces.is_empty() {
                let mut surf_remap = vec![UNSET; self.face_surfaces.len()];
                let mut new_surfs = Vec::new();
                for (old, &used) in surf_used.iter().enumerate() {
                    if used {
                        surf_remap[old] = new_surfs.len();
                        new_surfs.push(self.face_surfaces[old].clone());
                    }
                }
                face_surfaces = new_surfs;
            }
        }

        Self {
            vertices: new_vertices,
            faces: new_faces,
            attributes_vertex,
            attributes_face,
            source: self.source.clone(),
            materials,
            face_surfaces,
            density: self.density,
            ..Default::default()
        }
    }

    /// Split the mesh into sub-meshes.
    ///
    /// - `on = None`: split by face-adjacency connected components.
    /// - `on = Some(kind)`: split by the matching face grouping attribute.
    ///
    /// Returns an empty vec for empty meshes, or a single-element vec
    /// (clone of self) if the mesh cannot be split.
    pub fn split(&self, on: Option<&GroupingKind>) -> Vec<Self> {
        if self.faces.is_empty() {
            return vec![];
        }

        let groups = match on {
            None => {
                let components =
                    crate::graph::connected_components(self.faces.len(), self.face_adjacency());
                if components.len() <= 1 {
                    return vec![self.clone()];
                }
                components
            }
            Some(kind) => {
                let grouping = self
                    .attributes_face
                    .groupings
                    .iter()
                    .find(|g| &g.kind == kind);

                match grouping {
                    None => return vec![self.clone()],
                    Some(g) => {
                        let groups = split_by_grouping(g);
                        if groups.len() <= 1 {
                            return vec![self.clone()];
                        }
                        groups
                    }
                }
            }
        };

        groups.iter().map(|fi| self.submesh(fi)).collect()
    }

    /// Calculate an axis-aligned bounding box (AABB) for the mesh,
    /// or None if the mesh is empty or degenerate.
    pub fn bounds(&self) -> Option<(Point3<f64>, Point3<f64>)> {
        if self.vertices.is_empty() {
            return None;
        }

        let (lower, upper) = self.vertices.par_iter().map(|v| (*v, *v)).reduce(
            || (self.vertices[0], self.vertices[0]),
            |(l1, u1), (l2, u2)| (l1.inf(&l2), u1.sup(&u2)),
        );

        if lower == upper {
            return None;
        }

        Some((lower, upper))
    }
}

/// Partition face indices by grouping value.
///
/// Returns groups of face indices, one per distinct grouping value.
/// UNSET faces are collected into a final group. Empty bins are filtered out.
fn split_by_grouping(grouping: &crate::attributes::Grouping) -> Vec<Vec<usize>> {
    let n_names = grouping.names.len();
    // One bin per name + one bin for UNSET
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); n_names + 1];

    for (fi, &idx) in grouping.indices.iter().enumerate() {
        if idx == UNSET || idx >= n_names {
            bins[n_names].push(fi);
        } else {
            bins[idx].push(fi);
        }
    }

    bins.into_iter().filter(|b| !b.is_empty()).collect()
}

impl From<crate::cleanup::CleanupResult> for Trimesh {
    fn from(result: crate::cleanup::CleanupResult) -> Self {
        Self {
            vertices: result.vertices,
            faces: result.faces,
            attributes_vertex: result.attributes_vertex,
            attributes_face: result.attributes_face,
            ..Default::default()
        }
    }
}

impl From<SimplifyResult> for Trimesh {
    fn from(result: SimplifyResult) -> Self {
        Self {
            vertices: result.vertices,
            faces: result.faces,
            attributes_vertex: result.attributes_vertex.unwrap_or_default(),
            attributes_face: result.attributes_face.unwrap_or_default(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::creation::create_box;
    use crate::exchange::{FileType, load};
    use crate::geometry::Geometry;
    use approx::{assert_relative_eq, relative_eq};

    #[test]
    fn test_mesh_normals() {
        let m = Trimesh::from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[0, 1, 2])
            .unwrap();
        let normals = m.face_normals();
        assert_eq!(normals.len(), 1);
        assert!(relative_eq!(
            normals[0],
            Vector3::new(0.0, 0.0, 1.0),
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_bounds() {
        let cube = create_box(&[1.0, 2.0, 3.0]);
        let bounds = cube.bounds().unwrap();
        assert!(relative_eq!(
            bounds.0,
            Point3::new(-0.5, -1.0, -1.5),
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_mesh_box() {
        let box_mesh = create_box(&[1.0, 1.0, 1.0]);
        assert_eq!(box_mesh.vertices.len(), 8);
        assert_eq!(box_mesh.faces.len(), 12);

        let bounds = box_mesh.bounds().unwrap();
        assert_eq!(bounds.0, Point3::new(-0.5, -0.5, -0.5));
        assert_eq!(bounds.1, Point3::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn test_mesh_stl() {
        let stl_data = include_bytes!("../../../test/data/unit_cube.STL");

        // STL doesn't need a resolver, pass None
        let scene = load(stl_data, Some(FileType::STL), None).unwrap();
        let mesh = match &scene.geometry[0] {
            Geometry::Mesh(m) => m,
            _ => panic!("Expected Mesh geometry"),
        };

        assert_eq!(mesh.vertices.len(), 36);
        assert_eq!(mesh.faces.len(), 12);
    }

    #[test]
    fn test_mesh_adj() {
        let box_mesh = create_box(&[1.0, 1.0, 1.0]);
        let adj = box_mesh.face_adjacency();
        let ang = box_mesh.face_adjacency_angles();
        assert_eq!(adj.len(), 18);
        assert_eq!(ang.len(), 18);

        // angles for a box should always be 0 or 90 degrees
        for a in ang.iter() {
            assert!(
                relative_eq!(*a, 0.0, epsilon = 1e-10)
                    | relative_eq!(*a, std::f64::consts::PI / 2.0, epsilon = 1e-10)
            );
        }
    }

    #[test]
    fn test_mass_properties() {
        let box_mesh = create_box(&[1.0, 1.0, 1.0]);

        // Unit cube should have volume 1
        assert!(relative_eq!(box_mesh.volume().abs(), 1.0, epsilon = 1e-10));

        // Mass equals volume for default density of 1
        assert!(relative_eq!(box_mesh.mass().abs(), 1.0, epsilon = 1e-10));

        // Center of mass at origin (use larger epsilon for floating point)
        let com = box_mesh.center_mass();
        assert!(relative_eq!(com.x, 0.0, epsilon = 1e-6));
        assert!(relative_eq!(com.y, 0.0, epsilon = 1e-6));
        assert!(relative_eq!(com.z, 0.0, epsilon = 1e-6));

        // Inertia tensor diagonal should be ~1/6 for unit cube
        let inertia = box_mesh.moment_inertia().unwrap();
        let expected = 1.0 / 6.0;
        assert!(relative_eq!(
            inertia[(0, 0)].abs(),
            expected,
            epsilon = 1e-3
        ));
        assert!(relative_eq!(
            inertia[(1, 1)].abs(),
            expected,
            epsilon = 1e-3
        ));
        assert!(relative_eq!(
            inertia[(2, 2)].abs(),
            expected,
            epsilon = 1e-3
        ));
    }

    #[test]
    fn test_is_watertight() {
        let box_mesh = create_box(&[1.0, 1.0, 1.0]);
        assert!(box_mesh.is_watertight());
    }

    #[test]
    fn test_surface_area() {
        let box_mesh = create_box(&[1.0, 1.0, 1.0]);
        // Unit cube has surface area 6
        assert!(relative_eq!(box_mesh.area(), 6.0, epsilon = 1e-10));
    }

    #[test]
    fn test_cache_performance() {
        use std::time::Instant;

        // Create a synthetic mesh with 100K faces
        const NUM_FACES: usize = 100_000;
        const NUM_VERTICES: usize = 50_000;

        // Simple LCG for deterministic pseudo-random numbers
        let mut seed: u64 = 12345;
        let mut next_random = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            seed
        };

        // Generate random vertices
        let vertices: Vec<Point3<f64>> = (0..NUM_VERTICES)
            .map(|_| {
                Point3::new(
                    (next_random() % 10000) as f64 / 100.0,
                    (next_random() % 10000) as f64 / 100.0,
                    (next_random() % 10000) as f64 / 100.0,
                )
            })
            .collect();

        // Generate random faces (indices into vertices)
        let faces: Vec<[usize; 3]> = (0..NUM_FACES)
            .map(|_| {
                [
                    next_random() as usize % NUM_VERTICES,
                    next_random() as usize % NUM_VERTICES,
                    next_random() as usize % NUM_VERTICES,
                ]
            })
            .collect();

        let mesh = Trimesh {
            vertices,
            faces,
            ..Default::default()
        };

        // First call - should compute and cache
        let start1 = Instant::now();
        let normals1 = mesh.face_normals();
        let duration1 = start1.elapsed();

        // Second call - should return cached value instantly
        let start2 = Instant::now();
        let normals2 = mesh.face_normals();
        let duration2 = start2.elapsed();

        // Verify we got the same data (same pointer)
        assert!(std::ptr::eq(normals1.as_ptr(), normals2.as_ptr()));
        assert_eq!(normals1.len(), NUM_FACES);

        // The cached call should be at least 100x faster
        // First call typically takes milliseconds, cached call takes nanoseconds
        assert!(
            duration2 < duration1 / 100,
            "Cache not effective: first call {:?}, second call {:?}",
            duration1,
            duration2
        );
    }

    #[test]
    fn test_project_to_3d_planes() {
        // Project a cube along Z and verify that transformed path
        // vertices lie on the expected planes: origin + normal * level.
        let cube = create_box(&[2.0, 2.0, 2.0]);
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let origin = Point3::new(0.0, 0.0, 0.0);
        let levels = vec![-0.5, 0.0, 0.5];

        let results = cube.project(&normal, &origin, &levels);

        for (i, level) in levels.iter().enumerate() {
            let paths = results[i].as_ref().expect("should have projection");
            for path in paths {
                let to_3d = path.to_3d.expect("should have to_3d");
                for p in &path.vertices {
                    let p3 = to_3d.transform_point(&Point3::new(p.x, p.y, 0.0));
                    let height = (p3 - origin).dot(&normal);
                    assert_relative_eq!(height, *level, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_project_to_3d_diagonal() {
        // Same test but with a diagonal normal and non-zero origin
        // to exercise both the rotation and translation in transform_to_2d.
        let cube = create_box(&[2.0, 2.0, 2.0]);
        let normal = Vector3::new(1.0, 1.0, 1.0).normalize();
        let origin = Point3::new(0.0, 0.0, 0.0);
        let levels = vec![-0.3, 0.0, 0.3];

        let results = cube.project(&normal, &origin, &levels);

        for (i, level) in levels.iter().enumerate() {
            if let Some(paths) = &results[i] {
                for path in paths {
                    let to_3d = path.to_3d.expect("should have to_3d");
                    for p in &path.vertices {
                        let p3 = to_3d.transform_point(&Point3::new(p.x, p.y, 0.0));
                        let height = (p3 - origin).dot(&normal);
                        assert_relative_eq!(height, *level, epsilon = 1e-6);
                    }
                }
            }
        }
    }

    #[test]
    fn test_project_to_path3d() {
        // Verify Polygon2D::to_path3d() produces 3D vertices on the plane.
        let cube = create_box(&[2.0, 2.0, 2.0]);
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let origin = Point3::origin();
        let levels = vec![0.0];

        let results = cube.project(&normal, &origin, &levels);
        let polys = results[0].as_ref().expect("should have projection");

        for poly in polys {
            let path3d = poly.to_path3d().expect("should produce Path3D");
            assert!(!path3d.vertices.is_empty());
            assert!(!path3d.segments.is_empty());

            // All 3D vertices should lie on the Z=0 plane
            for v in &path3d.vertices {
                assert_relative_eq!(v.z, 0.0, epsilon = 1e-6);
            }
        }
    }

    /// Test that projecting a BREP model with cylindrical holes detects
    /// the correct number of analytical circles on each cardinal axis.
    ///
    /// The featuretype model is a plate with drilled holes. Cylinder edge
    /// snapping in `snap_to_cylinder_edges` must project intersection
    /// vertices onto circles so `ring_to_segments` can detect them.
    #[test]
    fn test_project_cylinder_hole_detected_as_circle() {
        use crate::path::Segment2D;

        // Load the featuretype GLB (plate with cylindrical holes, includes BREP data)
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data/featuretype.glb");
        if !path.exists() {
            eprintln!("Skipping test: {:?} not found", path);
            return;
        }
        let data = std::fs::read(&path).unwrap();
        let scene = load(&data, Some(FileType::GLB), None).unwrap();

        // Extract the mesh
        let mesh = scene
            .geometry
            .values()
            .find_map(|g| match g {
                Geometry::Mesh(m) => Some(m.as_ref()),
                _ => None,
            })
            .expect("featuretype.glb should contain a mesh");

        // The model must have face surfaces from BREP data
        assert!(
            !mesh.face_surfaces.is_empty(),
            "featuretype.glb should have BREP face surface data"
        );

        // Helper: project along `normal` at level 0, count Circle segments
        let count_circles = |normal: Vector3<f64>| -> usize {
            let results = mesh.project(&normal, &Point3::origin(), &[0.0]);
            results[0]
                .as_ref()
                .map(|paths| {
                    paths
                        .iter()
                        .flat_map(|p| p.segments.iter())
                        .filter(|s| matches!(s, Segment2D::Circle(_)))
                        .count()
                })
                .unwrap_or(0)
        };

        // Y axis: 1 hole visible
        assert_eq!(count_circles(Vector3::y()), 1, "Y: expected 1 circle");

        // Verify the Y-axis circle has a reasonable radius (~0.0049m for this model)
        let y_results = mesh.project(&Vector3::y(), &Point3::origin(), &[0.0]);
        let circle = y_results[0]
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(|p| p.segments.iter())
            .find_map(|s| match s {
                Segment2D::Circle(c) => Some(c),
                _ => None,
            })
            .expect("Y projection should contain a circle");
        assert!(
            circle.radius > 0.001 && circle.radius < 0.05,
            "circle radius {} outside expected range [0.001, 0.05]",
            circle.radius
        );

        // Z axis: 8 holes visible from top
        assert_eq!(count_circles(Vector3::z()), 8, "Z: expected 8 circles");

        // X axis: no holes aligned with X
        assert_eq!(count_circles(Vector3::x()), 0, "X: expected 0 circles");
    }

    /// Verify that the TM_brep_faces extension loads the correct number
    /// of surfaces and that the per-face index mapping is valid.
    #[test]
    fn test_brep_faces_loaded_correctly() {
        use crate::attributes::UNSET;

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data/featuretype.glb");
        if !path.exists() {
            eprintln!("Skipping test: {:?} not found", path);
            return;
        }
        let data = std::fs::read(&path).unwrap();
        let scene = load(&data, Some(FileType::GLB), None).unwrap();

        let mesh = scene
            .geometry
            .values()
            .find_map(|g| match g {
                Geometry::Mesh(m) => Some(m.as_ref()),
                _ => None,
            })
            .expect("featuretype.glb should contain a mesh");

        // The GLB has 96 BREP face entries, 1 null → 95 valid surfaces.
        assert_eq!(
            mesh.face_surfaces.len(),
            95,
            "expected 95 non-null BREP surfaces, got {}",
            mesh.face_surfaces.len()
        );

        // Find the Surface grouping.
        let grouping = mesh
            .attributes_face
            .groupings
            .iter()
            .find(|g| matches!(g.kind, GroupingKind::Surface))
            .expect("mesh should have a Surface grouping");

        // One index per face.
        assert_eq!(
            grouping.indices.len(),
            mesh.faces.len(),
            "grouping indices length {} != face count {}",
            grouping.indices.len(),
            mesh.faces.len()
        );

        // Every non-UNSET index must be in range [0, face_surfaces.len()).
        for (fi, &si) in grouping.indices.iter().enumerate() {
            if si != UNSET {
                assert!(
                    si < mesh.face_surfaces.len(),
                    "face {} has surface index {} but only {} surfaces",
                    fi,
                    si,
                    mesh.face_surfaces.len()
                );
            }
        }

        // At least two distinct non-UNSET indices (planes and cylinders).
        let distinct: std::collections::HashSet<usize> = grouping
            .indices
            .iter()
            .copied()
            .filter(|&i| i != UNSET)
            .collect();
        assert!(
            distinct.len() >= 2,
            "expected multiple distinct surface indices, got {}",
            distinct.len()
        );

        // Some faces should be UNSET (the null BREP face).
        let n_unset = grouping.indices.iter().filter(|&&i| i == UNSET).count();
        assert!(
            n_unset > 0,
            "expected some UNSET indices for the null BREP face"
        );

        // Count surface types.
        let n_cylinders = mesh
            .face_surfaces
            .iter()
            .filter(|s| matches!(s, crate::boundary::Surface::Cylinder(_)))
            .count();
        let n_planes = mesh
            .face_surfaces
            .iter()
            .filter(|s| matches!(s, crate::boundary::Surface::Plane(_)))
            .count();
        assert_eq!(n_cylinders, 46, "expected 46 cylinder surfaces");
        assert_eq!(n_planes, 49, "expected 49 plane surfaces");
    }

    #[test]
    fn test_submesh_basic() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        // Take the first 2 faces
        let sub = cube.submesh(&[0, 1]);
        assert_eq!(sub.faces.len(), 2);
        // At most 4 unique vertices for 2 triangles on the same face of a cube
        assert!(sub.vertices.len() <= 4);
        // All face indices should be valid
        for face in &sub.faces {
            for &vi in face {
                assert!(vi < sub.vertices.len());
            }
        }
    }

    #[test]
    fn test_submesh_empty() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let sub = cube.submesh(&[]);
        assert!(sub.faces.is_empty());
        assert!(sub.vertices.is_empty());
    }

    #[test]
    fn test_split_connected_components_disjoint() {
        // Two separate triangles → should split into 2 parts
        let mesh = Trimesh::new(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(5.0, 5.0, 0.0),
                Point3::new(6.0, 5.0, 0.0),
                Point3::new(5.0, 6.0, 0.0),
            ],
            vec![[0, 1, 2], [3, 4, 5]],
            None,
            None,
        )
        .unwrap();

        let parts = mesh.split(None);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].faces.len(), 1);
        assert_eq!(parts[1].faces.len(), 1);
        // Each part should have 3 vertices
        assert_eq!(parts[0].vertices.len(), 3);
        assert_eq!(parts[1].vertices.len(), 3);
    }

    #[test]
    fn test_split_connected_components_single() {
        // A box is one connected component → returns clone of self
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let parts = cube.split(None);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].faces.len(), 12);
        assert_eq!(parts[0].vertices.len(), 8);
    }

    #[test]
    fn test_split_empty_mesh() {
        let mesh = Trimesh::default();
        let parts = mesh.split(None);
        assert!(parts.is_empty());
    }

    #[test]
    fn test_split_by_material() {
        use crate::attributes::{Grouping, GroupingKind};

        // Two triangles sharing vertices but with different materials
        let mut mesh = Trimesh::new(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [1, 3, 2]],
            None,
            None,
        )
        .unwrap();

        mesh.attributes_face.groupings.push(Grouping {
            kind: GroupingKind::Material,
            names: vec!["red".into(), "blue".into()],
            indices: vec![0, 1], // face 0 = red, face 1 = blue
        });

        let parts = mesh.split(Some(&GroupingKind::Material));
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].faces.len(), 1);
        assert_eq!(parts[1].faces.len(), 1);
    }

    #[test]
    fn test_split_by_material_with_unset() {
        use crate::attributes::{Grouping, GroupingKind, UNSET};

        let mut mesh = Trimesh::new(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
                Point3::new(2.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [1, 3, 2], [1, 4, 5]],
            None,
            None,
        )
        .unwrap();

        mesh.attributes_face.groupings.push(Grouping {
            kind: GroupingKind::Material,
            names: vec!["red".into()],
            indices: vec![0, 0, UNSET], // faces 0,1 = red, face 2 = UNSET
        });

        let parts = mesh.split(Some(&GroupingKind::Material));
        assert_eq!(parts.len(), 2);
        // First group: faces with material "red"
        assert_eq!(parts[0].faces.len(), 2);
        // Second group: UNSET face
        assert_eq!(parts[1].faces.len(), 1);
    }

    #[test]
    fn test_split_round_trip() {
        // Split a mesh into connected components, verify total face count
        // equals original and all vertex positions are present in some part.
        let mesh = Trimesh::new(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(5.0, 5.0, 0.0),
                Point3::new(6.0, 5.0, 0.0),
                Point3::new(5.0, 6.0, 0.0),
            ],
            vec![[0, 1, 2], [3, 4, 5]],
            None,
            None,
        )
        .unwrap();

        let parts = mesh.split(None);
        assert_eq!(parts.len(), 2);

        // Total face count preserved
        let total_faces: usize = parts.iter().map(|p| p.faces.len()).sum();
        assert_eq!(total_faces, mesh.faces.len());

        // All original vertex positions appear in exactly one part
        for v in &mesh.vertices {
            let found = parts.iter().any(|p| {
                p.vertices
                    .iter()
                    .any(|pv| relative_eq!(pv, v, epsilon = 1e-10))
            });
            assert!(found, "vertex {v:?} not found in any split part");
        }
    }

    #[test]
    fn test_split_no_matching_grouping() {
        let mesh = create_box(&[1.0, 1.0, 1.0]);
        // No material grouping exists → returns clone
        let parts = mesh.split(Some(&GroupingKind::Material));
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].faces.len(), 12);
    }

    #[test]
    fn test_split_preserves_vertex_attributes() {
        use crate::attributes::Attributes;
        use nalgebra::Vector4;

        // Two disjoint triangles with vertex colors
        let colors: Vec<Vector4<u8>> = vec![
            Vector4::new(255, 0, 0, 255),
            Vector4::new(0, 255, 0, 255),
            Vector4::new(0, 0, 255, 255),
            Vector4::new(255, 255, 0, 255),
            Vector4::new(0, 255, 255, 255),
            Vector4::new(255, 0, 255, 255),
        ];

        let attr_vertex = Attributes {
            colors: vec![colors.clone()],
            ..Default::default()
        };

        let mesh = Trimesh::new(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(5.0, 5.0, 0.0),
                Point3::new(6.0, 5.0, 0.0),
                Point3::new(5.0, 6.0, 0.0),
            ],
            vec![[0, 1, 2], [3, 4, 5]],
            Some(attr_vertex),
            None,
        )
        .unwrap();

        let parts = mesh.split(None);
        assert_eq!(parts.len(), 2);

        // First part should have the first 3 colors
        assert_eq!(parts[0].attributes_vertex.colors.len(), 1);
        assert_eq!(parts[0].attributes_vertex.colors[0].len(), 3);
        assert_eq!(parts[0].attributes_vertex.colors[0][0], colors[0]);
        assert_eq!(parts[0].attributes_vertex.colors[0][1], colors[1]);
        assert_eq!(parts[0].attributes_vertex.colors[0][2], colors[2]);

        // Second part should have the last 3 colors
        assert_eq!(parts[1].attributes_vertex.colors.len(), 1);
        assert_eq!(parts[1].attributes_vertex.colors[0].len(), 3);
        assert_eq!(parts[1].attributes_vertex.colors[0][0], colors[3]);
        assert_eq!(parts[1].attributes_vertex.colors[0][1], colors[4]);
        assert_eq!(parts[1].attributes_vertex.colors[0][2], colors[5]);
    }

    #[test]
    fn test_split_obj_by_object() {
        // Load two_objects.obj which has "o cube" and "o tetra" object groupings
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data/two_objects.obj");
        if !path.exists() {
            eprintln!("Skipping test: {:?} not found", path);
            return;
        }
        let data = std::fs::read(&path).unwrap();
        let scene = load(&data, Some(FileType::OBJ), None).unwrap();

        let mesh = scene
            .geometry
            .values()
            .find_map(|g| match g {
                Geometry::Mesh(m) => Some(m.as_ref()),
                _ => None,
            })
            .expect("two_objects.obj should contain a mesh");

        // Should have Object grouping
        assert!(
            mesh.attributes_face
                .groupings
                .iter()
                .any(|g| matches!(g.kind, GroupingKind::Object)),
            "mesh should have Object grouping"
        );

        // Split by object: should get cube (12 faces) and tetra (4 faces)
        let parts = mesh.split(Some(&GroupingKind::Object));
        assert_eq!(parts.len(), 2, "expected 2 objects");

        let mut face_counts: Vec<usize> = parts.iter().map(|p| p.faces.len()).collect();
        face_counts.sort_unstable();
        assert_eq!(face_counts, vec![4, 12], "expected tetra(4) + cube(12)");

        // Split by connected components should also give 2 parts
        // (cube and tetra don't share vertices)
        let cc_parts = mesh.split(None);
        assert_eq!(cc_parts.len(), 2, "expected 2 connected components");
    }

    #[test]
    fn test_concatenate_basic() {
        use crate::creation::create_icosphere;

        let a = create_icosphere(1.0, 3);
        let b = create_icosphere(2.0, 3);

        assert!(a.is_watertight(), "icosphere a not watertight");
        assert!(b.is_watertight(), "icosphere b not watertight");

        let mesh = Trimesh::concatenate(&[&a, &b]);
        assert_eq!(mesh.vertices.len(), a.vertices.len() + b.vertices.len());
        assert_eq!(mesh.faces.len(), a.faces.len() + b.faces.len());

        // Verify all face indices are valid
        for (fi, face) in mesh.faces.iter().enumerate() {
            for &vi in face {
                assert!(
                    vi < mesh.vertices.len(),
                    "face {fi} has vertex index {vi} but only {} vertices",
                    mesh.vertices.len()
                );
            }
        }

        let parts = mesh.split(None);
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn test_concatenate_empty() {
        let mesh = Trimesh::concatenate(&[]);
        assert!(mesh.vertices.is_empty());
        assert!(mesh.faces.is_empty());
    }

    #[test]
    fn test_concatenate_single() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let mesh = Trimesh::concatenate(&[&cube]);
        assert_eq!(mesh.vertices.len(), cube.vertices.len());
        assert_eq!(mesh.faces.len(), cube.faces.len());
    }

    #[test]
    fn test_concatenate_mixed_attributes() {
        use crate::attributes::{Attributes, DEFAULT_COLOR, Grouping, GroupingKind, UNSET};
        use nalgebra::Vector4;

        // Mesh A: 1 triangle with vertex colors and a material grouping
        let colors_a = vec![
            Vector4::new(255, 0, 0, 255),
            Vector4::new(0, 255, 0, 255),
            Vector4::new(0, 0, 255, 255),
        ];
        let a = Trimesh::new(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
            Some(Attributes {
                colors: vec![colors_a.clone()],
                ..Default::default()
            }),
            Some(Attributes {
                groupings: vec![Grouping {
                    kind: GroupingKind::Material,
                    names: vec!["red".into()],
                    indices: vec![0],
                }],
                ..Default::default()
            }),
        )
        .unwrap();

        // Mesh B: 1 triangle with NO vertex colors and NO material grouping
        let b = Trimesh::new(
            vec![
                Point3::new(5.0, 0.0, 0.0),
                Point3::new(6.0, 0.0, 0.0),
                Point3::new(5.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
            None,
            None,
        )
        .unwrap();

        // Mesh C: 1 triangle with vertex colors but NO material grouping
        let colors_c = vec![
            Vector4::new(255, 255, 0, 255),
            Vector4::new(0, 255, 255, 255),
            Vector4::new(255, 0, 255, 255),
        ];
        let c = Trimesh::new(
            vec![
                Point3::new(10.0, 0.0, 0.0),
                Point3::new(11.0, 0.0, 0.0),
                Point3::new(10.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
            Some(Attributes {
                colors: vec![colors_c.clone()],
                ..Default::default()
            }),
            None,
        )
        .unwrap();

        let mesh = Trimesh::concatenate(&[&a, &b, &c]);
        assert_eq!(mesh.vertices.len(), 9);
        assert_eq!(mesh.faces.len(), 3);

        // Vertex colors: A's colors, then DEFAULT_COLOR for B, then C's colors
        assert_eq!(mesh.attributes_vertex.colors.len(), 1);
        let colors = &mesh.attributes_vertex.colors[0];
        assert_eq!(colors.len(), 9);
        assert_eq!(colors[0..3], colors_a[..]);
        assert_eq!(colors[3], DEFAULT_COLOR);
        assert_eq!(colors[4], DEFAULT_COLOR);
        assert_eq!(colors[5], DEFAULT_COLOR);
        assert_eq!(colors[6..9], colors_c[..]);

        // Face groupings: Material grouping should exist with UNSET for B and C
        let mat_grouping = mesh
            .attributes_face
            .groupings
            .iter()
            .find(|g| g.kind == GroupingKind::Material)
            .expect("should have Material grouping");
        assert_eq!(mat_grouping.indices.len(), 3);
        assert_ne!(mat_grouping.indices[0], UNSET); // A has material
        assert_eq!(mat_grouping.indices[1], UNSET); // B has no material
        assert_eq!(mat_grouping.indices[2], UNSET); // C has no material
    }

    #[test]
    fn test_split_benchmark() {
        use crate::creation::create_icosphere;
        use std::time::Instant;

        // Two icospheres with 6 subdivisions each
        let a = create_icosphere(1.0, 6);
        let b = create_icosphere(2.0, 6);

        assert!(a.is_watertight(), "icosphere a not watertight");
        assert!(b.is_watertight(), "icosphere b not watertight");

        let mesh = Trimesh::concatenate(&[&a, &b]);
        assert!(mesh.is_watertight(), "concatenated mesh not watertight");
        eprintln!(
            "split benchmark: {} vertices, {} faces",
            mesh.vertices.len(),
            mesh.faces.len()
        );

        // Time each stage of split individually (cold — no cached edges)
        // Use a fresh clone so caches from is_watertight() don't affect timing
        let fresh = mesh.clone();
        let t0 = Instant::now();
        let _edges = fresh.edges_sorted();
        let t1 = Instant::now();
        let _groups = fresh.edges_grouped();
        let t2 = Instant::now();
        let adj = fresh.face_adjacency();
        let t3 = Instant::now();
        let components = crate::graph::connected_components(fresh.faces.len(), adj);
        let t4 = Instant::now();
        let parts: Vec<Trimesh> = components.iter().map(|fi| fresh.submesh(fi)).collect();
        let t5 = Instant::now();

        eprintln!(
            "  edges_sorted:        {:>8.1}ms",
            (t1 - t0).as_secs_f64() * 1000.0
        );
        eprintln!(
            "  edges_grouped:       {:>8.1}ms",
            (t2 - t1).as_secs_f64() * 1000.0
        );
        eprintln!(
            "  face_adjacency:      {:>8.1}ms",
            (t3 - t2).as_secs_f64() * 1000.0
        );
        eprintln!(
            "  connected_components:{:>8.1}ms",
            (t4 - t3).as_secs_f64() * 1000.0
        );
        eprintln!(
            "  submesh (x{}):       {:>8.1}ms",
            parts.len(),
            (t5 - t4).as_secs_f64() * 1000.0
        );
        eprintln!(
            "  total:               {:>8.1}ms",
            (t5 - t0).as_secs_f64() * 1000.0
        );

        let expected_faces = 20 * 4_usize.pow(6);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].faces.len(), expected_faces);
        assert_eq!(parts[1].faces.len(), expected_faces);

        // Both parts should be watertight
        for (i, part) in parts.iter().enumerate() {
            assert!(part.is_watertight(), "part {i} not watertight");
        }
    }

    #[test]
    fn test_edges_boundary_watertight() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        assert!(cube.is_watertight());
        assert!(cube.edges_boundary().is_empty());
    }

    #[test]
    fn test_edges_boundary_open_cube() {
        // Cube with one face removed (2 triangles) → leaves a square hole with 4 boundary edges.
        let cube = create_box(&[1.0, 1.0, 1.0]);
        // Remove the last 2 faces (one quad face of the cube)
        let open = cube.submesh(&(0..10).collect::<Vec<_>>());
        let boundary = open.edges_boundary();
        assert_eq!(
            boundary.len(),
            4,
            "expected 4 boundary edges, got {}",
            boundary.len()
        );

        // Boundary edges should form a closed loop
        let mut next_map = std::collections::HashMap::new();
        for &[a, b] in &boundary {
            next_map.insert(a, b);
        }
        // Walk the loop from the first edge
        let start = boundary[0][0];
        let mut current = start;
        for _ in 0..4 {
            current = *next_map.get(&current).expect("broken loop");
        }
        assert_eq!(current, start, "boundary edges don't form a closed loop");
    }

    #[test]
    fn test_fill_holes_open_cube() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let open = cube.submesh(&(0..10).collect::<Vec<_>>());
        assert!(!open.is_watertight());

        let filled = open.fill_holes();
        assert!(filled.is_watertight(), "filled mesh should be watertight");
        // Original 10 faces + 2 fill faces (fan-triangulating a 4-vertex loop)
        assert_eq!(filled.faces.len(), 12);
    }

    #[test]
    fn test_fill_holes_already_watertight() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        assert!(cube.is_watertight());
        let filled = cube.fill_holes();
        assert_eq!(filled.faces.len(), cube.faces.len());
        assert!(filled.is_watertight());
    }

    #[test]
    fn test_fill_holes_multiple_holes() {
        // Remove bottom (faces 0,1) and top (faces 2,3) from a cube
        // This creates 2 separate 4-edge boundary loops
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let keep: Vec<usize> = (4..12).collect();
        let open = cube.submesh(&keep);
        assert!(!open.is_watertight());

        let filled = open.fill_holes();
        assert!(
            filled.is_watertight(),
            "filled mesh should be watertight after filling 2 holes"
        );
        // 8 original + 2*2 fill = 12
        assert_eq!(filled.faces.len(), 12);
    }

    #[test]
    fn test_smooth_vertex_normals_cube_smooth() {
        // A cube with 30° threshold: all edges are 90° > 30°, so every face
        // should get its own flat normals (equivalent to faceted for a cube).
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let normals = cube.smooth_vertex_normals(std::f64::consts::FRAC_PI_6, None);
        assert_eq!(normals.len(), cube.faces.len() * 3);

        // Each face's 3 corner normals should match the face normal
        let face_normals = cube.face_normals();
        for (fi, _face) in cube.faces.iter().enumerate() {
            for ci in 0..3 {
                let cn = normals[fi * 3 + ci];
                assert_relative_eq!(cn, face_normals[fi], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_smooth_vertex_normals_cube_full() {
        // A cube with PI threshold: all edges are smooth, so corners should
        // average to normalize(±1, ±1, ±1) at each vertex.
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let normals = cube.smooth_vertex_normals(std::f64::consts::PI, None);
        assert_eq!(normals.len(), cube.faces.len() * 3);

        // Every corner normal at a cube vertex should point toward ±1,±1,±1
        for (fi, face) in cube.faces.iter().enumerate() {
            for (ci, &vi) in face.iter().enumerate() {
                let cn = normals[fi * 3 + ci];
                let v = cube.vertices[vi];
                let expected = Vector3::new(v.x.signum(), v.y.signum(), v.z.signum()).normalize();
                assert_relative_eq!(cn, expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_smooth_vertex_normals_unit_length() {
        // All returned corner normals should be unit length.
        let cube = create_box(&[2.0, 3.0, 4.0]);
        for &threshold in &[std::f64::consts::FRAC_PI_6, std::f64::consts::PI] {
            let normals = cube.smooth_vertex_normals(threshold, None);
            for n in normals.iter() {
                assert_relative_eq!(n.norm(), 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_smooth_vertex_normals_full_shared() {
        // With PI threshold, all faces sharing a vertex should produce the
        // same normal at that vertex (within tolerance).
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let normals = cube.smooth_vertex_normals(std::f64::consts::PI, None);

        // Build vertex→corner normals map
        let mut vert_normals: Vec<Vec<Vector3<f64>>> = vec![Vec::new(); cube.vertices.len()];
        for (fi, face) in cube.faces.iter().enumerate() {
            for (ci, &vi) in face.iter().enumerate() {
                vert_normals[vi].push(normals[fi * 3 + ci]);
            }
        }

        // All normals at the same vertex should be identical
        for (_vi, vn) in vert_normals.iter().enumerate() {
            if vn.len() < 2 {
                continue;
            }
            for n in &vn[1..] {
                assert_relative_eq!(*n, vn[0], epsilon = 1e-10);
            }
        }
    }
}
