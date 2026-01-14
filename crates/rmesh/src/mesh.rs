use std::sync::OnceLock;

use anyhow::Result;

use crate::{
    attributes::{Attributes, LoadSource, Material},
    graph::adjacency,
    simplify::{SimplifyOptions, SimplifyResult, simplify_mesh},
    triangles::inertia::{self, MassProperties},
};
use nalgebra::{Matrix3, Point3, Vector2, Vector3};
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
/// 4. **Important**: Expensive compute functions (e.g., `adjacency::check_topology`)
///    should ONLY be called inside `get_or_init` closures, never directly.
///
/// # Mutation Warning
///
/// If you mutate `vertices` or `faces` after calling any cached method,
/// the cached values will be stale. Create a new `Trimesh` instead of mutating.
#[derive(Debug)]
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

    /// Optional density for mass property calculations (default 1.0)
    pub density: Option<f64>,

    // Cached derived values - computed lazily on first access
    cache_faces_cross: OnceLock<Vec<Vector3<f64>>>,
    cache_face_normals: OnceLock<Vec<Vector3<f64>>>,
    cache_faces_area: OnceLock<Vec<f64>>,
    cache_area: OnceLock<f64>,
    cache_edges: OnceLock<Vec<[usize; 2]>>,
    cache_face_adjacency: OnceLock<Vec<(usize, usize)>>,
    cache_mass_properties: OnceLock<MassProperties>,
    cache_topology: OnceLock<(bool, bool)>, // (is_watertight, is_winding_consistent)
}

impl Default for Trimesh {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
            attributes_vertex: Attributes::default(),
            attributes_face: Attributes::default(),
            source: LoadSource::default(),
            materials: Vec::new(),
            density: None,
            cache_faces_cross: OnceLock::new(),
            cache_face_normals: OnceLock::new(),
            cache_faces_area: OnceLock::new(),
            cache_area: OnceLock::new(),
            cache_edges: OnceLock::new(),
            cache_face_adjacency: OnceLock::new(),
            cache_mass_properties: OnceLock::new(),
            cache_topology: OnceLock::new(),
        }
    }
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
            density: self.density,
            // Fresh cache - will recompute on demand
            cache_faces_cross: OnceLock::new(),
            cache_face_normals: OnceLock::new(),
            cache_faces_area: OnceLock::new(),
            cache_area: OnceLock::new(),
            cache_edges: OnceLock::new(),
            cache_face_adjacency: OnceLock::new(),
            cache_mass_properties: OnceLock::new(),
            cache_topology: OnceLock::new(),
        }
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

    /// Simplify the mesh to a target face count.
    ///
    /// Preserves vertex attributes (normals, UVs, colors) through interpolation.
    #[must_use]
    pub fn simplify(&self, target_count: usize, aggressiveness: f64) -> Self {
        let options = SimplifyOptions {
            target_count,
            aggressiveness,
            preserve_attributes: true,
            ..Default::default()
        };

        let result = simplify_mesh(
            &self.vertices,
            &self.faces,
            Some(&self.attributes_vertex),
            Some(&self.attributes_face),
            options,
        );

        Self {
            vertices: result.vertices,
            faces: result.faces,
            attributes_vertex: result.attributes_vertex,
            attributes_face: result.attributes_face,
            ..Default::default()
        }
    }

    /// Simplify the mesh with full control over options.
    ///
    /// Returns the full SimplifyResult including quality metrics if requested.
    #[must_use]
    pub fn simplify_with_options(&self, options: SimplifyOptions) -> SimplifyResult {
        simplify_mesh(
            &self.vertices,
            &self.faces,
            Some(&self.attributes_vertex),
            Some(&self.attributes_face),
            options,
        )
    }

    /// Subdivide the mesh by splitting each triangle into 4 triangles.
    ///
    /// Each edge is split at its midpoint. After N iterations,
    /// the face count is multiplied by 4^N.
    /// Face attributes (colors) are propagated to child faces.
    #[must_use]
    pub fn subdivide(&self, iterations: usize) -> Self {
        let (vertices, faces, attributes_face) = crate::subdivide::subdivide_with_attributes(
            &self.vertices,
            &self.faces,
            &self.attributes_face,
            iterations,
        );
        Self {
            vertices,
            faces,
            attributes_face,
            ..Default::default()
        }
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
                .map(|cross| cross.normalize())
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

    /// A helper method to get the UV coordinate attributes
    /// stored in `mesh.attributes_vertex`.
    pub fn uv(&self) -> Option<&Vec<Vector2<f64>>> {
        self.attributes_vertex.uv.first()
    }

    /// What are the pairs of face indices that share an edge?
    ///
    /// Cached on first access. Uses sort-based algorithm for cache efficiency.
    pub fn face_adjacency(&self) -> &[(usize, usize)] {
        self.cache_face_adjacency
            .get_or_init(|| adjacency::face_adjacency(&self.faces))
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

    /// Get cached topology info: (is_watertight, is_winding_consistent).
    fn topology(&self) -> (bool, bool) {
        *self
            .cache_topology
            .get_or_init(|| adjacency::check_topology(&self.faces))
    }

    /// Check if the mesh is watertight (all edges shared by exactly 2 faces).
    pub fn is_watertight(&self) -> bool {
        self.topology().0
    }

    /// Check if face winding is consistent across the mesh.
    ///
    /// For consistent winding, adjacent faces must traverse their shared edge
    /// in opposite directions. If face A has edge (v0→v1), face B must have (v1→v0).
    pub fn is_winding_consistent(&self) -> bool {
        self.topology().1
    }

    /// Check if the mesh represents a valid volume.
    ///
    /// A mesh is a volume if it is watertight, has consistent winding,
    /// and has positive volume (outward-facing normals).
    pub fn is_volume(&self) -> bool {
        let (watertight, consistent) = self.topology();
        watertight && consistent && self.volume() > 0.0
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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::creation::create_box;
    use crate::exchange::{MeshFormat, load_mesh};
    use approx::relative_eq;

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

        let mesh = load_mesh(stl_data, MeshFormat::STL).unwrap();

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
}
