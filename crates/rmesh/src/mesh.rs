use std::sync::RwLock;

use ahash::AHashMap;

use anyhow::Result;

use crate::{
    attributes::{Attributes, LoadSource},
    simplify::simplify_mesh,
};
use nalgebra::{Point3, Vector2, Vector3};
use rayon::prelude::*;
use rmesh_macro::cache_access;

// The inner cache for the Trimesh struct. Any field that uses
// the `#[cache_access]` macro will be stored here.
#[derive(Default, Debug, Clone)]
pub struct InnerCache {
    pub face_adjacency: Option<Vec<(usize, usize)>>,
    pub face_normals: Option<Vec<Vector3<f64>>>, // cache for face normals

    pub edges: Option<Vec<[usize; 2]>>,

    pub face_adjacency_angles: Option<Vec<f64>>,
    pub faces_cross: Option<Vec<Vector3<f64>>>,
    pub faces_area: Option<Vec<f64>>,
    pub area: Option<f64>,
}

#[derive(Default, Debug)]
pub struct Trimesh {
    pub vertices: Vec<Point3<f64>>,
    pub faces: Vec<(usize, usize, usize)>,

    // A flat list of attributes so we can define things like
    // multiple colors, normals, uv coordinates, etc and can pick
    // which ones we want to use at runtime or at the application level
    pub attributes_vertex: Attributes,
    pub attributes_face: Attributes,

    // information about where the mesh came from
    pub source: LoadSource,

    // the cached values computed for the mesh
    pub _cache: RwLock<InnerCache>,
}

impl Clone for Trimesh {
    fn clone(&self) -> Self {
        let cache = self._cache.read().unwrap();
        Self {
            vertices: self.vertices.clone(),
            faces: self.faces.clone(),
            _cache: RwLock::new(cache.clone()),
            ..Default::default()
        }
    }
}

impl Trimesh {
    /// Create a new trimesh from a vec of tuple values.
    pub fn new(
        vertices: Vec<Point3<f64>>,
        faces: Vec<(usize, usize, usize)>,
        attributes_vertex: Option<Attributes>,
        attributes_face: Option<Attributes>,
    ) -> Result<Self> {
        Ok(Self {
            vertices,
            faces,
            attributes_vertex: attributes_vertex.unwrap_or_default(),
            attributes_face: attributes_face.unwrap_or_default(),
            _cache: RwLock::new(InnerCache::default()),
            ..Default::default()
        })
    }

    /// Create a Trimesh from flat slices of vertices and faces.
    pub fn from_slice(vertices: &[f64], faces: &[usize]) -> Result<Self> {
        let vertices: Vec<Point3<f64>> = vertices
            .chunks_exact(3)
            .map(|chunk| Point3::new(chunk[0], chunk[1], chunk[2]))
            .collect();

        let faces: Vec<(usize, usize, usize)> = faces
            .chunks_exact(3)
            .map(|chunk| (chunk[0], chunk[1], chunk[2]))
            .collect();

        Ok(Self {
            vertices,
            faces,
            _cache: RwLock::new(InnerCache::default()),
            ..Default::default()
        })
    }

    pub fn simplify(&self, target_count: usize, aggressiveness: f64) -> Self {
        let (vertices, faces) = simplify_mesh(
            &self.vertices,
            &self.faces,
            target_count,
            aggressiveness,
            false,
        );

        Self {
            vertices,
            faces,
            _cache: RwLock::new(InnerCache::default()),
            ..Default::default()
        }
    }

    /// Calculate the normals for each face of the mesh.
    #[cache_access]
    pub fn face_normals(&self) -> Vec<Vector3<f64>> {
        self.faces_cross()
            .par_iter()
            .map(|cross| cross.normalize())
            .collect()
    }

    // Get the edges calculated from the faces
    #[cache_access]
    pub fn edges(&self) -> Vec<[usize; 2]> {
        self.faces
            .par_iter()
            .flat_map(|face| vec![[face.0, face.1], [face.1, face.2], [face.2, face.0]])
            .collect()
    }

    /// The non-normalized cross product of every face.
    #[cache_access]
    pub fn faces_cross(&self) -> Vec<Vector3<f64>> {
        self.faces
            .par_iter()
            .map(|face| {
                let v0 = self.vertices[face.0];
                let v1 = self.vertices[face.1];
                let v2 = self.vertices[face.2];
                (v1 - v0).cross(&(v2 - v0))
            })
            .collect()
    }

    /// The area for each triangle in the mesh.
    #[cache_access]
    pub fn faces_area(&self) -> Vec<f64> {
        self.faces_cross()
            .par_iter()
            .map(|cross| cross.norm() / 2.0)
            .collect()
    }

    /// The summed area of every triangle in the mesh.
    #[cache_access]
    pub fn area(&self) -> f64 {
        self.faces_area().iter().sum()
    }

    /// A helper method to get the UV coordinate attributes
    /// stored in `mesh.attributes_vertex`.
    pub fn uv(&self) -> Option<&Vec<Vector2<f64>>> {
        self.attributes_vertex.uv.get(0)
    }

    // What are the pairs of face indices that share an edge?
    #[cache_access]
    pub fn face_adjacency(&self) -> Vec<(usize, usize)> {
        let mut edge_map = AHashMap::new();
        let mut adjacency = Vec::new();

        for (i, edge) in self.edges().iter().enumerate() {
            // there are 3 edges per triangle
            let face_index = i / 3;
            // sorted edge for querying
            let edge = [edge[0].min(edge[1]), edge[0].max(edge[1])];
            if let Some(other) = edge_map.get(&edge) {
                // add the face index to the adjacency list
                adjacency.push((*other, face_index));
            } else {
                // add the edge to the map for checking later
                edge_map.insert(edge, face_index);
            }
        }

        adjacency
    }

    // Calculate the angles between adjacent faces.
    pub fn face_adjacency_angles(&self) -> Vec<f64> {
        let adjacency = self.face_adjacency();
        let normals = self.face_normals();
        adjacency
            .par_iter()
            .map(|adj| normals[adj.0].angle(&normals[adj.1]))
            .collect()
    }

    pub fn smooth_shaded(&self, threshold: f64) {
        // get the angles between adjacent faces
        let angles = self.face_adjacency_angles();
        let index: Vec<usize> = (0..angles.len())
            .into_par_iter()
            .filter(|i| angles[*i] < threshold)
            .collect();

        let adjacency = self.face_adjacency();
    }

    /// Calculate an axis-aligned bounding box (AABB) for the mesh,
    /// or an error if the mesh is empty.
    ///
    /// Returns
    /// ------------
    /// bounds
    ///   The axis-aligned bounding box of the mesh.
    pub fn bounds(&self) -> Result<(Point3<f64>, Point3<f64>)> {
        if self.vertices.is_empty() {
            return Err(anyhow::anyhow!("Mesh has no vertices"));
        }

        let (mut lower, mut upper) = (self.vertices[0], self.vertices[0]);
        for vertex in self.vertices.iter().skip(1) {
            // use componentwise min/max
            lower = lower.inf(vertex);
            upper = upper.sup(vertex);
        }

        if lower == upper {
            return Err(anyhow::anyhow!("All vertices are the same"));
        }

        Ok((lower, upper))
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
}
