use std::sync::{Arc, RwLock};

use ahash::AHashMap;

use anyhow::{anyhow, Result};
use itertools::Itertools;
use nalgebra::{convert, Point3, Vector3};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::simplify::simplify_mesh;

#[derive(Default, Debug, Clone)]
struct InnerCache {
    face_adjacency: Option<Arc<Vec<(usize, usize)>>>, // cache for face adjacency
    face_normals: Option<Arc<Vec<Vector3<f64>>>>,     // cache for face normals
}

#[pyclass]
pub struct Trimesh {
    pub vertices: Vec<Point3<f64>>,
    pub faces: Vec<(usize, usize, usize)>,

    _cache: RwLock<InnerCache>,
}

impl Clone for Trimesh {
    // Implement a custom clone method to avoid copying the cache
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            faces: self.faces.clone(),
            _cache: RwLock::new(InnerCache::default()),
        }
    }
}

impl Trimesh {
    pub fn new(vertices: Vec<Point3<f64>>, faces: Vec<(usize, usize, usize)>) -> Self {
        Self {
            vertices,
            faces,
            _cache: RwLock::new(InnerCache::default()),
        }
    }

    pub fn simplify(&self, target_count: usize) -> Self {
        let (vertices, faces) =
            simplify_mesh(&self.vertices, &self.faces, target_count, 1.0, false);

        Trimesh {
            vertices,
            faces,
            _cache: RwLock::new(InnerCache::default()),
        }
    }

    /// Create a Trimesh from flat slices of vertices and faces.
    pub fn from_slice(vertices: &[f64], faces: &[usize]) -> Result<Self> {
        if vertices.len() % 3 != 0 {
            return Err(anyhow::anyhow!("Vertices must be a multiple of 3"));
        }
        if faces.len() % 3 != 0 {
            return Err(anyhow::anyhow!("Faces must be a multiple of 3"));
        }

        let v = vertices
            .chunks_exact(3)
            .map(|chunk| Point3::new(chunk[0], chunk[1], chunk[2]))
            .collect::<Vec<_>>();

        let f: Vec<(usize, usize, usize)> = faces
            .chunks_exact(3)
            .map(|chunk| (chunk[0], chunk[1], chunk[2]))
            .collect();

        Ok(Self {
            vertices: v,
            faces: f,

            _cache: RwLock::new(InnerCache::default()),
        })
    }

    /// Calculate the normals for each face of the mesh.
    pub fn face_normals(&self) -> Arc<Vec<Vector3<f64>>> {
        if self._cache.read().unwrap().face_normals.is_none() {
            let vertices = &self.vertices;

            let temp = self
                .faces
                .par_iter()
                .map(|face| {
                    let v0 = vertices[face.0];
                    let v1 = vertices[face.1];
                    let v2 = vertices[face.2];
                    ((v1 - v0).cross(&(v2 - v0))).normalize()
                })
                .collect();
            let mut cache = self._cache.write().unwrap();
            cache.face_normals = Some(Arc::new(temp));
        }

        self._cache
            .read()
            .unwrap()
            .face_normals
            .as_ref()
            .unwrap()
            .clone()
    }

    // Get the edges calculated from the faces
    pub fn edges(&self) -> Vec<[usize; 2]> {
        self.faces
            .par_iter()
            .flat_map(|face| vec![[face.0, face.1], [face.1, face.2], [face.2, face.0]])
            .collect()
    }

    // What are the pairs of face indices that share an edge?
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

    /// Calculate an axis-aligned bounding box (AABB) for the mesh.
    pub fn bounds(&self) -> Result<(Point3<f64>, Point3<f64>)> {
        if self.vertices.is_empty() {
            return Err(anyhow::anyhow!("Mesh has no vertices"));
        }

        // start with bounds from the first vertex
        let (mut lower, mut upper) = (self.vertices[0].clone(), self.vertices[0].clone());
        for vertex in self.vertices.iter().skip(1) {
            // use componentwise min/max
            lower = lower.inf(&vertex);
            upper = upper.sup(&vertex);
        }

        if lower == upper {
            return Err(anyhow::anyhow!("All vertices are the same"));
        }

        Ok((lower, upper))
    }
}

pub struct BinaryStl {
    pub header: String,
    pub triangles: Vec<BinaryStlTriangle>,
}
#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BinaryStlTriangle {
    pub normal: Vector3<f32>,
    pub vertices: [Point3<f32>; 3],
    pub attributes: u16,
}

impl BinaryStlTriangle {
    pub fn normal(&self) -> Vector3<f32> {
        self.normal
    }

    pub fn convert_vertices(&self) -> [Point3<f64>; 3] {
        [
            convert(self.vertices[0]),
            convert(self.vertices[1]),
            convert(self.vertices[2]),
        ]
    }
}

impl BinaryStl {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 84 {
            return Err(anyhow::anyhow!("STL file too short"));
        }

        let header = String::from_utf8_lossy(&bytes[0..80]).to_string();
        // let triangle_count = u32::from_le_bytes(bytes[80..84].try_into().unwrap());

        let triangles: &[BinaryStlTriangle] = bytemuck::try_cast_slice(&bytes[84..])
            .map_err(|_e| anyhow!("Could not interpret bytes as STL triangles!"))?;

        Ok(Self {
            header,
            triangles: triangles.to_vec(),
        })
    }

    pub fn to_mesh(&self) -> Trimesh {
        // convert STL f32 vertices to f64
        let vertices: Vec<Point3<f64>> = self
            .triangles
            .par_iter()
            .flat_map(|triangle| triangle.convert_vertices())
            .collect();

        let faces: Vec<(usize, usize, usize)> = (0..vertices.len()).tuples().collect();

        Trimesh::new(vertices, faces)
    }
}

// An enum to represent the different mesh file formats.
enum MeshFileFormat {
    STL,
    OBJ,
    PLY,
}

/// Load a mesh from a file, doing no initial processing.
pub fn load_mesh(file_data: &[u8], file_type: MeshFileFormat) -> Result<Trimesh> {
    match file_type {
        MeshFileFormat::STL => Ok(BinaryStl::from_bytes(file_data)?.to_mesh()),
        MeshFileFormat::OBJ => todo!(),
        MeshFileFormat::PLY => todo!(),
    }
}

pub fn create_box(extents: &[f64; 3]) -> Trimesh {
    let half_extents = [extents[0] / 2.0, extents[1] / 2.0, extents[2] / 2.0];
    let vertices = vec![
        Point3::new(-half_extents[0], -half_extents[1], -half_extents[2]),
        Point3::new(half_extents[0], -half_extents[1], -half_extents[2]),
        Point3::new(half_extents[0], half_extents[1], -half_extents[2]),
        Point3::new(-half_extents[0], half_extents[1], -half_extents[2]),
        Point3::new(-half_extents[0], -half_extents[1], half_extents[2]),
        Point3::new(half_extents[0], -half_extents[1], half_extents[2]),
        Point3::new(half_extents[0], half_extents[1], half_extents[2]),
        Point3::new(-half_extents[0], half_extents[1], half_extents[2]),
    ];

    let faces = vec![
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (2, 3, 7),
        (2, 7, 6),
        (1, 2, 6),
        (1, 6, 5),
        (3, 0, 4),
        (3, 4, 7),
    ];

    Trimesh::new(vertices, faces)
}

#[cfg(test)]
mod tests {

    use super::*;
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
        let stl_data = include_bytes!("../test/data/unit_cube.STL");

        let mesh = load_mesh(stl_data, MeshFileFormat::STL).unwrap();

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
