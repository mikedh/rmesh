//! Mesh creation utilities
//!
//! This module provides utilities for creating and manipulating meshes:
//! - Primitive creation (boxes, etc.)
//! - Triangulation (earcut-based 2D/3D triangulation)
//! - Plane fitting and projection
//! - Feature-based CAD system

pub mod feature;

use anyhow::Result;
use nalgebra::{
    Isometry3, Matrix3, Matrix4, Point2, Point3, Rotation3, Unit, UnitQuaternion, Vector3,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::mesh::Trimesh;

/// Create a mesh of a box centered at the origin with the
/// specified axis aligned bounding box size.
///
/// Parameters
/// -------------
/// extents
///   The size of the box in each dimension.
///
/// Returns
/// -------------
///  A Trimesh representing the box.
pub fn create_box(extents: &[f64; 3]) -> Trimesh {
    // half extents for the box
    let half = [extents[0] / 2.0, extents[1] / 2.0, extents[2] / 2.0];

    // Vertices as Vec<Point3<f64>>
    let vertices = vec![
        Point3::new(-half[0], -half[1], -half[2]),
        Point3::new(half[0], -half[1], -half[2]),
        Point3::new(half[0], half[1], -half[2]),
        Point3::new(-half[0], half[1], -half[2]),
        Point3::new(-half[0], -half[1], half[2]),
        Point3::new(half[0], -half[1], half[2]),
        Point3::new(half[0], half[1], half[2]),
        Point3::new(-half[0], half[1], half[2]),
    ];

    // Faces as Vec<[usize; 3]> - CCW winding for outward normals
    let faces = vec![
        // Bottom face (-Z) - viewed from below, vertices go CCW
        [0, 2, 1],
        [0, 3, 2],
        // Top face (+Z) - viewed from above, vertices go CCW
        [4, 5, 6],
        [4, 6, 7],
        // Front face (-Y)
        [0, 1, 5],
        [0, 5, 4],
        // Back face (+Y)
        [2, 3, 7],
        [2, 7, 6],
        // Right face (+X)
        [1, 2, 6],
        [1, 6, 5],
        // Left face (-X)
        [0, 4, 7],
        [0, 7, 3],
    ];

    // use the constructor to properly initialize cache fields
    Trimesh::new(vertices, faces, None, None).unwrap()
}

/// Create a mesh of a regular tetrahedron centered at the origin.
///
/// Uses the symmetric embedding where vertices are at:
/// `(s, s, s), (s, -s, -s), (-s, s, -s), (-s, -s, s)`
/// with `s = edge / (2 * sqrt(2))`.
///
/// Parameters
/// -------------
/// edge
///   The edge length of the tetrahedron.
///
/// Returns
/// -------------
///  A Trimesh representing the regular tetrahedron.
pub fn create_tetrahedron(edge: f64) -> Trimesh {
    let s = edge / (2.0 * std::f64::consts::SQRT_2);

    let vertices = vec![
        Point3::new(s, s, s),
        Point3::new(s, -s, -s),
        Point3::new(-s, s, -s),
        Point3::new(-s, -s, s),
    ];

    // CCW outward winding for each face.
    // Each face is the triangle opposite to the vertex not included.
    let faces = vec![
        [0, 3, 1], // opposite vertex 2
        [0, 1, 2], // opposite vertex 3
        [0, 2, 3], // opposite vertex 1
        [1, 3, 2], // opposite vertex 0
    ];

    Trimesh::new(vertices, faces, None, None).unwrap()
}

/// Create a regular icosahedron mesh centered at the origin.
///
/// 12 vertices, 20 faces, inscribed in a sphere of the given radius.
pub fn create_icosahedron(radius: f64) -> Trimesh {
    // Golden ratio φ = (1 + √5) / 2, not a midpoint calculation
    #[allow(clippy::manual_midpoint)]
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let len = (1.0 + phi * phi).sqrt();
    let a = radius / len;
    let b = radius * phi / len;

    let vertices = vec![
        Point3::new(-a, b, 0.0),
        Point3::new(a, b, 0.0),
        Point3::new(-a, -b, 0.0),
        Point3::new(a, -b, 0.0),
        Point3::new(0.0, -a, b),
        Point3::new(0.0, a, b),
        Point3::new(0.0, -a, -b),
        Point3::new(0.0, a, -b),
        Point3::new(b, 0.0, -a),
        Point3::new(b, 0.0, a),
        Point3::new(-b, 0.0, -a),
        Point3::new(-b, 0.0, a),
    ];

    let faces = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    Trimesh::new(vertices, faces, None, None).unwrap()
}

/// Create an icosphere mesh centered at the origin.
///
/// Subdivides a regular icosahedron and projects vertices onto the sphere.
/// Face count = `20 * 4^subdivisions`.
pub fn create_icosphere(radius: f64, subdivisions: usize) -> Trimesh {
    let ico = create_icosahedron(radius);
    let mut vertices = ico.vertices;
    let mut faces = ico.faces;

    for _ in 0..subdivisions {
        (vertices, faces) = {
            let (v, f, _) = crate::subdivide::subdivide(&vertices, &faces, None, 1);
            (v, f)
        };
        for v in &mut vertices {
            let len = v.coords.norm();
            if len > 0.0 {
                v.coords *= radius / len;
            }
        }
    }

    Trimesh::new(vertices, faces, None, None).unwrap()
}

mod earcut;
use earcut::Earcut;

/// A wrapper object for a triangulator
pub struct Triangulator {
    // lazily initialized earcut triangulator
    earcut: Option<Earcut<f64>>,
}

impl Default for Triangulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Triangulator {
    pub fn new() -> Self {
        Triangulator { earcut: None }
    }

    /// Triangulate a 2D polygon using the earcut algorithm.
    ///
    /// Parameters
    /// -------------
    /// exterior
    ///   The exterior of the polygon to triangulate as
    ///   indices of `vertices`
    /// interiors
    ///   The interior holes of the polygon to triangulate.
    /// vertices
    ///   The 2D vertices of the polygon.
    /// local_indices
    ///   If true, return triangle indices local to the polygon
    ///   (0..exterior.len()). If false, remap through `exterior`
    ///   to return indices into `vertices`.
    ///
    /// Returns
    /// ------------
    /// triangles
    ///  The triangles referencing `vertices`
    pub fn triangulate_2d(
        &mut self,
        exterior: &[usize],
        interiors: &[Vec<usize>],
        vertices: &[Point2<f64>],
        local_indices: bool,
    ) -> Vec<[usize; 3]> {
        let earcut = self.earcut.get_or_insert_with(Earcut::new);

        // start with a flattening of the exterior
        let mut flat = exterior
            .iter()
            .map(|i| [vertices[*i].x, vertices[*i].y])
            .collect::<Vec<[f64; 2]>>();

        // the holes are represented as offsets into the flat array
        // for wherever the interior holes start
        let mut holes = vec![];
        for interior in interiors {
            holes.push(flat.len());
            flat.extend(
                interior
                    .iter()
                    .map(|i| [vertices[*i].x, vertices[*i].y])
                    .collect::<Vec<[f64; 2]>>(),
            );
        }

        // run the triangulator
        let mut result: Vec<usize> = vec![];
        earcut.earcut(flat, &holes, &mut result);

        if local_indices {
            // return indices into the polygon (exterior then interiors)
            return result
                .chunks_exact(3)
                .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                .collect();
        }

        // Build index mapping: earcut returns indices into `flat`, we need original vertex indices
        // flat[0..exterior.len()] maps to exterior, then interiors follow
        let mut index_map: Vec<usize> = exterior.to_vec();
        for interior in interiors {
            index_map.extend(interior);
        }

        // convert the flat result into triangles with original vertex indices
        result
            .chunks_exact(3)
            .map(|chunk| {
                [
                    index_map[chunk[0]],
                    index_map[chunk[1]],
                    index_map[chunk[2]],
                ]
            })
            .collect()
    }

    /// Triangulate a polygon in 3D space by fitting a plane to the exterior
    /// and then triangulating the projected points in 2D space.
    ///
    /// Parameters
    /// -------------
    /// exterior
    ///   The exterior of the polygon to triangulate as
    ///   indices of `vertices`
    /// interiors
    ///   The interior holes of the polygon to triangulate.
    /// vertices
    ///   The 3D vertices of the polygon.
    /// local_indices
    ///   If true, return triangle indices local to the polygon
    ///   (0..exterior.len()). If false, remap through `exterior`
    ///   to return indices into `vertices`.
    ///
    /// Returns
    /// ------------
    /// triangles
    ///  The triangles referencing `vertices`
    pub fn triangulate_3d(
        &mut self,
        exterior: &[usize],
        interiors: &[Vec<usize>],
        vertices: &[Point3<f64>],
        local_indices: bool,
        fan_fallback: bool,
    ) -> Result<Vec<[usize; 3]>> {
        // find a plane for the vertices in our exterior as not every vertex may be referenced
        let fittable: Vec<Point3<f64>> = exterior.iter().map(|i| vertices[*i]).collect();
        // use the cross product method to find a plane which works well for exactly planar points
        let result = Plane::from_points(&fittable, true).map(|plane| {
            let on_plane = plane.to_2d(vertices);
            self.triangulate_2d(exterior, interiors, &on_plane, local_indices)
        });

        match result {
            Ok(tris) => Ok(tris),
            Err(e) if fan_fallback => Ok(triangulate_fan(exterior, local_indices)),
            Err(e) => Err(e),
        }
    }
}

/// Triangulate a polygon using a triangle fan. This requires no knowledge
/// of the position of the vertices but may produce incorrect triangulations
/// for non-convex polygons and does not support interiors.
///
/// Parameters
/// -------------
/// exterior
///   The exterior of the polygon as indices of a vertex list.
/// local_indices
///   If true, return triangle indices local to the polygon
///   (0..exterior.len()). If false, remap through `exterior`
///   to return indices into the original vertex array.
///
/// Returns
/// ------------
/// triangles
///  The triangles referencing vertex indexes.
pub fn triangulate_fan(exterior: &[usize], local_indices: bool) -> Vec<[usize; 3]> {
    if local_indices {
        (1..exterior.len() - 1).map(|i| [0, i, i + 1]).collect()
    } else {
        (1..exterior.len() - 1)
            .map(|i| [exterior[0], exterior[i], exterior[i + 1]])
            .collect()
    }
}

/// A plane defined by a normal vector and origin point.
///
/// Used for projecting 3D points to 2D and fitting planes to point clouds.
pub struct Plane {
    pub normal: Vector3<f64>,
    pub origin: Point3<f64>,
}

impl Plane {
    /// Create a new plane with the specified normal vector and origin point.
    ///
    /// Parameters
    /// -------------
    /// normal
    ///   The normal vector of the plane.
    /// origin
    ///  The origin point of the plane.
    ///
    /// Returns
    /// ------------
    /// plane
    ///  The new plane object.
    pub fn new(normal: Vector3<f64>, origin: Point3<f64>) -> Self {
        Plane { normal, origin }
    }

    /// Fit a plane to a point cloud using either lazy minimal cross products
    /// for points that we know should lie exactly on a plane (i.e. polygon face
    /// on a mesh), or using a least squares method for points that may not be
    /// exactly planar.
    ///
    /// Parameters
    /// -------------
    /// points
    ///   The points to fit our current plane to
    /// method_cross
    ///   Picks three arbitrary points that meet a heuristic for "probably not
    ///   colinear" and then runs the cross product to find the normal. If not
    ///   set will use optimization methods to fit a plane.
    ///
    /// Returns
    /// ------------
    /// plane
    ///   The plane that best fits the points using the specified method.
    pub fn from_points(points: &[Point3<f64>], method_cross: bool) -> Result<Self> {
        if points.len() < 3 {
            return Err(anyhow::anyhow!(
                "At least 3 points are required to define a plane."
            ));
        }
        if method_cross {
            // Use the minimal cross-product method with a point-picking strategy
            let third = points.len() / 3;

            // if all the points are on the same plane we just
            // need to find a subset of 3 of them that aren't colinear
            // this loops through the points offsetting by a third of the
            // array length, which if the points have "locality" should give
            // us a good change of finding a nicely distant non-colinear group
            for i in 0..third {
                // pick 3 arbitrary points
                let p0 = points[i];
                let p1 = points[third + i];
                let p2 = points[2 * third + i];

                // get the two vectors
                let v1 = p1 - p0;
                let v2 = p2 - p0;

                // run the cross product
                let normal = v1.cross(&v2);
                // this should only be zero if the points are colinear or identical
                if normal.norm() > 1e-10 {
                    // we have a nonzero norm so return a plane
                    return Ok(Plane::new(normal.normalize(), p0));
                }
            }
        }

        // get the centroid of the points
        let centroid = points
            .iter()
            .fold(Vector3::zeros(), |acc, p| acc + p.coords)
            / points.len() as f64;

        // calculate the covariance matrix with parallelism
        let covariance = points
            .par_iter()
            .map(|p| {
                let centered = p.coords - centroid;
                centered * centered.transpose()
            })
            .reduce(Matrix3::zeros, |a, b| a + b);

        // eigen decomposition for least squares plane fit
        let eig = covariance.symmetric_eigen();
        let normal = eig.eigenvectors.column(0).normalize();

        Ok(Plane::new(normal, Point3::from(centroid)))
    }

    /// Calculate an arbitrary but deterministic homogeneous transformation
    /// that moves from the XY plane to the plane defined by this object.
    ///
    /// Returns
    /// -------------
    /// transform
    ///   The transformation matrix that moves from the XY plane to this plane.
    pub fn transform_to_2d(&self) -> Matrix4<f64> {
        // Rotation that maps our normal onto Z, then translate so
        // the plane origin maps to the world origin.
        let rotation = align_vectors(self.normal, Vector3::z());
        let translation = (rotation * (-self.origin.coords)).into();
        let rotation = UnitQuaternion::from_rotation_matrix(&rotation);
        Isometry3::from_parts(translation, rotation).to_homogeneous()
    }

    /// Project 3D points onto the plane defined by this object.
    ///
    /// Parameters
    /// -------------
    /// points
    ///  The points to project onto the plane.
    /// Returns
    /// -------------
    /// projected
    ///   The projected points in 2D space.
    pub fn to_2d(&self, points: &[Point3<f64>]) -> Vec<Point2<f64>> {
        let transform = self.transform_to_2d();
        points
            .par_iter()
            .map(|p| {
                let p = Point3::from_homogeneous(transform * p.to_homogeneous()).unwrap();
                Point2::new(p.x, p.y)
            })
            .collect()
    }

    /// Convert 2D points into 3D points by applying the inverse
    /// of the transformation matrix defined by this object.
    ///
    /// Parameters
    /// -------------
    /// points
    ///   The 2D points to convert into 3D points.
    ///
    /// Returns
    /// -------------
    /// converted
    ///   The converted points in 3D space.
    pub fn to_3d(&self, points: &[Point2<f64>]) -> Vec<Point3<f64>> {
        let transform = self.transform_to_2d().try_inverse().unwrap();
        points
            .par_iter()
            .map(|p| {
                Point3::from_homogeneous(transform * Point3::new(p.x, p.y, 0.0).to_homogeneous())
                    .unwrap()
            })
            .collect()
    }
}

/// Align two vectors in 3D space by calculating the rotation matrix
/// that rotates the first vector to the second vector.
///
/// Parameters
/// -------------
/// a
///   The first vector.
/// b
///   The second vector.
///
/// Returns
/// -------------
/// rotation
///   The rotation matrix that rotates `a` to `b`.
pub fn align_vectors(a: Vector3<f64>, b: Vector3<f64>) -> Rotation3<f64> {
    let a = Unit::new_normalize(a);
    let b = Unit::new_normalize(b);

    // `rotation_between` returns None for anti-parallel vectors
    Rotation3::rotation_between(a.as_ref(), b.as_ref()).unwrap_or_else(|| {
        let perp = Unit::new_normalize(perpendicular(&a));
        Rotation3::from_axis_angle(&perp, std::f64::consts::PI)
    })
}

/// Find an arbitrary vector that is perpendicular to a
/// given 3D vector, or if the input vector is zero will
/// return a zero vector.
///
/// Parameters
/// -------------
/// vec
///  The vector to find a perpendicular vector to.
///
/// Returns
/// -------------
/// perpendicular
///   Any perpendicular vector to `v`.
pub fn perpendicular(vec: &Vector3<f64>) -> Vector3<f64> {
    if vec.norm() < f64::EPSILON {
        // a zero vector should return a zero vector
        Vector3::new(0.0, 0.0, 0.0)
    } else if vec.x.abs() > vec.y.abs() {
        // if the x component is the largest, we can use the y and z components
        Vector3::new(-vec.z, 0.0, vec.x).normalize()
    } else {
        // otherwise we can use the x and z components
        Vector3::new(0.0, vec.z, -vec.y).normalize()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    /// Helper function to create a linear space of values
    fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
        let step = (end - start) / (count as f64 - 1.0);
        (0..count).map(|i| start + i as f64 * step).collect()
    }

    #[test]
    fn test_mesh_normals() {
        let m = Trimesh::from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[0, 1, 2])
            .unwrap();
        let normals = m.face_normals();
        assert_eq!(normals.len(), 1);
        assert_relative_eq!(normals[0], Vector3::new(0.0, 0.0, 1.0), epsilon = 1e-6);
    }

    #[test]
    fn test_align_vectors() {
        for theta in linspace(0.0, 360.0, 10000) {
            let a = Vector3::new(1.0, 0.0, 0.0);
            let b = Rotation3::from_axis_angle(&Vector3::z_axis(), (theta / 10.0).to_radians())
                .transform_vector(&a);
            let rotation = align_vectors(a, b);

            // Check if the rotation matrix rotates a to b
            let rotated_a = rotation * a;
            assert_relative_eq!(rotated_a.x, b.x, epsilon = 1e-6);
            assert_relative_eq!(rotated_a.y, b.y, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_plane_2d() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let plane = Plane::from_points(&points, true).unwrap();

        assert_eq!(plane.normal, Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(plane.origin, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(plane.normal.norm(), 1.0);

        let projected = plane.to_2d(&points);
        assert_eq!(projected.len(), points.len());
        assert_relative_eq!(projected[0], Point2::new(0.0, 0.0), epsilon = 1e-6);
        assert_relative_eq!(projected[1], Point2::new(1.0, 0.0), epsilon = 1e-6);

        let back = plane.to_3d(&projected);
        assert_eq!(back.len(), points.len());
        for i in 0..points.len() {
            assert_relative_eq!(back[i], points[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_perpendicular() {
        // check through a grid of of vectors including the cardinal axes
        // should always return a perpendicular vector or if
        // the input is zero return a zero vector
        for x in linspace(-1.0, 1.0, 20) {
            for y in linspace(-1.0, 1.0, 20) {
                for z in linspace(-1.0, 1.0, 20) {
                    let v = Vector3::new(x, y, z);
                    if v.norm() > 0.0 {
                        let perp = perpendicular(&v);
                        // should never include NaN or Inf
                        assert!(perp.x.is_finite() && perp.y.is_finite() && perp.z.is_finite());

                        // a zero vector should return a zero vector
                        if v.x == 0.0 && v.y == 0.0 && v.z == 0.0 {
                            assert_eq!(perp, Vector3::new(0.0, 0.0, 0.0));
                        }

                        // the dot product of the two vectors should always be zero
                        let dot = v.dot(&perp);
                        assert!(dot.is_finite());
                        assert!(dot.abs() < 1e-10, "v: {v:?}, perp: {perp:?}");
                    }
                }
            }
        }
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
    fn test_tetrahedron_basic() {
        let tet = create_tetrahedron(1.0);
        assert_eq!(tet.vertices.len(), 4);
        assert_eq!(tet.faces.len(), 4);
        assert!(tet.is_watertight());
        assert!(tet.is_convex());
        assert!(tet.volume() > 0.0);
    }

    #[test]
    fn test_tetrahedron_edge_lengths() {
        let edge = 2.5;
        let tet = create_tetrahedron(edge);
        let lengths = tet.edges_unique_length();
        assert_eq!(lengths.len(), 6);
        for &len in &lengths {
            assert_relative_eq!(len, edge, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tetrahedron_volume() {
        let edge = 3.0;
        let tet = create_tetrahedron(edge);
        let expected = edge.powi(3) / (6.0 * std::f64::consts::SQRT_2);
        assert_relative_eq!(tet.volume(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_icosahedron() {
        let ico = create_icosahedron(1.0);
        assert_eq!(ico.vertices.len(), 12);
        assert_eq!(ico.faces.len(), 20);
        assert!(ico.is_watertight());
        assert!(ico.is_convex());
        assert!(ico.volume() > 0.0);
        // All vertices on the unit sphere
        for v in &ico.vertices {
            assert_relative_eq!(v.coords.norm(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_icosphere_basic() {
        let sphere = create_icosphere(1.0, 0);
        assert_eq!(sphere.vertices.len(), 12);
        assert_eq!(sphere.faces.len(), 20);
        assert!(sphere.is_watertight());
        assert!(sphere.is_convex());
        assert!(sphere.volume() > 0.0);
    }

    #[test]
    fn test_icosphere_subdivisions() {
        for sub in 1..=4 {
            let sphere = create_icosphere(1.0, sub);
            let expected_faces = 20 * 4_usize.pow(sub as u32);
            assert_eq!(sphere.faces.len(), expected_faces);
            assert!(sphere.is_watertight());
            assert!(sphere.volume() > 0.0);
        }
    }

    #[test]
    fn test_icosphere_radius() {
        let radius = 2.5;
        let sphere = create_icosphere(radius, 3);
        // All vertices should be on the sphere surface
        for v in &sphere.vertices {
            assert_relative_eq!(v.coords.norm(), radius, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_icosphere_volume_converges() {
        let radius: f64 = 1.0;
        let expected = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        // Volume should converge toward 4/3 pi r^3
        let vol_3 = create_icosphere(radius, 3).volume();
        let vol_5 = create_icosphere(radius, 5).volume();
        // 5 subdivisions should be closer to the analytical value
        assert!(
            (vol_5 - expected).abs() < (vol_3 - expected).abs(),
            "vol_3={vol_3}, vol_5={vol_5}, expected={expected}"
        );
        assert_relative_eq!(vol_5, expected, epsilon = 5e-3);
    }
}
