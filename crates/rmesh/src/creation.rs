use anyhow::Result;
use approx::relative_eq;
use nalgebra::{Matrix3, Matrix4, Point2, Point3, Rotation3, SVD, Transform3, Unit, Vector3};

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
    let half_extents = [extents[0] / 2.0, extents[1] / 2.0, extents[2] / 2.0];

    // Flattened vertices array
    let vertices = vec![
        -half_extents[0],
        -half_extents[1],
        -half_extents[2],
        half_extents[0],
        -half_extents[1],
        -half_extents[2],
        half_extents[0],
        half_extents[1],
        -half_extents[2],
        -half_extents[0],
        half_extents[1],
        -half_extents[2],
        -half_extents[0],
        -half_extents[1],
        half_extents[2],
        half_extents[0],
        -half_extents[1],
        half_extents[2],
        half_extents[0],
        half_extents[1],
        half_extents[2],
        -half_extents[0],
        half_extents[1],
        half_extents[2],
    ];

    // Flattened faces array
    let faces = vec![
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5,
        3, 0, 4, 3, 4, 7,
    ];

    // Create the mesh using Trimesh::from_slice
    Trimesh::from_slice(&vertices, &faces).unwrap()
}

use earcut::Earcut;

/// A wrapper object for a triangulator
pub struct Triangulator {
    earcut: Option<Earcut<f64>>,
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
    ///
    /// Returns
    /// ------------
    /// triangles
    ///  The triangles referencing `vertices`
    pub fn trianglate_2d(
        &mut self,
        exterior: &[usize],
        interiors: &[Vec<usize>],
        vertices: &[Point2<f64>],
    ) -> Vec<(usize, usize, usize)> {
        // lazily initialize the earcut triangulator
        if self.earcut.is_none() {
            self.earcut = Some(Earcut::new());
        }
        let earcut = self.earcut.as_mut().unwrap();

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

        // convert the flat result into a list of triangles
        result
            .chunks_exact(3)
            .map(|chunk| (chunk[0], chunk[1], chunk[2]))
            .collect()
    }

    /// Triangulate a polygon in 3D space by fitting a plane to the exterior
    /// and then triangulating the projected points in 2D space returning
    /// the indices of the triangles in the original 3D space.
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
    ) -> Result<Vec<(usize, usize, usize)>> {
        // find a plane for the vertices in our exterior as not every vertex may be referenced
        let fittable: Vec<Point3<f64>> = exterior.iter().map(|i| vertices[*i]).collect();
        let plane = Plane::from_points(&fittable, true)?;
        // project the 3D vertices into the plane so we can triangulate them in 2D
        let on_plane = plane.to_2d(vertices);

        Ok(self.trianglate_2d(exterior, interiors, &on_plane))
    }
}

/// Triangulate a polygon using a triangle fan. This requires no knowledge
/// of the position of the vertices, but may produce incorrect triangulations
/// for non-convex polygons and does not support interiors.
///
/// Parameters
/// -------------
/// exterior
///   The exterior of the polygon as indices of a vertex list
///
/// Returns
/// ------------
/// triangles
///  The triangles referencing vertex indexes.
pub fn triangulate_fan(exterior: &[usize]) -> Vec<(usize, usize, usize)> {
    (1..exterior.len() - 1)
        .map(|i| (exterior[0], exterior[i], exterior[i + 1]))
        .collect()
}
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
    /// on a mesh), or using the SVD method for points that may be noisy like a laser scan.
    ///
    /// Parameters
    /// -------------
    /// points
    ///   The points to fit our current plane to
    /// method_cross
    ///   Picks some arbitrary points that meet a heuristic for "probably not colinear"
    ///   and then runs the cross product to find the normal
    ///   Otherwise use the optimization methods to find the best fit plane.
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
            // need to find a 3-subset of them that aren't colinear
            // this loops through the points offsetting by a third of the
            // array length, which if the points have "locality" should give
            // us a good change of finding a nicely distant non-colinear group
            for i in 0..third {
                // pick 3 points
                let p0 = points[i];
                let p1 = points[third + i];
                let p2 = points[2 * third + i];

                // get the two vectors
                let v1 = p1 - p0;
                let v2 = p2 - p0;

                // run the cross product
                let normal = v1.cross(&v2);
                if normal.norm() > 1e-10 {
                    // return a plane
                    return Ok(Plane::new(normal.normalize(), p0));
                }
            }
        }

        // todo : this should probably be least squares?
        // Use the SVD method
        let centroid = points
            .iter()
            .fold(Vector3::zeros(), |acc, p| acc + p.coords)
            / points.len() as f64;

        let mut covariance = Matrix3::zeros();
        for p in points {
            let centered = p.coords - centroid;
            covariance += centered * centered.transpose();
        }

        let svd = SVD::new(covariance, true, true);
        let normal = svd.v_t.unwrap().row(2).transpose().normalize();

        Ok(Plane::new(normal, Point3::from(centroid)))
    }

    /// Calculate an arbitrary but deterministic homogeneous transformation
    /// that moves from the XY plane to the plane defined by this object.
    ///
    /// Returns
    /// -------------
    /// transform
    ///   The transformation matrix that moves from the XY plane to this plane.
    pub fn to_plane(&self) -> Matrix4<f64> {
        // this transform aligns the vectors then offsets the origin
        align_vectors(self.normal, Vector3::z()).append_translation(&Vector3::new(
            -self.origin.x,
            -self.origin.y,
            -self.origin.z,
        ))
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
        let transform = self.to_plane();
        points
            .iter()
            .map(|p| {
                let p = Point3::from_homogeneous(transform * p.to_homogeneous()).unwrap();
                Point2::new(p.x, p.y)
            })
            .collect()
    }

    /// Convert 2D points into 3D points by applying the inverse of the
    /// transformation matrix defined by this object.
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
        let transform = self.to_plane().try_inverse().unwrap();
        points
            .iter()
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
pub fn align_vectors(a: Vector3<f64>, b: Vector3<f64>) -> Matrix4<f64> {
    // Normalize the input vectors
    let a = Unit::new_normalize(a);
    let b = Unit::new_normalize(b);

    // if they are the same vector we can just return the identity matrix
    if relative_eq!(a, b, epsilon = f64::EPSILON) {
        return Transform3::identity().to_homogeneous();
    }

    // find the axis as the mutually perpendicular vector from the cross product
    let axis = a.cross(&b);
    // find the angle between the two vectors
    let angle = a.dot(&b).acos();

    if axis.norm() < f64::EPSILON {
        // If the axis is zero, it means the vectors are opposite
        // We can rotate by 180 degrees around any perpendicular axis
        let perp = Unit::new_normalize(perpendicular(&a));
        return Rotation3::from_axis_angle(&perp, std::f64::consts::PI).to_homogeneous();
    }

    // Normalize the axis and create the rotation matrix
    let axis = Unit::new_normalize(axis);
    Rotation3::from_axis_angle(&axis, angle).to_homogeneous()
}

/// Find an arbitrary vector that is perpendicular to the given
/// 3D vector.
///
/// Parameters
/// -------------
/// v
///  The vector to find a perpendicular vector to.
///
/// Returns
/// -------------
/// perpendicular
///   Any perpendicular vector to `v`.
///
pub fn perpendicular(v: &Vector3<f64>) -> Vector3<f64> {
    // Find a vector that is perpendicular to v
    if v.x.abs() > v.y.abs() {
        Vector3::new(-v.z, 0.0, v.x) / v.x.abs()
    } else {
        Vector3::new(0.0, v.z, -v.y) / v.y.abs()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

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
        for theta in 0..3600 {
            let a = Vector3::new(1.0, 0.0, 0.0);
            let b = Rotation3::from_axis_angle(
                &Vector3::z_axis(),
                ((theta as f64) / 10.0).to_radians(),
            )
            .transform_vector(&a);
            let rotation = align_vectors(a, b);

            // Check if the rotation matrix rotates a to b
            let rotated_a = rotation * a.to_homogeneous();
            assert_relative_eq!(rotated_a.x, b.x, epsilon = 1e-6);
            assert_relative_eq!(rotated_a.y, b.y, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_plane_2D() {
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
    fn test_mesh_box() {
        let box_mesh = create_box(&[1.0, 1.0, 1.0]);
        assert_eq!(box_mesh.vertices.len(), 8);
        assert_eq!(box_mesh.faces.len(), 12);

        let bounds = box_mesh.bounds().unwrap();
        assert_eq!(bounds.0, Point3::new(-0.5, -0.5, -0.5));
        assert_eq!(bounds.1, Point3::new(0.5, 0.5, 0.5));
    }
}
