use nalgebra::{
    Matrix3, Matrix4, Point2, Point3, Rotation3, SVD, Transform3, Translation3, Unit, Vector3,
};

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

/// Triangulate a face using a triangle fan, which requires no knowledge
/// of the positions of the vertices. This works for convex polygons.
fn triangulate_fan(f: &[usize]) -> Vec<(usize, usize, usize)> {
    (1..f.len() - 1).map(|i| (f[0], f[i], f[i + 1])).collect()
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

    pub fn triangulate_3d(
        &mut self,
        exterior: &[usize],
        interiors: &[Vec<usize>],
        vertices: &[Point3<f64>],
    ) -> Vec<(usize, usize, usize)> {
        if self.earcut.is_none() {
            // lazily initialize the earcut triangulator
            self.earcut = Some(Earcut::new());
        }
        let earcut = self.earcut.as_mut().unwrap();

        // Convert the 3D points to 2D points using a plane fit
        let plane = Plane::from_points(vertices, true);
        let projected = plane.to_2D(vertices);
        let mut flat = exterior
            .iter()
            .map(|i| {
                let coords = projected[*i].coords;
                [coords[0], coords[1]]
            })
            .collect::<Vec<[f64; 2]>>();

        // todo : extend this to support holes
        let mut holes = vec![];
        for interior in interiors {
            holes.push(flat.len());
            flat.extend(
                interior
                    .iter()
                    .map(|i| {
                        let coords = projected[*i].coords;
                        [coords[0], coords[1]]
                    })
                    .collect::<Vec<[f64; 2]>>(),
            );
        }

        let mut result: Vec<usize> = vec![];
        earcut.earcut(flat, &[], &mut result);

        result
            .chunks_exact(3)
            .map(|chunk| (chunk[0], chunk[1], chunk[2]))
            .collect()
    }
}

pub struct Plane {
    pub normal: Vector3<f64>,
    pub origin: Point3<f64>,
}

impl Plane {
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
    ///   and then runs the cross product to find the normal.
    ///   Otherwise use the SVD method to find the best fit plane.
    ///
    /// Returns
    /// ------------
    /// plane
    ///   The plane that best fits the points using the specified method.
    ///  

    pub fn from_points(points: &[Point3<f64>], method_cross: bool) -> Self {
        assert!(
            points.len() >= 3,
            "At least 3 points are required to define a plane."
        );

        if method_cross {
            // Use the minimal cross-product method with a point-picking strategy
            let third = points.len() / 3;

            for i in 0..=third {
                let p0 = points[i];
                let p1 = points[third + i];
                let p2 = points[2 * third + i];

                let v1 = p1 - p0;
                let v2 = p2 - p0;

                let normal = v1.cross(&v2);
                if normal.norm() > 1e-6 {
                    return Plane::new(normal.normalize(), p0);
                }
            }
        }

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

        Plane::new(normal, Point3::from(centroid))
    }

    /// Generate arbitrary but deterministic basis vectors for the plane's
    /// X and Y axes that are orthogonal to the normal vector and each other,
    /// with the plane normal serving as the Z axis.
    ///
    /// Returns
    /// ----------
    /// x_axis
    ///   A unit vector on the plane that we've picked as an arbitray X direction.
    /// y_axis
    ///   A unit vector on the plane that we've picked as an arbitray Y direction.
    pub fn basis(&self) -> (Vector3<f64>, Vector3<f64>) {
        let normal = self.normal;

        let mut x = Vector3::new(-normal.y, normal.x, 0.0);
        if x.norm() < 1e-6 {
            // If the normal is aligned with the y-axis, use the z-axis instead
            x = Vector3::new(-normal.z, normal.y, 0.0);
        }
        let x_axis = x.normalize();
        let y_axis = normal.cross(&x_axis).normalize();

        (x_axis, y_axis)
    }

    pub fn to_plane(&self) -> Matrix4<f64> {
        let o = self.origin;
        align_vectors(self.normal, Vector3::z()).append_translation(&Vector3::new(-o.x, -o.y, -o.z))
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
    pub fn to_2D(&self, points: &[Point3<f64>]) -> Vec<Point2<f64>> {
        let transform = self.to_plane();
        points
            .iter()
            .map(|p| {
                let p = Point3::from_homogeneous(transform * p.to_homogeneous()).unwrap();
                Point2::new(p.x, p.y)
            })
            .collect()
    }

    pub fn to_3D(&self, points: &[Point2<f64>]) -> Vec<Point3<f64>> {
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

pub fn align_vectors(a: Vector3<f64>, b: Vector3<f64>) -> Matrix4<f64> {
    let a = Unit::new_normalize(a);
    let b = Unit::new_normalize(b);

    //let mut m = Transformation3::identity();
    if a == b {
        // No rotation needed
        return Transform3::identity().to_homogeneous();
    }
    let axis = a.cross(&b);
    let angle = a.dot(&b).acos();

    // todo : check for zero axis and return a reverse

    // Normalize the axis and create the rotation matrix
    let axis = Unit::new_normalize(axis);
    Rotation3::from_axis_angle(&axis, angle).to_homogeneous()
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
    fn test_plane_2D() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let plane = Plane::from_points(&points, true);

        assert_eq!(plane.normal, Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(plane.origin, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(plane.normal.norm(), 1.0);

        let projected = plane.to_2D(&points);
        assert_eq!(projected.len(), points.len());
        assert_relative_eq!(projected[0], Point2::new(0.0, 0.0), epsilon = 1e-6);
        assert_relative_eq!(projected[1], Point2::new(1.0, 0.0), epsilon = 1e-6);

        let back = plane.to_3D(&projected);
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
