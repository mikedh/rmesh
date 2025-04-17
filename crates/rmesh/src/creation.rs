use nalgebra::Point3;

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
    use nalgebra::Vector3;

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
}
