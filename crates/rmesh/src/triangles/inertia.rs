//! Mass properties computation for triangle meshes.
//!
//! Implements the polyhedral mass properties algorithm from:
//! http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf

use nalgebra::{Matrix3, Point3, SymmetricEigen, Vector3};
use rayon::prelude::*;

/// Mass properties computed from a closed triangle mesh.
#[derive(Debug, Clone)]
pub struct MassProperties {
    pub density: f64,
    pub mass: f64,
    pub volume: f64,
    pub center_mass: Vector3<f64>,
    pub inertia: Option<Matrix3<f64>>,
}

impl Default for MassProperties {
    fn default() -> Self {
        Self {
            density: 1.0,
            mass: 0.0,
            volume: 0.0,
            center_mass: Vector3::zeros(),
            inertia: None,
        }
    }
}

impl MassProperties {
    /// Compute principal inertia components (eigenvalues) and vectors (eigenvectors).
    ///
    /// Returns (eigenvalues sorted descending, eigenvectors as columns of matrix).
    /// The eigenvectors form an orthonormal basis aligned with the principal axes.
    pub fn principal_inertia(&self) -> Option<(Vector3<f64>, Matrix3<f64>)> {
        let inertia = self.inertia.as_ref()?;
        let eigen = SymmetricEigen::new(*inertia);

        // Sort eigenvalues descending, reorder eigenvectors to match
        let mut indices: Vec<usize> = (0..3).collect();
        indices.sort_by(|&a, &b| {
            eigen.eigenvalues[b]
                .partial_cmp(&eigen.eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let values = Vector3::new(
            eigen.eigenvalues[indices[0]],
            eigen.eigenvalues[indices[1]],
            eigen.eigenvalues[indices[2]],
        );
        let vectors = Matrix3::from_columns(&[
            eigen.eigenvectors.column(indices[0]).clone_owned(),
            eigen.eigenvectors.column(indices[1]).clone_owned(),
            eigen.eigenvectors.column(indices[2]).clone_owned(),
        ]);

        Some((values, vectors))
    }
}

/// Compute the 10 integral components for a single triangle.
#[inline]
fn compute_face_integrals(v0: &Point3<f64>, v1: &Point3<f64>, v2: &Point3<f64>) -> [f64; 10] {
    let cross = (v1 - v0).cross(&(v2 - v0));

    let f1 = v0.coords + v1.coords + v2.coords;
    let f2 = v0.coords.component_mul(&v0.coords)
        + v1.coords.component_mul(&v1.coords)
        + v0.coords.component_mul(&v1.coords)
        + v2.coords.component_mul(&f1);
    let f3 = v0
        .coords
        .component_mul(&v0.coords)
        .component_mul(&v0.coords)
        + v0.coords
            .component_mul(&v0.coords)
            .component_mul(&v1.coords)
        + v0.coords
            .component_mul(&v1.coords)
            .component_mul(&v1.coords)
        + v1.coords
            .component_mul(&v1.coords)
            .component_mul(&v1.coords)
        + v2.coords.component_mul(&f2);

    let g0 = f2 + (v0.coords + f1).component_mul(&v0.coords);
    let g1 = f2 + (v1.coords + f1).component_mul(&v1.coords);
    let g2 = f2 + (v2.coords + f1).component_mul(&v2.coords);

    [
        cross.x * f1.x,
        cross.x * f2.x,
        cross.y * f2.y,
        cross.z * f2.z,
        cross.x * f3.x,
        cross.y * f3.y,
        cross.z * f3.z,
        cross.x * (v0.y * g0.x + v1.y * g1.x + v2.y * g2.x),
        cross.y * (v0.z * g0.y + v1.z * g1.y + v2.z * g2.y),
        cross.z * (v0.x * g0.z + v1.x * g1.z + v2.x * g2.z),
    ]
}

#[inline]
fn add_integrals(a: [f64; 10], b: [f64; 10]) -> [f64; 10] {
    std::array::from_fn(|i| a[i] + b[i])
}

/// Compute mass properties from triangles using polyhedral integration.
pub fn mass_properties(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    density: f64,
    skip_inertia: bool,
) -> MassProperties {
    if faces.is_empty() {
        return MassProperties {
            density,
            ..Default::default()
        };
    }

    let integrals: [f64; 10] = faces
        .par_iter()
        .map(|[i0, i1, i2]| compute_face_integrals(&vertices[*i0], &vertices[*i1], &vertices[*i2]))
        .reduce(|| [0.0; 10], add_integrals);

    let volume = integrals[0] / 6.0;
    let center_mass = if volume.abs() > f64::EPSILON {
        Vector3::new(
            integrals[1] / 24.0 / volume,
            integrals[2] / 24.0 / volume,
            integrals[3] / 24.0 / volume,
        )
    } else {
        let sum: Vector3<f64> = vertices
            .par_iter()
            .map(|v| v.coords)
            .reduce(Vector3::zeros, |a, b| a + b);
        sum / vertices.len() as f64
    };

    let mass = density * volume;

    if skip_inertia {
        return MassProperties {
            density,
            mass,
            volume,
            center_mass,
            inertia: None,
        };
    }

    let (x, y, z) = (center_mass.x, center_mass.y, center_mass.z);
    let ixx = integrals[4] / 60.0;
    let iyy = integrals[5] / 60.0;
    let izz = integrals[6] / 60.0;
    let ixy = integrals[7] / 120.0;
    let iyz = integrals[8] / 120.0;
    let ixz = integrals[9] / 120.0;

    let i_xx = iyy + izz - volume * (y * y + z * z);
    let i_yy = ixx + izz - volume * (x * x + z * z);
    let i_zz = ixx + iyy - volume * (x * x + y * y);
    let i_xy = -(ixy - volume * x * y);
    let i_yz = -(iyz - volume * y * z);
    let i_xz = -(ixz - volume * x * z);

    let inertia = Matrix3::new(i_xx, i_xy, i_xz, i_xy, i_yy, i_yz, i_xz, i_yz, i_zz) * density;

    MassProperties {
        density,
        mass,
        volume,
        center_mass,
        inertia: Some(inertia),
    }
}

/// Compute just the signed volume (faster than full mass_properties).
pub fn volume(vertices: &[Point3<f64>], faces: &[[usize; 3]]) -> f64 {
    if faces.is_empty() {
        return 0.0;
    }
    faces
        .par_iter()
        .map(|[i0, i1, i2]| {
            let v0 = &vertices[*i0];
            let v1 = &vertices[*i1];
            let v2 = &vertices[*i2];
            let cross = (v1 - v0).cross(&(v2 - v0));
            cross.x * (v0.x + v1.x + v2.x)
        })
        .sum::<f64>()
        / 6.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::create_box;
    use approx::assert_relative_eq;

    #[test]
    fn test_cube_volume() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let vol = volume(&cube.vertices, &cube.faces);
        assert_relative_eq!(vol.abs(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cube_mass_properties() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let props = mass_properties(&cube.vertices, &cube.faces, 1.0, false);

        assert_relative_eq!(props.volume.abs(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(props.mass.abs(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(props.center_mass.x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(props.center_mass.y, 0.0, epsilon = 1e-6);
        assert_relative_eq!(props.center_mass.z, 0.0, epsilon = 1e-6);

        let inertia = props.inertia.unwrap();
        let expected = 1.0 / 6.0;
        assert_relative_eq!(inertia[(0, 0)].abs(), expected, epsilon = 1e-3);
        assert_relative_eq!(inertia[(1, 1)].abs(), expected, epsilon = 1e-3);
        assert_relative_eq!(inertia[(2, 2)].abs(), expected, epsilon = 1e-3);
    }

    #[test]
    fn test_empty_mesh() {
        let vertices: Vec<Point3<f64>> = vec![];
        let faces: Vec<[usize; 3]> = vec![];
        let props = mass_properties(&vertices, &faces, 1.0, false);
        assert_relative_eq!(props.volume, 0.0);
    }

    #[test]
    fn test_density_scaling() {
        let cube = create_box(&[1.0, 1.0, 1.0]);
        let props = mass_properties(&cube.vertices, &cube.faces, 2.5, false);
        assert_relative_eq!(props.mass.abs(), 2.5, epsilon = 1e-10);
    }
}
