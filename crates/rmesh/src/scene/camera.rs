//! Camera types and projection matrices.

use nalgebra::Matrix4;

/// Camera projection type.
#[derive(Debug, Clone)]
pub enum CameraProjection {
    /// Perspective projection with field of view.
    Perspective {
        /// Vertical field of view in radians.
        fov_y: f64,
        /// Aspect ratio (width/height). If None, uses viewport aspect.
        aspect: Option<f64>,
        /// Near clipping plane distance.
        znear: f64,
        /// Far clipping plane distance.
        zfar: f64,
    },
    /// Orthographic projection.
    Orthographic {
        /// Half-width of the view volume.
        xmag: f64,
        /// Half-height of the view volume.
        ymag: f64,
        /// Near clipping plane distance.
        znear: f64,
        /// Far clipping plane distance.
        zfar: f64,
    },
}

impl Default for CameraProjection {
    fn default() -> Self {
        CameraProjection::Perspective {
            fov_y: std::f64::consts::FRAC_PI_4, // 45 degrees
            aspect: None,
            znear: 0.1,
            zfar: 1000.0,
        }
    }
}

/// A camera in the scene.
#[derive(Debug, Clone, Default)]
pub struct Camera {
    /// Human-readable name.
    pub name: String,
    /// Projection parameters.
    pub projection: CameraProjection,
}

impl Camera {
    /// Create a perspective camera with the given parameters.
    pub fn perspective(fov_y: f64, znear: f64, zfar: f64) -> Self {
        Camera {
            name: String::new(),
            projection: CameraProjection::Perspective {
                fov_y,
                aspect: None,
                znear,
                zfar,
            },
        }
    }

    /// Create an orthographic camera with the given parameters.
    pub fn orthographic(xmag: f64, ymag: f64, znear: f64, zfar: f64) -> Self {
        Camera {
            name: String::new(),
            projection: CameraProjection::Orthographic {
                xmag,
                ymag,
                znear,
                zfar,
            },
        }
    }

    /// Compute the projection matrix for the given aspect ratio.
    pub fn projection_matrix(&self, viewport_aspect: f64) -> Matrix4<f64> {
        match &self.projection {
            CameraProjection::Perspective {
                fov_y,
                aspect,
                znear,
                zfar,
            } => {
                let aspect = aspect.unwrap_or(viewport_aspect);
                let f = 1.0 / (fov_y / 2.0).tan();
                let nf = 1.0 / (znear - zfar);

                Matrix4::new(
                    f / aspect, 0.0, 0.0, 0.0,
                    0.0, f, 0.0, 0.0,
                    0.0, 0.0, (zfar + znear) * nf, 2.0 * zfar * znear * nf,
                    0.0, 0.0, -1.0, 0.0,
                )
            }
            CameraProjection::Orthographic {
                xmag,
                ymag,
                znear,
                zfar,
            } => {
                let nf = 1.0 / (znear - zfar);

                Matrix4::new(
                    1.0 / xmag, 0.0, 0.0, 0.0,
                    0.0, 1.0 / ymag, 0.0, 0.0,
                    0.0, 0.0, 2.0 * nf, (zfar + znear) * nf,
                    0.0, 0.0, 0.0, 1.0,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perspective_camera() {
        let cam = Camera::perspective(std::f64::consts::FRAC_PI_4, 0.1, 100.0);
        let proj = cam.projection_matrix(1.0);
        // Check that projection matrix is valid (non-zero determinant would fail for perspective)
        assert!(proj[(0, 0)] > 0.0);
        assert!(proj[(1, 1)] > 0.0);
    }

    #[test]
    fn test_orthographic_camera() {
        let cam = Camera::orthographic(10.0, 10.0, 0.1, 100.0);
        let proj = cam.projection_matrix(1.0);
        assert!((proj[(0, 0)] - 0.1).abs() < 1e-10);
        assert!((proj[(1, 1)] - 0.1).abs() < 1e-10);
    }
}
