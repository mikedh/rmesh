//! Analytical surface types from BREP faces (ISO 10303-42 elementary surfaces).
//!
//! When a STEP file is loaded via cascadio, each triangle can be tagged with the
//! BREP face it came from. These types capture the analytical surface definition
//! so that projection and buffer operations can preserve exact geometry (e.g.
//! cylinders project as circles, arcs from fillets are preserved through offset).

use nalgebra::{Point2, Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::creation::{Plane, perpendicular};

// ============================================================================
// Tolerance Constants
// ============================================================================
//
// These constants define numerical tolerances used throughout surface computations.
// They are organized by purpose to ensure consistent behavior.

/// Tolerance for detecting zero-length vectors and degenerate geometry.
/// Used for normalizing directions and checking for degenerate normals.
pub const GEOMETRY_TOL: f64 = 1e-12;

/// Tolerance for detecting zero curvature (effectively planar regions).
pub const CURVATURE_TOL: f64 = 1e-12;

/// Tolerance for Newton-Raphson convergence in parameter finding.
pub const NEWTON_TOL: f64 = 1e-10;

/// Tolerance for knot vector comparisons in B-spline evaluation.
/// This is tighter than geometry tolerance due to the sensitivity of basis functions.
pub const KNOT_TOL: f64 = 1e-14;

/// Tolerance for detecting degenerate metric tensors (first fundamental form).
pub const METRIC_TOL: f64 = 1e-14;

// ============================================================================
// Surface Curvature
// ============================================================================

/// Principal curvatures and derived quantities at a point on a surface.
///
/// Used for curvature-aware tessellation: the maximum edge length that maintains
/// chord tolerance ε is L_max = sqrt(8ε / κ_max).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceCurvature {
    /// Maximum principal curvature (κ₁)
    pub kappa_1: f64,
    /// Minimum principal curvature (κ₂)
    pub kappa_2: f64,
    /// Gaussian curvature K = κ₁ × κ₂
    pub gaussian: f64,
    /// Mean curvature H = (κ₁ + κ₂) / 2
    pub mean: f64,
}

impl SurfaceCurvature {
    /// Create curvature from principal curvatures.
    pub fn from_principal(kappa_1: f64, kappa_2: f64) -> Self {
        Self {
            kappa_1,
            kappa_2,
            gaussian: kappa_1 * kappa_2,
            mean: f64::midpoint(kappa_1, kappa_2),
        }
    }

    /// Zero curvature (for planar surfaces).
    pub const fn zero() -> Self {
        Self {
            kappa_1: 0.0,
            kappa_2: 0.0,
            gaussian: 0.0,
            mean: 0.0,
        }
    }

    /// Maximum absolute principal curvature.
    pub fn kappa_max(&self) -> f64 {
        self.kappa_1.abs().max(self.kappa_2.abs())
    }

    /// Maximum edge length that maintains the given chord tolerance.
    ///
    /// From the chord error formula: ε = κL²/8, we get L_max = sqrt(8ε/κ).
    /// Returns infinity for zero curvature (planar surfaces).
    pub fn max_edge_length(&self, tolerance: f64) -> f64 {
        let k = self.kappa_max();
        if k < CURVATURE_TOL {
            f64::INFINITY
        } else {
            (8.0 * tolerance / k).sqrt()
        }
    }
}

/// An analytical surface from a BREP model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Surface {
    Plane(SurfacePlane),
    Cylinder(Cylinder),
    Cone(Cone),
    Sphere(Sphere),
    Torus(Torus),
    BSpline(SurfaceBSpline),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfacePlane {
    pub origin: Point3<f64>,
    pub normal: Vector3<f64>,
}

impl SurfacePlane {
    /// Curvature at any point on a plane is zero.
    pub fn curvature_at(&self, _u: f64, _v: f64) -> SurfaceCurvature {
        SurfaceCurvature::zero()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(into = "CylinderSerde")]
pub struct Cylinder {
    pub origin: Point3<f64>,
    pub axis: Vector3<f64>,
    pub radius: f64,
    // Precomputed orthonormal basis perpendicular to axis
    axis_unit: Vector3<f64>,
    x_basis: Vector3<f64>,
    y_basis: Vector3<f64>,
}

#[derive(Serialize, Deserialize)]
struct CylinderSerde {
    origin: Point3<f64>,
    axis: Vector3<f64>,
    radius: f64,
}

impl From<CylinderSerde> for Cylinder {
    fn from(s: CylinderSerde) -> Self {
        Self::new(s.origin, s.axis, s.radius)
    }
}

impl From<Cylinder> for CylinderSerde {
    fn from(c: Cylinder) -> Self {
        Self { origin: c.origin, axis: c.axis, radius: c.radius }
    }
}

impl<'de> Deserialize<'de> for Cylinder {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        CylinderSerde::deserialize(deserializer).map(Into::into)
    }
}

impl Cylinder {
    pub fn new(origin: Point3<f64>, axis: Vector3<f64>, radius: f64) -> Self {
        let axis_unit = axis.normalize();
        let x_basis = perpendicular(&axis_unit).normalize();
        let y_basis = axis_unit.cross(&x_basis);
        Self { origin, axis, radius, axis_unit, x_basis, y_basis }
    }

    /// Precomputed orthonormal basis perpendicular to axis.
    #[inline]
    pub fn basis(&self) -> (Vector3<f64>, Vector3<f64>) {
        (self.x_basis, self.y_basis)
    }

    /// Precomputed unit axis direction.
    #[inline]
    pub fn axis_unit(&self) -> Vector3<f64> {
        self.axis_unit
    }

    /// Curvature of a cylinder is constant: κ₁ = 1/r around the circumference, κ₂ = 0 along the axis.
    pub fn curvature_at(&self, _theta: f64, _h: f64) -> SurfaceCurvature {
        SurfaceCurvature::from_principal(1.0 / self.radius, 0.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(into = "ConeSerde")]
pub struct Cone {
    pub apex: Point3<f64>,
    pub axis: Vector3<f64>,
    pub half_angle: f64,
    // Precomputed orthonormal basis perpendicular to axis
    axis_unit: Vector3<f64>,
    x_basis: Vector3<f64>,
    y_basis: Vector3<f64>,
}

#[derive(Serialize, Deserialize)]
struct ConeSerde {
    apex: Point3<f64>,
    axis: Vector3<f64>,
    half_angle: f64,
}

impl From<ConeSerde> for Cone {
    fn from(s: ConeSerde) -> Self {
        Self::new(s.apex, s.axis, s.half_angle)
    }
}

impl From<Cone> for ConeSerde {
    fn from(c: Cone) -> Self {
        Self { apex: c.apex, axis: c.axis, half_angle: c.half_angle }
    }
}

impl<'de> Deserialize<'de> for Cone {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ConeSerde::deserialize(deserializer).map(Into::into)
    }
}

impl Cone {
    pub fn new(apex: Point3<f64>, axis: Vector3<f64>, half_angle: f64) -> Self {
        let axis_unit = axis.normalize();
        let x_basis = perpendicular(&axis_unit).normalize();
        let y_basis = axis_unit.cross(&x_basis);
        Self { apex, axis, half_angle, axis_unit, x_basis, y_basis }
    }

    /// Precomputed orthonormal basis perpendicular to axis.
    #[inline]
    pub fn basis(&self) -> (Vector3<f64>, Vector3<f64>) {
        (self.x_basis, self.y_basis)
    }

    /// Precomputed unit axis direction.
    #[inline]
    pub fn axis_unit(&self) -> Vector3<f64> {
        self.axis_unit
    }

    /// Curvature of a cone varies with distance from apex.
    ///
    /// At distance d from the apex:
    /// - Local radius r(d) = d × tan(α)
    /// - κ₁ = cos(α) / r(d) = cos(α) / (d × tan(α)) around the circumference
    /// - κ₂ = 0 along generators
    ///
    /// Near the apex (d → 0), curvature approaches infinity.
    pub fn curvature_at(&self, _theta: f64, d: f64) -> SurfaceCurvature {
        let d_abs = d.abs();
        if d_abs < GEOMETRY_TOL {
            // Near apex: curvature is infinite (singularity)
            SurfaceCurvature::from_principal(f64::INFINITY, 0.0)
        } else {
            let r = d_abs * self.half_angle.tan();
            let kappa_1 = self.half_angle.cos() / r;
            SurfaceCurvature::from_principal(kappa_1, 0.0)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sphere {
    pub center: Point3<f64>,
    pub radius: f64,
}

impl Sphere {
    /// Curvature of a sphere is constant and isotropic: κ₁ = κ₂ = 1/r (umbilical surface).
    pub fn curvature_at(&self, _lon: f64, _lat: f64) -> SurfaceCurvature {
        let k = 1.0 / self.radius;
        SurfaceCurvature::from_principal(k, k)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(into = "TorusSerde")]
pub struct Torus {
    pub center: Point3<f64>,
    pub axis: Vector3<f64>,
    pub major_radius: f64,
    pub minor_radius: f64,
    // Precomputed orthonormal basis perpendicular to axis
    axis_unit: Vector3<f64>,
    x_basis: Vector3<f64>,
    y_basis: Vector3<f64>,
}

#[derive(Serialize, Deserialize)]
struct TorusSerde {
    center: Point3<f64>,
    axis: Vector3<f64>,
    major_radius: f64,
    minor_radius: f64,
}

impl From<TorusSerde> for Torus {
    fn from(s: TorusSerde) -> Self {
        Self::new(s.center, s.axis, s.major_radius, s.minor_radius)
    }
}

impl From<Torus> for TorusSerde {
    fn from(t: Torus) -> Self {
        Self { center: t.center, axis: t.axis, major_radius: t.major_radius, minor_radius: t.minor_radius }
    }
}

impl<'de> Deserialize<'de> for Torus {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        TorusSerde::deserialize(deserializer).map(Into::into)
    }
}

impl Torus {
    pub fn new(center: Point3<f64>, axis: Vector3<f64>, major_radius: f64, minor_radius: f64) -> Self {
        let axis_unit = axis.normalize();
        let x_basis = perpendicular(&axis_unit).normalize();
        let y_basis = axis_unit.cross(&x_basis);
        Self { center, axis, major_radius, minor_radius, axis_unit, x_basis, y_basis }
    }

    /// Precomputed orthonormal basis perpendicular to axis.
    #[inline]
    pub fn basis(&self) -> (Vector3<f64>, Vector3<f64>) {
        (self.x_basis, self.y_basis)
    }

    /// Precomputed unit axis direction.
    #[inline]
    pub fn axis_unit(&self) -> Vector3<f64> {
        self.axis_unit
    }

    /// Curvature of a torus varies around the tube.
    ///
    /// At tube angle φ (minor angle):
    /// - κ₁ = 1 / r_minor (constant around the tube)
    /// - κ₂ = cos(φ) / (R + r_minor × cos(φ))
    ///
    /// Where R = major_radius, r = minor_radius.
    /// - At outer equator (φ = 0): κ₂ = 1/(R+r), both curvatures positive (elliptic)
    /// - At inner equator (φ = π): κ₂ = -1/(R-r), negative (hyperbolic/saddle)
    /// - At top/bottom (φ = ±π/2): κ₂ = 0 (parabolic)
    pub fn curvature_at(&self, _major: f64, minor: f64) -> SurfaceCurvature {
        let r = self.minor_radius;
        let big_r = self.major_radius;

        let kappa_1 = 1.0 / r;

        let cos_phi = minor.cos();
        let denom = big_r + r * cos_phi;

        // Avoid division by zero (would happen if R = r and φ = π, i.e., spindle torus)
        let kappa_2 = if denom.abs() < GEOMETRY_TOL {
            // At the singular inner edge of a spindle torus - curvature is infinite
            // Sign follows from the limit: cos(φ)/denom → -∞ as denom → 0⁺ when cos(φ) < 0
            if cos_phi < 0.0 {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        } else {
            cos_phi / denom
        };

        SurfaceCurvature::from_principal(kappa_1, kappa_2)
    }
}

/// A B-spline surface in 3D space (optionally rational / NURBS).
/// Uses De Boor's algorithm for evaluation (Algorithm A3.5 from "The NURBS Book").
/// When `weights` is `Some`, the surface is rational (NURBS) and the
/// evaluation uses the weighted formula:
///   S(u,v) = Σ(w_ij * N_i(u) * M_j(v) * P_ij) / Σ(w_ij * N_i(u) * M_j(v))
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceBSpline {
    /// Polynomial degree in U direction
    pub u_degree: usize,
    /// Polynomial degree in V direction
    pub v_degree: usize,
    /// Control points grid [u_index][v_index]
    /// Grid dimensions: (n+1) x (m+1) where n = len(u_knots) - u_degree - 2
    pub control_points: Vec<Vec<Point3<f64>>>,
    /// Knot vector in U direction (length = control_points.len() + u_degree + 1)
    pub u_knots: Vec<f64>,
    /// Knot vector in V direction (length = control_points[0].len() + v_degree + 1)
    pub v_knots: Vec<f64>,
    /// Optional weights grid for rational B-spline (NURBS).
    /// When present, dimensions must match control_points.
    pub weights: Option<Vec<Vec<f64>>>,
}

impl SurfaceBSpline {
    /// Create a B-spline surface from STEP-style multiplicities.
    /// `weights` is `Some` for rational B-spline surfaces (NURBS).
    #[allow(clippy::too_many_arguments)]
    pub fn from_multiplicities(
        u_degree: usize,
        v_degree: usize,
        control_points: Vec<Vec<Point3<f64>>>,
        u_knot_values: &[f64],
        u_multiplicities: &[usize],
        v_knot_values: &[f64],
        v_multiplicities: &[usize],
        weights: Option<Vec<Vec<f64>>>,
    ) -> Self {
        let expand_knots = |values: &[f64], mults: &[usize]| -> Vec<f64> {
            values
                .iter()
                .zip(mults.iter())
                .flat_map(|(&k, &m)| std::iter::repeat_n(k, m))
                .collect()
        };

        Self {
            u_degree,
            v_degree,
            control_points,
            u_knots: expand_knots(u_knot_values, u_multiplicities),
            v_knots: expand_knots(v_knot_values, v_multiplicities),
            weights,
        }
    }

    /// Get the valid parameter domain [(u_min, u_max), (v_min, v_max)].
    pub fn domain(&self) -> ((f64, f64), (f64, f64)) {
        let p = self.u_degree;
        let q = self.v_degree;
        let u_domain = (self.u_knots[p], self.u_knots[self.u_knots.len() - 1 - p]);
        let v_domain = (self.v_knots[q], self.v_knots[self.v_knots.len() - 1 - q]);
        (u_domain, v_domain)
    }

    /// Find the knot span index for parameter u in the U direction.
    fn find_u_span(&self, u: f64) -> usize {
        Self::find_span(u, self.u_degree, &self.u_knots, self.control_points.len())
    }

    /// Find the knot span index for parameter v in the V direction.
    fn find_v_span(&self, v: f64) -> usize {
        Self::find_span(
            v,
            self.v_degree,
            &self.v_knots,
            self.control_points[0].len(),
        )
    }

    /// Find knot span index (Algorithm A2.1 from "The NURBS Book").
    fn find_span(u: f64, degree: usize, knots: &[f64], n_control: usize) -> usize {
        let n = n_control - 1;
        let p = degree;

        if u >= knots[n + 1] {
            return n;
        }
        if u <= knots[p] {
            return p;
        }

        let mut low = p;
        let mut high = n + 1;
        let mut mid = usize::midpoint(low, high);

        while u < knots[mid] || u >= knots[mid + 1] {
            if u < knots[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = usize::midpoint(low, high);
        }
        mid
    }

    /// Compute the non-vanishing basis functions (Algorithm A2.2).
    ///
    /// Note on handling zero denominators: When the denominator (right[r+1] + left[j-r])
    /// is zero, this means we have a repeated knot causing 0/0. According to the NURBS
    /// Book convention, 0/0 = 0 in this context, which maintains the partition of unity
    /// property. We handle this by setting temp = 0 when denom is small.
    fn basis_funs(span: usize, u: f64, degree: usize, knots: &[f64]) -> Vec<f64> {
        let p = degree;
        let mut n = vec![0.0; p + 1];
        let mut left = vec![0.0; p + 1];
        let mut right = vec![0.0; p + 1];

        n[0] = 1.0;
        for j in 1..=p {
            left[j] = u - knots[span + 1 - j];
            right[j] = knots[span + j] - u;
            let mut saved = 0.0;
            for r in 0..j {
                let denom = right[r + 1] + left[j - r];
                // Handle 0/0 case: by NURBS Book convention, treat as 0
                // This maintains partition of unity when there are repeated knots
                let temp = if denom.abs() < KNOT_TOL {
                    0.0
                } else {
                    n[r] / denom
                };
                n[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            n[j] = saved;
        }
        n
    }

    /// Evaluate the surface at parameters (u, v).
    /// Algorithm A3.5 from "The NURBS Book" (extended for rational/NURBS).
    #[allow(clippy::needless_range_loop)]
    pub fn evaluate(&self, u: f64, v: f64) -> Point3<f64> {
        let uspan = self.find_u_span(u);
        let vspan = self.find_v_span(v);

        let nu = Self::basis_funs(uspan, u, self.u_degree, &self.u_knots);
        let nv = Self::basis_funs(vspan, v, self.v_degree, &self.v_knots);

        let p = self.u_degree;
        let q = self.v_degree;
        let uind = uspan - p;

        if let Some(ref weights) = self.weights {
            let mut numerator = Vector3::zeros();
            let mut denominator = 0.0;
            for l in 0..=q {
                let vind = vspan - q + l;
                for k in 0..=p {
                    let wn = weights[uind + k][vind] * nu[k] * nv[l];
                    numerator += wn * self.control_points[uind + k][vind].coords;
                    denominator += wn;
                }
            }
            Point3::from(numerator / denominator)
        } else {
            let mut s = Point3::origin();
            for l in 0..=q {
                let mut temp = Vector3::zeros();
                let vind = vspan - q + l;
                for k in 0..=p {
                    temp += nu[k] * self.control_points[uind + k][vind].coords;
                }
                s.coords += nv[l] * temp;
            }
            s
        }
    }

    /// Compute the partial derivatives at (u, v).
    /// Returns (dS/du, dS/dv) - the first partial derivatives.
    /// For rational (NURBS) surfaces, uses the quotient rule.
    #[allow(clippy::needless_range_loop)]
    pub fn derivatives(&self, u: f64, v: f64) -> (Vector3<f64>, Vector3<f64>) {
        let uspan = self.find_u_span(u);
        let vspan = self.find_v_span(v);

        let nu = Self::basis_funs(uspan, u, self.u_degree, &self.u_knots);
        let nv = Self::basis_funs(vspan, v, self.v_degree, &self.v_knots);

        // Compute derivative basis functions
        let nu_der = self.basis_funs_ders(uspan, u, self.u_degree, &self.u_knots, 1);
        let nv_der = self.basis_funs_ders(vspan, v, self.v_degree, &self.v_knots, 1);

        let p = self.u_degree;
        let q = self.v_degree;
        let uind = uspan - p;

        if let Some(ref weights) = self.weights {
            // For rational surfaces, we need: A(u,v) = Σ w_ij N_i M_j P_ij
            //                                  w(u,v) = Σ w_ij N_i M_j
            // S = A/w, dS/du = (dA/du - dw/du * S) / w
            let mut a = Vector3::zeros();
            let mut a_du = Vector3::zeros();
            let mut a_dv = Vector3::zeros();
            let mut w = 0.0;
            let mut w_du = 0.0;
            let mut w_dv = 0.0;

            for l in 0..=q {
                let vind = vspan - q + l;
                for k in 0..=p {
                    let wt = weights[uind + k][vind];
                    let pt = self.control_points[uind + k][vind].coords;
                    let wpt = wt * pt;

                    let n0m0 = nu[k] * nv[l];
                    let n1m0 = nu_der[1][k] * nv[l];
                    let n0m1 = nu[k] * nv_der[1][l];

                    a += n0m0 * wpt;
                    a_du += n1m0 * wpt;
                    a_dv += n0m1 * wpt;

                    w += wt * n0m0;
                    w_du += wt * n1m0;
                    w_dv += wt * n0m1;
                }
            }

            let s = a / w;
            let du = (a_du - w_du * s) / w;
            let dv = (a_dv - w_dv * s) / w;
            (du, dv)
        } else {
            // Compute dS/du
            let mut du = Vector3::zeros();
            for l in 0..=q {
                let mut temp = Vector3::zeros();
                let vind = vspan - q + l;
                for k in 0..=p {
                    temp += nu_der[1][k] * self.control_points[uind + k][vind].coords;
                }
                du += nv[l] * temp;
            }

            // Compute dS/dv
            let mut dv = Vector3::zeros();
            for l in 0..=q {
                let mut temp = Vector3::zeros();
                let vind = vspan - q + l;
                for k in 0..=p {
                    temp += nu[k] * self.control_points[uind + k][vind].coords;
                }
                dv += nv_der[1][l] * temp;
            }

            (du, dv)
        }
    }

    /// Compute basis function derivatives (Algorithm A2.3).
    #[allow(
        clippy::unused_self,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::needless_range_loop
    )]
    fn basis_funs_ders(
        &self,
        span: usize,
        u: f64,
        degree: usize,
        knots: &[f64],
        n_ders: usize,
    ) -> Vec<Vec<f64>> {
        let p = degree;
        let mut ders = vec![vec![0.0; p + 1]; n_ders + 1];
        let mut ndu = vec![vec![0.0; p + 1]; p + 1];
        let mut left = vec![0.0; p + 1];
        let mut right = vec![0.0; p + 1];

        ndu[0][0] = 1.0;
        for j in 1..=p {
            left[j] = u - knots[span + 1 - j];
            right[j] = knots[span + j] - u;
            let mut saved = 0.0;
            for r in 0..j {
                ndu[j][r] = right[r + 1] + left[j - r];
                let temp = ndu[r][j - 1] / ndu[j][r];
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }

        // Load basis functions
        for j in 0..=p {
            ders[0][j] = ndu[j][p];
        }

        // Compute derivatives
        let mut a = vec![vec![0.0; p + 1]; 2];
        for r in 0..=p {
            let mut s1 = 0;
            let mut s2 = 1;
            a[0][0] = 1.0;

            for k in 1..=n_ders.min(p) {
                let mut d = 0.0;
                let rk = r as i32 - k as i32;
                let pk = (p as i32 - k as i32) as usize;

                if rk >= 0 {
                    a[s2][0] = a[s1][0] / ndu[pk + 1][rk as usize];
                    d = a[s2][0] * ndu[rk as usize][pk];
                }

                let j1 = if rk >= -1 { 1 } else { (-rk) as usize };
                let j2 = if (r as i32 - 1) <= pk as i32 {
                    k - 1
                } else {
                    p - r
                };

                for j in j1..=j2 {
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][(rk + j as i32) as usize];
                    d += a[s2][j] * ndu[(rk + j as i32) as usize][pk];
                }

                if r <= pk {
                    a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                    d += a[s2][k] * ndu[r][pk];
                }

                ders[k][r] = d;
                std::mem::swap(&mut s1, &mut s2);
            }
        }

        // Multiply by factorial terms
        let mut r = p as f64;
        for k in 1..=n_ders.min(p) {
            for j in 0..=p {
                ders[k][j] *= r;
            }
            r *= (p - k) as f64;
        }

        ders
    }

    /// Find the (u, v) parameters for a 3D point on the surface using Newton-Raphson.
    /// Returns the parameter pair that produces the closest point on the surface.
    pub fn parameter_at(&self, point: &Point3<f64>) -> Point2<f64> {
        let ((u_min, u_max), (v_min, v_max)) = self.domain();

        // Initial guess: start at center of domain
        let mut u = f64::midpoint(u_min, u_max);
        let mut v = f64::midpoint(v_min, v_max);

        // Try a 3×3 grid of initial guesses to find the best starting point
        // (reduced from 4×4 = 16 evaluations to 9 evaluations)
        let mut best_dist = f64::MAX;
        for ui in 0..3 {
            for vi in 0..3 {
                let test_u = u_min + (u_max - u_min) * (f64::from(ui) + 0.5) / 3.0;
                let test_v = v_min + (v_max - v_min) * (f64::from(vi) + 0.5) / 3.0;
                let test_p = self.evaluate(test_u, test_v);
                let dist = (test_p - point).norm_squared();
                if dist < best_dist {
                    best_dist = dist;
                    u = test_u;
                    v = test_v;
                }
            }
        }

        // Newton-Raphson iteration
        const MAX_ITER: usize = 20;
        const TOL: f64 = 1e-10;

        for _ in 0..MAX_ITER {
            let s = self.evaluate(u, v);
            let (su, sv) = self.derivatives(u, v);

            let delta = s - point;

            // Check convergence
            if delta.norm_squared() < TOL * TOL {
                break;
            }

            // Solve 2x2 system: [su·su  su·sv] [du]   [delta·su]
            //                   [su·sv  sv·sv] [dv] = [delta·sv]
            let a11 = su.dot(&su);
            let a12 = su.dot(&sv);
            let a22 = sv.dot(&sv);
            let b1 = delta.dot(&su);
            let b2 = delta.dot(&sv);

            let det = a11 * a22 - a12 * a12;
            if det.abs() < METRIC_TOL {
                break;
            }

            let du = (a22 * b1 - a12 * b2) / det;
            let dv = (a11 * b2 - a12 * b1) / det;

            u -= du;
            v -= dv;

            // Clamp to domain
            u = u.clamp(u_min, u_max);
            v = v.clamp(v_min, v_max);
        }

        Point2::new(u, v)
    }

    /// Compute the surface normal at (u, v).
    ///
    /// If the cross product of the partial derivatives is degenerate (zero-length),
    /// this function attempts to compute a fallback normal by sampling nearby points.
    pub fn normal_at(&self, u: f64, v: f64) -> Vector3<f64> {
        let (su, sv) = self.derivatives(u, v);
        let n = su.cross(&sv);
        let len = n.norm();

        if len < GEOMETRY_TOL {
            // Degenerate normal - try fallback by sampling nearby points
            self.fallback_normal(u, v)
        } else {
            n / len
        }
    }

    /// Compute a fallback normal for degenerate points (e.g., poles, singularities).
    ///
    /// Samples a small offset in parameter space to find a non-degenerate normal.
    fn fallback_normal(&self, u: f64, v: f64) -> Vector3<f64> {
        let ((u_min, u_max), (v_min, v_max)) = self.domain();
        let u_span = u_max - u_min;
        let v_span = v_max - v_min;

        // Try small offsets in various directions (relative to domain size)
        let eps_u = u_span * 1e-6;
        let eps_v = v_span * 1e-6;

        let offsets = [
            (eps_u, 0.0),
            (-eps_u, 0.0),
            (0.0, eps_v),
            (0.0, -eps_v),
            (eps_u, eps_v),
            (-eps_u, -eps_v),
        ];

        for (du, dv) in offsets {
            let test_u = (u + du).clamp(u_min, u_max);
            let test_v = (v + dv).clamp(v_min, v_max);

            let (su, sv) = self.derivatives(test_u, test_v);
            let n = su.cross(&sv);
            let len = n.norm();

            if len > GEOMETRY_TOL {
                return n / len;
            }
        }

        // Last resort: return Z axis (shouldn't normally reach here)
        Vector3::z()
    }

    /// Compute partial derivatives up to order d at (u, v).
    ///
    /// Returns a 2D array SKL where SKL[k][l] = ∂^(k+l)S / ∂u^k ∂v^l.
    /// Algorithm A3.6 from "The NURBS Book".
    #[allow(clippy::needless_range_loop)]
    pub fn derivatives_order(&self, u: f64, v: f64, d: usize) -> Vec<Vec<Vector3<f64>>> {
        let p = self.u_degree;
        let q = self.v_degree;

        let du = d.min(p);
        let dv = d.min(q);

        // Initialize result array
        let mut skl = vec![vec![Vector3::zeros(); d + 1]; d + 1];

        let uspan = self.find_u_span(u);
        let vspan = self.find_v_span(v);

        // Compute basis function derivatives
        let nu_ders = self.basis_funs_ders(uspan, u, p, &self.u_knots, du);
        let nv_ders = self.basis_funs_ders(vspan, v, q, &self.v_knots, dv);

        let uind = uspan - p;

        // Compute surface point and derivatives
        for k in 0..=du {
            let mut temp = vec![Vector3::zeros(); q + 1];
            for s in 0..=q {
                let vind = vspan - q + s;
                for r in 0..=p {
                    temp[s] += nu_ders[k][r] * self.control_points[uind + r][vind].coords;
                }
            }

            let dd = (d - k).min(dv);
            for l in 0..=dd {
                for s in 0..=q {
                    skl[k][l] += nv_ders[l][s] * temp[s];
                }
            }
        }

        skl
    }

    /// Compute principal curvatures at (u, v) using the first and second fundamental forms.
    ///
    /// First fundamental form coefficients: E = S_u · S_u, F = S_u · S_v, G = S_v · S_v
    /// Second fundamental form coefficients: L = S_uu · n, M = S_uv · n, N = S_vv · n
    ///
    /// Gaussian curvature: K = (LN - M²) / (EG - F²)
    /// Mean curvature: H = (EN - 2FM + GL) / (2(EG - F²))
    /// Principal curvatures: κ₁,₂ = H ± sqrt(H² - K)
    pub fn curvature_at(&self, u: f64, v: f64) -> SurfaceCurvature {
        let d = self.derivatives_order(u, v, 2);

        let s_u = d[1][0];
        let s_v = d[0][1];
        let s_uu = d[2][0];
        let s_uv = d[1][1];
        let s_vv = d[0][2];

        // Surface normal
        let n = s_u.cross(&s_v);
        let n_len = n.norm();
        if n_len < METRIC_TOL {
            // Degenerate surface normal (e.g., at a pole)
            return SurfaceCurvature::zero();
        }
        let n = n / n_len;

        // First fundamental form coefficients
        let e = s_u.dot(&s_u);
        let f = s_u.dot(&s_v);
        let g = s_v.dot(&s_v);

        // Second fundamental form coefficients
        let l = s_uu.dot(&n);
        let m = s_uv.dot(&n);
        let nn = s_vv.dot(&n); // Using 'nn' to avoid shadowing the normal

        // Determinant of first fundamental form
        let eg_f2 = e * g - f * f;
        if eg_f2.abs() < METRIC_TOL {
            // Degenerate metric
            return SurfaceCurvature::zero();
        }

        // Gaussian curvature: K = (LN - M²) / (EG - F²)
        let gaussian = (l * nn - m * m) / eg_f2;

        // Mean curvature: H = (EN - 2FM + GL) / (2(EG - F²))
        let mean = (e * nn - 2.0 * f * m + g * l) / (2.0 * eg_f2);

        // Principal curvatures: κ = H ± sqrt(H² - K)
        let discriminant = mean * mean - gaussian;
        let sqrt_disc = if discriminant > 0.0 {
            discriminant.sqrt()
        } else {
            0.0 // Near umbilical point
        };

        let kappa_1 = mean + sqrt_disc;
        let kappa_2 = mean - sqrt_disc;

        SurfaceCurvature {
            kappa_1,
            kappa_2,
            gaussian,
            mean,
        }
    }
}

impl Surface {
    /// Whether this surface uses angular parametrization that can have ±π discontinuities.
    ///
    /// For these surfaces, midpoint computation should be done in 3D to avoid
    /// issues when the UV midpoint crosses the angular discontinuity.
    pub fn is_angular(&self) -> bool {
        matches!(
            self,
            Surface::Cylinder(_) | Surface::Cone(_) | Surface::Sphere(_) | Surface::Torus(_)
        )
    }

    /// Whether both parametric coordinates are angular (can have ±π discontinuities).
    ///
    /// Torus has (major_angle, minor_angle) — both are atan2 outputs in [-π, π].
    /// Other angular surfaces only have one angular coordinate (u/theta).
    pub fn is_doubly_angular(&self) -> bool {
        matches!(self, Surface::Torus(_))
    }

    /// Whether this is a planar surface (no subdivision needed).
    pub fn is_planar(&self) -> bool {
        matches!(self, Surface::Plane(_))
    }

    /// Estimate maximum curvature over a UV region by sampling a 3×3 grid.
    ///
    /// This catches interior extrema that corner+center sampling would miss.
    pub fn estimate_max_curvature(&self, u_min: f64, u_max: f64, v_min: f64, v_max: f64) -> f64 {
        let mut max_kappa = 0.0f64;
        for i in 0..3 {
            let u = u_min + (u_max - u_min) * (f64::from(i) / 2.0);
            for j in 0..3 {
                let v = v_min + (v_max - v_min) * (f64::from(j) / 2.0);
                let kappa = self.curvature_at(u, v).kappa_max();
                // Handle infinite curvature (singularities) - use a large finite value
                // for tessellation purposes to ensure adequate refinement
                let kappa_finite = if kappa.is_finite() { kappa } else { 1e6 };
                max_kappa = max_kappa.max(kappa_finite);
            }
        }
        max_kappa
    }

    /// Generate interior UV samples for tessellation.
    ///
    /// Uses curvature-based density (chord error formula) with a minimum grid
    /// for robustness. Returns empty for planar surfaces.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn generate_interior_samples(
        &self,
        u_min: f64,
        u_max: f64,
        v_min: f64,
        v_max: f64,
        tolerance: f64,
    ) -> Vec<Point2<f64>> {
        // Planar surfaces don't need interior points
        if self.is_planar() {
            return Vec::new();
        }

        let u_span = u_max - u_min;
        let v_span = v_max - v_min;

        // Grid size bounds
        const N_MIN: usize = 1; // Curvature drives density; Phase 3 fills in as needed
        const N_MAX: usize = 8; // Maximum to avoid excessive triangles

        // Additional density from curvature for quality
        let max_kappa = self.estimate_max_curvature(u_min, u_max, v_min, v_max);
        let (n_u_curv, n_v_curv) = if max_kappa > 1e-12 {
            let step = (8.0 * tolerance / max_kappa).sqrt();
            (
                (u_span / step).ceil() as usize,
                (v_span / step).ceil() as usize,
            )
        } else {
            (0, 0)
        };

        // N_MIN=1 lets curvature drive the grid — near-flat curved surfaces
        // get fewer interior points, with Phase 3 refinement filling in as needed.
        let n_u = N_MIN.max(n_u_curv).min(N_MAX);
        let n_v = N_MIN.max(n_v_curv).min(N_MAX);

        // Generate interior grid (not on boundary edges)
        let mut samples = Vec::with_capacity(n_u * n_v);
        for i in 1..=n_u {
            for j in 1..=n_v {
                let u = u_min + u_span * i as f64 / (n_u + 1) as f64;
                let v = v_min + v_span * j as f64 / (n_v + 1) as f64;
                samples.push(Point2::new(u, v));
            }
        }
        samples
    }

    /// If this surface projects as a circle onto the given plane,
    /// return (center_2d, radius).
    ///
    /// Currently handles cylinders whose axis is approximately parallel to the
    /// projection plane normal (covers drill holes, bores, fillets, rounds).
    pub fn project_as_circle(&self, plane: &Plane) -> Option<(Point2<f64>, f64)> {
        match self {
            Surface::Cylinder(c) => {
                let axis_norm = c.axis.normalize();
                // Cylinder axis must be ~parallel to the plane normal
                const AXIS_PARALLEL_TOL: f64 = 1e-6;
                if axis_norm.dot(&plane.normal).abs() > 1.0 - AXIS_PARALLEL_TOL {
                    let center_2d = plane.to_2d(&[c.origin]);
                    Some((center_2d[0], c.radius))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Human-readable kind name for grouping labels.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Surface::Plane(_) => "Plane",
            Surface::Cylinder(_) => "Cylinder",
            Surface::Cone(_) => "Cone",
            Surface::Sphere(_) => "Sphere",
            Surface::Torus(_) => "Torus",
            Surface::BSpline(_) => "BSpline",
        }
    }

    /// Compute the principal curvatures at parametric coordinates (u, v).
    pub fn curvature_at(&self, u: f64, v: f64) -> SurfaceCurvature {
        match self {
            Surface::Plane(p) => p.curvature_at(u, v),
            Surface::Cylinder(c) => c.curvature_at(u, v),
            Surface::Cone(c) => c.curvature_at(u, v),
            Surface::Sphere(s) => s.curvature_at(u, v),
            Surface::Torus(t) => t.curvature_at(u, v),
            Surface::BSpline(b) => b.curvature_at(u, v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Vector3};

    #[test]
    fn test_cylinder_project_as_circle_aligned() {
        let cyl = Surface::Cylinder(Cylinder::new(
            Point3::new(1.0, 2.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            5.0,
        ));
        let plane = Plane::new(Vector3::new(0.0, 0.0, 1.0), Point3::origin());
        let (center, radius) = cyl.project_as_circle(&plane).unwrap();
        assert_relative_eq!(radius, 5.0, epsilon = 1e-10);
        assert_relative_eq!(center.x, 1.0, epsilon = 1e-6);
        assert_relative_eq!(center.y, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cylinder_project_as_circle_misaligned() {
        let cyl = Surface::Cylinder(Cylinder::new(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            5.0,
        ));
        let plane = Plane::new(Vector3::new(0.0, 0.0, 1.0), Point3::origin());
        assert!(cyl.project_as_circle(&plane).is_none());
    }

    #[test]
    fn test_plane_surface_no_circle() {
        let prim = Surface::Plane(SurfacePlane {
            origin: Point3::origin(),
            normal: Vector3::z(),
        });
        let plane = Plane::new(Vector3::z(), Point3::origin());
        assert!(prim.project_as_circle(&plane).is_none());
    }

    #[test]
    fn test_serde_roundtrip() {
        let cyl = Surface::Cylinder(Cylinder::new(
            Point3::new(1.0, 2.0, 3.0),
            Vector3::new(0.0, 0.0, 1.0),
            5.0,
        ));
        let json = serde_json::to_string(&cyl).unwrap();
        let deserialized: Surface = serde_json::from_str(&json).unwrap();
        assert_eq!(cyl, deserialized);
    }

    #[test]
    fn test_kind_name() {
        assert_eq!(
            Surface::Plane(SurfacePlane {
                origin: Point3::origin(),
                normal: Vector3::z(),
            })
            .kind_name(),
            "Plane"
        );
        assert_eq!(
            Surface::Cylinder(Cylinder::new(
                Point3::origin(),
                Vector3::z(),
                1.0,
            ))
            .kind_name(),
            "Cylinder"
        );
        assert_eq!(
            Surface::Cone(Cone::new(
                Point3::origin(),
                Vector3::z(),
                0.5,
            ))
            .kind_name(),
            "Cone"
        );
        assert_eq!(
            Surface::Sphere(Sphere {
                center: Point3::origin(),
                radius: 1.0,
            })
            .kind_name(),
            "Sphere"
        );
        assert_eq!(
            Surface::Torus(Torus::new(
                Point3::origin(),
                Vector3::z(),
                2.0,
                0.5,
            ))
            .kind_name(),
            "Torus"
        );
        assert_eq!(
            Surface::BSpline(SurfaceBSpline {
                u_degree: 1,
                v_degree: 1,
                control_points: vec![
                    vec![Point3::origin(), Point3::new(0.0, 1.0, 0.0)],
                    vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
                ],
                u_knots: vec![0.0, 0.0, 1.0, 1.0],
                v_knots: vec![0.0, 0.0, 1.0, 1.0],
                weights: None,
            })
            .kind_name(),
            "BSpline"
        );
    }

    #[test]
    fn test_bspline_surface_bilinear() {
        // A bilinear patch (degree 1 in both directions) is a simple plane segment
        // Control points form a unit square in the XY plane:
        //  (0,1) --- (1,1)
        //    |         |
        //  (0,0) --- (1,0)
        let surface = SurfaceBSpline {
            u_degree: 1,
            v_degree: 1,
            control_points: vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            u_knots: vec![0.0, 0.0, 1.0, 1.0],
            v_knots: vec![0.0, 0.0, 1.0, 1.0],
            weights: None,
        };

        // Test corner points
        let p00 = surface.evaluate(0.0, 0.0);
        assert_relative_eq!(p00.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(p00.y, 0.0, epsilon = 1e-10);

        let p10 = surface.evaluate(1.0, 0.0);
        assert_relative_eq!(p10.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(p10.y, 0.0, epsilon = 1e-10);

        let p01 = surface.evaluate(0.0, 1.0);
        assert_relative_eq!(p01.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(p01.y, 1.0, epsilon = 1e-10);

        let p11 = surface.evaluate(1.0, 1.0);
        assert_relative_eq!(p11.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(p11.y, 1.0, epsilon = 1e-10);

        // Test center point
        let center = surface.evaluate(0.5, 0.5);
        assert_relative_eq!(center.x, 0.5, epsilon = 1e-10);
        assert_relative_eq!(center.y, 0.5, epsilon = 1e-10);
        assert_relative_eq!(center.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bspline_surface_parameter_at() {
        // Create a bilinear surface
        let surface = SurfaceBSpline {
            u_degree: 1,
            v_degree: 1,
            control_points: vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            u_knots: vec![0.0, 0.0, 1.0, 1.0],
            v_knots: vec![0.0, 0.0, 1.0, 1.0],
            weights: None,
        };

        // Test that parameter_at inverts evaluate
        for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let point = surface.evaluate(u, v);
                let uv = surface.parameter_at(&point);
                assert_relative_eq!(uv.x, u, epsilon = 1e-6);
                assert_relative_eq!(uv.y, v, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_bspline_surface_normal() {
        // Bilinear surface in XY plane should have normal pointing in Z
        let surface = SurfaceBSpline {
            u_degree: 1,
            v_degree: 1,
            control_points: vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            u_knots: vec![0.0, 0.0, 1.0, 1.0],
            v_knots: vec![0.0, 0.0, 1.0, 1.0],
            weights: None,
        };

        let normal = surface.normal_at(0.5, 0.5);
        assert_relative_eq!(normal.z.abs(), 1.0, epsilon = 1e-10);
    }

    // =========================================================================
    // Curvature computation tests
    // =========================================================================

    #[test]
    fn test_curvature_plane_zero() {
        // A plane has zero curvature everywhere
        let plane = SurfacePlane {
            origin: Point3::origin(),
            normal: Vector3::z(),
        };

        let curvature = plane.curvature_at(0.0, 0.0);
        assert_relative_eq!(curvature.kappa_1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(curvature.kappa_2, 0.0, epsilon = 1e-10);
        assert_relative_eq!(curvature.gaussian, 0.0, epsilon = 1e-10);
        assert_relative_eq!(curvature.mean, 0.0, epsilon = 1e-10);
        assert_eq!(curvature.max_edge_length(0.001), f64::INFINITY);
    }

    #[test]
    fn test_curvature_cylinder() {
        // Cylinder with radius r has κ₁ = 1/r, κ₂ = 0
        let radius = 5.0;
        let cyl = Cylinder::new(Point3::origin(), Vector3::z(), radius);

        // Curvature should be constant everywhere on the cylinder
        for theta in [0.0, 1.0, 2.0, 3.0] {
            for h in [-1.0, 0.0, 1.0, 5.0] {
                let curvature = cyl.curvature_at(theta, h);
                assert_relative_eq!(curvature.kappa_1, 1.0 / radius, epsilon = 1e-10);
                assert_relative_eq!(curvature.kappa_2, 0.0, epsilon = 1e-10);
                assert_relative_eq!(curvature.gaussian, 0.0, epsilon = 1e-10);
                assert_relative_eq!(curvature.mean, 1.0 / (2.0 * radius), epsilon = 1e-10);
            }
        }

        // Test max edge length computation
        // For tolerance 0.01 and κ = 0.2, L_max = sqrt(8 * 0.01 / 0.2) ≈ 0.632
        let tolerance = 0.01;
        let curvature = cyl.curvature_at(0.0, 0.0);
        let l_max = curvature.max_edge_length(tolerance);
        let expected = (8.0 * tolerance / (1.0 / radius)).sqrt();
        assert_relative_eq!(l_max, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_curvature_sphere_isotropic() {
        // Sphere with radius r has κ₁ = κ₂ = 1/r (umbilical surface)
        let radius = 3.0;
        let sphere = Sphere {
            center: Point3::origin(),
            radius,
        };

        // Test at various points on the sphere
        for lon in [0.0, 1.0, 2.0] {
            for lat in [-0.5, 0.0, 0.5] {
                let curvature = sphere.curvature_at(lon, lat);
                let expected_k = 1.0 / radius;
                assert_relative_eq!(curvature.kappa_1, expected_k, epsilon = 1e-10);
                assert_relative_eq!(curvature.kappa_2, expected_k, epsilon = 1e-10);
                // Gaussian = κ₁ × κ₂ = 1/r²
                assert_relative_eq!(curvature.gaussian, 1.0 / (radius * radius), epsilon = 1e-10);
                // Mean = (κ₁ + κ₂)/2 = 1/r
                assert_relative_eq!(curvature.mean, expected_k, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_curvature_torus_varies() {
        // Torus: κ₁ = 1/r_minor (constant), κ₂ = cos(φ)/(R + r*cos(φ)) (varies)
        let major_r = 5.0;
        let minor_r = 1.0;
        let torus = Torus::new(Point3::origin(), Vector3::z(), major_r, minor_r);

        // κ₁ should always be 1/r_minor
        let curvature_outer = torus.curvature_at(0.0, 0.0); // outer equator (φ = 0)
        assert_relative_eq!(curvature_outer.kappa_1, 1.0 / minor_r, epsilon = 1e-10);

        // At outer equator (φ = 0): κ₂ = 1/(R+r)
        let expected_k2_outer = 1.0 / (major_r + minor_r);
        assert_relative_eq!(curvature_outer.kappa_2, expected_k2_outer, epsilon = 1e-10);
        // Gaussian positive (elliptic point)
        assert!(curvature_outer.gaussian > 0.0);

        // At inner equator (φ = π): κ₂ = -1/(R-r)
        let curvature_inner = torus.curvature_at(0.0, std::f64::consts::PI);
        let expected_k2_inner = -1.0 / (major_r - minor_r);
        assert_relative_eq!(curvature_inner.kappa_2, expected_k2_inner, epsilon = 1e-10);
        // Gaussian negative (hyperbolic/saddle point)
        assert!(curvature_inner.gaussian < 0.0);

        // At top/bottom (φ = π/2): κ₂ = 0 (parabolic)
        let curvature_top = torus.curvature_at(0.0, std::f64::consts::FRAC_PI_2);
        assert_relative_eq!(curvature_top.kappa_2, 0.0, epsilon = 1e-10);
        assert_relative_eq!(curvature_top.gaussian, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_curvature_cone_varies_with_distance() {
        // Cone: κ₁ = cos(α)/r(d) where r(d) = d*tan(α), κ₂ = 0
        let half_angle = std::f64::consts::FRAC_PI_6; // 30 degrees
        let cone = Cone::new(Point3::origin(), Vector3::z(), half_angle);

        // At distance d = 1: r = tan(30°) ≈ 0.577, κ₁ = cos(30°)/r
        let d = 1.0;
        let curvature = cone.curvature_at(0.0, d);

        let r = d * half_angle.tan();
        let expected_k1 = half_angle.cos() / r;

        assert_relative_eq!(curvature.kappa_1, expected_k1, epsilon = 1e-10);
        assert_relative_eq!(curvature.kappa_2, 0.0, epsilon = 1e-10);

        // Curvature should decrease with distance from apex
        let curvature_far = cone.curvature_at(0.0, 10.0);
        assert!(curvature_far.kappa_max() < curvature.kappa_max());
    }

    #[test]
    fn test_curvature_bspline_bilinear_flat() {
        // A bilinear B-spline (degree 1 in both directions) forming a flat plane
        // should have zero curvature
        let surface = SurfaceBSpline {
            u_degree: 1,
            v_degree: 1,
            control_points: vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            u_knots: vec![0.0, 0.0, 1.0, 1.0],
            v_knots: vec![0.0, 0.0, 1.0, 1.0],
            weights: None,
        };

        // For a bilinear surface, the second derivatives are zero
        // so curvature should be zero
        let curvature = surface.curvature_at(0.5, 0.5);
        assert_relative_eq!(curvature.kappa_max(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_edge_length_formula() {
        // Verify the chord error formula: ε = κL²/8, so L_max = sqrt(8ε/κ)
        let tolerance = 0.001;

        // For a unit cylinder (κ = 1)
        let cyl = Cylinder::new(Point3::origin(), Vector3::z(), 1.0);
        let curvature = cyl.curvature_at(0.0, 0.0);
        let l_max = curvature.max_edge_length(tolerance);

        // L_max = sqrt(8 * 0.001 / 1) ≈ 0.0894
        let expected = (8.0 * tolerance).sqrt();
        assert_relative_eq!(l_max, expected, epsilon = 1e-10);

        // Verify that using this edge length gives the right chord error
        // For an arc of length L, chord error ε = κL²/8
        let chord_error = curvature.kappa_max() * l_max * l_max / 8.0;
        assert_relative_eq!(chord_error, tolerance, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_enum_curvature_dispatch() {
        // Test that Surface enum dispatches to correct surface type
        let surfaces: Vec<Surface> = vec![
            Surface::Plane(SurfacePlane {
                origin: Point3::origin(),
                normal: Vector3::z(),
            }),
            Surface::Cylinder(Cylinder::new(Point3::origin(), Vector3::z(), 2.0)),
            Surface::Sphere(Sphere {
                center: Point3::origin(),
                radius: 3.0,
            }),
        ];

        let curvatures: Vec<f64> = surfaces
            .iter()
            .map(|s| s.curvature_at(0.0, 0.0).kappa_max())
            .collect();

        // Plane: κ = 0
        assert_relative_eq!(curvatures[0], 0.0, epsilon = 1e-10);
        // Cylinder r=2: κ = 0.5
        assert_relative_eq!(curvatures[1], 0.5, epsilon = 1e-10);
        // Sphere r=3: κ = 1/3
        assert_relative_eq!(curvatures[2], 1.0 / 3.0, epsilon = 1e-10);
    }
}
