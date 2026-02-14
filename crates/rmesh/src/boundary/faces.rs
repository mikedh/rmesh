//! Analytical surface types from BREP faces (ISO 10303-42 elementary surfaces).
//!
//! When a STEP file is loaded via cascadio, each triangle can be tagged with the
//! BREP face it came from. These types capture the analytical surface definition
//! so that projection and buffer operations can preserve exact geometry (e.g.
//! cylinders project as circles, arcs from fillets are preserved through offset).

use nalgebra::{Point2, Point3, Vector3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use super::bspline::SurfaceBSpline;
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

/// Compute an orthonormal basis from an axis direction.
///
/// Returns `(axis_unit, x_basis, y_basis)` where `axis_unit` is the normalized
/// axis, and `x_basis`/`y_basis` form an orthonormal frame perpendicular to it.
/// Used by [`Cylinder`], [`Cone`], and [`Torus`] constructors.
fn compute_axis_basis(axis: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let axis_unit = axis.normalize();
    let x_basis = perpendicular(&axis_unit);
    let y_basis = axis_unit.cross(&x_basis);
    (axis_unit, x_basis, y_basis)
}

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
    Offset(Box<OffsetSurface>),
}

/// A surface offset from a base surface by a constant distance along its normal.
///
/// `evaluate(u,v) = base.evaluate(u,v) + distance * base.normal_at(u,v)`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OffsetSurface {
    pub base: Surface,
    pub distance: f64,
}

impl OffsetSurface {
    /// Map 3D point to (u, v) parameters via Newton-Raphson starting from base.to_parametric.
    pub fn to_parametric(&self, point: &Point3<f64>) -> Point2<f64> {
        // Start from the base surface's parameter for the query point
        let uv0 = self.base.to_parametric(point);
        let mut u = uv0.x;
        let mut v = uv0.y;

        // Newton-Raphson refinement
        for _ in 0..10 {
            let s = self.evaluate(u, v);
            let delta = s - point;
            if delta.norm_squared() < NEWTON_TOL * NEWTON_TOL {
                break;
            }
            // Use finite differences for derivatives of the offset surface
            let eps = 1e-8;
            let su = (self.evaluate(u + eps, v) - self.evaluate(u - eps, v)) / (2.0 * eps);
            let sv = (self.evaluate(u, v + eps) - self.evaluate(u, v - eps)) / (2.0 * eps);

            let a11 = su.dot(&su);
            let a12 = su.dot(&sv);
            let a22 = sv.dot(&sv);
            let b1 = delta.dot(&su);
            let b2 = delta.dot(&sv);

            let det = a11 * a22 - a12 * a12;
            if det.abs() < METRIC_TOL {
                break;
            }
            u -= (a22 * b1 - a12 * b2) / det;
            v -= (a11 * b2 - a12 * b1) / det;
        }

        Point2::new(u, v)
    }

    /// Evaluate the offset surface at (u, v).
    pub fn evaluate(&self, u: f64, v: f64) -> Point3<f64> {
        let base_pt = self.base.evaluate(u, v);
        let n = self.base.normal_at(u, v);
        base_pt + self.distance * n
    }

    /// Surface normal via finite differences.
    pub fn normal_at(&self, u: f64, v: f64) -> Vector3<f64> {
        let eps = 1e-8;
        let su = self.evaluate(u + eps, v) - self.evaluate(u - eps, v);
        let sv = self.evaluate(u, v + eps) - self.evaluate(u, v - eps);
        let n = su.cross(&sv);
        let len = n.norm();
        if len < GEOMETRY_TOL {
            self.base.normal_at(u, v)
        } else {
            n / len
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(into = "SurfacePlaneSerde")]
pub struct SurfacePlane {
    pub origin: Point3<f64>,
    pub normal: Vector3<f64>,
    // Precomputed orthonormal basis in the plane
    x_basis: Vector3<f64>,
    y_basis: Vector3<f64>,
}

#[derive(Serialize, Deserialize)]
struct SurfacePlaneSerde {
    origin: Point3<f64>,
    normal: Vector3<f64>,
}

impl From<SurfacePlaneSerde> for SurfacePlane {
    fn from(s: SurfacePlaneSerde) -> Self {
        Self::new(s.origin, s.normal)
    }
}

impl From<SurfacePlane> for SurfacePlaneSerde {
    fn from(p: SurfacePlane) -> Self {
        Self {
            origin: p.origin,
            normal: p.normal,
        }
    }
}

impl<'de> Deserialize<'de> for SurfacePlane {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        SurfacePlaneSerde::deserialize(deserializer).map(Into::into)
    }
}

impl SurfacePlane {
    pub fn new(origin: Point3<f64>, normal: Vector3<f64>) -> Self {
        let n = normal.normalize();
        let x_basis = perpendicular(&n).normalize();
        let y_basis = n.cross(&x_basis);
        Self {
            origin,
            normal,
            x_basis,
            y_basis,
        }
    }

    /// Precomputed orthonormal basis in the plane.
    #[inline]
    pub fn basis(&self) -> (Vector3<f64>, Vector3<f64>) {
        (self.x_basis, self.y_basis)
    }

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
        Self {
            origin: c.origin,
            axis: c.axis,
            radius: c.radius,
        }
    }
}

impl<'de> Deserialize<'de> for Cylinder {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        CylinderSerde::deserialize(deserializer).map(Into::into)
    }
}

impl Cylinder {
    pub fn new(origin: Point3<f64>, axis: Vector3<f64>, radius: f64) -> Self {
        let (axis_unit, x_basis, y_basis) = compute_axis_basis(&axis);
        Self {
            origin,
            axis,
            radius,
            axis_unit,
            x_basis,
            y_basis,
        }
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
        Self {
            apex: c.apex,
            axis: c.axis,
            half_angle: c.half_angle,
        }
    }
}

impl<'de> Deserialize<'de> for Cone {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ConeSerde::deserialize(deserializer).map(Into::into)
    }
}

impl Cone {
    pub fn new(apex: Point3<f64>, axis: Vector3<f64>, half_angle: f64) -> Self {
        let (axis_unit, x_basis, y_basis) = compute_axis_basis(&axis);
        Self {
            apex,
            axis,
            half_angle,
            axis_unit,
            x_basis,
            y_basis,
        }
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
        Self {
            center: t.center,
            axis: t.axis,
            major_radius: t.major_radius,
            minor_radius: t.minor_radius,
        }
    }
}

impl<'de> Deserialize<'de> for Torus {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        TorusSerde::deserialize(deserializer).map(Into::into)
    }
}

impl Torus {
    pub fn new(
        center: Point3<f64>,
        axis: Vector3<f64>,
        major_radius: f64,
        minor_radius: f64,
    ) -> Self {
        let (axis_unit, x_basis, y_basis) = compute_axis_basis(&axis);
        Self {
            center,
            axis,
            major_radius,
            minor_radius,
            axis_unit,
            x_basis,
            y_basis,
        }
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

impl Surface {
    /// Whether this surface uses angular parametrization that can have ±π discontinuities.
    ///
    /// For these surfaces, midpoint computation should be done in 3D to avoid
    /// issues when the UV midpoint crosses the angular discontinuity.
    pub fn is_angular(&self) -> bool {
        match self {
            Surface::Cylinder(_) | Surface::Cone(_) | Surface::Sphere(_) | Surface::Torus(_) => {
                true
            }
            Surface::Offset(o) => o.base.is_angular(),
            _ => false,
        }
    }

    /// Whether both parametric coordinates are angular (can have ±π discontinuities).
    ///
    /// Torus has (major_angle, minor_angle) — both are atan2 outputs in [-π, π].
    /// Other angular surfaces only have one angular coordinate (u/theta).
    pub fn is_doubly_angular(&self) -> bool {
        match self {
            Surface::Torus(_) => true,
            Surface::Offset(o) => o.base.is_doubly_angular(),
            _ => false,
        }
    }

    /// Whether this is a planar surface (no subdivision needed).
    pub fn is_planar(&self) -> bool {
        match self {
            Surface::Plane(_) => true,
            Surface::Offset(o) => o.base.is_planar(),
            _ => false,
        }
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

    /// If this surface projects as a circle onto the given plane,
    /// return (center_2d, radius).
    ///
    /// Currently handles cylinders whose axis is approximately parallel to the
    /// projection plane normal (covers drill holes, bores, fillets, rounds).
    pub fn project_as_circle(&self, plane: &Plane) -> Option<(Point2<f64>, f64)> {
        match self {
            Surface::Cylinder(c) => {
                // Cylinder axis must be ~parallel to the plane normal
                const AXIS_PARALLEL_TOL: f64 = 1e-6;
                if c.axis_unit().dot(&plane.normal).abs() > 1.0 - AXIS_PARALLEL_TOL {
                    let center_2d = plane.to_2d(&[c.origin]);
                    Some((center_2d[0], c.radius))
                } else {
                    None
                }
            }
            Surface::Offset(o) => o.base.project_as_circle(plane),
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
            Surface::Offset(_) => "Offset",
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
            Surface::Offset(o) => o.base.curvature_at(u, v),
        }
    }

    /// First fundamental form coefficients (E, F, G) at parameter (u, v).
    /// Used to convert 3D distances to UV distances: ds² = E·du² + 2F·du·dv + G·dv²
    pub fn first_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64) {
        match self {
            // Plane: du and dv map 1:1 to 3D basis vectors
            Surface::Plane(_) => (1.0, 0.0, 1.0),
            // Cylinder: u=theta, v=h. dS/du = r*(-sin,cos,0), dS/dv = (0,0,1)
            // E = r², F = 0, G = 1
            Surface::Cylinder(c) => (c.radius * c.radius, 0.0, 1.0),
            // Sphere: u=lon, v=lat. dS/du = r*cos(lat)*(-sin(lon),cos(lon),0), dS/dv = r*(-sin(lat)*cos(lon),...)
            // E = r²cos²(lat), F = 0, G = r²
            Surface::Sphere(s) => {
                let cos_v = v.cos();
                let r2 = s.radius * s.radius;
                (r2 * cos_v * cos_v, 0.0, r2)
            }
            // Cone: u=theta, v=d (distance from apex). r(d) = d*tan(α)
            // dS/du = r*(-sin,cos,0), dS/dv = axis + tan(α)*(cos,sin,0)
            // E = r² = d²tan²(α), F = 0, G = 1 + tan²(α) = 1/cos²(α)
            Surface::Cone(c) => {
                let tan_a = c.half_angle.tan();
                let r = v.abs() * tan_a;
                let sec2 = 1.0 / (c.half_angle.cos() * c.half_angle.cos());
                (r * r, 0.0, sec2)
            }
            // Torus: u=major, v=minor. R=major_radius, r=minor_radius
            // E = (R + r*cos(v))², F = 0, G = r²
            Surface::Torus(t) => {
                let arm = t.major_radius + t.minor_radius * v.cos();
                (arm * arm, 0.0, t.minor_radius * t.minor_radius)
            }
            // BSpline: compute via finite differences of partial derivatives
            Surface::BSpline(b) => {
                let (s_u, s_v) = b.derivatives(u, v);
                let e = s_u.dot(&s_u);
                let f = s_u.dot(&s_v);
                let g = s_v.dot(&s_v);
                (e, f, g)
            }
            // Offset: delegate to base surface (approximate — offset changes metric slightly)
            Surface::Offset(o) => o.base.first_fundamental_form(u, v),
        }
    }
}

// ============================================================================
// SurfaceDict — canonical serde helper for dict ↔ Surface conversion
// ============================================================================

/// Canonical dict format for `Surface` serialization.
///
/// Used by GLTF extensions and Python bindings as the single source of truth
/// for surface ↔ dict conversion. Tagged by `"kind"` with capitalized variant
/// names; lowercase aliases provide backward compatibility with older GLB files.
///
/// Point/vector fields use `[f64; 3]` rather than `Point3`/`Vector3` because
/// serde serializes nalgebra types as `{"x":…,"y":…,"z":…}`, while the JSON
/// format uses flat arrays `[x, y, z]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum SurfaceDict {
    #[serde(alias = "plane")]
    Plane {
        origin: [f64; 3],
        normal: [f64; 3],
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_dir: Option<[f64; 3]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extent_x: Option<[f64; 2]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extent_y: Option<[f64; 2]>,
    },
    #[serde(alias = "cylinder")]
    Cylinder {
        origin: [f64; 3],
        axis: [f64; 3],
        radius: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extent_angle: Option<[f64; 2]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extent_height: Option<[f64; 2]>,
    },
    #[serde(alias = "cone")]
    Cone {
        #[serde(alias = "semi_angle")]
        half_angle: f64,
        apex: [f64; 3],
        axis: [f64; 3],
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ref_radius: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extent_angle: Option<[f64; 2]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extent_distance: Option<[f64; 2]>,
    },
    #[serde(alias = "sphere")]
    Sphere {
        center: [f64; 3],
        radius: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extent_longitude: Option<[f64; 2]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extent_latitude: Option<[f64; 2]>,
    },
    #[serde(alias = "torus")]
    Torus {
        center: [f64; 3],
        axis: [f64; 3],
        major_radius: f64,
        minor_radius: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extent_major_angle: Option<[f64; 2]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extent_minor_angle: Option<[f64; 2]>,
    },
    /// Placeholder for BSpline faces — state is too large for dict format.
    /// Exists so GLTF deserialization doesn't fail; `TryFrom` returns `Err`.
    #[serde(alias = "bspline")]
    BSpline,
}

impl From<&Surface> for SurfaceDict {
    fn from(surface: &Surface) -> Self {
        match surface {
            Surface::Plane(p) => SurfaceDict::Plane {
                origin: p.origin.into(),
                normal: p.normal.into(),
                x_dir: None,
                extent_x: None,
                extent_y: None,
            },
            Surface::Cylinder(c) => SurfaceDict::Cylinder {
                origin: c.origin.into(),
                axis: c.axis.into(),
                radius: c.radius,
                extent_angle: None,
                extent_height: None,
            },
            Surface::Cone(c) => SurfaceDict::Cone {
                half_angle: c.half_angle,
                apex: c.apex.into(),
                axis: c.axis.into(),
                ref_radius: None,
                extent_angle: None,
                extent_distance: None,
            },
            Surface::Sphere(s) => SurfaceDict::Sphere {
                center: s.center.into(),
                radius: s.radius,
                extent_longitude: None,
                extent_latitude: None,
            },
            Surface::Torus(t) => SurfaceDict::Torus {
                center: t.center.into(),
                axis: t.axis.into(),
                major_radius: t.major_radius,
                minor_radius: t.minor_radius,
                extent_major_angle: None,
                extent_minor_angle: None,
            },
            Surface::BSpline(_) | Surface::Offset(_) => SurfaceDict::BSpline,
        }
    }
}

impl TryFrom<SurfaceDict> for Surface {
    type Error = String;

    fn try_from(dict: SurfaceDict) -> Result<Self, String> {
        Ok(match dict {
            SurfaceDict::Plane { origin, normal, .. } => {
                Surface::Plane(SurfacePlane::new(origin.into(), normal.into()))
            }
            SurfaceDict::Cylinder {
                origin,
                axis,
                radius,
                ..
            } => Surface::Cylinder(Cylinder::new(origin.into(), axis.into(), radius)),
            SurfaceDict::Cone {
                apex,
                axis,
                half_angle,
                ..
            } => Surface::Cone(Cone::new(apex.into(), axis.into(), half_angle)),
            SurfaceDict::Sphere { center, radius, .. } => Surface::Sphere(Sphere {
                center: center.into(),
                radius,
            }),
            SurfaceDict::Torus {
                center,
                axis,
                major_radius,
                minor_radius,
                ..
            } => Surface::Torus(Torus::new(
                center.into(),
                axis.into(),
                major_radius,
                minor_radius,
            )),
            SurfaceDict::BSpline => {
                return Err("BSpline surfaces cannot be round-tripped through dict format".into());
            }
        })
    }
}

impl Surface {
    /// Deserialize a `Surface` from a JSON value in the canonical dict format.
    ///
    /// The dict must have a `"kind"` tag (e.g. `{"kind": "Plane", ...}`).
    /// Lowercase variant names and `"semi_angle"` are accepted for backward compat.
    pub fn from_dict(value: Value) -> Result<Self, String> {
        let dict: SurfaceDict =
            serde_json::from_value(value).map_err(|e| format!("invalid surface dict: {e}"))?;
        dict.try_into()
    }

    /// Serialize this `Surface` to a JSON value in the canonical dict format.
    pub fn to_dict(&self) -> Value {
        let dict = SurfaceDict::from(self);
        serde_json::to_value(dict).expect("SurfaceDict is always serializable")
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
        let prim = Surface::Plane(SurfacePlane::new(Point3::origin(), Vector3::z()));
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
            Surface::Plane(SurfacePlane::new(Point3::origin(), Vector3::z(),)).kind_name(),
            "Plane"
        );
        assert_eq!(
            Surface::Cylinder(Cylinder::new(Point3::origin(), Vector3::z(), 1.0,)).kind_name(),
            "Cylinder"
        );
        assert_eq!(
            Surface::Cone(Cone::new(Point3::origin(), Vector3::z(), 0.5,)).kind_name(),
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
            Surface::Torus(Torus::new(Point3::origin(), Vector3::z(), 2.0, 0.5,)).kind_name(),
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
        let plane = SurfacePlane::new(Point3::origin(), Vector3::z());

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
            Surface::Plane(SurfacePlane::new(Point3::origin(), Vector3::z())),
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

    // =========================================================================
    // SurfaceDict round-trip tests
    // =========================================================================

    #[test]
    fn test_surface_dict_roundtrip_plane() {
        let surface = Surface::Plane(SurfacePlane::new(
            Point3::new(1.0, 2.0, 3.0),
            Vector3::new(0.0, 0.0, 1.0),
        ));
        let dict = surface.to_dict();
        let back = Surface::from_dict(dict).unwrap();
        assert_eq!(surface, back);
    }

    #[test]
    fn test_surface_dict_roundtrip_cylinder() {
        let surface = Surface::Cylinder(Cylinder::new(
            Point3::new(1.0, 2.0, 3.0),
            Vector3::new(0.0, 1.0, 0.0),
            0.005,
        ));
        let dict = surface.to_dict();
        let back = Surface::from_dict(dict).unwrap();
        assert_eq!(surface, back);
    }

    #[test]
    fn test_surface_dict_roundtrip_cone() {
        let surface = Surface::Cone(Cone::new(
            Point3::new(0.0, 0.0, 5.0),
            Vector3::new(0.0, 0.0, 1.0),
            0.3,
        ));
        let dict = surface.to_dict();
        let back = Surface::from_dict(dict).unwrap();
        assert_eq!(surface, back);
    }

    #[test]
    fn test_surface_dict_roundtrip_sphere() {
        let surface = Surface::Sphere(Sphere {
            center: Point3::new(1.0, 2.0, 3.0),
            radius: 1.0,
        });
        let dict = surface.to_dict();
        let back = Surface::from_dict(dict).unwrap();
        assert_eq!(surface, back);
    }

    #[test]
    fn test_surface_dict_roundtrip_torus() {
        let surface = Surface::Torus(Torus::new(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            2.0,
            0.5,
        ));
        let dict = surface.to_dict();
        let back = Surface::from_dict(dict).unwrap();
        assert_eq!(surface, back);
    }

    #[test]
    fn test_surface_dict_backward_compat_lowercase() {
        // Old GLB format uses lowercase variant names
        let json = serde_json::json!({
            "kind": "plane",
            "origin": [1.0, 2.0, 3.0],
            "normal": [0.0, 0.0, 1.0]
        });
        let surface = Surface::from_dict(json).unwrap();
        assert!(matches!(surface, Surface::Plane(_)));
    }

    #[test]
    fn test_surface_dict_backward_compat_semi_angle() {
        // Old GLB format uses "semi_angle" instead of "half_angle"
        let json = serde_json::json!({
            "kind": "cone",
            "apex": [0.0, 0.0, 5.0],
            "axis": [0.0, 0.0, 1.0],
            "semi_angle": 0.3
        });
        let surface = Surface::from_dict(json).unwrap();
        match surface {
            Surface::Cone(c) => assert_relative_eq!(c.half_angle, 0.3),
            _ => panic!("expected Cone"),
        }
    }

    #[test]
    fn test_surface_dict_with_extra_fields() {
        // Extra extent fields should be accepted and ignored
        let json = serde_json::json!({
            "kind": "Cylinder",
            "origin": [0.0, 0.0, 0.0],
            "axis": [0.0, 1.0, 0.0],
            "radius": 0.005,
            "extent_angle": [-3.14, 3.14],
            "extent_height": [0.0, 1.0]
        });
        let surface = Surface::from_dict(json).unwrap();
        assert!(matches!(surface, Surface::Cylinder(_)));
    }

    #[test]
    fn test_surface_dict_bspline_error() {
        let json = serde_json::json!({"kind": "BSpline"});
        assert!(Surface::from_dict(json).is_err());
    }

    #[test]
    fn test_surface_dict_missing_fields() {
        // Missing "normal" — should error, not panic
        let json = serde_json::json!({"kind": "Plane", "origin": [0.0, 0.0, 0.0]});
        assert!(Surface::from_dict(json).is_err());
    }

    #[test]
    fn test_surface_to_dict_structure() {
        let surface = Surface::Cone(Cone::new(Point3::new(0.0, 0.0, 5.0), Vector3::z(), 0.3));
        let dict = surface.to_dict();
        assert_eq!(dict["kind"], "Cone");
        assert_eq!(dict["half_angle"], 0.3);
        assert_eq!(dict["apex"], serde_json::json!([0.0, 0.0, 5.0]));
        // Extent fields should be absent (not null)
        assert!(dict.get("extent_angle").is_none());
    }
}
