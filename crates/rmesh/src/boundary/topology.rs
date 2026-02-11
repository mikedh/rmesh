//! BREP topology types following ISO 10303-42.
//!
//! These types represent the topological structure of solid models:
//! - Vertices (0D) - points in space
//! - Edges (1D) - curve segments bounded by vertices
//! - Loops - closed chains of oriented edges
//! - Faces (2D) - surface regions bounded by loops
//! - Shells - connected collections of faces
//! - Solids (3D) - volumes bounded by shells

use std::collections::HashMap;

use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

use super::Surface;
use super::faces::{GEOMETRY_TOL, NEWTON_TOL};

// ============================================================================
// Curve types (1D geometry)
// ============================================================================

/// A 1D curve that can be used as edge geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Curve {
    Line(CurveLine),
    Circle(CurveCircle),
    Ellipse(CurveEllipse),
    BSpline(CurveBSpline),
}

/// An infinite line defined by origin and direction.
/// Parameter t gives: origin + t * direction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveLine {
    pub origin: Point3<f64>,
    pub direction: Vector3<f64>,
}

/// A circle in 3D space.
/// Parameter t in [0, 2π) gives: center + radius * (cos(t) * x_axis + sin(t) * y_axis)
/// where y_axis = axis × x_axis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveCircle {
    pub center: Point3<f64>,
    pub axis: Vector3<f64>,
    pub x_axis: Vector3<f64>,
    pub radius: f64,
}

/// An ellipse in 3D space.
/// Parameter t in [0, 2π) gives: center + semi_major * cos(t) * x_axis + semi_minor * sin(t) * y_axis
/// where y_axis = axis × x_axis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveEllipse {
    pub center: Point3<f64>,
    pub axis: Vector3<f64>,
    pub x_axis: Vector3<f64>,
    pub semi_major: f64,
    pub semi_minor: f64,
}

/// A B-spline curve in 3D space (optionally rational / NURBS).
/// Uses De Boor's algorithm for evaluation.
/// When `weights` is `Some`, the curve is rational (NURBS) and the
/// evaluation uses the weighted formula:
///   C(u) = Σ(w_i * N_i(u) * P_i) / Σ(w_i * N_i(u))
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveBSpline {
    /// Polynomial degree (order = degree + 1)
    pub degree: usize,
    /// Control points
    pub control_points: Vec<Point3<f64>>,
    /// Knot vector (length = control_points.len() + degree + 1)
    pub knots: Vec<f64>,
    /// Optional weights for rational B-spline (NURBS).
    /// When present, length must equal control_points.len().
    pub weights: Option<Vec<f64>>,
}

/// Compute binomial coefficient C(n, k).
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Compute basis function derivatives (Algorithm A2.3 from "The NURBS Book").
/// Returns `ders[k][j]` = k-th derivative of N_{span-degree+j, degree}(u).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]
fn basis_funs_ders(
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

impl CurveBSpline {
    /// Create a B-spline from STEP-style multiplicities.
    /// `knot_values` are the unique knot positions, `multiplicities` are their repetition counts.
    /// `weights` is `Some` for rational B-splines (NURBS).
    pub fn from_multiplicities(
        degree: usize,
        control_points: Vec<Point3<f64>>,
        knot_values: &[f64],
        multiplicities: &[usize],
        weights: Option<Vec<f64>>,
    ) -> Self {
        assert_eq!(knot_values.len(), multiplicities.len());
        let knots: Vec<f64> = knot_values
            .iter()
            .zip(multiplicities.iter())
            .flat_map(|(&k, &m)| std::iter::repeat_n(k, m))
            .collect();
        Self {
            degree,
            control_points,
            knots,
            weights,
        }
    }

    /// Find the knot span index for parameter u.
    /// Returns i such that knots[i] <= u < knots[i+1].
    /// Algorithm A2.1 from "The NURBS Book"
    fn find_span(&self, u: f64) -> usize {
        let n = self.control_points.len() - 1;
        let p = self.degree;

        // Special cases at domain boundaries
        if u >= self.knots[n + 1] {
            return n;
        }
        if u <= self.knots[p] {
            return p;
        }

        // Binary search
        let mut low = p;
        let mut high = n + 1;
        let mut mid = usize::midpoint(low, high);

        while u < self.knots[mid] || u >= self.knots[mid + 1] {
            if u < self.knots[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = usize::midpoint(low, high);
        }
        mid
    }

    /// Compute the non-vanishing basis functions at parameter u.
    /// Returns array N[0..=degree] where N[j] = N_{span-degree+j, degree}(u).
    /// Algorithm A2.2 from "The NURBS Book"
    fn basis_funs(&self, span: usize, u: f64) -> Vec<f64> {
        let p = self.degree;
        let mut n = vec![0.0; p + 1];
        let mut left = vec![0.0; p + 1];
        let mut right = vec![0.0; p + 1];

        n[0] = 1.0;
        for j in 1..=p {
            left[j] = u - self.knots[span + 1 - j];
            right[j] = self.knots[span + j] - u;
            let mut saved = 0.0;
            for r in 0..j {
                let temp = n[r] / (right[r + 1] + left[j - r]);
                n[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            n[j] = saved;
        }
        n
    }

    /// Evaluate the curve at parameter u.
    /// Algorithm A3.1 from "The NURBS Book" (extended for rational/NURBS).
    #[allow(clippy::needless_range_loop)]
    pub fn evaluate(&self, u: f64) -> Point3<f64> {
        let span = self.find_span(u);
        let basis = self.basis_funs(span, u);
        let p = self.degree;

        if let Some(ref weights) = self.weights {
            // Rational (NURBS): C(u) = Σ(w_i * N_i * P_i) / Σ(w_i * N_i)
            let mut numerator = Vector3::zeros();
            let mut denominator = 0.0;
            for i in 0..=p {
                let idx = span - p + i;
                let wn = weights[idx] * basis[i];
                numerator += wn * self.control_points[idx].coords;
                denominator += wn;
            }
            Point3::from(numerator / denominator)
        } else {
            let mut point = Point3::origin();
            for i in 0..=p {
                let idx = span - p + i;
                point.coords += basis[i] * self.control_points[idx].coords;
            }
            point
        }
    }

    /// Get the valid parameter range [u_min, u_max].
    pub fn domain(&self) -> (f64, f64) {
        let p = self.degree;
        (self.knots[p], self.knots[self.knots.len() - 1 - p])
    }

    /// Evaluate C(u) and derivatives C'(u), C''(u), ... up to order `n_ders`
    /// in a single find_span + basis_funs_ders pass (Algorithm A2.3).
    /// For rational (NURBS) curves, uses Algorithm A4.2 from "The NURBS Book".
    #[allow(clippy::needless_range_loop)]
    pub fn evaluate_with_derivatives(&self, u: f64, n_ders: usize) -> Vec<Vector3<f64>> {
        let span = self.find_span(u);
        let p = self.degree;
        let ders = basis_funs_ders(span, u, p, &self.knots, n_ders);

        if let Some(ref weights) = self.weights {
            // Compute weighted derivatives: Aw[k] = Σ w_i * N_i^(k) * P_i
            //                               wders[k] = Σ w_i * N_i^(k)
            let mut a_ders = vec![Vector3::zeros(); n_ders + 1];
            let mut w_ders = vec![0.0; n_ders + 1];
            for k in 0..=n_ders {
                for j in 0..=p {
                    let idx = span - p + j;
                    let w = weights[idx];
                    a_ders[k] += (w * ders[k][j]) * self.control_points[idx].coords;
                    w_ders[k] += w * ders[k][j];
                }
            }

            // Apply quotient rule (Algorithm A4.2):
            // CK[0] = Aw[0] / w[0]  (the point itself)
            // CK[k] = (Aw[k] - Σ_{i=1}^{k} C(k,i) * w[i] * CK[k-i]) / w[0]
            let mut ck = vec![Vector3::zeros(); n_ders + 1];
            for k in 0..=n_ders {
                let mut v = a_ders[k];
                for i in 1..=k {
                    let bin = binomial(k, i) as f64;
                    v -= (bin * w_ders[i]) * ck[k - i];
                }
                ck[k] = v / w_ders[0];
            }
            ck
        } else {
            let mut result = vec![Vector3::zeros(); n_ders + 1];
            for k in 0..=n_ders {
                for j in 0..=p {
                    let idx = span - p + j;
                    result[k] += ders[k][j] * self.control_points[idx].coords;
                }
            }
            result
        }
    }

    /// Find parameter t for a point on the curve using Newton-Raphson.
    /// Uses control polygon for initial guess, then refines with analytical derivatives.
    pub fn parameter_at(&self, point: &Point3<f64>) -> f64 {
        let (u_min, u_max) = self.domain();
        let eps = NEWTON_TOL;
        let max_iter = 20;

        // Control polygon initial guess: find closest control polygon edge
        // and map projection to parameter domain. O(n) dot products, no curve evals.
        let n_cp = self.control_points.len();
        let p = self.degree;
        let best_u = if n_cp >= 2 {
            let mut best_dist_sq = f64::MAX;
            let mut best_idx = 0;
            let mut best_t = 0.0;

            for i in 0..n_cp - 1 {
                let a = &self.control_points[i];
                let b = &self.control_points[i + 1];
                let ab = b - a;
                let ab_len_sq = ab.norm_squared();
                let t = if ab_len_sq < 1e-30 {
                    0.0
                } else {
                    ((point - a).dot(&ab) / ab_len_sq).clamp(0.0, 1.0)
                };
                let proj = a.coords + t * ab;
                let dist_sq = (proj - point.coords).norm_squared();
                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_idx = i;
                    best_t = t;
                }
            }

            // Map control polygon index to parameter domain using Greville abscissae.
            // Greville abscissa for control point i: avg(knots[i+1..=i+p])
            let greville = |idx: usize| -> f64 {
                if p == 0 {
                    return (u_min + u_max) * 0.5;
                }
                let sum: f64 = (1..=p).map(|j| self.knots[idx + j]).sum();
                sum / p as f64
            };
            let g_a = greville(best_idx);
            let g_b = greville(best_idx + 1);
            (g_a + best_t * (g_b - g_a)).clamp(u_min, u_max)
        } else {
            (u_min + u_max) * 0.5
        };

        // Newton-Raphson refinement with analytical derivatives
        let mut u = best_u;
        for _ in 0..max_iter {
            let d = self.evaluate_with_derivatives(u, 2);
            let c = Point3::from(d[0]);
            let c_prime = d[1];
            let c_double_prime = d[2];

            let r = c - point;
            if r.norm() < eps {
                break;
            }

            // f(u) = (C(u) - P) · C'(u) = 0 at the closest point
            // f'(u) = C'(u) · C'(u) + (C(u) - P) · C''(u)
            let f = r.dot(&c_prime);
            let f_prime = c_prime.norm_squared() + r.dot(&c_double_prime);

            if f_prime.abs() < GEOMETRY_TOL {
                let f_prime_simple = c_prime.norm_squared();
                if f_prime_simple.abs() < GEOMETRY_TOL {
                    break;
                }
                u = (u - f / f_prime_simple).clamp(u_min, u_max);
            } else {
                let du = -f / f_prime;
                u = (u + du).clamp(u_min, u_max);
                if du.abs() < eps {
                    break;
                }
            }
        }

        u
    }
}

impl Curve {
    /// Human-readable kind name.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Curve::Line(_) => "Line",
            Curve::Circle(_) => "Circle",
            Curve::Ellipse(_) => "Ellipse",
            Curve::BSpline(_) => "BSpline",
        }
    }

    /// Evaluate the curve at parameter t.
    pub fn evaluate(&self, t: f64) -> Point3<f64> {
        match self {
            Curve::Line(line) => line.origin + t * line.direction,
            Curve::Circle(circle) => {
                let y_axis = circle.axis.cross(&circle.x_axis);
                circle.center + circle.radius * (t.cos() * circle.x_axis + t.sin() * y_axis)
            }
            Curve::Ellipse(ellipse) => {
                let y_axis = ellipse.axis.cross(&ellipse.x_axis);
                ellipse.center
                    + ellipse.semi_major * t.cos() * ellipse.x_axis
                    + ellipse.semi_minor * t.sin() * y_axis
            }
            Curve::BSpline(bspline) => bspline.evaluate(t),
        }
    }

    /// Whether this curve is periodic (closes on itself).
    pub fn is_periodic(&self) -> bool {
        match self {
            Curve::Line(_) | Curve::BSpline(_) => false,
            Curve::Circle(_) | Curve::Ellipse(_) => true,
        }
    }

    /// Compute the parameter t for a point on the curve.
    /// Returns the parameter value that, when passed to evaluate(), gives a point
    /// closest to the input point.
    pub fn parameter_at(&self, point: &Point3<f64>) -> f64 {
        match self {
            Curve::Line(line) => {
                // Project point onto line: t = (point - origin) · direction / |direction|²
                let d = point - line.origin;
                let dir_len_sq = line.direction.norm_squared();
                if dir_len_sq < GEOMETRY_TOL {
                    0.0
                } else {
                    d.dot(&line.direction) / dir_len_sq
                }
            }
            Curve::Circle(circle) => {
                // Transform point to local coordinates and use atan2
                let y_axis = circle.axis.cross(&circle.x_axis);
                let d = point - circle.center;
                // Project onto the circle's plane
                let x = d.dot(&circle.x_axis);
                let y = d.dot(&y_axis);
                y.atan2(x)
            }
            Curve::Ellipse(ellipse) => {
                // Transform point to local coordinates and use atan2
                let y_axis = ellipse.axis.cross(&ellipse.x_axis);
                let d = point - ellipse.center;
                // Project onto the ellipse's plane, accounting for semi-axes
                let x = d.dot(&ellipse.x_axis) / ellipse.semi_major;
                let y = d.dot(&y_axis) / ellipse.semi_minor;
                y.atan2(x)
            }
            Curve::BSpline(bspline) => {
                // Use Newton-Raphson to find the parameter
                bspline.parameter_at(point)
            }
        }
    }
}

// ============================================================================
// Topology types
// ============================================================================

/// A vertex (0D topology) - a point in space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrepVertex {
    pub point: Point3<f64>,
}

/// An edge (1D topology) - a curve segment bounded by vertices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrepEdge {
    /// Index into BrepModel.curves
    pub curve: usize,
    /// Index into BrepModel.vertices (start point)
    pub start_vertex: usize,
    /// Index into BrepModel.vertices (end point)
    pub end_vertex: usize,
    /// Curve parameter at start vertex
    pub t_start: f64,
    /// Curve parameter at end vertex
    pub t_end: f64,
}

/// An edge with orientation information.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OrientedEdge {
    /// Index into BrepModel.edges
    pub edge: usize,
    /// true = same direction as edge, false = reversed
    pub same_sense: bool,
}

/// A loop - a closed chain of oriented edges forming a boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrepLoop {
    pub edges: Vec<OrientedEdge>,
}

/// A face (2D topology) - a surface region bounded by loops.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrepFace {
    /// Index into BrepModel.face_surfaces
    pub surface: usize,
    /// Index into BrepModel.loops (outer boundary)
    pub outer_loop: usize,
    /// Indices into BrepModel.loops (holes)
    pub inner_loops: Vec<usize>,
    /// true = surface normal matches face normal
    pub same_sense: bool,
}

/// A shell - a connected set of faces forming a boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrepShell {
    /// Indices into BrepModel.faces
    pub faces: Vec<usize>,
}

/// A solid (3D topology) - a volume bounded by shells.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrepSolid {
    /// Index into BrepModel.shells (outer boundary)
    pub outer_shell: usize,
    /// Indices into BrepModel.shells (voids/cavities)
    pub void_shells: Vec<usize>,
}

// ============================================================================
// Edge adjacency tracking
// ============================================================================

/// Describes how a face uses a specific edge.
///
/// Used by the global refinement algorithm to find all faces that share an edge.
/// When an edge is subdivided, all faces using it must be updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeUse {
    /// Index of the face that uses this edge
    pub face_idx: usize,
    /// Index of the loop (outer or inner) containing the edge
    pub loop_idx: usize,
    /// Position of the oriented edge within the loop
    pub edge_position: usize,
    /// Whether the edge is used in its natural direction
    pub same_sense: bool,
}

// ============================================================================
// BrepModel - the container for all topology
// ============================================================================

/// A complete BREP model containing all topology and geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrepModel {
    pub curves: Vec<Curve>,
    pub vertices: Vec<BrepVertex>,
    pub edges: Vec<BrepEdge>,
    pub loops: Vec<BrepLoop>,
    pub face_surfaces: Vec<Surface>,
    pub faces: Vec<BrepFace>,
    pub shells: Vec<BrepShell>,
    pub solids: Vec<BrepSolid>,
}

impl Default for BrepModel {
    fn default() -> Self {
        Self {
            curves: Vec::new(),
            vertices: Vec::new(),
            edges: Vec::new(),
            loops: Vec::new(),
            face_surfaces: Vec::new(),
            faces: Vec::new(),
            shells: Vec::new(),
            solids: Vec::new(),
        }
    }
}

/// Errors found during BREP model validation.
#[derive(Debug, Clone, PartialEq)]
pub enum BrepError {
    InvalidIndex {
        entity: &'static str,
        index: usize,
        max: usize,
    },
    InvalidParameterRange {
        edge: usize,
        t_start: f64,
        t_end: f64,
    },
    LoopNotClosed {
        loop_: usize,
        gap_after_edge: usize,
    },
    /// Edge is not used exactly twice with opposite orientations.
    /// For a watertight mesh, each edge must be shared by exactly 2 faces
    /// with opposite same_sense values.
    EdgeSharingInvalid {
        edge: usize,
        forward_uses: usize,
        reverse_uses: usize,
    },
}

impl BrepModel {
    /// Create a new empty BREP model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a curve and return its index.
    pub fn add_curve(&mut self, curve: Curve) -> usize {
        let idx = self.curves.len();
        self.curves.push(curve);
        idx
    }

    /// Add a vertex and return its index.
    pub fn add_vertex(&mut self, point: Point3<f64>) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(BrepVertex { point });
        idx
    }

    /// Add an edge and return its index.
    pub fn add_edge(
        &mut self,
        curve: usize,
        start_vertex: usize,
        end_vertex: usize,
        t_start: f64,
        t_end: f64,
    ) -> usize {
        let idx = self.edges.len();
        self.edges.push(BrepEdge {
            curve,
            start_vertex,
            end_vertex,
            t_start,
            t_end,
        });
        idx
    }

    /// Add a loop and return its index.
    pub fn add_loop(&mut self, edges: Vec<OrientedEdge>) -> usize {
        let idx = self.loops.len();
        self.loops.push(BrepLoop { edges });
        idx
    }

    /// Add a surface and return its index.
    pub fn add_surface(&mut self, surface: Surface) -> usize {
        let idx = self.face_surfaces.len();
        self.face_surfaces.push(surface);
        idx
    }

    /// Add a face and return its index.
    pub fn add_face(
        &mut self,
        surface: usize,
        outer_loop: usize,
        inner_loops: Vec<usize>,
        same_sense: bool,
    ) -> usize {
        let idx = self.faces.len();
        self.faces.push(BrepFace {
            surface,
            outer_loop,
            inner_loops,
            same_sense,
        });
        idx
    }

    /// Add a shell and return its index.
    pub fn add_shell(&mut self, faces: Vec<usize>) -> usize {
        let idx = self.shells.len();
        self.shells.push(BrepShell { faces });
        idx
    }

    /// Add a solid and return its index.
    pub fn add_solid(&mut self, outer_shell: usize, void_shells: Vec<usize>) -> usize {
        let idx = self.solids.len();
        self.solids.push(BrepSolid {
            outer_shell,
            void_shells,
        });
        idx
    }

    /// Validate all index references in the model.
    pub fn validate(&self) -> Vec<BrepError> {
        let mut errors = Vec::new();

        // Validate edges
        for (i, edge) in self.edges.iter().enumerate() {
            if edge.curve >= self.curves.len() {
                errors.push(BrepError::InvalidIndex {
                    entity: "edge.curve",
                    index: edge.curve,
                    max: self.curves.len(),
                });
            }
            if edge.start_vertex >= self.vertices.len() {
                errors.push(BrepError::InvalidIndex {
                    entity: "edge.start_vertex",
                    index: edge.start_vertex,
                    max: self.vertices.len(),
                });
            }
            if edge.end_vertex >= self.vertices.len() {
                errors.push(BrepError::InvalidIndex {
                    entity: "edge.end_vertex",
                    index: edge.end_vertex,
                    max: self.vertices.len(),
                });
            }
            // Check parameter range (for non-periodic curves, t_start should be < t_end)
            if edge.curve < self.curves.len()
                && !self.curves[edge.curve].is_periodic()
                && edge.t_start >= edge.t_end
            {
                errors.push(BrepError::InvalidParameterRange {
                    edge: i,
                    t_start: edge.t_start,
                    t_end: edge.t_end,
                });
            }
        }

        // Validate loops
        for loop_ in &self.loops {
            for oe in &loop_.edges {
                if oe.edge >= self.edges.len() {
                    errors.push(BrepError::InvalidIndex {
                        entity: "loop.edge",
                        index: oe.edge,
                        max: self.edges.len(),
                    });
                }
            }
        }

        // Validate faces
        for face in &self.faces {
            if face.surface >= self.face_surfaces.len() {
                errors.push(BrepError::InvalidIndex {
                    entity: "face.surface",
                    index: face.surface,
                    max: self.face_surfaces.len(),
                });
            }
            if face.outer_loop >= self.loops.len() {
                errors.push(BrepError::InvalidIndex {
                    entity: "face.outer_loop",
                    index: face.outer_loop,
                    max: self.loops.len(),
                });
            }
            for &inner in &face.inner_loops {
                if inner >= self.loops.len() {
                    errors.push(BrepError::InvalidIndex {
                        entity: "face.inner_loop",
                        index: inner,
                        max: self.loops.len(),
                    });
                }
            }
        }

        // Validate shells
        for shell in &self.shells {
            for &face_idx in &shell.faces {
                if face_idx >= self.faces.len() {
                    errors.push(BrepError::InvalidIndex {
                        entity: "shell.face",
                        index: face_idx,
                        max: self.faces.len(),
                    });
                }
            }
        }

        // Validate solids
        for solid in &self.solids {
            if solid.outer_shell >= self.shells.len() {
                errors.push(BrepError::InvalidIndex {
                    entity: "solid.outer_shell",
                    index: solid.outer_shell,
                    max: self.shells.len(),
                });
            }
            for &void in &solid.void_shells {
                if void >= self.shells.len() {
                    errors.push(BrepError::InvalidIndex {
                        entity: "solid.void_shell",
                        index: void,
                        max: self.shells.len(),
                    });
                }
            }
        }

        errors
    }

    /// Validate that loops form closed chains (edge endpoints connect).
    pub fn validate_loop_closure(&self, tolerance: f64) -> Vec<BrepError> {
        let mut errors = Vec::new();

        for (loop_idx, loop_) in self.loops.iter().enumerate() {
            if loop_.edges.is_empty() {
                continue;
            }

            for i in 0..loop_.edges.len() {
                let current = &loop_.edges[i];
                let next = &loop_.edges[(i + 1) % loop_.edges.len()];

                // Skip if edge indices are out of bounds
                if current.edge >= self.edges.len() || next.edge >= self.edges.len() {
                    continue;
                }

                let current_edge = &self.edges[current.edge];
                let next_edge = &self.edges[next.edge];

                // Get end vertex of current oriented edge
                let current_end = if current.same_sense {
                    current_edge.end_vertex
                } else {
                    current_edge.start_vertex
                };

                // Get start vertex of next oriented edge
                let next_start = if next.same_sense {
                    next_edge.start_vertex
                } else {
                    next_edge.end_vertex
                };

                // Skip if vertex indices are out of bounds
                if current_end >= self.vertices.len() || next_start >= self.vertices.len() {
                    continue;
                }

                let end_point = &self.vertices[current_end].point;
                let start_point = &self.vertices[next_start].point;

                if (end_point - start_point).norm() > tolerance {
                    errors.push(BrepError::LoopNotClosed {
                        loop_: loop_idx,
                        gap_after_edge: i,
                    });
                }
            }
        }

        errors
    }

    /// Validate that each edge is used exactly twice with opposite orientations.
    ///
    /// For a valid watertight BREP, each edge must be shared by exactly 2 faces:
    /// - Once with `same_sense: true` (forward direction)
    /// - Once with `same_sense: false` (reverse direction)
    ///
    /// This is a prerequisite for producing a watertight mesh from tessellation.
    pub fn validate_edge_sharing(&self) -> Vec<BrepError> {
        let adjacency = self.build_edge_adjacency();
        let mut errors = Vec::new();

        for (edge_idx, uses) in &adjacency {
            let forward = uses.iter().filter(|u| u.same_sense).count();
            let reverse = uses.len() - forward;
            if forward != 1 || reverse != 1 {
                errors.push(BrepError::EdgeSharingInvalid {
                    edge: *edge_idx,
                    forward_uses: forward,
                    reverse_uses: reverse,
                });
            }
        }

        errors
    }

    /// Check if the model passes all validation.
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// Build a map from edge index to all faces that use that edge.
    ///
    /// Returns a HashMap where:
    /// - Key: edge index
    /// - Value: list of EdgeUse structs describing how each face uses that edge
    ///
    /// This is essential for the global refinement algorithm: when an edge needs
    /// to be subdivided, ALL faces using that edge must be updated simultaneously.
    pub fn build_edge_adjacency(&self) -> HashMap<usize, Vec<EdgeUse>> {
        let mut adjacency: HashMap<usize, Vec<EdgeUse>> = HashMap::new();

        for (face_idx, face) in self.faces.iter().enumerate() {
            // Process outer loop
            if face.outer_loop < self.loops.len() {
                let outer_loop = &self.loops[face.outer_loop];
                for (edge_position, oe) in outer_loop.edges.iter().enumerate() {
                    adjacency.entry(oe.edge).or_default().push(EdgeUse {
                        face_idx,
                        loop_idx: face.outer_loop,
                        edge_position,
                        same_sense: oe.same_sense,
                    });
                }
            }

            // Process inner loops (holes)
            for &inner_loop_idx in &face.inner_loops {
                if inner_loop_idx < self.loops.len() {
                    let inner_loop = &self.loops[inner_loop_idx];
                    for (edge_position, oe) in inner_loop.edges.iter().enumerate() {
                        adjacency.entry(oe.edge).or_default().push(EdgeUse {
                            face_idx,
                            loop_idx: inner_loop_idx,
                            edge_position,
                            same_sense: oe.same_sense,
                        });
                    }
                }
            }
        }

        adjacency
    }

    /// Compute the bounding-box diagonal length in model units.
    ///
    /// Used as the characteristic length for relative tolerance computation.
    /// Returns 0.0 if the model has no vertices.
    pub fn characteristic_length(&self) -> f64 {
        match self.bounds() {
            Some((min, max)) => (max - min).norm(),
            None => 0.0,
        }
    }

    /// Compute the axis-aligned bounding box of all vertices.
    pub fn bounds(&self) -> Option<(Point3<f64>, Point3<f64>)> {
        if self.vertices.is_empty() {
            return None;
        }
        let mut min = self.vertices[0].point;
        let mut max = self.vertices[0].point;
        for v in &self.vertices[1..] {
            min.x = min.x.min(v.point.x);
            min.y = min.y.min(v.point.y);
            min.z = min.z.min(v.point.z);
            max.x = max.x.max(v.point.x);
            max.y = max.y.max(v.point.y);
            max.z = max.z.max(v.point.z);
        }
        Some((min, max))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    use crate::boundary::faces::SurfacePlane;

    #[test]
    fn test_curve_line_evaluate() {
        let line = Curve::Line(CurveLine {
            origin: Point3::new(1.0, 2.0, 3.0),
            direction: Vector3::new(1.0, 0.0, 0.0),
        });

        let p0 = line.evaluate(0.0);
        assert_relative_eq!(p0.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(p0.y, 2.0, epsilon = 1e-10);
        assert_relative_eq!(p0.z, 3.0, epsilon = 1e-10);

        let p1 = line.evaluate(5.0);
        assert_relative_eq!(p1.x, 6.0, epsilon = 1e-10);
        assert_relative_eq!(p1.y, 2.0, epsilon = 1e-10);
        assert_relative_eq!(p1.z, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_curve_circle_evaluate() {
        let circle = Curve::Circle(CurveCircle {
            center: Point3::origin(),
            axis: Vector3::z(),
            x_axis: Vector3::x(),
            radius: 2.0,
        });

        // t=0 should give (2, 0, 0)
        let p0 = circle.evaluate(0.0);
        assert_relative_eq!(p0.x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(p0.y, 0.0, epsilon = 1e-10);

        // t=π/2 should give (0, 2, 0)
        let p1 = circle.evaluate(std::f64::consts::FRAC_PI_2);
        assert_relative_eq!(p1.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(p1.y, 2.0, epsilon = 1e-10);

        // t=π should give (-2, 0, 0)
        let p2 = circle.evaluate(std::f64::consts::PI);
        assert_relative_eq!(p2.x, -2.0, epsilon = 1e-10);
        assert_relative_eq!(p2.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_curve_ellipse_evaluate() {
        let ellipse = Curve::Ellipse(CurveEllipse {
            center: Point3::origin(),
            axis: Vector3::z(),
            x_axis: Vector3::x(),
            semi_major: 3.0,
            semi_minor: 1.0,
        });

        // t=0 should give (3, 0, 0)
        let p0 = ellipse.evaluate(0.0);
        assert_relative_eq!(p0.x, 3.0, epsilon = 1e-10);
        assert_relative_eq!(p0.y, 0.0, epsilon = 1e-10);

        // t=π/2 should give (0, 1, 0)
        let p1 = ellipse.evaluate(std::f64::consts::FRAC_PI_2);
        assert_relative_eq!(p1.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(p1.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_curve_kind_name() {
        assert_eq!(
            Curve::Line(CurveLine {
                origin: Point3::origin(),
                direction: Vector3::x(),
            })
            .kind_name(),
            "Line"
        );
        assert_eq!(
            Curve::Circle(CurveCircle {
                center: Point3::origin(),
                axis: Vector3::z(),
                x_axis: Vector3::x(),
                radius: 1.0,
            })
            .kind_name(),
            "Circle"
        );
        assert_eq!(
            Curve::Ellipse(CurveEllipse {
                center: Point3::origin(),
                axis: Vector3::z(),
                x_axis: Vector3::x(),
                semi_major: 2.0,
                semi_minor: 1.0,
            })
            .kind_name(),
            "Ellipse"
        );
    }

    #[test]
    fn test_curve_is_periodic() {
        let line = Curve::Line(CurveLine {
            origin: Point3::origin(),
            direction: Vector3::x(),
        });
        assert!(!line.is_periodic());

        let circle = Curve::Circle(CurveCircle {
            center: Point3::origin(),
            axis: Vector3::z(),
            x_axis: Vector3::x(),
            radius: 1.0,
        });
        assert!(circle.is_periodic());

        let ellipse = Curve::Ellipse(CurveEllipse {
            center: Point3::origin(),
            axis: Vector3::z(),
            x_axis: Vector3::x(),
            semi_major: 2.0,
            semi_minor: 1.0,
        });
        assert!(ellipse.is_periodic());
    }

    #[test]
    fn test_curve_line_parameter_at() {
        let line = Curve::Line(CurveLine {
            origin: Point3::new(1.0, 2.0, 3.0),
            direction: Vector3::new(2.0, 0.0, 0.0),
        });

        // t=0 should give origin
        let t0 = line.parameter_at(&Point3::new(1.0, 2.0, 3.0));
        assert_relative_eq!(t0, 0.0, epsilon = 1e-10);

        // t=1 should give origin + direction
        let t1 = line.parameter_at(&Point3::new(3.0, 2.0, 3.0));
        assert_relative_eq!(t1, 1.0, epsilon = 1e-10);

        // t=0.5 should give midpoint
        let t_half = line.parameter_at(&Point3::new(2.0, 2.0, 3.0));
        assert_relative_eq!(t_half, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_curve_circle_parameter_at() {
        let circle = Curve::Circle(CurveCircle {
            center: Point3::origin(),
            axis: Vector3::z(),
            x_axis: Vector3::x(),
            radius: 2.0,
        });

        // t=0 should give (2, 0, 0)
        let t0 = circle.parameter_at(&Point3::new(2.0, 0.0, 0.0));
        assert_relative_eq!(t0, 0.0, epsilon = 1e-10);

        // t=π/2 should give (0, 2, 0)
        let t_pi2 = circle.parameter_at(&Point3::new(0.0, 2.0, 0.0));
        assert_relative_eq!(t_pi2, std::f64::consts::FRAC_PI_2, epsilon = 1e-10);

        // t=π should give (-2, 0, 0)
        let t_pi = circle.parameter_at(&Point3::new(-2.0, 0.0, 0.0));
        assert_relative_eq!(t_pi, std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn test_curve_ellipse_parameter_at() {
        let ellipse = Curve::Ellipse(CurveEllipse {
            center: Point3::origin(),
            axis: Vector3::z(),
            x_axis: Vector3::x(),
            semi_major: 3.0,
            semi_minor: 1.0,
        });

        // t=0 should give (3, 0, 0)
        let t0 = ellipse.parameter_at(&Point3::new(3.0, 0.0, 0.0));
        assert_relative_eq!(t0, 0.0, epsilon = 1e-10);

        // t=π/2 should give (0, 1, 0)
        let t_pi2 = ellipse.parameter_at(&Point3::new(0.0, 1.0, 0.0));
        assert_relative_eq!(t_pi2, std::f64::consts::FRAC_PI_2, epsilon = 1e-10);
    }

    #[test]
    fn test_brep_model_builder() {
        let mut model = BrepModel::new();

        let curve = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::origin(),
            direction: Vector3::x(),
        }));
        assert_eq!(curve, 0);

        let v0 = model.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = model.add_vertex(Point3::new(1.0, 0.0, 0.0));
        assert_eq!(v0, 0);
        assert_eq!(v1, 1);

        let edge = model.add_edge(curve, v0, v1, 0.0, 1.0);
        assert_eq!(edge, 0);

        assert_eq!(model.curves.len(), 1);
        assert_eq!(model.vertices.len(), 2);
        assert_eq!(model.edges.len(), 1);
    }

    #[test]
    fn test_validate_empty_model() {
        let model = BrepModel::new();
        assert!(model.is_valid());
        assert!(model.validate().is_empty());
    }

    #[test]
    fn test_validate_invalid_curve_index() {
        let mut model = BrepModel::new();
        model.add_vertex(Point3::origin());
        model.add_vertex(Point3::new(1.0, 0.0, 0.0));
        // Edge references curve 99 which doesn't exist
        model.add_edge(99, 0, 1, 0.0, 1.0);

        let errors = model.validate();
        assert!(!errors.is_empty());
        assert!(matches!(
            errors[0],
            BrepError::InvalidIndex {
                entity: "edge.curve",
                index: 99,
                ..
            }
        ));
    }

    #[test]
    fn test_validate_invalid_parameter_range() {
        let mut model = BrepModel::new();
        model.add_curve(Curve::Line(CurveLine {
            origin: Point3::origin(),
            direction: Vector3::x(),
        }));
        model.add_vertex(Point3::origin());
        model.add_vertex(Point3::new(1.0, 0.0, 0.0));
        // t_start > t_end for a line (non-periodic) is invalid
        model.add_edge(0, 0, 1, 5.0, 1.0);

        let errors = model.validate();
        assert!(errors.iter().any(|e| matches!(
            e,
            BrepError::InvalidParameterRange {
                edge: 0,
                t_start: 5.0,
                t_end: 1.0
            }
        )));
    }

    #[test]
    fn test_periodic_curve_wrap_allowed() {
        let mut model = BrepModel::new();
        model.add_curve(Curve::Circle(CurveCircle {
            center: Point3::origin(),
            axis: Vector3::z(),
            x_axis: Vector3::x(),
            radius: 1.0,
        }));
        model.add_vertex(Point3::new(1.0, 0.0, 0.0));
        model.add_vertex(Point3::new(0.0, 1.0, 0.0));
        // t_start > t_end is allowed for periodic curves (wrapping around)
        model.add_edge(0, 0, 1, 5.0, 1.0);

        let errors = model.validate();
        // Should not contain InvalidParameterRange for periodic curves
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e, BrepError::InvalidParameterRange { .. }))
        );
    }

    #[test]
    fn test_validate_loop_closure_triangle() {
        let mut model = BrepModel::new();

        // Create a triangle with 3 vertices
        let v0 = model.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = model.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = model.add_vertex(Point3::new(0.5, 1.0, 0.0));

        // Create 3 line curves
        let c0 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(1.0, 0.0, 0.0),
        }));
        let c1 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(1.0, 0.0, 0.0),
            direction: Vector3::new(-0.5, 1.0, 0.0),
        }));
        let c2 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::new(0.5, 1.0, 0.0),
            direction: Vector3::new(-0.5, -1.0, 0.0),
        }));

        // Create 3 edges: v0->v1, v1->v2, v2->v0
        let e0 = model.add_edge(c0, v0, v1, 0.0, 1.0);
        let e1 = model.add_edge(c1, v1, v2, 0.0, 1.0);
        let e2 = model.add_edge(c2, v2, v0, 0.0, 1.0);

        // Create a closed loop
        model.add_loop(vec![
            OrientedEdge {
                edge: e0,
                same_sense: true,
            },
            OrientedEdge {
                edge: e1,
                same_sense: true,
            },
            OrientedEdge {
                edge: e2,
                same_sense: true,
            },
        ]);

        let errors = model.validate_loop_closure(1e-10);
        assert!(errors.is_empty(), "Triangle loop should be closed");
    }

    #[test]
    fn test_validate_loop_closure_gap() {
        let mut model = BrepModel::new();

        // Create vertices with a gap
        let v0 = model.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = model.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = model.add_vertex(Point3::new(2.0, 0.0, 0.0)); // Gap!
        let v3 = model.add_vertex(Point3::new(0.5, 1.0, 0.0));

        let c0 = model.add_curve(Curve::Line(CurveLine {
            origin: Point3::origin(),
            direction: Vector3::x(),
        }));

        // Edge from v0->v1, but next edge starts at v2 (gap)
        let e0 = model.add_edge(c0, v0, v1, 0.0, 1.0);
        let e1 = model.add_edge(c0, v2, v3, 0.0, 1.0);

        model.add_loop(vec![
            OrientedEdge {
                edge: e0,
                same_sense: true,
            },
            OrientedEdge {
                edge: e1,
                same_sense: true,
            },
        ]);

        let errors = model.validate_loop_closure(1e-10);
        assert!(!errors.is_empty(), "Should detect gap in loop");
        assert!(matches!(
            errors[0],
            BrepError::LoopNotClosed {
                loop_: 0,
                gap_after_edge: 0
            }
        ));
    }

    #[test]
    fn test_serde_roundtrip_curve() {
        let curve = Curve::Circle(CurveCircle {
            center: Point3::new(1.0, 2.0, 3.0),
            axis: Vector3::z(),
            x_axis: Vector3::x(),
            radius: 5.0,
        });

        let json = serde_json::to_string(&curve).unwrap();
        let deserialized: Curve = serde_json::from_str(&json).unwrap();
        assert_eq!(curve, deserialized);
    }

    #[test]
    fn test_serde_roundtrip_brep_model() {
        let mut model = BrepModel::new();

        model.add_curve(Curve::Line(CurveLine {
            origin: Point3::origin(),
            direction: Vector3::x(),
        }));
        let v0 = model.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = model.add_vertex(Point3::new(1.0, 0.0, 0.0));
        model.add_edge(0, v0, v1, 0.0, 1.0);
        model.add_loop(vec![OrientedEdge {
            edge: 0,
            same_sense: true,
        }]);

        model.add_surface(Surface::Plane(SurfacePlane::new(
            Point3::origin(),
            Vector3::z(),
        )));
        model.add_face(0, 0, vec![], true);
        model.add_shell(vec![0]);
        model.add_solid(0, vec![]);

        let json = serde_json::to_string(&model).unwrap();
        let deserialized: BrepModel = serde_json::from_str(&json).unwrap();
        assert_eq!(model, deserialized);
    }
}
