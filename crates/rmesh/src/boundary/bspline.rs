//! B-spline curve and surface types with shared evaluation primitives.
//!
//! Implements Algorithms A2.1–A3.6 from "The NURBS Book" (Piegl & Tiller).
//! Both `CurveBSpline` and `SurfaceBSpline` share the same `find_span`,
//! `basis_funs`, and `basis_funs_ders` free functions.

use nalgebra::{Point2, Point3, Vector3};
use serde::{Deserialize, Serialize};

use super::faces::{GEOMETRY_TOL, METRIC_TOL, NEWTON_TOL};

// ============================================================================
// Shared B-spline primitives
// ============================================================================

/// Tolerance for knot vector comparisons in B-spline evaluation.
const KNOT_TOL: f64 = 1e-14;

/// Compute binomial coefficient C(n, k).
pub(crate) fn binomial(n: usize, k: usize) -> usize {
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

/// Find the knot span index for parameter `u` (Algorithm A2.1 from "The NURBS Book").
///
/// Returns `i` such that `knots[i] <= u < knots[i+1]`.
pub(crate) fn find_span(u: f64, degree: usize, knots: &[f64], n_control: usize) -> usize {
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

    let mut iters = 0usize;
    while u < knots[mid] || u >= knots[mid + 1] {
        iters += 1;
        if iters > 64 {
            break;
        }
        if u < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = usize::midpoint(low, high);
    }
    mid
}

/// Compute the non-vanishing basis functions at parameter `u` (Algorithm A2.2).
///
/// Returns array `N[0..=degree]` where `N[j] = N_{span-degree+j, degree}(u)`.
///
/// When the denominator `right[r+1] + left[j-r]` is zero (repeated knots),
/// we follow the NURBS Book convention of treating 0/0 = 0, which maintains
/// the partition-of-unity property.
pub(crate) fn basis_funs(span: usize, u: f64, degree: usize, knots: &[f64]) -> Vec<f64> {
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

/// Compute basis function derivatives (Algorithm A2.3 from "The NURBS Book").
///
/// Returns `ders[k][j]` = k-th derivative of `N_{span-degree+j, degree}(u)`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]
pub(crate) fn basis_funs_ders(
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
            let temp = if ndu[j][r].abs() < KNOT_TOL {
                0.0
            } else {
                ndu[r][j - 1] / ndu[j][r]
            };
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

// ============================================================================
// CurveBSpline
// ============================================================================

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
    fn find_span(&self, u: f64) -> usize {
        find_span(u, self.degree, &self.knots, self.control_points.len())
    }

    /// Compute the non-vanishing basis functions at parameter u.
    fn basis_funs(&self, span: usize, u: f64) -> Vec<f64> {
        basis_funs(span, u, self.degree, &self.knots)
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

    /// Compute the number of segments needed to approximate the curve between
    /// `t_start` and `t_end` within the given chord tolerance.
    ///
    /// Probes the curve at a coarse resolution, measures the maximum chord
    /// error (distance from curve midpoint to linear interpolation), then
    /// scales segment count using the quadratic relationship (error ∝ 1/n²).
    pub fn sample_count(
        &self,
        t_start: f64,
        t_end: f64,
        chord_tol: f64,
        max_segments: usize,
    ) -> usize {
        let n_probe = 8_usize;
        let mut max_error = 0.0_f64;
        let t_range = t_end - t_start;
        for i in 0..n_probe {
            let t0 = t_start + t_range * (i as f64 / n_probe as f64);
            let t1 = t_start + t_range * ((i as f64 + 0.5) / n_probe as f64);
            let t2 = t_start + t_range * ((i + 1) as f64 / n_probe as f64);
            let p0 = self.evaluate(t0);
            let p2 = self.evaluate(t2);
            let p_mid = self.evaluate(t1);
            let p_linear = p0.coords.lerp(&p2.coords, 0.5);
            max_error = max_error.max((p_mid - Point3::from(p_linear)).norm());
        }
        if max_error < chord_tol {
            n_probe
        } else {
            // error ∝ 1/n², so n_needed = n_probe * sqrt(error / tolerance)
            let ratio = (max_error / chord_tol).sqrt();
            let n = (n_probe as f64 * ratio).ceil() as usize;
            n.clamp(n_probe, max_segments)
        }
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

    /// Compute the segment count needed for a parameter range `[t_start, t_end]`
    /// to stay within a given chord-error tolerance.
    ///
    /// Samples at `n_probe` equally-spaced intervals, measures the maximum
    /// midpoint deviation from the chord, then extrapolates using the
    /// O(1/n²) chord-error scaling law.
    pub fn chord_error_segments(&self, t_start: f64, t_end: f64, tolerance: f64) -> usize {
        let n_probe = 8usize;
        let dt = (t_end - t_start) / n_probe as f64;
        let mut max_dev = 0.0_f64;
        for i in 0..n_probe {
            let t0 = t_start + dt * i as f64;
            let t1 = t0 + dt;
            let p0 = self.evaluate(t0);
            let p1 = self.evaluate(t1);
            let p_mid = self.evaluate(f64::midpoint(t0, t1));
            let linear_mid = Point3::new(
                f64::midpoint(p0.x, p1.x),
                f64::midpoint(p0.y, p1.y),
                f64::midpoint(p0.z, p1.z),
            );
            max_dev = max_dev.max((p_mid - linear_mid).norm());
        }
        if max_dev <= tolerance {
            n_probe
        } else {
            let ratio = (max_dev / tolerance).sqrt();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                (n_probe as f64 * ratio).ceil() as usize
            }
        }
    }
}

// ============================================================================
// SurfaceBSpline
// ============================================================================

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
        find_span(u, self.u_degree, &self.u_knots, self.control_points.len())
    }

    /// Find the knot span index for parameter v in the V direction.
    fn find_v_span(&self, v: f64) -> usize {
        find_span(
            v,
            self.v_degree,
            &self.v_knots,
            self.control_points[0].len(),
        )
    }

    /// Evaluate the surface at parameters (u, v).
    /// Algorithm A3.5 from "The NURBS Book" (extended for rational/NURBS).
    #[allow(clippy::needless_range_loop)]
    pub fn evaluate(&self, u: f64, v: f64) -> Point3<f64> {
        let uspan = self.find_u_span(u);
        let vspan = self.find_v_span(v);

        let nu = basis_funs(uspan, u, self.u_degree, &self.u_knots);
        let nv = basis_funs(vspan, v, self.v_degree, &self.v_knots);

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

        let nu = basis_funs(uspan, u, self.u_degree, &self.u_knots);
        let nv = basis_funs(vspan, v, self.v_degree, &self.v_knots);

        // Compute derivative basis functions
        let nu_der = basis_funs_ders(uspan, u, self.u_degree, &self.u_knots, 1);
        let nv_der = basis_funs_ders(vspan, v, self.v_degree, &self.v_knots, 1);

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

        for _ in 0..MAX_ITER {
            let s = self.evaluate(u, v);
            let (su, sv) = self.derivatives(u, v);

            let delta = s - point;

            // Check convergence
            if delta.norm_squared() < NEWTON_TOL * NEWTON_TOL {
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
        let nu_ders = basis_funs_ders(uspan, u, p, &self.u_knots, du);
        let nv_ders = basis_funs_ders(vspan, v, q, &self.v_knots, dv);

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
    pub fn curvature_at(&self, u: f64, v: f64) -> super::faces::SurfaceCurvature {
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
            return super::faces::SurfaceCurvature::zero();
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
            return super::faces::SurfaceCurvature::zero();
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

        super::faces::SurfaceCurvature {
            kappa_1,
            kappa_2,
            gaussian,
            mean,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // =========================================================================
    // Shared primitive tests
    // =========================================================================

    #[test]
    fn test_basis_funs_partition_of_unity() {
        // Basis functions should sum to 1.0 for any valid parameter
        let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        let n_control = 5; // 9 knots - 3 - 1 = 5

        for &u in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0] {
            let span = find_span(u, degree, &knots, n_control);
            let basis = basis_funs(span, u, degree, &knots);
            let sum: f64 = basis.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_find_span_boundaries() {
        // Knots: [0, 0, 0, 1, 2, 3, 3, 3] for degree 2, 5 control points
        let knots = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let degree = 2;
        let n_control = 5;

        // At start of domain
        assert_eq!(find_span(0.0, degree, &knots, n_control), 2);
        // At end of domain
        assert_eq!(find_span(3.0, degree, &knots, n_control), 4);
        // Below domain
        assert_eq!(find_span(-1.0, degree, &knots, n_control), 2);
        // Above domain
        assert_eq!(find_span(5.0, degree, &knots, n_control), 4);
        // Interior
        assert_eq!(find_span(1.5, degree, &knots, n_control), 3);
    }

    #[test]
    fn test_binomial_values() {
        assert_eq!(binomial(0, 0), 1);
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(6, 3), 20);
        assert_eq!(binomial(3, 5), 0); // k > n
    }

    // =========================================================================
    // CurveBSpline tests
    // =========================================================================

    #[test]
    fn test_bspline_curve_line() {
        // Degree-1 BSpline with two control points is a line
        let bs = CurveBSpline {
            degree: 1,
            control_points: vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)],
            knots: vec![0.0, 0.0, 1.0, 1.0],
            weights: None,
        };

        let p0 = bs.evaluate(0.0);
        assert_relative_eq!(p0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(p0.y, 0.0, epsilon = 1e-10);

        let p1 = bs.evaluate(1.0);
        assert_relative_eq!(p1.x, 10.0, epsilon = 1e-10);

        let mid = bs.evaluate(0.5);
        assert_relative_eq!(mid.x, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bspline_curve_cubic() {
        // Cubic Bezier: 4 control points, knots [0,0,0,0,1,1,1,1]
        let bs = CurveBSpline {
            degree: 3,
            control_points: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(3.0, 2.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
            ],
            knots: vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            weights: None,
        };

        // At t=0.5: Bezier formula gives (2.0, 1.5, 0.0)
        let mid = bs.evaluate(0.5);
        assert_relative_eq!(mid.x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(mid.y, 1.5, epsilon = 1e-10);
        assert_relative_eq!(mid.z, 0.0, epsilon = 1e-10);

        // Endpoints
        let p0 = bs.evaluate(0.0);
        assert_relative_eq!(p0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(p0.y, 0.0, epsilon = 1e-10);

        let p1 = bs.evaluate(1.0);
        assert_relative_eq!(p1.x, 4.0, epsilon = 1e-10);
        assert_relative_eq!(p1.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bspline_nurbs_circle() {
        // Rational quadratic BSpline reproducing a quarter-circle of radius 1.
        // Control points: (1,0), (1,1), (0,1)
        // Weights: 1, 1/√2, 1
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let bs = CurveBSpline {
            degree: 2,
            control_points: vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            weights: Some(vec![1.0, w, 1.0]),
        };

        // At t=0: (1, 0)
        let p0 = bs.evaluate(0.0);
        assert_relative_eq!(p0.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(p0.y, 0.0, epsilon = 1e-10);

        // At t=1: (0, 1)
        let p1 = bs.evaluate(1.0);
        assert_relative_eq!(p1.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(p1.y, 1.0, epsilon = 1e-10);

        // At t=0.5: should be (cos(45°), sin(45°)) = (√2/2, √2/2)
        let mid = bs.evaluate(0.5);
        assert_relative_eq!(mid.x, w, epsilon = 1e-10);
        assert_relative_eq!(mid.y, w, epsilon = 1e-10);

        // All points should lie on the unit circle
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let p = bs.evaluate(t);
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert_relative_eq!(r, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bspline_derivatives() {
        // Verify C'(u) with finite differences
        let bs = CurveBSpline {
            degree: 3,
            control_points: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(3.0, 2.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
            ],
            knots: vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            weights: None,
        };

        let u = 0.4;
        let eps = 1e-7;
        let d = bs.evaluate_with_derivatives(u, 1);
        let c_prime = d[1];

        // Finite difference
        let p_plus = bs.evaluate(u + eps);
        let p_minus = bs.evaluate(u - eps);
        let fd = (p_plus - p_minus) / (2.0 * eps);

        assert_relative_eq!(c_prime.x, fd.x, epsilon = 1e-4);
        assert_relative_eq!(c_prime.y, fd.y, epsilon = 1e-4);
        assert_relative_eq!(c_prime.z, fd.z, epsilon = 1e-4);
    }

    #[test]
    fn test_bspline_parameter_at() {
        // evaluate → parameter_at roundtrip
        let bs = CurveBSpline {
            degree: 3,
            control_points: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 3.0, 0.0),
                Point3::new(3.0, 3.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
            ],
            knots: vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            weights: None,
        };

        for &u in &[0.0, 0.2, 0.5, 0.8, 1.0] {
            let point = bs.evaluate(u);
            let recovered = bs.parameter_at(&point);
            assert_relative_eq!(recovered, u, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_chord_error_segments_straight() {
        // A degree-1 (line) BSpline should return ~n_probe (minimal segments)
        let bs = CurveBSpline {
            degree: 1,
            control_points: vec![Point3::new(0.0, 0.0, 0.0), Point3::new(100.0, 0.0, 0.0)],
            knots: vec![0.0, 0.0, 1.0, 1.0],
            weights: None,
        };

        let n = bs.chord_error_segments(0.0, 1.0, 0.01);
        // A perfectly straight line has zero chord error → returns n_probe = 8
        assert_eq!(n, 8);
    }

    #[test]
    fn test_chord_error_segments_curved() {
        // A curved BSpline should return more segments than a straight one
        let bs = CurveBSpline {
            degree: 3,
            control_points: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 10.0, 0.0),
                Point3::new(3.0, 10.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
            ],
            knots: vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            weights: None,
        };

        let tight = bs.chord_error_segments(0.0, 1.0, 0.001);
        let loose = bs.chord_error_segments(0.0, 1.0, 1.0);

        // Tighter tolerance should require more segments
        assert!(tight > loose, "tight={tight} should be > loose={loose}");
        // Curved BSpline should need more than the minimum 8
        assert!(tight > 8, "tight={tight} should be > 8");
    }
}
