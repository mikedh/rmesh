//! Sparse constraint solver using Levenberg-Marquardt with sparse Cholesky
//!
//! Solves the nonlinear least squares problem: min_x ||r(x)||²
//! where r(x) is the vector of constraint residuals.
//!
//! # Algorithm
//!
//! Uses Levenberg-Marquardt: solve (J^T J + λI) δ = -J^T r
//! with adaptive damping and backtracking line search.
//!
//! # Complexity
//!
//! For typical CAD sketches where each vertex has bounded degree d:
//! - Jacobian J: O(m) non-zeros where m = constraints
//! - J^T J: O(n·d²) non-zeros ≈ O(n) for bounded d
//! - Sparse Cholesky: O(n) for banded structure
//! - Total per iteration: O(n)

use nalgebra::DVector;
use nalgebra_sparse::factorization::CscCholesky;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use serde::{Deserialize, Serialize};

use super::types::{Constraint, PointRef};
use crate::creation::feature::environment::Environment;
use crate::creation::feature::error::{FeatureError, Result};
use crate::creation::feature::sketch::Sketch;

// Solver parameters
const MAX_ITERATIONS: usize = 100;
const RESIDUAL_TOL: f64 = 1e-6; // Convergence threshold for ||r||
const STEP_TOL: f64 = 1e-10; // Convergence threshold for ||δ||
const LAMBDA_INIT: f64 = 1e-6; // Initial LM damping
const LAMBDA_MIN: f64 = 1e-10; // Minimum damping
const LAMBDA_MAX: f64 = 1e12; // Maximum damping before giving up
const DIST_EPSILON: f64 = 1e-12; // Regularization for zero-distance Jacobian
const LINE_SEARCH_STEPS: usize = 10;
const LINE_SEARCH_FACTOR: f64 = 0.5;

/// Result of constraint solving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveResult {
    pub positions: Vec<f64>,
    pub status: SolveStatus,
    pub iterations: usize,
    pub residual_norm: f64,
}

impl SolveResult {
    pub fn vertex(&self, index: usize) -> Option<(f64, f64)> {
        let i = index * 2;
        if i + 1 < self.positions.len() {
            Some((self.positions[i], self.positions[i + 1]))
        } else {
            None
        }
    }

    pub fn is_success(&self) -> bool {
        matches!(self.status, SolveStatus::Converged)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolveStatus {
    Converged,
    MaxIterations,
    Inconsistent,
}

/// Internal resolved constraint with concrete vertex indices and values
#[derive(Debug, Clone)]
enum ResolvedConstraint {
    Fixed { vertex: usize, x: f64, y: f64 },
    Distance { a: usize, b: usize, target: f64 },
    Coincident { a: usize, b: usize },
    Horizontal { start: usize, end: usize },
    Vertical { start: usize, end: usize },
    HorizontalDistance { a: usize, b: usize, target: f64 },
    VerticalDistance { a: usize, b: usize, target: f64 },
}

impl ResolvedConstraint {
    /// Number of scalar residuals this constraint produces
    const fn residual_count(&self) -> usize {
        match self {
            Self::Fixed { .. } | Self::Coincident { .. } => 2,
            _ => 1,
        }
    }

    /// Number of non-zero Jacobian entries this constraint produces
    const fn jacobian_nnz(&self) -> usize {
        match self {
            Self::Fixed { .. } | Self::Coincident { .. } | Self::Distance { .. } => 4,
            _ => 2,
        }
    }

    /// Compute residuals into output slice
    fn eval_residuals(&self, pos: &[f64], out: &mut [f64]) {
        match self {
            Self::Fixed { vertex, x, y } => {
                let i = vertex * 2;
                out[0] = pos[i] - x;
                out[1] = pos[i + 1] - y;
            }
            Self::Distance { a, b, target } => {
                let dx = pos[a * 2] - pos[b * 2];
                let dy = pos[a * 2 + 1] - pos[b * 2 + 1];
                out[0] = dx.hypot(dy) - target;
            }
            Self::Coincident { a, b } => {
                out[0] = pos[a * 2] - pos[b * 2];
                out[1] = pos[a * 2 + 1] - pos[b * 2 + 1];
            }
            Self::Horizontal { start, end } => {
                out[0] = pos[end * 2 + 1] - pos[start * 2 + 1];
            }
            Self::Vertical { start, end } => {
                out[0] = pos[end * 2] - pos[start * 2];
            }
            Self::HorizontalDistance { a, b, target } => {
                out[0] = (pos[b * 2] - pos[a * 2]) - target;
            }
            Self::VerticalDistance { a, b, target } => {
                out[0] = (pos[b * 2 + 1] - pos[a * 2 + 1]) - target;
            }
        }
    }

    /// Add Jacobian entries to COO matrix
    fn add_jacobian_entries(&self, pos: &[f64], row: usize, coo: &mut CooMatrix<f64>) {
        match self {
            Self::Fixed { vertex, .. } => {
                let col = vertex * 2;
                coo.push(row, col, 1.0);
                coo.push(row + 1, col + 1, 1.0);
            }
            Self::Distance { a, b, .. } => {
                let dx = pos[a * 2] - pos[b * 2];
                let dy = pos[a * 2 + 1] - pos[b * 2 + 1];
                let dist = dx.hypot(dy).max(DIST_EPSILON);
                let (nx, ny) = (dx / dist, dy / dist);
                coo.push(row, a * 2, nx);
                coo.push(row, a * 2 + 1, ny);
                coo.push(row, b * 2, -nx);
                coo.push(row, b * 2 + 1, -ny);
            }
            Self::Coincident { a, b } => {
                coo.push(row, a * 2, 1.0);
                coo.push(row, b * 2, -1.0);
                coo.push(row + 1, a * 2 + 1, 1.0);
                coo.push(row + 1, b * 2 + 1, -1.0);
            }
            Self::Horizontal { start, end } => {
                coo.push(row, start * 2 + 1, -1.0);
                coo.push(row, end * 2 + 1, 1.0);
            }
            Self::Vertical { start, end } => {
                coo.push(row, start * 2, -1.0);
                coo.push(row, end * 2, 1.0);
            }
            Self::HorizontalDistance { a, b, .. } => {
                coo.push(row, a * 2, -1.0);
                coo.push(row, b * 2, 1.0);
            }
            Self::VerticalDistance { a, b, .. } => {
                coo.push(row, a * 2 + 1, -1.0);
                coo.push(row, b * 2 + 1, 1.0);
            }
        }
    }
}

pub struct Solver2D {
    positions: DVector<f64>,
    constraints: Vec<ResolvedConstraint>,
    num_residuals: usize,
    jacobian_nnz: usize,
}

impl Solver2D {
    pub fn new(sketch: &Sketch, env: &Environment) -> Result<Self> {
        let positions = DVector::from_iterator(
            sketch.vertices.len() * 2,
            sketch.vertices.iter().flat_map(|v| [v.x, v.y]),
        );

        let mut constraints = Vec::with_capacity(sketch.constraints.len());
        let mut num_residuals = 0;
        let mut jacobian_nnz = 0;

        for constraint in &sketch.constraints {
            let resolved = Self::resolve_constraint(constraint, sketch, env)?;
            num_residuals += resolved.residual_count();
            jacobian_nnz += resolved.jacobian_nnz();
            constraints.push(resolved);
        }

        Ok(Self {
            positions,
            constraints,
            num_residuals,
            jacobian_nnz,
        })
    }

    fn resolve_constraint(
        constraint: &Constraint,
        sketch: &Sketch,
        env: &Environment,
    ) -> Result<ResolvedConstraint> {
        match constraint {
            Constraint::Fixed { point, x, y } => Ok(ResolvedConstraint::Fixed {
                vertex: Self::resolve_point_ref(point, sketch)?,
                x: x.resolve(env)?,
                y: y.resolve(env)?,
            }),
            Constraint::Distance { a, b, value } => Ok(ResolvedConstraint::Distance {
                a: Self::resolve_point_ref(a, sketch)?,
                b: Self::resolve_point_ref(b, sketch)?,
                target: value.resolve(env)?,
            }),
            Constraint::Coincident { a, b } => Ok(ResolvedConstraint::Coincident {
                a: Self::resolve_point_ref(a, sketch)?,
                b: Self::resolve_point_ref(b, sketch)?,
            }),
            Constraint::Horizontal { entity } => {
                let (start, end) = sketch.entity_endpoints(*entity)?;
                Ok(ResolvedConstraint::Horizontal { start, end })
            }
            Constraint::Vertical { entity } => {
                let (start, end) = sketch.entity_endpoints(*entity)?;
                Ok(ResolvedConstraint::Vertical { start, end })
            }
            Constraint::HorizontalDistance { a, b, value } => {
                Ok(ResolvedConstraint::HorizontalDistance {
                    a: Self::resolve_point_ref(a, sketch)?,
                    b: Self::resolve_point_ref(b, sketch)?,
                    target: value.resolve(env)?,
                })
            }
            Constraint::VerticalDistance { a, b, value } => {
                Ok(ResolvedConstraint::VerticalDistance {
                    a: Self::resolve_point_ref(a, sketch)?,
                    b: Self::resolve_point_ref(b, sketch)?,
                    target: value.resolve(env)?,
                })
            }
            _ => Err(FeatureError::InvalidOperation(format!(
                "Constraint type not yet implemented: {:?}",
                constraint
            ))),
        }
    }

    fn resolve_point_ref(point_ref: &PointRef, sketch: &Sketch) -> Result<usize> {
        match point_ref {
            PointRef::Vertex { index } => {
                if *index < sketch.vertices.len() {
                    Ok(*index)
                } else {
                    Err(FeatureError::InvalidSketch(format!(
                        "Vertex index {} out of bounds",
                        index
                    )))
                }
            }
            PointRef::Start { entity } => Ok(sketch.entity_endpoints(*entity)?.0),
            PointRef::End { entity } => Ok(sketch.entity_endpoints(*entity)?.1),
            PointRef::Center { entity } => sketch.entity_center(*entity),
        }
    }

    /// Compute residual vector and return squared norm
    fn compute_residuals_into(&self, pos: &[f64], out: &mut [f64]) -> f64 {
        let mut row = 0;
        for c in &self.constraints {
            let count = c.residual_count();
            c.eval_residuals(pos, &mut out[row..row + count]);
            row += count;
        }
        out.iter().map(|r| r * r).sum()
    }

    /// Build sparse Jacobian J where J_ij = ∂r_i/∂x_j
    fn build_jacobian(&self, pos: &[f64]) -> CscMatrix<f64> {
        let mut coo = CooMatrix::try_from_triplets(
            self.num_residuals,
            pos.len(),
            Vec::with_capacity(self.jacobian_nnz),
            Vec::with_capacity(self.jacobian_nnz),
            Vec::with_capacity(self.jacobian_nnz),
        )
        .unwrap();

        let mut row = 0;
        for c in &self.constraints {
            c.add_jacobian_entries(pos, row, &mut coo);
            row += c.residual_count();
        }

        CscMatrix::from(&coo)
    }

    /// Solve: find positions that minimize ||r(x)||²
    pub fn solve(mut self) -> Result<SolveResult> {
        let n = self.positions.len();
        let mut lambda = LAMBDA_INIT;
        let mut residuals = vec![0.0; self.num_residuals];
        let mut trial_pos = vec![0.0; n];

        for iter in 0..MAX_ITERATIONS {
            // Compute residuals and check convergence
            let cost = self.compute_residuals_into(self.positions.as_slice(), &mut residuals);
            let residual_norm = cost.sqrt();

            if residual_norm < RESIDUAL_TOL {
                return Ok(SolveResult {
                    positions: self.positions.as_slice().to_vec(),
                    status: SolveStatus::Converged,
                    iterations: iter + 1,
                    residual_norm,
                });
            }

            // Build Jacobian and normal equations: (J^T J + λI) δ = -J^T r
            let j = self.build_jacobian(self.positions.as_slice());
            let jt = j.transpose();
            let jtj = &jt * &j;
            let r = DVector::from_column_slice(&residuals);
            let jtr = &jt * &r;

            // Add Levenberg-Marquardt damping and solve
            let jtj_damped = add_diagonal(&jtj, lambda);

            let Ok(chol) = CscCholesky::factor(&jtj_damped) else {
                lambda *= 10.0;
                if lambda > LAMBDA_MAX {
                    return Ok(SolveResult {
                        positions: self.positions.as_slice().to_vec(),
                        status: SolveStatus::Inconsistent,
                        iterations: iter + 1,
                        residual_norm,
                    });
                }
                continue;
            };

            let delta = chol.solve(&-&jtr);

            // Backtracking line search (in-place trial updates)
            let mut step = 1.0;
            let mut accepted = false;

            for _ in 0..LINE_SEARCH_STEPS {
                // Trial: x_trial = x + step * δ
                for i in 0..n {
                    trial_pos[i] = self.positions[i] + step * delta[i];
                }

                let new_cost = self.compute_residuals_into(&trial_pos, &mut residuals);

                if new_cost < cost {
                    // Accept: copy trial to positions
                    self.positions.copy_from_slice(&trial_pos);
                    lambda = (lambda * LINE_SEARCH_FACTOR).max(LAMBDA_MIN);
                    accepted = true;
                    break;
                }
                step *= LINE_SEARCH_FACTOR;
            }

            if !accepted {
                lambda *= 10.0;
                if lambda > LAMBDA_MAX {
                    return Ok(SolveResult {
                        positions: self.positions.as_slice().to_vec(),
                        status: SolveStatus::Inconsistent,
                        iterations: iter + 1,
                        residual_norm,
                    });
                }
            }

            // Check step size convergence
            if delta.norm() < STEP_TOL {
                let final_cost =
                    self.compute_residuals_into(self.positions.as_slice(), &mut residuals);
                let final_norm = final_cost.sqrt();
                return Ok(SolveResult {
                    positions: self.positions.as_slice().to_vec(),
                    status: if final_norm < RESIDUAL_TOL {
                        SolveStatus::Converged
                    } else {
                        SolveStatus::Inconsistent
                    },
                    iterations: iter + 1,
                    residual_norm: final_norm,
                });
            }
        }

        // Max iterations reached
        let final_cost = self.compute_residuals_into(self.positions.as_slice(), &mut residuals);
        let final_norm = final_cost.sqrt();
        Ok(SolveResult {
            positions: self.positions.as_slice().to_vec(),
            status: if final_norm < RESIDUAL_TOL {
                SolveStatus::Converged
            } else if final_norm > 1.0 {
                SolveStatus::Inconsistent
            } else {
                SolveStatus::MaxIterations
            },
            iterations: MAX_ITERATIONS,
            residual_norm: final_norm,
        })
    }
}

/// Add scalar to diagonal: returns A + λI
fn add_diagonal(mat: &CscMatrix<f64>, lambda: f64) -> CscMatrix<f64> {
    let n = mat.nrows();
    let nnz = mat.nnz();

    // Pre-allocate with capacity for existing entries + potentially n new diagonal entries
    let mut rows = Vec::with_capacity(nnz + n);
    let mut cols = Vec::with_capacity(nnz + n);
    let mut vals = Vec::with_capacity(nnz + n);
    let mut seen_diag = vec![false; n];

    for (i, j, &v) in mat.triplet_iter() {
        rows.push(i);
        cols.push(j);
        if i == j {
            vals.push(v + lambda);
            seen_diag[i] = true;
        } else {
            vals.push(v);
        }
    }

    // Add missing diagonal entries
    for (i, seen) in seen_diag.into_iter().enumerate() {
        if !seen {
            rows.push(i);
            cols.push(i);
            vals.push(lambda);
        }
    }

    CooMatrix::try_from_triplets(n, n, rows, cols, vals)
        .map(|coo| CscMatrix::from(&coo))
        .expect("valid triplets")
}
