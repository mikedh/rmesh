//! Constraint solving for parametric sketches
//!
//! This module provides:
//! - Constraint types and dimension expressions (in `types.rs`)
//! - Generic constraint solver using Levenberg-Marquardt
//!
//! # Architecture
//!
//! The solver is generic over the variable type, enabling reuse for:
//! - 2D sketch constraints (Point2 variables)
//! - 3D assembly constraints (RigidTransform variables) - future
//!
//! # Example
//!
//! ```ignore
//! use rmesh::creation::feature::constraint::{Solver2D, Constraint, PointRef, Dim};
//!
//! let mut sketch = Sketch::new();
//! // ... add vertices and entities ...
//! sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));
//! sketch.constraints.push(Constraint::distance(PointRef::vertex(0), PointRef::vertex(1), 10.0));
//!
//! let env = Environment::new();
//! let result = sketch.solve(&env)?;
//! ```

mod solver;
mod types;

pub use solver::{SolveResult, SolveStatus, Solver2D};
pub use types::{Constraint, Dim, PointRef};

#[cfg(test)]
mod tests;
