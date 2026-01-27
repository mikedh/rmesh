//! Feature-based CAD system
//!
//! This module provides a feature list CAD system where models are built from
//! a sequence of operations (extrude, revolve, sweep, etc.) applied to 2D sketches.
//!
//! # Structure
//!
//! - [`sketch`] - 2D sketches with entity IDs on planes
//! - [`constraint`] - Geometric constraints (Fixed, Coincident, Horizontal, etc.)
//! - [`operation`] - CAD operations (Extrude, Revolve, Sweep, Loft, Fillet, Chamfer)
//! - [`environment`] - Units, variables, and Starlark expression evaluation
//! - [`plane`] - Sketch plane definition with quaternion orientation
//! - [`backend`] - Backend trait for mesh generation
//! - [`error`] - Error types
//!
//! # Example
//!
//! ```
//! use rmesh::creation::feature::{FeatureModel, Sketch, Extrude, Sign};
//! use rmesh::creation::feature::environment::Units;
//!
//! // Create a simple extruded box
//! let model = FeatureModel::new()
//!     .with_units(Units::Millimeters)
//!     .with_variable("width", 100.0)
//!     .with_variable("height", 50.0)
//!     .with_operation(Extrude::simple(
//!         Sketch::rectangle(100.0, 50.0),
//!         25.0
//!     ));
//!
//! assert_eq!(model.operations.len(), 1);
//! ```

pub mod backend;
pub mod backends;
pub mod constraint;
pub mod environment;
pub mod error;
pub mod exchange;
pub mod operation;
pub mod plane;
pub mod sketch;

// Re-export commonly used types
pub use backend::{BackendSettings, DefaultSettings, FeatureBackend};
pub use constraint::{Constraint, Dim, PointRef};
pub use environment::{Environment, Units};
pub use error::{FeatureError, Result};
pub use operation::{
    Chamfer, EdgeSelection, Extrude, Fillet, Loft, Operation, Revolve, Sign, Sweep,
};
pub use plane::SketchPlane;
pub use sketch::{EntityId, Sketch, SketchEntity};

use serde::{Deserialize, Serialize};

/// A feature-based CAD model
///
/// Contains an environment with units and variables, plus an ordered list of operations.
/// Operations are executed sequentially to build the final geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FeatureModel {
    /// Environment with units, variables, and equations
    pub environment: Environment,
    /// Ordered list of operations
    pub operations: Vec<Operation>,
}

impl FeatureModel {
    /// Create a new empty model
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the units
    pub fn with_units(mut self, units: Units) -> Self {
        self.environment.units = units;
        self
    }

    /// Add a variable
    pub fn with_variable(mut self, name: impl Into<String>, value: f64) -> Self {
        self.environment.variables.insert(name.into(), value);
        self
    }

    /// Add an equation
    pub fn with_equation(mut self, name: impl Into<String>, expr: impl Into<String>) -> Self {
        self.environment.equations.insert(name.into(), expr.into());
        self
    }

    /// Add an operation
    pub fn with_operation(mut self, op: impl Into<Operation>) -> Self {
        self.operations.push(op.into());
        self
    }

    /// Add multiple operations
    pub fn with_operations(mut self, ops: impl IntoIterator<Item = Operation>) -> Self {
        self.operations.extend(ops);
        self
    }

    /// Get the number of operations
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the model has no operations
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Iterate over operations
    pub fn iter(&self) -> impl Iterator<Item = &Operation> {
        self.operations.iter()
    }

    /// Get mutable iterator over operations
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Operation> {
        self.operations.iter_mut()
    }

    /// Get operation by index
    pub fn get(&self, index: usize) -> Option<&Operation> {
        self.operations.get(index)
    }

    /// Get mutable operation by index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Operation> {
        self.operations.get_mut(index)
    }

    /// Remove an operation by index
    pub fn remove(&mut self, index: usize) -> Option<Operation> {
        if index < self.operations.len() {
            Some(self.operations.remove(index))
        } else {
            None
        }
    }

    /// Insert an operation at a specific index
    pub fn insert(&mut self, index: usize, op: impl Into<Operation>) {
        self.operations.insert(index, op.into());
    }

    /// Clear all operations
    pub fn clear(&mut self) {
        self.operations.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_model() {
        let model = FeatureModel::new()
            .with_units(Units::Millimeters)
            .with_variable("width", 10.0)
            .with_operation(Extrude::simple(Sketch::rectangle(10.0, 10.0), 5.0));

        assert_eq!(model.operations.len(), 1);
        assert_eq!(model.environment.units, Units::Millimeters);
    }

    #[test]
    fn test_model_builder() {
        let model = FeatureModel::new()
            .with_variable("w", 100.0)
            .with_variable("h", 50.0)
            .with_equation("area", "w * h")
            .with_operation(Extrude::simple(Sketch::rectangle(100.0, 50.0), 10.0))
            .with_operation(Fillet::new(5.0));

        assert_eq!(model.len(), 2);
        assert!(model.environment.has_variable("w"));
        assert!(model.environment.has_variable("h"));
    }

    #[test]
    fn test_model_operations() {
        let mut model =
            FeatureModel::new().with_operation(Extrude::simple(Sketch::circle(10.0), 5.0));

        assert_eq!(model.len(), 1);
        assert!(!model.is_empty());

        model.insert(0, Extrude::simple(Sketch::rectangle(20.0, 20.0), 1.0));
        assert_eq!(model.len(), 2);

        let removed = model.remove(0);
        assert!(removed.is_some());
        assert_eq!(model.len(), 1);

        model.clear();
        assert!(model.is_empty());
    }

    #[test]
    fn test_serde_roundtrip() {
        let model = FeatureModel::new()
            .with_units(Units::Inches)
            .with_variable("depth", 25.4)
            .with_operation(Extrude::simple(Sketch::rectangle(10.0, 5.0), 5.0));

        let json = serde_json::to_string_pretty(&model).unwrap();
        let parsed: FeatureModel = serde_json::from_str(&json).unwrap();

        assert_eq!(model.environment.units, parsed.environment.units);
        assert_eq!(model.operations.len(), parsed.operations.len());
        assert_eq!(
            model.environment.get_variable("depth"),
            parsed.environment.get_variable("depth")
        );
    }
}
