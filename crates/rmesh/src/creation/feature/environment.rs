//! Environment settings for feature models
//!
//! Contains units, variables, and parametric equations for CAD models.
//! Expression evaluation uses Starlark for full scripting support.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::error::Result;
use super::evaluator;

/// Units for dimensions in the model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Units {
    /// Meters (SI base unit)
    #[default]
    Meters,
    /// Millimeters (1/1000 of a meter)
    Millimeters,
    /// Inches (25.4 mm)
    Inches,
    /// Feet (12 inches)
    Feet,
}

impl Units {
    /// Conversion factor from this unit to meters
    pub fn to_meters(&self) -> f64 {
        match self {
            Units::Meters => 1.0,
            Units::Millimeters => 0.001,
            Units::Inches => 0.0254,
            Units::Feet => 0.3048,
        }
    }

    /// Conversion factor from meters to this unit
    pub fn from_meters(&self) -> f64 {
        1.0 / self.to_meters()
    }

    /// Convert a value from one unit to another
    pub fn convert(value: f64, from: Units, to: Units) -> f64 {
        let meters = value * from.to_meters();
        meters * to.from_meters()
    }
}

/// Environment settings for a feature model
///
/// Contains units, named variables, and parametric equations.
/// Variables can be referenced in dimension expressions using Starlark syntax.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Environment {
    /// Units for all dimensions in the model
    pub units: Units,
    /// Named variables for parametric design (e.g., "width" -> 10.0)
    pub variables: HashMap<String, f64>,
    /// Parametric equations (evaluated in order)
    /// Key is variable name, value is expression string
    pub equations: HashMap<String, String>,
}

impl Environment {
    /// Create a new empty environment with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the units
    #[must_use]
    pub fn with_units(mut self, units: Units) -> Self {
        self.units = units;
        self
    }

    /// Add a variable
    #[must_use]
    pub fn with_variable(mut self, name: impl Into<String>, value: f64) -> Self {
        self.variables.insert(name.into(), value);
        self
    }

    /// Add an equation
    #[must_use]
    pub fn with_equation(mut self, name: impl Into<String>, expr: impl Into<String>) -> Self {
        self.equations.insert(name.into(), expr.into());
        self
    }

    /// Get a variable value by name
    pub fn get_variable(&self, name: &str) -> Option<f64> {
        self.variables.get(name).copied()
    }

    /// Set a variable value
    pub fn set_variable(&mut self, name: impl Into<String>, value: f64) {
        self.variables.insert(name.into(), value);
    }

    /// Evaluate an expression using Starlark
    ///
    /// Supports full Starlark syntax including:
    /// - Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
    /// - Comparisons: `<`, `>`, `<=`, `>=`, `==`, `!=`
    /// - Boolean: `and`, `or`, `not`
    /// - Conditionals: `x if condition else y`
    /// - Lists: `[1, 2, 3]`, list comprehensions
    /// - String operations
    /// - Math module: `math.sqrt`, `math.sin`, `math.cos`, `math.pi`, etc.
    ///
    /// All environment variables are available by name.
    ///
    /// # Examples
    ///
    /// ```
    /// use rmesh::creation::feature::Environment;
    ///
    /// let env = Environment::new()
    ///     .with_variable("d0", 50.0)
    ///     .with_variable("d1", 30.0);
    ///
    /// assert_eq!(env.evaluate("d0 * 2").unwrap(), 100.0);
    /// assert_eq!(env.evaluate("d0 + d1").unwrap(), 80.0);
    /// assert_eq!(env.evaluate("math.sqrt(d0)").unwrap(), 50.0_f64.sqrt());
    /// assert_eq!(env.evaluate("d0 if d0 > d1 else d1").unwrap(), 50.0);
    /// ```
    pub fn evaluate(&self, expr: &str) -> Result<f64> {
        evaluator::evaluate(expr, &self.variables)
    }

    /// Get all variable names
    pub fn variable_names(&self) -> impl Iterator<Item = &String> {
        self.variables.keys()
    }

    /// Check if a variable exists
    pub fn has_variable(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_units_conversion() {
        assert_relative_eq!(Units::Millimeters.to_meters(), 0.001, epsilon = 1e-10);
        assert_relative_eq!(Units::Inches.to_meters(), 0.0254, epsilon = 1e-10);

        let inches = 10.0;
        let mm = Units::convert(inches, Units::Inches, Units::Millimeters);
        assert_relative_eq!(mm, 254.0, epsilon = 1e-10);
    }

    #[test]
    fn test_environment_variables() {
        let env = Environment::new()
            .with_variable("width", 100.0)
            .with_variable("height", 50.0);

        assert_eq!(env.get_variable("width"), Some(100.0));
        assert_eq!(env.get_variable("height"), Some(50.0));
        assert_eq!(env.get_variable("unknown"), None);
    }

    #[test]
    fn test_environment_evaluate() {
        let env = Environment::new()
            .with_variable("width", 100.0)
            .with_variable("height", 4.0);

        assert_relative_eq!(env.evaluate("width").unwrap(), 100.0, epsilon = 1e-10);
        assert_relative_eq!(
            env.evaluate("width * height").unwrap(),
            400.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.sqrt(width)").unwrap(),
            10.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let env = Environment::new()
            .with_units(Units::Millimeters)
            .with_variable("width", 100.0);

        let json = serde_json::to_string(&env).unwrap();
        let parsed: Environment = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.units, Units::Millimeters);
        assert_eq!(parsed.get_variable("width"), Some(100.0));
    }
}
