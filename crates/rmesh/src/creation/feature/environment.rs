//! Environment settings for feature models
//!
//! Contains units, variables, and parametric equations for CAD models.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

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
/// Variables can be referenced in dimension expressions.
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
    pub fn with_units(mut self, units: Units) -> Self {
        self.units = units;
        self
    }

    /// Add a variable
    pub fn with_variable(mut self, name: impl Into<String>, value: f64) -> Self {
        self.variables.insert(name.into(), value);
        self
    }

    /// Add an equation
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

    /// Evaluate a simple arithmetic expression with variable substitution
    ///
    /// Supports: +, -, *, /, parentheses, sqrt(), and variable names.
    /// For more complex expressions, use the Starlark evaluator (when available).
    pub fn evaluate(&self, expr: &str) -> super::error::Result<f64> {
        evaluate_simple_expr(expr, &self.variables)
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

/// Simple expression evaluator supporting basic arithmetic and variables
fn evaluate_simple_expr(
    expr: &str,
    variables: &HashMap<String, f64>,
) -> super::error::Result<f64> {
    let expr = expr.trim();

    // Try to parse as a simple number first
    if let Ok(n) = expr.parse::<f64>() {
        return Ok(n);
    }

    // Try to look up as a variable
    if let Some(&value) = variables.get(expr) {
        return Ok(value);
    }

    // Handle sqrt() function
    if let Some(inner) = expr.strip_prefix("sqrt(").and_then(|s| s.strip_suffix(')')) {
        let value = evaluate_simple_expr(inner, variables)?;
        return Ok(value.sqrt());
    }

    // Handle parentheses
    if expr.starts_with('(') && expr.ends_with(')') {
        return evaluate_simple_expr(&expr[1..expr.len() - 1], variables);
    }

    // Find the last + or - at the top level (not inside parentheses)
    let mut paren_depth = 0;
    let mut last_add_sub = None;
    for (i, c) in expr.char_indices() {
        match c {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '+' | '-' if paren_depth == 0 && i > 0 => {
                last_add_sub = Some((i, c));
            }
            _ => {}
        }
    }

    if let Some((idx, op)) = last_add_sub {
        let left = evaluate_simple_expr(&expr[..idx], variables)?;
        let right = evaluate_simple_expr(&expr[idx + 1..], variables)?;
        return Ok(match op {
            '+' => left + right,
            '-' => left - right,
            _ => unreachable!(),
        });
    }

    // Find the last * or / at the top level
    let mut last_mul_div = None;
    paren_depth = 0;
    for (i, c) in expr.char_indices() {
        match c {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '*' | '/' if paren_depth == 0 => {
                last_mul_div = Some((i, c));
            }
            _ => {}
        }
    }

    if let Some((idx, op)) = last_mul_div {
        let left = evaluate_simple_expr(&expr[..idx], variables)?;
        let right = evaluate_simple_expr(&expr[idx + 1..], variables)?;
        return Ok(match op {
            '*' => left * right,
            '/' => {
                if right.abs() < 1e-15 {
                    return Err(super::error::FeatureError::ExpressionError(
                        "Division by zero".into(),
                    ));
                }
                left / right
            }
            _ => unreachable!(),
        });
    }

    // Unrecognized expression
    Err(super::error::FeatureError::ExpressionError(format!(
        "Cannot evaluate expression: '{}'",
        expr
    )))
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
    fn test_evaluate_simple() {
        let env = Environment::new();

        assert_relative_eq!(env.evaluate("42").unwrap(), 42.0, epsilon = 1e-10);
        assert_relative_eq!(env.evaluate("3.14").unwrap(), 3.14, epsilon = 1e-10);
    }

    #[test]
    fn test_evaluate_arithmetic() {
        let env = Environment::new();

        assert_relative_eq!(env.evaluate("2 + 3").unwrap(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(env.evaluate("10 - 4").unwrap(), 6.0, epsilon = 1e-10);
        assert_relative_eq!(env.evaluate("3 * 4").unwrap(), 12.0, epsilon = 1e-10);
        assert_relative_eq!(env.evaluate("20 / 4").unwrap(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_evaluate_with_variables() {
        let env = Environment::new()
            .with_variable("width", 100.0)
            .with_variable("height", 4.0);

        assert_relative_eq!(env.evaluate("width").unwrap(), 100.0, epsilon = 1e-10);
        assert_relative_eq!(env.evaluate("width * height").unwrap(), 400.0, epsilon = 1e-10);
    }

    #[test]
    fn test_evaluate_sqrt() {
        let env = Environment::new().with_variable("x", 16.0);

        assert_relative_eq!(env.evaluate("sqrt(16)").unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(env.evaluate("sqrt(x)").unwrap(), 4.0, epsilon = 1e-10);
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
