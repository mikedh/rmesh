//! Environment settings for feature models
//!
//! Contains units, variables, and parametric equations for CAD models.
//! Expression evaluation uses Starlark for full scripting support.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use starlark::environment::{GlobalsBuilder, Module};
use starlark::eval::Evaluator;
use starlark::starlark_module;
use starlark::syntax::{AstModule, Dialect};
use starlark::values::list::{AllocList, ListRef};
use starlark::values::{Heap, Value, ValueLike};

use super::error::{FeatureError, Result};

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
        evaluate_starlark(expr, &self.variables)
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

/// Extract f64 from a Starlark Value (int or float)
fn unpack_f64(v: Value) -> anyhow::Result<f64> {
    if let Some(f) = v.downcast_ref::<starlark::values::float::StarlarkFloat>() {
        Ok(f.0)
    } else if let Some(i) = v.unpack_i32() {
        Ok(f64::from(i))
    } else {
        Err(anyhow::anyhow!("Expected number, got {}", v.get_type()))
    }
}

/// Apply a unary f64 function to a Value (scalar or list).
fn apply_unary<'v, F>(x: Value<'v>, heap: &'v Heap, f: F) -> anyhow::Result<Value<'v>>
where
    F: Fn(f64) -> f64,
{
    if let Some(list) = ListRef::from_value(x) {
        let results: Vec<Value> = list
            .iter()
            .map(|v| Ok(heap.alloc(f(unpack_f64(v)?))))
            .collect::<anyhow::Result<_>>()?;
        Ok(heap.alloc(AllocList(results)))
    } else {
        Ok(heap.alloc(f(unpack_f64(x)?)))
    }
}

/// Apply a binary f64 function to two Values with broadcasting support.
fn apply_binary<'v, F>(
    a: Value<'v>,
    b: Value<'v>,
    heap: &'v Heap,
    f: F,
) -> anyhow::Result<Value<'v>>
where
    F: Fn(f64, f64) -> f64,
{
    let list_a = ListRef::from_value(a);
    let list_b = ListRef::from_value(b);

    match (list_a, list_b) {
        (Some(la), Some(lb)) => {
            // element-wise
            let results: Vec<Value> = la
                .iter()
                .zip(lb.iter())
                .map(|(va, vb)| Ok(heap.alloc(f(unpack_f64(va)?, unpack_f64(vb)?))))
                .collect::<anyhow::Result<_>>()?;
            Ok(heap.alloc(AllocList(results)))
        }
        (Some(la), None) => {
            // broadcast b
            let b_val = unpack_f64(b)?;
            let results: Vec<Value> = la
                .iter()
                .map(|va| Ok(heap.alloc(f(unpack_f64(va)?, b_val))))
                .collect::<anyhow::Result<_>>()?;
            Ok(heap.alloc(AllocList(results)))
        }
        (None, Some(lb)) => {
            // broadcast a
            let a_val = unpack_f64(a)?;
            let results: Vec<Value> = lb
                .iter()
                .map(|vb| Ok(heap.alloc(f(a_val, unpack_f64(vb)?))))
                .collect::<anyhow::Result<_>>()?;
            Ok(heap.alloc(AllocList(results)))
        }
        (None, None) => Ok(heap.alloc(f(unpack_f64(a)?, unpack_f64(b)?))),
    }
}

/// # Math Functions
///
/// All math functions work on both scalars and vectors (lists).
/// When given a list, the function is applied element-wise.
///
/// ## Function Reference
///
/// | Category | Functions |
/// |----------|-----------|
/// | Trig (radians) | `sin`, `cos`, `tan` |
/// | Inverse trig | `asin`, `acos`, `atan`, `atan2(y, x)` |
/// | Hyperbolic | `sinh`, `cosh`, `tanh` |
/// | Inverse hyperbolic | `asinh`, `acosh`, `atanh` |
/// | Roots | `sqrt`, `cbrt` |
/// | Exponential | `exp`, `exp2`, `exp_m1` |
/// | Logarithmic | `ln`, `log2`, `log10`, `ln_1p` |
/// | Rounding | `floor`, `ceil`, `round`, `trunc`, `fract` |
/// | Sign/absolute | `abs`, `signum`, `copysign(x, sign)` |
/// | Bounds | `min(a, b)`, `max(a, b)`, `clamp(x, lo, hi)` |
/// | Distance | `hypot(x, y)` |
/// | Conversion | `degrees`, `radians` |
/// | Other | `recip`, `pow(x, y)` |
/// | Checks | `is_nan`, `is_finite`, `is_infinite` (return 0.0/1.0) |
/// | Constants | `pi`, `tau`, `e` |
#[starlark_module]
fn math_functions(builder: &mut GlobalsBuilder) {
    // === Trigonometric functions (radians) ===

    #[starlark(speculative_exec_safe)]
    fn sin<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::sin)
    }

    #[starlark(speculative_exec_safe)]
    fn cos<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::cos)
    }

    #[starlark(speculative_exec_safe)]
    fn tan<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::tan)
    }

    // === Inverse trigonometric functions ===

    #[starlark(speculative_exec_safe)]
    fn asin<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::asin)
    }

    #[starlark(speculative_exec_safe)]
    fn acos<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::acos)
    }

    #[starlark(speculative_exec_safe)]
    fn atan<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::atan)
    }

    #[starlark(speculative_exec_safe)]
    fn atan2<'v>(y: Value<'v>, x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_binary(y, x, heap, f64::atan2)
    }

    // === Hyperbolic functions ===

    #[starlark(speculative_exec_safe)]
    fn sinh<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::sinh)
    }

    #[starlark(speculative_exec_safe)]
    fn cosh<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::cosh)
    }

    #[starlark(speculative_exec_safe)]
    fn tanh<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::tanh)
    }

    // === Inverse hyperbolic functions ===

    #[starlark(speculative_exec_safe)]
    fn asinh<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::asinh)
    }

    #[starlark(speculative_exec_safe)]
    fn acosh<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::acosh)
    }

    #[starlark(speculative_exec_safe)]
    fn atanh<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::atanh)
    }

    // === Root functions ===

    #[starlark(speculative_exec_safe)]
    fn sqrt<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::sqrt)
    }

    #[starlark(speculative_exec_safe)]
    fn cbrt<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::cbrt)
    }

    // === Exponential functions ===

    #[starlark(speculative_exec_safe)]
    fn exp<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::exp)
    }

    #[starlark(speculative_exec_safe)]
    fn exp2<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::exp2)
    }

    #[starlark(speculative_exec_safe)]
    fn exp_m1<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::exp_m1)
    }

    // === Logarithmic functions ===

    #[starlark(speculative_exec_safe)]
    fn ln<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::ln)
    }

    #[starlark(speculative_exec_safe)]
    fn log2<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::log2)
    }

    #[starlark(speculative_exec_safe)]
    fn log10<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::log10)
    }

    #[starlark(speculative_exec_safe)]
    fn ln_1p<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::ln_1p)
    }

    // === Rounding functions ===

    #[starlark(speculative_exec_safe)]
    fn floor<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::floor)
    }

    #[starlark(speculative_exec_safe)]
    fn ceil<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::ceil)
    }

    #[starlark(speculative_exec_safe)]
    fn round<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::round)
    }

    #[starlark(speculative_exec_safe)]
    fn trunc<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::trunc)
    }

    #[starlark(speculative_exec_safe)]
    fn fract<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::fract)
    }

    // === Sign and absolute value ===

    #[starlark(speculative_exec_safe)]
    fn abs<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::abs)
    }

    #[starlark(speculative_exec_safe)]
    fn signum<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::signum)
    }

    #[starlark(speculative_exec_safe)]
    fn copysign<'v>(x: Value<'v>, sign: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_binary(x, sign, heap, f64::copysign)
    }

    // === Bounds ===

    #[starlark(speculative_exec_safe)]
    fn min<'v>(a: Value<'v>, b: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_binary(a, b, heap, f64::min)
    }

    #[starlark(speculative_exec_safe)]
    fn max<'v>(a: Value<'v>, b: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_binary(a, b, heap, f64::max)
    }

    #[starlark(speculative_exec_safe)]
    fn clamp<'v>(
        x: Value<'v>,
        lo: Value<'v>,
        hi: Value<'v>,
        heap: &'v Heap,
    ) -> anyhow::Result<Value<'v>> {
        // For simplicity, we broadcast lo and hi as scalars when x is a list
        if let Some(list) = ListRef::from_value(x) {
            let lo_val = unpack_f64(lo)?;
            let hi_val = unpack_f64(hi)?;
            let results: Vec<Value> = list
                .iter()
                .map(|v| Ok(heap.alloc(unpack_f64(v)?.clamp(lo_val, hi_val))))
                .collect::<anyhow::Result<_>>()?;
            Ok(heap.alloc(AllocList(results)))
        } else {
            Ok(heap.alloc(unpack_f64(x)?.clamp(unpack_f64(lo)?, unpack_f64(hi)?)))
        }
    }

    // === Distance ===

    #[starlark(speculative_exec_safe)]
    fn hypot<'v>(x: Value<'v>, y: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_binary(x, y, heap, f64::hypot)
    }

    // === Conversion ===

    #[starlark(speculative_exec_safe)]
    fn degrees<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::to_degrees)
    }

    #[starlark(speculative_exec_safe)]
    fn radians<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::to_radians)
    }

    // === Other ===

    #[starlark(speculative_exec_safe)]
    fn recip<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, f64::recip)
    }

    #[starlark(speculative_exec_safe)]
    fn pow<'v>(x: Value<'v>, y: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_binary(x, y, heap, f64::powf)
    }

    #[starlark(speculative_exec_safe)]
    fn powf<'v>(x: Value<'v>, y: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_binary(x, y, heap, f64::powf)
    }

    // === Checks (return 0.0/1.0) ===

    #[starlark(speculative_exec_safe)]
    fn is_nan<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, |v| if v.is_nan() { 1.0 } else { 0.0 })
    }

    #[starlark(speculative_exec_safe)]
    fn is_finite<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, |v| if v.is_finite() { 1.0 } else { 0.0 })
    }

    #[starlark(speculative_exec_safe)]
    fn is_infinite<'v>(x: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        apply_unary(x, heap, |v| if v.is_infinite() { 1.0 } else { 0.0 })
    }

    // === Constants ===

    const pi: f64 = std::f64::consts::PI;
    const tau: f64 = std::f64::consts::TAU;
    const e: f64 = std::f64::consts::E;
}

/// Evaluate an expression using Starlark
fn evaluate_starlark(expr: &str, variables: &HashMap<String, f64>) -> Result<f64> {
    // Build globals with math functions in a `math` namespace
    let globals = GlobalsBuilder::standard()
        .with_struct("math", math_functions)
        .build();

    // Create module and inject variables
    let module = Module::new();
    let heap = module.heap();
    for (name, value) in variables {
        module.set(name, heap.alloc(*value));
    }

    // Parse as an expression (not a statement)
    let ast = AstModule::parse("expression", expr.to_string(), &Dialect::Extended)
        .map_err(|e| FeatureError::ExpressionError(format!("Parse error: {}", e)))?;

    // Evaluate and get result directly
    let mut eval = Evaluator::new(&module);
    let result = eval
        .eval_module(ast, &globals)
        .map_err(|e| FeatureError::ExpressionError(format!("Evaluation error: {}", e)))?;

    // Convert to f64
    value_to_f64(result)
}

/// Convert a Starlark Value to f64
fn value_to_f64(value: Value) -> Result<f64> {
    // Try to unpack as i32 (Starlark's int type)
    if let Some(i) = value.unpack_i32() {
        return Ok(f64::from(i));
    }

    // Try to unpack as f64 (Starlark float)
    if let Some(f) = value.downcast_ref::<starlark::values::float::StarlarkFloat>() {
        return Ok(f.0);
    }

    // Try bool (True = 1.0, False = 0.0)
    if let Some(b) = value.unpack_bool() {
        return Ok(if b { 1.0 } else { 0.0 });
    }

    Err(FeatureError::ExpressionError(format!(
        "Cannot convert to number: {}",
        value
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
        assert_relative_eq!(
            env.evaluate("width * height").unwrap(),
            400.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evaluate_sqrt() {
        let env = Environment::new().with_variable("x", 16.0);

        assert_relative_eq!(env.evaluate("math.sqrt(16)").unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(env.evaluate("math.sqrt(x)").unwrap(), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_evaluate_trig() {
        let env = Environment::new();

        assert_relative_eq!(env.evaluate("math.sin(0)").unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(env.evaluate("math.cos(0)").unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(
            env.evaluate("math.sin(math.pi / 2)").unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evaluate_conditional() {
        let env = Environment::new()
            .with_variable("a", 10.0)
            .with_variable("b", 20.0);

        assert_relative_eq!(
            env.evaluate("a if a > b else b").unwrap(),
            20.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("a if a < b else b").unwrap(),
            10.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evaluate_list_operations() {
        let env = Environment::new();

        // List indexing
        assert_relative_eq!(env.evaluate("[1, 2, 3][1]").unwrap(), 2.0, epsilon = 1e-10);

        // Sum-like operations using reduce pattern
        assert_relative_eq!(
            env.evaluate("1 + 2 + 3 + 4").unwrap(),
            10.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evaluate_complex_expression() {
        let env = Environment::new()
            .with_variable("d0", 50.0)
            .with_variable("d1", 30.0);

        // Expression like "d0 * 2" that was mentioned
        assert_relative_eq!(env.evaluate("d0 * 2").unwrap(), 100.0, epsilon = 1e-10);

        // More complex
        assert_relative_eq!(
            env.evaluate("(d0 + d1) / 2").unwrap(),
            40.0,
            epsilon = 1e-10
        );

        // With functions
        assert_relative_eq!(
            env.evaluate("math.sqrt(d0 * d0 + d1 * d1)").unwrap(),
            (50.0_f64.powi(2) + 30.0_f64.powi(2)).sqrt(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evaluate_radians_degrees() {
        let env = Environment::new();

        assert_relative_eq!(
            env.evaluate("math.radians(180)").unwrap(),
            std::f64::consts::PI,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.degrees(math.pi)").unwrap(),
            180.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evaluate_min_max() {
        let env = Environment::new()
            .with_variable("a", 10.0)
            .with_variable("b", 20.0);

        assert_relative_eq!(
            env.evaluate("math.min(a, b)").unwrap(),
            10.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.max(a, b)").unwrap(),
            20.0,
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

    #[test]
    fn test_starlark_math_scalar_and_vector() {
        let env = Environment::default();

        // Scalar operations
        assert_relative_eq!(env.evaluate("math.sin(0.0)").unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(env.evaluate("math.cos(0.0)").unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(
            env.evaluate("math.sqrt(4.0)").unwrap(),
            2.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.abs(-5.0)").unwrap(),
            5.0,
            epsilon = 1e-10
        );

        // Vector operations - index into result to get f64
        assert_relative_eq!(
            env.evaluate("math.sin([0.0, 0.0, 0.0])[0]").unwrap(),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.sqrt([4.0, 9.0, 16.0])[1]").unwrap(),
            3.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.sqrt([1.0, 4.0, 9.0])[2]").unwrap(),
            3.0,
            epsilon = 1e-10
        );

        // Binary with broadcasting
        assert_relative_eq!(
            env.evaluate("math.pow([2.0, 3.0], 2.0)[0]").unwrap(),
            4.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.pow([2.0, 3.0], 2.0)[1]").unwrap(),
            9.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_starlark_math_additional_functions() {
        let env = Environment::default();

        // Hyperbolic functions
        assert_relative_eq!(
            env.evaluate("math.sinh(0.0)").unwrap(),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.cosh(0.0)").unwrap(),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.tanh(0.0)").unwrap(),
            0.0,
            epsilon = 1e-10
        );

        // Cube root
        assert_relative_eq!(
            env.evaluate("math.cbrt(8.0)").unwrap(),
            2.0,
            epsilon = 1e-10
        );

        // Exponential functions
        assert_relative_eq!(env.evaluate("math.exp(0.0)").unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(
            env.evaluate("math.exp2(3.0)").unwrap(),
            8.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.exp_m1(0.0)").unwrap(),
            0.0,
            epsilon = 1e-10
        );

        // Logarithmic functions
        assert_relative_eq!(env.evaluate("math.ln(1.0)").unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(
            env.evaluate("math.log2(8.0)").unwrap(),
            3.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.log10(100.0)").unwrap(),
            2.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.ln_1p(0.0)").unwrap(),
            0.0,
            epsilon = 1e-10
        );

        // Rounding functions
        assert_relative_eq!(
            env.evaluate("math.trunc(3.7)").unwrap(),
            3.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.fract(3.7)").unwrap(),
            0.7,
            epsilon = 1e-10
        );

        // Sign functions
        assert_relative_eq!(
            env.evaluate("math.signum(-5.0)").unwrap(),
            -1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.signum(5.0)").unwrap(),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.copysign(5.0, -1.0)").unwrap(),
            -5.0,
            epsilon = 1e-10
        );

        // Reciprocal
        assert_relative_eq!(
            env.evaluate("math.recip(2.0)").unwrap(),
            0.5,
            epsilon = 1e-10
        );

        // Hypot and clamp
        assert_relative_eq!(
            env.evaluate("math.hypot(3.0, 4.0)").unwrap(),
            5.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.clamp(5.0, 0.0, 3.0)").unwrap(),
            3.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.clamp(-1.0, 0.0, 3.0)").unwrap(),
            0.0,
            epsilon = 1e-10
        );

        // Check functions (return 0.0/1.0)
        assert_relative_eq!(
            env.evaluate("math.is_nan(1.0)").unwrap(),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.is_finite(1.0)").unwrap(),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.is_infinite(1.0)").unwrap(),
            0.0,
            epsilon = 1e-10
        );

        // Constants
        assert_relative_eq!(
            env.evaluate("math.pi").unwrap(),
            std::f64::consts::PI,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.tau").unwrap(),
            std::f64::consts::TAU,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            env.evaluate("math.e").unwrap(),
            std::f64::consts::E,
            epsilon = 1e-10
        );
    }
}
