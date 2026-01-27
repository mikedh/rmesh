//! Starlark expression evaluator
//!
//! Provides standalone evaluation of mathematical expressions using Starlark.
//! Supports variables, arithmetic, conditionals, and a comprehensive math module.

use std::collections::HashMap;

use starlark::environment::{GlobalsBuilder, Module};
use starlark::eval::Evaluator;
use starlark::starlark_module;
use starlark::syntax::{AstModule, Dialect};
use starlark::values::list::{AllocList, ListRef};
use starlark::values::{Heap, Value, ValueLike};

use super::error::{FeatureError, Result};

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
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use rmesh::creation::feature::evaluator;
///
/// let mut vars = HashMap::new();
/// vars.insert("width".to_string(), 100.0);
/// vars.insert("height".to_string(), 50.0);
///
/// assert_eq!(evaluator::evaluate("width * 2", &vars).unwrap(), 200.0);
/// assert_eq!(evaluator::evaluate("width + height", &vars).unwrap(), 150.0);
/// assert_eq!(evaluator::evaluate("math.sqrt(width)", &vars).unwrap(), 10.0);
/// ```
pub fn evaluate(expr: &str, variables: &HashMap<String, f64>) -> Result<f64> {
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
pub fn value_to_f64(value: Value) -> Result<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn eval(expr: &str) -> f64 {
        evaluate(expr, &HashMap::new()).unwrap()
    }

    fn eval_with(expr: &str, vars: &[(&str, f64)]) -> f64 {
        let variables: HashMap<String, f64> =
            vars.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        evaluate(expr, &variables).unwrap()
    }

    #[test]
    fn test_evaluate_simple() {
        assert_relative_eq!(eval("42"), 42.0, epsilon = 1e-10);
        assert_relative_eq!(eval("3.14"), 3.14, epsilon = 1e-10);
    }

    #[test]
    fn test_evaluate_arithmetic() {
        assert_relative_eq!(eval("2 + 3"), 5.0, epsilon = 1e-10);
        assert_relative_eq!(eval("10 - 4"), 6.0, epsilon = 1e-10);
        assert_relative_eq!(eval("3 * 4"), 12.0, epsilon = 1e-10);
        assert_relative_eq!(eval("20 / 4"), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_evaluate_with_variables() {
        assert_relative_eq!(
            eval_with("width", &[("width", 100.0)]),
            100.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            eval_with("width * height", &[("width", 100.0), ("height", 4.0)]),
            400.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evaluate_sqrt() {
        assert_relative_eq!(eval("math.sqrt(16)"), 4.0, epsilon = 1e-10);
        assert_relative_eq!(
            eval_with("math.sqrt(x)", &[("x", 16.0)]),
            4.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evaluate_trig() {
        assert_relative_eq!(eval("math.sin(0)"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.cos(0)"), 1.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.sin(math.pi / 2)"), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_evaluate_conditional() {
        assert_relative_eq!(
            eval_with("a if a > b else b", &[("a", 10.0), ("b", 20.0)]),
            20.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            eval_with("a if a < b else b", &[("a", 10.0), ("b", 20.0)]),
            10.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evaluate_list_operations() {
        // List indexing
        assert_relative_eq!(eval("[1, 2, 3][1]"), 2.0, epsilon = 1e-10);

        // Sum-like operations using reduce pattern
        assert_relative_eq!(eval("1 + 2 + 3 + 4"), 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_evaluate_complex_expression() {
        // Expression like "d0 * 2" that was mentioned
        assert_relative_eq!(
            eval_with("d0 * 2", &[("d0", 50.0), ("d1", 30.0)]),
            100.0,
            epsilon = 1e-10
        );

        // More complex
        assert_relative_eq!(
            eval_with("(d0 + d1) / 2", &[("d0", 50.0), ("d1", 30.0)]),
            40.0,
            epsilon = 1e-10
        );

        // With functions
        assert_relative_eq!(
            eval_with(
                "math.sqrt(d0 * d0 + d1 * d1)",
                &[("d0", 50.0), ("d1", 30.0)]
            ),
            (50.0_f64.powi(2) + 30.0_f64.powi(2)).sqrt(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evaluate_radians_degrees() {
        assert_relative_eq!(eval("math.radians(180)"), std::f64::consts::PI, epsilon = 1e-10);
        assert_relative_eq!(eval("math.degrees(math.pi)"), 180.0, epsilon = 1e-10);
    }

    #[test]
    fn test_evaluate_min_max() {
        assert_relative_eq!(
            eval_with("math.min(a, b)", &[("a", 10.0), ("b", 20.0)]),
            10.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            eval_with("math.max(a, b)", &[("a", 10.0), ("b", 20.0)]),
            20.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_starlark_math_scalar_and_vector() {
        // Scalar operations
        assert_relative_eq!(eval("math.sin(0.0)"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.cos(0.0)"), 1.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.sqrt(4.0)"), 2.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.abs(-5.0)"), 5.0, epsilon = 1e-10);

        // Vector operations - index into result to get f64
        assert_relative_eq!(eval("math.sin([0.0, 0.0, 0.0])[0]"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.sqrt([4.0, 9.0, 16.0])[1]"), 3.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.sqrt([1.0, 4.0, 9.0])[2]"), 3.0, epsilon = 1e-10);

        // Binary with broadcasting
        assert_relative_eq!(eval("math.pow([2.0, 3.0], 2.0)[0]"), 4.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.pow([2.0, 3.0], 2.0)[1]"), 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_starlark_math_additional_functions() {
        // Hyperbolic functions
        assert_relative_eq!(eval("math.sinh(0.0)"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.cosh(0.0)"), 1.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.tanh(0.0)"), 0.0, epsilon = 1e-10);

        // Cube root
        assert_relative_eq!(eval("math.cbrt(8.0)"), 2.0, epsilon = 1e-10);

        // Exponential functions
        assert_relative_eq!(eval("math.exp(0.0)"), 1.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.exp2(3.0)"), 8.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.exp_m1(0.0)"), 0.0, epsilon = 1e-10);

        // Logarithmic functions
        assert_relative_eq!(eval("math.ln(1.0)"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.log2(8.0)"), 3.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.log10(100.0)"), 2.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.ln_1p(0.0)"), 0.0, epsilon = 1e-10);

        // Rounding functions
        assert_relative_eq!(eval("math.trunc(3.7)"), 3.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.fract(3.7)"), 0.7, epsilon = 1e-10);

        // Sign functions
        assert_relative_eq!(eval("math.signum(-5.0)"), -1.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.signum(5.0)"), 1.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.copysign(5.0, -1.0)"), -5.0, epsilon = 1e-10);

        // Reciprocal
        assert_relative_eq!(eval("math.recip(2.0)"), 0.5, epsilon = 1e-10);

        // Hypot and clamp
        assert_relative_eq!(eval("math.hypot(3.0, 4.0)"), 5.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.clamp(5.0, 0.0, 3.0)"), 3.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.clamp(-1.0, 0.0, 3.0)"), 0.0, epsilon = 1e-10);

        // Check functions (return 0.0/1.0)
        assert_relative_eq!(eval("math.is_nan(1.0)"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.is_finite(1.0)"), 1.0, epsilon = 1e-10);
        assert_relative_eq!(eval("math.is_infinite(1.0)"), 0.0, epsilon = 1e-10);

        // Constants
        assert_relative_eq!(eval("math.pi"), std::f64::consts::PI, epsilon = 1e-10);
        assert_relative_eq!(eval("math.tau"), std::f64::consts::TAU, epsilon = 1e-10);
        assert_relative_eq!(eval("math.e"), std::f64::consts::E, epsilon = 1e-10);
    }
}
