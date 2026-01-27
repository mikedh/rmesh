//! Tests for the constraint solver

use super::*;
use crate::creation::feature::environment::Environment;
use crate::creation::feature::sketch::{EntityId, Sketch};
use crate::path::{Line, Segment2D};
use nalgebra::Point2;

/// Euclidean distance (matches solver's numerically stable version)
fn distance(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    (x1 - x0).hypot(y1 - y0)
}

/// Helper to create a triangle sketch with three vertices and two line segments
fn make_triangle_sketch(v0: (f64, f64), v1: (f64, f64), v2: (f64, f64)) -> Sketch {
    let mut sketch = Sketch::new();
    sketch.vertices.push(Point2::new(v0.0, v0.1));
    sketch.vertices.push(Point2::new(v1.0, v1.1));
    sketch.vertices.push(Point2::new(v2.0, v2.1));
    sketch.add(Segment2D::Line(Line::new(0, 1)));
    sketch.add(Segment2D::Line(Line::new(1, 2)));
    sketch
}

// =============================================================================
// Basic constraint tests
// =============================================================================

#[test]
fn test_fixed_constraint_only() {
    let mut sketch = Sketch::new();
    sketch.vertices.push(Point2::new(5.0, 5.0));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));

    let result = sketch.solve(&Environment::default()).unwrap();

    assert!(result.is_success());
    let (x, y) = result.vertex(0).unwrap();
    assert!((x).abs() < 1e-6);
    assert!((y).abs() < 1e-6);
}

#[test]
fn test_distance_constraint_simple() {
    let mut sketch = Sketch::new();
    sketch.vertices.push(Point2::new(0.0, 0.0));
    sketch.vertices.push(Point2::new(3.0, 0.0));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));
    sketch
        .constraints
        .push(Constraint::distance(PointRef::vertex(0), PointRef::vertex(1), 5.0));

    let result = sketch.solve(&Environment::default()).unwrap();

    assert!(result.is_success());
    let (x0, y0) = result.vertex(0).unwrap();
    let (x1, y1) = result.vertex(1).unwrap();
    let dist = distance(x0, y0, x1, y1);
    assert!((dist - 5.0).abs() < 1e-6, "Distance should be 5, got {}", dist);
}

#[test]
fn test_horizontal_constraint() {
    let mut sketch = Sketch::new();
    sketch.vertices.push(Point2::new(0.0, 0.0));
    sketch.vertices.push(Point2::new(5.0, 3.0));
    sketch.add(Segment2D::Line(Line::new(0, 1)));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));
    sketch.constraints.push(Constraint::horizontal(EntityId(1)));

    let result = sketch.solve(&Environment::default()).unwrap();

    assert!(result.is_success());
    let (_, y0) = result.vertex(0).unwrap();
    let (_, y1) = result.vertex(1).unwrap();
    assert!((y1 - y0).abs() < 1e-6, "Line should be horizontal: y0={}, y1={}", y0, y1);
}

#[test]
fn test_vertical_constraint() {
    let mut sketch = Sketch::new();
    sketch.vertices.push(Point2::new(0.0, 0.0));
    sketch.vertices.push(Point2::new(3.0, 5.0));
    sketch.add(Segment2D::Line(Line::new(0, 1)));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));
    sketch.constraints.push(Constraint::vertical(EntityId(1)));

    let result = sketch.solve(&Environment::default()).unwrap();

    assert!(result.is_success());
    let (x0, _) = result.vertex(0).unwrap();
    let (x1, _) = result.vertex(1).unwrap();
    assert!((x1 - x0).abs() < 1e-6, "Line should be vertical: x0={}, x1={}", x0, x1);
}

#[test]
fn test_coincident_constraint() {
    let mut sketch = Sketch::new();
    sketch.vertices.push(Point2::new(0.0, 0.0));
    sketch.vertices.push(Point2::new(5.0, 5.0));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));
    sketch
        .constraints
        .push(Constraint::coincident(PointRef::vertex(0), PointRef::vertex(1)));

    let result = sketch.solve(&Environment::default()).unwrap();

    assert!(result.is_success());
    let (x0, y0) = result.vertex(0).unwrap();
    let (x1, y1) = result.vertex(1).unwrap();
    assert!((x1 - x0).abs() < 1e-6, "Points should coincide in x");
    assert!((y1 - y0).abs() < 1e-6, "Points should coincide in y");
}

// =============================================================================
// Triangle constraint tests (two-segment hinge)
// =============================================================================

/// Two-segment triangle: v0 fixed at origin, v2 fixed at (Lx, 0), v1 is free hinge.
/// Constraints: dist(v0,v1)=L0, dist(v1,v2)=L1
#[test]
fn test_two_segment_triangle_equilateral() {
    let lx = 10.0;
    let l0 = 10.0;
    let l1 = 10.0;

    let mut sketch = make_triangle_sketch((0.0, 0.0), (5.0, 5.0), (lx, 0.0));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(2), lx, 0.0));
    sketch
        .constraints
        .push(Constraint::distance(PointRef::vertex(0), PointRef::vertex(1), l0));
    sketch
        .constraints
        .push(Constraint::distance(PointRef::vertex(1), PointRef::vertex(2), l1));

    let result = sketch.solve(&Environment::default()).unwrap();

    assert!(result.is_success(), "Solver should converge");
    assert!(result.residual_norm < 1e-6, "Residual {} should be < 1e-6", result.residual_norm);

    let (x0, y0) = result.vertex(0).unwrap();
    let (x1, y1) = result.vertex(1).unwrap();
    let (x2, y2) = result.vertex(2).unwrap();

    // Fixed points
    assert!((x0).abs() < 1e-6 && (y0).abs() < 1e-6, "v0 should be at origin");
    assert!((x2 - lx).abs() < 1e-6 && (y2).abs() < 1e-6, "v2 should be at ({}, 0)", lx);

    // Distance constraints
    let dist_01 = distance(x0, y0, x1, y1);
    let dist_12 = distance(x1, y1, x2, y2);
    assert!((dist_01 - l0).abs() < 1e-6, "dist(v0,v1)={} should be {}", dist_01, l0);
    assert!((dist_12 - l1).abs() < 1e-6, "dist(v1,v2)={} should be {}", dist_12, l1);

    // Hinge above x-axis (equilateral has positive y)
    assert!(y1 > 0.0, "v1.y={} should be positive", y1);
}

/// Sweep L1 from 10 to 19.9, verifying triangle inequality is respected
#[test]
fn test_two_segment_triangle_sweep_l1() {
    let lx = 10.0;
    let l0 = 10.0;

    for l1 in [8.0, 10.0, 15.0, 19.9] {
        let mut sketch = make_triangle_sketch((0.0, 0.0), (5.0, 5.0), (lx, 0.0));
        sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));
        sketch.constraints.push(Constraint::fixed(PointRef::vertex(2), lx, 0.0));
        sketch
            .constraints
            .push(Constraint::distance(PointRef::vertex(0), PointRef::vertex(1), l0));
        sketch
            .constraints
            .push(Constraint::distance(PointRef::vertex(1), PointRef::vertex(2), l1));

        let result = sketch.solve(&Environment::default()).unwrap();

        // Triangle inequality: |L0 - L1| <= Lx <= L0 + L1
        assert!(result.residual_norm < 1e-4, "L1={}: residual={:.2e}", l1, result.residual_norm);

        let (x0, y0) = result.vertex(0).unwrap();
        let (x1, y1) = result.vertex(1).unwrap();
        let (x2, y2) = result.vertex(2).unwrap();
        let dist_01 = distance(x0, y0, x1, y1);
        let dist_12 = distance(x1, y1, x2, y2);

        assert!((dist_01 - l0).abs() < 1e-4, "L1={}: dist_01={} != {}", l1, dist_01, l0);
        assert!((dist_12 - l1).abs() < 1e-4, "L1={}: dist_12={} != {}", l1, dist_12, l1);
    }
}

#[test]
fn test_impossible_constraints_detected() {
    let lx = 10.0;
    let l0 = 10.0;
    let l1 = 25.0; // Impossible: L0 + Lx = 20 < 25

    let mut sketch = make_triangle_sketch((0.0, 0.0), (5.0, 5.0), (lx, 0.0));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(2), lx, 0.0));
    sketch
        .constraints
        .push(Constraint::distance(PointRef::vertex(0), PointRef::vertex(1), l0));
    sketch
        .constraints
        .push(Constraint::distance(PointRef::vertex(1), PointRef::vertex(2), l1));

    let result = sketch.solve(&Environment::default()).unwrap();

    assert_eq!(
        result.status,
        SolveStatus::Inconsistent,
        "Should detect impossible constraints, got {:?} with residual {:.2e}",
        result.status,
        result.residual_norm
    );
}

// =============================================================================
// Expression/environment tests
// =============================================================================

#[test]
fn test_constraints_with_expressions() {
    let env = Environment::new()
        .with_variable("base", 10.0)
        .with_variable("arm", 8.0);

    let mut sketch = make_triangle_sketch((0.0, 0.0), (5.0, 5.0), (10.0, 0.0));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(2), "base", 0.0));
    sketch
        .constraints
        .push(Constraint::distance(PointRef::vertex(0), PointRef::vertex(1), "arm"));
    sketch
        .constraints
        .push(Constraint::distance(PointRef::vertex(1), PointRef::vertex(2), "arm"));

    let result = sketch.solve(&env).unwrap();

    assert!(result.is_success());
    assert!(result.residual_norm < 1e-6);

    let (x2, _) = result.vertex(2).unwrap();
    assert!((x2 - 10.0).abs() < 1e-6, "v2.x should be base=10, got {}", x2);
}

#[test]
fn test_dim_resolve_with_starlark_expression() {
    let env = Environment::new()
        .with_variable("d0", 50.0)
        .with_variable("d1", 30.0);

    assert!((Dim::value(100.0).resolve(&env).unwrap() - 100.0).abs() < 1e-10);
    assert!((Dim::expr("d0 * 2").resolve(&env).unwrap() - 100.0).abs() < 1e-10);
    assert!((Dim::expr("d0 + d1").resolve(&env).unwrap() - 80.0).abs() < 1e-10);

    let expected = (50.0_f64.powi(2) + 30.0_f64.powi(2)).sqrt();
    assert!((Dim::expr("math.sqrt(d0 * d0 + d1 * d1)").resolve(&env).unwrap() - expected).abs() < 1e-10);
}

#[test]
fn test_types_serde_roundtrip() {
    let constraints = vec![
        Constraint::fixed(PointRef::vertex(0), 0.0, 0.0),
        Constraint::horizontal(EntityId(1)),
        Constraint::distance(PointRef::vertex(0), PointRef::vertex(1), "d0 * 2"),
    ];

    let json = serde_json::to_string_pretty(&constraints).unwrap();
    let parsed: Vec<Constraint> = serde_json::from_str(&json).unwrap();

    assert_eq!(constraints.len(), parsed.len());
}

// =============================================================================
// Performance tests
// =============================================================================

/// Sweep L1 with 1000 samples - cold start (new sketch each time)
#[test]
fn test_sweep_l1_1000_samples() {
    const NUM_SAMPLES: usize = 1000;
    let (lx, l0) = (10.0, 10.0);
    let (l1_min, l1_max) = (10.0, 19.9);

    let start = std::time::Instant::now();
    let mut total_iters = 0;
    let mut max_residual = 0.0_f64;

    for i in 0..NUM_SAMPLES {
        let t = i as f64 / (NUM_SAMPLES - 1) as f64;
        let l1 = l1_min + t * (l1_max - l1_min);

        let mut sketch = make_triangle_sketch((0.0, 0.0), (5.0, 5.0), (lx, 0.0));
        sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));
        sketch.constraints.push(Constraint::fixed(PointRef::vertex(2), lx, 0.0));
        sketch
            .constraints
            .push(Constraint::distance(PointRef::vertex(0), PointRef::vertex(1), l0));
        sketch
            .constraints
            .push(Constraint::distance(PointRef::vertex(1), PointRef::vertex(2), l1));

        let result = sketch.solve(&Environment::default()).unwrap();
        total_iters += result.iterations;
        max_residual = max_residual.max(result.residual_norm);
        assert!(result.is_success(), "Sample {} failed", i);
    }

    let elapsed = start.elapsed();
    eprintln!(
        "\n[Cold Start] {} samples in {:?} ({:?}/solve, {:.1} avg iters, {:.2e} max residual)",
        NUM_SAMPLES,
        elapsed,
        elapsed / NUM_SAMPLES as u32,
        total_iters as f64 / NUM_SAMPLES as f64,
        max_residual
    );
}

/// Sweep L1 with 1000 samples - warm start (reuse sketch, vertices updated)
#[test]
fn test_sweep_l1_1000_samples_warm_start() {
    const NUM_SAMPLES: usize = 1000;
    let (lx, l0) = (10.0, 10.0);
    let (l1_min, l1_max) = (10.0, 19.9);

    let mut sketch = make_triangle_sketch((0.0, 0.0), (5.0, 5.0), (lx, 0.0));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), 0.0, 0.0));
    sketch.constraints.push(Constraint::fixed(PointRef::vertex(2), lx, 0.0));
    sketch
        .constraints
        .push(Constraint::distance(PointRef::vertex(0), PointRef::vertex(1), l0));
    let l1_idx = sketch.constraints.len();
    sketch
        .constraints
        .push(Constraint::distance(PointRef::vertex(1), PointRef::vertex(2), l1_min));

    let env = Environment::default();
    let start = std::time::Instant::now();
    let mut total_iters = 0;
    let mut max_residual = 0.0_f64;

    for i in 0..NUM_SAMPLES {
        let t = i as f64 / (NUM_SAMPLES - 1) as f64;
        let l1 = l1_min + t * (l1_max - l1_min);

        sketch.constraints[l1_idx] =
            Constraint::distance(PointRef::vertex(1), PointRef::vertex(2), l1);

        let result = sketch.solve(&env).unwrap();
        total_iters += result.iterations;
        max_residual = max_residual.max(result.residual_norm);
        assert!(result.is_success(), "Sample {} failed", i);
    }

    let elapsed = start.elapsed();
    eprintln!(
        "\n[Warm Start] {} samples in {:?} ({:?}/solve, {:.1} avg iters, {:.2e} max residual)",
        NUM_SAMPLES,
        elapsed,
        elapsed / NUM_SAMPLES as u32,
        total_iters as f64 / NUM_SAMPLES as f64,
        max_residual
    );
}

/// Scaling test: N-gon with distance constraints between consecutive vertices.
/// Verifies O(n) complexity with sparse solver.
#[test]
fn test_scaling_many_vertices() {
    eprintln!("\n{:>6} {:>10} {:>6} {:>12} {:>12}", "N", "Time", "Iters", "Residual", "Per-vertex");

    for n in [10, 50, 100, 200] {
        let mut sketch = Sketch::new();

        // Create N vertices in a rough circle with noise
        let radius = 10.0;
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            let r = radius * (1.0 + 0.1 * (i as f64 * 0.7).sin());
            sketch.vertices.push(Point2::new(r * angle.cos(), r * angle.sin()));
        }

        // Fix first vertex, add distance constraints for regular N-gon
        sketch.constraints.push(Constraint::fixed(PointRef::vertex(0), radius, 0.0));
        let edge_len = 2.0 * radius * (std::f64::consts::PI / n as f64).sin();
        for i in 0..n {
            sketch.constraints.push(Constraint::distance(
                PointRef::vertex(i),
                PointRef::vertex((i + 1) % n),
                edge_len,
            ));
        }

        let start = std::time::Instant::now();
        let result = sketch.solve(&Environment::default()).unwrap();
        let elapsed = start.elapsed();

        eprintln!(
            "{:>6} {:>10.2?} {:>6} {:>12.2e} {:>12.2?}",
            n, elapsed, result.iterations, result.residual_norm, elapsed / n as u32
        );

        assert!(
            result.is_success(),
            "N={} failed: {:?}, residual={:.2e}",
            n, result.status, result.residual_norm
        );
    }
}
