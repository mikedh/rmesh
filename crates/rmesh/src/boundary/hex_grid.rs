//! Hex grid interior point generation for surface tessellation.
//!
//! Places interior points on a BREP face in UV space using a hex grid pattern
//! with curvature-adaptive spacing. Points are filtered to lie inside the
//! boundary polygon and outside any holes, then passed to CDT triangulation.

#![allow(clippy::manual_midpoint)]

use nalgebra::Point2;

use super::Surface;
use super::faces::{CURVATURE_TOL, GEOMETRY_TOL};
use super::tesselate::point_in_polygon;

/// Maximum interior points per face.
const MAX_INTERIOR_POINTS: usize = 256;

/// Generate interior UV points for a BREP face using a hex grid.
/// Returns interior UV positions only (boundary vertices are already in the caller's state).
pub fn generate_interior_points(
    surface: &Surface,
    boundary_uvs: &[Point2<f64>],
    hole_uvs: &[&[Point2<f64>]],
    tolerance: f64,
) -> Vec<Point2<f64>> {
    // Compute UV bounding box from boundary
    let (u_min, u_max, v_min, v_max) = uv_bounds(boundary_uvs, hole_uvs);

    // Compute average spacing at face center
    let u_mid = (u_min + u_max) / 2.0;
    let v_mid = (v_min + v_max) / 2.0;
    let avg_spacing = node_spacing_uv(surface, u_mid, v_mid, tolerance);

    // If spacing is huge (nearly flat), no interior points needed
    let u_span = u_max - u_min;
    let v_span = v_max - v_min;
    if avg_spacing > u_span.max(v_span) * 2.0 {
        return Vec::new();
    }

    // Clamp spacing to something reasonable relative to the face size
    let min_dim = u_span.min(v_span);
    let spacing = avg_spacing.min(min_dim);
    if spacing < GEOMETRY_TOL {
        return Vec::new();
    }

    generate_hex_grid(
        surface,
        boundary_uvs,
        hole_uvs,
        u_min,
        u_max,
        v_min,
        v_max,
        spacing,
        tolerance,
    )
}

/// Compute node spacing in UV space at a given (u, v) location.
/// Maps curvature to point spacing, corrected for UV metric distortion.
fn node_spacing_uv(surface: &Surface, u: f64, v: f64, tolerance: f64) -> f64 {
    let kappa = surface.curvature_at(u, v).kappa_max();
    let kappa = if kappa.is_finite() { kappa } else { 1e6 };

    let spacing_3d = if kappa > CURVATURE_TOL {
        (8.0 * tolerance / kappa).sqrt()
    } else {
        // Planar region: use a large spacing (will be clamped by caller)
        return f64::MAX;
    };

    // Convert 3D spacing to UV via first fundamental form
    let (e, _f, g) = surface.first_fundamental_form(u, v);
    let scale = ((e + g) / 2.0).sqrt();
    if scale > GEOMETRY_TOL {
        spacing_3d / scale
    } else {
        spacing_3d
    }
}

/// Compute UV bounding box.
fn uv_bounds(boundary: &[Point2<f64>], holes: &[&[Point2<f64>]]) -> (f64, f64, f64, f64) {
    let mut u_min = f64::MAX;
    let mut u_max = f64::MIN;
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;

    for p in boundary {
        u_min = u_min.min(p.x);
        u_max = u_max.max(p.x);
        v_min = v_min.min(p.y);
        v_max = v_max.max(p.y);
    }
    for &hole in holes {
        for p in hole {
            u_min = u_min.min(p.x);
            u_max = u_max.max(p.x);
            v_min = v_min.min(p.y);
            v_max = v_max.max(p.y);
        }
    }

    (u_min, u_max, v_min, v_max)
}

/// Generate a hex grid of interior points, keeping only those inside the boundary.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn generate_hex_grid(
    surface: &Surface,
    boundary: &[Point2<f64>],
    holes: &[&[Point2<f64>]],
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
    spacing: f64,
    tolerance: f64,
) -> Vec<Point2<f64>> {
    // Scale spacing up if the grid would exceed MAX_INTERIOR_POINTS
    let mut spacing = spacing;
    let mut scale_iters = 0usize;
    loop {
        scale_iters += 1;
        if scale_iters > 50 {
            return Vec::new();
        }
        let row_sp = spacing;
        let col_sp = spacing * (3.0_f64).sqrt() / 2.0;
        let inset = spacing * 0.5;
        let us = u_min + inset;
        let ue = u_max - inset;
        let vs = v_min + inset;
        let ve = v_max - inset;
        if us >= ue || vs >= ve {
            return Vec::new();
        }
        let nr = ((ve - vs) / col_sp).ceil() as usize + 1;
        let nc = ((ue - us) / row_sp).ceil() as usize + 1;
        if nr * nc <= MAX_INTERIOR_POINTS {
            break;
        }
        spacing *= 1.5;
    }

    let row_spacing = spacing;
    let col_spacing = spacing * (3.0_f64).sqrt() / 2.0;

    // Inset from boundary to avoid placing points right on edges
    let inset = spacing * 0.5;
    let u_start = u_min + inset;
    let u_end = u_max - inset;
    let v_start = v_min + inset;
    let v_end = v_max - inset;

    if u_start >= u_end || v_start >= v_end {
        return Vec::new();
    }

    let n_rows = ((v_end - v_start) / col_spacing).ceil() as usize + 1;
    let n_cols = ((u_end - u_start) / row_spacing).ceil() as usize + 1;

    let mut points = Vec::with_capacity(n_rows * n_cols);

    for row in 0..n_rows {
        let v = v_start + row as f64 * col_spacing;
        if v > v_end {
            break;
        }
        // Hex offset: odd rows shift by half spacing
        let u_offset = if row % 2 == 1 { row_spacing * 0.5 } else { 0.0 };

        for col in 0..n_cols {
            let u = u_start + col as f64 * row_spacing + u_offset;
            if u > u_end {
                break;
            }

            let pt = Point2::new(u, v);

            // Must be inside outer boundary
            if !point_in_polygon(&pt, boundary) {
                continue;
            }

            // Must be outside all holes
            if holes.iter().any(|&h| point_in_polygon(&pt, h)) {
                continue;
            }

            let r = node_spacing_uv(surface, u, v, tolerance);
            // Skip if spacing is huge (planar region)
            if !r.is_finite() || r > (u_max - u_min).max(v_max - v_min) {
                continue;
            }

            points.push(pt);
        }
    }

    points
}
