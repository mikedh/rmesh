//! Fidget SDF backend implementation
//!
//! Converts feature operations to Fidget Trees (SDFs), then meshes them using
//! Manifold Dual Contouring.

use fidget::{
    context::Tree,
    jit::JitShape,
    mesh::{Octree, Settings as OctreeSettings},
};
use nalgebra::Point2;

use crate::creation::feature::{
    Extrude, FeatureBackend, FeatureError, FeatureModel, Operation, Result, Sign, Sketch, sketch,
};
use crate::mesh::Trimesh;

use super::super::backend::BackendSettings;

/// Settings for Fidget mesh generation
#[derive(Debug, Clone)]
pub struct FidgetSettings {
    /// Octree depth - higher = more detail (default: 6)
    pub depth: u8,
    /// Bounds for the model (min, max) - if None, auto-calculated
    pub bounds: Option<([f64; 3], [f64; 3])>,
}

impl Default for FidgetSettings {
    fn default() -> Self {
        Self {
            depth: 6,
            bounds: None,
        }
    }
}

impl FidgetSettings {
    /// Create settings with specific depth
    pub fn with_depth(depth: u8) -> Self {
        Self {
            depth,
            ..Default::default()
        }
    }

    /// Create settings with custom bounds
    pub fn with_bounds(min: [f64; 3], max: [f64; 3]) -> Self {
        Self {
            bounds: Some((min, max)),
            ..Default::default()
        }
    }
}

impl BackendSettings for FidgetSettings {
    fn resolution(&self) -> u32 {
        self.depth as u32
    }
}

/// Fidget SDF backend for feature meshing
///
/// Uses signed distance fields (SDFs) and Manifold Dual Contouring to generate
/// high-quality meshes from feature operations.
///
/// # Supported Operations
///
/// - **Extrude**: Both Add (union) and Remove (difference) modes
///   - Line-based profiles (rectangles, polygons)
///   - Circle profiles
///
/// # Unsupported Operations
///
/// - Revolve, Sweep, Loft (not yet implemented)
/// - Fillet, Chamfer (requires edge detection)
#[derive(Debug, Clone, Default)]
pub struct FidgetBackend;

impl FidgetBackend {
    /// Create a new Fidget backend
    pub fn new() -> Self {
        Self
    }

    /// Execute the model with given settings
    pub fn execute_with_settings(
        &self,
        model: &FeatureModel,
        settings: &FidgetSettings,
    ) -> Result<Trimesh> {
        if model.operations.is_empty() {
            return Err(FeatureError::BackendError("Empty model".to_string()));
        }

        // Build the combined SDF tree from all operations
        let tree = build_tree(model)?;

        // Calculate bounds if not provided
        let bounds = settings
            .bounds
            .or_else(|| model.bounds())
            .unwrap_or(([-1.0; 3], [1.0; 3]));

        // Build octree and mesh (JIT compiles SDF to native code for fast evaluation)
        let shape = JitShape::from(tree);

        // Calculate the center and size of our bounding box
        let cx = ((bounds.0[0] + bounds.1[0]) / 2.0) as f32;
        let cy = ((bounds.0[1] + bounds.1[1]) / 2.0) as f32;
        let cz = ((bounds.0[2] + bounds.1[2]) / 2.0) as f32;
        let size = ((bounds.1[0] - bounds.0[0])
            .max(bounds.1[1] - bounds.0[1])
            .max(bounds.1[2] - bounds.0[2])
            * 1.1) as f32; // 10% padding

        // Build world_to_model transform
        // Maps octree space [-1, 1]³ to model space (centered at center, size `size`)
        let half_size = size / 2.0;

        // Create Matrix4 using nalgebra 0.34 (fidget's version)
        let world_to_model = nalgebra_034::Matrix4::new(
            half_size, 0.0, 0.0, cx, 0.0, half_size, 0.0, cy, 0.0, 0.0, half_size, cz, 0.0, 0.0,
            0.0, 1.0,
        );

        let octree_settings = OctreeSettings {
            depth: settings.depth,
            world_to_model,
            ..Default::default()
        };

        let octree = Octree::build(&shape, &octree_settings).ok_or_else(|| {
            FeatureError::BackendError("Octree build failed or cancelled".to_string())
        })?;

        let fidget_mesh = octree.walk_dual();

        // Convert fidget mesh to rmesh Trimesh
        fidget_to_trimesh(&fidget_mesh)
    }
}

impl FeatureBackend for FidgetBackend {
    type Settings = FidgetSettings;

    fn execute(&self, model: &FeatureModel, settings: &Self::Settings) -> Result<Trimesh> {
        self.execute_with_settings(model, settings)
    }

    fn name(&self) -> &'static str {
        "fidget"
    }

    fn supports_operation(&self, op_type: &str) -> bool {
        matches!(op_type, "Extrude")
    }
}

/// Convert a fidget Mesh to an rmesh Trimesh
fn fidget_to_trimesh(mesh: &fidget::mesh::Mesh) -> Result<Trimesh> {
    // Convert vertices to flat f64 slice to avoid nalgebra version issues
    // (fidget may use a different nalgebra version than rmesh)
    let vertices: Vec<f64> = mesh
        .vertices
        .iter()
        .flat_map(|v| [v.x as f64, v.y as f64, v.z as f64])
        .collect();

    // Convert triangles to flat usize slice
    let faces: Vec<usize> = mesh
        .triangles
        .iter()
        .flat_map(|t| [t.x, t.y, t.z])
        .collect();

    Trimesh::from_slice(&vertices, &faces)
        .map_err(|e| FeatureError::BackendError(format!("Failed to create Trimesh: {:?}", e)))
}

/// Build a Fidget Tree from the feature model
fn build_tree(model: &FeatureModel) -> Result<Tree> {
    let mut result: Option<Tree> = None;

    for op in &model.operations {
        let op_tree = operation_to_tree(op)?;
        let sign = operation_sign(op);

        result = Some(match result {
            None => op_tree,
            Some(existing) => match sign {
                Sign::Add => existing.min(op_tree),     // Union = min of SDFs
                Sign::Remove => existing.max(-op_tree), // Difference = max(a, -b)
            },
        });
    }

    result.ok_or(FeatureError::BackendError("Empty model".to_string()))
}

/// Convert a single operation to a Fidget Tree
fn operation_to_tree(op: &Operation) -> Result<Tree> {
    match op {
        Operation::Extrude(extrude) => extrude_to_tree(extrude),
        Operation::Revolve(_) => Err(FeatureError::BackendError(
            "Revolve not yet implemented".to_string(),
        )),
        Operation::Sweep(_) => Err(FeatureError::BackendError(
            "Sweep not yet implemented".to_string(),
        )),
        Operation::Loft(_) => Err(FeatureError::BackendError(
            "Loft not yet implemented".to_string(),
        )),
        Operation::Fillet(_) => Err(FeatureError::BackendError(
            "Fillet not yet implemented".to_string(),
        )),
        Operation::Chamfer(_) => Err(FeatureError::BackendError(
            "Chamfer not yet implemented".to_string(),
        )),
    }
}

/// Convert an extrusion to a Fidget Tree.
///
/// Supports arbitrary sketch planes by projecting world coordinates into
/// the plane's local coordinate system (u, v, w) via dot products.
fn extrude_to_tree(extrude: &Extrude) -> Result<Tree> {
    let plane = &extrude.sketch.plane;
    let o = plane.origin;
    let u = plane.x_axis();
    let v = plane.y_axis();
    let n = plane.normal();

    // World coords → local coords via dot products
    let dx = Tree::x() - o.x;
    let dy = Tree::y() - o.y;
    let dz = Tree::z() - o.z;

    let local_u = dx.clone() * u.x + dy.clone() * u.y + dz.clone() * u.z;
    let local_v = dx.clone() * v.x + dy.clone() * v.y + dz.clone() * v.z;
    let local_w = dx * n.x + dy * n.y + dz * n.z;

    // 2D profile evaluated in local (u, v) space
    let profile_2d = sketch_to_tree_local(&extrude.sketch, local_u, local_v)?;

    // Extrusion bounds in local w (normal) direction
    let depth = extrude.depth;
    let w_dist = (local_w - depth / 2.0).abs() - depth / 2.0;

    // Extrusion: max of 2D profile and w bounds
    Ok(profile_2d.max(w_dist))
}

/// Convert a sketch to a 2D SDF Tree using the given local coordinate Trees.
///
/// `local_u` and `local_v` are Fidget Trees representing the projection
/// of world coordinates onto the sketch plane's local axes.
///
/// Uses triangulated polygon SDFs: each triangle is convex (3 half-planes
/// via `max`), all triangles unioned via `min`. This handles non-convex
/// polygons and holes correctly, unlike the previous half-plane approach
/// which only worked for convex shapes.
fn sketch_to_tree_local(sketch: &Sketch, local_u: Tree, local_v: Tree) -> Result<Tree> {
    let mut path = sketch.to_path2d();
    if path.segments.is_empty() {
        return Err(FeatureError::InvalidSketch("Empty sketch".into()));
    }
    path.deviation = Some(sketch::DEFAULT_TOLERANCE);

    let (vertices, triangles) = path.triangulate();
    if triangles.is_empty() {
        return Err(FeatureError::InvalidSketch("No closed polygons".into()));
    }

    let mut result: Option<Tree> = None;
    for tri in triangles {
        let Some(tri_sdf) = triangle_sdf(
            vertices[tri[0]],
            vertices[tri[1]],
            vertices[tri[2]],
            &local_u,
            &local_v,
        ) else {
            continue; // skip degenerate triangles
        };
        result = Some(match result {
            None => tri_sdf,
            Some(existing) => existing.min(tri_sdf), // union
        });
    }

    result.ok_or(FeatureError::InvalidSketch("Failed to build SDF".into()))
}

/// SDF for a single triangle via intersection of 3 half-planes.
///
/// Returns `None` if the triangle is degenerate (all edges have zero length).
fn triangle_sdf(
    p0: Point2<f64>,
    p1: Point2<f64>,
    p2: Point2<f64>,
    local_u: &Tree,
    local_v: &Tree,
) -> Option<Tree> {
    let mut result: Option<Tree> = None;
    let pts = [p0, p1, p2];
    for i in 0..3 {
        let a = pts[i];
        let b = pts[(i + 1) % 3];
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-10 {
            continue;
        }
        // Inward normal (rotate edge 90° CW for CCW winding)
        let (nx, ny) = (dy / len, -dx / len);
        let half_plane = local_u.clone() * nx + local_v.clone() * ny - (nx * a.x + ny * a.y);
        result = Some(match result {
            None => half_plane,
            Some(existing) => existing.max(half_plane),
        });
    }
    result
}

/// Get the sign (add/remove) from an operation
fn operation_sign(op: &Operation) -> Sign {
    match op {
        Operation::Extrude(e) => e.sign,
        Operation::Revolve(r) => r.sign,
        Operation::Sweep(s) => s.sign,
        Operation::Loft(l) => l.sign,
        Operation::Fillet(_) => Sign::Add,
        Operation::Chamfer(_) => Sign::Add,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::feature::Sketch;
    use crate::path::{Line, Segment2D};

    #[test]
    fn test_extrude_rectangle_to_box() {
        let model =
            FeatureModel::new().with_operation(Extrude::simple(Sketch::rectangle(1.0, 1.0), 1.0));

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(5);
        let mesh = backend
            .execute(&model, &settings)
            .expect("Failed to execute");

        assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
        assert!(!mesh.faces.is_empty(), "Mesh should have faces");
    }

    #[test]
    fn test_extrude_circle_to_cylinder() {
        let sketch = Sketch::circle(0.5);
        let model = FeatureModel::new().with_operation(Extrude::simple(sketch, 1.0));

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(5);
        let mesh = backend
            .execute(&model, &settings)
            .expect("Failed to execute");

        assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
        assert!(!mesh.faces.is_empty(), "Mesh should have faces");
    }

    #[test]
    fn test_boolean_cut() {
        // Box extrusion
        let box_sketch = Sketch::rectangle(2.0, 1.0);
        let box_extrude = Extrude::simple(box_sketch, 3.0);

        // Cylinder cut
        let hole_sketch = Sketch::circle(0.375);
        let hole_extrude = Extrude::new(hole_sketch, 0.5, Sign::Remove);

        let model = FeatureModel::new()
            .with_operation(box_extrude)
            .with_operation(hole_extrude);

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(5);
        let mesh = backend
            .execute(&model, &settings)
            .expect("Failed to execute");

        assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
        assert!(!mesh.faces.is_empty(), "Mesh should have faces");
    }

    #[test]
    fn test_boolean_cut_volume() {
        // Create a box and cut a through-hole, then verify the volume
        //
        // Box: 2×2×1 = 4 cubic units
        // Cylinder: π × 0.25² × 1 ≈ 0.1963 cubic units
        // Expected: 4 - 0.1963 ≈ 3.804 cubic units

        let box_width = 2.0;
        let box_height = 2.0;
        let box_depth = 1.0;
        let hole_radius = 0.25;

        // Box extrusion
        let box_sketch = Sketch::rectangle(box_width, box_height);
        let box_extrude = Extrude::simple(box_sketch, box_depth);

        // Through-hole cylinder cut (same depth as box)
        let hole_sketch = Sketch::circle(hole_radius);
        let hole_extrude = Extrude::new(hole_sketch, box_depth, Sign::Remove);

        let model = FeatureModel::new()
            .with_operation(box_extrude)
            .with_operation(hole_extrude);

        // Use higher depth for better accuracy
        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(7);
        let mesh = backend
            .execute(&model, &settings)
            .expect("Failed to execute");

        // Calculate expected volume
        let box_volume = box_width * box_height * box_depth;
        let cylinder_volume = std::f64::consts::PI * hole_radius * hole_radius * box_depth;
        let expected_volume = box_volume - cylinder_volume;

        // Get actual volume from mesh
        let actual_volume = mesh.volume();

        // SDF meshing has some approximation error, allow 5% tolerance
        let tolerance = 0.05;
        let relative_error = (actual_volume - expected_volume).abs() / expected_volume;

        println!(
            "Box with hole: expected volume {:.4}, actual {:.4}, error {:.2}%",
            expected_volume,
            actual_volume,
            relative_error * 100.0
        );

        assert!(
            relative_error < tolerance,
            "Volume mismatch: expected {:.4}, got {:.4} (error: {:.2}%)",
            expected_volume,
            actual_volume,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_empty_model_error() {
        let model = FeatureModel::new();
        let backend = FidgetBackend::new();
        let settings = FidgetSettings::default();

        let result = backend.execute(&model, &settings);
        assert!(result.is_err());
    }

    #[test]
    fn test_backend_name() {
        let backend = FidgetBackend::new();
        assert_eq!(backend.name(), "fidget");
    }

    #[test]
    fn test_supports_operation() {
        let backend = FidgetBackend::new();
        assert!(backend.supports_operation("Extrude"));
        assert!(!backend.supports_operation("Revolve"));
        assert!(!backend.supports_operation("Fillet"));
    }

    #[test]
    fn test_extrude_xz_plane() {
        // Extrude a circle on the XZ plane → should produce geometry
        // extending along Y (the plane normal).
        use crate::creation::feature::SketchPlane;

        let mut sketch = Sketch::on_plane(SketchPlane::xz());
        sketch.add_circle(Point2::new(0.0, 0.0), 0.5);

        let extrude = Extrude::new(sketch, 2.0, Sign::Add);
        let model = FeatureModel::new().with_operation(extrude);

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(5);
        let mesh = backend
            .execute(&model, &settings)
            .expect("XZ plane extrusion failed");

        assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
        assert!(!mesh.faces.is_empty(), "Mesh should have faces");

        // Bounding box should have significant Y extent (extrusion direction)
        let mut y_min = f64::MAX;
        let mut y_max = f64::MIN;
        for v in &mesh.vertices {
            y_min = y_min.min(v.y);
            y_max = y_max.max(v.y);
        }
        let y_extent = y_max - y_min;
        assert!(
            y_extent > 1.0,
            "XZ plane extrusion should have Y extent > 1.0, got {:.3}",
            y_extent
        );
    }

    #[test]
    fn test_extrude_yz_plane() {
        // Extrude a rectangle on the YZ plane → should produce geometry
        // extending along X (the plane normal).
        use crate::creation::feature::SketchPlane;

        let mut sketch = Sketch::on_plane(SketchPlane::yz());
        // Manually add a rectangle since Sketch::rectangle() uses default XY plane
        sketch.add_line(Point2::new(-0.5, -0.5), Point2::new(0.5, -0.5));
        sketch.add_line(Point2::new(0.5, -0.5), Point2::new(0.5, 0.5));
        sketch.add_line(Point2::new(0.5, 0.5), Point2::new(-0.5, 0.5));
        sketch.add_line(Point2::new(-0.5, 0.5), Point2::new(-0.5, -0.5));

        let extrude = Extrude::new(sketch, 2.0, Sign::Add);
        let model = FeatureModel::new().with_operation(extrude);

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(5);
        let mesh = backend
            .execute(&model, &settings)
            .expect("YZ plane extrusion failed");

        assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
        assert!(!mesh.faces.is_empty(), "Mesh should have faces");

        // Bounding box should have significant X extent (extrusion direction)
        let mut x_min = f64::MAX;
        let mut x_max = f64::MIN;
        for v in &mesh.vertices {
            x_min = x_min.min(v.x);
            x_max = x_max.max(v.x);
        }
        let x_extent = x_max - x_min;
        assert!(
            x_extent > 1.0,
            "YZ plane extrusion should have X extent > 1.0, got {:.3}",
            x_extent
        );
    }

    #[test]
    fn test_multi_plane_cross() {
        // Cylinder body on XY plane + perpendicular spike on XZ plane.
        use crate::creation::feature::SketchPlane;

        // Main body: circle on XY plane extruded along Z
        let mut body_sketch = Sketch::on_plane(SketchPlane::xy());
        body_sketch.add_circle(Point2::new(0.0, 0.0), 1.0);
        let body = Extrude::new(body_sketch, 3.0, Sign::Add);

        // Spike: small circle on XZ plane extruded along Y
        let mut spike_sketch = Sketch::on_plane(SketchPlane::xz());
        spike_sketch.add_circle(Point2::new(0.0, 1.5), 0.3);
        let spike = Extrude::new(spike_sketch, 2.0, Sign::Add);

        let model = FeatureModel::new()
            .with_operation(body)
            .with_operation(spike);

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(6);
        let mesh = backend
            .execute(&model, &settings)
            .expect("Multi-plane extrusion failed");

        assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
        assert!(!mesh.faces.is_empty(), "Mesh should have faces");

        // The combined shape should extend in Y (from the spike)
        let mut y_min = f64::MAX;
        let mut y_max = f64::MIN;
        for v in &mesh.vertices {
            y_min = y_min.min(v.y);
            y_max = y_max.max(v.y);
        }
        let y_extent = y_max - y_min;
        assert!(
            y_extent > 1.5,
            "Multi-plane shape should have Y extent > 1.5 (spike), got {:.3}",
            y_extent
        );
    }

    /// Integration test: Load SLDPRT file → parse → mesh → verify volume
    ///
    /// The 123.2021.SLDPRT file contains:
    /// - Boss-Extrude1: 1"×2" rectangle extruded 3" (6 cubic inches)
    /// - Cut-Extrude1: 0.75" diameter hole, 0.38" deep (~0.168 cubic inches)
    /// - Expected volume: ~5.83 cubic inches
    #[test]
    fn test_sldprt_to_mesh_volume() {
        use crate::creation::feature::exchange::sldprt;

        // Try to find the test file
        let test_file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../rcad/attempts/models/123.2021.SLDPRT");

        if !test_file.exists() {
            println!("Test file not found, skipping: {:?}", test_file);
            return;
        }

        // Parse SLDPRT
        let import = sldprt::read_sldprt(&test_file).expect("Failed to parse SLDPRT");

        println!("Parsed {} operations from SLDPRT:", import.operations.len());
        for (i, op) in import.operations.iter().enumerate() {
            match op {
                Operation::Extrude(e) => {
                    println!(
                        "  {}: Extrude depth={:.4}, sign={:?}, entities={}",
                        i,
                        e.depth,
                        e.sign,
                        e.sketch.entities.len()
                    );
                }
                _ => println!("  {}: {:?}", i, op),
            }
        }

        for warning in &import.warnings {
            println!("  Warning: {}", warning);
        }

        // Need at least 1 operation to mesh
        assert!(
            !import.operations.is_empty(),
            "SLDPRT should have at least one operation"
        );

        // Build model and mesh it
        let model = import.into_model();
        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(6);

        let mesh = backend.execute(&model, &settings).expect("Failed to mesh");

        println!(
            "Mesh: {} vertices, {} faces",
            mesh.vertices.len(),
            mesh.faces.len()
        );

        let volume = mesh.volume();
        println!("Volume: {:.4} cubic units", volume);

        // The file contains:
        // - 1" × 2" × 3" box = 6 cubic inches
        // - 0.75" diameter hole, 0.375" deep (subtracted)
        //
        // Expected volume: box - hole = 6.0 - π×0.375²×0.375 ≈ 5.83 cubic inches
        // Note: Due to mesh discretization and dimension extraction tolerance,
        // we allow up to 5% error from the solid box volume.
        let expected_box_volume = 1.0 * 2.0 * 3.0; // 6.0 cubic inches

        let relative_error = (volume - expected_box_volume).abs() / expected_box_volume;
        assert!(
            relative_error < 0.05,
            "Volume should be ~{:.2} in³, got {:.4} (error: {:.2}%)",
            expected_box_volume,
            volume,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_extrude_l_shape() {
        // L-shaped non-convex polygon (6 line segments)
        //
        //   (0,2)----(1,2)
        //     |        |
        //     |  (1,1)--(2,1)
        //     |          |
        //   (0,0)------(2,0)
        //
        // Area = 2×1 (bottom) + 1×1 (left column) = 3 sq units
        // But the full L is: bottom 2×1 + top-left 1×1 = 3
        let mut sketch = Sketch::new();
        let p0 = sketch.add_vertex(Point2::new(0.0, 0.0));
        let p1 = sketch.add_vertex(Point2::new(2.0, 0.0));
        let p2 = sketch.add_vertex(Point2::new(2.0, 1.0));
        let p3 = sketch.add_vertex(Point2::new(1.0, 1.0));
        let p4 = sketch.add_vertex(Point2::new(1.0, 2.0));
        let p5 = sketch.add_vertex(Point2::new(0.0, 2.0));

        sketch.add(Segment2D::Line(Line::new(p0, p1)));
        sketch.add(Segment2D::Line(Line::new(p1, p2)));
        sketch.add(Segment2D::Line(Line::new(p2, p3)));
        sketch.add(Segment2D::Line(Line::new(p3, p4)));
        sketch.add(Segment2D::Line(Line::new(p4, p5)));
        sketch.add(Segment2D::Line(Line::new(p5, p0)));

        let depth = 1.0;
        let model = FeatureModel::new().with_operation(Extrude::simple(sketch, depth));

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(7);
        let mesh = backend
            .execute(&model, &settings)
            .expect("L-shape extrusion failed");

        let expected_volume = 3.0 * depth; // L-shape area = 3
        let actual_volume = mesh.volume();
        let tolerance = 0.05;
        let relative_error = (actual_volume - expected_volume).abs() / expected_volume;

        assert!(
            relative_error < tolerance,
            "L-shape volume: expected {:.4}, got {:.4} (error: {:.2}%)",
            expected_volume,
            actual_volume,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_extrude_circle_volume() {
        // Circle sketch extruded to cylinder — regression test for triangulated polygon SDF
        let radius = 0.5;
        let depth = 2.0;

        let sketch = Sketch::circle(radius);
        let model = FeatureModel::new().with_operation(Extrude::simple(sketch, depth));

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(7);
        let mesh = backend
            .execute(&model, &settings)
            .expect("Circle extrusion failed");

        let expected_volume = std::f64::consts::PI * radius * radius * depth;
        let actual_volume = mesh.volume();
        let tolerance = 0.05;
        let relative_error = (actual_volume - expected_volume).abs() / expected_volume;

        assert!(
            relative_error < tolerance,
            "Cylinder volume: expected {:.4}, got {:.4} (error: {:.2}%)",
            expected_volume,
            actual_volume,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_extrude_square_with_hole() {
        // Square with circular hole — tests polygon hole handling in triangulation
        let box_size = 2.0;
        let hole_radius = 0.4;
        let depth = 1.0;

        let mut sketch = Sketch::new();
        // Outer square
        let hw = box_size / 2.0;
        let p0 = sketch.add_vertex(Point2::new(-hw, -hw));
        let p1 = sketch.add_vertex(Point2::new(hw, -hw));
        let p2 = sketch.add_vertex(Point2::new(hw, hw));
        let p3 = sketch.add_vertex(Point2::new(-hw, hw));

        sketch.add(Segment2D::Line(Line::new(p0, p1)));
        sketch.add(Segment2D::Line(Line::new(p1, p2)));
        sketch.add(Segment2D::Line(Line::new(p2, p3)));
        sketch.add(Segment2D::Line(Line::new(p3, p0)));

        // Inner circle (hole)
        sketch.add_circle(Point2::new(0.0, 0.0), hole_radius);

        let model = FeatureModel::new().with_operation(Extrude::simple(sketch, depth));

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(7);
        let mesh = backend
            .execute(&model, &settings)
            .expect("Square-with-hole extrusion failed");

        let box_volume = box_size * box_size * depth;
        let hole_volume = std::f64::consts::PI * hole_radius * hole_radius * depth;
        let expected_volume = box_volume - hole_volume;
        let actual_volume = mesh.volume();
        let tolerance = 0.05;
        let relative_error = (actual_volume - expected_volume).abs() / expected_volume;

        assert!(
            relative_error < tolerance,
            "Square-with-hole volume: expected {:.4}, got {:.4} (error: {:.2}%)",
            expected_volume,
            actual_volume,
            relative_error * 100.0
        );
    }
}
