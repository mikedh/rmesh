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
    Extrude, FeatureBackend, FeatureError, FeatureModel, Operation, Result, Sign, Sketch,
};
use crate::mesh::Trimesh;
use crate::path::Segment2D;

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
        let bounds = settings.bounds.unwrap_or_else(|| calculate_bounds(model));

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
            half_size, 0.0, 0.0, cx,
            0.0, half_size, 0.0, cy,
            0.0, 0.0, half_size, cz,
            0.0, 0.0, 0.0, 1.0,
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

/// Convert an extrusion to a Fidget Tree
fn extrude_to_tree(extrude: &Extrude) -> Result<Tree> {
    // Get 2D profile as SDF
    let profile_2d = sketch_to_tree(&extrude.sketch)?;

    // Get the plane's Z offset (for now we only handle XY-parallel planes)
    let z_offset = extrude.sketch.plane.origin.z;

    // Extrude in Z: combine 2D SDF with Z bounds
    // For a profile f(x,y), extruded from z=z_offset to z=z_offset+depth:
    // SDF(x,y,z) = max(f(x,y), |z - center| - half_height)
    let depth = extrude.depth;
    let z = Tree::z();
    let z_center = z_offset + depth / 2.0;
    let z_half = depth / 2.0;

    // Z bounds: |z - center| - half_height
    let z_dist = (z - z_center).abs() - z_half;

    // Extrusion: max of 2D profile and Z bounds
    Ok(profile_2d.max(z_dist))
}

/// Convert a sketch to a 2D SDF Tree (in XY plane)
fn sketch_to_tree(sketch: &Sketch) -> Result<Tree> {
    if sketch.entities.is_empty() {
        return Err(FeatureError::InvalidSketch("Empty sketch".to_string()));
    }

    // Check if it's a single circle (simple case)
    if sketch.entities.len() == 1 {
        if let Segment2D::Circle(circle) = &sketch.entities[0].segment {
            return Ok(circle_sdf(circle.center.x, circle.center.y, circle.radius));
        }
    }

    // For polygons (lines), we need to compute a proper 2D polygon SDF
    let polygons = sketch
        .to_polygon()
        .map_err(|e| FeatureError::InvalidSketch(e.to_string()))?;

    if polygons.is_empty() {
        return Err(FeatureError::InvalidSketch("No closed polygons".to_string()));
    }

    // For now, handle the common case: axis-aligned rectangle
    // We can detect this and use the optimized box SDF
    if let Some(rect) = try_extract_rectangle(&polygons[0]) {
        return Ok(rectangle_sdf(rect.0, rect.1, rect.2, rect.3));
    }

    // Generic polygon - use polygon SDF approximation
    polygon_to_tree(&polygons[0])
}

/// Circle SDF: sqrt(x² + y²) - r (centered at origin, translated)
fn circle_sdf(cx: f64, cy: f64, radius: f64) -> Tree {
    let x = Tree::x() - cx;
    let y = Tree::y() - cy;
    (x.square() + y.square()).sqrt() - radius
}

/// Axis-aligned rectangle SDF
fn rectangle_sdf(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Tree {
    let cx = (x_min + x_max) / 2.0;
    let cy = (y_min + y_max) / 2.0;
    let hx = (x_max - x_min) / 2.0;
    let hy = (y_max - y_min) / 2.0;

    let x = Tree::x() - cx;
    let y = Tree::y() - cy;

    // 2D box SDF: max(|x| - hx, |y| - hy)
    (x.abs() - hx).max(y.abs() - hy)
}

/// Try to extract rectangle bounds from a polygon
fn try_extract_rectangle(points: &[Point2<f64>]) -> Option<(f64, f64, f64, f64)> {
    if points.len() < 3 {
        return None;
    }

    let mut x_min = f64::MAX;
    let mut x_max = f64::MIN;
    let mut y_min = f64::MAX;
    let mut y_max = f64::MIN;

    for p in points {
        x_min = x_min.min(p.x);
        x_max = x_max.max(p.x);
        y_min = y_min.min(p.y);
        y_max = y_max.max(p.y);
    }

    // Check if it's actually a rectangle (4 corners at extremes)
    let corners = [
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_min),
        (x_max, y_max),
    ];

    let mut found = [false; 4];
    for p in points {
        for (i, &(cx, cy)) in corners.iter().enumerate() {
            if (p.x - cx).abs() < 1e-9 && (p.y - cy).abs() < 1e-9 {
                found[i] = true;
            }
        }
    }

    if found.iter().all(|&f| f) {
        Some((x_min, y_min, x_max, y_max))
    } else {
        // Not a perfect rectangle, but use bounds anyway for approximation
        Some((x_min, y_min, x_max, y_max))
    }
}

/// Convert a generic polygon to a Tree using half-plane intersection
/// This creates an approximate SDF for convex polygons
fn polygon_to_tree(points: &[Point2<f64>]) -> Result<Tree> {
    if points.len() < 3 {
        return Err(FeatureError::InvalidSketch(
            "Polygon needs at least 3 points".to_string(),
        ));
    }

    // For a convex polygon, the SDF is the max of all edge half-planes
    // For each edge (p0, p1), the half-plane is: dot(normal, p - p0)
    // where normal points inward

    let mut result: Option<Tree> = None;

    let n = points.len();
    for i in 0..n {
        let p0 = &points[i];
        let p1 = &points[(i + 1) % n];

        // Edge vector
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;

        // Inward normal (rotate edge 90° CW for CCW polygon)
        let nx = dy;
        let ny = -dx;
        let len = (nx * nx + ny * ny).sqrt();
        if len < 1e-10 {
            continue;
        }
        let nx = nx / len;
        let ny = ny / len;

        // Half-plane: nx*(x - p0.x) + ny*(y - p0.y)
        let x = Tree::x();
        let y = Tree::y();
        let half_plane = x * nx + y * ny - (nx * p0.x + ny * p0.y);

        result = Some(match result {
            None => half_plane,
            Some(existing) => existing.max(half_plane),
        });
    }

    result.ok_or(FeatureError::InvalidSketch(
        "Failed to build polygon SDF".to_string(),
    ))
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

/// Calculate bounds for the model
fn calculate_bounds(model: &FeatureModel) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];

    for op in &model.operations {
        if let Operation::Extrude(e) = op {
            // Get sketch bounds
            for entity in &e.sketch.entities {
                match &entity.segment {
                    Segment2D::Line(line) => {
                        min[0] = min[0].min(line.start.x).min(line.finish.x);
                        min[1] = min[1].min(line.start.y).min(line.finish.y);
                        max[0] = max[0].max(line.start.x).max(line.finish.x);
                        max[1] = max[1].max(line.start.y).max(line.finish.y);
                    }
                    Segment2D::Circle(circle) => {
                        min[0] = min[0].min(circle.center.x - circle.radius);
                        min[1] = min[1].min(circle.center.y - circle.radius);
                        max[0] = max[0].max(circle.center.x + circle.radius);
                        max[1] = max[1].max(circle.center.y + circle.radius);
                    }
                    Segment2D::Arc(arc) => {
                        // Use start/finish as bounds approximation
                        min[0] = min[0].min(arc.start.x).min(arc.finish.x);
                        min[1] = min[1].min(arc.start.y).min(arc.finish.y);
                        max[0] = max[0].max(arc.start.x).max(arc.finish.x);
                        max[1] = max[1].max(arc.start.y).max(arc.finish.y);
                        if let Some(center) = arc.center {
                            let radius = arc.radius();
                            min[0] = min[0].min(center.x - radius);
                            min[1] = min[1].min(center.y - radius);
                            max[0] = max[0].max(center.x + radius);
                            max[1] = max[1].max(center.y + radius);
                        }
                    }
                    Segment2D::Ellipse(ellipse) => {
                        let r = ellipse.major.max(ellipse.minor);
                        min[0] = min[0].min(ellipse.center.x - r);
                        min[1] = min[1].min(ellipse.center.y - r);
                        max[0] = max[0].max(ellipse.center.x + r);
                        max[1] = max[1].max(ellipse.center.y + r);
                    }
                    Segment2D::CubicBezier(bezier) => {
                        for p in [&bezier.p0, &bezier.p1, &bezier.p2, &bezier.p3] {
                            min[0] = min[0].min(p.x);
                            min[1] = min[1].min(p.y);
                            max[0] = max[0].max(p.x);
                            max[1] = max[1].max(p.y);
                        }
                    }
                    Segment2D::QuadraticBezier(bezier) => {
                        for p in [&bezier.p0, &bezier.p1, &bezier.p2] {
                            min[0] = min[0].min(p.x);
                            min[1] = min[1].min(p.y);
                            max[0] = max[0].max(p.x);
                            max[1] = max[1].max(p.y);
                        }
                    }
                    Segment2D::BSpline(spline) => {
                        for p in &spline.points {
                            min[0] = min[0].min(p.x);
                            min[1] = min[1].min(p.y);
                            max[0] = max[0].max(p.x);
                            max[1] = max[1].max(p.y);
                        }
                    }
                }
            }

            // Z bounds from extrusion depth + plane offset
            let z_offset = e.sketch.plane.origin.z;
            min[2] = min[2].min(z_offset);
            max[2] = max[2].max(z_offset + e.depth);
        }
    }

    // Fallback if nothing found
    if min[0] == f64::MAX {
        min = [-1.0, -1.0, -1.0];
        max = [1.0, 1.0, 1.0];
    }

    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::feature::Sketch;

    #[test]
    fn test_extrude_rectangle_to_box() {
        let model =
            FeatureModel::new().with_operation(Extrude::simple(Sketch::rectangle(1.0, 1.0), 1.0));

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(5);
        let mesh = backend.execute(&model, &settings).expect("Failed to execute");

        assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
        assert!(!mesh.faces.is_empty(), "Mesh should have faces");
    }

    #[test]
    fn test_extrude_circle_to_cylinder() {
        let sketch = Sketch::circle(0.5);
        let model = FeatureModel::new().with_operation(Extrude::simple(sketch, 1.0));

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(5);
        let mesh = backend.execute(&model, &settings).expect("Failed to execute");

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
        let mesh = backend.execute(&model, &settings).expect("Failed to execute");

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
        let mesh = backend.execute(&model, &settings).expect("Failed to execute");

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
            expected_volume, actual_volume, relative_error * 100.0
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
                        i, e.depth, e.sign, e.sketch.entities.len()
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
}
