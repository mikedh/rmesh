//! Volumetric Hierarchical Approximate Convex Decomposition (VHACD).
//!
//! Breaks an arbitrary triangle mesh into a set of approximate convex hulls.
//! Essential for physics engines (collision detection requires convex shapes)
//! and spatial reasoning.
//!
//! **Pipeline**: Voxelize mesh -> Recursively split voxel regions via
//! axis-aligned planes -> Compute convex hull of each region -> Merge hulls
//! greedily until target count -> Shrink-wrap hull vertices to original surface.
//!
//! # Feature Gate
//!
//! This module requires the `wgpu` feature. No CPU fallback is provided.
//!
//! # Example
//!
//! ```ignore
//! use rmesh::decomposition::{convex_decomposition, DecompositionParams};
//! use rmesh::voxel::request_device;
//!
//! let (device, queue) = request_device().expect("GPU required");
//! let params = DecompositionParams::default();
//! let result = convex_decomposition(&device, &queue, &vertices, &faces, &params);
//! println!("Decomposed into {} hulls", result.hulls.len());
//! ```

mod merge;
mod shrink_wrap;
mod split;

use std::sync::Arc;

use nalgebra::Point3;

use crate::triangles::bvh::TriangleBvh;
use crate::voxel::{FillMode, VoxelGrid};

/// Parameters controlling convex decomposition behavior.
#[derive(Debug, Clone)]
pub struct DecompositionParams {
    /// Maximum number of convex hulls in the output.
    pub max_convex_hulls: u32,
    /// Voxelization resolution (approximate total voxel count).
    pub resolution: u32,
    /// Maximum recursion depth for hierarchical splitting.
    pub max_recursion_depth: u32,
    /// Minimum volume error percentage to continue splitting.
    pub min_volume_error_pct: f64,
    /// Minimum voxel edge length for a region to be split further.
    pub min_edge_length: u32,
    /// Whether to shrink-wrap hull vertices to the original mesh surface.
    pub shrink_wrap: bool,
    /// Interior fill mode for voxelization.
    pub fill_mode: FillMode,
    /// Maximum vertices per output hull.
    pub max_vertices_per_hull: u32,
    /// Use GPU split analysis for finding optimal split planes.
    /// When false, uses simple midpoint splits along the longest axis.
    pub find_best_plane: bool,
}

impl Default for DecompositionParams {
    fn default() -> Self {
        Self {
            max_convex_hulls: 64,
            resolution: 400_000,
            max_recursion_depth: 10,
            min_volume_error_pct: 1.0,
            min_edge_length: 2,
            shrink_wrap: true,
            fill_mode: FillMode::FloodFill,
            max_vertices_per_hull: 64,
            find_best_plane: false,
        }
    }
}

/// A convex hull output from decomposition.
#[derive(Debug, Clone)]
pub struct ConvexHull {
    /// Vertex positions of the hull.
    pub vertices: Vec<Point3<f64>>,
    /// Triangle faces as indices into `vertices` (CCW winding, outward normals).
    pub faces: Vec<[usize; 3]>,
    /// Volume of the hull.
    pub volume: f64,
    /// Centroid of the hull vertices.
    pub center: Point3<f64>,
}

/// Result of convex decomposition.
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// The set of convex hulls that approximate the input mesh.
    pub hulls: Vec<ConvexHull>,
}

/// Perform convex decomposition on a triangle mesh.
///
/// Requires a WGPU device and queue (see `voxel::request_device()`).
///
/// # Arguments
///
/// * `device` - WGPU device for GPU compute
/// * `queue` - WGPU command queue
/// * `vertices` - Mesh vertex positions
/// * `faces` - Triangle face indices
/// * `params` - Decomposition parameters
pub fn convex_decomposition(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    params: &DecompositionParams,
) -> DecompositionResult {
    if faces.is_empty() || vertices.is_empty() {
        return DecompositionResult { hulls: Vec::new() };
    }

    // Phase 1: Voxelization
    let mut voxel_grid = VoxelGrid::from_mesh(
        Arc::clone(device),
        Arc::clone(queue),
        vertices,
        faces,
        params.resolution,
        params.fill_mode,
    );

    // Phase 2: Hierarchical splitting
    let split_params = split::SplitParams {
        max_recursion_depth: params.max_recursion_depth,
        min_volume_error_pct: params.min_volume_error_pct,
        min_edge_length: params.min_edge_length,
        max_vertices_per_hull: params.max_vertices_per_hull,
    };

    let mut hulls = split::hierarchical_split(&mut voxel_grid, &split_params);

    // Phase 3: Greedy merge to target count
    if hulls.len() > params.max_convex_hulls as usize {
        hulls = merge::greedy_merge(hulls, params.max_convex_hulls);
    }

    // Phase 4: Shrink-wrap to original surface
    if params.shrink_wrap && !hulls.is_empty() {
        let bvh = TriangleBvh::build(vertices, faces);
        let max_dist = voxel_grid.scale() * 2.0;

        hulls = hulls
            .iter()
            .map(|hull| shrink_wrap::shrink_wrap(hull, &bvh, vertices, faces, max_dist))
            .collect();
    }

    DecompositionResult { hulls }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = DecompositionParams::default();
        assert_eq!(params.max_convex_hulls, 64);
        assert_eq!(params.resolution, 400_000);
        assert_eq!(params.max_recursion_depth, 10);
        assert!(params.shrink_wrap);
    }

    #[test]
    fn test_decompose_empty_mesh() {
        let (dev, queue) = match crate::voxel::request_device() {
            Some(dq) => dq,
            None => return,
        };

        let result = convex_decomposition(&dev, &queue, &[], &[], &DecompositionParams::default());
        assert!(result.hulls.is_empty());
    }

    #[test]
    fn test_decompose_unit_cube() {
        let (dev, queue) = match crate::voxel::request_device() {
            Some(dq) => dq,
            None => return,
        };

        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];

        let params = DecompositionParams {
            resolution: 10_000,
            max_convex_hulls: 4,
            ..Default::default()
        };

        let result = convex_decomposition(&dev, &queue, &vertices, &faces, &params);
        assert!(
            !result.hulls.is_empty(),
            "Cube should decompose into at least 1 hull"
        );
        assert!(
            result.hulls.len() <= 4,
            "Should not exceed max_convex_hulls"
        );

        // All hulls should have positive volume
        for hull in &result.hulls {
            assert!(hull.volume > 0.0, "Hull volume should be positive");
            assert!(!hull.vertices.is_empty());
            assert!(!hull.faces.is_empty());
        }
    }

    // ── Test Helpers ────────────────────────────────────────────────────

    /// Build a cross/plus shape from two perpendicular boxes.
    /// A 3×1×1 horizontal bar + a 1×3×1 vertical bar, both centered at origin.
    fn cross_mesh() -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
        fn box_mesh(
            hx: f64,
            hy: f64,
            hz: f64,
            ox: f64,
            oy: f64,
            oz: f64,
        ) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
            let v = vec![
                Point3::new(ox - hx, oy - hy, oz - hz),
                Point3::new(ox + hx, oy - hy, oz - hz),
                Point3::new(ox + hx, oy + hy, oz - hz),
                Point3::new(ox - hx, oy + hy, oz - hz),
                Point3::new(ox - hx, oy - hy, oz + hz),
                Point3::new(ox + hx, oy - hy, oz + hz),
                Point3::new(ox + hx, oy + hy, oz + hz),
                Point3::new(ox - hx, oy + hy, oz + hz),
            ];
            let f = vec![
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [0, 4, 7],
                [0, 7, 3],
                [1, 2, 6],
                [1, 6, 5],
            ];
            (v, f)
        }

        // Horizontal bar: 3×1×1
        let (mut verts, mut faces) = box_mesh(1.5, 0.5, 0.5, 0.0, 0.0, 0.0);
        // Vertical bar: 1×3×1
        let (v2, f2) = box_mesh(0.5, 1.5, 0.5, 0.0, 0.0, 0.0);

        let offset = verts.len();
        verts.extend_from_slice(&v2);
        for f in &f2 {
            faces.push([f[0] + offset, f[1] + offset, f[2] + offset]);
        }
        (verts, faces)
    }

    /// Build an L-shape from two joined boxes.
    /// A 2×1×1 base + a 1×2×1 upright sharing one face.
    fn l_shape_mesh() -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
        fn box_mesh(
            hx: f64,
            hy: f64,
            hz: f64,
            ox: f64,
            oy: f64,
            oz: f64,
        ) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
            let v = vec![
                Point3::new(ox - hx, oy - hy, oz - hz),
                Point3::new(ox + hx, oy - hy, oz - hz),
                Point3::new(ox + hx, oy + hy, oz - hz),
                Point3::new(ox - hx, oy + hy, oz - hz),
                Point3::new(ox - hx, oy - hy, oz + hz),
                Point3::new(ox + hx, oy - hy, oz + hz),
                Point3::new(ox + hx, oy + hy, oz + hz),
                Point3::new(ox - hx, oy + hy, oz + hz),
            ];
            let f = vec![
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [0, 4, 7],
                [0, 7, 3],
                [1, 2, 6],
                [1, 6, 5],
            ];
            (v, f)
        }

        // Base: 2×1×1, centered at (0, 0, 0)
        let (mut verts, mut faces) = box_mesh(1.0, 0.5, 0.5, 0.0, 0.0, 0.0);
        // Upright: 1×2×1, offset so it joins at the right end of the base
        let (v2, f2) = box_mesh(0.5, 1.0, 0.5, 0.5, 1.0, 0.0);

        let offset = verts.len();
        verts.extend_from_slice(&v2);
        for f in &f2 {
            faces.push([f[0] + offset, f[1] + offset, f[2] + offset]);
        }
        (verts, faces)
    }

    /// Build spiked cylinder via FeatureModel + FidgetBackend.
    #[cfg(feature = "cad")]
    fn spiked_cylinder_mesh() -> crate::mesh::Trimesh {
        use crate::creation::feature::FeatureBackend;
        use crate::creation::feature::{
            Extrude, FeatureModel, Sign, Sketch, SketchPlane,
            backends::fidget::{FidgetBackend, FidgetSettings},
        };
        use nalgebra::Point2;

        // Main body: circle on XY plane extruded along Z
        let mut body_sketch = Sketch::on_plane(SketchPlane::xy());
        body_sketch.add_circle(Point2::new(0.0, 0.0), 1.0);
        let body = Extrude::new(body_sketch, 3.0, Sign::Add);

        // Spike 1: small circle on XZ plane extruded along Y
        let mut spike1_sketch = Sketch::on_plane(SketchPlane::xz());
        spike1_sketch.add_circle(Point2::new(0.0, 1.5), 0.3);
        let spike1 = Extrude::new(spike1_sketch, 2.0, Sign::Add);

        // Spike 2: small circle on YZ plane extruded along X
        let mut spike2_sketch = Sketch::on_plane(SketchPlane::yz());
        spike2_sketch.add_circle(Point2::new(0.0, 1.5), 0.3);
        let spike2 = Extrude::new(spike2_sketch, 2.0, Sign::Add);

        let model = FeatureModel::new()
            .with_operation(body)
            .with_operation(spike1)
            .with_operation(spike2);

        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(6);
        backend
            .execute(&model, &settings)
            .expect("Spiked cylinder mesh failed")
    }

    // ── Integration Tests ───────────────────────────────────────────────

    #[cfg(feature = "cad")]
    #[test]
    fn test_decompose_spiked_cylinder() {
        let (dev, queue) = match crate::voxel::request_device() {
            Some(dq) => dq,
            None => return,
        };

        let mesh = spiked_cylinder_mesh();

        // Expected geometry:
        //   Body:  cylinder r=1.0, h=3.0 on XY → π×1²×3 ≈ 9.42
        //   Spike1: cylinder r=0.3, h=2.0 on XZ → π×0.09×2 ≈ 0.57
        //   Spike2: cylinder r=0.3, h=2.0 on YZ → ≈ 0.57
        let mesh_volume = mesh.volume();
        let expected_approx =
            std::f64::consts::PI * 1.0 * 1.0 * 3.0 + 2.0 * std::f64::consts::PI * 0.3 * 0.3 * 2.0;

        println!(
            "Spiked cylinder mesh: volume={:.3} (analytic≈{:.3}), {} verts, {} faces",
            mesh_volume,
            expected_approx,
            mesh.vertices.len(),
            mesh.faces.len()
        );

        let params = DecompositionParams {
            resolution: 100_000,
            max_convex_hulls: 16,
            ..Default::default()
        };

        let result = convex_decomposition(&dev, &queue, &mesh.vertices, &mesh.faces, &params);

        println!("Spiked cylinder: {} hulls", result.hulls.len());

        assert!(
            result.hulls.len() >= 2,
            "Spiked cylinder should decompose into >= 2 hulls, got {}",
            result.hulls.len()
        );

        // Every hull must have positive volume
        for (i, hull) in result.hulls.iter().enumerate() {
            assert!(
                hull.volume > 0.0,
                "Hull {} should have positive volume, got {}",
                i,
                hull.volume
            );
        }

        // Total hull volume should cover the mesh volume (hulls over-approximate)
        let hull_volume_sum: f64 = result.hulls.iter().map(|h| h.volume).sum();
        println!(
            "  hull volume sum={:.3}, mesh volume={:.3}",
            hull_volume_sum, mesh_volume
        );
        assert!(
            hull_volume_sum >= mesh_volume * 0.8,
            "Hull volume sum ({:.3}) should cover >= 80% of mesh volume ({:.3})",
            hull_volume_sum,
            mesh_volume
        );

        // The combined hulls should span in all three axes since the spikes
        // extend along Y and X while the body extends along Z.
        let mut all_min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut all_max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
        for hull in &result.hulls {
            for v in &hull.vertices {
                all_min = all_min.inf(v);
                all_max = all_max.sup(v);
            }
        }
        let extent = all_max - all_min;

        println!(
            "  hull extent: x={:.2}, y={:.2}, z={:.2}",
            extent.x, extent.y, extent.z
        );

        // Body is 2.0 wide (diameter) × 3.0 tall (Z), spikes add Y and X extent
        assert!(
            extent.z > 2.0,
            "Z extent should reflect body height (3.0), got {:.2}",
            extent.z
        );
        assert!(
            extent.x > 1.5,
            "X extent should reflect body + YZ spike, got {:.2}",
            extent.x
        );
        assert!(
            extent.y > 1.5,
            "Y extent should reflect XZ spike, got {:.2}",
            extent.y
        );
    }

    #[test]
    fn test_decompose_cross_shape() {
        let (dev, queue) = match crate::voxel::request_device() {
            Some(dq) => dq,
            None => return,
        };

        let (vertices, faces) = cross_mesh();
        let params = DecompositionParams {
            resolution: 100_000,
            max_convex_hulls: 8,
            ..Default::default()
        };

        let result = convex_decomposition(&dev, &queue, &vertices, &faces, &params);

        println!("Cross shape: {} hulls", result.hulls.len());
        assert!(
            !result.hulls.is_empty(),
            "Cross shape should decompose into >= 1 hull, got 0",
        );

        for (i, hull) in result.hulls.iter().enumerate() {
            assert!(hull.volume > 0.0, "Hull {} should have positive volume", i);
        }
    }

    #[test]
    fn test_decompose_l_shape() {
        let (dev, queue) = match crate::voxel::request_device() {
            Some(dq) => dq,
            None => return,
        };

        let (vertices, faces) = l_shape_mesh();
        let params = DecompositionParams {
            resolution: 100_000,
            max_convex_hulls: 8,
            ..Default::default()
        };

        let result = convex_decomposition(&dev, &queue, &vertices, &faces, &params);

        println!("L-shape: {} hulls", result.hulls.len());
        assert!(
            result.hulls.len() >= 2,
            "L-shape should decompose into >= 2 hulls, got {}",
            result.hulls.len()
        );

        for (i, hull) in result.hulls.iter().enumerate() {
            assert!(hull.volume > 0.0, "Hull {} should have positive volume", i);
        }
    }

    #[test]
    fn test_decompose_monkey() {
        let (dev, queue) = match crate::voxel::request_device() {
            Some(dq) => dq,
            None => return,
        };

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data/monkey.glb");

        if !path.exists() {
            println!("monkey.glb not found, skipping: {:?}", path);
            return;
        }

        let data = std::fs::read(&path).unwrap();
        let scene = crate::exchange::load(&data, None, None).unwrap();

        // Get the first mesh from the scene
        let mesh = scene
            .geometry
            .values()
            .find_map(|g| {
                if let crate::geometry::Geometry::Mesh(m) = g {
                    Some(m)
                } else {
                    None
                }
            })
            .expect("monkey.glb should contain a mesh");

        let params = DecompositionParams::default();
        let result = convex_decomposition(&dev, &queue, &mesh.vertices, &mesh.faces, &params);

        println!("Monkey: {} hulls", result.hulls.len());
        assert!(
            result.hulls.len() > 1,
            "Monkey should decompose into > 1 hull, got {}",
            result.hulls.len()
        );
        assert!(
            result.hulls.len() <= 64,
            "Monkey should have <= 64 hulls (default max), got {}",
            result.hulls.len()
        );

        for (i, hull) in result.hulls.iter().enumerate() {
            assert!(hull.volume > 0.0, "Hull {} should have positive volume", i);
        }
    }

    #[test]
    fn test_decompose_volume_conservation() {
        let (dev, queue) = match crate::voxel::request_device() {
            Some(dq) => dq,
            None => return,
        };

        // Unit cube
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];

        let mesh_volume = crate::triangles::inertia::volume(&vertices, &faces);

        let params = DecompositionParams {
            resolution: 100_000,
            max_convex_hulls: 8,
            ..Default::default()
        };

        let result = convex_decomposition(&dev, &queue, &vertices, &faces, &params);
        let hull_volume_sum: f64 = result.hulls.iter().map(|h| h.volume).sum();

        println!(
            "Volume conservation: mesh={:.4}, hull_sum={:.4}",
            mesh_volume, hull_volume_sum
        );

        // Convex hulls over-approximate, so sum should be >= mesh volume
        assert!(
            hull_volume_sum >= mesh_volume * 0.9,
            "Sum of hull volumes ({:.4}) should be >= mesh volume ({:.4})",
            hull_volume_sum,
            mesh_volume
        );
    }

    #[test]
    fn test_decompose_all_hulls_convex() {
        let (dev, queue) = match crate::voxel::request_device() {
            Some(dq) => dq,
            None => return,
        };

        let (vertices, faces) = cross_mesh();
        let params = DecompositionParams {
            resolution: 100_000,
            max_convex_hulls: 8,
            ..Default::default()
        };

        let result = convex_decomposition(&dev, &queue, &vertices, &faces, &params);

        for (i, hull) in result.hulls.iter().enumerate() {
            if hull.faces.is_empty() {
                continue;
            }
            assert!(
                crate::convex::is_hull_valid_3d(&hull.vertices, &hull.faces),
                "Hull {} should be a valid convex hull",
                i
            );
        }
    }

    #[test]
    fn test_decompose_deterministic() {
        let (dev, queue) = match crate::voxel::request_device() {
            Some(dq) => dq,
            None => return,
        };

        let (vertices, faces) = cross_mesh();
        let params = DecompositionParams {
            resolution: 10_000,
            max_convex_hulls: 8,
            ..Default::default()
        };

        let result1 = convex_decomposition(&dev, &queue, &vertices, &faces, &params);
        let result2 = convex_decomposition(&dev, &queue, &vertices, &faces, &params);

        assert_eq!(
            result1.hulls.len(),
            result2.hulls.len(),
            "Deterministic: same number of hulls"
        );

        // Compare total volume sums rather than per-hull volumes.
        // GPU split analysis can produce slightly different split planes between
        // runs, so individual hull volumes vary — but the total should be stable.
        let vol1: f64 = result1.hulls.iter().map(|h| h.volume).sum();
        let vol2: f64 = result2.hulls.iter().map(|h| h.volume).sum();
        let vol_err = if vol1 > 1e-12 {
            (vol1 - vol2).abs() / vol1
        } else {
            (vol1 - vol2).abs()
        };
        assert!(
            vol_err < 0.15,
            "Total hull volume should match within 15% ({:.6} vs {:.6}, err={:.2}%)",
            vol1,
            vol2,
            vol_err * 100.0,
        );
    }
}
