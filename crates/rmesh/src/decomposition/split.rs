//! Hierarchical splitting of voxel regions via distance-field concavity detection.
//!
//! Uses a GPU split analysis shader to find "necks" (concavities) in each region,
//! then splits regions at those points. After splitting completes, extracts surface
//! voxels per region and computes convex hulls on CPU.

use std::sync::Arc;

use ahash::AHashMap;
use nalgebra::Point3;
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::voxel::VoxelGrid;

use super::ConvexHull;

/// A split decision read back from the GPU.
#[derive(Debug, Clone, Copy)]
struct SplitDecision {
    axis: u32,
    position: u32,
    neck_depth: f32,
    region_id: u32,
}

/// GPU-side split decision for upload to the region update shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSplitDecision {
    parent_region: u32,
    child_left: u32,
    child_right: u32,
    axis: u32,
    position: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// GPU-side split analysis result.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSplitResult {
    axis: u32,
    position: u32,
    neck_depth: f32,
    region_id: u32,
}

/// Parameters controlling the splitting process.
pub struct SplitParams {
    pub max_recursion_depth: u32,
    pub min_volume_error_pct: f64,
    pub min_edge_length: u32,
    pub max_vertices_per_hull: u32,
}

/// Run hierarchical splitting on a voxel grid.
///
/// Returns a list of convex hulls, one per final region.
pub fn hierarchical_split(grid: &mut VoxelGrid, params: &SplitParams) -> Vec<ConvexHull> {
    let device = Arc::clone(grid.device());
    let queue = Arc::clone(grid.queue());
    let dims = grid.dims();
    let total_voxels = grid.total_voxels();
    let voxel_scale = grid.scale();

    // Read grid data and extract scale before mutable borrow for distance_field
    let grid_data = grid.to_array();

    // Compute the distance field (cached by VoxelGrid) - takes &mut self
    let distance_buf = grid.distance_field();

    // Create region_ids buffer (all 0 = single region)
    let region_ids_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("region_ids"),
        size: total_voxels * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Initialize region_ids: set surface/inside voxels to region 0,
    // outside/undefined voxels to u32::MAX (no region).
    let region_init: Vec<u32> = grid_data
        .iter()
        .map(|v| {
            use crate::voxel::VoxelValue;
            match v {
                VoxelValue::Surface | VoxelValue::Inside => 0,
                _ => u32::MAX,
            }
        })
        .collect();
    queue.write_buffer(&region_ids_buf, 0, bytemuck::cast_slice(&region_init));

    let mut active_regions: Vec<u32> = vec![0];
    let mut next_region_id: u32 = 1;
    let mut completed_regions: Vec<u32> = Vec::new();

    // Load shaders
    let analysis_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cs_split_analysis"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cs_split_analysis.wgsl").into()),
    });

    let update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cs_region_update"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cs_region_update.wgsl").into()),
    });

    // Analysis pipeline
    let analysis_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("split_analysis_bgl"),
        entries: &[
            bgl_uniform(0),
            bgl_storage_ro(1),
            bgl_storage_ro(2),
            bgl_storage_ro(3),
            bgl_storage_rw(4),
        ],
    });

    let analysis_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("split_analysis_pl"),
        bind_group_layouts: &[&analysis_bgl],
        immediate_size: 0,
    });

    let analysis_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("split_analysis_pipeline"),
        layout: Some(&analysis_pl),
        module: &analysis_shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Update pipeline
    let update_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("region_update_bgl"),
        entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
    });

    let update_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("region_update_pl"),
        bind_group_layouts: &[&update_bgl],
        immediate_size: 0,
    });

    let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("region_update_pipeline"),
        layout: Some(&update_pl),
        module: &update_shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let wg = [
        dims[0].div_ceil(4),
        dims[1].div_ceil(4),
        dims[2].div_ceil(4),
    ];

    #[allow(clippy::cast_possible_truncation)]
    let min_neck_depth = params.min_volume_error_pct as f32 * voxel_scale as f32;

    // Iterative splitting
    for _depth in 0..params.max_recursion_depth {
        if active_regions.is_empty() {
            break;
        }

        let num_active = active_regions.len();

        // Upload active regions
        let active_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("active_regions"),
            contents: bytemuck::cast_slice(&active_regions),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Results buffer
        let results_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("split_results"),
            size: (num_active * std::mem::size_of::<GpuSplitResult>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Analysis params
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct AnalysisParams {
            dims: [u32; 3],
            num_regions: u32,
            max_dim: u32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let max_dim = dims[0].max(dims[1]).max(dims[2]);
        let analysis_params = AnalysisParams {
            dims,
            #[allow(clippy::cast_possible_truncation)]
            num_regions: num_active as u32,
            max_dim,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let analysis_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("analysis_params"),
            contents: bytemuck::bytes_of(&analysis_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let analysis_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("split_analysis_bg"),
            layout: &analysis_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: analysis_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: distance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: region_ids_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: active_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        // Run analysis
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("split_analysis_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("split_analysis_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&analysis_pipeline);
            pass.set_bind_group(0, &analysis_bg, &[]);
            #[allow(clippy::cast_possible_truncation)]
            let workgroup_count = (num_active as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Readback results
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("split_readback"),
            size: (num_active * std::mem::size_of::<GpuSplitResult>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &results_buf,
            0,
            &readback,
            0,
            (num_active * std::mem::size_of::<GpuSplitResult>()) as u64,
        );
        queue.submit(Some(encoder.finish()));

        // Read back split decisions
        let data = crate::gpu::gpu_read_buffer(&device, &readback);
        let results: &[GpuSplitResult] = bytemuck::cast_slice(&data);
        let decisions: Vec<SplitDecision> = results
            .iter()
            .map(|r| SplitDecision {
                axis: r.axis,
                position: r.position,
                neck_depth: r.neck_depth,
                region_id: r.region_id,
            })
            .collect();

        // Decide which regions to split
        let mut splits = Vec::new();
        let mut new_active = Vec::new();

        for decision in &decisions {
            if decision.neck_depth < min_neck_depth || decision.position < params.min_edge_length {
                // Region is complete (no good split found)
                completed_regions.push(decision.region_id);
                continue;
            }

            // Check if split position leaves enough room on both sides
            let axis_dim = match decision.axis {
                0 => dims[0],
                1 => dims[1],
                _ => dims[2],
            };
            if decision.position < params.min_edge_length
                || axis_dim - decision.position < params.min_edge_length
            {
                completed_regions.push(decision.region_id);
                continue;
            }

            let child_left = next_region_id;
            let child_right = next_region_id + 1;
            next_region_id += 2;

            splits.push(GpuSplitDecision {
                parent_region: decision.region_id,
                child_left,
                child_right,
                axis: decision.axis,
                position: decision.position,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            });

            new_active.push(child_left);
            new_active.push(child_right);
        }

        if splits.is_empty() {
            break;
        }

        // Upload split decisions and run region update
        let splits_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splits"),
            contents: bytemuck::cast_slice(&splits),
            usage: wgpu::BufferUsages::STORAGE,
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct UpdateParams {
            dims: [u32; 3],
            num_splits: u32,
        }

        let update_params = UpdateParams {
            dims,
            #[allow(clippy::cast_possible_truncation)]
            num_splits: splits.len() as u32,
        };
        let update_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("update_params"),
            contents: bytemuck::bytes_of(&update_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let update_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("region_update_bg"),
            layout: &update_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: update_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: splits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: region_ids_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("region_update_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("region_update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&update_pipeline);
            pass.set_bind_group(0, &update_bg, &[]);
            pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
        }
        queue.submit(Some(encoder.finish()));

        active_regions = new_active;
    }

    // Any remaining active regions are completed
    completed_regions.extend(active_regions);

    // Read back final region_ids and grid data
    let region_data = readback_buffer(&device, &queue, &region_ids_buf);

    // Extract surface voxels per region and compute hulls
    extract_hulls(grid, &region_data, &completed_regions, params)
}

/// Extract convex hulls from finalized regions.
fn extract_hulls(
    grid: &VoxelGrid,
    region_data: &[u32],
    regions: &[u32],
    params: &SplitParams,
) -> Vec<ConvexHull> {
    let dims = grid.dims();
    let dx = dims[0] as usize;
    let dy = dims[1] as usize;

    // Group voxels by region
    let mut region_voxels: AHashMap<u32, Vec<Point3<f64>>> = AHashMap::new();

    for (idx, &region_id) in region_data.iter().enumerate() {
        if region_id == u32::MAX {
            continue;
        }
        if !regions.contains(&region_id) {
            continue;
        }

        #[allow(clippy::cast_possible_truncation)]
        let x = (idx % dx) as u32;
        #[allow(clippy::cast_possible_truncation)]
        let y = ((idx / dx) % dy) as u32;
        #[allow(clippy::cast_possible_truncation)]
        let z = (idx / (dx * dy)) as u32;
        let world = grid.voxel_to_world(x, y, z);

        region_voxels.entry(region_id).or_default().push(world);
    }

    // Compute convex hulls in parallel
    let region_list: Vec<(u32, Vec<Point3<f64>>)> = region_voxels.into_iter().collect();

    region_list
        .into_par_iter()
        .filter_map(|(_region_id, points)| {
            if points.len() < 4 {
                return None;
            }

            build_hull_from_points(&points, params.max_vertices_per_hull)
        })
        .collect()
}

/// Build a ConvexHull from a set of points, with vertex count limit.
fn build_hull_from_points(points: &[Point3<f64>], max_vertices: u32) -> Option<ConvexHull> {
    // Try convex hull first
    let Ok(faces) = crate::convex::convex_hull_3d(points) else {
        // Fall back to AABB as 12-triangle box hull
        return aabb_fallback(points);
    };

    // Collect unique vertices referenced by hull faces
    let mut used = vec![false; points.len()];
    for [a, b, c] in &faces {
        used[*a] = true;
        used[*b] = true;
        used[*c] = true;
    }
    let hull_vertices: Vec<Point3<f64>> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| used[*i])
        .map(|(_, p)| *p)
        .collect();

    // If too many vertices, simplify by recomputing hull with subset
    let final_vertices;
    let final_faces;

    if hull_vertices.len() > max_vertices as usize {
        // Sample down to max_vertices by taking evenly spaced points
        let step = hull_vertices.len() / max_vertices as usize;
        let subset: Vec<Point3<f64>> = hull_vertices
            .iter()
            .step_by(step.max(1))
            .take(max_vertices as usize)
            .copied()
            .collect();
        match crate::convex::convex_hull_3d(&subset) {
            Ok(f) => {
                final_vertices = subset;
                final_faces = f;
            }
            Err(_) => return aabb_fallback(points),
        }
    } else {
        final_vertices = hull_vertices;
        // Remap face indices to the new vertex list
        let mut index_map = vec![0usize; points.len()];
        let mut count = 0;
        for (i, &u) in used.iter().enumerate() {
            if u {
                index_map[i] = count;
                count += 1;
            }
        }
        final_faces = faces
            .iter()
            .map(|[a, b, c]| [index_map[*a], index_map[*b], index_map[*c]])
            .collect();
    }

    let volume = crate::triangles::inertia::volume(&final_vertices, &final_faces).abs();
    let center = centroid(&final_vertices);

    Some(ConvexHull {
        vertices: final_vertices,
        faces: final_faces,
        volume,
        center,
    })
}

/// Build an AABB box hull as fallback for degenerate inputs.
fn aabb_fallback(points: &[Point3<f64>]) -> Option<ConvexHull> {
    if points.is_empty() {
        return None;
    }

    let mut min = points[0];
    let mut max = points[0];
    for p in points {
        min = min.inf(p);
        max = max.sup(p);
    }

    // Ensure non-degenerate box
    let eps = 1e-10;
    if (max.x - min.x) < eps || (max.y - min.y) < eps || (max.z - min.z) < eps {
        return None;
    }

    let vertices = vec![
        Point3::new(min.x, min.y, min.z),
        Point3::new(max.x, min.y, min.z),
        Point3::new(max.x, max.y, min.z),
        Point3::new(min.x, max.y, min.z),
        Point3::new(min.x, min.y, max.z),
        Point3::new(max.x, min.y, max.z),
        Point3::new(max.x, max.y, max.z),
        Point3::new(min.x, max.y, max.z),
    ];
    let faces = vec![
        [0, 2, 1],
        [0, 3, 2], // -Z
        [4, 5, 6],
        [4, 6, 7], // +Z
        [0, 1, 5],
        [0, 5, 4], // -Y
        [2, 3, 7],
        [2, 7, 6], // +Y
        [0, 4, 7],
        [0, 7, 3], // -X
        [1, 2, 6],
        [1, 6, 5], // +X
    ];

    let volume = (max.x - min.x) * (max.y - min.y) * (max.z - min.z);
    let center = Point3::from((min.coords + max.coords) * 0.5);

    Some(ConvexHull {
        vertices,
        faces,
        volume,
        center,
    })
}

fn centroid(points: &[Point3<f64>]) -> Point3<f64> {
    let sum = points
        .iter()
        .fold(nalgebra::Vector3::zeros(), |acc, p| acc + p.coords);
    Point3::from(sum / points.len() as f64)
}

// ── Buffer helpers ──────────────────────────────────────────────────────

fn readback_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let size = buffer.size();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("split_readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("split_readback_enc"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let data = crate::gpu::gpu_read_buffer(device, &staging);
    data.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_fallback() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];

        let hull = aabb_fallback(&points).unwrap();
        assert_eq!(hull.vertices.len(), 8);
        assert_eq!(hull.faces.len(), 12);
        assert!((hull.volume - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb_fallback_empty() {
        assert!(aabb_fallback(&[]).is_none());
    }

    #[test]
    fn test_build_hull_from_points() {
        let points: Vec<Point3<f64>> = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.5, 0.5, 0.5), // interior point
        ];

        let hull = build_hull_from_points(&points, 64).unwrap();
        assert!(!hull.vertices.is_empty());
        assert!(!hull.faces.is_empty());
        assert!(hull.volume > 0.0);
    }

    #[test]
    fn test_centroid() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(0.0, 0.0, 2.0),
        ];
        let c = centroid(&points);
        assert!((c.x - 0.5).abs() < 1e-10);
        assert!((c.y - 0.5).abs() < 1e-10);
        assert!((c.z - 0.5).abs() < 1e-10);
    }
}
