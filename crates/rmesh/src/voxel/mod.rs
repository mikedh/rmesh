//! GPU-accelerated voxel grid for mesh processing.
//!
//! Provides triangle mesh voxelization, flood-fill interior detection,
//! and 3D Jump Flooding Algorithm (JFA) distance field computation.
//! All operations run as WGPU compute shaders.
//!
//! # Feature Gate
//!
//! This module requires the `wgpu` feature.

use std::sync::Arc;

use nalgebra::Point3;

/// Voxel classification values matching the GPU shader conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum VoxelValue {
    Undefined = 0,
    Surface = 1,
    Inside = 2,
    Outside = 3,
}

impl VoxelValue {
    fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Surface,
            2 => Self::Inside,
            3 => Self::Outside,
            _ => Self::Undefined,
        }
    }
}

/// Fill mode for interior detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillMode {
    /// Flood fill from boundary - fast but requires watertight mesh.
    #[default]
    FloodFill,
    /// Surface only - no interior detection.
    SurfaceOnly,
    /// Raycast parity - robust for non-watertight meshes.
    Raycast,
}

/// A 3D voxel grid backed by GPU buffers.
///
/// Created from a triangle mesh via `from_mesh`. Supports distance field
/// computation (3D JFA) and CPU readback of voxel data.
pub struct VoxelGrid {
    /// Grid dimensions per axis.
    dims: [u32; 3],
    /// World-space AABB min corner.
    origin: Point3<f64>,
    /// Uniform voxel edge length in world space.
    scale: f64,
    /// GPU buffer: 4 bytes per voxel (VoxelValue as u32).
    buffer: wgpu::Buffer,
    /// Cached 3D JFA distance field (f32 per voxel), lazily computed.
    distance: Option<wgpu::Buffer>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl VoxelGrid {
    /// Create a voxel grid from a triangle mesh.
    ///
    /// `resolution` controls the total number of voxels (approximate).
    /// The grid dimension is `max(32, resolution^(1/3) * 1.5)` along the longest axis,
    /// scaled proportionally for other axes.
    pub fn from_mesh(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        vertices: &[Point3<f64>],
        faces: &[[usize; 3]],
        resolution: u32,
        fill_mode: FillMode,
    ) -> Self {
        // Compute AABB
        let (aabb_min, aabb_max) = compute_aabb(vertices);
        let extent = aabb_max - aabb_min;
        let longest = extent.x.max(extent.y).max(extent.z);

        // Grid sizing: target dim along longest axis
        let target_dim = 32_u32.max(((resolution as f64).cbrt() * 1.5) as u32);
        // Clamp to 1023 since we pack coords into 10 bits in JFA shaders
        let target_dim = target_dim.min(1023);
        let scale = longest / f64::from(target_dim);

        let dims = [
            2.max((extent.x / scale).ceil() as u32 + 2),
            2.max((extent.y / scale).ceil() as u32 + 2),
            2.max((extent.z / scale).ceil() as u32 + 2),
        ];
        // Clamp all dims to 1023
        let dims = [dims[0].min(1023), dims[1].min(1023), dims[2].min(1023)];

        // Offset origin by one voxel for boundary padding
        let origin = Point3::new(aabb_min.x - scale, aabb_min.y - scale, aabb_min.z - scale);

        let total_voxels = dims[0] as u64 * dims[1] as u64 * dims[2] as u64;

        // Create grid buffer (initialized to 0 = Undefined)
        let grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_grid"),
            size: total_voxels * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid = Self {
            dims,
            origin,
            scale,
            buffer: grid_buffer,
            distance: None,
            device,
            queue,
        };

        // Run voxelization
        grid.run_voxelize(vertices, faces);

        // Run interior detection
        match fill_mode {
            FillMode::FloodFill => grid.run_flood_fill(),
            FillMode::Raycast => grid.run_raycast_fill(vertices, faces),
            FillMode::SurfaceOnly => {}
        }

        grid
    }

    /// Grid dimensions as [x, y, z].
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// World-space origin (AABB min corner minus padding).
    pub fn origin(&self) -> Point3<f64> {
        self.origin
    }

    /// Voxel edge length in world space.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Total number of voxels.
    pub fn total_voxels(&self) -> u64 {
        self.dims[0] as u64 * self.dims[1] as u64 * self.dims[2] as u64
    }

    /// Reference to the underlying GPU grid buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Reference to the GPU device.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Reference to the GPU queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Compute or return the cached 3D JFA distance field.
    ///
    /// The distance field gives the Euclidean distance from each voxel to
    /// the nearest surface voxel, in world-space units.
    pub fn distance_field(&mut self) -> &wgpu::Buffer {
        if self.distance.is_none() {
            let buf = self.compute_distance_field();
            self.distance = Some(buf);
        }
        self.distance.as_ref().expect("just computed")
    }

    /// Read the voxel grid back to CPU.
    pub fn to_array(&self) -> Vec<VoxelValue> {
        let data = self.readback_buffer(&self.buffer);
        data.iter().map(|&v| VoxelValue::from_u32(v)).collect()
    }

    /// Read the distance field back to CPU. Computes it if not yet cached.
    pub fn distance_to_array(&mut self) -> Vec<f32> {
        self.distance_field();
        let buf = self.distance.as_ref().expect("just computed");
        let raw = self.readback_buffer_raw(buf);
        // Reinterpret u8 as f32
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Get coordinates of all surface voxels.
    pub fn surface_voxels(&self) -> Vec<[u32; 3]> {
        self.voxels_with_value(VoxelValue::Surface)
    }

    /// Get coordinates of all interior voxels.
    pub fn interior_voxels(&self) -> Vec<[u32; 3]> {
        self.voxels_with_value(VoxelValue::Inside)
    }

    /// Convert voxel coordinates to world-space point (voxel center).
    pub fn voxel_to_world(&self, x: u32, y: u32, z: u32) -> Point3<f64> {
        let half = self.scale * 0.5;
        Point3::new(
            self.origin.x + f64::from(x) * self.scale + half,
            self.origin.y + f64::from(y) * self.scale + half,
            self.origin.z + f64::from(z) * self.scale + half,
        )
    }

    /// Convert world-space point to voxel coordinates (clamped to grid).
    pub fn world_to_voxel(&self, p: &Point3<f64>) -> [u32; 3] {
        let inv = 1.0 / self.scale;
        [
            ((p.x - self.origin.x) * inv)
                .floor()
                .clamp(0.0, f64::from(self.dims[0] - 1)) as u32,
            ((p.y - self.origin.y) * inv)
                .floor()
                .clamp(0.0, f64::from(self.dims[1] - 1)) as u32,
            ((p.z - self.origin.z) * inv)
                .floor()
                .clamp(0.0, f64::from(self.dims[2] - 1)) as u32,
        ]
    }

    /// Volume and surface area computed in a single GPU pass.
    ///
    /// Returns `(volume, area)` in world-space units (cubed / squared).
    pub fn measure(&self) -> (f64, f64) {
        let (filled, faces) = self.run_measure_shader();
        let volume = filled as f64 * self.scale.powi(3);
        let area = faces as f64 * self.scale.powi(2);
        (volume, area)
    }

    /// Approximate volume of the filled voxels in world-space cubic units.
    pub fn volume(&self) -> f64 {
        self.measure().0
    }

    /// Approximate surface area from exposed voxel faces in world-space square units.
    pub fn area(&self) -> f64 {
        self.measure().1
    }

    // ── GPU Pipeline Internals ──────────────────────────────────────────

    fn run_voxelize(&self, vertices: &[Point3<f64>], faces: &[[usize; 3]]) {
        if faces.is_empty() {
            return;
        }

        // Pack triangles as 9 f32 per triangle
        let mut tri_data: Vec<f32> = Vec::with_capacity(faces.len() * 9);
        for &[i0, i1, i2] in faces {
            for &vi in &[i0, i1, i2] {
                let v = &vertices[vi];
                tri_data.push(v.x as f32);
                tri_data.push(v.y as f32);
                tri_data.push(v.z as f32);
            }
        }

        let tri_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelize_triangles"),
                contents: bytemuck::cast_slice(&tri_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Params: dims (3xu32) + num_triangles (u32) + origin (3xf32) + scale (f32)
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct VoxelizeParams {
            dims: [u32; 3],
            num_triangles: u32,
            origin: [f32; 3],
            scale: f32,
        }

        let params = VoxelizeParams {
            dims: self.dims,
            num_triangles: faces.len() as u32,
            origin: [
                self.origin.x as f32,
                self.origin.y as f32,
                self.origin.z as f32,
            ],
            scale: self.scale as f32,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelize_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_voxelize"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cs_voxelize.wgsl").into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("voxelize_bgl"),
                    entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("voxelize_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("voxelize_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxelize_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tri_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxelize_enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxelize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (faces.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    fn run_flood_fill(&self) {
        let total = self.total_voxels();

        // Create second grid buffer for ping-pong
        let grid_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flood_fill_b"),
            size: total * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Changed counter buffer
        let changed_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flood_changed"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct FloodParams {
            dims: [u32; 3],
            _pad: u32,
        }

        let params = FloodParams {
            dims: self.dims,
            _pad: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flood_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_flood_fill"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cs_flood_fill.wgsl").into()),
            });

        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("flood_bgl"),
                entries: &[
                    bgl_uniform(0),
                    bgl_storage_ro(1),
                    bgl_storage_rw(2),
                    bgl_storage_rw(3),
                ],
            });

        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("flood_pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("flood_pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let finalize_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("flood_finalize_pipeline"),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some("finalize"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        let wg = [
            self.dims[0].div_ceil(4),
            self.dims[1].div_ceil(4),
            self.dims[2].div_ceil(4),
        ];

        // Readback buffer for changed counter
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flood_readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Iterative flood fill with ping-pong
        let max_iters = (self.dims[0] + self.dims[1] + self.dims[2]) as usize;
        let mut forward = true; // true: self.buffer -> grid_b, false: grid_b -> self.buffer

        for _ in 0..max_iters {
            // Zero changed counter
            self.queue.write_buffer(&changed_buffer, 0, &[0u8; 4]);

            let (src, dst) = if forward {
                (&self.buffer, &grid_b)
            } else {
                (&grid_b, &self.buffer)
            };

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flood_bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: changed_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flood_enc"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("flood_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
            }

            encoder.copy_buffer_to_buffer(&changed_buffer, 0, &readback, 0, 4);
            self.queue.submit(Some(encoder.finish()));

            // Read changed count
            let changed = {
                let slice = readback.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                self.device.poll(wgpu::Maintain::Wait);
                let data = slice.get_mapped_range();
                let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                drop(data);
                readback.unmap();
                val
            };

            forward = !forward;

            if changed == 0 {
                break;
            }
        }

        // Ensure final result is in self.buffer
        if !forward {
            // Last write went to grid_b, copy back
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flood_copy_back"),
                });
            encoder.copy_buffer_to_buffer(&grid_b, 0, &self.buffer, 0, total * 4);
            self.queue.submit(Some(encoder.finish()));
        }

        // Run finalize pass to mark remaining Undefined as Inside.
        // The finalize entry point only uses grid_out (binding 2).
        // Bind grid_b at binding 1 (unused by finalize) to avoid
        // conflicting read-only/read-write usages on self.buffer.
        let final_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flood_finalize_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: changed_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flood_finalize_enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flood_finalize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&finalize_pipeline);
            pass.set_bind_group(0, &final_bg, &[]);
            pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn run_raycast_fill(&self, vertices: &[Point3<f64>], faces: &[[usize; 3]]) {
        // Raycast fill: for each non-surface voxel, cast a ray in +X direction
        // and count surface crossings. Odd = inside, even = outside.
        // This is a CPU fallback for non-watertight meshes.
        use crate::triangles::bvh::TriangleBvh;

        let bvh = TriangleBvh::build(vertices, faces);
        let grid_data = self.readback_buffer(&self.buffer);
        let mut result = grid_data.clone();

        let dx = self.dims[0] as usize;
        let dy = self.dims[1] as usize;

        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let idx = x as usize + y as usize * dx + z as usize * dx * dy;
                    if grid_data[idx] == 1 {
                        continue; // Surface stays Surface
                    }

                    let center = self.voxel_to_world(x, y, z);
                    let dir = nalgebra::Vector3::new(1.0, 0.0, 0.0);

                    // Count intersections
                    let mut count = 0u32;
                    let mut origin = center;
                    loop {
                        match bvh.trace_ray(&origin, &dir, vertices, faces) {
                            Some(hit) => {
                                count += 1;
                                origin = hit.point + dir * 1e-8;
                            }
                            None => break,
                        }
                    }

                    result[idx] = if count % 2 == 1 { 2 } else { 3 }; // Inside or Outside
                }
            }
        }

        // Upload result back to GPU
        self.queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&result));
    }

    fn run_measure_shader(&self) -> (u32, u32) {
        let total = self.total_voxels();

        // Copy grid to a dedicated read-only buffer to avoid usage conflicts
        // with the main grid buffer (which may retain RW tracking from prior passes).
        let grid_ro = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("measure_grid_ro"),
            size: total * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("measure_copy_enc"),
                });
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &grid_ro, 0, total * 4);
            self.queue.submit(Some(encoder.finish()));
        }

        // Counter buffer: [filled_count, exposed_face_count]
        let counters_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("measure_counters"),
            size: 8, // 2 × u32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero-initialize counters
        self.queue.write_buffer(&counters_buffer, 0, &[0u8; 8]);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct MeasureDims {
            dims: [u32; 3],
            _pad: u32,
        }

        let dims_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("measure_dims"),
                contents: bytemuck::bytes_of(&MeasureDims {
                    dims: self.dims,
                    _pad: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_measure"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cs_measure.wgsl").into()),
            });

        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("measure_bgl"),
                entries: &[bgl_storage_ro(0), bgl_storage_rw(1), bgl_uniform(2)],
            });

        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("measure_pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("measure_pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("measure_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_ro.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("measure_enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("measure_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.dims[0].div_ceil(4),
                self.dims[1].div_ceil(4),
                self.dims[2].div_ceil(4),
            );
        }

        // Readback counters
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("measure_readback"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&counters_buffer, 0, &readback, 0, 8);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let filled = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let faces = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        drop(data);
        readback.unmap();

        (filled, faces)
    }

    fn compute_distance_field(&self) -> wgpu::Buffer {
        let total = self.total_voxels();

        // Create seed buffers for JFA ping-pong
        let seeds_a = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jfa_seeds_a"),
            size: total * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let seeds_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jfa_seeds_b"),
            size: total * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let distance_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("distance_field"),
            size: total * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── JFA Init ──
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct JfaInitParams {
            dims: [u32; 3],
            _pad: u32,
        }

        let init_params = JfaInitParams {
            dims: self.dims,
            _pad: 0,
        };
        let init_params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jfa_init_params"),
                contents: bytemuck::bytes_of(&init_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let init_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_jump_init_3d"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/cs_jump_init_3d.wgsl").into(),
                ),
            });

        let init_bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("jfa_init_bgl"),
                entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
            });

        let init_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("jfa_init_pl"),
                bind_group_layouts: &[&init_bgl],
                push_constant_ranges: &[],
            });

        let init_pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jfa_init_pipeline"),
                layout: Some(&init_pl),
                module: &init_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let init_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jfa_init_bg"),
            layout: &init_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: init_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: seeds_a.as_entire_binding(),
                },
            ],
        });

        // ── JFA Passes ──
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct JfaParams {
            dims: [u32; 3],
            step_size: u32,
        }

        let jfa_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_jump_3d"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cs_jump_3d.wgsl").into()),
            });

        let jfa_bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("jfa_bgl"),
                entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
            });

        let jfa_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("jfa_pl"),
                bind_group_layouts: &[&jfa_bgl],
                push_constant_ranges: &[],
            });

        let jfa_pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jfa_pipeline"),
                layout: Some(&jfa_pl),
                module: &jfa_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // ── Distance shader ──
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct DistParams {
            dims: [u32; 3],
            scale: f32,
        }

        let dist_params = DistParams {
            dims: self.dims,
            scale: self.scale as f32,
        };
        let dist_params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dist_params"),
                contents: bytemuck::bytes_of(&dist_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let dist_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_distance_3d"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/cs_distance_3d.wgsl").into(),
                ),
            });

        let dist_bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dist_bgl"),
                entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
            });

        let dist_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dist_pl"),
                bind_group_layouts: &[&dist_bgl],
                push_constant_ranges: &[],
            });

        let dist_pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("dist_pipeline"),
                layout: Some(&dist_pl),
                module: &dist_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let wg = [
            self.dims[0].div_ceil(4),
            self.dims[1].div_ceil(4),
            self.dims[2].div_ceil(4),
        ];

        // Build encoder for full JFA pipeline
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jfa_enc"),
            });

        // Init pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jfa_init_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&init_pipeline);
            pass.set_bind_group(0, &init_bg, &[]);
            pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
        }

        // JFA passes: step sizes from max_dim/2 down to 1
        let max_dim = self.dims[0].max(self.dims[1]).max(self.dims[2]);
        let mut step = max_dim / 2;
        let mut forward = true; // true: seeds_a -> seeds_b

        while step >= 1 {
            let jfa_params = JfaParams {
                dims: self.dims,
                step_size: step,
            };
            let jfa_params_buf =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("jfa_params"),
                        contents: bytemuck::bytes_of(&jfa_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let (src, dst) = if forward {
                (&seeds_a, &seeds_b)
            } else {
                (&seeds_b, &seeds_a)
            };

            let jfa_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jfa_bg"),
                layout: &jfa_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: jfa_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dst.as_entire_binding(),
                    },
                ],
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("jfa_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&jfa_pipeline);
                pass.set_bind_group(0, &jfa_bg, &[]);
                pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
            }

            forward = !forward;
            step /= 2;
        }

        // Distance conversion pass - reads from whichever seed buffer was last written
        let final_seeds = if forward { &seeds_a } else { &seeds_b };

        let dist_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dist_bg"),
            layout: &dist_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dist_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: final_seeds.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: distance_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dist_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&dist_pipeline);
            pass.set_bind_group(0, &dist_bg, &[]);
            pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
        }

        self.queue.submit(Some(encoder.finish()));
        distance_buf
    }

    // ── Buffer Readback Helpers ─────────────────────────────────────────

    fn readback_buffer(&self, buffer: &wgpu::Buffer) -> Vec<u32> {
        let raw = self.readback_buffer_raw(buffer);
        raw.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn readback_buffer_raw(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let size = buffer.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_enc"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging.unmap();
        result
    }

    fn voxels_with_value(&self, value: VoxelValue) -> Vec<[u32; 3]> {
        let data = self.readback_buffer(&self.buffer);
        let target = value as u32;
        let dx = self.dims[0] as usize;
        let dy = self.dims[1] as usize;

        let mut result = Vec::new();
        for (idx, &val) in data.iter().enumerate() {
            if val == target {
                let x = idx % dx;
                let y = (idx / dx) % dy;
                let z = idx / (dx * dy);
                result.push([x as u32, y as u32, z as u32]);
            }
        }
        result
    }
}

// ── Bind Group Layout Helpers ───────────────────────────────────────────

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

// ── Utility ─────────────────────────────────────────────────────────────

fn compute_aabb(vertices: &[Point3<f64>]) -> (Point3<f64>, Point3<f64>) {
    let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);

    for v in vertices {
        min = min.inf(v);
        max = max.sup(v);
    }
    (min, max)
}

/// Request a WGPU device and queue suitable for compute work.
///
/// This is a convenience function for creating the GPU context needed
/// by `VoxelGrid` and the decomposition module.
pub fn request_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("rmesh_compute"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;

    Some((Arc::new(device), Arc::new(queue)))
}

// Re-export wgpu buffer init for convenience
use wgpu::util::DeviceExt;

#[cfg(test)]
mod tests {
    use super::*;

    fn get_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        request_device()
    }

    fn unit_cube_mesh() -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
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
        (vertices, faces)
    }

    #[test]
    fn test_grid_sizing_min_32() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return, // No GPU available
        };

        let (v, f) = unit_cube_mesh();
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 8, FillMode::SurfaceOnly);

        // Minimum dimension should be at least 32
        assert!(grid.dims[0] >= 2);
        assert!(grid.dims[1] >= 2);
        assert!(grid.dims[2] >= 2);
    }

    #[test]
    fn test_voxelize_unit_cube() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = unit_cube_mesh();
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 1000, FillMode::SurfaceOnly);

        let surface = grid.surface_voxels();
        // A unit cube should produce a reasonable number of surface voxels
        assert!(!surface.is_empty(), "Expected surface voxels for unit cube");
    }

    #[test]
    fn test_flood_fill_closed() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = unit_cube_mesh();
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 1000, FillMode::FloodFill);

        let interior = grid.interior_voxels();
        let surface = grid.surface_voxels();

        assert!(!surface.is_empty(), "Expected surface voxels");
        assert!(
            !interior.is_empty(),
            "Expected interior voxels for closed mesh"
        );
    }

    #[test]
    fn test_surface_only_mode() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = unit_cube_mesh();
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 1000, FillMode::SurfaceOnly);

        let interior = grid.interior_voxels();
        assert!(
            interior.is_empty(),
            "SurfaceOnly mode should not produce interior voxels"
        );
    }

    #[test]
    fn test_voxel_to_world_roundtrip() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = unit_cube_mesh();
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 1000, FillMode::SurfaceOnly);

        let world_point = grid.voxel_to_world(5, 5, 5);
        let voxel = grid.world_to_voxel(&world_point);
        assert_eq!(voxel, [5, 5, 5]);
    }

    fn scaled_box_mesh(sx: f64, sy: f64, sz: f64) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
        let hx = sx / 2.0;
        let hy = sy / 2.0;
        let hz = sz / 2.0;
        let vertices = vec![
            Point3::new(-hx, -hy, -hz),
            Point3::new(hx, -hy, -hz),
            Point3::new(hx, hy, -hz),
            Point3::new(-hx, hy, -hz),
            Point3::new(-hx, -hy, hz),
            Point3::new(hx, -hy, hz),
            Point3::new(hx, hy, hz),
            Point3::new(-hx, hy, hz),
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
        (vertices, faces)
    }

    #[test]
    fn test_volume_unit_cube() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = unit_cube_mesh();
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 100_000, FillMode::FloodFill);
        let vol = grid.volume();

        let error = (vol - 1.0).abs() / 1.0;
        println!(
            "test_volume_unit_cube: volume={:.4}, error={:.2}%",
            vol,
            error * 100.0
        );
        assert!(
            error < 0.10,
            "Unit cube volume should be ~1.0, got {:.4} (error {:.2}%)",
            vol,
            error * 100.0
        );
    }

    #[test]
    fn test_volume_scaled_box() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = scaled_box_mesh(2.0, 3.0, 4.0);
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 100_000, FillMode::FloodFill);
        let vol = grid.volume();
        let expected = 24.0;

        let error = (vol - expected).abs() / expected;
        println!(
            "test_volume_scaled_box: volume={:.4}, expected={:.1}, error={:.2}%",
            vol,
            expected,
            error * 100.0
        );
        assert!(
            error < 0.10,
            "2x3x4 box volume should be ~{:.1}, got {:.4} (error {:.2}%)",
            expected,
            vol,
            error * 100.0
        );
    }

    #[test]
    fn test_area_unit_cube() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = unit_cube_mesh();
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 100_000, FillMode::FloodFill);
        let area = grid.area();
        let expected = 6.0;

        let error = (area - expected).abs() / expected;
        println!(
            "test_area_unit_cube: area={:.4}, expected={:.1}, error={:.2}%",
            area,
            expected,
            error * 100.0
        );
        assert!(
            error < 0.15,
            "Unit cube area should be ~{:.1}, got {:.4} (error {:.2}%)",
            expected,
            area,
            error * 100.0
        );
    }

    #[test]
    fn test_area_scaled_box() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = scaled_box_mesh(2.0, 3.0, 4.0);
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 100_000, FillMode::FloodFill);
        let area = grid.area();
        let expected = 52.0; // 2*(2*3 + 2*4 + 3*4) = 2*(6+8+12) = 52

        let error = (area - expected).abs() / expected;
        println!(
            "test_area_scaled_box: area={:.4}, expected={:.1}, error={:.2}%",
            area,
            expected,
            error * 100.0
        );
        assert!(
            error < 0.15,
            "2x3x4 box area should be ~{:.1}, got {:.4} (error {:.2}%)",
            expected,
            area,
            error * 100.0
        );
    }

    #[test]
    fn test_volume_vs_mesh_volume() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = unit_cube_mesh();
        let mesh_vol = crate::triangles::inertia::volume(&v, &f);
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 100_000, FillMode::FloodFill);
        let grid_vol = grid.volume();

        let error = (grid_vol - mesh_vol).abs() / mesh_vol.abs();
        println!(
            "test_volume_vs_mesh_volume: grid={:.4}, mesh={:.4}, error={:.2}%",
            grid_vol,
            mesh_vol,
            error * 100.0
        );
        assert!(
            error < 0.10,
            "Grid volume ({:.4}) and mesh volume ({:.4}) should agree within 10%, error={:.2}%",
            grid_vol,
            mesh_vol,
            error * 100.0
        );
    }

    #[test]
    fn test_area_surface_only() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = unit_cube_mesh();
        // With FloodFill, we get the full solid volume
        let grid_full = VoxelGrid::from_mesh(
            dev.clone(),
            queue.clone(),
            &v,
            &f,
            100_000,
            FillMode::FloodFill,
        );
        let vol_full = grid_full.volume();

        // With SurfaceOnly, only a thin shell is filled → much less volume
        let grid_shell = VoxelGrid::from_mesh(dev, queue, &v, &f, 100_000, FillMode::SurfaceOnly);
        let vol_shell = grid_shell.volume();

        println!(
            "test_area_surface_only: full={:.4}, shell={:.4}",
            vol_full, vol_shell
        );
        assert!(
            vol_shell < vol_full * 0.5,
            "SurfaceOnly volume ({:.4}) should be much less than FloodFill volume ({:.4})",
            vol_shell,
            vol_full
        );
    }
}
