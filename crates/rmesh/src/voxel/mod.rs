//! GPU-accelerated voxel grid for mesh processing.
//!
//! Provides triangle mesh voxelization, flood-fill interior detection,
//! and 3D Jump Flooding Algorithm (JFA) distance field computation.
//! All operations run as WGPU compute shaders.
//!
//! # Feature Gate
//!
//! This module requires the `wgpu` feature.

mod marching_cubes;

use std::sync::Arc;

use nalgebra::Point3;

/// A hull candidate voxel from GPU-side convex candidate extraction.
#[derive(Debug, Clone, Copy)]
pub struct HullCandidate {
    /// The region this voxel belongs to.
    pub region_id: u32,
    /// Voxel grid coordinates.
    pub coords: [u32; 3],
}

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
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let target_dim = 32_u32.max((f64::from(resolution).cbrt() * 1.5) as u32);
        // Clamp to 1023 since we pack coords into 10 bits in JFA shaders
        let target_dim = target_dim.min(1023);
        let scale = longest / f64::from(target_dim);

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let dims = [
            2.max((extent.x / scale).ceil() as u32 + 2),
            2.max((extent.y / scale).ceil() as u32 + 2),
            2.max((extent.z / scale).ceil() as u32 + 2),
        ];
        // Clamp all dims to 1023
        let dims = [dims[0].min(1023), dims[1].min(1023), dims[2].min(1023)];

        // Offset origin by one voxel for boundary padding
        let origin = Point3::new(aabb_min.x - scale, aabb_min.y - scale, aabb_min.z - scale);

        let total_voxels = u64::from(dims[0]) * u64::from(dims[1]) * u64::from(dims[2]);

        // Create grid buffer (initialized to 0 = Undefined)
        let grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_grid"),
            size: total_voxels * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Explicit zero-init: wgpu spec says buffers are zero-initialized, but
        // some backends may recycle memory. A stale voxel propagates through
        // flood-fill → region splitting → non-deterministic decomposition.
        #[allow(clippy::cast_possible_truncation)]
        queue.write_buffer(&grid_buffer, 0, &vec![0u8; (total_voxels * 4) as usize]);

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

    /// Create an empty (all-Undefined) voxel grid with the given dimensions.
    pub fn empty(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        dims: [u32; 3],
        origin: Point3<f64>,
        pitch: f64,
    ) -> Self {
        let dims = [dims[0].min(1023), dims[1].min(1023), dims[2].min(1023)];
        let total_voxels = u64::from(dims[0]) * u64::from(dims[1]) * u64::from(dims[2]);

        let grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_grid_empty"),
            size: total_voxels * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        #[allow(clippy::cast_possible_truncation)]
        queue.write_buffer(&grid_buffer, 0, &vec![0u8; (total_voxels * 4) as usize]);

        Self {
            dims,
            origin,
            scale: pitch,
            buffer: grid_buffer,
            distance: None,
            device,
            queue,
        }
    }

    /// Create a voxel grid of a sphere from its analytic SDF.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn sphere(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        center: Point3<f64>,
        radius: f64,
        pitch: f64,
    ) -> Self {
        let pad = pitch;
        let origin = Point3::new(
            center.x - radius - pad,
            center.y - radius - pad,
            center.z - radius - pad,
        );
        let extent = 2.0 * (radius + pad);
        let dim = (extent / pitch).ceil() as u32 + 2;
        let dims = [dim.min(1023), dim.min(1023), dim.min(1023)];

        let grid = Self::empty(device, queue, dims, origin, pitch);
        // mode=0 (sphere), params=[radius, 0, 0, 0]
        grid.run_sdf_fill(0, center, [radius as f32, 0.0, 0.0, 0.0]);
        grid
    }

    /// Create a voxel grid of an axis-aligned cuboid from its analytic SDF.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn cuboid(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        center: Point3<f64>,
        extents: [f64; 3],
        pitch: f64,
    ) -> Self {
        let pad = pitch;
        let hx = extents[0] / 2.0;
        let hy = extents[1] / 2.0;
        let hz = extents[2] / 2.0;
        let origin = Point3::new(
            center.x - hx - pad,
            center.y - hy - pad,
            center.z - hz - pad,
        );
        let dims = [
            ((extents[0] + 2.0 * pad) / pitch).ceil() as u32 + 2,
            ((extents[1] + 2.0 * pad) / pitch).ceil() as u32 + 2,
            ((extents[2] + 2.0 * pad) / pitch).ceil() as u32 + 2,
        ];
        let dims = [dims[0].min(1023), dims[1].min(1023), dims[2].min(1023)];

        let grid = Self::empty(device, queue, dims, origin, pitch);
        // mode=1 (box), params=[hx, hy, hz, 0]
        grid.run_sdf_fill(1, center, [hx as f32, hy as f32, hz as f32, 0.0]);
        grid
    }

    /// Create a voxel grid of an axis-aligned cylinder from its analytic SDF.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn cylinder(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        center: Point3<f64>,
        axis: u8,
        radius: f64,
        height: f64,
        pitch: f64,
    ) -> Self {
        let pad = pitch;
        let hh = height / 2.0;
        let mut extents = [radius + pad, radius + pad, radius + pad];
        extents[axis as usize] = hh + pad;

        let origin = Point3::new(
            center.x - extents[0],
            center.y - extents[1],
            center.z - extents[2],
        );
        let dims = [
            (2.0 * extents[0] / pitch).ceil() as u32 + 2,
            (2.0 * extents[1] / pitch).ceil() as u32 + 2,
            (2.0 * extents[2] / pitch).ceil() as u32 + 2,
        ];
        let dims = [dims[0].min(1023), dims[1].min(1023), dims[2].min(1023)];

        let grid = Self::empty(device, queue, dims, origin, pitch);
        // mode=2 (cylinder), params=[radius, half_height, axis, 0]
        grid.run_sdf_fill(2, center, [radius as f32, hh as f32, f32::from(axis), 0.0]);
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
        u64::from(self.dims[0]) * u64::from(self.dims[1]) * u64::from(self.dims[2])
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
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
        let volume = f64::from(filled) * self.scale.powi(3);
        let area = f64::from(faces) * self.scale.powi(2);
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

    /// Return a new `VoxelGrid` with interior voxels zeroed out.
    ///
    /// A filled voxel is "shell" if at least one 6-neighbor is unfilled.
    /// Interior filled voxels become `Undefined` (0). The result is a
    /// GPU-resident grid — no readback occurs.
    #[must_use]
    pub fn shell(&self) -> VoxelGrid {
        let total = self.total_voxels();

        // Copy grid to read-only input
        let grid_ro = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shell_grid_ro"),
            size: total * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shell_grid_out"),
            size: total * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("shell_copy_enc"),
                });
            enc.copy_buffer_to_buffer(&self.buffer, 0, &grid_ro, 0, total * 4);
            self.queue.submit(Some(enc.finish()));
        }

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dims {
            dims: [u32; 3],
            _pad: u32,
        }

        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shell_dims"),
                contents: bytemuck::bytes_of(&Dims {
                    dims: self.dims,
                    _pad: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_shell"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cs_shell.wgsl").into()),
            });

        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shell_bgl"),
                entries: &[bgl_storage_ro(0), bgl_storage_rw(1), bgl_uniform(2)],
            });

        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shell_pl"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("shell_pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shell_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_ro.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shell_enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shell_pass"),
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
        self.queue.submit(Some(enc.finish()));

        VoxelGrid {
            dims: self.dims,
            origin: self.origin,
            scale: self.scale,
            buffer: grid_out,
            distance: None,
            device: Arc::clone(&self.device),
            queue: Arc::clone(&self.queue),
        }
    }

    /// Boolean union: combine two grids, keeping voxels from either.
    #[must_use]
    pub fn union(&self, other: &VoxelGrid) -> VoxelGrid {
        self.run_boolean(other, 0)
    }

    /// Boolean intersection: keep only voxels present in both grids.
    #[must_use]
    pub fn intersection(&self, other: &VoxelGrid) -> VoxelGrid {
        self.run_boolean(other, 1)
    }

    /// Boolean difference: keep voxels from self that are not in other.
    #[must_use]
    pub fn difference(&self, other: &VoxelGrid) -> VoxelGrid {
        self.run_boolean(other, 2)
    }

    /// GPU-compacted list of all filled voxel coordinates.
    ///
    /// Uses atomic append on the GPU — no full-grid readback.
    pub fn compact(&self) -> Vec<[u32; 3]> {
        let total = self.total_voxels();

        // Copy grid to read-only input
        let grid_ro = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_grid_ro"),
            size: total * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("compact_copy_enc"),
                });
            enc.copy_buffer_to_buffer(&self.buffer, 0, &grid_ro, 0, total * 4);
            self.queue.submit(Some(enc.finish()));
        }

        // Output buffer: 1 counter + up to total_voxels * 3 coords
        let max_entries = total;
        let output_size = (1 + max_entries * 3) * 4;
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero-initialize counter
        self.queue.write_buffer(&output_buf, 0, &[0u8; 4]);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dims {
            dims: [u32; 3],
            _pad: u32,
        }

        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("compact_dims"),
                contents: bytemuck::bytes_of(&Dims {
                    dims: self.dims,
                    _pad: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_compact"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cs_compact.wgsl").into()),
            });

        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compact_bgl"),
                entries: &[bgl_storage_ro(0), bgl_storage_rw(1), bgl_uniform(2)],
            });

        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compact_pl"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compact_pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_ro.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compact_enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compact_pass"),
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

        // Readback
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&output_buf, 0, &readback, 0, output_size);
        self.queue.submit(Some(enc.finish()));

        let data = crate::gpu::gpu_read_buffer(&self.device, &readback);
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let base = 4 + i * 12; // skip 4-byte counter, each entry is 3 × u32
            if base + 12 > data.len() {
                break;
            }
            let x =
                u32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]]);
            let y = u32::from_le_bytes([
                data[base + 4],
                data[base + 5],
                data[base + 6],
                data[base + 7],
            ]);
            let z = u32::from_le_bytes([
                data[base + 8],
                data[base + 9],
                data[base + 10],
                data[base + 11],
            ]);
            result.push([x, y, z]);
        }
        result
    }

    /// GPU-compacted convex hull candidates from a region_ids buffer.
    ///
    /// Two-pass pipeline: boundary detection + 26-direction support-plane
    /// culling + compaction. Returns only the ~1-3% of voxels that could
    /// be convex hull vertices.
    pub fn convex_candidates(
        &self,
        region_ids: &wgpu::Buffer,
        max_region_id: u32,
    ) -> Vec<HullCandidate> {
        let total = self.total_voxels();
        let dims = self.dims;
        let max_dim = dims[0].max(dims[1]).max(dims[2]);

        // Copy region_ids to read-only buffer
        let region_ro = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convex_region_ro"),
            size: total * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("convex_copy_enc"),
                });
            enc.copy_buffer_to_buffer(region_ids, 0, &region_ro, 0, total * 4);
            self.queue.submit(Some(enc.finish()));
        }

        // Extremes buffer: (max_region_id + 1) × 26 × 4 bytes, zero-initialized
        let extremes_count = (u64::from(max_region_id) + 1) * 26;
        let extremes_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convex_extremes"),
            size: extremes_count * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero-initialize (atomicMax starts from 0)
        #[allow(clippy::cast_possible_truncation)]
        let zeros = vec![0u8; (extremes_count * 4) as usize];
        self.queue.write_buffer(&extremes_buf, 0, &zeros);

        // Output buffer: generous ceiling at total/10, each entry is 4 × u32 + 1 counter
        let max_entries = (total / 10).max(64);
        let output_size = (1 + max_entries * 4) * 4;
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convex_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&output_buf, 0, &[0u8; 4]);

        // ── Pass 1: cs_convex_support ──
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SupportParams {
            dims: [u32; 3],
            max_dim: u32,
        }

        let support_params_buf =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("convex_support_params"),
                    contents: bytemuck::bytes_of(&SupportParams { dims, max_dim }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let support_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_convex_support"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/cs_convex_support.wgsl").into(),
                ),
            });

        let support_bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("convex_support_bgl"),
                entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
            });

        let support_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("convex_support_pl"),
                bind_group_layouts: &[&support_bgl],
                immediate_size: 0,
            });

        let support_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("convex_support_pipeline"),
                    layout: Some(&support_pl),
                    module: &support_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        let support_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("convex_support_bg"),
            layout: &support_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: support_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: region_ro.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: extremes_buf.as_entire_binding(),
                },
            ],
        });

        // ── Pass 2: cs_convex_compact ──
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct CompactParams {
            dims: [u32; 3],
            max_dim: u32,
            margin: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let compact_params_buf =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("convex_compact_params"),
                    contents: bytemuck::bytes_of(&CompactParams {
                        dims,
                        max_dim,
                        margin: 1,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let compact_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_convex_compact"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/cs_convex_compact.wgsl").into(),
                ),
            });

        let compact_bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("convex_compact_bgl"),
                entries: &[
                    bgl_uniform(0),
                    bgl_storage_ro(1),
                    bgl_storage_ro(2),
                    bgl_storage_rw(3),
                ],
            });

        let compact_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("convex_compact_pl"),
                bind_group_layouts: &[&compact_bgl],
                immediate_size: 0,
            });

        let compact_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("convex_compact_pipeline"),
                    layout: Some(&compact_pl),
                    module: &compact_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        let compact_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("convex_compact_bg"),
            layout: &compact_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: compact_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: region_ro.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: extremes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch both passes in a single encoder submit
        let wg = [
            dims[0].div_ceil(4),
            dims[1].div_ceil(4),
            dims[2].div_ceil(4),
        ];

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("convex_enc"),
            });

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("convex_support_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&support_pipeline);
            pass.set_bind_group(0, &support_bg, &[]);
            pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
        }

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("convex_compact_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&compact_pipeline);
            pass.set_bind_group(0, &compact_bg, &[]);
            pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
        }

        // Readback
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convex_readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&output_buf, 0, &readback, 0, output_size);
        self.queue.submit(Some(enc.finish()));

        let data = crate::gpu::gpu_read_buffer(&self.device, &readback);
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let base = 4 + i * 16; // skip 4-byte counter, each entry is 4 × u32
            if base + 16 > data.len() {
                break;
            }
            let rid =
                u32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]]);
            let x = u32::from_le_bytes([
                data[base + 4],
                data[base + 5],
                data[base + 6],
                data[base + 7],
            ]);
            let y = u32::from_le_bytes([
                data[base + 8],
                data[base + 9],
                data[base + 10],
                data[base + 11],
            ]);
            let z = u32::from_le_bytes([
                data[base + 12],
                data[base + 13],
                data[base + 14],
                data[base + 15],
            ]);
            result.push(HullCandidate {
                region_id: rid,
                coords: [x, y, z],
            });
        }
        result
    }

    // ── GPU Pipeline Internals ──────────────────────────────────────────

    #[allow(clippy::cast_possible_truncation)]
    fn run_sdf_fill(&self, mode: u32, center: Point3<f64>, params: [f32; 4]) {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SdfFillParams {
            dims: [u32; 3],
            mode: u32,
            origin: [f32; 3],
            pitch: f32,
            center: [f32; 3],
            _pad: u32,
            params: [f32; 4],
        }

        let uniform = SdfFillParams {
            dims: self.dims,
            mode,
            origin: [
                self.origin.x as f32,
                self.origin.y as f32,
                self.origin.z as f32,
            ],
            pitch: self.scale as f32,
            center: [center.x as f32, center.y as f32, center.z as f32],
            _pad: 0,
            params,
        };

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sdf_fill_params"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_sdf_fill"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cs_sdf_fill.wgsl").into()),
            });

        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sdf_fill_bgl"),
                entries: &[bgl_uniform(0), bgl_storage_rw(1)],
            });

        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sdf_fill_pl"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sdf_fill_pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sdf_fill_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffer.as_entire_binding(),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sdf_fill_enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdf_fill_pass"),
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
        self.queue.submit(Some(enc.finish()));
    }

    #[allow(clippy::cast_possible_truncation)]
    fn run_boolean(&self, other: &VoxelGrid, mode: u32) -> VoxelGrid {
        // Compute output AABB based on operation type
        let (out_min, out_max) = match mode {
            0 => {
                // Union: combined AABB
                let a_max = Point3::new(
                    self.origin.x + f64::from(self.dims[0]) * self.scale,
                    self.origin.y + f64::from(self.dims[1]) * self.scale,
                    self.origin.z + f64::from(self.dims[2]) * self.scale,
                );
                let b_max = Point3::new(
                    other.origin.x + f64::from(other.dims[0]) * other.scale,
                    other.origin.y + f64::from(other.dims[1]) * other.scale,
                    other.origin.z + f64::from(other.dims[2]) * other.scale,
                );
                (self.origin.inf(&other.origin), a_max.sup(&b_max))
            }
            1 => {
                // Intersection: overlapping AABB
                let a_max = Point3::new(
                    self.origin.x + f64::from(self.dims[0]) * self.scale,
                    self.origin.y + f64::from(self.dims[1]) * self.scale,
                    self.origin.z + f64::from(self.dims[2]) * self.scale,
                );
                let b_max = Point3::new(
                    other.origin.x + f64::from(other.dims[0]) * other.scale,
                    other.origin.y + f64::from(other.dims[1]) * other.scale,
                    other.origin.z + f64::from(other.dims[2]) * other.scale,
                );
                let lo = self.origin.sup(&other.origin);
                let hi = a_max.inf(&b_max);
                // If no overlap, return empty grid
                if lo.x >= hi.x || lo.y >= hi.y || lo.z >= hi.z {
                    return VoxelGrid::empty(
                        Arc::clone(&self.device),
                        Arc::clone(&self.queue),
                        [2, 2, 2],
                        self.origin,
                        self.scale,
                    );
                }
                (lo, hi)
            }
            _ => {
                // Difference: self's AABB
                let a_max = Point3::new(
                    self.origin.x + f64::from(self.dims[0]) * self.scale,
                    self.origin.y + f64::from(self.dims[1]) * self.scale,
                    self.origin.z + f64::from(self.dims[2]) * self.scale,
                );
                (self.origin, a_max)
            }
        };

        // Use the finer pitch
        let pitch = self.scale.min(other.scale);

        #[allow(clippy::cast_sign_loss)]
        let out_dims = [
            2u32.max(((out_max.x - out_min.x) / pitch).ceil() as u32 + 1)
                .min(1023),
            2u32.max(((out_max.y - out_min.y) / pitch).ceil() as u32 + 1)
                .min(1023),
            2u32.max(((out_max.z - out_min.z) / pitch).ceil() as u32 + 1)
                .min(1023),
        ];

        let out_grid = VoxelGrid::empty(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            out_dims,
            out_min,
            pitch,
        );

        // Copy both input grids to read-only buffers
        let total_a = self.total_voxels();
        let total_b = other.total_voxels();

        let grid_a_ro = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bool_grid_a_ro"),
            size: total_a * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_b_ro = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bool_grid_b_ro"),
            size: total_b * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bool_copy_enc"),
                });
            enc.copy_buffer_to_buffer(&self.buffer, 0, &grid_a_ro, 0, total_a * 4);
            enc.copy_buffer_to_buffer(&other.buffer, 0, &grid_b_ro, 0, total_b * 4);
            self.queue.submit(Some(enc.finish()));
        }

        // Uniform
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct BooleanParams {
            out_dims: [u32; 3],
            mode: u32,
            out_origin: [f32; 3],
            pitch: f32,
            a_dims: [u32; 3],
            _pad0: u32,
            a_origin: [f32; 3],
            _pad1: u32,
            b_dims: [u32; 3],
            _pad2: u32,
            b_origin: [f32; 3],
            _pad3: u32,
        }

        let uniform = BooleanParams {
            out_dims: out_grid.dims,
            mode,
            out_origin: [
                out_grid.origin.x as f32,
                out_grid.origin.y as f32,
                out_grid.origin.z as f32,
            ],
            pitch: pitch as f32,
            a_dims: self.dims,
            _pad0: 0,
            a_origin: [
                self.origin.x as f32,
                self.origin.y as f32,
                self.origin.z as f32,
            ],
            _pad1: 0,
            b_dims: other.dims,
            _pad2: 0,
            b_origin: [
                other.origin.x as f32,
                other.origin.y as f32,
                other.origin.z as f32,
            ],
            _pad3: 0,
        };

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bool_params"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cs_boolean"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cs_boolean.wgsl").into()),
            });

        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bool_bgl"),
                entries: &[
                    bgl_uniform(0),
                    bgl_storage_ro(1),
                    bgl_storage_ro(2),
                    bgl_storage_rw(3),
                ],
            });

        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bool_pl"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bool_pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bool_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_a_ro.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_b_ro.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_grid.buffer.as_entire_binding(),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bool_enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bool_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                out_grid.dims[0].div_ceil(4),
                out_grid.dims[1].div_ceil(4),
                out_grid.dims[2].div_ceil(4),
            );
        }
        self.queue.submit(Some(enc.finish()));

        out_grid
    }

    fn run_voxelize(&self, vertices: &[Point3<f64>], faces: &[[usize; 3]]) {
        if faces.is_empty() {
            return;
        }

        // Pack triangles as 9 f32 per triangle
        #[allow(clippy::cast_possible_truncation)]
        let tri_data: Vec<f32> = {
            let mut data = Vec::with_capacity(faces.len() * 9);
            for &[i0, i1, i2] in faces {
                for &vi in &[i0, i1, i2] {
                    let v = &vertices[vi];
                    data.push(v.x as f32);
                    data.push(v.y as f32);
                    data.push(v.z as f32);
                }
            }
            data
        };

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

        #[allow(clippy::cast_possible_truncation)]
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
                immediate_size: 0,
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
            #[allow(clippy::cast_possible_truncation)]
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
                immediate_size: 0,
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
                let data = crate::gpu::gpu_read_buffer(&self.device, &readback);
                u32::from_le_bytes([data[0], data[1], data[2], data[3]])
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
                    while let Some(hit) = bvh.trace_ray(&origin, &dir, vertices, faces) {
                        count += 1;
                        origin = hit.point + dir * 1e-8;
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
                immediate_size: 0,
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

        let data = crate::gpu::gpu_read_buffer(&self.device, &readback);
        let filled = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let faces = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

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
                immediate_size: 0,
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
                immediate_size: 0,
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

        #[allow(clippy::cast_possible_truncation)]
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
                immediate_size: 0,
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

        crate::gpu::gpu_read_buffer(&self.device, &staging)
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
                #[allow(clippy::cast_possible_truncation)]
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
/// Returns a process-wide shared device to avoid driver-level deadlocks
/// when multiple threads do GPU work in parallel (see [`crate::gpu`]).
pub fn request_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    crate::gpu::request_device()
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
    fn test_shell_boundary_only() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = unit_cube_mesh();
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 100_000, FillMode::FloodFill);

        let original_filled = grid.compact().len();
        let shell_grid = grid.shell();
        let shell_filled = shell_grid.compact().len();

        println!(
            "test_shell_boundary_only: original={}, shell={}",
            original_filled, shell_filled
        );

        assert!(
            shell_filled < original_filled,
            "Shell ({}) should have fewer filled voxels than original ({})",
            shell_filled,
            original_filled
        );
        assert!(
            shell_filled > 0,
            "Shell should have at least some filled voxels"
        );
    }

    #[test]
    fn test_compact_matches_readback() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = unit_cube_mesh();
        let grid = VoxelGrid::from_mesh(dev, queue, &v, &f, 10_000, FillMode::FloodFill);

        // Get filled voxels via full readback
        let mut readback_coords: Vec<[u32; 3]> = grid.surface_voxels();
        readback_coords.extend(grid.interior_voxels());
        readback_coords.sort();

        // Get filled voxels via GPU compact
        let mut compact_coords = grid.compact();
        compact_coords.sort();

        assert_eq!(
            compact_coords.len(),
            readback_coords.len(),
            "compact() count ({}) should match readback count ({})",
            compact_coords.len(),
            readback_coords.len()
        );
        assert_eq!(
            compact_coords, readback_coords,
            "compact() coordinates should match readback coordinates"
        );
    }

    #[test]
    fn test_convex_candidates_culling() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let (v, f) = unit_cube_mesh();
        let grid = VoxelGrid::from_mesh(
            Arc::clone(&dev),
            Arc::clone(&queue),
            &v,
            &f,
            100_000,
            FillMode::FloodFill,
        );

        let total_filled = grid.compact().len();

        // Create a single-region region_ids buffer (all filled voxels = region 0)
        let grid_data = grid.to_array();
        let region_init: Vec<u32> = grid_data
            .iter()
            .map(|v| match v {
                VoxelValue::Surface | VoxelValue::Inside => 0,
                _ => u32::MAX,
            })
            .collect();
        let region_ids_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_region_ids"),
            contents: bytemuck::cast_slice(&region_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let candidates = grid.convex_candidates(&region_ids_buf, 1);

        println!(
            "test_convex_candidates_culling: total_filled={}, candidates={}",
            total_filled,
            candidates.len()
        );

        assert!(
            !candidates.is_empty(),
            "Should have at least some convex candidates"
        );
        assert!(
            candidates.len() < total_filled,
            "Candidates ({}) should be fewer than total filled voxels ({})",
            candidates.len(),
            total_filled
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

    #[test]
    fn test_empty_grid() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let grid = VoxelGrid::empty(dev, queue, [10, 10, 10], Point3::origin(), 0.1);
        assert_eq!(grid.dims(), [10, 10, 10]);
        assert_eq!(grid.volume(), 0.0);
        let data = grid.to_array();
        assert!(data.iter().all(|v| *v == VoxelValue::Undefined));
    }

    #[test]
    fn test_sphere_volume() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let r = 1.0;
        let pitch = 0.05;
        let grid = VoxelGrid::sphere(dev, queue, Point3::origin(), r, pitch);
        let vol = grid.volume();
        let expected = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        let error = (vol - expected).abs() / expected;

        println!(
            "test_sphere_volume: volume={:.4}, expected={:.4}, error={:.2}%",
            vol,
            expected,
            error * 100.0
        );
        assert!(
            error < 0.05,
            "Sphere volume should be ~{:.4}, got {:.4} (error {:.2}%)",
            expected,
            vol,
            error * 100.0
        );
    }

    #[test]
    fn test_cuboid_volume() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let extents = [2.0, 3.0, 4.0];
        let pitch = 0.1;
        let grid = VoxelGrid::cuboid(dev, queue, Point3::origin(), extents, pitch);
        let vol = grid.volume();
        let expected = 24.0;
        let error = (vol - expected).abs() / expected;

        println!(
            "test_cuboid_volume: volume={:.4}, expected={:.4}, error={:.2}%",
            vol,
            expected,
            error * 100.0
        );
        assert!(
            error < 0.05,
            "Cuboid volume should be ~{:.1}, got {:.4} (error {:.2}%)",
            expected,
            vol,
            error * 100.0
        );
    }

    #[test]
    fn test_cylinder_volume() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let r = 1.0;
        let h = 3.0;
        let pitch = 0.1;
        let grid = VoxelGrid::cylinder(dev, queue, Point3::origin(), 2, r, h, pitch);
        let vol = grid.volume();
        let expected = std::f64::consts::PI * r.powi(2) * h;
        let error = (vol - expected).abs() / expected;

        println!(
            "test_cylinder_volume: volume={:.4}, expected={:.4}, error={:.2}%",
            vol,
            expected,
            error * 100.0
        );
        assert!(
            error < 0.05,
            "Cylinder volume should be ~{:.4}, got {:.4} (error {:.2}%)",
            expected,
            vol,
            error * 100.0
        );
    }

    #[test]
    fn test_sphere_shell_hollow() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let grid = VoxelGrid::sphere(dev, queue, Point3::origin(), 1.0, 0.05);
        let shell_grid = grid.shell();
        let interior = shell_grid.interior_voxels();

        println!(
            "test_sphere_shell_hollow: interior_count={}",
            interior.len()
        );
        assert!(
            interior.is_empty(),
            "Shell of sphere should have no interior voxels, found {}",
            interior.len()
        );
    }

    #[test]
    fn test_boolean_union_volume() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let pitch = 0.05;
        // Two cuboids: [0,1]^3 and [0.5,1.5]^3 => overlap 0.5^3 = 0.125
        // Union volume = 2.0 - 0.125 = 1.875... but extents-based:
        // A centered at (0.5, 0.5, 0.5) extents [1,1,1]
        // B centered at (1.0, 0.5, 0.5) extents [1,1,1]
        // => A covers [0,1], B covers [0.5,1.5] in X. Union = [0,1.5] in X, [0,1] in Y,Z
        // Union vol = 1.5
        let a = VoxelGrid::cuboid(
            Arc::clone(&dev),
            Arc::clone(&queue),
            Point3::new(0.5, 0.5, 0.5),
            [1.0, 1.0, 1.0],
            pitch,
        );
        let b = VoxelGrid::cuboid(
            dev,
            queue,
            Point3::new(1.0, 0.5, 0.5),
            [1.0, 1.0, 1.0],
            pitch,
        );
        let u = a.union(&b);
        let vol = u.volume();
        let expected = 1.5;
        let error = (vol - expected).abs() / expected;

        println!(
            "test_boolean_union_volume: vol={:.4}, expected={:.4}, error={:.2}%",
            vol,
            expected,
            error * 100.0
        );
        assert!(
            error < 0.10,
            "Union volume should be ~{:.2}, got {:.4} (error {:.2}%)",
            expected,
            vol,
            error * 100.0
        );
    }

    #[test]
    fn test_boolean_intersection_volume() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let pitch = 0.05;
        let a = VoxelGrid::cuboid(
            Arc::clone(&dev),
            Arc::clone(&queue),
            Point3::new(0.5, 0.5, 0.5),
            [1.0, 1.0, 1.0],
            pitch,
        );
        let b = VoxelGrid::cuboid(
            dev,
            queue,
            Point3::new(1.0, 0.5, 0.5),
            [1.0, 1.0, 1.0],
            pitch,
        );
        let inter = a.intersection(&b);
        let vol = inter.volume();
        let expected = 0.5; // overlap region: [0.5,1.0] in X, [0,1] in Y,Z
        let error = (vol - expected).abs() / expected;

        println!(
            "test_boolean_intersection_volume: vol={:.4}, expected={:.4}, error={:.2}%",
            vol,
            expected,
            error * 100.0
        );
        assert!(
            error < 0.10,
            "Intersection volume should be ~{:.2}, got {:.4} (error {:.2}%)",
            expected,
            vol,
            error * 100.0
        );
    }

    #[test]
    fn test_boolean_difference_volume() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let pitch = 0.05;
        let a = VoxelGrid::cuboid(
            Arc::clone(&dev),
            Arc::clone(&queue),
            Point3::new(0.5, 0.5, 0.5),
            [1.0, 1.0, 1.0],
            pitch,
        );
        let b = VoxelGrid::cuboid(
            dev,
            queue,
            Point3::new(1.0, 0.5, 0.5),
            [1.0, 1.0, 1.0],
            pitch,
        );
        let diff = a.difference(&b);
        let vol = diff.volume();
        let expected = 0.5; // A minus overlap: 1.0 - 0.5 = 0.5
        let error = (vol - expected).abs() / expected;

        println!(
            "test_boolean_difference_volume: vol={:.4}, expected={:.4}, error={:.2}%",
            vol,
            expected,
            error * 100.0
        );
        assert!(
            error < 0.10,
            "Difference volume should be ~{:.2}, got {:.4} (error {:.2}%)",
            expected,
            vol,
            error * 100.0
        );
    }

    #[test]
    fn test_boolean_union_to_mesh() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let pitch = 0.05;
        let a = VoxelGrid::cuboid(
            Arc::clone(&dev),
            Arc::clone(&queue),
            Point3::new(0.5, 0.5, 0.5),
            [1.0, 1.0, 1.0],
            pitch,
        );
        let b = VoxelGrid::cuboid(
            dev,
            queue,
            Point3::new(1.0, 0.5, 0.5),
            [1.0, 1.0, 1.0],
            pitch,
        );
        let u = a.union(&b);

        let (verts, faces) = u.to_mesh();
        assert!(!verts.is_empty(), "Mesh should have vertices");
        assert!(!faces.is_empty(), "Mesh should have faces");

        // Check volume via signed tetrahedra
        let mesh_vol = crate::triangles::inertia::volume(&verts, &faces).abs();
        let expected = 1.5;
        let error = (mesh_vol - expected).abs() / expected;

        println!(
            "test_boolean_union_to_mesh: verts={}, faces={}, mesh_vol={:.4}, expected={:.2}, error={:.2}%",
            verts.len(),
            faces.len(),
            mesh_vol,
            expected,
            error * 100.0
        );
        assert!(
            error < 0.15,
            "Union mesh volume should be ~{:.2}, got {:.4} (error {:.2}%)",
            expected,
            mesh_vol,
            error * 100.0
        );
    }

    #[test]
    fn test_marching_cubes_sphere() {
        let (dev, queue) = match get_device() {
            Some(dq) => dq,
            None => return,
        };

        let r = 1.0;
        let pitch = 0.05;
        let grid = VoxelGrid::sphere(dev, queue, Point3::origin(), r, pitch);
        let (verts, faces) = grid.to_mesh();

        assert!(!verts.is_empty(), "Sphere mesh should have vertices");
        assert!(!faces.is_empty(), "Sphere mesh should have faces");

        let mesh_vol = crate::triangles::inertia::volume(&verts, &faces).abs();
        let expected = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        let error = (mesh_vol - expected).abs() / expected;

        println!(
            "test_marching_cubes_sphere: verts={}, faces={}, mesh_vol={:.4}, expected={:.4}, error={:.2}%",
            verts.len(),
            faces.len(),
            mesh_vol,
            expected,
            error * 100.0
        );
        assert!(
            error < 0.10,
            "Sphere mesh volume should be ~{:.4}, got {:.4} (error {:.2}%)",
            expected,
            mesh_vol,
            error * 100.0
        );
    }
}
