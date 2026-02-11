use wgpu::util::DeviceExt;
use wgpu::{
    DepthBiasState, MultisampleState, PipelineCompilationOptions, StencilState, TextureFormat,
};

use super::device::DEPTH_FORMAT;
use super::upload::{GpuPath, LineVertex};

pub struct LineRenderer {
    pipeline: wgpu::RenderPipeline,
    grid_vertex_buffer: wgpu::Buffer,
    grid_vertex_count: u32,
    axes_vertex_buffer: wgpu::Buffer,
    axes_vertex_count: u32,
}

impl LineRenderer {
    /// Create a line renderer for 2D (no depth buffer).
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_2d(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        format: TextureFormat,
    ) -> Self {
        Self::new_inner(device, camera_bgl, format, false)
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn new_with_format(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        format: TextureFormat,
    ) -> Self {
        Self::new_inner(device, camera_bgl, format, true)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn new_inner(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        format: TextureFormat,
        use_depth: bool,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/line.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("line_pipeline_layout"),
            bind_group_layouts: &[camera_bgl],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: if use_depth {
                Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                })
            } else {
                None
            },
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Pre-generate grid and axes with unit scale (will be used with scene_extent)
        let (grid_verts, axes_verts) = generate_overlays(1.0);

        let grid_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid_vertices"),
            contents: bytemuck::cast_slice(&grid_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let axes_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("axes_vertices"),
            contents: bytemuck::cast_slice(&axes_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            pipeline,
            grid_vertex_buffer,
            grid_vertex_count: grid_verts.len() as u32,
            axes_vertex_buffer,
            axes_vertex_count: axes_verts.len() as u32,
        }
    }

    /// Regenerate grid/axes buffers for a new scene extent.
    #[allow(clippy::cast_possible_truncation)]
    pub fn update_overlays(&mut self, device: &wgpu::Device, scene_extent: f32) {
        let (grid_verts, axes_verts) = generate_overlays(scene_extent);

        self.grid_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid_vertices"),
            contents: bytemuck::cast_slice(&grid_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.grid_vertex_count = grid_verts.len() as u32;

        self.axes_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("axes_vertices"),
            contents: bytemuck::cast_slice(&axes_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.axes_vertex_count = axes_verts.len() as u32;
    }

    pub fn draw_grid<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, _scene_extent: f32) {
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
        pass.draw(0..self.grid_vertex_count, 0..1);
    }

    pub fn draw_axes<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, _scene_extent: f32) {
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.axes_vertex_buffer.slice(..));
        pass.draw(0..self.axes_vertex_count, 0..1);
    }

    pub fn draw_paths<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, paths: &'a [GpuPath]) {
        pass.set_pipeline(&self.pipeline);
        for path in paths {
            pass.set_vertex_buffer(0, path.vertex_buffer.slice(..));
            pass.draw(0..path.vertex_count, 0..1);
        }
    }

    /// Draw a single line buffer (used by 2D renderer).
    pub fn draw_buffer<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, buf: &'a GpuPath) {
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, buf.vertex_buffer.slice(..));
        pass.draw(0..buf.vertex_count, 0..1);
    }
}

/// Generate grid and axes overlay vertices.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn generate_overlays(extent: f32) -> (Vec<LineVertex>, Vec<LineVertex>) {
    let grid_color = [0.3, 0.3, 0.3];
    let half = extent;
    let lines = 21i32;
    let step = 2.0 * half / (lines - 1) as f32;

    let mut grid = Vec::with_capacity((lines as usize) * 4);
    for i in 0..lines {
        let t = -half + i as f32 * step;
        // Lines parallel to Y axis
        grid.push(LineVertex {
            position: [t, -half, 0.0],
            color: grid_color,
        });
        grid.push(LineVertex {
            position: [t, half, 0.0],
            color: grid_color,
        });
        // Lines parallel to X axis
        grid.push(LineVertex {
            position: [-half, t, 0.0],
            color: grid_color,
        });
        grid.push(LineVertex {
            position: [half, t, 0.0],
            color: grid_color,
        });
    }

    let len = extent;
    let axes = vec![
        // X axis - red
        LineVertex {
            position: [0.0, 0.0, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        LineVertex {
            position: [len, 0.0, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        // Y axis - green
        LineVertex {
            position: [0.0, 0.0, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        LineVertex {
            position: [0.0, len, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        // Z axis - blue
        LineVertex {
            position: [0.0, 0.0, 0.0],
            color: [0.0, 0.0, 1.0],
        },
        LineVertex {
            position: [0.0, 0.0, len],
            color: [0.0, 0.0, 1.0],
        },
    ];

    (grid, axes)
}
