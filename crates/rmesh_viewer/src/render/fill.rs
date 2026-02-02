use wgpu::util::DeviceExt;
use wgpu::{MultisampleState, PipelineCompilationOptions, TextureFormat};

/// GPU-ready 2D fill vertex (just position, color is a uniform).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FillVertex {
    pub position: [f32; 2],
}

/// An uploaded polygon fill draw call.
pub struct GpuFill {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    #[allow(dead_code)] // Holds ownership; GPU buffer lives as long as this struct.
    pub color_buffer: wgpu::Buffer,
    pub color_bind_group: wgpu::BindGroup,
}

pub struct FillRenderer {
    pipeline: wgpu::RenderPipeline,
    pub color_bind_group_layout: wgpu::BindGroupLayout,
}

impl FillRenderer {
    pub fn new(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        format: TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fill2d_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shader_src/fill2d.wgsl").into()),
        });

        let color_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fill_color_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fill2d_pipeline_layout"),
            bind_group_layouts: &[camera_bgl, &color_bind_group_layout],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<FillVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            }],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fill2d_pipeline"),
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            color_bind_group_layout,
        }
    }

    /// Upload a polygon fill to the GPU.
    #[allow(clippy::cast_possible_truncation)]
    pub fn upload(
        &self,
        device: &wgpu::Device,
        vertices: &[[f32; 2]],
        indices: &[u32],
        color: [f32; 4],
    ) -> GpuFill {
        let verts: Vec<FillVertex> = vertices
            .iter()
            .map(|&p| FillVertex { position: p })
            .collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fill_vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fill_indices"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fill_color"),
            contents: bytemuck::cast_slice(&color),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let color_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fill_color_bg"),
            layout: &self.color_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_buffer.as_entire_binding(),
            }],
        });

        GpuFill {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            color_buffer,
            color_bind_group,
        }
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, fills: &'a [GpuFill]) {
        pass.set_pipeline(&self.pipeline);
        for fill in fills {
            pass.set_bind_group(1, &fill.color_bind_group, &[]);
            pass.set_vertex_buffer(0, fill.vertex_buffer.slice(..));
            pass.set_index_buffer(fill.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..fill.index_count, 0, 0..1);
        }
    }
}
