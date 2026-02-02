use wgpu::{
    DepthBiasState, MultisampleState, PipelineCompilationOptions, StencilState, TextureFormat,
};

use crate::gpu::DEPTH_FORMAT;
use crate::upload::{GpuPointCloud, PointVertex};

pub struct PointRenderer {
    pipeline: wgpu::RenderPipeline,
}

impl PointRenderer {
    pub fn new_with_format(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        format: TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shader_src/point.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point_pipeline_layout"),
            bind_group_layouts: &[camera_bgl],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PointVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_pipeline"),
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
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self { pipeline }
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, point_clouds: &'a [GpuPointCloud]) {
        pass.set_pipeline(&self.pipeline);
        for pc in point_clouds {
            pass.set_vertex_buffer(0, pc.vertex_buffer.slice(..));
            pass.draw(0..pc.point_count, 0..1);
        }
    }
}
