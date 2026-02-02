use nalgebra::Matrix4;
use wgpu::util::DeviceExt;

use super::CameraUniforms;
use super::fill::{FillRenderer, GpuFill};
use super::line::LineRenderer;
use crate::gpu::GpuContext;
use crate::upload::GpuPath;

/// Orchestrates 2D rendering: fill polygons + line outlines/grid via reused LineRenderer.
pub struct Scene2DRenderer {
    #[allow(dead_code)]
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub line_renderer: LineRenderer,
    pub fill_renderer: FillRenderer,
}

impl Scene2DRenderer {
    pub fn new(gpu: &GpuContext) -> Self {
        let format = gpu.surface_format();
        let device = &gpu.device;
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera2d_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera2d_uniform"),
            contents: bytemuck::bytes_of(&CameraUniforms {
                view_proj: Matrix4::<f32>::identity().into(),
                camera_pos: [0.0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera2d_bg"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Line renderer reused from 3D but fed with ortho projection — no depth stencil needed.
        // We create it with the same format but it'll use the same line.wgsl shader.
        let line_renderer = LineRenderer::new_2d(device, &camera_bind_group_layout, format);
        let fill_renderer = FillRenderer::new(device, &camera_bind_group_layout, format);

        Self {
            camera_bind_group_layout,
            camera_buffer,
            camera_bind_group,
            line_renderer,
            fill_renderer,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, view_proj: &Matrix4<f32>) {
        let uniforms = CameraUniforms {
            view_proj: (*view_proj).into(),
            camera_pos: [0.0; 4],
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        grid_buf: Option<&GpuPath>,
        axes_buf: Option<&GpuPath>,
        fills: &[GpuFill],
        outlines: &[GpuPath],
        zoom_box_fill: Option<&GpuFill>,
        zoom_box_lines: Option<&GpuPath>,
        background: [f32; 3],
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("2d_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: f64::from(background[0]),
                        g: f64::from(background[1]),
                        b: f64::from(background[2]),
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_bind_group(0, &self.camera_bind_group, &[]);

        // 1. Grid
        if let Some(grid) = grid_buf {
            self.line_renderer.draw_buffer(&mut pass, grid);
        }

        // 2. Axes
        if let Some(axes) = axes_buf {
            self.line_renderer.draw_buffer(&mut pass, axes);
        }

        // 3. Polygon fills
        self.fill_renderer.draw(&mut pass, fills);

        // 4. Outlines
        for outline in outlines {
            self.line_renderer.draw_buffer(&mut pass, outline);
        }

        // 5. Zoom box overlay
        if let Some(zf) = zoom_box_fill {
            self.fill_renderer.draw(&mut pass, std::slice::from_ref(zf));
        }
        if let Some(zl) = zoom_box_lines {
            self.line_renderer.draw_buffer(&mut pass, zl);
        }
    }
}
