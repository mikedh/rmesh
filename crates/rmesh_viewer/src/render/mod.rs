pub mod fill;
pub mod line;
pub mod mesh;
pub mod point;
pub mod scene2d;
pub mod voxel;

use nalgebra::Matrix4;
use wgpu::TextureFormat;
use wgpu::util::DeviceExt;

use crate::gpu::GpuContext;
use crate::input::RenderToggles;
use crate::upload::SceneGpuData;

/// Camera uniform buffer data.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

/// Orchestrates all sub-renderers.
pub struct SceneRenderer {
    #[allow(dead_code)]
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub mesh_renderer: mesh::MeshRenderer,
    pub line_renderer: line::LineRenderer,
    pub point_renderer: point::PointRenderer,
    #[allow(dead_code)]
    pub voxel_renderer: voxel::VoxelRenderer,
}

impl SceneRenderer {
    pub fn new(gpu: &GpuContext) -> Self {
        Self::new_with_format(&gpu.device, &gpu.queue, gpu.surface_format())
    }

    pub fn new_with_format(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: TextureFormat,
    ) -> Self {
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bgl"),
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
            label: Some("camera_uniform"),
            contents: bytemuck::bytes_of(&CameraUniforms {
                view_proj: Matrix4::<f32>::identity().into(),
                camera_pos: [0.0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let mesh_renderer =
            mesh::MeshRenderer::new_with_format(device, queue, &camera_bind_group_layout, format);
        let line_renderer =
            line::LineRenderer::new_with_format(device, &camera_bind_group_layout, format);
        let point_renderer =
            point::PointRenderer::new_with_format(device, &camera_bind_group_layout, format);
        let voxel_renderer =
            voxel::VoxelRenderer::new_with_format(device, &camera_bind_group_layout, format);

        Self {
            camera_bind_group_layout,
            camera_buffer,
            camera_bind_group,
            mesh_renderer,
            line_renderer,
            point_renderer,
            voxel_renderer,
        }
    }

    pub fn update_camera(
        &self,
        queue: &wgpu::Queue,
        view_proj: &Matrix4<f32>,
        camera_pos: [f32; 3],
    ) {
        let uniforms = CameraUniforms {
            view_proj: (*view_proj).into(),
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        scene_data: &SceneGpuData,
        mesh_bind_groups: &[mesh::MeshBindGroups],
        toggles: &RenderToggles,
        background: [f32; 3],
        scene_extent: f32,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main_pass"),
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
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

        // Draw grid
        if toggles.grid {
            self.line_renderer.draw_grid(&mut render_pass, scene_extent);
        }

        // Draw axes
        if toggles.axes {
            self.line_renderer.draw_axes(&mut render_pass, scene_extent);
        }

        // Draw meshes
        self.mesh_renderer.draw(
            &mut render_pass,
            &scene_data.meshes,
            mesh_bind_groups,
            toggles.wireframe,
        );

        // Draw paths
        self.line_renderer
            .draw_paths(&mut render_pass, &scene_data.paths);

        // Draw points
        self.point_renderer
            .draw(&mut render_pass, &scene_data.points);
    }

    pub fn update_overlays(&mut self, device: &wgpu::Device, scene_extent: f32) {
        self.line_renderer.update_overlays(device, scene_extent);
    }
}
