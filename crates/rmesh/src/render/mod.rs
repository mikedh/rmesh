pub mod device;
pub mod fill;
pub mod line;
pub mod mesh;
pub mod point;
pub mod scene2d;
pub mod shaders;
pub mod upload;
pub mod view2d;
pub mod voxel;

use anyhow::Result;
use nalgebra::Matrix4;
use wgpu::TextureFormat;
use wgpu::util::DeviceExt;

use shaders::CameraUniforms;
use upload::SceneGpuData;

/// Shading mode for mesh rendering.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum ShadingMode {
    /// Angle-threshold smooth groups (sharp edges preserved at creases >30deg).
    #[default]
    Smooth,
    /// Per-face normals (standard flat shading).
    Flat,
    /// Fully smooth, no sharp edges anywhere (PI threshold).
    Full,
}

impl ShadingMode {
    pub fn next(self) -> Self {
        match self {
            Self::Smooth => Self::Flat,
            Self::Flat => Self::Full,
            Self::Full => Self::Smooth,
        }
    }
}

impl std::fmt::Display for ShadingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Smooth => write!(f, "Smooth"),
            Self::Flat => write!(f, "Flat"),
            Self::Full => write!(f, "Full"),
        }
    }
}

/// Toggle states for rendering options.
#[allow(clippy::struct_excessive_bools)]
pub struct RenderToggles {
    pub wireframe: bool,
    pub grid: bool,
    pub axes: bool,
    pub env_light: bool,
    pub shading_mode: ShadingMode,
}

impl Default for RenderToggles {
    fn default() -> Self {
        Self {
            wireframe: false,
            grid: false,
            axes: false,
            env_light: true,
            shading_mode: ShadingMode::default(),
        }
    }
}

/// Options for headless rendering.
pub struct RenderOptions {
    pub width: u32,
    pub height: u32,
    pub background: [f32; 3],
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            background: [0.9, 0.9, 0.92],
        }
    }
}

/// Convert a nalgebra Matrix4 to a flat [f32; 16] array (column-major).
pub fn mat4_to_array(m: &Matrix4<f32>) -> [f32; 16] {
    let s = m.as_slice();
    let mut out = [0.0f32; 16];
    out.copy_from_slice(s);
    out
}

#[allow(clippy::cast_possible_truncation)]
pub fn mat4_f64_to_f32(m: &Matrix4<f64>) -> Matrix4<f32> {
    Matrix4::new(
        m[(0, 0)] as f32,
        m[(0, 1)] as f32,
        m[(0, 2)] as f32,
        m[(0, 3)] as f32,
        m[(1, 0)] as f32,
        m[(1, 1)] as f32,
        m[(1, 2)] as f32,
        m[(1, 3)] as f32,
        m[(2, 0)] as f32,
        m[(2, 1)] as f32,
        m[(2, 2)] as f32,
        m[(2, 3)] as f32,
        m[(3, 0)] as f32,
        m[(3, 1)] as f32,
        m[(3, 2)] as f32,
        m[(3, 3)] as f32,
    )
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
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: TextureFormat) -> Self {
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
                view_proj: mat4_to_array(&Matrix4::<f32>::identity()),
                camera_pos: [0.0; 4],
                use_env_light: 1,
                _pad_0: [0; 12],
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
            mesh::MeshRenderer::new(device, queue, &camera_bind_group_layout, format);
        let line_renderer =
            line::LineRenderer::new_with_format(device, &camera_bind_group_layout, format);
        let point_renderer = point::PointRenderer::new(device, &camera_bind_group_layout, format);
        let voxel_renderer = voxel::VoxelRenderer::new(device, &camera_bind_group_layout, format);

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
        env_light: bool,
    ) {
        let uniforms = CameraUniforms {
            view_proj: mat4_to_array(view_proj),
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
            use_env_light: u32::from(env_light),
            _pad_0: [0; 12],
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

/// Render a scene to an RGBA pixel buffer (headless, no window).
pub fn render_to_image(scene: &crate::scene::Scene, options: &RenderOptions) -> Result<Vec<u8>> {
    let (device, queue) = device::create_device()?;
    let format = TextureFormat::Rgba8UnormSrgb;
    let w = options.width;
    let h = options.height;

    // Offscreen color texture
    let color_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen_color"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let (_, depth_view) = device::create_depth_texture(&device, w, h);

    // Upload scene data
    let scene_data = upload::upload_scene(&device, &queue, scene, ShadingMode::Smooth);

    // Create renderer for offscreen format
    let mut renderer = SceneRenderer::new(&device, &queue, format);
    let bind_groups = renderer
        .mesh_renderer
        .prepare_bind_groups(&device, &scene_data.meshes);

    #[allow(clippy::cast_possible_truncation)]
    let scene_extent = (scene_data.bounds_max - scene_data.bounds_min).norm() as f32;
    let scene_extent = if scene_extent > 0.001 {
        scene_extent
    } else {
        2.0
    };
    renderer.update_overlays(&device, scene_extent);

    // Camera auto-fit
    let mut trackball = crate::scene::Trackball::default();
    trackball.fit(scene_data.bounds_min, scene_data.bounds_max);
    let camera = crate::scene::Camera::perspective(
        std::f64::consts::FRAC_PI_4,
        f64::from(scene_extent) * 0.001,
        f64::from(scene_extent) * 100.0,
    );
    let view = trackball.view_matrix();
    let proj = camera.projection_matrix(f64::from(w) / f64::from(h));
    let view_proj = mat4_f64_to_f32(&(proj * view));
    let cam_pos = trackball.position();
    renderer.update_camera(
        &queue,
        &view_proj,
        #[allow(clippy::cast_possible_truncation)]
        [cam_pos.x as f32, cam_pos.y as f32, cam_pos.z as f32],
        true,
    );

    // Render
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let toggles = RenderToggles::default();
    renderer.render(
        &mut encoder,
        &color_view,
        &depth_view,
        &scene_data,
        &bind_groups,
        &toggles,
        options.background,
        scene_extent,
    );

    // Readback: wgpu requires 256-byte row alignment
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = w * bytes_per_pixel;
    let align = 256u32;
    let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(padded_bytes_per_row) * u64::from(h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        color_tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    // Map and strip padding
    let slice = output_buffer.slice(..);
    #[allow(clippy::disallowed_methods)]
    {
        slice.map_async(wgpu::MapMode::Read, |_| {});
    }
    #[allow(clippy::disallowed_methods)]
    device
        .poll(wgpu::PollType::Wait {
            timeout: None,
            submission_index: None,
        })
        .ok();

    let data = slice.get_mapped_range();
    let mut rgba = Vec::with_capacity((w * h * bytes_per_pixel) as usize);
    for row in 0..h {
        let start = (row * padded_bytes_per_row) as usize;
        let end = start + (w * bytes_per_pixel) as usize;
        rgba.extend_from_slice(&data[start..end]);
    }
    Ok(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create the full SceneRenderer to validate all shaders compile on the GPU.
    #[test]
    fn shaders_compile() {
        let (device, queue) = device::create_device().unwrap();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        SceneRenderer::new(&device, &queue, format);
    }
}
