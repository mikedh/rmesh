mod app;
pub(crate) mod gpu;
pub(crate) mod input;
pub(crate) mod render;
pub(crate) mod upload;

use nalgebra::Matrix4;
use rmesh::scene::{Camera, Scene, Trackball};
use wgpu::TextureFormat;

use gpu::GpuContext;
use input::RenderToggles;
use render::SceneRenderer;

/// Options for configuring the viewer window.
pub struct ViewerOptions {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub background: [f32; 3],
}

impl Default for ViewerOptions {
    fn default() -> Self {
        Self {
            title: "rmesh viewer".to_string(),
            width: 1280,
            height: 720,
            background: [0.15, 0.15, 0.18],
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
            background: [0.15, 0.15, 0.18],
        }
    }
}

/// Trait for displaying scenes in an interactive viewer window.
pub trait SceneViewer {
    /// Open an interactive viewer window. Blocks until closed.
    fn show(&self);

    /// Open an interactive viewer window with custom options. Blocks until closed.
    fn show_with_options(&self, options: ViewerOptions);

    /// Render the scene to an RGBA pixel buffer (headless, no window).
    fn render_to_image(&self, options: &RenderOptions) -> Vec<u8>;
}

#[allow(clippy::cast_possible_truncation)]
fn mat4_f64_to_f32(m: &Matrix4<f64>) -> Matrix4<f32> {
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

impl SceneViewer for Scene {
    fn show(&self) {
        self.show_with_options(ViewerOptions::default());
    }

    fn show_with_options(&self, options: ViewerOptions) {
        env_logger::try_init().ok();
        app::run(self, options);
    }

    fn render_to_image(&self, options: &RenderOptions) -> Vec<u8> {
        let (device, queue) = GpuContext::create_device();
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
        let (_, depth_view) = GpuContext::create_depth_texture(&device, w, h);

        // Upload scene data
        let scene_data = upload::upload_scene(&device, &queue, self);

        // Create renderer for offscreen format
        let mut renderer = SceneRenderer::new_with_format(&device, &queue, format);
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
        let mut trackball = Trackball::default();
        trackball.fit(scene_data.bounds_min, scene_data.bounds_max);
        let camera = Camera::perspective(
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
        slice.map_async(wgpu::MapMode::Read, |_| {});
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
        rgba
    }
}
