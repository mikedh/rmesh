mod app;
mod app2d;
pub(crate) mod gpu;
pub(crate) mod input;
pub(crate) mod render;
pub(crate) mod upload;
pub(crate) mod view2d;
mod viewer_thread;

use nalgebra::{Matrix4, Point2};
use rmesh::path::Path2D;
use rmesh::path::polygon::Polygon2D;
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

/// Intermediate geometry for the 2D viewer.
#[derive(Clone)]
pub struct View2DData {
    /// Polylines: each is (points, RGBA color).
    pub lines: Vec<(Vec<Point2<f64>>, [f32; 4])>,
    /// Filled polygons: each is (vertices, triangles, RGBA color).
    pub fills: Vec<(Vec<Point2<f64>>, Vec<[usize; 3]>, [f32; 4])>,
    /// Axis-aligned bounding box.
    pub bounds: (Point2<f64>, Point2<f64>),
}

/// Trait for displaying 2D geometry in a viewer window.
pub trait Viewer2D {
    fn view_2d_data(&self) -> View2DData;

    fn show_2d(&self) {
        self.show_2d_with_options(ViewerOptions {
            title: "rmesh 2D".to_string(),
            background: [1.0, 1.0, 1.0],
            ..Default::default()
        });
    }

    fn show_2d_with_options(&self, options: ViewerOptions) {
        env_logger::try_init().ok();
        let data = self.view_2d_data();
        app2d::run(&data, options);
    }
}

impl Viewer2D for Path2D {
    fn view_2d_data(&self) -> View2DData {
        let segments = self.discretize();
        let color = [0.0_f32, 0.8, 0.2, 1.0]; // green

        let mut lines = Vec::new();
        let mut min = Point2::new(f64::INFINITY, f64::INFINITY);
        let mut max = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);

        for seg in &segments {
            if seg.len() < 2 {
                continue;
            }
            for p in seg {
                min.x = min.x.min(p.x);
                min.y = min.y.min(p.y);
                max.x = max.x.max(p.x);
                max.y = max.y.max(p.y);
            }
            lines.push((seg.clone(), color));
        }

        if min.x > max.x {
            min = Point2::new(-1.0, -1.0);
            max = Point2::new(1.0, 1.0);
        }

        View2DData {
            lines,
            fills: Vec::new(),
            bounds: (min, max),
        }
    }
}

impl Viewer2D for Polygon2D {
    fn view_2d_data(&self) -> View2DData {
        let fill_color = [0.2_f32, 0.5, 0.8, 0.3];
        let outline_color = [0.2_f32, 0.4, 0.8, 1.0];

        let (bmin, bmax) = self
            .bounds()
            .unwrap_or((Point2::new(-1.0, -1.0), Point2::new(1.0, 1.0)));

        // Build lines from exterior + interiors
        let mut lines = Vec::new();
        let mut ext_line = self.exterior.clone();
        if !ext_line.is_empty() {
            ext_line.push(ext_line[0]); // close
            lines.push((ext_line, outline_color));
        }
        for hole in &self.interiors {
            let mut h = hole.clone();
            if !h.is_empty() {
                h.push(h[0]); // close
                lines.push((h, outline_color));
            }
        }

        // Triangulate for fill using earcut-style approach
        let fills = triangulate_polygon(self, fill_color);

        View2DData {
            lines,
            fills,
            bounds: (bmin, bmax),
        }
    }
}

impl Viewer2D for [Polygon2D] {
    fn view_2d_data(&self) -> View2DData {
        let fill_color = [0.2_f32, 0.5, 0.8, 0.3];
        let outline_color = [0.2_f32, 0.4, 0.8, 1.0];

        let mut lines = Vec::new();
        let mut fills = Vec::new();
        let mut min = Point2::new(f64::INFINITY, f64::INFINITY);
        let mut max = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);

        for poly in self {
            if let Some((bmin, bmax)) = poly.bounds() {
                min.x = min.x.min(bmin.x);
                min.y = min.y.min(bmin.y);
                max.x = max.x.max(bmax.x);
                max.y = max.y.max(bmax.y);
            }

            let mut ext_line = poly.exterior.clone();
            if !ext_line.is_empty() {
                ext_line.push(ext_line[0]);
                lines.push((ext_line, outline_color));
            }
            for hole in &poly.interiors {
                let mut h = hole.clone();
                if !h.is_empty() {
                    h.push(h[0]);
                    lines.push((h, outline_color));
                }
            }

            fills.extend(triangulate_polygon(poly, fill_color));
        }

        if min.x > max.x {
            min = Point2::new(-1.0, -1.0);
            max = Point2::new(1.0, 1.0);
        }

        View2DData {
            lines,
            fills,
            bounds: (min, max),
        }
    }
}

/// Triangulate a single polygon into fill data using earcut.
fn triangulate_polygon(
    poly: &Polygon2D,
    color: [f32; 4],
) -> Vec<(Vec<Point2<f64>>, Vec<[usize; 3]>, [f32; 4])> {
    // Build a flat vertex array and use earcut
    let mut flat: Vec<[f64; 2]> = Vec::new();
    let mut hole_starts = Vec::new();

    // Exterior ring
    for p in &poly.exterior {
        flat.push([p.x, p.y]);
    }

    // Interior rings
    for hole in &poly.interiors {
        hole_starts.push(flat.len());
        for p in hole {
            flat.push([p.x, p.y]);
        }
    }

    // Earcut triangulation
    let mut earcut = earcut::Earcut::new();
    let mut result: Vec<usize> = Vec::new();
    earcut.earcut(flat.iter().copied(), &hole_starts, &mut result);

    if result.is_empty() {
        return Vec::new();
    }

    let vertices: Vec<Point2<f64>> = flat.iter().map(|p| Point2::new(p[0], p[1])).collect();
    let tris: Vec<[usize; 3]> = result.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

    vec![(vertices, tris, color)]
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
