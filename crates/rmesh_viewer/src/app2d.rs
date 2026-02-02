use std::sync::Arc;

use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

use crate::View2DData;
use crate::ViewerOptions;
use crate::gpu::GpuContext;
use crate::input::ZoomBoxState;
use crate::render::fill::GpuFill;
use crate::render::scene2d::Scene2DRenderer;
use crate::upload::{GpuPath, LineVertex};
use crate::view2d::View2D;

/// All GPU-uploaded 2D scene data.
struct Gpu2DData {
    fills: Vec<GpuFill>,
    outlines: Vec<GpuPath>,
}

struct Viewer2DState {
    window: Arc<Window>,
    gpu: GpuContext,
    renderer: Scene2DRenderer,
    view: View2D,
    gpu_data: Gpu2DData,
    grid_buf: Option<GpuPath>,
    axes_buf: Option<GpuPath>,
    zoom_box: ZoomBoxState,
    zoom_box_fill: Option<GpuFill>,
    zoom_box_lines: Option<GpuPath>,
    show_grid: bool,
    left_down: bool,
    right_down: bool,
    last_mouse: Option<(f64, f64)>,
    options: ViewerOptions,
}

struct Viewer2DApp<'a> {
    data: &'a View2DData,
    options: ViewerOptions,
    state: Option<Viewer2DState>,
}

impl<'a> Viewer2DApp<'a> {
    fn new(data: &'a View2DData, options: ViewerOptions) -> Self {
        Self {
            data,
            options,
            state: None,
        }
    }
}

impl ApplicationHandler for Viewer2DApp<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attrs = WindowAttributes::default()
            .with_title(&self.options.title)
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.options.width,
                self.options.height,
            ));

        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("failed to create window"),
        );

        let gpu = GpuContext::new(window.clone(), None);
        let renderer = Scene2DRenderer::new(&gpu);

        let view = View2D::new(self.data.bounds.0, self.data.bounds.1);

        let gpu_data = upload_2d_data(&gpu.device, &renderer, self.data);

        let grid_buf = Some(generate_grid(&gpu.device, &view, gpu.aspect_ratio()));
        let axes_buf = Some(generate_axes(&gpu.device, &view, gpu.aspect_ratio()));

        let state = Viewer2DState {
            window,
            gpu,
            renderer,
            view,
            gpu_data,
            grid_buf,
            axes_buf,
            zoom_box: ZoomBoxState::default(),
            zoom_box_fill: None,
            zoom_box_lines: None,
            show_grid: true,
            left_down: false,
            right_down: false,
            last_mouse: None,
            options: ViewerOptions {
                title: self.options.title.clone(),
                width: self.options.width,
                height: self.options.height,
                background: self.options.background,
            },
        };

        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                self.state.take();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                state.gpu.resize(size.width, size.height);
                regenerate_overlays(state);
                state.window.request_redraw();
            }
            WindowEvent::MouseInput {
                button, state: s, ..
            } => {
                let pressed = s == ElementState::Pressed;
                match button {
                    MouseButton::Left => state.left_down = pressed,
                    MouseButton::Right => {
                        if pressed {
                            // Start zoom box
                            if let Some((mx, my)) = state.last_mouse {
                                state.right_down = true;
                                state.zoom_box.active = true;
                                state.zoom_box.start = (mx, my);
                                state.zoom_box.current = (mx, my);
                            }
                        } else if state.zoom_box.active {
                            // End zoom box
                            state.right_down = false;
                            state.zoom_box.active = false;
                            let (sx, sy) = state.zoom_box.start;
                            let (cx, cy) = state.zoom_box.current;
                            let dx = (cx - sx).abs();
                            let dy = (cy - sy).abs();
                            if dx > 5.0 && dy > 5.0 {
                                let size = state.gpu.surface_config.width;
                                state.view.zoom_to_box(
                                    sx,
                                    sy,
                                    cx,
                                    cy,
                                    state.gpu.surface_config.width,
                                    state.gpu.surface_config.height,
                                );
                                let _ = size; // suppress unused
                            }
                            state.zoom_box_fill = None;
                            state.zoom_box_lines = None;
                            regenerate_overlays(state);
                            state.window.request_redraw();
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let x = position.x;
                let y = position.y;

                // Update title with cursor data coords
                let aspect = state.gpu.aspect_ratio();
                let data_pt = state.view.pixel_to_data(
                    x,
                    y,
                    state.gpu.surface_config.width,
                    state.gpu.surface_config.height,
                    aspect,
                );
                state.window.set_title(&format!(
                    "{} — x: {:.4}, y: {:.4}",
                    state.options.title, data_pt.x, data_pt.y
                ));

                // Pan with left drag
                if state.left_down {
                    if let Some((lx, ly)) = state.last_mouse {
                        let w = f64::from(state.gpu.surface_config.width);
                        let h = f64::from(state.gpu.surface_config.height);
                        let dx = (x - lx) / w;
                        let dy = (y - ly) / h;
                        state.view.pan(dx, dy, aspect);
                        regenerate_overlays(state);
                        state.window.request_redraw();
                    }
                }

                // Update zoom box
                if state.right_down && state.zoom_box.active {
                    state.zoom_box.current = (x, y);
                    update_zoom_box_gpu(state);
                    state.window.request_redraw();
                }

                state.last_mouse = Some((x, y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(y),
                    MouseScrollDelta::PixelDelta(pos) => pos.y / 50.0,
                };
                if y.abs() > 0.001 {
                    let factor = if y > 0.0 { 0.9 } else { 1.1 };
                    let (cx, cy) = state.last_mouse.unwrap_or((
                        f64::from(state.gpu.surface_config.width) / 2.0,
                        f64::from(state.gpu.surface_config.height) / 2.0,
                    ));
                    state.view.zoom_toward(
                        factor,
                        cx,
                        cy,
                        state.gpu.surface_config.width,
                        state.gpu.surface_config.height,
                    );
                    regenerate_overlays(state);
                    state.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                match &event.logical_key {
                    Key::Named(NamedKey::Escape) => {
                        self.state.take();
                        event_loop.exit();
                    }
                    Key::Character(c) => match c.as_ref() {
                        "q" => {
                            self.state.take();
                            event_loop.exit();
                        }
                        "z" => {
                            state.view.fit_to_data();
                            regenerate_overlays(state);
                            state.window.request_redraw();
                        }
                        "g" => {
                            state.show_grid = !state.show_grid;
                            state.window.request_redraw();
                        }
                        "f" => {
                            let current = state.window.fullscreen();
                            if current.is_some() {
                                state.window.set_fullscreen(None);
                            } else {
                                state
                                    .window
                                    .set_fullscreen(Some(Fullscreen::Borderless(None)));
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                render_frame_2d(state);
            }
            _ => {}
        }
    }
}

fn render_frame_2d(state: &Viewer2DState) {
    let output = match state.gpu.surface.get_current_texture() {
        Ok(t) => t,
        Err(wgpu::SurfaceError::Lost) => return,
        Err(wgpu::SurfaceError::OutOfMemory) => {
            log::error!("out of GPU memory");
            return;
        }
        Err(e) => {
            log::warn!("surface error: {e:?}");
            return;
        }
    };

    let color_view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let aspect = state.gpu.aspect_ratio();
    let view_proj = state.view.view_proj_matrix(aspect);
    state.renderer.update_camera(&state.gpu.queue, &view_proj);

    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame2d_encoder"),
        });

    state.renderer.render(
        &mut encoder,
        &color_view,
        if state.show_grid {
            state.grid_buf.as_ref()
        } else {
            None
        },
        if state.show_grid {
            state.axes_buf.as_ref()
        } else {
            None
        },
        &state.gpu_data.fills,
        &state.gpu_data.outlines,
        state.zoom_box_fill.as_ref(),
        state.zoom_box_lines.as_ref(),
        state.options.background,
    );

    state.gpu.queue.submit(Some(encoder.finish()));
    output.present();
}

/// Upload the View2DData (lines + fills) to GPU buffers.
#[allow(clippy::cast_possible_truncation)]
fn upload_2d_data(
    device: &wgpu::Device,
    renderer: &Scene2DRenderer,
    data: &View2DData,
) -> Gpu2DData {
    let mut fills = Vec::new();
    let mut outlines = Vec::new();

    // Upload fills
    for (vertices, triangles, color) in &data.fills {
        let verts: Vec<[f32; 2]> = vertices.iter().map(|p| [p.x as f32, p.y as f32]).collect();
        let indices: Vec<u32> = triangles
            .iter()
            .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
            .collect();
        if !indices.is_empty() {
            fills.push(
                renderer
                    .fill_renderer
                    .upload(device, &verts, &indices, *color),
            );
        }
    }

    // Upload line outlines
    for (points, color) in &data.lines {
        if points.len() < 2 {
            continue;
        }
        let mut vertices = Vec::with_capacity(points.len() * 2);
        for window in points.windows(2) {
            vertices.push(LineVertex {
                position: [window[0].x as f32, window[0].y as f32, 0.0],
                color: [color[0], color[1], color[2]],
            });
            vertices.push(LineVertex {
                position: [window[1].x as f32, window[1].y as f32, 0.0],
                color: [color[0], color[1], color[2]],
            });
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("outline2d_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        outlines.push(GpuPath {
            vertex_buffer,
            vertex_count: vertices.len() as u32,
        });
    }

    Gpu2DData { fills, outlines }
}

/// Generate nice tick positions using the {1, 2, 5} * 10^n algorithm.
fn nice_ticks(min: f64, max: f64, max_ticks: usize) -> Vec<f64> {
    let range = max - min;
    if range <= 0.0 || max_ticks < 2 {
        return vec![];
    }
    let rough_step = range / max_ticks as f64;
    let exp = rough_step.log10().floor();
    let base = 10.0_f64.powf(exp);
    let frac = rough_step / base;
    let nice_step = if frac <= 1.5 {
        base
    } else if frac <= 3.5 {
        2.0 * base
    } else if frac <= 7.5 {
        5.0 * base
    } else {
        10.0 * base
    };

    let start = (min / nice_step).ceil() as i64;
    let end = (max / nice_step).floor() as i64;
    (start..=end).map(|i| i as f64 * nice_step).collect()
}

/// Generate grid lines for the current view.
#[allow(clippy::cast_possible_truncation)]
fn generate_grid(device: &wgpu::Device, view: &View2D, aspect: f64) -> GpuPath {
    let hw = view.half_height * aspect;
    let left = view.center.x - hw;
    let right = view.center.x + hw;
    let bottom = view.center.y - view.half_height;
    let top = view.center.y + view.half_height;

    let grid_color = [0.85_f32, 0.85, 0.85];
    let max_ticks = 20;

    let x_ticks = nice_ticks(left, right, max_ticks);
    let y_ticks = nice_ticks(bottom, top, max_ticks);

    let mut verts = Vec::with_capacity((x_ticks.len() + y_ticks.len()) * 2);

    // Vertical lines
    for &x in &x_ticks {
        verts.push(LineVertex {
            position: [x as f32, bottom as f32, 0.0],
            color: grid_color,
        });
        verts.push(LineVertex {
            position: [x as f32, top as f32, 0.0],
            color: grid_color,
        });
    }

    // Horizontal lines
    for &y in &y_ticks {
        verts.push(LineVertex {
            position: [left as f32, y as f32, 0.0],
            color: grid_color,
        });
        verts.push(LineVertex {
            position: [right as f32, y as f32, 0.0],
            color: grid_color,
        });
    }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grid2d_vertices"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });

    GpuPath {
        vertex_buffer,
        vertex_count: verts.len() as u32,
    }
}

/// Generate X/Y axis lines through the origin.
#[allow(clippy::cast_possible_truncation)]
fn generate_axes(device: &wgpu::Device, view: &View2D, aspect: f64) -> GpuPath {
    let hw = view.half_height * aspect;
    let left = view.center.x - hw;
    let right = view.center.x + hw;
    let bottom = view.center.y - view.half_height;
    let top = view.center.y + view.half_height;

    let axis_color = [0.3_f32, 0.3, 0.3];

    let verts = vec![
        // X axis (horizontal through y=0)
        LineVertex {
            position: [left as f32, 0.0, 0.0],
            color: axis_color,
        },
        LineVertex {
            position: [right as f32, 0.0, 0.0],
            color: axis_color,
        },
        // Y axis (vertical through x=0)
        LineVertex {
            position: [0.0, bottom as f32, 0.0],
            color: axis_color,
        },
        LineVertex {
            position: [0.0, top as f32, 0.0],
            color: axis_color,
        },
    ];

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("axes2d_vertices"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });

    GpuPath {
        vertex_buffer,
        vertex_count: verts.len() as u32,
    }
}

/// Regenerate grid + axes for the current view.
fn regenerate_overlays(state: &mut Viewer2DState) {
    let aspect = state.gpu.aspect_ratio();
    state.grid_buf = Some(generate_grid(&state.gpu.device, &state.view, aspect));
    state.axes_buf = Some(generate_axes(&state.gpu.device, &state.view, aspect));
}

/// Update the zoom box GPU buffers from current state.
#[allow(clippy::cast_possible_truncation)]
fn update_zoom_box_gpu(state: &mut Viewer2DState) {
    let (sx, sy) = state.zoom_box.start;
    let (cx, cy) = state.zoom_box.current;
    let aspect = state.gpu.aspect_ratio();
    let w = state.gpu.surface_config.width;
    let h = state.gpu.surface_config.height;

    let d0 = state.view.pixel_to_data(sx, sy, w, h, aspect);
    let d1 = state.view.pixel_to_data(cx, cy, w, h, aspect);

    let x0 = d0.x.min(d1.x) as f32;
    let y0 = d0.y.min(d1.y) as f32;
    let x1 = d0.x.max(d1.x) as f32;
    let y1 = d0.y.max(d1.y) as f32;

    // Fill: 2 triangles
    let verts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    let indices = [0u32, 1, 2, 0, 2, 3];
    let fill_color = [0.2_f32, 0.5, 0.8, 0.15];
    state.zoom_box_fill = Some(state.renderer.fill_renderer.upload(
        &state.gpu.device,
        &verts,
        &indices,
        fill_color,
    ));

    // Lines: 4 edges
    let line_color = [0.2_f32, 0.5, 0.8];
    let line_verts = vec![
        LineVertex {
            position: [x0, y0, 0.0],
            color: line_color,
        },
        LineVertex {
            position: [x1, y0, 0.0],
            color: line_color,
        },
        LineVertex {
            position: [x1, y0, 0.0],
            color: line_color,
        },
        LineVertex {
            position: [x1, y1, 0.0],
            color: line_color,
        },
        LineVertex {
            position: [x1, y1, 0.0],
            color: line_color,
        },
        LineVertex {
            position: [x0, y1, 0.0],
            color: line_color,
        },
        LineVertex {
            position: [x0, y1, 0.0],
            color: line_color,
        },
        LineVertex {
            position: [x0, y0, 0.0],
            color: line_color,
        },
    ];

    let vertex_buffer = state
        .gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("zoom_box_lines"),
            contents: bytemuck::cast_slice(&line_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

    state.zoom_box_lines = Some(GpuPath {
        vertex_buffer,
        vertex_count: line_verts.len() as u32,
    });
}

/// Run the 2D viewer event loop. Blocks until the window is closed.
///
/// Uses a singleton viewer thread so the event loop can be reused
/// across multiple calls (winit only allows one EventLoop per process).
pub fn run(data: &View2DData, options: ViewerOptions) {
    crate::viewer_thread::show_2d(data.clone(), options);
}

/// Run on an existing event loop (called from viewer thread).
pub(crate) fn run_on(event_loop: &mut EventLoop<()>, data: &View2DData, options: ViewerOptions) {
    use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
    let mut app = Viewer2DApp::new(data, options);
    let _ = event_loop.run_app_on_demand(&mut app);
}
