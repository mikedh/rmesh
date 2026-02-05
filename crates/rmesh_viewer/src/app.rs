use std::sync::Arc;

use nalgebra::Matrix4;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

use rmesh::scene::{Camera, Scene, Trackball};

use crate::ViewerOptions;
use crate::gpu::GpuContext;
use crate::input::{InputCommand, InputState, RenderToggles};
use crate::render::SceneRenderer;
use crate::render::mesh::MeshBindGroups;
use crate::upload::{self, SceneGpuData};

struct ViewerState {
    window: Arc<Window>,
    gpu: GpuContext,
    renderer: SceneRenderer,
    scene_data: SceneGpuData,
    mesh_bind_groups: Vec<MeshBindGroups>,
    trackball: Trackball,
    camera: Camera,
    input: InputState,
    toggles: RenderToggles,
    options: ViewerOptions,
    scene_extent: f32,
}

struct ViewerApp<'a> {
    scene: &'a Scene,
    options: ViewerOptions,
    state: Option<ViewerState>,
}

impl<'a> ViewerApp<'a> {
    fn new(scene: &'a Scene, options: ViewerOptions) -> Self {
        Self {
            scene,
            options,
            state: None,
        }
    }
}

impl ApplicationHandler for ViewerApp<'_> {
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
        let renderer = SceneRenderer::new(&gpu);

        let scene_data = upload::upload_scene(&gpu.device, &gpu.queue, self.scene);

        let mesh_bind_groups = renderer
            .mesh_renderer
            .prepare_bind_groups(&gpu.device, &scene_data.meshes);

        let mut trackball = Trackball::default();
        trackball.fit(scene_data.bounds_min, scene_data.bounds_max);

        #[allow(clippy::cast_possible_truncation)]
        let scene_extent = (scene_data.bounds_max - scene_data.bounds_min).norm() as f32;
        let scene_extent = if scene_extent > 0.001 {
            scene_extent
        } else {
            2.0
        };

        let camera = Camera::perspective(
            std::f64::consts::FRAC_PI_4,
            f64::from(scene_extent) * 0.001,
            f64::from(scene_extent) * 100.0,
        );

        let size = window.inner_size();
        let input = InputState {
            window_size: (size.width, size.height),
            ..Default::default()
        };

        let mut state = ViewerState {
            window,
            gpu,
            renderer,
            scene_data,
            mesh_bind_groups,
            trackball,
            camera,
            input,
            toggles: RenderToggles::default(),
            options: ViewerOptions {
                title: self.options.title.clone(),
                width: self.options.width,
                height: self.options.height,
                background: self.options.background,
            },
            scene_extent,
        };

        state
            .renderer
            .update_overlays(&state.gpu.device, scene_extent);

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
                state.input.window_size = (size.width, size.height);
                state.window.request_redraw();
            }
            WindowEvent::MouseInput {
                button, state: s, ..
            } => {
                if let Some(cmd) = state.input.handle_mouse_button(button, s) {
                    if matches!(cmd, InputCommand::Close) {
                        self.state.take();
                        event_loop.exit();
                        return;
                    }
                    handle_command(state, &cmd);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(cmd) = state.input.handle_mouse_move(position.x, position.y) {
                    handle_command(state, &cmd);
                    state.window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(cmd) = state.input.handle_scroll(delta) {
                    handle_command(state, &cmd);
                    state.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(cmd) = state.input.handle_key(event.logical_key, event.state) {
                    if matches!(cmd, InputCommand::Close) {
                        self.state.take();
                        event_loop.exit();
                        return;
                    }
                    handle_command(state, &cmd);
                    state.window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                state.input.ctrl_held = modifiers.state().control_key();
            }
            WindowEvent::RedrawRequested => {
                render_frame(state);
            }
            _ => {}
        }
    }
}

fn handle_command(state: &mut ViewerState, cmd: &InputCommand) {
    match cmd {
        InputCommand::Close => {
            return;
        }
        InputCommand::ResetView => {
            state
                .trackball
                .fit(state.scene_data.bounds_min, state.scene_data.bounds_max);
        }
        InputCommand::ToggleFullscreen => {
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
    }
    state
        .input
        .apply_command(cmd, &mut state.trackball, &mut state.toggles);
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

fn render_frame(state: &ViewerState) {
    let output = match state.gpu.surface.get_current_texture() {
        Ok(t) => t,
        Err(wgpu::SurfaceError::Lost) => {
            return;
        }
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

    // Update camera
    let view = state.trackball.view_matrix();
    let proj = state.camera.projection_matrix(state.gpu.aspect_ratio());
    let view_proj = mat4_f64_to_f32(&(proj * view));

    let cam_pos = state.trackball.position();
    #[allow(clippy::cast_possible_truncation)]
    let cam_pos_f32 = [cam_pos.x as f32, cam_pos.y as f32, cam_pos.z as f32];
    state.renderer.update_camera(
        &state.gpu.queue,
        &view_proj,
        cam_pos_f32,
        state.toggles.env_light,
    );

    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame_encoder"),
        });

    state.renderer.render(
        &mut encoder,
        &color_view,
        &state.gpu.depth_view,
        &state.scene_data,
        &state.mesh_bind_groups,
        &state.toggles,
        state.options.background,
        state.scene_extent,
    );

    state.gpu.queue.submit(Some(encoder.finish()));
    output.present();
}

/// Run the viewer event loop. Blocks until the window is closed.
///
/// Uses a singleton viewer thread so the event loop can be reused
/// across multiple calls (winit only allows one EventLoop per process).
pub fn run(scene: &Scene, options: ViewerOptions) {
    crate::viewer_thread::show_scene(scene, options);
}

/// Run on an existing event loop (called from viewer thread).
pub(crate) fn run_on(event_loop: &mut EventLoop<()>, scene: &Scene, options: ViewerOptions) {
    use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
    let mut app = ViewerApp::new(scene, options);
    let _ = event_loop.run_app_on_demand(&mut app);
}
