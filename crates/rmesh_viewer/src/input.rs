use nalgebra::Vector2;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::{Key, NamedKey};

use rmesh::scene::Trackball;

/// Toggle states for rendering options.
#[allow(clippy::struct_excessive_bools)]
pub struct RenderToggles {
    pub wireframe: bool,
    pub grid: bool,
    pub axes: bool,
    pub backface_culling: bool,
}

impl Default for RenderToggles {
    fn default() -> Self {
        Self {
            wireframe: false,
            grid: true,
            axes: true,
            backface_culling: false,
        }
    }
}

/// Input state for mouse tracking.
pub struct InputState {
    pub left_down: bool,
    pub middle_down: bool,
    pub ctrl_held: bool,
    pub last_mouse: Option<(f64, f64)>,
    pub window_size: (u32, u32),
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            left_down: false,
            middle_down: false,
            ctrl_held: false,
            last_mouse: None,
            window_size: (1280, 720),
        }
    }
}

/// Commands produced by input processing.
pub enum InputCommand {
    Rotate(Vector2<f64>),
    Pan(Vector2<f64>),
    Zoom(f64),
    ResetView,
    ToggleWireframe,
    ToggleGrid,
    ToggleAxes,
    ToggleCulling,
    ToggleFullscreen,
    Close,
}

impl InputState {
    pub fn handle_mouse_button(
        &mut self,
        button: MouseButton,
        state: ElementState,
    ) -> Option<InputCommand> {
        let pressed = state == ElementState::Pressed;
        match button {
            MouseButton::Left => self.left_down = pressed,
            MouseButton::Middle => self.middle_down = pressed,
            _ => {}
        }
        None
    }

    pub fn handle_mouse_move(&mut self, x: f64, y: f64) -> Option<InputCommand> {
        let result = if let Some((lx, ly)) = self.last_mouse {
            let dx = (x - lx) / f64::from(self.window_size.0);
            let dy = (y - ly) / f64::from(self.window_size.1);

            if self.left_down && (self.ctrl_held || self.middle_down) {
                Some(InputCommand::Pan(Vector2::new(dx, dy)))
            } else if self.left_down {
                Some(InputCommand::Rotate(Vector2::new(dx, dy)))
            } else if self.middle_down {
                Some(InputCommand::Pan(Vector2::new(dx, dy)))
            } else {
                None
            }
        } else {
            None
        };
        self.last_mouse = Some((x, y));
        result
    }

    #[allow(clippy::unused_self)]
    pub fn handle_scroll(&mut self, delta: MouseScrollDelta) -> Option<InputCommand> {
        let y = match delta {
            MouseScrollDelta::LineDelta(_, y) => f64::from(y),
            MouseScrollDelta::PixelDelta(pos) => pos.y / 50.0,
        };
        if y.abs() > 0.001 {
            let factor = if y > 0.0 { 0.9 } else { 1.1 };
            Some(InputCommand::Zoom(factor))
        } else {
            None
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn handle_key(&mut self, key: Key, state: ElementState) -> Option<InputCommand> {
        // Track ctrl state
        if key == Key::Named(NamedKey::Control) {
            self.ctrl_held = state == ElementState::Pressed;
            return None;
        }

        if state != ElementState::Pressed {
            return None;
        }

        match &key {
            Key::Named(NamedKey::Escape) => Some(InputCommand::Close),
            Key::Character(c) => match c.as_ref() {
                "q" => Some(InputCommand::Close),
                "z" => Some(InputCommand::ResetView),
                "w" => Some(InputCommand::ToggleWireframe),
                "g" => Some(InputCommand::ToggleGrid),
                "a" => Some(InputCommand::ToggleAxes),
                "c" => Some(InputCommand::ToggleCulling),
                "f" => Some(InputCommand::ToggleFullscreen),
                _ => None,
            },
            _ => None,
        }
    }

    #[allow(clippy::unused_self)]
    pub fn apply_command(
        &self,
        cmd: &InputCommand,
        trackball: &mut Trackball,
        toggles: &mut RenderToggles,
    ) {
        match cmd {
            InputCommand::Rotate(delta) => trackball.rotate(*delta),
            InputCommand::Pan(delta) => trackball.pan(*delta),
            InputCommand::Zoom(factor) => trackball.zoom(*factor),
            InputCommand::ToggleWireframe => toggles.wireframe = !toggles.wireframe,
            InputCommand::ToggleGrid => toggles.grid = !toggles.grid,
            InputCommand::ToggleAxes => toggles.axes = !toggles.axes,
            InputCommand::ToggleCulling => toggles.backface_culling = !toggles.backface_culling,
            InputCommand::ResetView | InputCommand::Close | InputCommand::ToggleFullscreen => {}
        }
    }
}
