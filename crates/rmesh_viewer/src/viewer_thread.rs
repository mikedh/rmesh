//! Singleton EventLoop stored in a thread-local.
//!
//! Winit only allows one `EventLoop` per process. We store it in a
//! thread-local `RefCell` and reuse it across multiple `.show()` calls
//! via `run_app_on_demand`.

use std::cell::RefCell;

use winit::event_loop::{ControlFlow, EventLoop};

use crate::{View2DData, ViewerOptions};
use rmesh::scene::Scene;

thread_local! {
    static EVENT_LOOP: RefCell<EventLoop<()>> = RefCell::new({
        let el = EventLoop::builder().build().expect("failed to create event loop");
        el.set_control_flow(ControlFlow::Wait);
        el
    });
}

/// Show a 3D scene. Blocks until the window is closed.
pub(crate) fn show_scene(scene: Scene, options: ViewerOptions) {
    EVENT_LOOP.with(|el| {
        crate::app::run_on(&mut el.borrow_mut(), &scene, options);
    });
}

/// Show 2D geometry. Blocks until the window is closed.
pub(crate) fn show_2d(data: View2DData, options: ViewerOptions) {
    EVENT_LOOP.with(|el| {
        crate::app2d::run_on(&mut el.borrow_mut(), &data, options);
    });
}
