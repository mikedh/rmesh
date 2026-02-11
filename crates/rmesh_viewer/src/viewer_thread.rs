//! Singleton EventLoop stored in a thread-local.
//!
//! Winit only allows one `EventLoop` per process. We store it in a
//! thread-local `RefCell` and reuse it across multiple `.show()` calls
//! via `run_app_on_demand`.

use std::cell::RefCell;

use anyhow::{Context, Result};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::{View2DData, ViewerOptions};
use rmesh::scene::Scene;

thread_local! {
    static EVENT_LOOP: RefCell<Option<EventLoop<()>>> = const { RefCell::new(None) };
}

fn with_event_loop<F, R>(f: F) -> Result<R>
where
    F: FnOnce(&mut EventLoop<()>) -> Result<R>,
{
    EVENT_LOOP.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            let el = EventLoop::builder()
                .build()
                .context("failed to create event loop")?;
            el.set_control_flow(ControlFlow::Wait);
            *opt = Some(el);
        }
        let el = opt.as_mut().context("event loop not initialized")?;
        f(el)
    })
}

/// Show a 3D scene. Blocks until the window is closed.
pub(crate) fn show_scene(scene: &Scene, options: ViewerOptions) -> Result<()> {
    with_event_loop(|el| crate::app::run_on(el, scene, options))
}

/// Show 2D geometry. Blocks until the window is closed.
pub(crate) fn show_2d(data: &View2DData, options: ViewerOptions) -> Result<()> {
    with_event_loop(|el| crate::app2d::run_on(el, data, options))
}
