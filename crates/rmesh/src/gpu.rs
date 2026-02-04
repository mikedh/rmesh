//! GPU helpers: shared device, serialized readbacks.
//!
//! Multiple threads creating separate `wgpu::Device` instances and calling
//! `map_async` + `device.poll(Wait)` concurrently deadlock on many drivers.
//!
//! This module provides:
//! - [`request_device`] — returns a process-wide shared device/queue pair.
//! - [`gpu_read_buffer`] — maps, polls, and reads a `MAP_READ` buffer under a
//!   global lock so only one readback runs at a time.
//!
//! Direct calls to `wgpu::Device::poll` and `wgpu::BufferSlice::map_async`
//! are banned by clippy (see `clippy.toml`). Use [`gpu_read_buffer`] instead.

use std::sync::{Arc, Mutex, OnceLock};

/// Global lock that serializes GPU map/poll/read sequences.
static GPU_READBACK_LOCK: Mutex<()> = Mutex::new(());

/// Process-wide shared compute device.
static SHARED_DEVICE: OnceLock<Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)>> = OnceLock::new();

/// Return a shared `(Device, Queue)` pair, creating it on the first call.
///
/// Every caller in the process gets the same device, which avoids
/// driver-level deadlocks when multiple threads do GPU work in parallel.
pub fn request_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    SHARED_DEVICE
        .get_or_init(|| {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..wgpu::InstanceDescriptor::default()
            });

            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .ok()?;

            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("rmesh_compute"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    trace: wgpu::Trace::Off,
                }))
                .ok()?;

            Some((Arc::new(device), Arc::new(queue)))
        })
        .clone()
}

/// Map a `MAP_READ` buffer, poll the device, and return the contents as bytes.
///
/// The buffer must already be the target of a submitted copy command.
/// Acquires [`GPU_READBACK_LOCK`] so only one readback runs at a time.
#[allow(clippy::disallowed_methods)]
pub fn gpu_read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let _lock = GPU_READBACK_LOCK.lock().unwrap();
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            timeout: None,
            submission_index: None,
        })
        .ok();
    let data = slice.get_mapped_range();
    let result = data.to_vec();
    drop(data);
    buffer.unmap();
    result
}
