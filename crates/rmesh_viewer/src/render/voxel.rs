/// Placeholder voxel renderer. Full implementation would bind VoxelGrid's
/// GPU buffer directly and ray-march through it.
pub struct VoxelRenderer {
    // Will hold pipeline and bind group layout when implemented
}

impl VoxelRenderer {
    pub fn new_with_format(
        _device: &wgpu::Device,
        _camera_bgl: &wgpu::BindGroupLayout,
        _format: wgpu::TextureFormat,
    ) -> Self {
        Self {}
    }
}
