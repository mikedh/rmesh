struct CameraUniforms {
    @align(16) view_proj: mat4x4f,
    @align(16) camera_pos: vec4f,
    @align(4)  use_env_light: u32,
}

struct ModelUniforms {
    @align(16) model: mat4x4f,
    @align(16) normal_matrix: mat4x4f,
}

struct MaterialUniforms {
    @align(16) base_color: vec4f,
    @align(4)  metallic: f32,
    @align(4)  roughness: f32,
    @align(4)  use_vertex_color: u32,
    @align(4)  has_texture: u32,
    @align(16) emissive_factor: vec3f,
    @align(4)  occlusion_strength: f32,
    @align(4)  normal_scale: f32,
    @align(4)  has_mr_texture: u32,
    @align(4)  has_normal_texture: u32,
    @align(4)  has_occlusion_texture: u32,
    @align(4)  has_emissive_texture: u32,
}
