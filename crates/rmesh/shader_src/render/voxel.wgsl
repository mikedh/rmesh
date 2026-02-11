// Voxel ray-march shader - fullscreen triangle rendering of voxel grids

#include "uniforms.inc.wgsl"

struct VoxelUniforms {
    inv_view_proj: mat4x4<f32>,
    origin: vec4<f32>,   // xyz = world origin, w = scale
    dims: vec4<u32>,     // xyz = grid dims, w = unused
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> voxel_params: VoxelUniforms;
@group(1) @binding(1) var<storage, read> voxel_data: array<u32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate fullscreen triangle: vertices at (-1,-1), (3,-1), (-1,3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * voxel_params.dims.x + z * voxel_params.dims.x * voxel_params.dims.y;
}

fn sample_voxel(x: i32, y: i32, z: i32) -> u32 {
    if x < 0 || y < 0 || z < 0 {
        return 0u;
    }
    let ux = u32(x);
    let uy = u32(y);
    let uz = u32(z);
    if ux >= voxel_params.dims.x || uy >= voxel_params.dims.y || uz >= voxel_params.dims.z {
        return 0u;
    }
    return voxel_data[voxel_index(ux, uy, uz)];
}

fn intersect_aabb(ray_origin: vec3<f32>, ray_dir_inv: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let t1 = (box_min - ray_origin) * ray_dir_inv;
    let t2 = (box_max - ray_origin) * ray_dir_inv;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_enter, t_exit);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let scale = voxel_params.origin.w;
    let origin = voxel_params.origin.xyz;
    let dims = vec3<f32>(voxel_params.dims.xyz);

    // Reconstruct ray from screen coordinates
    let ndc = vec4<f32>(in.uv, 0.0, 1.0);
    let world_near = voxel_params.inv_view_proj * vec4<f32>(in.uv, -1.0, 1.0);
    let world_far = voxel_params.inv_view_proj * vec4<f32>(in.uv, 1.0, 1.0);
    let ray_origin = world_near.xyz / world_near.w;
    let ray_end = world_far.xyz / world_far.w;
    let ray_dir = normalize(ray_end - ray_origin);

    // AABB of voxel grid in world space
    let box_min = origin;
    let box_max = origin + dims * scale;

    let ray_dir_inv = 1.0 / ray_dir;
    let t_range = intersect_aabb(ray_origin, ray_dir_inv, box_min, box_max);

    if t_range.x > t_range.y || t_range.y < 0.0 {
        discard;
    }

    let t_start = max(t_range.x, 0.0);
    let inv_scale = 1.0 / scale;

    // DDA ray march through voxel grid
    var pos = ray_origin + ray_dir * (t_start + 0.001 * scale);
    let max_steps = u32(dims.x + dims.y + dims.z) * 2u;

    // Grid-space position
    var grid_pos = (pos - origin) * inv_scale;
    var cell = vec3<i32>(floor(grid_pos));

    let step = vec3<i32>(sign(ray_dir));
    let t_delta = abs(vec3<f32>(1.0) / ray_dir) * scale;
    var t_max_v = ((vec3<f32>(cell + max(step, vec3<i32>(0))) * scale + origin) - ray_origin) / ray_dir;

    for (var i = 0u; i < max_steps; i++) {
        if cell.x < 0 || cell.y < 0 || cell.z < 0 {
            // exited grid
            break;
        }
        if u32(cell.x) >= voxel_params.dims.x || u32(cell.y) >= voxel_params.dims.y || u32(cell.z) >= voxel_params.dims.z {
            break;
        }

        let val = sample_voxel(cell.x, cell.y, cell.z);
        if val == 1u || val == 2u {
            // Surface or Inside voxel - shade it
            // Estimate normal from gradient
            let nx = f32(i32(sample_voxel(cell.x + 1, cell.y, cell.z) > 0u)) - f32(i32(sample_voxel(cell.x - 1, cell.y, cell.z) > 0u));
            let ny = f32(i32(sample_voxel(cell.x, cell.y + 1, cell.z) > 0u)) - f32(i32(sample_voxel(cell.x, cell.y - 1, cell.z) > 0u));
            let nz = f32(i32(sample_voxel(cell.x, cell.y, cell.z + 1) > 0u)) - f32(i32(sample_voxel(cell.x, cell.y, cell.z - 1) > 0u));
            var normal = normalize(vec3<f32>(-nx, -ny, -nz));
            if length(vec3<f32>(nx, ny, nz)) < 0.001 {
                normal = normalize(camera.camera_pos.xyz - pos);
            }

            let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
            let ndl = max(dot(normal, light_dir), 0.0);
            let ambient = 0.15;

            var color: vec3<f32>;
            if val == 1u {
                color = vec3<f32>(0.4, 0.6, 0.8); // Surface: blue-ish
            } else {
                color = vec3<f32>(0.8, 0.4, 0.3); // Inside: red-ish
            }
            color = color * (ambient + ndl * 0.85);
            // Gamma
            color = pow(color, vec3<f32>(1.0 / 2.2));
            return vec4<f32>(color, 1.0);
        }

        // DDA step: advance to next cell boundary
        if t_max_v.x < t_max_v.y && t_max_v.x < t_max_v.z {
            cell.x += step.x;
            t_max_v.x += t_delta.x;
        } else if t_max_v.y < t_max_v.z {
            cell.y += step.y;
            t_max_v.y += t_delta.y;
        } else {
            cell.z += step.z;
            t_max_v.z += t_delta.z;
        }
    }

    discard;
}
