// Convert JFA seeds to Euclidean distance values.
//
// For each voxel, computes the Euclidean distance to its nearest surface
// voxel (as determined by the JFA seed buffer).

struct Params {
    dims: vec3<u32>,
    scale: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> seeds: array<u32>;
@group(0) @binding(2) var<storage, read_write> distance: array<f32>;

fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.dims.x + z * params.dims.x * params.dims.y;
}

fn unpack_seed(packed: u32) -> vec3<u32> {
    return vec3<u32>(
        packed & 0x3FFu,
        (packed >> 10u) & 0x3FFu,
        (packed >> 20u) & 0x3FFu,
    );
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z {
        return;
    }

    let idx = voxel_index(gid.x, gid.y, gid.z);
    let seed = seeds[idx];

    if seed == 0xFFFFFFFFu {
        distance[idx] = 1e30;
        return;
    }

    let s = unpack_seed(seed);
    let d = vec3<f32>(gid) - vec3<f32>(s);
    // Distance in voxel units scaled to world space
    distance[idx] = length(d) * params.scale;
}
