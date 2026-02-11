// Jump Flooding Algorithm - 3D seed initialization.
//
// Seeds surface voxels with their own coordinates. Non-surface voxels
// get an invalid sentinel value (0xFFFFFFFF for each component).

struct Params {
    dims: vec3<u32>,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
// Seeds: packed as (x | y<<10 | z<<20) or 0xFFFFFFFF for no seed
@group(0) @binding(2) var<storage, read_write> seeds: array<u32>;

fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.dims.x + z * params.dims.x * params.dims.y;
}

fn pack_seed(x: u32, y: u32, z: u32) -> u32 {
    return x | (y << 10u) | (z << 20u);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z {
        return;
    }

    let idx = voxel_index(gid.x, gid.y, gid.z);

    if grid[idx] == 1u { // Surface
        seeds[idx] = pack_seed(gid.x, gid.y, gid.z);
    } else {
        seeds[idx] = 0xFFFFFFFFu; // No seed
    }
}
