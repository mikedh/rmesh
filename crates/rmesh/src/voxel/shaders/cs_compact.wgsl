// GPU-compact filled voxel coordinates into a dense output buffer.
//
// For each filled voxel (Surface=1 or Inside=2), atomically appends
// its (x, y, z) coordinates to the output buffer.
//
// output layout: [count, x0, y0, z0, x1, y1, z1, ...]
// counter is at output[0], entries start at output[1].

@group(0) @binding(0) var<storage, read> grid: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> dims: vec3u;

fn is_filled(val: u32) -> bool {
    return val == 1u || val == 2u;
}

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * dims.x + z * dims.x * dims.y;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if any(gid >= dims) {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    let val = grid[idx];

    if !is_filled(val) {
        return;
    }

    let slot = atomicAdd(&output[0], 1u);
    let base = 1u + slot * 3u;
    atomicStore(&output[base], gid.x);
    atomicStore(&output[base + 1u], gid.y);
    atomicStore(&output[base + 2u], gid.z);
}
