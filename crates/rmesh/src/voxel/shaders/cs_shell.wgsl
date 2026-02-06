// Compute shell of a voxel grid: keep only boundary voxels.
//
// A filled voxel (Surface=1 or Inside=2) is "shell" if at least one
// 6-neighbor is unfilled (Outside, Undefined, or out-of-bounds).
// Interior filled voxels are zeroed to Undefined (0).

@group(0) @binding(0) var<storage, read> grid_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> grid_out: array<u32>;
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
    let val = grid_in[idx];

    if !is_filled(val) {
        grid_out[idx] = val;
        return;
    }

    // Check 6-neighbors: if any is unfilled or out-of-bounds, this is a shell voxel
    var is_boundary = false;

    if gid.x == 0u || !is_filled(grid_in[grid_index(gid.x - 1u, gid.y, gid.z)]) {
        is_boundary = true;
    }
    if gid.x + 1u >= dims.x || !is_filled(grid_in[grid_index(gid.x + 1u, gid.y, gid.z)]) {
        is_boundary = true;
    }
    if gid.y == 0u || !is_filled(grid_in[grid_index(gid.x, gid.y - 1u, gid.z)]) {
        is_boundary = true;
    }
    if gid.y + 1u >= dims.y || !is_filled(grid_in[grid_index(gid.x, gid.y + 1u, gid.z)]) {
        is_boundary = true;
    }
    if gid.z == 0u || !is_filled(grid_in[grid_index(gid.x, gid.y, gid.z - 1u)]) {
        is_boundary = true;
    }
    if gid.z + 1u >= dims.z || !is_filled(grid_in[grid_index(gid.x, gid.y, gid.z + 1u)]) {
        is_boundary = true;
    }

    if is_boundary {
        grid_out[idx] = val;
    } else {
        grid_out[idx] = 0u; // Undefined — interior voxel removed
    }
}
