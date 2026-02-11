// Compute volume (filled voxel count) and surface area (exposed face count)
// in a single pass over the voxel grid.
//
// counters[0] = number of filled voxels  (→ volume = count * scale³)
// counters[1] = number of exposed faces  (→ area   = count * scale²)
//
// A voxel is "filled" if its value is Surface (1) or Inside (2).
// An exposed face is one whose neighbor is unfilled or out-of-bounds.

@group(0) @binding(0) var<storage, read> grid: array<u32>;
@group(0) @binding(1) var<storage, read_write> counters: array<atomic<u32>>;
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

    // Count filled voxel for volume
    atomicAdd(&counters[0], 1u);

    // Count exposed faces for surface area
    var faces = 0u;

    // -X neighbor
    if gid.x == 0u || !is_filled(grid[grid_index(gid.x - 1u, gid.y, gid.z)]) {
        faces += 1u;
    }
    // +X neighbor
    if gid.x + 1u >= dims.x || !is_filled(grid[grid_index(gid.x + 1u, gid.y, gid.z)]) {
        faces += 1u;
    }
    // -Y neighbor
    if gid.y == 0u || !is_filled(grid[grid_index(gid.x, gid.y - 1u, gid.z)]) {
        faces += 1u;
    }
    // +Y neighbor
    if gid.y + 1u >= dims.y || !is_filled(grid[grid_index(gid.x, gid.y + 1u, gid.z)]) {
        faces += 1u;
    }
    // -Z neighbor
    if gid.z == 0u || !is_filled(grid[grid_index(gid.x, gid.y, gid.z - 1u)]) {
        faces += 1u;
    }
    // +Z neighbor
    if gid.z + 1u >= dims.z || !is_filled(grid[grid_index(gid.x, gid.y, gid.z + 1u)]) {
        faces += 1u;
    }

    atomicAdd(&counters[1], faces);
}
