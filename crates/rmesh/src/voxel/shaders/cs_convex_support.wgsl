// Pass 1: Boundary detection + 26-direction support-plane extremes.
//
// For each filled voxel that is a boundary of its region (at least one
// 6-neighbor has a different region_id), compute dot products with 26
// canonical directions and atomicMax into per-region extremes.

struct Params {
    dims: vec3u,
    max_dim: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> region_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> extremes: array<atomic<u32>>;

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.dims.x + z * params.dims.x * params.dims.y;
}

// 26 directions: 6 face + 12 edge + 8 corner normals
const DIRS: array<vec3i, 26> = array<vec3i, 26>(
    // 6 face normals
    vec3i( 1,  0,  0),
    vec3i(-1,  0,  0),
    vec3i( 0,  1,  0),
    vec3i( 0, -1,  0),
    vec3i( 0,  0,  1),
    vec3i( 0,  0, -1),
    // 12 edge normals
    vec3i( 1,  1,  0),
    vec3i( 1, -1,  0),
    vec3i(-1,  1,  0),
    vec3i(-1, -1,  0),
    vec3i( 1,  0,  1),
    vec3i( 1,  0, -1),
    vec3i(-1,  0,  1),
    vec3i(-1,  0, -1),
    vec3i( 0,  1,  1),
    vec3i( 0,  1, -1),
    vec3i( 0, -1,  1),
    vec3i( 0, -1, -1),
    // 8 corner normals
    vec3i( 1,  1,  1),
    vec3i( 1,  1, -1),
    vec3i( 1, -1,  1),
    vec3i( 1, -1, -1),
    vec3i(-1,  1,  1),
    vec3i(-1,  1, -1),
    vec3i(-1, -1,  1),
    vec3i(-1, -1, -1),
);

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if any(gid >= params.dims) {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    let rid = region_ids[idx];

    // Skip unfilled voxels
    if rid == 0xFFFFFFFFu {
        return;
    }

    // 6-neighbor boundary check: skip if all neighbors have same region
    let pos = vec3i(gid);
    var is_boundary = false;

    if gid.x == 0u || region_ids[grid_index(gid.x - 1u, gid.y, gid.z)] != rid {
        is_boundary = true;
    }
    if !is_boundary && (gid.x + 1u >= params.dims.x || region_ids[grid_index(gid.x + 1u, gid.y, gid.z)] != rid) {
        is_boundary = true;
    }
    if !is_boundary && (gid.y == 0u || region_ids[grid_index(gid.x, gid.y - 1u, gid.z)] != rid) {
        is_boundary = true;
    }
    if !is_boundary && (gid.y + 1u >= params.dims.y || region_ids[grid_index(gid.x, gid.y + 1u, gid.z)] != rid) {
        is_boundary = true;
    }
    if !is_boundary && (gid.z == 0u || region_ids[grid_index(gid.x, gid.y, gid.z - 1u)] != rid) {
        is_boundary = true;
    }
    if !is_boundary && (gid.z + 1u >= params.dims.z || region_ids[grid_index(gid.x, gid.y, gid.z + 1u)] != rid) {
        is_boundary = true;
    }

    if !is_boundary {
        return;
    }

    // Bias to make all dot products non-negative
    let bias = i32(3u * params.max_dim);

    // Update 26-direction extremes for this region
    for (var k = 0u; k < 26u; k++) {
        let d = DIRS[k];
        let val = u32(dot(pos, d) + bias);
        atomicMax(&extremes[rid * 26u + k], val);
    }
}
