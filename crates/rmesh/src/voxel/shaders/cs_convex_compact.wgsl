// Pass 2: Support-plane filter + compaction.
//
// For each boundary voxel, check if it lies within `margin` of the
// extreme in at least one of 26 directions. If so, it's a hull
// candidate and gets appended to the compact output buffer.

struct Params {
    dims: vec3u,
    max_dim: u32,
    margin: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> region_ids: array<u32>;
@group(0) @binding(2) var<storage, read> extremes: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<atomic<u32>>;

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.dims.x + z * params.dims.x * params.dims.y;
}

// Same 26 directions as cs_convex_support.wgsl
const DIRS: array<vec3i, 26> = array<vec3i, 26>(
    vec3i( 1,  0,  0),
    vec3i(-1,  0,  0),
    vec3i( 0,  1,  0),
    vec3i( 0, -1,  0),
    vec3i( 0,  0,  1),
    vec3i( 0,  0, -1),
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

    if rid == 0xFFFFFFFFu {
        return;
    }

    // 6-neighbor boundary check
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

    // Support-plane filter: check if near extreme in any of 26 directions
    let bias = i32(3u * params.max_dim);
    var is_candidate = false;

    for (var k = 0u; k < 26u; k++) {
        let d = DIRS[k];
        let val = u32(dot(pos, d) + bias);
        let extreme = extremes[rid * 26u + k];
        if val + params.margin >= extreme {
            is_candidate = true;
            break;
        }
    }

    if !is_candidate {
        return;
    }

    // Append to output: [count, (rid, x, y, z), ...]
    let slot = atomicAdd(&output[0], 1u);
    let base = 1u + slot * 4u;
    atomicStore(&output[base], rid);
    atomicStore(&output[base + 1u], gid.x);
    atomicStore(&output[base + 2u], gid.y);
    atomicStore(&output[base + 3u], gid.z);
}
