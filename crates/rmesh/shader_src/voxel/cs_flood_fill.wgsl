// Iterative flood fill for interior detection.
//
// Seeds boundary voxels as Outside (3), then propagates to Undefined (0)
// neighbors using 6-connectivity. After convergence, remaining Undefined
// voxels are marked as Inside (2).
//
// Values: 0=Undefined, 1=Surface, 2=Inside, 3=Outside

struct Params {
    dims: vec3<u32>,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grid_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> grid_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> changed: atomic<u32>;

fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.dims.x + z * params.dims.x * params.dims.y;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z {
        return;
    }

    let idx = voxel_index(gid.x, gid.y, gid.z);
    let val = grid_in[idx];

    // Only propagate to Undefined voxels
    if val != 0u {
        grid_out[idx] = val;
        return;
    }

    // Check if on boundary (boundary voxels that are Undefined become Outside)
    if gid.x == 0u || gid.x == params.dims.x - 1u ||
       gid.y == 0u || gid.y == params.dims.y - 1u ||
       gid.z == 0u || gid.z == params.dims.z - 1u {
        grid_out[idx] = 3u; // Outside
        atomicAdd(&changed, 1u);
        return;
    }

    // Check 6-connected neighbors for Outside propagation
    let neighbors = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1),
    );

    for (var i = 0u; i < 6u; i++) {
        let n = vec3<i32>(gid) + neighbors[i];
        let ni = voxel_index(u32(n.x), u32(n.y), u32(n.z));
        if grid_in[ni] == 3u {
            grid_out[idx] = 3u; // Outside
            atomicAdd(&changed, 1u);
            return;
        }
    }

    // Still undefined - keep as is (will be resolved in later passes)
    grid_out[idx] = 0u;
}

// Second pass: mark remaining Undefined as Inside
@compute @workgroup_size(4, 4, 4)
fn finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z {
        return;
    }

    let idx = voxel_index(gid.x, gid.y, gid.z);
    let val = grid_out[idx];

    if val == 0u {
        grid_out[idx] = 2u; // Inside
    }
}
