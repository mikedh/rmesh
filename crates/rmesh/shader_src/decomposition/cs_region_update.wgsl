// Update region IDs after a split decision.
//
// For each voxel belonging to the parent region, assigns it to either
// the left or right child region based on its position relative to the
// split plane.

struct SplitDecision {
    parent_region: u32,
    child_left: u32,
    child_right: u32,
    axis: u32,       // 0=X, 1=Y, 2=Z
    position: u32,   // split position along axis
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

struct Params {
    dims: vec3<u32>,
    num_splits: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> splits: array<SplitDecision>;
@group(0) @binding(2) var<storage, read_write> region_ids: array<u32>;

fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.dims.x + z * params.dims.x * params.dims.y;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z {
        return;
    }

    let idx = voxel_index(gid.x, gid.y, gid.z);
    let current_region = region_ids[idx];

    // Check all splits to find if this voxel's region was split
    for (var i = 0u; i < params.num_splits; i++) {
        let split = splits[i];
        if current_region != split.parent_region { continue; }

        let coord = select(select(gid.z, gid.y, split.axis == 1u), gid.x, split.axis == 0u);

        if coord < split.position {
            region_ids[idx] = split.child_left;
        } else {
            region_ids[idx] = split.child_right;
        }
        return;
    }
}
