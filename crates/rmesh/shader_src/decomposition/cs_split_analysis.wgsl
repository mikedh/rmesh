// Per-region concavity detection for hierarchical splitting.
//
// For each active region, analyzes the distance field along each axis to
// find the "neck" (local minimum of max-distance profile). The neck
// indicates where the shape narrows, suggesting a good split plane.
//
// Output: per-region best split axis, position, and neck depth.

struct Params {
    dims: vec3<u32>,
    num_regions: u32,
    max_dim: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

struct SplitResult {
    axis: u32,       // 0=X, 1=Y, 2=Z
    position: u32,   // slice index along axis
    neck_depth: f32, // how pronounced the narrowing is (higher = better split)
    region_id: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> distance: array<f32>;
@group(0) @binding(2) var<storage, read> region_ids: array<u32>;
@group(0) @binding(3) var<storage, read> active_regions: array<u32>;
@group(0) @binding(4) var<storage, read_write> results: array<SplitResult>;

fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.dims.x + z * params.dims.x * params.dims.y;
}

// Analyze one axis for one region. Returns (best_position, neck_depth).
fn analyze_axis(region_id: u32, axis: u32) -> vec2<f32> {
    let dim = select(select(params.dims.z, params.dims.y, axis == 1u), params.dims.x, axis == 0u);

    // For each slice along the axis, find max distance within the region
    var max_dist_profile: array<f32, 1024>;
    var has_voxels: array<bool, 1024>;

    for (var s = 0u; s < dim && s < 1024u; s++) {
        var slice_max: f32 = 0.0;
        var found = false;

        // Iterate the other two dimensions
        let d1 = select(select(params.dims.x, params.dims.x, axis == 1u), params.dims.y, axis == 0u);
        let d2 = select(select(params.dims.y, params.dims.z, axis == 1u), params.dims.z, axis == 0u);

        for (var i = 0u; i < d1; i++) {
            for (var j = 0u; j < d2; j++) {
                var x: u32; var y: u32; var z: u32;
                if axis == 0u {
                    x = s; y = i; z = j;
                } else if axis == 1u {
                    x = i; y = s; z = j;
                } else {
                    x = i; y = j; z = s;
                }

                let idx = voxel_index(x, y, z);
                if region_ids[idx] == region_id {
                    let d = distance[idx];
                    slice_max = max(slice_max, d);
                    found = true;
                }
            }
        }
        max_dist_profile[s] = slice_max;
        has_voxels[s] = found;
    }

    // Find the local minimum of the max-distance profile (the "neck")
    // Skip first and last slices as they're boundary effects
    var best_pos: u32 = dim / 2u;
    var best_depth: f32 = 0.0;

    for (var s = 2u; s < dim - 2u && s < 1022u; s++) {
        if !has_voxels[s] { continue; }

        // Find max in left and right neighborhoods
        var left_max: f32 = 0.0;
        var right_max: f32 = 0.0;

        for (var k = 0u; k < s; k++) {
            if has_voxels[k] {
                left_max = max(left_max, max_dist_profile[k]);
            }
        }
        for (var k = s + 1u; k < dim && k < 1024u; k++) {
            if has_voxels[k] {
                right_max = max(right_max, max_dist_profile[k]);
            }
        }

        let surround = min(left_max, right_max);
        let depth = surround - max_dist_profile[s];

        if depth > best_depth {
            best_depth = depth;
            best_pos = s;
        }
    }

    return vec2<f32>(f32(best_pos), best_depth);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let region_idx = gid.x;
    if region_idx >= params.num_regions { return; }

    let region_id = active_regions[region_idx];

    // Analyze all 3 axes
    let result_x = analyze_axis(region_id, 0u);
    let result_y = analyze_axis(region_id, 1u);
    let result_z = analyze_axis(region_id, 2u);

    // Pick axis with deepest neck
    var best_axis = 0u;
    var best_pos = result_x.x;
    var best_depth = result_x.y;

    if result_y.y > best_depth {
        best_axis = 1u;
        best_pos = result_y.x;
        best_depth = result_y.y;
    }
    if result_z.y > best_depth {
        best_axis = 2u;
        best_pos = result_z.x;
        best_depth = result_z.y;
    }

    results[region_idx] = SplitResult(best_axis, u32(best_pos), best_depth, region_id);
}
