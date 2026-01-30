// Jump Flooding Algorithm - 3D pass with 26 neighbors.
//
// Each voxel checks its 26 neighbors at distance `step_size` and keeps
// the closest seed. Runs with decreasing step sizes: max_dim/2, max_dim/4, ..., 1.

struct Params {
    dims: vec3<u32>,
    step_size: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> seeds_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> seeds_out: array<u32>;

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

fn dist_sq(a: vec3<u32>, b: vec3<u32>) -> u32 {
    let d = vec3<i32>(a) - vec3<i32>(b);
    return u32(d.x * d.x + d.y * d.y + d.z * d.z);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z {
        return;
    }

    let idx = voxel_index(gid.x, gid.y, gid.z);
    var best_seed = seeds_in[idx];
    var best_dist = 0xFFFFFFFFu;

    if best_seed != 0xFFFFFFFFu {
        best_dist = dist_sq(gid, unpack_seed(best_seed));
    }

    let step = i32(params.step_size);

    // Check 26 neighbors at step distance
    for (var dz: i32 = -step; dz <= step; dz += step) {
        for (var dy: i32 = -step; dy <= step; dy += step) {
            for (var dx: i32 = -step; dx <= step; dx += step) {
                if dx == 0 && dy == 0 && dz == 0 { continue; }

                let nx = i32(gid.x) + dx;
                let ny = i32(gid.y) + dy;
                let nz = i32(gid.z) + dz;

                if nx < 0 || nx >= i32(params.dims.x) ||
                   ny < 0 || ny >= i32(params.dims.y) ||
                   nz < 0 || nz >= i32(params.dims.z) {
                    continue;
                }

                let ni = voxel_index(u32(nx), u32(ny), u32(nz));
                let ns = seeds_in[ni];
                if ns == 0xFFFFFFFFu { continue; }

                let nd = dist_sq(gid, unpack_seed(ns));
                if nd < best_dist {
                    best_dist = nd;
                    best_seed = ns;
                }
            }
        }
    }

    seeds_out[idx] = best_seed;
}
