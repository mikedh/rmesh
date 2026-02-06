// Voxel boolean operations: union (0), intersection (1), difference (2).
//
// Maps output voxel coords -> world position -> input A/B coords,
// samples both grids, applies the boolean, then classifies
// Surface vs Inside by checking 6 neighbors.

struct Params {
    out_dims: vec3u,
    mode: u32,          // 0=union, 1=intersection, 2=difference
    out_origin: vec3f,
    pitch: f32,
    a_dims: vec3u,
    _pad0: u32,
    a_origin: vec3f,
    _pad1: u32,
    b_dims: vec3u,
    _pad2: u32,
    b_origin: vec3f,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> grid_a: array<u32>;
@group(0) @binding(2) var<storage, read> grid_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> grid_out: array<u32>;

// Sample a grid: returns true if filled (Surface=1 or Inside=2).
fn sample_a(world: vec3f) -> bool {
    let rel = (world - p.a_origin) / p.pitch;
    let ic = vec3i(rel);
    if any(ic < vec3i(0)) || any(vec3u(ic) >= p.a_dims) {
        return false;
    }
    let coord = vec3u(ic);
    let idx = coord.x + coord.y * p.a_dims.x + coord.z * p.a_dims.x * p.a_dims.y;
    let v = grid_a[idx];
    return v == 1u || v == 2u;
}

fn sample_b(world: vec3f) -> bool {
    let rel = (world - p.b_origin) / p.pitch;
    let ic = vec3i(rel);
    if any(ic < vec3i(0)) || any(vec3u(ic) >= p.b_dims) {
        return false;
    }
    let coord = vec3u(ic);
    let idx = coord.x + coord.y * p.b_dims.x + coord.z * p.b_dims.x * p.b_dims.y;
    let v = grid_b[idx];
    return v == 1u || v == 2u;
}

fn boolean_op(a: bool, b: bool) -> bool {
    if p.mode == 0u {
        return a || b;   // union
    } else if p.mode == 1u {
        return a && b;   // intersection
    } else {
        return a && !b;  // difference
    }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if any(gid >= p.out_dims) {
        return;
    }

    let world = p.out_origin + vec3f(gid) * p.pitch + p.pitch * 0.5;
    let filled = boolean_op(sample_a(world), sample_b(world));

    let idx = gid.x + gid.y * p.out_dims.x + gid.z * p.out_dims.x * p.out_dims.y;

    if !filled {
        grid_out[idx] = 0u; // Undefined
        return;
    }

    // Check 6 neighbors for surface classification
    let step = p.pitch;
    var on_surface = false;
    let offsets = array<vec3f, 6>(
        vec3f(-step, 0.0, 0.0),
        vec3f( step, 0.0, 0.0),
        vec3f(0.0, -step, 0.0),
        vec3f(0.0,  step, 0.0),
        vec3f(0.0, 0.0, -step),
        vec3f(0.0, 0.0,  step),
    );

    for (var i = 0u; i < 6u; i = i + 1u) {
        let nw = world + offsets[i];
        if !boolean_op(sample_a(nw), sample_b(nw)) {
            on_surface = true;
            break;
        }
    }

    grid_out[idx] = select(2u, 1u, on_surface); // Inside=2, Surface=1
}
