// SDF-based voxel grid fill for analytic primitives.
//
// mode 0 = sphere:   params.x = radius
// mode 1 = box:      params.xyz = half-extents
// mode 2 = cylinder: params.x = radius, params.y = half-height, params.z = axis (0/1/2)

struct Params {
    dims: vec3u,
    mode: u32,
    origin: vec3f,
    pitch: f32,
    center: vec3f,
    _pad: u32,
    params: vec4f,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> grid: array<u32>;

fn sdf(pos: vec3f) -> f32 {
    let d = pos - p.center;
    if p.mode == 0u {
        // Sphere: distance to surface
        return length(d) - p.params.x;
    } else if p.mode == 1u {
        // Box: signed distance to axis-aligned box
        let q = abs(d) - p.params.xyz;
        return max(q.x, max(q.y, q.z));
    } else {
        // Cylinder: combine radial and axial distances
        let r = p.params.x;
        let hh = p.params.y;
        let axis = u32(p.params.z);
        var radial_sq: f32;
        var axial: f32;
        if axis == 0u {
            radial_sq = d.y * d.y + d.z * d.z;
            axial = abs(d.x);
        } else if axis == 1u {
            radial_sq = d.x * d.x + d.z * d.z;
            axial = abs(d.y);
        } else {
            radial_sq = d.x * d.x + d.y * d.y;
            axial = abs(d.z);
        }
        return max(sqrt(radial_sq) - r, axial - hh);
    }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if any(gid >= p.dims) {
        return;
    }

    let world = p.origin + vec3f(gid) * p.pitch + p.pitch * 0.5;
    let d = sdf(world);
    let idx = gid.x + gid.y * p.dims.x + gid.z * p.dims.x * p.dims.y;

    if d > 0.0 {
        grid[idx] = 0u; // Undefined (outside)
        return;
    }

    // Check 6 neighbors to classify Surface vs Inside
    let step = p.pitch;
    var on_surface = false;
    if sdf(world + vec3f(-step, 0.0, 0.0)) > 0.0 { on_surface = true; }
    if sdf(world + vec3f( step, 0.0, 0.0)) > 0.0 { on_surface = true; }
    if sdf(world + vec3f(0.0, -step, 0.0)) > 0.0 { on_surface = true; }
    if sdf(world + vec3f(0.0,  step, 0.0)) > 0.0 { on_surface = true; }
    if sdf(world + vec3f(0.0, 0.0, -step)) > 0.0 { on_surface = true; }
    if sdf(world + vec3f(0.0, 0.0,  step)) > 0.0 { on_surface = true; }

    grid[idx] = select(2u, 1u, on_surface); // Inside=2, Surface=1
}
