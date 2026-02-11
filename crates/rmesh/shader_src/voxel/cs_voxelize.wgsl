// Triangle-box SAT voxelization shader.
//
// One invocation per triangle. For each triangle, determines which voxels
// overlap using a separating axis test (13 axes) and marks them as Surface.

struct Params {
    dims: vec3<u32>,
    num_triangles: u32,
    origin: vec3<f32>,
    scale: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Triangles: 9 floats per triangle (v0.xyz, v1.xyz, v2.xyz)
@group(0) @binding(1) var<storage, read> triangles: array<f32>;
// Voxel grid: 0=Undefined, 1=Surface, 2=Inside, 3=Outside
@group(0) @binding(2) var<storage, read_write> grid: array<atomic<u32>>;

fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.dims.x + z * params.dims.x * params.dims.y;
}

// Test if triangle overlaps an axis-aligned box using SAT.
// Box is centered at `center` with half-extent `h`.
fn tri_box_overlap(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>,
                   center: vec3<f32>, h: f32) -> bool {
    // Translate triangle to box center
    let a = v0 - center;
    let b = v1 - center;
    let c = v2 - center;

    let e0 = b - a;
    let e1 = c - b;
    let e2 = a - c;

    // Test 9 cross-product axes (3 edges x 3 box normals)
    // Axis: e0 x (1,0,0) = (0, -e0.z, e0.y)
    var p0 = a.z * b.y - a.y * b.z;
    var p1 = c.z * e0.y - c.y * e0.z;
    var r = h * (abs(e0.z) + abs(e0.y));
    if min(p0, p1) > r || max(p0, p1) < -r { return false; }

    // e0 x (0,1,0) = (e0.z, 0, -e0.x)
    p0 = a.x * b.z - a.z * b.x;
    p1 = c.x * e0.z - c.z * e0.x;
    r = h * (abs(e0.z) + abs(e0.x));
    if min(p0, p1) > r || max(p0, p1) < -r { return false; }

    // e0 x (0,0,1) = (-e0.y, e0.x, 0)
    p0 = a.y * b.x - a.x * b.y;
    p1 = c.y * e0.x - c.x * e0.y;
    r = h * (abs(e0.y) + abs(e0.x));
    if min(p0, p1) > r || max(p0, p1) < -r { return false; }

    // e1 x (1,0,0)
    p0 = b.z * c.y - b.y * c.z;
    p1 = a.z * e1.y - a.y * e1.z;
    r = h * (abs(e1.z) + abs(e1.y));
    if min(p0, p1) > r || max(p0, p1) < -r { return false; }

    // e1 x (0,1,0)
    p0 = b.x * c.z - b.z * c.x;
    p1 = a.x * e1.z - a.z * e1.x;
    r = h * (abs(e1.z) + abs(e1.x));
    if min(p0, p1) > r || max(p0, p1) < -r { return false; }

    // e1 x (0,0,1)
    p0 = b.y * c.x - b.x * c.y;
    p1 = a.y * e1.x - a.x * e1.y;
    r = h * (abs(e1.y) + abs(e1.x));
    if min(p0, p1) > r || max(p0, p1) < -r { return false; }

    // e2 x (1,0,0)
    p0 = c.z * a.y - c.y * a.z;
    p1 = b.z * e2.y - b.y * e2.z;
    r = h * (abs(e2.z) + abs(e2.y));
    if min(p0, p1) > r || max(p0, p1) < -r { return false; }

    // e2 x (0,1,0)
    p0 = c.x * a.z - c.z * a.x;
    p1 = b.x * e2.z - b.z * e2.x;
    r = h * (abs(e2.z) + abs(e2.x));
    if min(p0, p1) > r || max(p0, p1) < -r { return false; }

    // e2 x (0,0,1)
    p0 = c.y * a.x - c.x * a.y;
    p1 = b.y * e2.x - b.x * e2.y;
    r = h * (abs(e2.y) + abs(e2.x));
    if min(p0, p1) > r || max(p0, p1) < -r { return false; }

    // Test 3 AABB face normals
    let tri_min = min(min(a, b), c);
    let tri_max = max(max(a, b), c);
    if tri_min.x > h || tri_max.x < -h { return false; }
    if tri_min.y > h || tri_max.y < -h { return false; }
    if tri_min.z > h || tri_max.z < -h { return false; }

    // Test triangle normal
    let n = cross(e0, e1);
    let d = dot(n, a);
    let s = h * (abs(n.x) + abs(n.y) + abs(n.z));
    if d > s || d < -s { return false; }

    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tri_idx = gid.x;
    if tri_idx >= params.num_triangles { return; }

    let base = tri_idx * 9u;
    let v0 = vec3<f32>(triangles[base], triangles[base + 1u], triangles[base + 2u]);
    let v1 = vec3<f32>(triangles[base + 3u], triangles[base + 4u], triangles[base + 5u]);
    let v2 = vec3<f32>(triangles[base + 6u], triangles[base + 7u], triangles[base + 8u]);

    // Compute AABB of triangle in voxel space
    let inv_scale = 1.0 / params.scale;
    let tv0 = (v0 - params.origin) * inv_scale;
    let tv1 = (v1 - params.origin) * inv_scale;
    let tv2 = (v2 - params.origin) * inv_scale;

    let tri_min = min(min(tv0, tv1), tv2);
    let tri_max = max(max(tv0, tv1), tv2);

    let imin = vec3<u32>(clamp(vec3<i32>(floor(tri_min)), vec3<i32>(0), vec3<i32>(params.dims) - 1));
    let imax = vec3<u32>(clamp(vec3<i32>(ceil(tri_max)), vec3<i32>(0), vec3<i32>(params.dims) - 1));

    let half = params.scale * 0.5;

    for (var z = imin.z; z <= imax.z; z++) {
        for (var y = imin.y; y <= imax.y; y++) {
            for (var x = imin.x; x <= imax.x; x++) {
                let center = params.origin + vec3<f32>(
                    f32(x) * params.scale + half,
                    f32(y) * params.scale + half,
                    f32(z) * params.scale + half,
                );
                if tri_box_overlap(v0, v1, v2, center, half) {
                    atomicMax(&grid[voxel_index(x, y, z)], 1u);
                }
            }
        }
    }
}
