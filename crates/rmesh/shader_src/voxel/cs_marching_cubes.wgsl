// GPU marching cubes with edge-indexed output.
//
// Each thread processes one cell (8 corner voxels), emits 0-5 triangles
// as edge ID triplets. Vertex positions are reconstructed on CPU from
// edge IDs — no vertex buffer needed on GPU.
//
// Edge ID = (ex + ey * dims.x + ez * dims.x * dims.y) * 3 + orientation
// where orientation: 0=x-aligned, 1=y-aligned, 2=z-aligned.
// The vertex sits at the midpoint of that edge.

struct Params {
    dims: vec3u,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
@group(0) @binding(2) var<storage, read> edge_table: array<u32>;
@group(0) @binding(3) var<storage, read> tri_table: array<i32>;
@group(0) @binding(4) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> tri_indices: array<u32>;

fn is_filled(x: u32, y: u32, z: u32) -> bool {
    if x >= p.dims.x || y >= p.dims.y || z >= p.dims.z {
        return false;
    }
    let idx = x + y * p.dims.x + z * p.dims.x * p.dims.y;
    let v = grid[idx];
    return v == 1u || v == 2u;
}

// Map a local MC edge (0-11) in cell (cx, cy, cz) to a global edge ID.
// Each grid position owns 3 edges (x/y/z aligned), giving a flat ID space.
fn edge_id(cx: u32, cy: u32, cz: u32, local_edge: u32) -> u32 {
    let dx = p.dims.x;
    let dxy = p.dims.x * p.dims.y;

    var ex: u32;
    var ey: u32;
    var ez: u32;
    var ori: u32;

    switch local_edge {
        //              grid position of edge       orientation
        case 0u  { ex = cx;      ey = cy;      ez = cz;      ori = 0u; } // c0-c1: x @ (cx, cy, cz)
        case 1u  { ex = cx + 1u; ey = cy;      ez = cz;      ori = 1u; } // c1-c2: y @ (cx+1, cy, cz)
        case 2u  { ex = cx;      ey = cy + 1u; ez = cz;      ori = 0u; } // c2-c3: x @ (cx, cy+1, cz)
        case 3u  { ex = cx;      ey = cy;      ez = cz;      ori = 1u; } // c3-c0: y @ (cx, cy, cz)
        case 4u  { ex = cx;      ey = cy;      ez = cz + 1u; ori = 0u; } // c4-c5: x @ (cx, cy, cz+1)
        case 5u  { ex = cx + 1u; ey = cy;      ez = cz + 1u; ori = 1u; } // c5-c6: y @ (cx+1, cy, cz+1)
        case 6u  { ex = cx;      ey = cy + 1u; ez = cz + 1u; ori = 0u; } // c6-c7: x @ (cx, cy+1, cz+1)
        case 7u  { ex = cx;      ey = cy;      ez = cz + 1u; ori = 1u; } // c7-c4: y @ (cx, cy, cz+1)
        case 8u  { ex = cx;      ey = cy;      ez = cz;      ori = 2u; } // c0-c4: z @ (cx, cy, cz)
        case 9u  { ex = cx + 1u; ey = cy;      ez = cz;      ori = 2u; } // c1-c5: z @ (cx+1, cy, cz)
        case 10u { ex = cx + 1u; ey = cy + 1u; ez = cz;      ori = 2u; } // c2-c6: z @ (cx+1, cy+1, cz)
        case 11u { ex = cx;      ey = cy + 1u; ez = cz;      ori = 2u; } // c3-c7: z @ (cx, cy+1, cz)
        default  { ex = 0u; ey = 0u; ez = 0u; ori = 0u; }
    }

    return (ex + ey * dx + ez * dxy) * 3u + ori;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= p.dims.x - 1u || gid.y >= p.dims.y - 1u || gid.z >= p.dims.z - 1u {
        return;
    }

    let x = gid.x;
    let y = gid.y;
    let z = gid.z;

    var cube_index = 0u;
    if is_filled(x,      y,      z     ) { cube_index |= 1u; }
    if is_filled(x + 1u, y,      z     ) { cube_index |= 2u; }
    if is_filled(x + 1u, y + 1u, z     ) { cube_index |= 4u; }
    if is_filled(x,      y + 1u, z     ) { cube_index |= 8u; }
    if is_filled(x,      y,      z + 1u) { cube_index |= 16u; }
    if is_filled(x + 1u, y,      z + 1u) { cube_index |= 32u; }
    if is_filled(x + 1u, y + 1u, z + 1u) { cube_index |= 64u; }
    if is_filled(x,      y + 1u, z + 1u) { cube_index |= 128u; }

    if cube_index == 0u || cube_index == 255u {
        return;
    }

    let edges = edge_table[cube_index];
    if edges == 0u {
        return;
    }

    let base_idx = cube_index * 16u;
    for (var i = 0u; i < 16u; i = i + 3u) {
        let e0 = tri_table[base_idx + i];
        if e0 < 0 {
            break;
        }
        let e1 = tri_table[base_idx + i + 1u];
        let e2 = tri_table[base_idx + i + 2u];

        let id0 = edge_id(x, y, z, u32(e0));
        let id1 = edge_id(x, y, z, u32(e1));
        let id2 = edge_id(x, y, z, u32(e2));

        let slot = atomicAdd(&counter, 1u);
        let off = slot * 3u;
        tri_indices[off + 0u] = id0;
        tri_indices[off + 1u] = id1;
        tri_indices[off + 2u] = id2;
    }
}
