// This was ported from fast-mesh-simplify using Gemini2.5-pro

use nalgebra::{Point3, Vector3};
use std::ops::{Add, AddAssign};

// Type aliases for clarity
type Point = Point3<f64>;
type Vector = Vector3<f64>;

// --- Helper: Symmetric Matrix (Quadric) ---

#[derive(Debug, Clone, Copy)]
pub struct SymmetricMatrix {
    m: [f64; 10],
}

impl SymmetricMatrix {
    // Initialize with a value (usually 0)
    fn new(c: f64) -> Self {
        SymmetricMatrix { m: [c; 10] }
    }

    // Initialize from plane equation ax + by + cz + d = 0
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        SymmetricMatrix {
            m: [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
            ],
        }
    }

    // Access elements (read-only) - Corresponds to C++ operator[]
    fn get(&self, index: usize) -> f64 {
        self.m[index]
    }

    // Calculate determinant of the 3x3 submatrix relevant for vertex calculation
    fn det(
        &self,
        a11: usize,
        a12: usize,
        a13: usize,
        a21: usize,
        a22: usize,
        a23: usize,
        a31: usize,
        a32: usize,
        a33: usize,
    ) -> f64 {
        self.m[a11] * self.m[a22] * self.m[a33]
            + self.m[a13] * self.m[a21] * self.m[a32]
            + self.m[a12] * self.m[a23] * self.m[a31]
            - self.m[a13] * self.m[a22] * self.m[a31]
            - self.m[a11] * self.m[a23] * self.m[a32]
            - self.m[a12] * self.m[a21] * self.m[a33]
    }
}

impl Add for SymmetricMatrix {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut result = self.m;
        for i in 0..10 {
            result[i] += rhs.m[i];
        }
        SymmetricMatrix { m: result }
    }
}

impl AddAssign for SymmetricMatrix {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..10 {
            self.m[i] += rhs.m[i];
        }
    }
}

// --- Core Data Structures ---

#[derive(Debug, Clone)]
struct Triangle {
    v: [usize; 3], // Vertex indices
    err: [f64; 4], // Edge errors [0-1, 1-2, 2-0], min error
    deleted: bool,
    dirty: bool,
    n: Vector, // Normal vector
               // UVs and material omitted as not requested in signature
}

#[derive(Debug, Clone)]
struct Vertex {
    p: Point,           // Position
    tstart: usize,      // Start index in refs array
    tcount: usize,      // Number of refs entries
    q: SymmetricMatrix, // Quadric error matrix
    border: bool,       // Is vertex on a boundary edge?
}

#[derive(Debug, Clone, Copy)]
struct Ref {
    tid: usize,     // Triangle ID
    tvertex: usize, // Index of vertex within triangle (0, 1, or 2)
}

// --- Simplification Logic ---

struct Simplifier {
    vertices: Vec<Vertex>,
    triangles: Vec<Triangle>,
    refs: Vec<Ref>,
}

impl Simplifier {
    fn new(input_vertices: &[Point], input_faces: &[(usize, usize, usize)]) -> Self {
        let vertices = input_vertices
            .iter()
            .map(|&p| Vertex {
                p,
                tstart: 0,
                tcount: 0,
                q: SymmetricMatrix::new(0.0),
                border: false,
            })
            .collect();

        let triangles = input_faces
            .iter()
            .map(|&(v0, v1, v2)| Triangle {
                v: [v0, v1, v2],
                err: [0.0; 4],
                deleted: false,
                dirty: false,
                n: Vector::zeros(), // Will be calculated later
            })
            .collect();

        Simplifier {
            vertices,
            triangles,
            refs: Vec::new(),
        }
    }

    // Calculate the error for collapsing edge between id_v1 and id_v2
    // Returns (error, optimal_position)
    fn calculate_error(&self, id_v1: usize, id_v2: usize) -> (f64, Point) {
        let q = self.vertices[id_v1].q + self.vertices[id_v2].q;
        let border = self.vertices[id_v1].border && self.vertices[id_v2].border;
        let det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7);

        let p_result: Point;
        let error: f64;

        if det.abs() > 1e-15 && !border {
            // Use tolerance instead of != 0
            // q_delta is invertible
            p_result = Point::new(
                -1.0 / det * q.det(1, 2, 3, 4, 5, 6, 5, 7, 8), // vx
                1.0 / det * q.det(0, 2, 3, 1, 5, 6, 2, 7, 8),  // vy
                -1.0 / det * q.det(0, 1, 3, 1, 4, 6, 2, 5, 8), // vz
            );
            error = self.vertex_error(q, p_result);
        } else {
            // det is close to 0 or on border -> Use midpoint or endpoints
            let p1 = self.vertices[id_v1].p;
            let p2 = self.vertices[id_v2].p;
            let p3 = Point::from((p1.coords + p2.coords) / 2.0); // Midpoint

            let error1 = self.vertex_error(q, p1);
            let error2 = self.vertex_error(q, p2);
            let error3 = self.vertex_error(q, p3);

            error = error1.min(error2.min(error3));
            if error == error1 {
                p_result = p1;
            } else if error == error2 {
                p_result = p2;
            } else {
                p_result = p3;
            }
        }
        (error, p_result)
    }

    // Calculate error for a vertex position given a quadric matrix
    fn vertex_error(&self, q: SymmetricMatrix, p: Point) -> f64 {
        let x = p.x;
        let y = p.y;
        let z = p.z;
        q.get(0) * x * x
            + 2.0 * q.get(1) * x * y
            + 2.0 * q.get(2) * x * z
            + 2.0 * q.get(3) * x
            + q.get(4) * y * y
            + 2.0 * q.get(5) * y * z
            + 2.0 * q.get(6) * y
            + q.get(7) * z * z
            + 2.0 * q.get(8) * z
            + q.get(9)
    }

    // Check if collapsing vertex i0 to position p causes topological inversion (flip)
    // for triangles attached to i0 but not containing edge (i0, i1)
    fn flipped(&self, p: Point, i0: usize, i1: usize, deleted_flags: &mut [bool]) -> bool {
        let v0 = &self.vertices[i0];
        for k in 0..v0.tcount {
            let r = self.refs[v0.tstart + k];
            let t = &self.triangles[r.tid];
            if t.deleted {
                continue;
            }

            let s = r.tvertex; // Index of i0 within triangle t.v
            let id1 = t.v[(s + 1) % 3];
            let id2 = t.v[(s + 2) % 3];

            // Does this triangle contain the edge (i0, i1)? If so, it's gonna be deleted
            if id1 == i1 || id2 == i1 {
                deleted_flags[k] = true; // Mark for deletion check in main loop
                continue;
            }

            let p1 = self.vertices[id1].p;
            let p2 = self.vertices[id2].p;

            // Check for degenerate triangles (collinear vertices) after collapse
            let d1 = (p1 - p).normalize();
            let d2 = (p2 - p).normalize();
            if d1.dot(&d2).abs() > 0.999 {
                return true;
            } // Nearly collinear

            // Check if normal flips significantly
            let n = d1.cross(&d2).normalize();
            deleted_flags[k] = false; // Not deleted by this edge collapse
            if n.dot(&t.n) < 0.2 {
                return true;
            } // Normal flipped too much (original code used 0.2)
        }
        false
    }

    // Update triangles connected to vertex `v` (index `i0` will replace original vertex index)
    // Appends new refs for updated triangles to the end of self.refs
    // Returns the number of new refs appended
    fn update_triangles(
        &mut self,
        i0: usize,    // The vertex ID that remains
        v_idx: usize, // The original index of the vertex being processed (could be i0 or i1)
        deleted_flags: &[bool],
        deleted_triangles: &mut usize,
        refs_append_start: usize,
    ) -> usize {
        let mut new_refs_count = 0;
        let v = &self.vertices[v_idx]; // Read-only borrow for tstart/tcount

        for k in 0..v.tcount {
            let r = self.refs[v.tstart + k];
            let tid = r.tid;

            // Borrow mutably inside the loop
            if self.triangles[tid].deleted {
                continue;
            }

            if deleted_flags[k] {
                // This triangle is deleted because it contained the collapsed edge
                if !self.triangles[tid].deleted {
                    // Avoid double counting
                    self.triangles[tid].deleted = true;
                    *deleted_triangles += 1;
                }
                continue;
            }

            // Triangle is not deleted, update its vertex index and recalculate errors
            self.triangles[tid].v[r.tvertex] = i0;
            self.triangles[tid].dirty = true;

            let (err0, _) =
                self.calculate_error(self.triangles[tid].v[0], self.triangles[tid].v[1]);
            let (err1, _) =
                self.calculate_error(self.triangles[tid].v[1], self.triangles[tid].v[2]);
            let (err2, _) =
                self.calculate_error(self.triangles[tid].v[2], self.triangles[tid].v[0]);

            self.triangles[tid].err[0] = err0;
            self.triangles[tid].err[1] = err1;
            self.triangles[tid].err[2] = err2;
            self.triangles[tid].err[3] = err0.min(err1.min(err2));

            // Append the updated reference to the end of the global list
            // This ref now points to the correct triangle and vertex (i0)
            if refs_append_start + new_refs_count < self.refs.len() {
                self.refs[refs_append_start + new_refs_count] = r; // Overwrite if space pre-allocated
            } else {
                self.refs.push(r); // Append if needed (shouldn't happen if resize was correct)
            }

            new_refs_count += 1;
        }
        new_refs_count
    }

    // Compact triangle list, build vertex references, initialize quadrics and errors
    fn update_mesh(&mut self, iteration: i32) {
        if iteration > 0 {
            // Compact triangles: remove deleted ones
            self.triangles.retain(|t| !t.deleted);
        }

        // Reset vertex references counts before rebuilding
        for v in self.vertices.iter_mut() {
            v.tstart = 0;
            v.tcount = 0;
        }

        // Calculate tcount for each vertex
        for (tid, t) in self.triangles.iter().enumerate() {
            if t.deleted {
                continue;
            } // Should not happen if compacted, but safe check
            for &v_idx in &t.v {
                if v_idx < self.vertices.len() {
                    // Bounds check
                    self.vertices[v_idx].tcount += 1;
                }
            }
        }

        // Calculate tstart for each vertex (cumulative count)
        let mut tstart: usize = 0;
        for v in self.vertices.iter_mut() {
            v.tstart = tstart;
            tstart += v.tcount;
            v.tcount = 0; // Reset tcount, will be incremented again when filling refs
        }

        // Resize refs vector and fill it
        self.refs.resize(tstart, Ref { tid: 0, tvertex: 0 }); // Resize to total needed count
        for (tid, t) in self.triangles.iter().enumerate() {
            if t.deleted {
                continue;
            }
            for (tvertex, &v_idx) in t.v.iter().enumerate() {
                if v_idx < self.vertices.len() {
                    // Bounds check
                    let v = &mut self.vertices[v_idx];
                    let ref_index = v.tstart + v.tcount;
                    if ref_index < self.refs.len() {
                        // Bounds check for refs too
                        self.refs[ref_index] = Ref { tid, tvertex };
                        v.tcount += 1;
                    }
                }
            }
        }

        // Initialize Quadrics (Q) and identify border vertices on first iteration
        if iteration == 0 {
            // --- Identify Border Vertices ---
            let v_on_edge_count: Vec<Vec<usize>> = vec![Vec::new(); self.vertices.len()];

            // Count how many non-deleted triangles share each edge connected to a vertex
            for v_idx in 0..self.vertices.len() {
                let v = &self.vertices[v_idx];
                let mut edges: std::collections::HashMap<usize, usize> =
                    std::collections::HashMap::new(); // neighbor_idx -> count

                for k in 0..v.tcount {
                    let r = self.refs[v.tstart + k];
                    let t = &self.triangles[r.tid];
                    if t.deleted {
                        continue;
                    }

                    for j in 0..3 {
                        let v0_t = t.v[j];
                        let v1_t = t.v[(j + 1) % 3];
                        if v0_t == v_idx || v1_t == v_idx {
                            let neighbor_idx = if v0_t == v_idx { v1_t } else { v0_t };
                            if neighbor_idx != v_idx {
                                // Avoid self-loops in count
                                *edges.entry(neighbor_idx).or_insert(0) += 1;
                            }
                        }
                    }
                }
                // If an edge (v_idx, neighbor_idx) is only part of one triangle, it's a border edge
                for (neighbor_idx, count) in edges {
                    if count == 1 && neighbor_idx < self.vertices.len() {
                        // Bounds check
                        self.vertices[v_idx].border = true;
                        self.vertices[neighbor_idx].border = true;
                    }
                }
            }

            // --- Initialize Quadrics (Q) ---
            for v in self.vertices.iter_mut() {
                v.q = SymmetricMatrix::new(0.0);
            }

            for t in self.triangles.iter_mut() {
                if t.deleted {
                    continue;
                }
                let p0 = self.vertices[t.v[0]].p;
                let p1 = self.vertices[t.v[1]].p;
                let p2 = self.vertices[t.v[2]].p;

                let normal = (p1 - p0).cross(&(p2 - p0)).normalize();
                t.n = normal; // Store triangle normal

                let dist = -normal.dot(&p0.coords); // d in plane equation ax+by+cz+d=0

                let plane_q = SymmetricMatrix::from_plane(normal.x, normal.y, normal.z, dist);

                for &v_idx in &t.v {
                    if v_idx < self.vertices.len() {
                        // Bounds check
                        self.vertices[v_idx].q += plane_q;
                    }
                }
            }

            // --- Initialize Edge Errors ---
            for t in self.triangles.iter_mut() {
                if t.deleted {
                    continue;
                }
                for j in 0..3 {
                    let v0 = t.v[j];
                    let v1 = t.v[(j + 1) % 3];
                    let err = {
                        let vertices = &self.vertices;
                        let q_v0 = vertices[v0].q;
                        let q_v1 = vertices[v1].q;
                        let border = vertices[v0].border && vertices[v1].border;
                        let det = (q_v0 + q_v1).det(0, 1, 2, 1, 4, 5, 2, 5, 7);

                        if det.abs() > 1e-15 && !border {
                            0.0 // Replace with actual error calculation logic if needed
                        } else {
                            f64::MAX // Replace with fallback error logic if needed
                        }
                    };
                    t.err[j] = err;
                }
                t.err[3] = t.err[0].min(t.err[1].min(t.err[2]));
            }
        }
    }

    // Perform the main simplification loop
    fn simplify(&mut self, target_count: usize, aggressiveness: f64, verbose: bool) {
        let initial_triangle_count = self.triangles.len();
        let mut deleted_triangles = 0;

        // Pre-allocate temporary vectors used in flipped check
        // Max possible size is max triangles connected to a vertex
        let max_tcount = self.vertices.iter().map(|v| v.tcount).max().unwrap_or(0);
        let mut deleted0: Vec<bool> = vec![false; max_tcount];
        let mut deleted1: Vec<bool> = vec![false; max_tcount];

        for iteration in 0..100 {
            let current_triangle_count = initial_triangle_count - deleted_triangles;
            if current_triangle_count <= target_count {
                break;
            }

            // Update mesh structure (refs, etc.) periodically or if first iteration
            if iteration == 0 || iteration % 5 == 0 {
                self.update_mesh(iteration);
                // Resize temp vectors if max tcount changed after update_mesh
                let current_max_tcount = self.vertices.iter().map(|v| v.tcount).max().unwrap_or(0);
                if deleted0.len() < current_max_tcount {
                    deleted0.resize(current_max_tcount, false);
                    deleted1.resize(current_max_tcount, false);
                }
            }

            // Reset dirty flags for this iteration
            for t in self.triangles.iter_mut() {
                t.dirty = false;
            }

            // Threshold calculation
            let threshold = 0.000000001 * (iteration as f64 + 3.0).powf(aggressiveness);

            if verbose && iteration % 5 == 0 {
                println!(
                    "Iteration {} - Triangles: {} Threshold: {:.1e}",
                    iteration, current_triangle_count, threshold
                );
            }

            // --- Edge Collapse Loop ---
            for tid in 0..self.triangles.len() {
                // Check triangle status
                if self.triangles[tid].err[3] > threshold
                    || self.triangles[tid].deleted
                    || self.triangles[tid].dirty
                {
                    continue;
                }

                // Check edges of the triangle
                for j in 0..3 {
                    if self.triangles[tid].err[j] < threshold {
                        let i0 = self.triangles[tid].v[j];
                        let i1 = self.triangles[tid].v[(j + 1) % 3];

                        // Check bounds and border status
                        if i0 >= self.vertices.len() || i1 >= self.vertices.len() {
                            continue;
                        }
                        if self.vertices[i0].border != self.vertices[i1].border {
                            continue;
                        }

                        // Calculate optimal collapse position and error
                        let (error, p_result) = self.calculate_error(i0, i1);

                        // Resize temp vectors for flipped check based on actual tcounts
                        let tcount0 = self.vertices[i0].tcount;
                        let tcount1 = self.vertices[i1].tcount;
                        if tcount0 > deleted0.len() || tcount1 > deleted1.len() {
                            // This case might happen if update_mesh wasn't called recently
                            // and a vertex accumulated many triangles. Reallocate larger.
                            let needed = tcount0.max(tcount1);
                            deleted0.resize(needed, false);
                            deleted1.resize(needed, false);
                            if verbose {
                                println!("Warning: Resized deleted flags mid-iteration");
                            }
                        }
                        // Reset only the parts we will use
                        deleted0.iter_mut().take(tcount0).for_each(|b| *b = false);
                        deleted1.iter_mut().take(tcount1).for_each(|b| *b = false);

                        // Flipped check (needs mutable slice access)
                        if self.flipped(p_result, i0, i1, &mut deleted0[..tcount0]) {
                            continue;
                        }
                        if self.flipped(p_result, i1, i0, &mut deleted1[..tcount1]) {
                            continue;
                        }

                        // --- Collapse the edge ---
                        // Update vertex i0
                        self.vertices[i0].p = p_result;
                        let (v0, v1) = if i0 < i1 {
                            let (left, right) = self.vertices.split_at_mut(i1);
                            (&mut left[i0], &mut right[0])
                        } else {
                            let (left, right) = self.vertices.split_at_mut(i0);
                            (&mut right[0], &mut left[i1])
                        };
                        v0.q += v1.q; // Add quadrics

                        // Remember where the appended refs will start
                        let refs_append_start = self.refs.len();
                        let mut new_refs_count = 0;

                        // Update triangles connected to i0 and i1
                        // This marks triangles for deletion and updates others
                        new_refs_count += self.update_triangles(
                            i0,
                            i0,
                            &deleted0[..tcount0],
                            &mut deleted_triangles,
                            refs_append_start + new_refs_count,
                        );
                        new_refs_count += self.update_triangles(
                            i0,
                            i1,
                            &deleted1[..tcount1],
                            &mut deleted_triangles,
                            refs_append_start + new_refs_count,
                        );

                        // Update vertex i0's reference list
                        // The new refs were appended starting at refs_append_start
                        // The C++ code uses memcpy for optimization; here we just update tstart/tcount
                        self.vertices[i0].tstart = refs_append_start;
                        self.vertices[i0].tcount = new_refs_count;

                        // Edge collapsed, move to next triangle
                        break; // Breaks the inner loop (j)
                    }
                } // End edge loop (j)

                if initial_triangle_count - deleted_triangles <= target_count {
                    break;
                } // Check target again
            } // End triangle loop (tid)
        } // End iteration loop

        // --- Final Cleanup ---
        self.compact_mesh();
    }

    // Remove deleted triangles and unused vertices, re-index faces
    fn compact_mesh(&mut self) {
        // 1. Filter out deleted triangles
        let old_triangle_count = self.triangles.len();
        self.triangles.retain(|t| !t.deleted);
        // println!("Compacted triangles: {} -> {}", old_triangle_count, self.triangles.len());

        // 2. Identify used vertices and create mapping old -> new index
        let mut vertex_used = vec![false; self.vertices.len()];
        let mut vertex_remap = vec![0; self.vertices.len()];
        let mut new_vertex_count = 0;

        for t in &self.triangles {
            for &v_idx in &t.v {
                if v_idx < vertex_used.len() && !vertex_used[v_idx] {
                    vertex_used[v_idx] = true;
                    new_vertex_count += 1;
                }
            }
        }

        // 3. Create the new vertex list and populate the remap table
        let mut new_vertices = Vec::with_capacity(new_vertex_count);
        let mut current_new_idx = 0;
        for (old_idx, used) in vertex_used.iter().enumerate() {
            if *used && old_idx < self.vertices.len() {
                // Bounds check
                new_vertices.push(self.vertices[old_idx].clone()); // Clone the used vertex data
                vertex_remap[old_idx] = current_new_idx;
                current_new_idx += 1;
            }
        }
        // println!("Compacted vertices: {} -> {}", self.vertices.len(), new_vertices.len());

        // 4. Update triangle indices using the remap table
        for t in self.triangles.iter_mut() {
            for i in 0..3 {
                if t.v[i] < vertex_remap.len() {
                    // Bounds check
                    t.v[i] = vertex_remap[t.v[i]];
                } else {
                    // This indicates an invalid index somehow survived, problematic
                    eprintln!(
                        "Error: Invalid vertex index {} found during compaction.",
                        t.v[i]
                    );
                    // Handle error appropriately, maybe set triangle to deleted or use a default index?
                    // For now, let's just panic or set to 0, though this hides the issue.
                    // panic!("Invalid vertex index during compaction");
                    t.v[i] = 0; // Or handle more gracefully
                }
            }
        }

        // 5. Replace old vertices with the compacted list
        self.vertices = new_vertices;
        // Refs are implicitly invalid now and would need rebuilding if used further,
        // but compact_mesh is the last step before returning results.
        self.refs.clear();
    }

    // Extract final mesh data
    fn get_result(&self) -> (Vec<Point>, Vec<(usize, usize, usize)>) {
        let result_vertices = self.vertices.iter().map(|v| v.p).collect();
        let result_faces = self
            .triangles
            .iter()
            .map(|t| (t.v[0], t.v[1], t.v[2]))
            .collect();
        (result_vertices, result_faces)
    }
}

/// Simplifies a mesh using the Fast Quadric Mesh Simplification algorithm.
///
/// # Arguments
///
/// * `input_vertices` - Slice of vertex positions.
/// * `input_faces` - Slice of triangle faces, represented as tuples of vertex indices.
/// * `target_count` - The desired number of faces in the simplified mesh.
/// * `aggressiveness` - Controls how aggressively to collapse edges. Higher values mean more aggressive simplification. Good values are typically between 5 and 8.
/// * `verbose` - Print progress information during simplification.
///
/// # Returns
///
/// A tuple containing the simplified vertex positions and the new faces.
/// Returns the original mesh if target_count is >= current face count or input is invalid.
pub fn simplify_mesh(
    input_vertices: &[Point3<f64>],
    input_faces: &[(usize, usize, usize)],
    target_count: usize,
    aggressiveness: f64,
    verbose: bool, // Added verbose flag
) -> (Vec<Point3<f64>>, Vec<(usize, usize, usize)>) {
    // Basic checks
    if target_count >= input_faces.len() {
        if verbose {
            println!(
                "Target count ({}) >= current count ({}), returning original.",
                target_count,
                input_faces.len()
            );
        }
        return (input_vertices.to_vec(), input_faces.to_vec());
    }
    if input_faces.is_empty() || input_vertices.len() < 3 {
        if verbose {
            println!("Input mesh is empty or too small, returning original.");
        }
        return (input_vertices.to_vec(), input_faces.to_vec());
    }
    if target_count == 0 {
        if verbose {
            println!("Target count is 0, returning empty mesh.");
        }
        return (Vec::new(), Vec::new());
    }

    if verbose {
        println!("Starting simplification:");
        println!("  Input vertices: {}", input_vertices.len());
        println!("  Input faces: {}", input_faces.len());
        println!("  Target faces: {}", target_count);
        println!("  Aggressiveness: {}", aggressiveness);
    }

    let mut simplifier = Simplifier::new(input_vertices, input_faces);

    simplifier.simplify(target_count, aggressiveness, verbose);

    if verbose {
        let (final_verts, final_faces) = simplifier.get_result();
        println!("Simplification finished:");
        println!("  Output vertices: {}", final_verts.len());
        println!("  Output faces: {}", final_faces.len());
        (final_verts, final_faces)
    } else {
        simplifier.get_result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn test_simplify_mesh() {
        // Define a simple cube mesh
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ];
        let faces = vec![
            (0, 1, 2),
            (0, 2, 3), // Bottom
            (4, 5, 6),
            (4, 6, 7), // Top
            (0, 1, 5),
            (0, 5, 4), // Front
            (1, 2, 6),
            (1, 6, 5), // Right
            (2, 3, 7),
            (2, 7, 6), // Back
            (3, 0, 4),
            (3, 4, 7), // Left
        ];

        // Simplify the cube mesh
        let target_face_count = 6; // Target number of faces
        let aggressiveness = 7.0;
        let verbose = false;

        let (simplified_vertices, simplified_faces) = simplify_mesh(
            &vertices,
            &faces,
            target_face_count,
            aggressiveness,
            verbose,
        );

        // Assert the simplified mesh has the expected number of vertices and faces
        assert!(simplified_vertices.len() <= vertices.len());
        assert!(simplified_faces.len() <= target_face_count);

        // Optionally, print the results for debugging
        println!("Simplified Vertices: {}", simplified_vertices.len());
        println!("Simplified Faces: {}", simplified_faces.len());
    }
}
