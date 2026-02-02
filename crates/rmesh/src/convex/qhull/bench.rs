use super::*;
use crate::convex::is_hull_valid_3d;
use crate::formatting::Table;
use std::time::Instant;

/// Simple deterministic LCG PRNG.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

/// Sweep point counts with varying hulls per level and print timing breakdown.
///
/// Run with:
/// ```sh
/// cargo test -p rmesh --release --features bench bench_sweep -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn bench_sweep() {
    // Scale down hulls per level for large point counts
    let hulls_for_level = |n: usize| -> usize {
        match n {
            0..=1000 => 100,
            1001..=10000 => 10,
            _ => 3,
        }
    };

    let levels: Vec<usize> = [
        4, 8, 16, 32, 64, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000,
    ]
    .into();

    // Pre-generate all point clouds so generation time isn't measured
    let mut rng = Lcg::new(42424242);
    let all_clouds: Vec<(usize, Vec<Vec<Point3<f64>>>)> = levels
        .iter()
        .map(|&n| {
            let hulls_per_level = hulls_for_level(n);
            let clouds: Vec<Vec<Point3<f64>>> = (0..hulls_per_level)
                .map(|_| {
                    (0..n)
                        .map(|_| {
                            Point3::new(
                                rng.next_range(-1.0, 1.0),
                                rng.next_range(-1.0, 1.0),
                                rng.next_range(-1.0, 1.0),
                            )
                        })
                        .collect()
                })
                .collect();
            (n, clouds)
        })
        .collect();

    let mut table = Table::new(&[
        "n",
        "hulls",
        "ok",
        "err",
        "total",
        "tol",
        "simplex",
        "partition",
        "loop",
        "extract",
        "validate",
    ]);

    let mut total_hulls = 0u64;
    let mut total_errors = 0u64;

    for (n, clouds) in &all_clouds {
        let n = *n;
        let mut t_tol = 0.0_f64;
        let mut t_simplex = 0.0_f64;
        let mut t_partition = 0.0_f64;
        let mut t_loop = 0.0_f64;
        let mut t_extract = 0.0_f64;
        let mut t_validate = 0.0_f64;
        let mut ok_count = 0u64;
        let mut err_count = 0u64;

        for pts in clouds {
            if pts.len() < 4 {
                err_count += 1;
                continue;
            }

            let t0 = Instant::now();
            let mut hull = QHull::new(pts);
            t_tol += t0.elapsed().as_secs_f64();

            let t1 = Instant::now();
            let simplex = hull.build_initial_simplex();
            t_simplex += t1.elapsed().as_secs_f64();

            let (i0, i1, i2, i3) = match simplex {
                Ok(s) => s,
                Err(_) => {
                    err_count += 1;
                    continue;
                }
            };

            let t2 = Instant::now();
            hull.create_initial_tetrahedron(i0, i1, i2, i3);
            hull.initial_partition(i0, i1, i2, i3);
            t_partition += t2.elapsed().as_secs_f64();

            let t3 = Instant::now();
            hull.build_hull();
            t_loop += t3.elapsed().as_secs_f64();

            let t4 = Instant::now();
            let faces = hull.to_result();
            t_extract += t4.elapsed().as_secs_f64();

            let t5 = Instant::now();
            is_hull_valid_3d(pts, &faces);
            t_validate += t5.elapsed().as_secs_f64();

            ok_count += 1;
        }

        let t_total = t_tol + t_simplex + t_partition + t_loop + t_extract;
        table.row(vec![
            format!("{n}"),
            format!("{}", clouds.len()),
            format!("{ok_count}"),
            format!("{err_count}"),
            format!("{t_total:.4}"),
            format!("{t_tol:.4}"),
            format!("{t_simplex:.4}"),
            format!("{t_partition:.4}"),
            format!("{t_loop:.4}"),
            format!("{t_extract:.4}"),
            format!("{t_validate:.4}"),
        ]);
        total_hulls += ok_count;
        total_errors += err_count;
    }

    println!("\n{table}");
    println!("{total_hulls} hulls computed, {total_errors} degenerate");
}

/// Run C qhull on a point cloud, return (faces, elapsed_secs).
fn run_c_qhull(pts: &[Point3<f64>]) -> (Vec<[usize; 3]>, f64) {
    let t0 = Instant::now();
    let qh = qhull::Qh::builder()
        .compute(true)
        .build_from_iter(pts.iter().map(|p| [p.x, p.y, p.z]))
        .expect("C qhull failed");
    let faces: Vec<[usize; 3]> = qh
        .simplices()
        .map(|simplex| {
            let verts: Vec<usize> = simplex
                .vertices()
                .expect("simplex has no vertices")
                .iter()
                .map(|v| v.index(&qh).expect("vertex has no index"))
                .collect();
            assert_eq!(verts.len(), 3, "C qhull returned non-triangular facet");
            [verts[0], verts[1], verts[2]]
        })
        .collect();
    let elapsed = t0.elapsed().as_secs_f64();
    (faces, elapsed)
}

/// Run our implementation, return (faces, elapsed_secs).
fn run_ours(pts: &[Point3<f64>]) -> (Vec<[usize; 3]>, f64) {
    let t0 = Instant::now();
    let faces = convex_hull_3d(pts).expect("our hull failed");
    let elapsed = t0.elapsed().as_secs_f64();
    (faces, elapsed)
}

/// Collect the set of vertex indices used in a face list.
fn hull_vertex_set(faces: &[[usize; 3]]) -> std::collections::BTreeSet<usize> {
    faces.iter().flat_map(|f| f.iter().copied()).collect()
}

/// Compare both implementations on a labeled set of point clouds, printing a table.
/// Rows with the same name are aggregated into a single output row.
fn compare_clouds(label: &str, clouds: &[(&str, Vec<Point3<f64>>)]) {
    struct Accum {
        n: usize,
        count: u64,
        t_ours: f64,
        t_c: f64,
        last_faces: usize,
        last_verts: usize,
        face_mismatches: u64,
        vert_mismatches: u64,
    }

    // Aggregate by name, preserving insertion order.
    let mut groups: Vec<(String, Accum)> = Vec::new();
    for (name, pts) in clouds {
        let (our_faces, dt_ours) = run_ours(pts);
        let (c_faces, dt_c) = run_c_qhull(pts);

        let our_verts = hull_vertex_set(&our_faces);
        let c_verts = hull_vertex_set(&c_faces);

        let face_mm = (our_faces.len() != c_faces.len()) as u64;
        let vert_mm = (our_verts != c_verts) as u64;

        if let Some((_, acc)) = groups.iter_mut().find(|(n, _)| n == name) {
            acc.count += 1;
            acc.t_ours += dt_ours;
            acc.t_c += dt_c;
            acc.last_faces = our_faces.len();
            acc.last_verts = our_verts.len();
            acc.face_mismatches += face_mm;
            acc.vert_mismatches += vert_mm;
        } else {
            groups.push((
                name.to_string(),
                Accum {
                    n: pts.len(),
                    count: 1,
                    t_ours: dt_ours,
                    t_c: dt_c,
                    last_faces: our_faces.len(),
                    last_verts: our_verts.len(),
                    face_mismatches: face_mm,
                    vert_mismatches: vert_mm,
                },
            ));
        }
    }

    let mut table = Table::new(&[
        "name", "n", "hulls", "ours(s)", "qhull(s)", "ratio", "faces", "verts", "<notes",
    ]);

    let mut total_ours = 0.0_f64;
    let mut total_c = 0.0_f64;

    for (name, acc) in &groups {
        let ratio = if acc.t_c > 0.0 {
            acc.t_ours / acc.t_c
        } else {
            f64::NAN
        };
        let mut notes = String::new();
        if acc.face_mismatches > 0 {
            notes += &format!("{} face mismatch ", acc.face_mismatches);
        }
        if acc.vert_mismatches > 0 {
            notes += &format!("{} vert mismatch ", acc.vert_mismatches);
        }

        table.row(vec![
            name.clone(),
            format!("{}", acc.n),
            format!("{}", acc.count),
            format!("{:.4}", acc.t_ours),
            format!("{:.4}", acc.t_c),
            format!("{ratio:.2}x"),
            format!("{}", acc.last_faces),
            format!("{}", acc.last_verts),
            notes,
        ]);

        total_ours += acc.t_ours;
        total_c += acc.t_c;
    }

    let ratio = if total_c > 0.0 {
        total_ours / total_c
    } else {
        f64::NAN
    };
    table.row(vec![
        "**TOTAL**".into(),
        String::new(),
        String::new(),
        format!("**{total_ours:.4}**"),
        format!("**{total_c:.4}**"),
        format!("**{ratio:.2}x**"),
        String::new(),
        String::new(),
        String::new(),
    ]);

    println!("\n{label}\n{table}");
}

/// Compare our convex hull against the C qhull library on random uniform clouds
/// and icosphere vertices.
///
/// Run with:
/// ```sh
/// cargo test -p rmesh --release --features bench bench_vs_qhull -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn bench_vs_qhull() {
    use crate::creation::create_icosphere;

    // --- Random uniform clouds ---
    let mut rng = Lcg::new(314159);
    let levels: &[(usize, usize)] = &[
        (10, 200),
        (50, 200),
        (100, 100),
        (500, 50),
        (1000, 20),
        (5000, 10),
        (10000, 5),
        (50000, 3),
        (100000, 3),
    ];

    let mut random_clouds: Vec<(&str, Vec<Point3<f64>>)> = Vec::new();
    // We need owned labels, but compare_clouds takes &str — use a Vec<String> for labels.
    let random_labels: Vec<String> = levels
        .iter()
        .map(|(n, h)| format!("rand {n}x{h}"))
        .collect();
    for (i, &(n, hulls)) in levels.iter().enumerate() {
        for _ in 0..hulls {
            let pts: Vec<Point3<f64>> = (0..n)
                .map(|_| {
                    Point3::new(
                        rng.next_range(-1.0, 1.0),
                        rng.next_range(-1.0, 1.0),
                        rng.next_range(-1.0, 1.0),
                    )
                })
                .collect();
            random_clouds.push((&random_labels[i], pts));
        }
    }
    compare_clouds("Random uniform clouds", &random_clouds);

    // --- Icosphere vertices (all points on hull) ---
    let sphere_labels: Vec<String> = (0..=7).map(|s| format!("ico sub={s}")).collect();
    let sphere_clouds: Vec<(&str, Vec<Point3<f64>>)> = (0..=7)
        .map(|subdivisions| {
            let mesh = create_icosphere(1.0, subdivisions);
            (&*sphere_labels[subdivisions], mesh.vertices)
        })
        .collect();
    compare_clouds("Icosphere vertices (all on hull)", &sphere_clouds);
}
