//! Mesh cleanup operations: merge vertices, remove degenerate faces, remove NaN/Inf.
//!
//! Unlike trimesh which uses global tolerances, all options are passed explicitly.

use ahash::AHashMap;

use nalgebra::Point3;
use rayon::prelude::*;

use crate::attributes::Attributes;

/// Options for mesh cleanup operations.
///
/// All options default to `None` or `false`, meaning no cleanup is performed
/// unless explicitly requested.
#[derive(Debug, Clone, Default)]
pub struct CleanupOptions {
    /// If Some(digits), merge duplicate vertices at this decimal precision.
    /// If None, don't merge vertices.
    /// Example: `Some(8)` means vertices within 1e-8 are considered equal.
    pub merge_vertices: Option<u32>,

    /// If Some(digits), also consider UV coordinates when merging vertices.
    /// If None, ignore UV coordinates (merge even if UVs differ).
    /// Only used when `merge_vertices` is Some.
    pub merge_tex: Option<u32>,

    /// If Some(digits), also consider vertex normals when merging vertices.
    /// If None, ignore normals (merge even if normals differ).
    /// Only used when `merge_vertices` is Some.
    pub merge_norm: Option<u32>,

    /// If Some(digits), remove degenerate faces where vertices are not unique
    /// at this decimal precision. If None, don't remove degenerate faces.
    /// Example: `Some(8)` means vertices within 1e-8 are considered equal.
    pub remove_degenerate: Option<u32>,

    /// Remove vertices and faces containing NaN or Inf values.
    pub remove_infinite: bool,

    /// Remove vertices that are not referenced by any face.
    pub remove_unreferenced: bool,
}

/// Result of a cleanup operation.
#[derive(Debug, Clone, Default)]
pub struct CleanupResult {
    /// New vertex positions after cleanup.
    pub vertices: Vec<Point3<f64>>,
    /// New face indices after cleanup.
    pub faces: Vec<[usize; 3]>,
    /// New vertex attributes after cleanup.
    pub attributes_vertex: Attributes,
    /// New face attributes after cleanup.
    pub attributes_face: Attributes,

    /// Number of vertices removed.
    pub vertices_removed: usize,
    /// Number of vertices merged.
    pub vertices_merged: usize,
    /// Number of degenerate faces removed.
    pub degenerate_faces_removed: usize,
    /// Number of infinite values removed.
    pub infinite_removed: usize,
}

/// Perform cleanup on mesh data.
#[allow(clippy::cast_possible_truncation)] // Coordinate values at typical mesh scales fit in i64
pub fn cleanup(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    attributes_vertex: Option<&Attributes>,
    attributes_face: Option<&Attributes>,
    options: &CleanupOptions,
) -> CleanupResult {
    let mut result = CleanupResult {
        vertices: vertices.to_vec(),
        faces: faces.to_vec(),
        attributes_vertex: attributes_vertex.cloned().unwrap_or_default(),
        attributes_face: attributes_face.cloned().unwrap_or_default(),
        ..Default::default()
    };

    // Step 1: Remove infinite values (NaN/Inf in vertices)
    if options.remove_infinite {
        let original_vertex_count = result.vertices.len();
        let original_face_count = result.faces.len();

        // Find vertices with infinite values
        let vertex_valid: Vec<bool> = result
            .vertices
            .par_iter()
            .map(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite())
            .collect();

        // Find faces referencing invalid vertices
        let face_valid: Vec<bool> = result
            .faces
            .par_iter()
            .map(|f| {
                f[0] < vertex_valid.len()
                    && f[1] < vertex_valid.len()
                    && f[2] < vertex_valid.len()
                    && vertex_valid[f[0]]
                    && vertex_valid[f[1]]
                    && vertex_valid[f[2]]
            })
            .collect();

        apply_face_mask(&mut result, &face_valid);

        // Rebuild vertex map after face filtering
        let mut referenced = vec![false; result.vertices.len()];
        for f in &result.faces {
            referenced[f[0]] = true;
            referenced[f[1]] = true;
            referenced[f[2]] = true;
        }

        // Combine with infinite check
        let vertex_mask: Vec<bool> = vertex_valid
            .iter()
            .zip(referenced.iter())
            .map(|(&valid, &used)| valid && used)
            .collect();

        apply_vertex_mask(&mut result, &vertex_mask);

        result.infinite_removed = (original_vertex_count - result.vertices.len())
            + (original_face_count - result.faces.len());
    }

    // Step 2: Remove degenerate faces
    if let Some(digits) = options.remove_degenerate {
        let original_count = result.faces.len();
        let scale = 10f64.powi(digits.cast_signed());

        // Quantize vertex positions
        let quantized: Vec<[i64; 3]> = result
            .vertices
            .par_iter()
            .map(|v| {
                [
                    (v.x * scale).round() as i64,
                    (v.y * scale).round() as i64,
                    (v.z * scale).round() as i64,
                ]
            })
            .collect();

        let face_valid: Vec<bool> = result
            .faces
            .par_iter()
            .map(|f| {
                if f[0] >= quantized.len() || f[1] >= quantized.len() || f[2] >= quantized.len() {
                    return false;
                }
                let q0 = &quantized[f[0]];
                let q1 = &quantized[f[1]];
                let q2 = &quantized[f[2]];
                q0 != q1 && q1 != q2 && q0 != q2
            })
            .collect();

        apply_face_mask(&mut result, &face_valid);
        result.degenerate_faces_removed = original_count - result.faces.len();
    }

    // Step 3: Merge vertices
    if let Some(digits) = options.merge_vertices
        && !result.vertices.is_empty()
    {
        let original_count = result.vertices.len();
        let scale_vertex = 10f64.powi(digits.cast_signed());

        // Get vertex normals and UVs if we need to consider them
        let (normals, scale_norm) = match options.merge_norm {
            Some(d) => (
                result.attributes_vertex.normals.first(),
                10f64.powi(d.cast_signed()),
            ),
            None => (None, 0.0),
        };
        let (uvs, scale_uv) = match options.merge_tex {
            Some(d) => (
                result.attributes_vertex.uv.first(),
                10f64.powi(d.cast_signed()),
            ),
            None => (None, 0.0),
        };

        // Create vertex keys - this determines which vertices are "equal"
        let vertex_keys: Vec<VertexKey> = result
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut key = VertexKey {
                    x: (v.x * scale_vertex).round() as i64,
                    y: (v.y * scale_vertex).round() as i64,
                    z: (v.z * scale_vertex).round() as i64,
                    nx: 0,
                    ny: 0,
                    nz: 0,
                    u: 0,
                    v: 0,
                };

                if let Some(n) = normals
                    && i < n.len()
                {
                    key.nx = (n[i].x * scale_norm).round() as i64;
                    key.ny = (n[i].y * scale_norm).round() as i64;
                    key.nz = (n[i].z * scale_norm).round() as i64;
                }

                if let Some(uv) = uvs
                    && i < uv.len()
                {
                    key.u = (uv[i].x * scale_uv).round() as i64;
                    key.v = (uv[i].y * scale_uv).round() as i64;
                }

                key
            })
            .collect();

        // Find unique vertices and build inverse mapping
        let mut unique_map: AHashMap<VertexKey, usize> = AHashMap::new();
        let mut inverse: Vec<usize> = vec![0; result.vertices.len()];
        let mut unique_indices: Vec<usize> = Vec::new();

        for (i, key) in vertex_keys.iter().enumerate() {
            if let Some(&existing) = unique_map.get(key) {
                inverse[i] = existing;
            } else {
                let new_idx = unique_indices.len();
                unique_map.insert(key.clone(), new_idx);
                inverse[i] = new_idx;
                unique_indices.push(i);
            }
        }

        // Build new vertex array
        let new_vertices: Vec<Point3<f64>> =
            unique_indices.iter().map(|&i| result.vertices[i]).collect();

        // Update face indices
        let new_faces: Vec<[usize; 3]> = result
            .faces
            .iter()
            .map(|f| [inverse[f[0]], inverse[f[1]], inverse[f[2]]])
            .collect();

        // Update vertex attributes
        let mut new_attrs = Attributes::default();
        if let Some(normals) = result.attributes_vertex.normals.first() {
            new_attrs.normals.push(
                unique_indices
                    .iter()
                    .filter_map(|&i| normals.get(i).copied())
                    .collect(),
            );
        }
        if let Some(uvs) = result.attributes_vertex.uv.first() {
            new_attrs.uv.push(
                unique_indices
                    .iter()
                    .filter_map(|&i| uvs.get(i).copied())
                    .collect(),
            );
        }
        if let Some(colors) = result.attributes_vertex.colors.first() {
            new_attrs.colors.push(
                unique_indices
                    .iter()
                    .filter_map(|&i| colors.get(i).copied())
                    .collect(),
            );
        }

        result.vertices = new_vertices;
        result.faces = new_faces;
        result.attributes_vertex = new_attrs;
        result.vertices_merged = original_count - result.vertices.len();
    }

    // Step 4: Remove unreferenced vertices
    if options.remove_unreferenced && !result.faces.is_empty() {
        let original_count = result.vertices.len();

        let mut referenced = vec![false; result.vertices.len()];
        for f in &result.faces {
            if f[0] < referenced.len() {
                referenced[f[0]] = true;
            }
            if f[1] < referenced.len() {
                referenced[f[1]] = true;
            }
            if f[2] < referenced.len() {
                referenced[f[2]] = true;
            }
        }

        apply_vertex_mask(&mut result, &referenced);
        result.vertices_removed += original_count - result.vertices.len();
    }

    result
}

/// Apply a boolean mask to faces, keeping only faces where mask[i] is true.
fn apply_face_mask(result: &mut CleanupResult, mask: &[bool]) {
    let new_faces: Vec<[usize; 3]> = result
        .faces
        .iter()
        .zip(mask.iter())
        .filter_map(|(f, &keep)| if keep { Some(*f) } else { None })
        .collect();

    // Update face attributes
    let mut new_face_attrs = Attributes::default();
    if let Some(colors) = result.attributes_face.colors.first() {
        new_face_attrs.colors.push(
            colors
                .iter()
                .zip(mask.iter())
                .filter_map(|(c, &keep)| if keep { Some(*c) } else { None })
                .collect(),
        );
    }

    result.faces = new_faces;
    result.attributes_face = new_face_attrs;
}

/// Apply a boolean mask to vertices, keeping only vertices where mask[i] is true.
/// Also updates face indices to point to the new vertex positions.
fn apply_vertex_mask(result: &mut CleanupResult, mask: &[bool]) {
    // Build inverse mapping: old index -> new index
    let mut inverse: Vec<usize> = vec![usize::MAX; mask.len()];
    let mut new_idx = 0;
    for (old_idx, &keep) in mask.iter().enumerate() {
        if keep {
            inverse[old_idx] = new_idx;
            new_idx += 1;
        }
    }

    // Filter vertices
    let new_vertices: Vec<Point3<f64>> = result
        .vertices
        .iter()
        .zip(mask.iter())
        .filter_map(|(v, &keep)| if keep { Some(*v) } else { None })
        .collect();

    // Update face indices
    let new_faces: Vec<[usize; 3]> = result
        .faces
        .iter()
        .filter_map(|f| {
            let i0 = inverse[f[0]];
            let i1 = inverse[f[1]];
            let i2 = inverse[f[2]];
            if i0 != usize::MAX && i1 != usize::MAX && i2 != usize::MAX {
                Some([i0, i1, i2])
            } else {
                None
            }
        })
        .collect();

    // Update vertex attributes
    let mut new_attrs = Attributes::default();
    if let Some(normals) = result.attributes_vertex.normals.first() {
        new_attrs.normals.push(
            normals
                .iter()
                .zip(mask.iter())
                .filter_map(|(n, &keep)| if keep { Some(*n) } else { None })
                .collect(),
        );
    }
    if let Some(uvs) = result.attributes_vertex.uv.first() {
        new_attrs.uv.push(
            uvs.iter()
                .zip(mask.iter())
                .filter_map(|(uv, &keep)| if keep { Some(*uv) } else { None })
                .collect(),
        );
    }
    if let Some(colors) = result.attributes_vertex.colors.first() {
        new_attrs.colors.push(
            colors
                .iter()
                .zip(mask.iter())
                .filter_map(|(c, &keep)| if keep { Some(*c) } else { None })
                .collect(),
        );
    }

    result.vertices = new_vertices;
    result.faces = new_faces;
    result.attributes_vertex = new_attrs;
}

/// Key for vertex comparison in merge operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VertexKey {
    x: i64,
    y: i64,
    z: i64,
    nx: i64,
    ny: i64,
    nz: i64,
    u: i64,
    v: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cleanup_options_default() {
        let opts = CleanupOptions::default();
        assert!(opts.merge_vertices.is_none());
        assert!(opts.remove_degenerate.is_none());
        assert!(!opts.remove_infinite);
    }

    #[test]
    fn test_cleanup_options_struct_init() {
        let opts = CleanupOptions {
            merge_vertices: Some(8),
            remove_degenerate: Some(12),
            remove_infinite: true,
            ..Default::default()
        };

        assert_eq!(opts.merge_vertices, Some(8));
        assert_eq!(opts.remove_degenerate, Some(12));
        assert!(opts.remove_infinite);
    }

    #[test]
    fn test_remove_degenerate_faces() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        // Face 0: valid, Face 1: degenerate (repeated index)
        let faces = vec![[0, 1, 2], [0, 1, 1]];

        let opts = CleanupOptions {
            remove_degenerate: Some(8), // 8 decimal places precision
            ..Default::default()
        };

        let result = cleanup(&vertices, &faces, None, None, &opts);

        assert_eq!(result.faces.len(), 1);
        assert_eq!(result.degenerate_faces_removed, 1);
    }

    #[test]
    fn test_remove_degenerate_faces_close_vertices() {
        // Test that vertices very close together are considered degenerate
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 0.000_000_001), // Very close to vertex 0
        ];
        // Face 0: valid, Face 1: degenerate (vertices 0 and 3 are equal at 8 digits)
        let faces = vec![[0, 1, 2], [0, 1, 3]];

        let opts = CleanupOptions {
            remove_degenerate: Some(8), // 8 decimal places - 0 and 3 will be same
            ..Default::default()
        };

        let result = cleanup(&vertices, &faces, None, None, &opts);

        assert_eq!(result.faces.len(), 1);
        assert_eq!(result.degenerate_faces_removed, 1);

        // But at higher precision they are different
        let opts_precise = CleanupOptions {
            remove_degenerate: Some(12), // 12 decimal places - 0 and 3 will be different
            ..Default::default()
        };

        let result_precise = cleanup(&vertices, &faces, None, None, &opts_precise);

        assert_eq!(result_precise.faces.len(), 2);
        assert_eq!(result_precise.degenerate_faces_removed, 0);
    }

    #[test]
    fn test_merge_vertices() {
        // Cube with duplicated vertices (like from STL)
        let vertices = vec![
            // Triangle 1
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            // Triangle 2 - shares vertices with triangle 1
            Point3::new(0.0, 0.0, 0.0), // duplicate of 0
            Point3::new(1.0, 0.0, 0.0), // duplicate of 1
            Point3::new(0.0, 0.0, 1.0),
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];

        let opts = CleanupOptions {
            merge_vertices: Some(8), // 8 decimal places
            ..Default::default()
        };

        let result = cleanup(&vertices, &faces, None, None, &opts);

        // Should have 4 unique vertices now
        assert_eq!(result.vertices.len(), 4);
        assert_eq!(result.vertices_merged, 2);
        // Faces should be re-indexed
        assert_eq!(result.faces.len(), 2);
    }

    #[test]
    fn test_remove_infinite() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(f64::NAN, 0.0, 0.0), // infinite
        ];
        let faces = vec![[0, 1, 2], [0, 1, 3]];

        let opts = CleanupOptions {
            remove_infinite: true,
            ..Default::default()
        };

        let result = cleanup(&vertices, &faces, None, None, &opts);

        // Face with NaN vertex should be removed
        assert_eq!(result.faces.len(), 1);
    }

    #[test]
    fn test_remove_unreferenced() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(99.0, 99.0, 99.0), // unreferenced
        ];
        let faces = vec![[0, 1, 2]];

        let opts = CleanupOptions {
            remove_unreferenced: true,
            ..Default::default()
        };

        let result = cleanup(&vertices, &faces, None, None, &opts);

        assert_eq!(result.vertices.len(), 3);
        assert_eq!(result.vertices_removed, 1);
    }
}
