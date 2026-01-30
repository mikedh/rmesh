# rmesh vs trimesh Comparison

## API Coverage

**34/135 trimesh.Trimesh attributes (25.2%)**

| Attribute                      | Status |
|--------------------------------|--------|
| `area`                         | ✓      |
| `area_faces`                   | ✓      |
| `bounds`                       | ✓      |
| `center_mass`                  | ✓      |
| `centroid`                     | ✓      |
| `convex_hull`                  | ✓      |
| `edges`                        | ✓      |
| `edges_sorted`                 | ✓      |
| `edges_unique`                 | ✓      |
| `edges_unique_inverse`         | ✓      |
| `edges_unique_length`          | ✓      |
| `euler_number`                 | ✓      |
| `extents`                      | ✓      |
| `face_adjacency`               | ✓      |
| `face_adjacency_angles`        | ✓      |
| `face_adjacency_convex`        | ✓      |
| `face_adjacency_projections`   | ✓      |
| `face_adjacency_unshared`      | ✓      |
| `face_attributes`              | ✓      |
| `face_normals`                 | ✓      |
| `faces`                        | ✓      |
| `is_convex`                    | ✓      |
| `is_volume`                    | ✓      |
| `is_watertight`                | ✓      |
| `is_winding_consistent`        | ✓      |
| `mass`                         | ✓      |
| `moment_inertia`               | ✓      |
| `principal_inertia_components` | ✓      |
| `principal_inertia_vectors`    | ✓      |
| `triangles`                    | ✓      |
| `triangles_center`             | ✓      |
| `vertex_attributes`            | ✓      |
| `vertices`                     | ✓      |
| `volume`                       | ✓      |

<details><summary>Not implemented (101 attributes)</summary>

- `apply_obb`
- `apply_scale`
- `apply_transform`
- `apply_translation`
- `body_count`
- `bounding_box`
- `bounding_box_oriented`
- `bounding_cylinder`
- `bounding_primitive`
- `bounding_sphere`
- `compute_stable_poses`
- `contains`
- `convert_units`
- `convex_decomposition`
- `copy`
- `density`
- `difference`
- `edges_face`
- `edges_sorted_tree`
- `edges_sparse`
- `eval_cached`
- `export`
- `face_adjacency_edges`
- `face_adjacency_edges_tree`
- `face_adjacency_radius`
- `face_adjacency_span`
- `face_adjacency_tree`
- `face_angles`
- `face_angles_sparse`
- `face_neighborhood`
- `faces_sparse`
- `faces_unique_edges`
- `facets`
- `facets_area`
- `facets_boundary`
- `facets_normal`
- `facets_on_hull`
- `facets_origin`
- `fill_holes`
- `fix_normals`
- `identifier`
- `identifier_hash`
- `integral_mean_curvature`
- `intersection`
- `invert`
- `is_empty`
- `kdtree`
- `mass_properties`
- `merge_vertices`
- `metadata`
- `moment_inertia_frame`
- `mutable`
- `nearest`
- `nondegenerate_faces`
- `outline`
- `permutate`
- `principal_inertia_transform`
- `process`
- `projected`
- `ray`
- `referenced_vertices`
- `register`
- `remove_infinite_values`
- `remove_unreferenced_vertices`
- `rezero`
- `sample`
- `scale`
- `scene`
- `section`
- `section_multiplane`
- `show`
- `simplify_quadric_decimation`
- `slice_plane`
- `smooth_shaded`
- `source`
- `split`
- `subdivide`
- `subdivide_loop`
- `subdivide_to_size`
- `submesh`
- `symmetry`
- `symmetry_axis`
- `symmetry_section`
- `to_dict`
- `triangles_cross`
- `triangles_tree`
- `union`
- `unique_faces`
- `units`
- `unmerge_vertices`
- `unwrap`
- `update_faces`
- `update_vertices`
- `vertex_adjacency_graph`
- `vertex_defects`
- `vertex_degree`
- `vertex_faces`
- `vertex_neighbors`
- `vertex_normals`
- `visual`
- `voxelized`

</details>

## Performance Comparison

| Property                       | Calls   | Equal  | Speedup (avg) | Speedup (min/max) |
|--------------------------------|---------|--------|---------------|-------------------|
| `area`                         | 117401  | 97.1%  | 0.6x          | 0.0x / 40.4x      |
| `area_faces`                   | 6199    | 89.3%  | 2.1x          | 0.0x / 16.7x      |
| `bounds`                       | 251390  | 90.8%  | 0.1x          | 0.0x / 3.3x       |
| `center_mass`                  | 5915    | 95.2%  | 1.6x          | 0.0x / 25.3x      |
| `centroid`                     | 11207   | 16.5%  | 0.0x          | 0.0x / 0.5x       |
| `edges`                        | 77927   | 92.6%  | 3.3x          | 0.0x / 103.6x     |
| `edges_sorted`                 | 42984   | 91.4%  | 1.7x          | 0.0x / 29.0x      |
| `edges_unique`                 | 3929    | 0.0%   | 1.5x          | 0.0x / 12.7x      |
| `edges_unique_inverse`         | 40782   | 0.3%   | 0.5x          | 0.0x / 9.0x       |
| `edges_unique_length`          | 40724   | 0.4%   | 0.2x          | 0.0x / 2.4x       |
| `euler_number`                 | 87297   | 98.7%  | 0.1x          | 0.0x / 20.7x      |
| `extents`                      | 9586    | 74.3%  | 0.1x          | 0.0x / 3.8x       |
| `face_adjacency`               | 35684   | 76.8%  | 2.3x          | 0.0x / 24.6x      |
| `face_adjacency_angles`        | 12336   | 2.3%   | 4.3x          | 0.0x / 28.3x      |
| `face_adjacency_convex`        | 4036    | 97.2%  | 0.0x          | 0.0x / 0.2x       |
| `face_adjacency_projections`   | 85      | 0.0%   | 0.1x          | 0.0x / 0.3x       |
| `face_adjacency_unshared`      | 3565    | 1.0%   | 1.8x          | 0.0x / 71.2x      |
| `face_attributes`              | 1       | 0.0%   | 0.7x          | 0.7x / 0.7x       |
| `face_normals`                 | 256074  | 80.6%  | 7.1x          | 0.0x / 79.9x      |
| `faces`                        | 1787165 | 70.0%  | 0.8x          | 0.0x / 98.5x      |
| `is_convex`                    | 9923    | 94.9%  | 0.0x          | 0.0x / 7.4x       |
| `is_volume`                    | 92703   | 97.3%  | 0.3x          | 0.0x / 18.4x      |
| `is_watertight`                | 98117   | 93.4%  | 1.2x          | 0.0x / 28.2x      |
| `is_winding_consistent`        | 8924    | 99.0%  | 2.9x          | 0.0x / 38.2x      |
| `mass`                         | 1179    | 60.7%  | 5.6x          | 0.0x / 29.9x      |
| `moment_inertia`               | 6444    | 54.1%  | 2.4x          | 0.0x / 14.0x      |
| `principal_inertia_components` | 4103    | 11.2%  | 1.0x          | 0.0x / 7.2x       |
| `principal_inertia_vectors`    | 4879    | 6.0%   | 3.2x          | 0.4x / 16.5x      |
| `triangles`                    | 18532   | 0.0%   | 0.0x          | 0.0x / 3.0x       |
| `triangles_center`             | 5897    | 100.0% | 0.1x          | 0.0x / 0.3x       |
| `vertex_attributes`            | 1       | 0.0%   | 1.3x          | 1.3x / 1.3x       |
| `vertices`                     | 1459560 | 79.2%  | 0.7x          | 0.0x / 157.7x     |
| `volume`                       | 126407  | 91.8%  | 2.0x          | 0.0x / 36.0x      |

## Summary

- **Total comparisons:** 4630956
- **Identical results:** 3532159 (76.3%)
- **Speedup:** 1.1x avg (0.0x min, 157.7x max)