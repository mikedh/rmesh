# rmesh vs trimesh Comparison

## API Coverage

**33/135 trimesh.Trimesh attributes (24.4%)**

| Attribute                      | Status |
|--------------------------------|--------|
| `area`                         | ✓      |
| `area_faces`                   | ✓      |
| `bounds`                       | ✓      |
| `center_mass`                  | ✓      |
| `centroid`                     | ✓      |
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

<details><summary>Not implemented (102 attributes)</summary>

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
- `convex_hull`
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

| Property                       | Calls | Equal  | Speedup (avg) | Speedup (min/max) |
|--------------------------------|-------|--------|---------------|-------------------|
| `area`                         | 1     | 100.0% | 12.3x         | 12.3x / 12.3x     |
| `area_faces`                   | 1     | 100.0% | 9.0x          | 9.0x / 9.0x       |
| `bounds`                       | 1     | 100.0% | 0.1x          | 0.1x / 0.1x       |
| `center_mass`                  | 1     | 100.0% | 7.6x          | 7.6x / 7.6x       |
| `centroid`                     | 1     | 100.0% | 0.1x          | 0.1x / 0.1x       |
| `edges`                        | 1     | 100.0% | 9.3x          | 9.3x / 9.3x       |
| `edges_sorted`                 | 1     | 100.0% | 2.7x          | 2.7x / 2.7x       |
| `edges_unique`                 | 1     | 0.0%   | 3.3x          | 3.3x / 3.3x       |
| `edges_unique_inverse`         | 1     | 0.0%   | 4.4x          | 4.4x / 4.4x       |
| `edges_unique_length`          | 1     | 0.0%   | 0.2x          | 0.2x / 0.2x       |
| `euler_number`                 | 1     | 100.0% | 17.5x         | 17.5x / 17.5x     |
| `extents`                      | 1     | 100.0% | 0.1x          | 0.1x / 0.1x       |
| `face_adjacency`               | 1     | 100.0% | 7.0x          | 7.0x / 7.0x       |
| `face_adjacency_angles`        | 1     | 0.0%   | 11.2x         | 11.2x / 11.2x     |
| `face_adjacency_convex`        | 1     | 0.0%   | 0.0x          | 0.0x / 0.0x       |
| `face_adjacency_projections`   | 1     | 0.0%   | 0.1x          | 0.1x / 0.1x       |
| `face_adjacency_unshared`      | 1     | 0.0%   | 0.2x          | 0.2x / 0.2x       |
| `face_attributes`              | 1     | 0.0%   | 0.7x          | 0.7x / 0.7x       |
| `face_normals`                 | 1     | 100.0% | 15.1x         | 15.1x / 15.1x     |
| `faces`                        | 2     | 100.0% | 1.5x          | 1.5x / 1.6x       |
| `is_convex`                    | 1     | 100.0% | 0.0x          | 0.0x / 0.0x       |
| `is_volume`                    | 2     | 100.0% | 12.4x         | 7.5x / 17.2x      |
| `is_watertight`                | 1     | 100.0% | 13.2x         | 13.2x / 13.2x     |
| `is_winding_consistent`        | 1     | 100.0% | 9.6x          | 9.6x / 9.6x       |
| `mass`                         | 1     | 100.0% | 11.2x         | 11.2x / 11.2x     |
| `moment_inertia`               | 1     | 100.0% | 5.6x          | 5.6x / 5.6x       |
| `principal_inertia_components` | 1     | 0.0%   | 3.4x          | 3.4x / 3.4x       |
| `principal_inertia_vectors`    | 1     | 0.0%   | 7.2x          | 7.2x / 7.2x       |
| `triangles`                    | 1     | 0.0%   | 0.0x          | 0.0x / 0.0x       |
| `triangles_center`             | 1     | 100.0% | 0.1x          | 0.1x / 0.1x       |
| `vertex_attributes`            | 1     | 0.0%   | 1.3x          | 1.3x / 1.3x       |
| `vertices`                     | 7     | 85.7%  | 1.4x          | 1.2x / 1.7x       |
| `volume`                       | 5     | 100.0% | 13.6x         | 12.7x / 15.0x     |

## Summary

- **Total comparisons:** 45
- **Identical results:** 32 (71.1%)
- **Speedup:** 5.7x avg (0.0x min, 17.5x max)