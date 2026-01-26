# rmesh vs trimesh Comparison

## API Coverage

**34/135 trimesh.Trimesh attributes (25.2%)**

| Attribute | Status |
|-----------|--------|
| `area` | ✓ |
| `area_faces` | ✓ |
| `bounds` | ✓ |
| `center_mass` | ✓ |
| `centroid` | ✓ |
| `edges` | ✓ |
| `edges_sorted` | ✓ |
| `edges_unique` | ✓ |
| `edges_unique_inverse` | ✓ |
| `edges_unique_length` | ✓ |
| `euler_number` | ✓ |
| `extents` | ✓ |
| `face_adjacency` | ✓ |
| `face_adjacency_angles` | ✓ |
| `face_adjacency_convex` | ✓ |
| `face_adjacency_projections` | ✓ |
| `face_adjacency_unshared` | ✓ |
| `face_attributes` | ✓ |
| `face_normals` | ✓ |
| `faces` | ✓ |
| `is_convex` | ✓ |
| `is_volume` | ✓ |
| `is_watertight` | ✓ |
| `is_winding_consistent` | ✓ |
| `mass` | ✓ |
| `moment_inertia` | ✓ |
| `principal_inertia_components` | ✓ |
| `principal_inertia_vectors` | ✓ |
| `triangles` | ✓ |
| `triangles_center` | ✓ |
| `vertex_attributes` | ✓ |
| `vertex_normals` | ✓ |
| `vertices` | ✓ |
| `volume` | ✓ |

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
- `visual`
- `voxelized`

</details>

## Performance Comparison

| Property | Calls | Equal | Speedup (avg) | Speedup (min/max) |
|----------|-------|-------|---------------|-------------------|
| `area` | 42 | 100.0% | 10.9x | 7.7x / 18.0x |
| `area_faces` | 16 | 100.0% | 3.0x | 0.0x / 5.8x |
| `center_mass` | 142 | 100.0% | 10.7x | 6.2x / 20.4x |
| `edges` | 300 | 0.0% | 10.4x | 5.6x / 19.1x |
| `edges_sorted` | 100 | 0.0% | 3.1x | 1.7x / 8.1x |
| `face_normals` | 403 | 74.7% | 14.2x | 7.3x / 26.3x |
| `faces` | 3294 | 87.0% | 1.2x | 0.6x / 2.4x |
| `is_volume` | 100 | 0.0% | 13.2x | 10.0x / 18.5x |
| `is_watertight` | 100 | 0.0% | 9.7x | 6.1x / 16.1x |
| `is_winding_consistent` | 100 | 100.0% | 10.1x | 5.6x / 17.7x |
| `mass` | 40 | 100.0% | 11.2x | 8.9x / 19.8x |
| `moment_inertia` | 318 | 99.4% | 7.1x | 3.5x / 18.6x |
| `triangles` | 1166 | 0.0% | 0.0x | 0.0x / 0.2x |
| `vertices` | 3878 | 76.4% | 1.2x | 0.7x / 2.4x |
| `volume` | 114 | 100.0% | 10.1x | 7.5x / 16.8x |

## Summary

- **Total comparisons:** 10113
- **Identical results:** 6902 (68.2%)
- **Speedup:** 2.7x avg (0.0x min, 26.3x max)