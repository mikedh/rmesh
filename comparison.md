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
| `area` | 14826 | 97.3% | 4.0x | 0.9x / 35.6x |
| `area_faces` | 15849 | 97.4% | 2.2x | 0.0x / 30.3x |
| `bounds` | 44537 | 87.6% | 0.2x | 0.0x / 11.2x |
| `center_mass` | 12667 | 17.1% | 3.1x | 0.8x / 53.5x |
| `centroid` | 945 | 13.4% | 0.1x | 0.0x / 0.4x |
| `edges` | 65263 | 91.4% | 7.3x | 0.8x / 28.8x |
| `edges_sorted` | 58669 | 94.4% | 1.7x | 0.0x / 15.6x |
| `edges_unique_inverse` | 5362 | 0.2% | 2.6x | 0.0x / 9.7x |
| `edges_unique_length` | 5510 | 0.2% | 0.1x | 0.0x / 0.8x |
| `euler_number` | 11789 | 96.1% | 2.0x | 0.8x / 15.0x |
| `extents` | 3702 | 93.8% | 0.1x | 0.0x / 9.4x |
| `face_adjacency` | 11496 | 88.7% | 10.0x | 1.8x / 25.7x |
| `face_adjacency_angles` | 1402 | 1.9% | 11.0x | 5.3x / 29.3x |
| `face_adjacency_convex` | 666 | 97.9% | 0.0x | 0.0x / 0.2x |
| `face_adjacency_projections` | 1612 | 0.1% | 0.0x | 0.0x / 0.3x |
| `face_adjacency_unshared` | 1909 | 0.3% | 0.5x | 0.0x / 15.4x |
| `face_normals` | 25629 | 94.2% | 14.1x | 6.1x / 41.6x |
| `faces` | 379725 | 77.2% | 1.2x | 0.4x / 3.8x |
| `is_convex` | 1676 | 98.6% | 0.0x | 0.0x / 0.6x |
| `is_volume` | 12932 | 95.4% | 3.1x | 0.7x / 42.8x |
| `is_watertight` | 29930 | 95.8% | 7.0x | 0.8x / 49.6x |
| `is_winding_consistent` | 14817 | 99.0% | 4.0x | 0.8x / 23.3x |
| `mass` | 165 | 61.8% | 11.5x | 5.0x / 27.0x |
| `moment_inertia` | 576 | 83.7% | 7.1x | 3.0x / 30.9x |
| `principal_inertia_components` | 1256 | 2.2% | 3.3x | 1.5x / 9.4x |
| `principal_inertia_vectors` | 264 | 1.1% | 363.2x | 186.9x / 641.9x |
| `triangles` | 43004 | 0.0% | 0.0x | 0.0x / 6.7x |
| `triangles_center` | 1896 | 91.9% | 0.1x | 0.0x / 1.4x |
| `vertex_normals` | 844 | 0.0% | 10.7x | 5.5x / 33.2x |
| `vertices` | 303251 | 71.3% | 1.2x | 0.3x / 3.2x |
| `volume` | 29379 | 96.9% | 4.4x | 0.8x / 50.5x |

## Summary

- **Total comparisons:** 1101548
- **Identical results:** 833447 (75.7%)
- **Speedup:** 2.4x avg (0.0x min, 641.9x max)