//! Handler for the `TM_brep_faces` glTF primitive extension.
//!
//! Parses BREP surface definitions (plane, cylinder, cone, sphere, torus)
//! from the extension JSON and maps per-face surface indices via an accessor.

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::attributes::{Grouping, GroupingKind, UNSET};
use crate::boundary::Surface;
use crate::boundary::faces::SurfaceDict;

use super::PrimitiveResult;

/// Top-level `TM_brep_faces` extension data.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrepFacesExt {
    face_indices: usize,
    #[serde(default)]
    faces: Vec<Option<SurfaceDict>>,
}

/// Handle the `TM_brep_faces` primitive extension.
///
/// `face_surfaces` in the result is a compact lookup table of surfaces.
/// `surface_grouping.indices` maps each triangle to a surface index, or
/// [`UNSET`] for triangles whose BREP face is null.
///
/// Backward compatibility transforms applied before parsing:
/// - `"type"` → `"kind"` key rename (old tag key)
/// - Lowercase variant aliases (e.g. `"plane"`) via `SurfaceDict` serde
/// - `"semi_angle"` → `"half_angle"` alias on `Cone` via `SurfaceDict` serde
pub fn handle_brep_faces(
    data: &Value,
    accessor_reader: &dyn Fn(usize) -> Result<Vec<usize>>,
) -> Result<PrimitiveResult> {
    // Backward compat: old files used "type" as tag key instead of "kind"
    let mut data = data.clone();
    if let Some(faces) = data.get_mut("faces").and_then(|f| f.as_array_mut()) {
        for face in faces {
            if let Some(obj) = face.as_object_mut()
                && let Some(v) = obj.remove("type")
            {
                obj.entry("kind").or_insert(v);
            }
        }
    }

    let ext: BrepFacesExt =
        serde_json::from_value(data).context("failed to parse TM_brep_faces")?;

    // Read per-face surface indices from the accessor
    let raw_indices = accessor_reader(ext.face_indices)?;

    // Build compact surface list, remapping old→new indices.
    let mut surfaces = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut old_to_new: Vec<usize> = vec![UNSET; ext.faces.len()];
    for (old_idx, entry) in ext.faces.into_iter().enumerate() {
        if let Some(dict) = entry
            && let Ok(surface) = Surface::try_from(dict)
        {
            let new_idx = surfaces.len();
            names.push(surface.kind_name().to_string());
            surfaces.push(surface);
            old_to_new[old_idx] = new_idx;
        }
    }

    // Remap per-face indices, preserving UNSET for null surfaces.
    let indices: Vec<usize> = raw_indices
        .iter()
        .map(|&old| {
            if old < old_to_new.len() {
                old_to_new[old]
            } else {
                UNSET
            }
        })
        .collect();

    let grouping = Grouping {
        kind: GroupingKind::Surface,
        names,
        indices,
    };

    Ok(PrimitiveResult {
        face_surfaces: surfaces,
        surface_grouping: Some(grouping),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn parse_face(json: serde_json::Value) -> Surface {
        Surface::from_dict(json).unwrap()
    }

    #[test]
    fn test_parse_plane() {
        let surface = parse_face(serde_json::json!({
            "kind": "Plane",
            "origin": [1.0, 2.0, 3.0],
            "normal": [0.0, 0.0, 1.0]
        }));
        match surface {
            Surface::Plane(p) => {
                assert_relative_eq!(p.origin.x, 1.0);
                assert_relative_eq!(p.origin.y, 2.0);
                assert_relative_eq!(p.origin.z, 3.0);
                assert_relative_eq!(p.normal.z, 1.0);
            }
            _ => panic!("expected Plane"),
        }
    }

    #[test]
    fn test_parse_cylinder() {
        let surface = parse_face(serde_json::json!({
            "kind": "Cylinder",
            "origin": [0.0, 0.0, 0.0],
            "axis": [0.0, 1.0, 0.0],
            "radius": 0.005
        }));
        match surface {
            Surface::Cylinder(c) => {
                assert_relative_eq!(c.axis.y, 1.0);
                assert_relative_eq!(c.radius, 0.005);
            }
            _ => panic!("expected Cylinder"),
        }
    }

    #[test]
    fn test_parse_cone() {
        let surface = parse_face(serde_json::json!({
            "kind": "Cone",
            "apex": [0.0, 0.0, 5.0],
            "axis": [0.0, 0.0, 1.0],
            "half_angle": 0.3
        }));
        match surface {
            Surface::Cone(c) => {
                assert_relative_eq!(c.apex.z, 5.0);
                assert_relative_eq!(c.half_angle, 0.3);
            }
            _ => panic!("expected Cone"),
        }
    }

    #[test]
    fn test_parse_sphere() {
        let surface = parse_face(serde_json::json!({
            "kind": "Sphere",
            "center": [1.0, 2.0, 3.0],
            "radius": 1.0
        }));
        match surface {
            Surface::Sphere(s) => {
                assert_relative_eq!(s.center.x, 1.0);
                assert_relative_eq!(s.radius, 1.0);
            }
            _ => panic!("expected Sphere"),
        }
    }

    #[test]
    fn test_parse_torus() {
        let surface = parse_face(serde_json::json!({
            "kind": "Torus",
            "center": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "major_radius": 2.0,
            "minor_radius": 0.5
        }));
        match surface {
            Surface::Torus(t) => {
                assert_relative_eq!(t.major_radius, 2.0);
                assert_relative_eq!(t.minor_radius, 0.5);
            }
            _ => panic!("expected Torus"),
        }
    }

    #[test]
    fn test_parse_unknown_type() {
        let json = serde_json::json!({"kind": "nurbs"});
        assert!(Surface::from_dict(json).is_err());
    }

    /// Backward compat: old GLBs use "type" + lowercase + "semi_angle"
    #[test]
    fn test_backward_compat_old_glb_format() {
        let data = serde_json::json!({
            "faceIndices": 0,
            "faces": [
                {"type": "plane", "origin": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
                {"type": "cone", "apex": [0.0, 0.0, 5.0], "axis": [0.0, 0.0, 1.0], "semi_angle": 0.3}
            ]
        });

        let mock_reader = |_idx: usize| -> Result<Vec<usize>> { Ok(vec![0, 1]) };
        let result = handle_brep_faces(&data, &mock_reader).unwrap();
        assert_eq!(result.face_surfaces.len(), 2);
        assert!(matches!(result.face_surfaces[0], Surface::Plane(_)));
        match &result.face_surfaces[1] {
            Surface::Cone(c) => assert_relative_eq!(c.half_angle, 0.3),
            _ => panic!("expected Cone"),
        }
    }

    #[test]
    fn test_handle_brep_faces() {
        let data = serde_json::json!({
            "faceIndices": 0,
            "faces": [
                {"kind": "Plane", "origin": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
                {"kind": "Cylinder", "origin": [0.0, 0.0, 0.0], "axis": [0.0, 1.0, 0.0], "radius": 0.005}
            ]
        });

        // Mock accessor reader: accessor 0 returns [0, 0, 1, 1, 0]
        let mock_reader = |idx: usize| -> Result<Vec<usize>> {
            assert_eq!(idx, 0);
            Ok(vec![0, 0, 1, 1, 0])
        };

        let result = handle_brep_faces(&data, &mock_reader).unwrap();
        // face_surfaces is the unique lookup table (2 entries)
        assert_eq!(result.face_surfaces.len(), 2);
        assert!(matches!(result.face_surfaces[0], Surface::Plane(_)));
        assert!(matches!(result.face_surfaces[1], Surface::Cylinder(_)));

        let grouping = result.surface_grouping.unwrap();
        assert_eq!(grouping.kind, GroupingKind::Surface);
        assert_eq!(grouping.names, vec!["Plane", "Cylinder"]);
        assert_eq!(grouping.indices, vec![0, 0, 1, 1, 0]);
    }

    #[test]
    fn test_handle_brep_faces_with_null() {
        let data = serde_json::json!({
            "faceIndices": 0,
            "faces": [
                {"kind": "Plane", "origin": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
                null,
                {"kind": "Cylinder", "origin": [0.0, 0.0, 0.0], "axis": [0.0, 1.0, 0.0], "radius": 0.005}
            ]
        });

        // face indices: 0=plane, 1=null, 2=cylinder
        let mock_reader = |_idx: usize| -> Result<Vec<usize>> { Ok(vec![0, 1, 2, 0]) };

        let result = handle_brep_faces(&data, &mock_reader).unwrap();
        // Only 2 real surfaces (null is excluded)
        assert_eq!(result.face_surfaces.len(), 2);
        assert!(matches!(result.face_surfaces[0], Surface::Plane(_)));
        assert!(matches!(result.face_surfaces[1], Surface::Cylinder(_)));

        let grouping = result.surface_grouping.unwrap();
        // Null face maps to UNSET
        assert_eq!(grouping.indices, vec![0, UNSET, 1, 0]);
    }
}
