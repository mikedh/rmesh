//! Handler for the `TM_brep_faces` glTF primitive extension.
//!
//! Parses BREP surface definitions (plane, cylinder, cone, sphere, torus)
//! from the extension JSON and maps per-face surface indices via an accessor.

use anyhow::{Context, Result, bail};
use nalgebra::{Point3, Vector3};
use serde_json::Value;

use crate::attributes::{Grouping, GroupingKind, UNSET};
use crate::boundary::Surface;
use crate::boundary::faces::{Cone, Cylinder, Sphere, SurfacePlane, Torus};

use super::PrimitiveResult;

/// Handle the `TM_brep_faces` primitive extension.
///
/// `face_surfaces` in the result is a compact lookup table of surfaces.
/// `surface_grouping.indices` maps each triangle to a surface index, or
/// [`UNSET`] for triangles whose BREP face is null.
pub fn handle_brep_faces(
    data: &Value,
    accessor_reader: &dyn Fn(usize) -> Result<Vec<usize>>,
) -> Result<PrimitiveResult> {
    // Read the faceIndices accessor index
    let face_indices_accessor = data
        .get("faceIndices")
        .and_then(Value::as_u64)
        .context("TM_brep_faces missing faceIndices")? as usize;

    // Read per-face surface indices from the accessor
    let raw_indices = accessor_reader(face_indices_accessor)?;

    // Parse the faces array — null entries become None.
    let faces_array = data
        .get("faces")
        .and_then(Value::as_array)
        .context("TM_brep_faces missing faces array")?;

    let mut parsed: Vec<Option<Surface>> = Vec::with_capacity(faces_array.len());
    for (i, face_json) in faces_array.iter().enumerate() {
        if face_json.is_null() {
            parsed.push(None);
        } else {
            parsed.push(Some(parse_surface(face_json).with_context(|| {
                format!("TM_brep_faces: failed to parse face {i}")
            })?));
        }
    }

    // Build compact surface list, remapping old→new indices.
    let mut surfaces: Vec<Surface> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut old_to_new: Vec<usize> = vec![UNSET; parsed.len()];
    for (old_idx, entry) in parsed.into_iter().enumerate() {
        if let Some(surface) = entry {
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

/// Parse a single surface definition from JSON.
fn parse_surface(json: &Value) -> Result<Surface> {
    let kind = json
        .get("type")
        .and_then(Value::as_str)
        .context("surface missing type")?;

    match kind {
        "plane" => Ok(Surface::Plane(SurfacePlane {
            origin: read_point3(json, "origin")?,
            normal: read_vector3(json, "normal")?,
        })),
        "cylinder" => Ok(Surface::Cylinder(Cylinder {
            origin: read_point3(json, "origin")?,
            axis: read_vector3(json, "axis")?,
            radius: read_f64(json, "radius")?,
        })),
        "cone" => Ok(Surface::Cone(Cone {
            apex: read_point3(json, "apex")?,
            axis: read_vector3(json, "axis")?,
            half_angle: read_f64(json, "semi_angle")?,
        })),
        "sphere" => Ok(Surface::Sphere(Sphere {
            center: read_point3(json, "center")?,
            radius: read_f64(json, "radius")?,
        })),
        "torus" => Ok(Surface::Torus(Torus {
            center: read_point3(json, "center")?,
            axis: read_vector3(json, "axis")?,
            major_radius: read_f64(json, "major_radius")?,
            minor_radius: read_f64(json, "minor_radius")?,
        })),
        _ => bail!("unknown surface type: {kind}"),
    }
}

/// Read a `Point3<f64>` from a JSON array field.
fn read_point3(json: &Value, field: &str) -> Result<Point3<f64>> {
    let arr = json
        .get(field)
        .and_then(Value::as_array)
        .with_context(|| format!("missing {field}"))?;
    if arr.len() < 3 {
        bail!("{field} must have at least 3 elements");
    }
    Ok(Point3::new(
        arr[0].as_f64().unwrap_or(0.0),
        arr[1].as_f64().unwrap_or(0.0),
        arr[2].as_f64().unwrap_or(0.0),
    ))
}

/// Read a `Vector3<f64>` from a JSON array field.
fn read_vector3(json: &Value, field: &str) -> Result<Vector3<f64>> {
    let arr = json
        .get(field)
        .and_then(Value::as_array)
        .with_context(|| format!("missing {field}"))?;
    if arr.len() < 3 {
        bail!("{field} must have at least 3 elements");
    }
    Ok(Vector3::new(
        arr[0].as_f64().unwrap_or(0.0),
        arr[1].as_f64().unwrap_or(0.0),
        arr[2].as_f64().unwrap_or(0.0),
    ))
}

/// Read a single `f64` from a JSON field.
fn read_f64(json: &Value, field: &str) -> Result<f64> {
    json.get(field)
        .and_then(Value::as_f64)
        .with_context(|| format!("missing {field}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parse_plane() {
        let json = serde_json::json!({
            "type": "plane",
            "origin": [1.0, 2.0, 3.0],
            "normal": [0.0, 0.0, 1.0]
        });
        let surface = parse_surface(&json).unwrap();
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
        let json = serde_json::json!({
            "type": "cylinder",
            "origin": [0.0, 0.0, 0.0],
            "axis": [0.0, 1.0, 0.0],
            "radius": 0.005
        });
        let surface = parse_surface(&json).unwrap();
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
        let json = serde_json::json!({
            "type": "cone",
            "apex": [0.0, 0.0, 5.0],
            "axis": [0.0, 0.0, 1.0],
            "semi_angle": 0.3
        });
        let surface = parse_surface(&json).unwrap();
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
        let json = serde_json::json!({
            "type": "sphere",
            "center": [1.0, 2.0, 3.0],
            "radius": 1.0
        });
        let surface = parse_surface(&json).unwrap();
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
        let json = serde_json::json!({
            "type": "torus",
            "center": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "major_radius": 2.0,
            "minor_radius": 0.5
        });
        let surface = parse_surface(&json).unwrap();
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
        let json = serde_json::json!({"type": "nurbs"});
        assert!(parse_surface(&json).is_err());
    }

    #[test]
    fn test_handle_brep_faces() {
        let data = serde_json::json!({
            "faceIndices": 0,
            "faces": [
                {"type": "plane", "origin": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
                {"type": "cylinder", "origin": [0.0, 0.0, 0.0], "axis": [0.0, 1.0, 0.0], "radius": 0.005}
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
                {"type": "plane", "origin": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
                null,
                {"type": "cylinder", "origin": [0.0, 0.0, 0.0], "axis": [0.0, 1.0, 0.0], "radius": 0.005}
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
