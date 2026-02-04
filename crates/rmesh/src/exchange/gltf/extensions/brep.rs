//! Handler for the `TM_brep_faces` glTF primitive extension.
//!
//! Parses BREP surface definitions (plane, cylinder, cone, sphere, torus)
//! from the extension JSON and maps per-face surface indices via an accessor.

use anyhow::{Context, Result};
use nalgebra::{Point3, Vector3};
use serde::Deserialize;
use serde_json::Value;

use crate::attributes::{Grouping, GroupingKind, UNSET};
use crate::boundary::Surface;
use crate::boundary::faces::{Cone, Cylinder, Sphere, SurfacePlane, Torus};

use super::PrimitiveResult;

/// Top-level `TM_brep_faces` extension data.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrepFacesExt {
    face_indices: usize,
    #[serde(default)]
    faces: Vec<Option<BrepFace>>,
}

/// A single BREP face surface definition, tagged by `"type"`.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[allow(dead_code)]
enum BrepFace {
    Plane {
        origin: Point3<f64>,
        normal: Vector3<f64>,
        #[serde(default)]
        x_dir: Option<Vector3<f64>>,
        #[serde(default)]
        extent_x: Option<[f64; 2]>,
        #[serde(default)]
        extent_y: Option<[f64; 2]>,
    },
    Cylinder {
        origin: Point3<f64>,
        axis: Vector3<f64>,
        radius: f64,
        #[serde(default)]
        extent_angle: Option<[f64; 2]>,
        #[serde(default)]
        extent_height: Option<[f64; 2]>,
    },
    Cone {
        apex: Point3<f64>,
        axis: Vector3<f64>,
        semi_angle: f64,
        #[serde(default)]
        ref_radius: Option<f64>,
        #[serde(default)]
        extent_angle: Option<[f64; 2]>,
        #[serde(default)]
        extent_distance: Option<[f64; 2]>,
    },
    Sphere {
        center: Point3<f64>,
        radius: f64,
        #[serde(default)]
        extent_longitude: Option<[f64; 2]>,
        #[serde(default)]
        extent_latitude: Option<[f64; 2]>,
    },
    Torus {
        center: Point3<f64>,
        axis: Vector3<f64>,
        major_radius: f64,
        minor_radius: f64,
        #[serde(default)]
        extent_major_angle: Option<[f64; 2]>,
        #[serde(default)]
        extent_minor_angle: Option<[f64; 2]>,
    },
}

impl From<BrepFace> for Surface {
    fn from(face: BrepFace) -> Self {
        match face {
            BrepFace::Plane { origin, normal, .. } => {
                Surface::Plane(SurfacePlane { origin, normal })
            }
            BrepFace::Cylinder {
                origin,
                axis,
                radius,
                ..
            } => Surface::Cylinder(Cylinder {
                origin,
                axis,
                radius,
            }),
            BrepFace::Cone {
                apex,
                axis,
                semi_angle,
                ..
            } => Surface::Cone(Cone {
                apex,
                axis,
                half_angle: semi_angle,
            }),
            BrepFace::Sphere { center, radius, .. } => Surface::Sphere(Sphere { center, radius }),
            BrepFace::Torus {
                center,
                axis,
                major_radius,
                minor_radius,
                ..
            } => Surface::Torus(Torus {
                center,
                axis,
                major_radius,
                minor_radius,
            }),
        }
    }
}

/// Handle the `TM_brep_faces` primitive extension.
///
/// `face_surfaces` in the result is a compact lookup table of surfaces.
/// `surface_grouping.indices` maps each triangle to a surface index, or
/// [`UNSET`] for triangles whose BREP face is null.
pub fn handle_brep_faces(
    data: &Value,
    accessor_reader: &dyn Fn(usize) -> Result<Vec<usize>>,
) -> Result<PrimitiveResult> {
    let ext: BrepFacesExt =
        serde_json::from_value(data.clone()).context("failed to parse TM_brep_faces")?;

    // Read per-face surface indices from the accessor
    let raw_indices = accessor_reader(ext.face_indices)?;

    // Build compact surface list, remapping old→new indices.
    let mut surfaces: Vec<Surface> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut old_to_new: Vec<usize> = vec![UNSET; ext.faces.len()];
    for (old_idx, entry) in ext.faces.into_iter().enumerate() {
        if let Some(face) = entry {
            let surface = Surface::from(face);
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
        Surface::from(serde_json::from_value::<BrepFace>(json).unwrap())
    }

    #[test]
    fn test_parse_plane() {
        let surface = parse_face(serde_json::json!({
            "type": "plane",
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
            "type": "cylinder",
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
            "type": "cone",
            "apex": [0.0, 0.0, 5.0],
            "axis": [0.0, 0.0, 1.0],
            "semi_angle": 0.3
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
            "type": "sphere",
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
            "type": "torus",
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
        let json = serde_json::json!({"type": "nurbs"});
        assert!(serde_json::from_value::<BrepFace>(json).is_err());
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
