//! CAD operations (extrude, revolve, sweep, loft, fillet, chamfer)

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use super::sketch::Sketch;
use crate::path::Path3D;

/// Whether an operation adds or removes material
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Sign {
    /// Add material (boss, pad)
    #[default]
    Add,
    /// Remove material (cut, pocket)
    Remove,
}

/// An extrusion operation - extrude a sketch along a direction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Extrude {
    /// The 2D sketch to extrude
    pub sketch: Sketch,
    /// Extrusion depth (positive = in normal direction)
    pub depth: f64,
    /// Whether to add or remove material
    #[serde(default)]
    pub sign: Sign,
    /// Optional draft angle in radians (0 = no draft)
    #[serde(default, skip_serializing_if = "is_zero")]
    pub draft_angle: f64,
}

fn is_zero(v: &f64) -> bool {
    v.abs() < 1e-10
}

impl Extrude {
    /// Create a new extrusion
    pub fn new(sketch: Sketch, depth: f64, sign: Sign) -> Self {
        Self {
            sketch,
            depth,
            sign,
            draft_angle: 0.0,
        }
    }

    /// Simple extrusion: sketch on XY plane, extrude in +Z, add material
    pub fn simple(sketch: Sketch, depth: f64) -> Self {
        Self {
            sketch,
            depth,
            sign: Sign::Add,
            draft_angle: 0.0,
        }
    }

    /// Set the draft angle
    pub fn with_draft(mut self, angle_radians: f64) -> Self {
        self.draft_angle = angle_radians;
        self
    }

    /// Get the extrusion direction (plane normal)
    pub fn direction(&self) -> Vector3<f64> {
        self.sketch.plane.normal()
    }
}

/// A revolve operation - revolve a sketch around an axis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Revolve {
    /// The 2D sketch to revolve
    pub sketch: Sketch,
    /// Revolution axis direction (in world coordinates)
    pub axis: Vector3<f64>,
    /// Point on the axis (in world coordinates)
    #[serde(default)]
    pub axis_origin: nalgebra::Point3<f64>,
    /// Revolution angle in radians (2*PI for full revolution)
    pub angle: f64,
    /// Whether to add or remove material
    #[serde(default)]
    pub sign: Sign,
}

impl Revolve {
    /// Create a new revolve operation
    pub fn new(
        sketch: Sketch,
        axis: Vector3<f64>,
        axis_origin: nalgebra::Point3<f64>,
        angle: f64,
        sign: Sign,
    ) -> Self {
        Self {
            sketch,
            axis: axis.normalize(),
            axis_origin,
            angle,
            sign,
        }
    }

    /// Full revolution around the Y axis through the origin
    pub fn full(sketch: Sketch) -> Self {
        Self {
            sketch,
            axis: Vector3::y(),
            axis_origin: nalgebra::Point3::origin(),
            angle: std::f64::consts::TAU,
            sign: Sign::Add,
        }
    }

    /// Set a partial revolution angle
    pub fn with_angle(mut self, angle_radians: f64) -> Self {
        self.angle = angle_radians;
        self
    }
}

/// A sweep operation - sweep a sketch along a path
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sweep {
    /// The 2D profile sketch to sweep
    pub profile: Sketch,
    /// The path to sweep along (3D curve)
    pub path: Path3D,
    /// Whether to add or remove material
    #[serde(default)]
    pub sign: Sign,
    /// Whether to keep the profile orientation fixed or follow the path
    #[serde(default)]
    pub fixed_orientation: bool,
}

impl Sweep {
    /// Create a new sweep operation
    pub fn new(profile: Sketch, path: Path3D, sign: Sign) -> Self {
        Self {
            profile,
            path,
            sign,
            fixed_orientation: false,
        }
    }

    /// Set fixed orientation mode
    pub fn with_fixed_orientation(mut self, fixed: bool) -> Self {
        self.fixed_orientation = fixed;
        self
    }
}

/// A loft operation - create a solid between multiple profiles
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Loft {
    /// The profiles to loft between (in order)
    pub profiles: Vec<Sketch>,
    /// Whether to add or remove material
    #[serde(default)]
    pub sign: Sign,
    /// Whether to close the loft (connect last profile to first)
    #[serde(default)]
    pub closed: bool,
}

impl Loft {
    /// Create a new loft operation
    pub fn new(profiles: Vec<Sketch>, sign: Sign) -> Self {
        Self {
            profiles,
            sign,
            closed: false,
        }
    }

    /// Set closed mode
    pub fn with_closed(mut self, closed: bool) -> Self {
        self.closed = closed;
        self
    }
}

/// Edge selection for fillet/chamfer operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeSelection {
    /// Apply to all edges
    All,
    /// Apply to specific edge indices
    Indices(Vec<usize>),
    /// Apply to edges matching a filter (e.g., "concave", "convex")
    Filter(String),
}

impl Default for EdgeSelection {
    fn default() -> Self {
        EdgeSelection::All
    }
}

/// A fillet operation - round edges
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fillet {
    /// Fillet radius
    pub radius: f64,
    /// Edge selection
    #[serde(default)]
    pub edges: EdgeSelection,
}

impl Fillet {
    /// Create a new fillet operation
    pub fn new(radius: f64) -> Self {
        Self {
            radius,
            edges: EdgeSelection::All,
        }
    }

    /// Apply to specific edges
    pub fn with_edges(mut self, edges: EdgeSelection) -> Self {
        self.edges = edges;
        self
    }
}

/// A chamfer operation - bevel edges
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chamfer {
    /// Chamfer distance
    pub distance: f64,
    /// Second distance for asymmetric chamfer (if different from first)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance2: Option<f64>,
    /// Edge selection
    #[serde(default)]
    pub edges: EdgeSelection,
}

impl Chamfer {
    /// Create a new symmetric chamfer operation
    pub fn new(distance: f64) -> Self {
        Self {
            distance,
            distance2: None,
            edges: EdgeSelection::All,
        }
    }

    /// Create an asymmetric chamfer
    pub fn asymmetric(distance1: f64, distance2: f64) -> Self {
        Self {
            distance: distance1,
            distance2: Some(distance2),
            edges: EdgeSelection::All,
        }
    }

    /// Apply to specific edges
    pub fn with_edges(mut self, edges: EdgeSelection) -> Self {
        self.edges = edges;
        self
    }
}

/// All possible CAD operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Operation {
    /// Extrude a sketch
    Extrude(Extrude),
    /// Revolve a sketch around an axis
    Revolve(Revolve),
    /// Sweep a profile along a path
    Sweep(Sweep),
    /// Loft between multiple profiles
    Loft(Loft),
    /// Round edges
    Fillet(Fillet),
    /// Bevel edges
    Chamfer(Chamfer),
}

impl From<Extrude> for Operation {
    fn from(e: Extrude) -> Self {
        Operation::Extrude(e)
    }
}

impl From<Revolve> for Operation {
    fn from(r: Revolve) -> Self {
        Operation::Revolve(r)
    }
}

impl From<Sweep> for Operation {
    fn from(s: Sweep) -> Self {
        Operation::Sweep(s)
    }
}

impl From<Loft> for Operation {
    fn from(l: Loft) -> Self {
        Operation::Loft(l)
    }
}

impl From<Fillet> for Operation {
    fn from(f: Fillet) -> Self {
        Operation::Fillet(f)
    }
}

impl From<Chamfer> for Operation {
    fn from(c: Chamfer) -> Self {
        Operation::Chamfer(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extrude_simple() {
        let sketch = Sketch::rectangle(10.0, 5.0);
        let extrude = Extrude::simple(sketch, 20.0);

        assert_eq!(extrude.depth, 20.0);
        assert_eq!(extrude.sign, Sign::Add);
        assert_eq!(extrude.draft_angle, 0.0);
    }

    #[test]
    fn test_revolve_full() {
        let sketch = Sketch::rectangle(10.0, 5.0);
        let revolve = Revolve::full(sketch);

        assert_eq!(revolve.angle, std::f64::consts::TAU);
        assert_eq!(revolve.axis, Vector3::y());
    }

    #[test]
    fn test_fillet() {
        let fillet = Fillet::new(2.0).with_edges(EdgeSelection::Indices(vec![0, 1, 2]));

        assert_eq!(fillet.radius, 2.0);
        assert!(matches!(fillet.edges, EdgeSelection::Indices(_)));
    }

    #[test]
    fn test_operation_from() {
        let extrude = Extrude::simple(Sketch::rectangle(10.0, 5.0), 5.0);
        let op: Operation = extrude.into();

        assert!(matches!(op, Operation::Extrude(_)));
    }

    #[test]
    fn test_serde_roundtrip() {
        let extrude = Extrude::simple(Sketch::rectangle(10.0, 5.0), 5.0);
        let op: Operation = extrude.into();

        let json = serde_json::to_string(&op).unwrap();
        let parsed: Operation = serde_json::from_str(&json).unwrap();

        if let (Operation::Extrude(orig), Operation::Extrude(loaded)) = (&op, &parsed) {
            assert_eq!(orig.depth, loaded.depth);
            assert_eq!(orig.sign, loaded.sign);
        } else {
            panic!("Expected Extrude operation");
        }
    }
}
