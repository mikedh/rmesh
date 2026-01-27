//! Constraint types and dimension expressions
//!
//! This module defines the core constraint types used in parametric sketches.
//! Dimension values can be literals or Starlark expressions.

use serde::{Deserialize, Serialize};

use crate::creation::feature::environment::Environment;
use crate::creation::feature::error::Result;
use crate::creation::feature::sketch::EntityId;

/// Reference to a point in the sketch
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PointRef {
    /// Direct vertex index
    Vertex { index: usize },
    /// Start point of an entity
    Start { entity: EntityId },
    /// End point of an entity
    End { entity: EntityId },
    /// Center of a circle or arc
    Center { entity: EntityId },
}

impl PointRef {
    /// Create a vertex reference
    pub fn vertex(index: usize) -> Self {
        Self::Vertex { index }
    }

    /// Create a start point reference
    pub fn start(entity: EntityId) -> Self {
        Self::Start { entity }
    }

    /// Create an end point reference
    pub fn end(entity: EntityId) -> Self {
        Self::End { entity }
    }

    /// Create a center point reference
    pub fn center(entity: EntityId) -> Self {
        Self::Center { entity }
    }
}

/// A dimension value - either a literal number or a Starlark expression
///
/// Expressions are evaluated against the environment's variables.
/// Examples: `100.0`, `"d0 * 2"`, `"width / 2"`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dim {
    /// Literal numeric value
    Value(f64),
    /// Starlark expression string (evaluated via Environment)
    Expr(String),
}

impl Dim {
    /// Create a literal dimension
    pub fn value(v: f64) -> Self {
        Self::Value(v)
    }

    /// Create an expression dimension
    pub fn expr(s: impl Into<String>) -> Self {
        Self::Expr(s.into())
    }

    /// Resolve the dimension to a concrete value
    ///
    /// Literals return directly; expressions are evaluated via the environment.
    pub fn resolve(&self, env: &Environment) -> Result<f64> {
        match self {
            Dim::Value(v) => Ok(*v),
            Dim::Expr(s) => env.evaluate(s),
        }
    }
}

impl From<f64> for Dim {
    fn from(v: f64) -> Self {
        Self::Value(v)
    }
}

impl From<&str> for Dim {
    fn from(s: &str) -> Self {
        Self::Expr(s.to_string())
    }
}

impl From<String> for Dim {
    fn from(s: String) -> Self {
        Self::Expr(s)
    }
}

/// Sketch constraints defining geometric relationships
///
/// Constraints are solved to determine final vertex positions.
/// Dimension values can be Starlark expressions referencing environment variables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Constraint {
    /// Fix a point to specific coordinates
    Fixed { point: PointRef, x: Dim, y: Dim },

    /// Two points share the same location
    Coincident { a: PointRef, b: PointRef },

    /// Two circles or arcs share the same center
    Concentric { a: EntityId, b: EntityId },

    /// A line segment is horizontal (parallel to X axis)
    Horizontal { entity: EntityId },

    /// A line segment is vertical (parallel to Y axis)
    Vertical { entity: EntityId },

    /// Two line segments are parallel
    Parallel { a: EntityId, b: EntityId },

    /// Two line segments are perpendicular
    Perpendicular { a: EntityId, b: EntityId },

    /// Distance between two points
    Distance {
        a: PointRef,
        b: PointRef,
        value: Dim,
    },

    /// Radius of a circle or arc
    Radius { entity: EntityId, value: Dim },

    /// Angle of a line segment (from horizontal, in radians)
    Angle { entity: EntityId, value: Dim },

    /// Two curves meet tangentially
    Tangent { a: EntityId, b: EntityId },

    /// A point lies on a curve
    PointOnCurve { point: PointRef, curve: EntityId },

    /// Two entities have equal length or radius
    Equal { a: EntityId, b: EntityId },

    /// An entity is symmetric about a construction line
    Symmetric { entity: EntityId, axis: EntityId },

    /// A line passes through the origin
    ThroughOrigin { entity: EntityId },

    /// Horizontal distance between two points
    HorizontalDistance {
        a: PointRef,
        b: PointRef,
        value: Dim,
    },

    /// Vertical distance between two points
    VerticalDistance {
        a: PointRef,
        b: PointRef,
        value: Dim,
    },
}

impl Constraint {
    /// Create a Fixed constraint
    pub fn fixed(point: PointRef, x: impl Into<Dim>, y: impl Into<Dim>) -> Self {
        Self::Fixed {
            point,
            x: x.into(),
            y: y.into(),
        }
    }

    /// Create a Coincident constraint
    pub fn coincident(a: PointRef, b: PointRef) -> Self {
        Self::Coincident { a, b }
    }

    /// Create a Concentric constraint
    pub fn concentric(a: EntityId, b: EntityId) -> Self {
        Self::Concentric { a, b }
    }

    /// Create a Horizontal constraint
    pub fn horizontal(entity: EntityId) -> Self {
        Self::Horizontal { entity }
    }

    /// Create a Vertical constraint
    pub fn vertical(entity: EntityId) -> Self {
        Self::Vertical { entity }
    }

    /// Create a Parallel constraint
    pub fn parallel(a: EntityId, b: EntityId) -> Self {
        Self::Parallel { a, b }
    }

    /// Create a Perpendicular constraint
    pub fn perpendicular(a: EntityId, b: EntityId) -> Self {
        Self::Perpendicular { a, b }
    }

    /// Create a Distance constraint
    pub fn distance(a: PointRef, b: PointRef, value: impl Into<Dim>) -> Self {
        Self::Distance {
            a,
            b,
            value: value.into(),
        }
    }

    /// Create a Radius constraint
    pub fn radius(entity: EntityId, value: impl Into<Dim>) -> Self {
        Self::Radius {
            entity,
            value: value.into(),
        }
    }

    /// Create an Angle constraint
    pub fn angle(entity: EntityId, value: impl Into<Dim>) -> Self {
        Self::Angle {
            entity,
            value: value.into(),
        }
    }

    /// Create a Tangent constraint
    pub fn tangent(a: EntityId, b: EntityId) -> Self {
        Self::Tangent { a, b }
    }

    /// Create a PointOnCurve constraint
    pub fn point_on_curve(point: PointRef, curve: EntityId) -> Self {
        Self::PointOnCurve { point, curve }
    }

    /// Create an Equal constraint
    pub fn equal(a: EntityId, b: EntityId) -> Self {
        Self::Equal { a, b }
    }

    /// Create a Symmetric constraint
    pub fn symmetric(entity: EntityId, axis: EntityId) -> Self {
        Self::Symmetric { entity, axis }
    }

    /// Create a HorizontalDistance constraint
    pub fn horizontal_distance(a: PointRef, b: PointRef, value: impl Into<Dim>) -> Self {
        Self::HorizontalDistance {
            a,
            b,
            value: value.into(),
        }
    }

    /// Create a VerticalDistance constraint
    pub fn vertical_distance(a: PointRef, b: PointRef, value: impl Into<Dim>) -> Self {
        Self::VerticalDistance {
            a,
            b,
            value: value.into(),
        }
    }
}
