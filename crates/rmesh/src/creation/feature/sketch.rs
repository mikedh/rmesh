//! Sketch module - collection of 2D entities on a plane with unique IDs
//!
//! Sketches are the foundation for feature operations like extrude and revolve.
//! Each entity has a stable ID that survives editing and serialization.
//!
//! Sketches use indexed vertices - all points are stored in a shared `vertices`
//! array and entities reference them by index.

use nalgebra::Point2;
use serde::{Deserialize, Serialize};

use crate::path::{Arc2, Circle2, Line, Segment2D, Winding};

use super::constraint::Constraint;
use super::environment::Environment;
use super::error::{FeatureError, Result};
use super::plane::SketchPlane;

/// Default tessellation tolerance in sketch units
pub const DEFAULT_TOLERANCE: f64 = 0.01;

/// Unique identifier for sketch entities
///
/// Stable across edits - assigned once on creation, never reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub u64);

impl EntityId {
    /// Create a new entity ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "e{}", self.0)
    }
}

/// A sketch entity with identity, geometry, and metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SketchEntity {
    /// Unique identifier (stable across edits)
    pub id: EntityId,

    /// Optional user-provided name (e.g., "base_edge")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// The geometry of this entity (references vertex indices)
    pub segment: Segment2D,

    /// Construction geometry doesn't form part of the profile
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub construction: bool,
}

impl SketchEntity {
    /// Create a new entity with the given ID and geometry
    pub fn new(id: EntityId, segment: Segment2D) -> Self {
        Self {
            id,
            name: None,
            segment,
            construction: false,
        }
    }

    /// Set the entity name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Mark as construction geometry
    pub fn as_construction(mut self) -> Self {
        self.construction = true;
        self
    }

    /// Get the start point of this entity using vertices array
    pub fn start(&self, vertices: &[Point2<f64>]) -> Option<Point2<f64>> {
        self.segment.start(vertices)
    }

    /// Get the end point of this entity using vertices array
    pub fn finish(&self, vertices: &[Point2<f64>]) -> Option<Point2<f64>> {
        self.segment.finish(vertices)
    }
}

/// A 2D sketch containing entities on a plane
///
/// Sketches are the foundation for operations like extrude, revolve, etc.
/// Each entity has a stable ID that survives editing and serialization.
///
/// Vertices are stored in a shared array and entities reference them by index.
/// Constraints define geometric relationships between entities and are solved
/// to determine final vertex positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sketch {
    /// The plane this sketch lies on
    #[serde(default)]
    pub plane: SketchPlane,

    /// Counter for generating unique entity IDs
    #[serde(default = "default_next_id")]
    next_id: u64,

    /// Shared vertex array
    #[serde(default)]
    pub vertices: Vec<Point2<f64>>,

    /// The entities in this sketch
    #[serde(default)]
    pub entities: Vec<SketchEntity>,

    /// Constraints defining geometric relationships
    ///
    /// Dimension values can be Starlark expressions referencing environment variables.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<Constraint>,
}

fn default_next_id() -> u64 {
    1
}

impl Default for Sketch {
    fn default() -> Self {
        Self::new()
    }
}

impl Sketch {
    /// Create a new empty sketch on the XY plane
    pub fn new() -> Self {
        Self {
            plane: SketchPlane::xy(),
            next_id: 1,
            vertices: Vec::new(),
            entities: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Create a sketch on a specific plane
    pub fn on_plane(plane: SketchPlane) -> Self {
        Self {
            plane,
            next_id: 1,
            vertices: Vec::new(),
            entities: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to the sketch
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Add a constraint and return self (builder pattern)
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Solve constraints and update vertex positions in place
    ///
    /// This is the main entry point for constraint solving. After solving,
    /// the sketch's vertices are updated to satisfy the constraints.
    ///
    /// For animations/sweeps, call this repeatedly - it automatically
    /// warm-starts from the current vertex positions.
    ///
    /// Returns the solve result with status and diagnostics.
    pub fn solve(&mut self, env: &Environment) -> Result<super::constraint::SolveResult> {
        let solver = super::constraint::Solver2D::new(self, env)?;
        let result = solver.solve()?;

        // Update vertices with solved positions (warm start for next solve)
        for (i, v) in self.vertices.iter_mut().enumerate() {
            if let Some((x, y)) = result.vertex(i) {
                v.x = x;
                v.y = y;
            }
        }

        Ok(result)
    }

    /// Generate a new unique entity ID
    fn next_entity_id(&mut self) -> EntityId {
        let id = EntityId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a vertex and return its index
    pub fn add_vertex(&mut self, point: Point2<f64>) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(point);
        idx
    }

    /// Add or get vertex index (deduplicates within tolerance)
    pub fn add_or_get_vertex(&mut self, point: Point2<f64>, tolerance: f64) -> usize {
        for (i, v) in self.vertices.iter().enumerate() {
            let dx = v.x - point.x;
            let dy = v.y - point.y;
            if dx * dx + dy * dy < tolerance * tolerance {
                return i;
            }
        }
        self.add_vertex(point)
    }

    /// Add a segment to the sketch, returning its assigned ID
    pub fn add(&mut self, segment: Segment2D) -> EntityId {
        let id = self.next_entity_id();
        self.entities.push(SketchEntity::new(id, segment));
        id
    }

    /// Add a segment with a name
    pub fn add_named(&mut self, name: impl Into<String>, segment: Segment2D) -> EntityId {
        let id = self.next_entity_id();
        self.entities
            .push(SketchEntity::new(id, segment).with_name(name));
        id
    }

    /// Add a line segment
    pub fn add_line(&mut self, start: Point2<f64>, end: Point2<f64>) -> EntityId {
        let start_idx = self.add_or_get_vertex(start, 1e-10);
        let end_idx = self.add_or_get_vertex(end, 1e-10);
        self.add(Segment2D::Line(Line::new(start_idx, end_idx)))
    }

    /// Add a circle
    pub fn add_circle(&mut self, center: Point2<f64>, radius: f64) -> EntityId {
        let center_idx = self.add_or_get_vertex(center, 1e-10);
        self.add(Segment2D::Circle(Circle2::new(center_idx, radius)))
    }

    /// Add an arc from center, angle, and winding
    ///
    /// The center is computed from the endpoints and angle.
    pub fn add_arc(
        &mut self,
        start: Point2<f64>,
        finish: Point2<f64>,
        angle: f64,
        winding: Winding,
    ) -> EntityId {
        let start_idx = self.add_or_get_vertex(start, 1e-10);
        let finish_idx = self.add_or_get_vertex(finish, 1e-10);
        self.add(Segment2D::Arc(Arc2::new(
            start_idx, finish_idx, angle, winding,
        )))
    }

    /// Get an entity by ID
    pub fn get(&self, id: EntityId) -> Option<&SketchEntity> {
        self.entities.iter().find(|e| e.id == id)
    }

    /// Get a mutable reference to an entity by ID
    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut SketchEntity> {
        self.entities.iter_mut().find(|e| e.id == id)
    }

    /// Get the start and end vertex indices for a line or arc entity
    ///
    /// Returns an error if the entity doesn't exist or has no endpoints.
    pub fn entity_endpoints(&self, id: EntityId) -> Result<(usize, usize)> {
        let entity = self.get(id).ok_or_else(|| {
            FeatureError::InvalidSketch(format!("Entity {} not found", id))
        })?;

        // Try to get endpoints using the Curve trait
        match entity.segment.end_indices() {
            Some([start, end]) => Ok((start, end)),
            None => Err(FeatureError::InvalidSketch(
                "Entity has no endpoints (e.g., closed circle)".into(),
            )),
        }
    }

    /// Get the center vertex index for a circle entity
    ///
    /// Returns an error if the entity doesn't exist or isn't a circle.
    pub fn entity_center(&self, id: EntityId) -> Result<usize> {
        let entity = self.get(id).ok_or_else(|| {
            FeatureError::InvalidSketch(format!("Entity {} not found", id))
        })?;

        match &entity.segment {
            Segment2D::Circle(circle) => Ok(circle.center),
            Segment2D::Arc(_)
            | Segment2D::Line(_)
            | Segment2D::Ellipse(_)
            | Segment2D::CubicBezier(_)
            | Segment2D::QuadraticBezier(_)
            | Segment2D::BSpline(_) => Err(FeatureError::GeometryError(
                "Entity has no center vertex".into(),
            )),
        }
    }

    /// Get an entity by name
    pub fn get_by_name(&self, name: &str) -> Option<&SketchEntity> {
        self.entities
            .iter()
            .find(|e| e.name.as_deref() == Some(name))
    }

    /// Get the number of entities (excluding construction geometry)
    pub fn profile_entity_count(&self) -> usize {
        self.entities.iter().filter(|e| !e.construction).count()
    }

    /// Create a rectangle sketch centered at origin
    pub fn rectangle(width: f64, height: f64) -> Self {
        let hw = width / 2.0;
        let hh = height / 2.0;

        let mut sketch = Self::new();
        let p0 = sketch.add_vertex(Point2::new(-hw, -hh));
        let p1 = sketch.add_vertex(Point2::new(hw, -hh));
        let p2 = sketch.add_vertex(Point2::new(hw, hh));
        let p3 = sketch.add_vertex(Point2::new(-hw, hh));

        sketch.add(Segment2D::Line(Line::new(p0, p1)));
        sketch.add(Segment2D::Line(Line::new(p1, p2)));
        sketch.add(Segment2D::Line(Line::new(p2, p3)));
        sketch.add(Segment2D::Line(Line::new(p3, p0)));

        sketch
    }

    /// Create a rectangle with corner at origin
    pub fn rectangle_corner(width: f64, height: f64) -> Self {
        let mut sketch = Self::new();
        let p0 = sketch.add_vertex(Point2::new(0.0, 0.0));
        let p1 = sketch.add_vertex(Point2::new(width, 0.0));
        let p2 = sketch.add_vertex(Point2::new(width, height));
        let p3 = sketch.add_vertex(Point2::new(0.0, height));

        sketch.add(Segment2D::Line(Line::new(p0, p1)));
        sketch.add(Segment2D::Line(Line::new(p1, p2)));
        sketch.add(Segment2D::Line(Line::new(p2, p3)));
        sketch.add(Segment2D::Line(Line::new(p3, p0)));

        sketch
    }

    /// Create a circle sketch centered at origin
    pub fn circle(radius: f64) -> Self {
        let mut sketch = Self::new();
        let center_idx = sketch.add_vertex(Point2::new(0.0, 0.0));
        sketch.add(Segment2D::Circle(Circle2::new(center_idx, radius)));
        sketch
    }

    /// Tessellate all entities into polygon(s)
    ///
    /// Returns a list of closed polygons. Each polygon is a list of points.
    /// The first polygon is the exterior, subsequent ones are holes (if winding differs).
    pub fn to_polygon(&self) -> Result<Vec<Vec<Point2<f64>>>> {
        self.to_polygon_with_tolerance(DEFAULT_TOLERANCE)
    }

    /// Tessellate with a specific tolerance
    pub fn to_polygon_with_tolerance(&self, tolerance: f64) -> Result<Vec<Vec<Point2<f64>>>> {
        // Filter out construction geometry
        let entities: Vec<_> = self.entities.iter().filter(|e| !e.construction).collect();

        if entities.is_empty() {
            return Err(FeatureError::InvalidSketch("Sketch has no entities".into()));
        }

        // Check for closed single entities (like Circle)
        if entities.len() == 1 && entities[0].segment.is_closed() {
            let mut points = entities[0].segment.discretize(&self.vertices, tolerance);
            // Close the polygon by adding the first point at the end
            if let Some(first) = points.first().copied() {
                points.push(first);
            }
            return Ok(vec![points]);
        }

        // For multiple entities, find connected chains
        let chains = find_connected_chains(&entities, &self.vertices, tolerance)?;

        let mut polygons = Vec::new();
        for chain in chains {
            let mut points = Vec::new();
            for idx in chain {
                let entity = &entities[idx];
                let discretized = entity.segment.discretize(&self.vertices, tolerance);
                points.extend(discretized);
            }
            if !points.is_empty() {
                polygons.push(points);
            }
        }

        if polygons.is_empty() {
            return Err(FeatureError::InvalidSketch(
                "Could not form closed polygon from entities".into(),
            ));
        }

        Ok(polygons)
    }
}

/// Find connected chains of entities that form closed loops
fn find_connected_chains(
    entities: &[&SketchEntity],
    vertices: &[Point2<f64>],
    tolerance: f64,
) -> Result<Vec<Vec<usize>>> {
    if entities.is_empty() {
        return Ok(vec![]);
    }

    let n = entities.len();
    let mut used = vec![false; n];
    let mut chains = Vec::new();

    // Simple greedy chain building
    while let Some(start_idx) = used.iter().position(|&u| !u) {
        let mut chain = vec![start_idx];
        used[start_idx] = true;

        let Some(chain_start) = entities[start_idx].start(vertices) else {
            continue; // Skip degenerate entities
        };
        let Some(mut chain_end) = entities[start_idx].finish(vertices) else {
            continue;
        };

        // Keep extending the chain
        loop {
            let mut found = false;

            for i in 0..n {
                if used[i] {
                    continue;
                }

                let Some(start) = entities[i].start(vertices) else {
                    continue;
                };
                let Some(end) = entities[i].finish(vertices) else {
                    continue;
                };

                // Check if this entity connects to the chain end
                if points_close(chain_end, start, tolerance) {
                    chain.push(i);
                    used[i] = true;
                    chain_end = end;
                    found = true;
                    break;
                }

                // Check if reversed entity connects
                if points_close(chain_end, end, tolerance) {
                    chain.push(i);
                    used[i] = true;
                    chain_end = start;
                    found = true;
                    break;
                }
            }

            if !found {
                break;
            }

            // Check if chain is closed
            if points_close(chain_end, chain_start, tolerance) {
                break;
            }
        }

        chains.push(chain);
    }

    Ok(chains)
}

fn points_close(a: Point2<f64>, b: Point2<f64>, tolerance: f64) -> bool {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy < tolerance * tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sketch_add_entity() {
        let mut sketch = Sketch::new();

        let id1 = sketch.add_line(Point2::new(0.0, 0.0), Point2::new(10.0, 0.0));
        let start_idx = sketch.add_or_get_vertex(Point2::new(10.0, 0.0), 1e-10);
        let end_idx = sketch.add_vertex(Point2::new(10.0, 10.0));
        let id2 = sketch.add_named("vertical", Segment2D::Line(Line::new(start_idx, end_idx)));

        assert_eq!(id1, EntityId(1));
        assert_eq!(id2, EntityId(2));
        assert_eq!(sketch.entities.len(), 2);
        assert_eq!(sketch.get_by_name("vertical").unwrap().id, id2);
    }

    #[test]
    fn test_rectangle_sketch() {
        let sketch = Sketch::rectangle(10.0, 5.0);

        assert_eq!(sketch.entities.len(), 4);
        assert_eq!(sketch.vertices.len(), 4);

        let polygons = sketch.to_polygon().unwrap();
        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].len(), 4);
    }

    #[test]
    fn test_circle_sketch() {
        let sketch = Sketch::circle(5.0);

        assert_eq!(sketch.entities.len(), 1);
        assert_eq!(sketch.vertices.len(), 1); // Just center

        let polygons = sketch.to_polygon().unwrap();
        assert_eq!(polygons.len(), 1);
        assert!(polygons[0].len() > 10); // Should have many points
    }

    #[test]
    fn test_sketch_serde_roundtrip() {
        let mut sketch = Sketch::rectangle(10.0, 5.0);
        sketch.plane = SketchPlane::xz();

        let json = serde_json::to_string_pretty(&sketch).unwrap();
        let parsed: Sketch = serde_json::from_str(&json).unwrap();

        assert_eq!(sketch.entities.len(), parsed.entities.len());
        assert_eq!(sketch.vertices.len(), parsed.vertices.len());

        // Entity IDs should be preserved
        for (orig, loaded) in sketch.entities.iter().zip(parsed.entities.iter()) {
            assert_eq!(orig.id, loaded.id);
        }
    }

    #[test]
    fn test_construction_geometry_excluded() {
        let mut sketch = Sketch::new();

        sketch.add_line(Point2::new(0.0, 0.0), Point2::new(10.0, 0.0));
        sketch.add_line(Point2::new(10.0, 0.0), Point2::new(10.0, 10.0));
        sketch.add_line(Point2::new(10.0, 10.0), Point2::new(0.0, 10.0));
        sketch.add_line(Point2::new(0.0, 10.0), Point2::new(0.0, 0.0));

        // Add a construction line
        let id = sketch.add_line(Point2::new(0.0, 0.0), Point2::new(10.0, 10.0));
        sketch.get_mut(id).unwrap().construction = true;

        let polygons = sketch.to_polygon().unwrap();
        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].len(), 4); // Construction line excluded

        assert_eq!(sketch.profile_entity_count(), 4);
    }

    #[test]
    fn test_entity_id_display() {
        assert_eq!(EntityId(42).to_string(), "e42");
    }

    #[test]
    fn test_vertex_deduplication() {
        let mut sketch = Sketch::new();
        let idx1 = sketch.add_or_get_vertex(Point2::new(0.0, 0.0), 0.01);
        let idx2 = sketch.add_or_get_vertex(Point2::new(0.001, 0.001), 0.01); // Within tolerance
        let idx3 = sketch.add_or_get_vertex(Point2::new(10.0, 0.0), 0.01); // New vertex

        assert_eq!(idx1, idx2);
        assert_ne!(idx1, idx3);
        assert_eq!(sketch.vertices.len(), 2);
    }
}
