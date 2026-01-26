//! Sketch module - collection of 2D entities on a plane with unique IDs
//!
//! Sketches are the foundation for feature operations like extrude and revolve.
//! Each entity has a stable ID that survives editing and serialization.

use nalgebra::Point2;
use serde::{Deserialize, Serialize};

use crate::path::{Segment2D, Line2D, Circle2D, Arc2D, Winding};
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

    /// The geometry of this entity
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

    /// Get the start point of this entity
    pub fn start(&self) -> Point2<f64> {
        self.segment.start()
    }

    /// Get the end point of this entity
    pub fn finish(&self) -> Point2<f64> {
        self.segment.finish()
    }
}

/// A 2D sketch containing entities on a plane
///
/// Sketches are the foundation for operations like extrude, revolve, etc.
/// Each entity has a stable ID that survives editing and serialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sketch {
    /// The plane this sketch lies on
    #[serde(default)]
    pub plane: SketchPlane,

    /// Counter for generating unique entity IDs
    #[serde(default = "default_next_id")]
    next_id: u64,

    /// The entities in this sketch
    #[serde(default)]
    pub entities: Vec<SketchEntity>,
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
            entities: Vec::new(),
        }
    }

    /// Create a sketch on a specific plane
    pub fn on_plane(plane: SketchPlane) -> Self {
        Self {
            plane,
            next_id: 1,
            entities: Vec::new(),
        }
    }

    /// Generate a new unique entity ID
    fn next_entity_id(&mut self) -> EntityId {
        let id = EntityId(self.next_id);
        self.next_id += 1;
        id
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
        self.entities.push(SketchEntity::new(id, segment).with_name(name));
        id
    }

    /// Add a line segment
    pub fn add_line(&mut self, start: Point2<f64>, end: Point2<f64>) -> EntityId {
        self.add(Segment2D::Line(Line2D::new(start, end)))
    }

    /// Add a circle
    pub fn add_circle(&mut self, center: Point2<f64>, radius: f64) -> EntityId {
        self.add(Segment2D::Circle(Circle2D::new(center, radius)))
    }

    /// Add an arc
    pub fn add_arc(
        &mut self,
        start: Point2<f64>,
        finish: Point2<f64>,
        center: Point2<f64>,
        winding: Winding,
    ) -> EntityId {
        self.add(Segment2D::Arc(Arc2D::new(start, finish, center, winding)))
    }

    /// Get an entity by ID
    pub fn get(&self, id: EntityId) -> Option<&SketchEntity> {
        self.entities.iter().find(|e| e.id == id)
    }

    /// Get a mutable reference to an entity by ID
    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut SketchEntity> {
        self.entities.iter_mut().find(|e| e.id == id)
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
        let p0 = Point2::new(-hw, -hh);
        let p1 = Point2::new(hw, -hh);
        let p2 = Point2::new(hw, hh);
        let p3 = Point2::new(-hw, hh);

        sketch.add_line(p0, p1);
        sketch.add_line(p1, p2);
        sketch.add_line(p2, p3);
        sketch.add_line(p3, p0);

        sketch
    }

    /// Create a rectangle with corner at origin
    pub fn rectangle_corner(width: f64, height: f64) -> Self {
        let mut sketch = Self::new();
        let p0 = Point2::new(0.0, 0.0);
        let p1 = Point2::new(width, 0.0);
        let p2 = Point2::new(width, height);
        let p3 = Point2::new(0.0, height);

        sketch.add_line(p0, p1);
        sketch.add_line(p1, p2);
        sketch.add_line(p2, p3);
        sketch.add_line(p3, p0);

        sketch
    }

    /// Create a circle sketch centered at origin
    pub fn circle(radius: f64) -> Self {
        let mut sketch = Self::new();
        sketch.add_circle(Point2::new(0.0, 0.0), radius);
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
            let mut points = entities[0].segment.tessellate(tolerance);
            // Close the polygon by adding the end point
            points.push(entities[0].finish());
            return Ok(vec![points]);
        }

        // For multiple entities, find connected chains
        let chains = find_connected_chains(&entities, tolerance)?;

        let mut polygons = Vec::new();
        for chain in chains {
            let mut points = Vec::new();
            for idx in chain {
                let entity = &entities[idx];
                let tessellated = entity.segment.tessellate(tolerance);
                points.extend(tessellated);
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
fn find_connected_chains(entities: &[&SketchEntity], tolerance: f64) -> Result<Vec<Vec<usize>>> {
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

        let chain_start = entities[start_idx].start();
        let mut chain_end = entities[start_idx].finish();

        // Keep extending the chain
        loop {
            let mut found = false;

            for i in 0..n {
                if used[i] {
                    continue;
                }

                let start = entities[i].start();
                let end = entities[i].finish();

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
        let id2 = sketch.add_named(
            "vertical",
            Segment2D::Line(Line2D::new(Point2::new(10.0, 0.0), Point2::new(10.0, 10.0))),
        );

        assert_eq!(id1, EntityId(1));
        assert_eq!(id2, EntityId(2));
        assert_eq!(sketch.entities.len(), 2);
        assert_eq!(sketch.get_by_name("vertical").unwrap().id, id2);
    }

    #[test]
    fn test_rectangle_sketch() {
        let sketch = Sketch::rectangle(10.0, 5.0);

        assert_eq!(sketch.entities.len(), 4);

        let polygons = sketch.to_polygon().unwrap();
        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].len(), 4);
    }

    #[test]
    fn test_circle_sketch() {
        let sketch = Sketch::circle(5.0);

        assert_eq!(sketch.entities.len(), 1);

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
}
