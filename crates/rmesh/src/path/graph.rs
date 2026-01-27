//! Entity connectivity graph for path analysis
//!
//! This module provides tools for analyzing segment connectivity,
//! finding connected components, and detecting closed rings (cycles).
//!
//! With indexed vertices, connectivity is determined by direct index comparison,
//! which is simpler and more efficient than tolerance-based point comparison.

use ahash::{HashMap, HashMapExt};

use super::{Path2D, Path3D, Segment2D, Segment3D};

/// Graph of segment connectivity based on shared vertex indices
#[derive(Debug, Clone)]
pub struct EntityGraph {
    /// Number of segments
    num_segments: usize,
    /// Adjacency list: segment index -> list of connected segment indices
    adjacency: Vec<Vec<usize>>,
    /// Map from vertex index to segment indices that use that endpoint
    endpoint_map: HashMap<usize, Vec<usize>>,
}

impl EntityGraph {
    /// Build a connectivity graph from a 2D path
    ///
    /// Uses direct index comparison for connectivity (no tolerance needed).
    pub fn from_path_2d(path: &Path2D) -> Self {
        let num_segments = path.segments.len();
        let mut endpoint_map: HashMap<usize, Vec<usize>> = HashMap::new();

        // Build endpoint map using vertex indices
        for (idx, segment) in path.segments.iter().enumerate() {
            let Some([start_idx, finish_idx]) = segment.end_indices() else {
                continue; // Skip degenerate segments
            };

            endpoint_map.entry(start_idx).or_default().push(idx);
            // Only add finish if different from start (avoid double-counting closed segments)
            if finish_idx != start_idx {
                endpoint_map.entry(finish_idx).or_default().push(idx);
            }
        }

        // Build adjacency list
        let mut adjacency = vec![Vec::new(); num_segments];
        for indices in endpoint_map.values() {
            // All segments sharing this endpoint are connected
            for &i in indices {
                for &j in indices {
                    if i != j && !adjacency[i].contains(&j) {
                        adjacency[i].push(j);
                    }
                }
            }
        }

        Self {
            num_segments,
            adjacency,
            endpoint_map,
        }
    }

    /// Build a connectivity graph from a 3D path
    pub fn from_path_3d(path: &Path3D) -> Self {
        let num_segments = path.segments.len();
        let mut endpoint_map: HashMap<usize, Vec<usize>> = HashMap::new();

        // Build endpoint map using vertex indices
        for (idx, segment) in path.segments.iter().enumerate() {
            let Some([start_idx, finish_idx]) = segment.end_indices() else {
                continue; // Skip degenerate segments
            };

            endpoint_map.entry(start_idx).or_default().push(idx);
            if finish_idx != start_idx {
                endpoint_map.entry(finish_idx).or_default().push(idx);
            }
        }

        // Build adjacency list
        let mut adjacency = vec![Vec::new(); num_segments];
        for indices in endpoint_map.values() {
            for &i in indices {
                for &j in indices {
                    if i != j && !adjacency[i].contains(&j) {
                        adjacency[i].push(j);
                    }
                }
            }
        }

        Self {
            num_segments,
            adjacency,
            endpoint_map,
        }
    }

    /// Find all connected components
    ///
    /// Returns groups of segment indices that are connected.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.num_segments];
        let mut components = Vec::new();

        for start in 0..self.num_segments {
            if visited[start] {
                continue;
            }

            let mut component = Vec::new();
            let mut stack = vec![start];

            while let Some(idx) = stack.pop() {
                if visited[idx] {
                    continue;
                }
                visited[idx] = true;
                component.push(idx);

                for &neighbor in &self.adjacency[idx] {
                    if !visited[neighbor] {
                        stack.push(neighbor);
                    }
                }
            }

            component.sort_unstable();
            components.push(component);
        }

        components
    }

    /// Find closed rings (cycles) in the graph for 2D segments
    ///
    /// A ring is a sequence of segments where:
    /// - Each segment connects to the next via a shared endpoint
    /// - The last segment connects back to the first
    /// - The segments form a proper cycle (not self-intersecting)
    pub fn find_rings(&self, segments: &[Segment2D]) -> Vec<Vec<usize>> {
        let mut rings = Vec::new();

        // First, check for single closed segments (circles, ellipses)
        for (idx, segment) in segments.iter().enumerate() {
            if segment.is_closed() {
                rings.push(vec![idx]);
            }
        }

        // Find multi-segment rings in each connected component
        for component in self.connected_components() {
            if component.len() < 2 {
                continue;
            }

            // Try to find a cycle starting from each segment in the component
            if let Some(ring) = self.find_cycle_from_component(&component, segments) {
                // Check if we already have this ring (avoid duplicates)
                let mut sorted_ring = ring.clone();
                sorted_ring.sort_unstable();
                let is_duplicate = rings.iter().any(|r| {
                    let mut sr = r.clone();
                    sr.sort_unstable();
                    sr == sorted_ring
                });

                if !is_duplicate {
                    rings.push(ring);
                }
            }
        }

        rings
    }

    /// Find closed rings for 3D segments
    pub fn find_rings_3d(&self, segments: &[Segment3D]) -> Vec<Vec<usize>> {
        let mut rings = Vec::new();

        // Check for single closed segments
        for (idx, segment) in segments.iter().enumerate() {
            if segment.is_closed() {
                rings.push(vec![idx]);
            }
        }

        // Find multi-segment rings
        for component in self.connected_components() {
            if component.len() < 2 {
                continue;
            }

            if let Some(ring) = self.find_cycle_from_component_3d(&component, segments) {
                let mut sorted_ring = ring.clone();
                sorted_ring.sort_unstable();
                let is_duplicate = rings.iter().any(|r| {
                    let mut sr = r.clone();
                    sr.sort_unstable();
                    sr == sorted_ring
                });

                if !is_duplicate {
                    rings.push(ring);
                }
            }
        }

        rings
    }

    /// Find a cycle within a connected component (2D)
    fn find_cycle_from_component(
        &self,
        component: &[usize],
        segments: &[Segment2D],
    ) -> Option<Vec<usize>> {
        if component.is_empty() {
            return None;
        }

        // Start from the first segment and try to walk back to it
        let start_idx = component[0];
        let [start_vertex, mut current_vertex] = segments[start_idx].end_indices()?;

        // DFS to find a path back to start
        let mut path = vec![start_idx];
        let mut visited = vec![false; self.num_segments];
        visited[start_idx] = true;

        loop {
            // Check if we've returned to start
            if current_vertex == start_vertex && path.len() > 1 {
                return Some(path);
            }

            // Find next unvisited segment sharing this vertex
            let mut found = false;
            if let Some(candidates) = self.endpoint_map.get(&current_vertex) {
                for &next_idx in candidates {
                    if visited[next_idx] || !component.contains(&next_idx) {
                        continue;
                    }

                    let next_seg = &segments[next_idx];
                    let Some([next_start, next_finish]) = next_seg.end_indices() else {
                        continue;
                    };

                    // Determine which endpoint is shared and which is the continuation
                    let (shares_start, shares_finish) =
                        (next_start == current_vertex, next_finish == current_vertex);

                    if shares_start || shares_finish {
                        visited[next_idx] = true;
                        path.push(next_idx);
                        current_vertex = if shares_start {
                            next_finish
                        } else {
                            next_start
                        };
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                // Dead end - this component doesn't form a complete ring
                break;
            }
        }

        None
    }

    /// Find a cycle within a connected component (3D)
    fn find_cycle_from_component_3d(
        &self,
        component: &[usize],
        segments: &[Segment3D],
    ) -> Option<Vec<usize>> {
        if component.is_empty() {
            return None;
        }

        let start_idx = component[0];
        let [start_vertex, mut current_vertex] = segments[start_idx].end_indices()?;

        let mut path = vec![start_idx];
        let mut visited = vec![false; self.num_segments];
        visited[start_idx] = true;

        loop {
            if current_vertex == start_vertex && path.len() > 1 {
                return Some(path);
            }

            let mut found = false;
            if let Some(candidates) = self.endpoint_map.get(&current_vertex) {
                for &next_idx in candidates {
                    if visited[next_idx] || !component.contains(&next_idx) {
                        continue;
                    }

                    let next_seg = &segments[next_idx];
                    let Some([next_start, next_finish]) = next_seg.end_indices() else {
                        continue;
                    };

                    let (shares_start, shares_finish) =
                        (next_start == current_vertex, next_finish == current_vertex);

                    if shares_start || shares_finish {
                        visited[next_idx] = true;
                        path.push(next_idx);
                        current_vertex = if shares_start {
                            next_finish
                        } else {
                            next_start
                        };
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                break;
            }
        }

        None
    }

    /// Get the number of segments in the graph
    pub fn num_segments(&self) -> usize {
        self.num_segments
    }

    /// Get the adjacency list for a segment
    pub fn neighbors(&self, segment_idx: usize) -> &[usize] {
        &self.adjacency[segment_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::Line;
    use nalgebra::{Point2, Point3};

    #[test]
    fn test_connected_components() {
        // Two separate line segments
        let path = Path2D::from_vertices_and_segments(
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(10.0, 0.0),
                Point2::new(11.0, 0.0),
            ],
            vec![
                Segment2D::Line(Line::new(0, 1)),
                Segment2D::Line(Line::new(2, 3)),
            ],
        );

        let graph = EntityGraph::from_path_2d(&path);
        let components = graph.connected_components();

        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_connected_segments() {
        // Two connected line segments sharing vertex 1
        let path = Path2D::from_vertices_and_segments(
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(2.0, 0.0),
            ],
            vec![
                Segment2D::Line(Line::new(0, 1)),
                Segment2D::Line(Line::new(1, 2)),
            ],
        );

        let graph = EntityGraph::from_path_2d(&path);
        let components = graph.connected_components();

        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 2);
    }

    #[test]
    fn test_find_ring() {
        // Square made of 4 line segments
        let path = Path2D::from_vertices_and_segments(
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(1.0, 1.0),
                Point2::new(0.0, 1.0),
            ],
            vec![
                Segment2D::Line(Line::new(0, 1)),
                Segment2D::Line(Line::new(1, 2)),
                Segment2D::Line(Line::new(2, 3)),
                Segment2D::Line(Line::new(3, 0)),
            ],
        );

        let graph = EntityGraph::from_path_2d(&path);
        let rings = graph.find_rings(&path.segments);

        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0].len(), 4);
    }

    #[test]
    fn test_single_closed_segment() {
        use crate::path::Circle2;

        let path = Path2D::from_vertices_and_segments(
            vec![Point2::origin()],
            vec![Segment2D::Circle(Circle2::new(0, 1.0))],
        );

        let graph = EntityGraph::from_path_2d(&path);
        let rings = graph.find_rings(&path.segments);

        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0], vec![0]);
    }

    #[test]
    fn test_3d_connected_components() {
        let path = Path3D::from_vertices_and_segments(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
            vec![
                Segment3D::Line(Line::new(0, 1)),
                Segment3D::Line(Line::new(1, 2)),
            ],
        );

        let graph = EntityGraph::from_path_3d(&path);
        let components = graph.connected_components();

        assert_eq!(components.len(), 1);
    }
}
