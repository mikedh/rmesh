//! Line/polyline entity

use serde::{Deserialize, Serialize};

use super::Curve;

/// A polyline defined by vertex indices.
///
/// Stores 2 or more point indices. Use `.windows(2)` or `tuple_windows`
/// to iterate over consecutive line segments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Line {
    /// Vertex indices (minimum 2 points)
    pub points: Vec<usize>,
}

impl Line {
    /// Create a simple two-point line segment
    pub fn new(start: usize, finish: usize) -> Self {
        Self {
            points: vec![start, finish],
        }
    }

    /// Create a polyline from multiple point indices
    pub fn from_points(points: Vec<usize>) -> Self {
        assert!(points.len() >= 2, "Line requires at least 2 points");
        Self { points }
    }

    /// Get the number of points in this polyline
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if this line has no points
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Iterate over consecutive point index pairs (for line segments)
    pub fn segments(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.points.windows(2).map(|w| (w[0], w[1]))
    }
}

impl Curve for Line {
    fn end_indices(&self) -> Option<[usize; 2]> {
        Some([*self.points.first()?, *self.points.last()?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_two_points() {
        let line = Line::new(0, 1);
        assert_eq!(line.end_indices(), Some([0, 1]));
        assert_eq!(line.len(), 2);
        assert!(!line.is_closed());
    }

    #[test]
    fn test_polyline() {
        let line = Line::from_points(vec![0, 1, 2, 3]);
        assert_eq!(line.end_indices(), Some([0, 3]));
        assert_eq!(line.len(), 4);

        let segments: Vec<_> = line.segments().collect();
        assert_eq!(segments, vec![(0, 1), (1, 2), (2, 3)]);
    }
}
