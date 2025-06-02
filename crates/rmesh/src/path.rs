use anyhow::Result;
use approx::relative_eq;
use nalgebra::{Matrix3, Matrix4, Point2, Point3, Rotation3, Transform3, Unit, Vector3};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::*;

use crate::mesh::Trimesh;



pub enum Curve {
    Line {
        // indexes of points on a line.
        points: Vec<usize>,
    },
    Circle {
        // start is either the start point of a circular arc
        // or any point on the full circle
        start: usize,

        // The end point of the circular are or any point 
        // on the full circle that isn't colinear with the 
        // center and start points as we need to know the
        // direction of the normal this cicle is in
        end: usize,

        // center is the center of the circle
        center: usize,

        // we need to know if the circle is closed as start and end
        // need to be different points in every case, even for a full circle
        closed: bool,
    },
    Bezier {
        // indexes of control points for the bezier curve
        points: Vec<usize>,
    }
}


impl Curve {

    pub fn length(&self, vertices: &[Point3<f64>]) -> f64 {
        match self {
            Curve::Line { points } => {
                if points.len() < 2 {
                    return 0.0;
                }
                let start = vertices[points[0]];
                let end = vertices[points[1]];
                (end - start).norm()
            }
            Curve::Circle { start, end, center, closed } => {

                // get the actual points from the indexes
                let center_point = vertices[*center];
                let start_point = vertices[*start];
                let end_point = vertices[*end];

                // Calculate the radius
                let radius = (start_point - center_point).norm();

                if *closed {
                    // If the circle is closed the length is the circumference
                    return 2.0 * std::f64::consts::PI * radius;
                }

                // Calculate the angle between the start and end points
                let angle_start = (start_point - center_point).angle(&Vector3::x_axis());
                let angle_end = (end_point - center_point).angle(&Vector3::x_axis());

                // Determine the direction of the circle
                let direction = if *closed || angle_end > angle_start {
                    1.0
                } else {
                    -1.0
                };

                // Calculate the arc length
                radius * direction * (angle_end - angle_start).abs()
            }
            Curve::Bezier { points } => {
                todo!("Bezier curve length calculation is not implemented yet");
                
            }
        }
    }

    pub fn discrete(&self, vertices: &[Point3<f64>], resolution: usize) -> Vec<Point3<f64>> {
        match self {
            Curve::Line { points } => {
                if points.len() < 2 {
                    return vec![];
                }
                let start = vertices[points[0]];
                let end = vertices[points[1]];
                let direction = (end - start).normalize();
                let step = (end - start).norm() / (resolution as f64);
                (0..resolution)
                    .map(|i| start + direction * (i as f64 * step))
                    .collect()
            }
            Curve::Circle { start, end, center, closed } => {
                let center_point = vertices[*center];
                let start_point = vertices[*start];
                let end_point = vertices[*end];

                // Calculate the radius
                let radius = (start_point - center_point).norm();

                // Calculate the angle between the start and end points
                let angle_start = (start_point - center_point).angle(&Vector3::x_axis());
                let angle_end = (end_point - center_point).angle(&Vector3::x_axis());

                // Determine the direction of the circle
                let direction = if *closed || angle_end > angle_start {
                    1.0
                } else {
                    -1.0
                };

                // Generate points along the circle
                (0..resolution)
                    .map(|i| {
                        let t = angle_start + direction * (i as f64 / resolution as f64) * (angle_end - angle_start);
                        center_point + Vector3::new(radius * t.cos(), radius * t.sin(), 0.0)
                    })
                    .collect()
            }
            Curve::Bezier { points } => {
                if points.len() < 2 {
                    return vec![];
                }
                // Collect control points
                let control: Vec<Point3<f64>> = points.iter().map(|&i| vertices[i]).collect();
                let n = control.len() - 1;

                // Precompute binomial coefficients
                fn binomial(n: usize, k: usize) -> f64 {
                    (0..=n).fold(1.0, |acc, i| {
                        if i == k { acc } else { acc * (n - i) as f64 / (i + 1) as f64 }
                    }) * if k == 0 || k == n { 1.0 } else { 1.0 }
                }
                let binoms: Vec<f64> = (0..=n).map(|k| binomial(n, k)).collect();

                // Sample points along the curve
                (0..resolution)
                    .map(|step| {
                        let t = step as f64 / (resolution - 1) as f64;
                        let one_minus_t = 1.0 - t;
                        let mut pt = Point3::origin();
                        for (i, p) in control.iter().enumerate() {
                            let coeff = binoms[i]
                                * one_minus_t.powi((n - i) as i32)
                                * t.powi(i as i32);
                            pt += p.coords * coeff;
                        }
                        Point3::from(pt)
                    })
                    .collect()
            }
        }
    }
}


pub struct Path {
    pub entities: Vec<Curve>,
    pub vertices: Vec<Point3<f64>>,
}

impl Path {
    /// Create a new Path from a list of vertices and curves.
    pub fn new(vertices: Vec<Point3<f64>>, entities: Vec<Curve>) -> Self {
        Self { vertices, entities }
    }




}



    /// Create a rectangle path (no rounded corners).
pub fn rectangle(width: f64, height: f64) -> Path {
        let w = width / 2.0;
        let h = height / 2.0;

        let vertices = vec![
            Point3::new(-w, -h, 0.0),
            Point3::new(w, -h, 0.0),
            Point3::new(w, h, 0.0),
            Point3::new(-w, h, 0.0),
        ];

        let entities = vec![
            Curve::Line { points: vec![0, 1] },
            Curve::Line { points: vec![1, 2] },
            Curve::Line { points: vec![2, 3] },
            Curve::Line { points: vec![3, 0] },
        ];

        Path::new(vertices, entities)
    }



#[cfg(test)]
mod tests {

    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    #[test]
    fn test_rectangle() {
        let path = rectangle(10.0, 5.0);
        assert_eq!(path.vertices.len(), 4);
        assert_eq!(path.entities.len(), 4);

        // Check vertices
        assert_relative_eq!(path.vertices[0], Point3::new(-5.0, -2.5, 0.0));
        assert_relative_eq!(path.vertices[1], Point3::new(5.0, -2.5, 0.0));
        assert_relative_eq!(path.vertices[2], Point3::new(5.0, 2.5, 0.0));
        assert_relative_eq!(path.vertices[3], Point3::new(-5.0, 2.5, 0.0));

        // Check curves
        if let Curve::Line { points } = &path.entities[0] {
            assert_eq!(*points, vec![0, 1]);
        } else {
            panic!("Expected Line curve");
        }
    }

}
