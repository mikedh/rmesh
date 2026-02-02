//! SVG path parsing and generation for Path2D
//!
//! This module handles conversion between SVG path strings and Path2D structures.
//!
//! ## Parsing
//! - `from_svg()` accepts full SVG documents or just `<path d="...">` elements
//! - Supports: rect, circle, ellipse, line, polyline, polygon, and path elements
//! - All elements are converted to path segments internally
//!
//! ## Export
//! - `to_svg()` only emits `<path d="...">` strings
//! - All geometry is converted to SVG path commands

use std::fmt::Write;

use nalgebra::Point2;

use super::entity::{Arc2, Circle2, CubicBezier, Ellipse2, Line, QuadraticBezier, Winding};
use super::{Path2D, Segment2D};

/// Error type for SVG parsing
#[derive(Debug, Clone)]
pub struct SvgError(pub String);

impl std::fmt::Display for SvgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SVG error: {}", self.0)
    }
}

impl std::error::Error for SvgError {}

type Result<T> = std::result::Result<T, SvgError>;

impl Path2D {
    /// Create a Path2D from an SVG string.
    ///
    /// Accepts either a full SVG document or a path d-string.
    /// Supports: path, rect, circle, ellipse, line, polyline, polygon elements.
    ///
    /// # Example
    /// ```
    /// use rmesh::path::Path2D;
    ///
    /// // From path d-string
    /// let path = Path2D::from_svg("M0,0 L10,0 L10,10 L0,10 Z").unwrap();
    ///
    /// // From SVG element
    /// let path = Path2D::from_svg("<rect x='0' y='0' width='10' height='10'/>").unwrap();
    /// ```
    pub fn from_svg(svg: &str) -> Result<Self> {
        let trimmed = svg.trim();

        // Check if this is a full SVG element or just a path d-string
        if trimmed.starts_with('<') {
            parse_svg_element(trimmed)
        } else {
            // Assume it's a path d-string
            parse_svg_path(trimmed)
        }
    }

    /// Convert this path to an SVG path d-string.
    ///
    /// All geometry is converted to SVG path commands (M, L, C, Q, A, Z).
    /// If the path forms a closed ring, Z is appended.
    pub fn to_svg(&self) -> String {
        let mut path = String::new();
        let mut current_pos: Option<Point2<f64>> = None;

        for segment in &self.segments {
            if let Some(svg) = segment_to_svg(segment, &self.vertices, &mut current_pos) {
                path.push_str(&svg);
            }
        }

        // Close path if it forms a ring (start and finish are the same)
        if !self.segments.is_empty()
            && let (Some(first_start), Some(last_finish)) = (
                self.segments[0].start(&self.vertices),
                self.segments.last().and_then(|s| s.finish(&self.vertices)),
            )
        {
            let is_closed = (first_start.x - last_finish.x).abs() < 1e-10
                && (first_start.y - last_finish.y).abs() < 1e-10;
            if is_closed && !path.is_empty() {
                path.push_str(" Z");
            }
        }

        path
    }

    /// Convert this path to a complete SVG document.
    ///
    /// Includes viewBox calculated from path bounds.
    pub fn to_svg_document(&self) -> String {
        let path_str = self.to_svg();
        let bounds = self.bounds();

        let (min_x, min_y, max_x, max_y) = match bounds {
            Some((min, max)) => (min.x, min.y, max.x, max.y),
            None => (0.0, 0.0, 100.0, 100.0),
        };

        let width = max_x - min_x;
        let height = max_y - min_y;
        let padding = (width.max(height)) * 0.1;

        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{} {} {} {}">
  <path d="{}" fill="none" stroke="black" stroke-width="1"/>
</svg>"#,
            min_x - padding,
            min_y - padding,
            width + padding * 2.0,
            height + padding * 2.0,
            path_str
        )
    }
}

// =============================================================================
// SVG Element Parsing
// =============================================================================

/// Parse an SVG element (path, rect, circle, etc.)
fn parse_svg_element(svg: &str) -> Result<Path2D> {
    let svg = svg.trim();

    // Simple tag detection
    if svg.starts_with("<path") {
        parse_path_element(svg)
    } else if svg.starts_with("<rect") {
        parse_rect_element(svg)
    } else if svg.starts_with("<circle") {
        parse_circle_element(svg)
    } else if svg.starts_with("<ellipse") {
        parse_ellipse_element(svg)
    } else if svg.starts_with("<line") {
        Ok(parse_line_element(svg))
    } else if svg.starts_with("<polyline") {
        parse_polyline_element(svg, false)
    } else if svg.starts_with("<polygon") {
        parse_polyline_element(svg, true)
    } else if svg.starts_with("<svg") {
        // Full SVG document - extract path elements
        parse_svg_document(svg)
    } else {
        Err(SvgError(format!(
            "Unknown SVG element: {}",
            &svg[..svg.len().min(20)]
        )))
    }
}

fn parse_path_element(svg: &str) -> Result<Path2D> {
    let d = extract_attribute(svg, "d").ok_or_else(|| SvgError("Missing 'd' attribute".into()))?;
    parse_svg_path(&d)
}

fn parse_rect_element(svg: &str) -> Result<Path2D> {
    let x: f64 = extract_attribute(svg, "x")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let y: f64 = extract_attribute(svg, "y")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let width: f64 = extract_attribute(svg, "width")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| SvgError("Missing 'width' attribute".into()))?;
    let height: f64 = extract_attribute(svg, "height")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| SvgError("Missing 'height' attribute".into()))?;

    // Optional corner radii
    let rx: f64 = extract_attribute(svg, "rx")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let ry: f64 = extract_attribute(svg, "ry")
        .and_then(|s| s.parse().ok())
        .unwrap_or(rx);

    if rx > 0.0 || ry > 0.0 {
        // Rounded rectangle - convert to path with arcs
        parse_svg_path(&rounded_rect_to_path(x, y, width, height, rx, ry))
    } else {
        // Simple rectangle
        let vertices = vec![
            Point2::new(x, y),
            Point2::new(x + width, y),
            Point2::new(x + width, y + height),
            Point2::new(x, y + height),
        ];
        let segments = vec![
            Segment2D::Line(Line::new(0, 1)),
            Segment2D::Line(Line::new(1, 2)),
            Segment2D::Line(Line::new(2, 3)),
            Segment2D::Line(Line::new(3, 0)),
        ];
        Ok(Path2D::from_vertices_and_segments(vertices, segments))
    }
}

fn rounded_rect_to_path(x: f64, y: f64, w: f64, h: f64, rx: f64, ry: f64) -> String {
    format!(
        "M{},{} h{} a{},{} 0 0 1 {},{} v{} a{},{} 0 0 1 -{},{} h-{} a{},{} 0 0 1 -{}-{} v-{} a{},{} 0 0 1 {}-{} Z",
        x + rx,
        y,
        w - 2.0 * rx,
        rx,
        ry,
        rx,
        ry,
        h - 2.0 * ry,
        rx,
        ry,
        rx,
        ry,
        w - 2.0 * rx,
        rx,
        ry,
        rx,
        ry,
        h - 2.0 * ry,
        rx,
        ry,
        rx,
        ry
    )
}

fn parse_circle_element(svg: &str) -> Result<Path2D> {
    let cx: f64 = extract_attribute(svg, "cx")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let cy: f64 = extract_attribute(svg, "cy")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let r: f64 = extract_attribute(svg, "r")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| SvgError("Missing 'r' attribute".into()))?;

    let vertices = vec![Point2::new(cx, cy)];
    let segments = vec![Segment2D::Circle(Circle2::new(0, r))];
    Ok(Path2D::from_vertices_and_segments(vertices, segments))
}

fn parse_ellipse_element(svg: &str) -> Result<Path2D> {
    let cx: f64 = extract_attribute(svg, "cx")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let cy: f64 = extract_attribute(svg, "cy")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let rx: f64 = extract_attribute(svg, "rx")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| SvgError("Missing 'rx' attribute".into()))?;
    let ry: f64 = extract_attribute(svg, "ry")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| SvgError("Missing 'ry' attribute".into()))?;

    let vertices = vec![Point2::new(cx, cy)];
    let segments = vec![Segment2D::Ellipse(Ellipse2::axis_aligned(
        0,
        rx.max(ry),
        rx.min(ry),
    ))];
    Ok(Path2D::from_vertices_and_segments(vertices, segments))
}

fn parse_line_element(svg: &str) -> Path2D {
    let x1: f64 = extract_attribute(svg, "x1")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let y1: f64 = extract_attribute(svg, "y1")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let x2: f64 = extract_attribute(svg, "x2")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let y2: f64 = extract_attribute(svg, "y2")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let vertices = vec![Point2::new(x1, y1), Point2::new(x2, y2)];
    let segments = vec![Segment2D::Line(Line::new(0, 1))];
    Path2D::from_vertices_and_segments(vertices, segments)
}

fn parse_polyline_element(svg: &str, close: bool) -> Result<Path2D> {
    let points_str = extract_attribute(svg, "points")
        .ok_or_else(|| SvgError("Missing 'points' attribute".into()))?;

    let points = parse_points_list(&points_str)?;
    if points.len() < 2 {
        return Err(SvgError("Polyline needs at least 2 points".into()));
    }

    let vertices = points.clone();
    let mut segments: Vec<Segment2D> = (0..points.len() - 1)
        .map(|i| Segment2D::Line(Line::new(i, i + 1)))
        .collect();

    if close && points.len() > 2 {
        segments.push(Segment2D::Line(Line::new(points.len() - 1, 0)));
    }

    Ok(Path2D::from_vertices_and_segments(vertices, segments))
}

fn parse_svg_document(svg: &str) -> Result<Path2D> {
    // Simple extraction of first path element from SVG document
    // A full implementation would parse all elements
    if let Some(start) = svg.find("<path") {
        let end = svg[start..]
            .find("/>")
            .or_else(|| svg[start..].find("</path>"));
        if let Some(end_idx) = end {
            let path_elem = &svg[start..start + end_idx + 2];
            return parse_path_element(path_elem);
        }
    }

    Err(SvgError("No path element found in SVG document".into()))
}

fn parse_points_list(s: &str) -> Result<Vec<Point2<f64>>> {
    let mut points = Vec::new();
    let mut chars = s.chars().peekable();

    while chars.peek().is_some() {
        skip_separators(&mut chars);
        if chars.peek().is_none() {
            break;
        }
        let x = parse_number(&mut chars)?;
        skip_separators(&mut chars);
        let y = parse_number(&mut chars)?;
        points.push(Point2::new(x, y));
    }

    Ok(points)
}

fn extract_attribute(svg: &str, name: &str) -> Option<String> {
    // Match name="value" or name='value'
    let pattern1 = format!("{}=\"", name);
    let pattern2 = format!("{}='", name);

    let (start, quote) = if let Some(pos) = svg.find(&pattern1) {
        (pos + pattern1.len(), '"')
    } else if let Some(pos) = svg.find(&pattern2) {
        (pos + pattern2.len(), '\'')
    } else {
        return None;
    };

    let rest = &svg[start..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

// =============================================================================
// SVG Path d-string Parsing
// =============================================================================

/// SVG path parser state
struct SvgPathParser {
    vertices: Vec<Point2<f64>>,
    segments: Vec<Segment2D>,
    current: Point2<f64>,
    start: Point2<f64>,
    last_control: Option<Point2<f64>>,
}

impl SvgPathParser {
    fn new() -> Self {
        Self {
            vertices: Vec::new(),
            segments: Vec::new(),
            current: Point2::new(0.0, 0.0),
            start: Point2::new(0.0, 0.0),
            last_control: None,
        }
    }

    /// Add a vertex and return its index (with deduplication)
    fn add_vertex(&mut self, point: Point2<f64>) -> usize {
        const TOL: f64 = 1e-10;
        for (i, v) in self.vertices.iter().enumerate() {
            let dx = v.x - point.x;
            let dy = v.y - point.y;
            if dx * dx + dy * dy < TOL * TOL {
                return i;
            }
        }
        let idx = self.vertices.len();
        self.vertices.push(point);
        idx
    }

    fn into_path(self) -> Path2D {
        Path2D::from_vertices_and_segments(self.vertices, self.segments)
    }
}

/// Parse an SVG path d-string into a Path2D
fn parse_svg_path(path: &str) -> Result<Path2D> {
    let mut parser = SvgPathParser::new();
    let mut chars = path.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            'M' => {
                let (x, y) = parse_coordinate_pair(&mut chars)?;
                parser.current = Point2::new(x, y);
                parser.start = parser.current;
                parser.last_control = None;

                // M can be followed by implicit L commands
                skip_separators(&mut chars);
                while chars
                    .peek()
                    .is_some_and(|c| c.is_ascii_digit() || *c == '-' || *c == '+' || *c == '.')
                {
                    let (x, y) = parse_coordinate_pair(&mut chars)?;
                    let end = Point2::new(x, y);
                    let start_idx = parser.add_vertex(parser.current);
                    let end_idx = parser.add_vertex(end);
                    parser
                        .segments
                        .push(Segment2D::Line(Line::new(start_idx, end_idx)));
                    parser.current = end;
                    skip_separators(&mut chars);
                }
            }
            'm' => {
                let (dx, dy) = parse_coordinate_pair(&mut chars)?;
                parser.current = Point2::new(parser.current.x + dx, parser.current.y + dy);
                parser.start = parser.current;
                parser.last_control = None;

                // m can be followed by implicit l commands
                skip_separators(&mut chars);
                while chars
                    .peek()
                    .is_some_and(|c| c.is_ascii_digit() || *c == '-' || *c == '+' || *c == '.')
                {
                    let (dx, dy) = parse_coordinate_pair(&mut chars)?;
                    let end = Point2::new(parser.current.x + dx, parser.current.y + dy);
                    let start_idx = parser.add_vertex(parser.current);
                    let end_idx = parser.add_vertex(end);
                    parser
                        .segments
                        .push(Segment2D::Line(Line::new(start_idx, end_idx)));
                    parser.current = end;
                    skip_separators(&mut chars);
                }
            }
            'L' => loop {
                let (x, y) = parse_coordinate_pair(&mut chars)?;
                let end = Point2::new(x, y);
                let start_idx = parser.add_vertex(parser.current);
                let end_idx = parser.add_vertex(end);
                parser
                    .segments
                    .push(Segment2D::Line(Line::new(start_idx, end_idx)));
                parser.current = end;
                parser.last_control = None;

                skip_separators(&mut chars);
                if !chars
                    .peek()
                    .is_some_and(|c| c.is_ascii_digit() || *c == '-' || *c == '+' || *c == '.')
                {
                    break;
                }
            },
            'l' => loop {
                let (dx, dy) = parse_coordinate_pair(&mut chars)?;
                let end = Point2::new(parser.current.x + dx, parser.current.y + dy);
                let start_idx = parser.add_vertex(parser.current);
                let end_idx = parser.add_vertex(end);
                parser
                    .segments
                    .push(Segment2D::Line(Line::new(start_idx, end_idx)));
                parser.current = end;
                parser.last_control = None;

                skip_separators(&mut chars);
                if !chars
                    .peek()
                    .is_some_and(|c| c.is_ascii_digit() || *c == '-' || *c == '+' || *c == '.')
                {
                    break;
                }
            },
            'H' => {
                let x = parse_number(&mut chars)?;
                let end = Point2::new(x, parser.current.y);
                let start_idx = parser.add_vertex(parser.current);
                let end_idx = parser.add_vertex(end);
                parser
                    .segments
                    .push(Segment2D::Line(Line::new(start_idx, end_idx)));
                parser.current = end;
                parser.last_control = None;
            }
            'h' => {
                let dx = parse_number(&mut chars)?;
                let end = Point2::new(parser.current.x + dx, parser.current.y);
                let start_idx = parser.add_vertex(parser.current);
                let end_idx = parser.add_vertex(end);
                parser
                    .segments
                    .push(Segment2D::Line(Line::new(start_idx, end_idx)));
                parser.current = end;
                parser.last_control = None;
            }
            'V' => {
                let y = parse_number(&mut chars)?;
                let end = Point2::new(parser.current.x, y);
                let start_idx = parser.add_vertex(parser.current);
                let end_idx = parser.add_vertex(end);
                parser
                    .segments
                    .push(Segment2D::Line(Line::new(start_idx, end_idx)));
                parser.current = end;
                parser.last_control = None;
            }
            'v' => {
                let dy = parse_number(&mut chars)?;
                let end = Point2::new(parser.current.x, parser.current.y + dy);
                let start_idx = parser.add_vertex(parser.current);
                let end_idx = parser.add_vertex(end);
                parser
                    .segments
                    .push(Segment2D::Line(Line::new(start_idx, end_idx)));
                parser.current = end;
                parser.last_control = None;
            }
            'C' => {
                let (x1, y1) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (x2, y2) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (x, y) = parse_coordinate_pair(&mut chars)?;

                let p1 = Point2::new(x1, y1);
                let p2 = Point2::new(x2, y2);
                let end = Point2::new(x, y);

                let p0_idx = parser.add_vertex(parser.current);
                let p1_idx = parser.add_vertex(p1);
                let p2_idx = parser.add_vertex(p2);
                let p3_idx = parser.add_vertex(end);

                parser
                    .segments
                    .push(Segment2D::CubicBezier(CubicBezier::new(
                        p0_idx, p1_idx, p2_idx, p3_idx,
                    )));
                parser.last_control = Some(p2);
                parser.current = end;
            }
            'c' => {
                let (dx1, dy1) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (dx2, dy2) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (dx, dy) = parse_coordinate_pair(&mut chars)?;

                let p1 = Point2::new(parser.current.x + dx1, parser.current.y + dy1);
                let p2 = Point2::new(parser.current.x + dx2, parser.current.y + dy2);
                let end = Point2::new(parser.current.x + dx, parser.current.y + dy);

                let p0_idx = parser.add_vertex(parser.current);
                let p1_idx = parser.add_vertex(p1);
                let p2_idx = parser.add_vertex(p2);
                let p3_idx = parser.add_vertex(end);

                parser
                    .segments
                    .push(Segment2D::CubicBezier(CubicBezier::new(
                        p0_idx, p1_idx, p2_idx, p3_idx,
                    )));
                parser.last_control = Some(p2);
                parser.current = end;
            }
            'S' => {
                // Smooth cubic bezier - reflects previous control point
                let p1 = match parser.last_control {
                    Some(lc) => {
                        Point2::new(2.0 * parser.current.x - lc.x, 2.0 * parser.current.y - lc.y)
                    }
                    None => parser.current,
                };

                let (x2, y2) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (x, y) = parse_coordinate_pair(&mut chars)?;

                let p2 = Point2::new(x2, y2);
                let end = Point2::new(x, y);

                let p0_idx = parser.add_vertex(parser.current);
                let p1_idx = parser.add_vertex(p1);
                let p2_idx = parser.add_vertex(p2);
                let p3_idx = parser.add_vertex(end);

                parser
                    .segments
                    .push(Segment2D::CubicBezier(CubicBezier::new(
                        p0_idx, p1_idx, p2_idx, p3_idx,
                    )));
                parser.last_control = Some(p2);
                parser.current = end;
            }
            's' => {
                let p1 = match parser.last_control {
                    Some(lc) => {
                        Point2::new(2.0 * parser.current.x - lc.x, 2.0 * parser.current.y - lc.y)
                    }
                    None => parser.current,
                };

                let (dx2, dy2) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (dx, dy) = parse_coordinate_pair(&mut chars)?;

                let p2 = Point2::new(parser.current.x + dx2, parser.current.y + dy2);
                let end = Point2::new(parser.current.x + dx, parser.current.y + dy);

                let p0_idx = parser.add_vertex(parser.current);
                let p1_idx = parser.add_vertex(p1);
                let p2_idx = parser.add_vertex(p2);
                let p3_idx = parser.add_vertex(end);

                parser
                    .segments
                    .push(Segment2D::CubicBezier(CubicBezier::new(
                        p0_idx, p1_idx, p2_idx, p3_idx,
                    )));
                parser.last_control = Some(p2);
                parser.current = end;
            }
            'Q' => {
                let (x1, y1) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (x, y) = parse_coordinate_pair(&mut chars)?;

                let p1 = Point2::new(x1, y1);
                let end = Point2::new(x, y);

                let p0_idx = parser.add_vertex(parser.current);
                let p1_idx = parser.add_vertex(p1);
                let p2_idx = parser.add_vertex(end);

                parser
                    .segments
                    .push(Segment2D::QuadraticBezier(QuadraticBezier::new(
                        p0_idx, p1_idx, p2_idx,
                    )));
                parser.last_control = Some(p1);
                parser.current = end;
            }
            'q' => {
                let (dx1, dy1) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (dx, dy) = parse_coordinate_pair(&mut chars)?;

                let p1 = Point2::new(parser.current.x + dx1, parser.current.y + dy1);
                let end = Point2::new(parser.current.x + dx, parser.current.y + dy);

                let p0_idx = parser.add_vertex(parser.current);
                let p1_idx = parser.add_vertex(p1);
                let p2_idx = parser.add_vertex(end);

                parser
                    .segments
                    .push(Segment2D::QuadraticBezier(QuadraticBezier::new(
                        p0_idx, p1_idx, p2_idx,
                    )));
                parser.last_control = Some(p1);
                parser.current = end;
            }
            'T' => {
                // Smooth quadratic bezier
                let p1 = match parser.last_control {
                    Some(lc) => {
                        Point2::new(2.0 * parser.current.x - lc.x, 2.0 * parser.current.y - lc.y)
                    }
                    None => parser.current,
                };

                let (x, y) = parse_coordinate_pair(&mut chars)?;
                let end = Point2::new(x, y);

                let p0_idx = parser.add_vertex(parser.current);
                let p1_idx = parser.add_vertex(p1);
                let p2_idx = parser.add_vertex(end);

                parser
                    .segments
                    .push(Segment2D::QuadraticBezier(QuadraticBezier::new(
                        p0_idx, p1_idx, p2_idx,
                    )));
                parser.last_control = Some(p1);
                parser.current = end;
            }
            't' => {
                let p1 = match parser.last_control {
                    Some(lc) => {
                        Point2::new(2.0 * parser.current.x - lc.x, 2.0 * parser.current.y - lc.y)
                    }
                    None => parser.current,
                };

                let (dx, dy) = parse_coordinate_pair(&mut chars)?;
                let end = Point2::new(parser.current.x + dx, parser.current.y + dy);

                let p0_idx = parser.add_vertex(parser.current);
                let p1_idx = parser.add_vertex(p1);
                let p2_idx = parser.add_vertex(end);

                parser
                    .segments
                    .push(Segment2D::QuadraticBezier(QuadraticBezier::new(
                        p0_idx, p1_idx, p2_idx,
                    )));
                parser.last_control = Some(p1);
                parser.current = end;
            }
            'A' | 'a' => {
                let is_relative = c == 'a';

                let rx = parse_number(&mut chars)?;
                skip_separators(&mut chars);
                let ry = parse_number(&mut chars)?;
                skip_separators(&mut chars);
                let _x_rotation = parse_number(&mut chars)?;
                skip_separators(&mut chars);
                let large_arc = parse_number(&mut chars)? != 0.0;
                skip_separators(&mut chars);
                let sweep = parse_number(&mut chars)? != 0.0;
                skip_separators(&mut chars);
                let (ex, ey) = parse_coordinate_pair(&mut chars)?;

                let end = if is_relative {
                    Point2::new(parser.current.x + ex, parser.current.y + ey)
                } else {
                    Point2::new(ex, ey)
                };

                if let Some(arc) = svg_arc_to_center_arc(&mut parser, end, rx, ry, large_arc, sweep)
                {
                    parser.segments.push(arc);
                }

                parser.current = end;
                parser.last_control = None;
            }
            'Z' | 'z' => {
                if (parser.current.x - parser.start.x).abs() > 1e-10
                    || (parser.current.y - parser.start.y).abs() > 1e-10
                {
                    let start_idx = parser.add_vertex(parser.current);
                    let end_idx = parser.add_vertex(parser.start);
                    parser
                        .segments
                        .push(Segment2D::Line(Line::new(start_idx, end_idx)));
                }
                parser.current = parser.start;
                parser.last_control = None;
            }
            // Skip whitespace and unknown commands
            _ => {}
        }
    }

    Ok(parser.into_path())
}

/// Convert SVG arc parameters to Arc segment
fn svg_arc_to_center_arc(
    parser: &mut SvgPathParser,
    p2: Point2<f64>,
    rx: f64,
    ry: f64,
    large_arc: bool,
    sweep: bool,
) -> Option<Segment2D> {
    let p1 = parser.current;

    // For elliptical arcs where rx != ry, approximate as circular using average radius
    if (rx - ry).abs() > 1e-10 {
        let r = f64::midpoint(rx, ry);
        return svg_arc_to_center_arc(parser, p2, r, r, large_arc, sweep);
    }

    let r = rx;

    // Midpoint of chord
    let mid = Point2::new(f64::midpoint(p1.x, p2.x), f64::midpoint(p1.y, p2.y));

    // Distance from midpoint to center
    let d = ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt() / 2.0;

    if d > r {
        // Points too far apart - return line instead
        let start_idx = parser.add_vertex(p1);
        let end_idx = parser.add_vertex(p2);
        return Some(Segment2D::Line(Line::new(start_idx, end_idx)));
    }

    let h = (r * r - d * d).sqrt();

    // Direction perpendicular to chord
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let len = (dx * dx + dy * dy).sqrt();
    let px = -dy / len;
    let py = dx / len;

    // Choose center based on large_arc and sweep flags
    let sign = if large_arc == sweep { -1.0 } else { 1.0 };
    let center = Point2::new(mid.x + sign * h * px, mid.y + sign * h * py);

    // Calculate sweep angle
    let start_angle = (p1.y - center.y).atan2(p1.x - center.x);
    let end_angle = (p2.y - center.y).atan2(p2.x - center.x);
    let mut sweep_angle = end_angle - start_angle;

    // Adjust sweep angle based on sweep direction
    if sweep {
        // Counter-clockwise
        if sweep_angle < 0.0 {
            sweep_angle += 2.0 * std::f64::consts::PI;
        }
    } else {
        // Clockwise
        if sweep_angle > 0.0 {
            sweep_angle -= 2.0 * std::f64::consts::PI;
        }
    }

    let winding = if sweep { Winding::Ccw } else { Winding::Cw };

    let start_idx = parser.add_vertex(p1);
    let end_idx = parser.add_vertex(p2);

    Some(Segment2D::Arc(Arc2::new(
        start_idx,
        end_idx,
        sweep_angle,
        winding,
    )))
}

// =============================================================================
// SVG Export
// =============================================================================

/// Convert a segment to SVG path commands
fn segment_to_svg(
    segment: &Segment2D,
    vertices: &[Point2<f64>],
    current_pos: &mut Option<Point2<f64>>,
) -> Option<String> {
    match segment {
        Segment2D::Line(line) => {
            if line.points.is_empty() {
                return None;
            }

            let start = *vertices.get(*line.points.first()?)?;
            let mut result = String::new();

            // Move to start if needed
            if needs_move(*current_pos, start) {
                let _ = write!(result, "M{},{} ", fmt_num(start.x), fmt_num(start.y));
            }

            // Draw lines to all subsequent points
            for &idx in line.points.iter().skip(1) {
                let p = *vertices.get(idx)?;
                let _ = write!(result, "L{},{} ", fmt_num(p.x), fmt_num(p.y));
            }

            *current_pos = line.points.last().and_then(|&i| vertices.get(i).copied());
            Some(result.trim_end().to_string())
        }

        Segment2D::Arc(arc) => {
            let start = *vertices.get(arc.start)?;
            let finish = *vertices.get(arc.finish)?;
            let mut result = String::new();

            if needs_move(*current_pos, start) {
                let _ = write!(result, "M{},{} ", fmt_num(start.x), fmt_num(start.y));
            }

            let radius = arc.radius(vertices).unwrap_or(0.0);
            let sweep_angle = arc.sweep_angle();
            let large_arc = i32::from(sweep_angle.abs() > std::f64::consts::PI);
            let sweep_flag = i32::from(sweep_angle > 0.0);

            let _ = write!(
                result,
                "A{},{} 0 {} {} {},{}",
                fmt_num(radius),
                fmt_num(radius),
                large_arc,
                sweep_flag,
                fmt_num(finish.x),
                fmt_num(finish.y)
            );
            *current_pos = Some(finish);
            Some(result)
        }

        Segment2D::Circle(circle) => {
            let center = *vertices.get(circle.center)?;
            // Circle as two arcs
            let right = Point2::new(center.x + circle.radius, center.y);
            let left = Point2::new(center.x - circle.radius, center.y);
            let r = circle.radius;

            *current_pos = Some(right);
            Some(format!(
                "M{},{} A{},{} 0 1 1 {},{} A{},{} 0 1 1 {},{}",
                fmt_num(right.x),
                fmt_num(right.y),
                fmt_num(r),
                fmt_num(r),
                fmt_num(left.x),
                fmt_num(left.y),
                fmt_num(r),
                fmt_num(r),
                fmt_num(right.x),
                fmt_num(right.y)
            ))
        }

        Segment2D::Ellipse(ellipse) => {
            let center = *vertices.get(ellipse.center)?;
            // Ellipse as two arcs
            let cos_r = ellipse.rotation.cos();
            let sin_r = ellipse.rotation.sin();
            let right = Point2::new(
                center.x + ellipse.major * cos_r,
                center.y + ellipse.major * sin_r,
            );
            let left = Point2::new(
                center.x - ellipse.major * cos_r,
                center.y - ellipse.major * sin_r,
            );

            *current_pos = Some(right);
            Some(format!(
                "M{},{} A{},{} {} 1 1 {},{} A{},{} {} 1 1 {},{}",
                fmt_num(right.x),
                fmt_num(right.y),
                fmt_num(ellipse.major),
                fmt_num(ellipse.minor),
                fmt_num(ellipse.rotation.to_degrees()),
                fmt_num(left.x),
                fmt_num(left.y),
                fmt_num(ellipse.major),
                fmt_num(ellipse.minor),
                fmt_num(ellipse.rotation.to_degrees()),
                fmt_num(right.x),
                fmt_num(right.y)
            ))
        }

        Segment2D::CubicBezier(bezier) => {
            let p0 = *vertices.get(bezier.p0)?;
            let p1 = *vertices.get(bezier.p1)?;
            let p2 = *vertices.get(bezier.p2)?;
            let p3 = *vertices.get(bezier.p3)?;
            let mut result = String::new();

            if needs_move(*current_pos, p0) {
                let _ = write!(result, "M{},{} ", fmt_num(p0.x), fmt_num(p0.y));
            }

            let _ = write!(
                result,
                "C{},{} {},{} {},{}",
                fmt_num(p1.x),
                fmt_num(p1.y),
                fmt_num(p2.x),
                fmt_num(p2.y),
                fmt_num(p3.x),
                fmt_num(p3.y)
            );
            *current_pos = Some(p3);
            Some(result)
        }

        Segment2D::QuadraticBezier(bezier) => {
            let p0 = *vertices.get(bezier.p0)?;
            let p1 = *vertices.get(bezier.p1)?;
            let p2 = *vertices.get(bezier.p2)?;
            let mut result = String::new();

            if needs_move(*current_pos, p0) {
                let _ = write!(result, "M{},{} ", fmt_num(p0.x), fmt_num(p0.y));
            }

            let _ = write!(
                result,
                "Q{},{} {},{}",
                fmt_num(p1.x),
                fmt_num(p1.y),
                fmt_num(p2.x),
                fmt_num(p2.y)
            );
            *current_pos = Some(p2);
            Some(result)
        }

        Segment2D::BSpline(spline) => {
            // Convert B-spline to polyline for SVG
            if spline.points.is_empty() {
                return None;
            }

            let mut result = String::new();
            let start = *vertices.get(*spline.points.first()?)?;

            if needs_move(*current_pos, start) {
                let _ = write!(result, "M{},{} ", fmt_num(start.x), fmt_num(start.y));
            }

            for &idx in spline.points.iter().skip(1) {
                let point = *vertices.get(idx)?;
                let _ = write!(result, "L{},{} ", fmt_num(point.x), fmt_num(point.y));
            }

            *current_pos = spline.points.last().and_then(|&i| vertices.get(i).copied());
            Some(result)
        }
    }
}

fn needs_move(current: Option<Point2<f64>>, target: Point2<f64>) -> bool {
    current.is_none_or(|p| (p.x - target.x).abs() > 1e-10 || (p.y - target.y).abs() > 1e-10)
}

/// Format a number for SVG output (remove trailing zeros)
fn fmt_num(n: f64) -> String {
    let s = format!("{:.6}", n);
    let s = s.trim_end_matches('0');
    let s = s.trim_end_matches('.');
    s.to_string()
}

// =============================================================================
// Parsing helpers
// =============================================================================

fn parse_coordinate_pair(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<(f64, f64)> {
    let x = parse_number(chars)?;
    skip_separators(chars);
    let y = parse_number(chars)?;
    Ok((x, y))
}

fn parse_number(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<f64> {
    skip_separators(chars);

    let mut num_str = String::new();

    // Handle sign
    if let Some(&c) = chars.peek()
        && (c == '-' || c == '+')
    {
        num_str.push(chars.next().unwrap());
    }

    // Collect digits, decimal point, and exponent
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' {
            num_str.push(chars.next().unwrap());
            // Handle exponent sign
            if (c == 'e' || c == 'E') && chars.peek().is_some_and(|&c| c == '-' || c == '+') {
                num_str.push(chars.next().unwrap());
            }
        } else {
            break;
        }
    }

    num_str
        .parse()
        .map_err(|_| SvgError(format!("Invalid number: '{}'", num_str)))
}

fn skip_separators(chars: &mut std::iter::Peekable<std::str::Chars>) {
    while let Some(&c) = chars.peek() {
        if c == ' ' || c == ',' || c == '\n' || c == '\t' || c == '\r' {
            chars.next();
        } else {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parse_simple_path() {
        let path = Path2D::from_svg("M0,0 L10,0 L10,10 L0,10 Z").unwrap();
        assert_eq!(path.segments.len(), 4);
        // Path forms a closed ring (start and end are the same)
        let first_start = path.segments[0].start(&path.vertices).unwrap();
        let last_finish = path
            .segments
            .last()
            .unwrap()
            .finish(&path.vertices)
            .unwrap();
        assert!((first_start.x - last_finish.x).abs() < 1e-10);
        assert!((first_start.y - last_finish.y).abs() < 1e-10);
    }

    #[test]
    fn test_parse_relative_commands() {
        let path = Path2D::from_svg("M0,0 l10,0 l0,10 l-10,0 z").unwrap();
        assert_eq!(path.segments.len(), 4);

        if let Segment2D::Line(line) = &path.segments[0] {
            let start = path.vertices[line.points[0]];
            let finish = path.vertices[*line.points.last().unwrap()];
            assert_relative_eq!(start.x, 0.0, epsilon = 1e-10);
            assert_relative_eq!(finish.x, 10.0, epsilon = 1e-10);
        } else {
            panic!("Expected Line");
        }
    }

    #[test]
    fn test_parse_hv_commands() {
        let path = Path2D::from_svg("M0,0 H10 V10 H0 V0").unwrap();
        assert_eq!(path.segments.len(), 4);
    }

    #[test]
    fn test_parse_cubic_bezier() {
        let path = Path2D::from_svg("M0,0 C1,2 3,2 4,0").unwrap();
        assert_eq!(path.segments.len(), 1);

        if let Segment2D::CubicBezier(bezier) = &path.segments[0] {
            let p0 = path.vertices[bezier.p0];
            let p3 = path.vertices[bezier.p3];
            assert_relative_eq!(p0.x, 0.0, epsilon = 1e-10);
            assert_relative_eq!(p3.x, 4.0, epsilon = 1e-10);
        } else {
            panic!("Expected CubicBezier");
        }
    }

    #[test]
    fn test_parse_rect_element() {
        let path = Path2D::from_svg("<rect x='0' y='0' width='10' height='5'/>").unwrap();
        assert_eq!(path.segments.len(), 4);
        // Rectangle should form a closed ring
        let first_start = path.segments[0].start(&path.vertices).unwrap();
        let last_finish = path
            .segments
            .last()
            .unwrap()
            .finish(&path.vertices)
            .unwrap();
        assert!((first_start.x - last_finish.x).abs() < 1e-10);
        assert!((first_start.y - last_finish.y).abs() < 1e-10);
    }

    #[test]
    fn test_parse_circle_element() {
        let path = Path2D::from_svg("<circle cx='5' cy='5' r='10'/>").unwrap();
        assert_eq!(path.segments.len(), 1);

        if let Segment2D::Circle(circle) = &path.segments[0] {
            let center = path.vertices[circle.center];
            assert_relative_eq!(center.x, 5.0, epsilon = 1e-10);
            assert_relative_eq!(circle.radius, 10.0, epsilon = 1e-10);
        } else {
            panic!("Expected Circle");
        }
    }

    #[test]
    fn test_svg_roundtrip() {
        let original = Path2D::from_svg("M0,0 L10,0 L10,10 L0,10 Z").unwrap();
        let svg = original.to_svg();
        let parsed = Path2D::from_svg(&svg).unwrap();

        assert_eq!(original.segments.len(), parsed.segments.len());
    }

    #[test]
    fn test_arc_parsing() {
        let path = Path2D::from_svg("M10,0 A10,10 0 1 1 -10,0 A10,10 0 1 1 10,0").unwrap();
        assert_eq!(path.segments.len(), 2);

        // Both should be arcs
        assert!(matches!(path.segments[0], Segment2D::Arc(_)));
        assert!(matches!(path.segments[1], Segment2D::Arc(_)));
    }

    #[test]
    fn test_to_svg_line() {
        let path = Path2D::from_vertices_and_segments(
            vec![Point2::new(0.0, 0.0), Point2::new(10.0, 5.0)],
            vec![Segment2D::Line(Line::new(0, 1))],
        );

        let svg = path.to_svg();
        assert!(svg.contains("M0,0"));
        assert!(svg.contains("L10,5"));
    }
}
