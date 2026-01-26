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

use nalgebra::Point2;

use super::entity::{
    Arc2D, Circle2D, CubicBezier2D, Ellipse2D, Line2D, QuadraticBezier2D, Winding,
};
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
    pub fn to_svg(&self) -> String {
        let mut path = String::new();
        let mut current_pos: Option<Point2<f64>> = None;

        for segment in &self.segments {
            let svg = segment_to_svg(segment, &mut current_pos);
            path.push_str(&svg);
        }

        // Close path if marked as closed
        if self.closed && !path.is_empty() {
            path.push_str(" Z");
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

    /// Calculate the bounding box of this path
    fn bounds(&self) -> Option<(Point2<f64>, Point2<f64>)> {
        use super::entity::Curve;

        let points: Vec<Point2<f64>> = self
            .segments
            .iter()
            .flat_map(|s| match s {
                Segment2D::Line(l) => vec![l.start(), l.finish()],
                Segment2D::Arc(a) => {
                    vec![a.start, a.finish]
                }
                Segment2D::Circle(c) => {
                    vec![
                        Point2::new(c.center.x - c.radius, c.center.y - c.radius),
                        Point2::new(c.center.x + c.radius, c.center.y + c.radius),
                    ]
                }
                Segment2D::Ellipse(e) => {
                    // Conservative bounds using major axis
                    vec![
                        Point2::new(e.center.x - e.major, e.center.y - e.major),
                        Point2::new(e.center.x + e.major, e.center.y + e.major),
                    ]
                }
                Segment2D::CubicBezier(b) => vec![b.p0, b.p1, b.p2, b.p3],
                Segment2D::QuadraticBezier(b) => vec![b.p0, b.p1, b.p2],
                Segment2D::BSpline(s) => s.points.clone(),
            })
            .collect();

        if points.is_empty() {
            return None;
        }

        let min_x = points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let min_y = points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let max_x = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let max_y = points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

        Some((Point2::new(min_x, min_y), Point2::new(max_x, max_y)))
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
        parse_line_element(svg)
    } else if svg.starts_with("<polyline") {
        parse_polyline_element(svg, false)
    } else if svg.starts_with("<polygon") {
        parse_polyline_element(svg, true)
    } else if svg.starts_with("<svg") {
        // Full SVG document - extract path elements
        parse_svg_document(svg)
    } else {
        Err(SvgError(format!("Unknown SVG element: {}", &svg[..svg.len().min(20)])))
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
        let segments = vec![
            Segment2D::Line(Line2D::new(Point2::new(x, y), Point2::new(x + width, y))),
            Segment2D::Line(Line2D::new(Point2::new(x + width, y), Point2::new(x + width, y + height))),
            Segment2D::Line(Line2D::new(Point2::new(x + width, y + height), Point2::new(x, y + height))),
            Segment2D::Line(Line2D::new(Point2::new(x, y + height), Point2::new(x, y))),
        ];
        Ok(Path2D {
            segments,
            closed: true,
        })
    }
}

fn rounded_rect_to_path(x: f64, y: f64, w: f64, h: f64, rx: f64, ry: f64) -> String {
    format!(
        "M{},{} h{} a{},{} 0 0 1 {},{} v{} a{},{} 0 0 1 -{},{} h-{} a{},{} 0 0 1 -{}-{} v-{} a{},{} 0 0 1 {}-{} Z",
        x + rx, y,
        w - 2.0 * rx,
        rx, ry, rx, ry,
        h - 2.0 * ry,
        rx, ry, rx, ry,
        w - 2.0 * rx,
        rx, ry, rx, ry,
        h - 2.0 * ry,
        rx, ry, rx, ry
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

    Ok(Path2D {
        segments: vec![Segment2D::Circle(Circle2D::new(Point2::new(cx, cy), r))],
        closed: true,
    })
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

    Ok(Path2D {
        segments: vec![Segment2D::Ellipse(Ellipse2D::axis_aligned(
            Point2::new(cx, cy),
            rx.max(ry),
            rx.min(ry),
        ))],
        closed: true,
    })
}

fn parse_line_element(svg: &str) -> Result<Path2D> {
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

    Ok(Path2D {
        segments: vec![Segment2D::Line(Line2D::new(
            Point2::new(x1, y1),
            Point2::new(x2, y2),
        ))],
        closed: false,
    })
}

fn parse_polyline_element(svg: &str, closed: bool) -> Result<Path2D> {
    let points_str = extract_attribute(svg, "points")
        .ok_or_else(|| SvgError("Missing 'points' attribute".into()))?;

    let points = parse_points_list(&points_str)?;
    if points.len() < 2 {
        return Err(SvgError("Polyline needs at least 2 points".into()));
    }

    let mut segments: Vec<Segment2D> = points
        .windows(2)
        .map(|w| Segment2D::Line(Line2D::new(w[0], w[1])))
        .collect();

    if closed && points.len() > 2 {
        segments.push(Segment2D::Line(Line2D::new(
            *points.last().unwrap(),
            points[0],
        )));
    }

    Ok(Path2D { segments, closed })
}

fn parse_svg_document(svg: &str) -> Result<Path2D> {
    // Simple extraction of first path element from SVG document
    // A full implementation would parse all elements
    if let Some(start) = svg.find("<path") {
        let end = svg[start..].find("/>").or_else(|| svg[start..].find("</path>"));
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

/// Parse an SVG path d-string into a Path2D
fn parse_svg_path(path: &str) -> Result<Path2D> {
    let mut segments = Vec::new();
    let mut chars = path.chars().peekable();
    let mut current = Point2::new(0.0, 0.0);
    let mut start = Point2::new(0.0, 0.0);
    let mut last_control: Option<Point2<f64>> = None;
    let mut closed = false;

    while let Some(c) = chars.next() {
        match c {
            'M' => {
                let (x, y) = parse_coordinate_pair(&mut chars)?;
                current = Point2::new(x, y);
                start = current;
                last_control = None;

                // M can be followed by implicit L commands
                skip_separators(&mut chars);
                while chars.peek().is_some_and(|c| c.is_ascii_digit() || *c == '-' || *c == '+' || *c == '.') {
                    let (x, y) = parse_coordinate_pair(&mut chars)?;
                    let end = Point2::new(x, y);
                    segments.push(Segment2D::Line(Line2D::new(current, end)));
                    current = end;
                    skip_separators(&mut chars);
                }
            }
            'm' => {
                let (dx, dy) = parse_coordinate_pair(&mut chars)?;
                current = Point2::new(current.x + dx, current.y + dy);
                start = current;
                last_control = None;

                // m can be followed by implicit l commands
                skip_separators(&mut chars);
                while chars.peek().is_some_and(|c| c.is_ascii_digit() || *c == '-' || *c == '+' || *c == '.') {
                    let (dx, dy) = parse_coordinate_pair(&mut chars)?;
                    let end = Point2::new(current.x + dx, current.y + dy);
                    segments.push(Segment2D::Line(Line2D::new(current, end)));
                    current = end;
                    skip_separators(&mut chars);
                }
            }
            'L' => {
                loop {
                    let (x, y) = parse_coordinate_pair(&mut chars)?;
                    let end = Point2::new(x, y);
                    segments.push(Segment2D::Line(Line2D::new(current, end)));
                    current = end;
                    last_control = None;

                    skip_separators(&mut chars);
                    if !chars.peek().is_some_and(|c| c.is_ascii_digit() || *c == '-' || *c == '+' || *c == '.') {
                        break;
                    }
                }
            }
            'l' => {
                loop {
                    let (dx, dy) = parse_coordinate_pair(&mut chars)?;
                    let end = Point2::new(current.x + dx, current.y + dy);
                    segments.push(Segment2D::Line(Line2D::new(current, end)));
                    current = end;
                    last_control = None;

                    skip_separators(&mut chars);
                    if !chars.peek().is_some_and(|c| c.is_ascii_digit() || *c == '-' || *c == '+' || *c == '.') {
                        break;
                    }
                }
            }
            'H' => {
                let x = parse_number(&mut chars)?;
                let end = Point2::new(x, current.y);
                segments.push(Segment2D::Line(Line2D::new(current, end)));
                current = end;
                last_control = None;
            }
            'h' => {
                let dx = parse_number(&mut chars)?;
                let end = Point2::new(current.x + dx, current.y);
                segments.push(Segment2D::Line(Line2D::new(current, end)));
                current = end;
                last_control = None;
            }
            'V' => {
                let y = parse_number(&mut chars)?;
                let end = Point2::new(current.x, y);
                segments.push(Segment2D::Line(Line2D::new(current, end)));
                current = end;
                last_control = None;
            }
            'v' => {
                let dy = parse_number(&mut chars)?;
                let end = Point2::new(current.x, current.y + dy);
                segments.push(Segment2D::Line(Line2D::new(current, end)));
                current = end;
                last_control = None;
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

                segments.push(Segment2D::CubicBezier(CubicBezier2D::new(current, p1, p2, end)));
                last_control = Some(p2);
                current = end;
            }
            'c' => {
                let (dx1, dy1) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (dx2, dy2) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (dx, dy) = parse_coordinate_pair(&mut chars)?;

                let p1 = Point2::new(current.x + dx1, current.y + dy1);
                let p2 = Point2::new(current.x + dx2, current.y + dy2);
                let end = Point2::new(current.x + dx, current.y + dy);

                segments.push(Segment2D::CubicBezier(CubicBezier2D::new(current, p1, p2, end)));
                last_control = Some(p2);
                current = end;
            }
            'S' => {
                // Smooth cubic bezier - reflects previous control point
                let p1 = match last_control {
                    Some(lc) => Point2::new(2.0 * current.x - lc.x, 2.0 * current.y - lc.y),
                    None => current,
                };

                let (x2, y2) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (x, y) = parse_coordinate_pair(&mut chars)?;

                let p2 = Point2::new(x2, y2);
                let end = Point2::new(x, y);

                segments.push(Segment2D::CubicBezier(CubicBezier2D::new(current, p1, p2, end)));
                last_control = Some(p2);
                current = end;
            }
            's' => {
                let p1 = match last_control {
                    Some(lc) => Point2::new(2.0 * current.x - lc.x, 2.0 * current.y - lc.y),
                    None => current,
                };

                let (dx2, dy2) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (dx, dy) = parse_coordinate_pair(&mut chars)?;

                let p2 = Point2::new(current.x + dx2, current.y + dy2);
                let end = Point2::new(current.x + dx, current.y + dy);

                segments.push(Segment2D::CubicBezier(CubicBezier2D::new(current, p1, p2, end)));
                last_control = Some(p2);
                current = end;
            }
            'Q' => {
                let (x1, y1) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (x, y) = parse_coordinate_pair(&mut chars)?;

                let p1 = Point2::new(x1, y1);
                let end = Point2::new(x, y);

                segments.push(Segment2D::QuadraticBezier(QuadraticBezier2D::new(current, p1, end)));
                last_control = Some(p1);
                current = end;
            }
            'q' => {
                let (dx1, dy1) = parse_coordinate_pair(&mut chars)?;
                skip_separators(&mut chars);
                let (dx, dy) = parse_coordinate_pair(&mut chars)?;

                let p1 = Point2::new(current.x + dx1, current.y + dy1);
                let end = Point2::new(current.x + dx, current.y + dy);

                segments.push(Segment2D::QuadraticBezier(QuadraticBezier2D::new(current, p1, end)));
                last_control = Some(p1);
                current = end;
            }
            'T' => {
                // Smooth quadratic bezier
                let p1 = match last_control {
                    Some(lc) => Point2::new(2.0 * current.x - lc.x, 2.0 * current.y - lc.y),
                    None => current,
                };

                let (x, y) = parse_coordinate_pair(&mut chars)?;
                let end = Point2::new(x, y);

                segments.push(Segment2D::QuadraticBezier(QuadraticBezier2D::new(current, p1, end)));
                last_control = Some(p1);
                current = end;
            }
            't' => {
                let p1 = match last_control {
                    Some(lc) => Point2::new(2.0 * current.x - lc.x, 2.0 * current.y - lc.y),
                    None => current,
                };

                let (dx, dy) = parse_coordinate_pair(&mut chars)?;
                let end = Point2::new(current.x + dx, current.y + dy);

                segments.push(Segment2D::QuadraticBezier(QuadraticBezier2D::new(current, p1, end)));
                last_control = Some(p1);
                current = end;
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
                    Point2::new(current.x + ex, current.y + ey)
                } else {
                    Point2::new(ex, ey)
                };

                if let Some(arc) = svg_arc_to_center_arc(current, end, rx, ry, large_arc, sweep) {
                    segments.push(arc);
                }

                current = end;
                last_control = None;
            }
            'Z' | 'z' => {
                if (current.x - start.x).abs() > 1e-10 || (current.y - start.y).abs() > 1e-10 {
                    segments.push(Segment2D::Line(Line2D::new(current, start)));
                }
                current = start;
                last_control = None;
                closed = true;
            }
            ' ' | ',' | '\n' | '\t' | '\r' => {
                // Skip whitespace
            }
            _ => {
                // Unknown command - skip
            }
        }
    }

    Ok(Path2D { segments, closed })
}

/// Convert SVG arc parameters to Arc2D segment
fn svg_arc_to_center_arc(
    p1: Point2<f64>,
    p2: Point2<f64>,
    rx: f64,
    ry: f64,
    large_arc: bool,
    sweep: bool,
) -> Option<Segment2D> {
    // For elliptical arcs where rx != ry, approximate as circular using average radius
    if (rx - ry).abs() > 1e-10 {
        let r = (rx + ry) / 2.0;
        return svg_arc_to_center_arc(p1, p2, r, r, large_arc, sweep);
    }

    let r = rx;

    // Midpoint of chord
    let mid = Point2::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);

    // Distance from midpoint to center
    let d = ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt() / 2.0;

    if d > r {
        // Points too far apart - return line instead
        return Some(Segment2D::Line(Line2D::new(p1, p2)));
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

    let winding = if sweep { Winding::Ccw } else { Winding::Cw };

    Some(Segment2D::Arc(Arc2D {
        start: p1,
        finish: p2,
        center: Some(center),
        winding,
    }))
}

// =============================================================================
// SVG Export
// =============================================================================

/// Convert a segment to SVG path commands
fn segment_to_svg(segment: &Segment2D, current_pos: &mut Option<Point2<f64>>) -> String {
    match segment {
        Segment2D::Line(line) => {
            let mut result = String::new();

            // Move to start if needed
            if needs_move(*current_pos, line.start) {
                result.push_str(&format!("M{},{} ", fmt_num(line.start.x), fmt_num(line.start.y)));
            }

            result.push_str(&format!("L{},{}", fmt_num(line.finish.x), fmt_num(line.finish.y)));
            *current_pos = Some(line.finish);
            result
        }

        Segment2D::Arc(arc) => {
            let mut result = String::new();

            if needs_move(*current_pos, arc.start) {
                result.push_str(&format!("M{},{} ", fmt_num(arc.start.x), fmt_num(arc.start.y)));
            }

            let radius = arc.radius();
            let sweep_angle = arc.angle();
            let large_arc = if sweep_angle.abs() > std::f64::consts::PI { 1 } else { 0 };
            let sweep_flag = if sweep_angle > 0.0 { 1 } else { 0 };

            result.push_str(&format!(
                "A{},{} 0 {} {} {},{}",
                fmt_num(radius),
                fmt_num(radius),
                large_arc,
                sweep_flag,
                fmt_num(arc.finish.x),
                fmt_num(arc.finish.y)
            ));
            *current_pos = Some(arc.finish);
            result
        }

        Segment2D::Circle(circle) => {
            // Circle as two arcs
            let right = Point2::new(circle.center.x + circle.radius, circle.center.y);
            let left = Point2::new(circle.center.x - circle.radius, circle.center.y);
            let r = circle.radius;

            *current_pos = Some(right);
            format!(
                "M{},{} A{},{} 0 1 1 {},{} A{},{} 0 1 1 {},{}",
                fmt_num(right.x), fmt_num(right.y),
                fmt_num(r), fmt_num(r),
                fmt_num(left.x), fmt_num(left.y),
                fmt_num(r), fmt_num(r),
                fmt_num(right.x), fmt_num(right.y)
            )
        }

        Segment2D::Ellipse(ellipse) => {
            // Ellipse as two arcs
            let cos_r = ellipse.rotation.cos();
            let sin_r = ellipse.rotation.sin();
            let right = Point2::new(
                ellipse.center.x + ellipse.major * cos_r,
                ellipse.center.y + ellipse.major * sin_r,
            );
            let left = Point2::new(
                ellipse.center.x - ellipse.major * cos_r,
                ellipse.center.y - ellipse.major * sin_r,
            );

            *current_pos = Some(right);
            format!(
                "M{},{} A{},{} {} 1 1 {},{} A{},{} {} 1 1 {},{}",
                fmt_num(right.x), fmt_num(right.y),
                fmt_num(ellipse.major), fmt_num(ellipse.minor),
                fmt_num(ellipse.rotation.to_degrees()),
                fmt_num(left.x), fmt_num(left.y),
                fmt_num(ellipse.major), fmt_num(ellipse.minor),
                fmt_num(ellipse.rotation.to_degrees()),
                fmt_num(right.x), fmt_num(right.y)
            )
        }

        Segment2D::CubicBezier(bezier) => {
            let mut result = String::new();

            if needs_move(*current_pos, bezier.p0) {
                result.push_str(&format!("M{},{} ", fmt_num(bezier.p0.x), fmt_num(bezier.p0.y)));
            }

            result.push_str(&format!(
                "C{},{} {},{} {},{}",
                fmt_num(bezier.p1.x), fmt_num(bezier.p1.y),
                fmt_num(bezier.p2.x), fmt_num(bezier.p2.y),
                fmt_num(bezier.p3.x), fmt_num(bezier.p3.y)
            ));
            *current_pos = Some(bezier.p3);
            result
        }

        Segment2D::QuadraticBezier(bezier) => {
            let mut result = String::new();

            if needs_move(*current_pos, bezier.p0) {
                result.push_str(&format!("M{},{} ", fmt_num(bezier.p0.x), fmt_num(bezier.p0.y)));
            }

            result.push_str(&format!(
                "Q{},{} {},{}",
                fmt_num(bezier.p1.x), fmt_num(bezier.p1.y),
                fmt_num(bezier.p2.x), fmt_num(bezier.p2.y)
            ));
            *current_pos = Some(bezier.p2);
            result
        }

        Segment2D::BSpline(spline) => {
            // Convert B-spline to cubic Bezier approximation for SVG
            // This is a simplification - a proper implementation would use the actual knot vector
            if spline.points.len() < 2 {
                return String::new();
            }

            let mut result = String::new();
            let start = spline.points[0];

            if needs_move(*current_pos, start) {
                result.push_str(&format!("M{},{} ", fmt_num(start.x), fmt_num(start.y)));
            }

            // Simple polyline approximation
            for point in spline.points.iter().skip(1) {
                result.push_str(&format!("L{},{} ", fmt_num(point.x), fmt_num(point.y)));
            }

            *current_pos = spline.points.last().copied();
            result
        }
    }
}

fn needs_move(current: Option<Point2<f64>>, target: Point2<f64>) -> bool {
    current.map_or(true, |p| {
        (p.x - target.x).abs() > 1e-10 || (p.y - target.y).abs() > 1e-10
    })
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
    if let Some(&c) = chars.peek() {
        if c == '-' || c == '+' {
            num_str.push(chars.next().unwrap());
        }
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
        assert!(path.closed);
    }

    #[test]
    fn test_parse_relative_commands() {
        let path = Path2D::from_svg("M0,0 l10,0 l0,10 l-10,0 z").unwrap();
        assert_eq!(path.segments.len(), 4);

        if let Segment2D::Line(line) = &path.segments[0] {
            assert_relative_eq!(line.start.x, 0.0, epsilon = 1e-10);
            assert_relative_eq!(line.finish.x, 10.0, epsilon = 1e-10);
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
            assert_relative_eq!(bezier.p0.x, 0.0, epsilon = 1e-10);
            assert_relative_eq!(bezier.p3.x, 4.0, epsilon = 1e-10);
        } else {
            panic!("Expected CubicBezier");
        }
    }

    #[test]
    fn test_parse_rect_element() {
        let path = Path2D::from_svg("<rect x='0' y='0' width='10' height='5'/>").unwrap();
        assert_eq!(path.segments.len(), 4);
        assert!(path.closed);
    }

    #[test]
    fn test_parse_circle_element() {
        let path = Path2D::from_svg("<circle cx='5' cy='5' r='10'/>").unwrap();
        assert_eq!(path.segments.len(), 1);

        if let Segment2D::Circle(circle) = &path.segments[0] {
            assert_relative_eq!(circle.center.x, 5.0, epsilon = 1e-10);
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
        let path = Path2D {
            segments: vec![Segment2D::Line(Line2D::new(
                Point2::new(0.0, 0.0),
                Point2::new(10.0, 5.0),
            ))],
            closed: false,
        };

        let svg = path.to_svg();
        assert!(svg.contains("M0,0"));
        assert!(svg.contains("L10,5"));
    }
}
