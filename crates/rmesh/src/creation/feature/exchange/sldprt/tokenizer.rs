//! Tokenizer for SLDPRT ResolvedFeatures binary payload
//!
//! This module extracts structured segments from the raw binary data in the
//! ResolvedFeatures stream of SLDPRT files.
//!
//! # Architecture
//!
//! The binary format is a sequence of **segments**, each starting with a type marker:
//!
//! ```text
//! [Type Marker][payload bytes...][Type Marker][payload bytes...]...
//! ```
//!
//! Each segment contains:
//! - A **type name** (ASCII string identifying the segment kind)
//! - Optional **name** (UTF-16 string like "Sketch1")
//! - **Payload** (raw bytes containing coords, dimensions, etc.)
//!
//! # Binary Format
//!
//! ## Type Markers (`FF FF 01 00`) - Segment delimiter
//!
//! ```text
//! FF FF 01 00 <len:u16 LE> <ASCII bytes...>
//! ```
//!
//! Common types: `sgLineHandle`, `sgArcHandle`, `sgCircleDim`, `moExtrusion_c`, etc.
//!
//! ## Name Markers (`FF FE FF`) - Within segment payload
//!
//! ```text
//! FF FE FF <len:u8> <UTF-16LE bytes...>
//! ```
//!
//! ## Coordinate Markers (`1E 00`) - Within segment payload
//!
//! ```text
//! 1E 00 <x:f64 LE> <y:f64 LE>
//! ```

/// Marker bytes for type strings (segment delimiter)
const TYPE_MARKER: [u8; 4] = [0xff, 0xff, 0x01, 0x00];
/// Marker bytes for UTF-16 name strings
const NAME_MARKER: [u8; 3] = [0xff, 0xfe, 0xff];
/// Marker bytes for 2D coordinates
const COORD_MARKER: [u8; 2] = [0x1e, 0x00];

/// A segment from the ResolvedFeatures payload
#[derive(Debug, Clone)]
pub struct Segment {
    /// Byte offset where this segment starts
    pub offset: usize,
    /// Type name (e.g., "sgLineHandle", "moExtrusion_c")
    pub type_name: String,
    /// Optional entity name (e.g., "Sketch1")
    pub name: Option<String>,
    /// Raw payload bytes (everything until next segment)
    pub payload: Vec<u8>,
}

impl Segment {
    /// Extract all 2D coordinates from this segment's payload
    pub fn coords(&self) -> Vec<(f64, f64)> {
        let mut coords = Vec::new();
        let mut i = 0;
        while i + 18 <= self.payload.len() {
            if self.payload[i..i + 2] == COORD_MARKER {
                if let (Ok(x_bytes), Ok(y_bytes)) = (
                    self.payload[i + 2..i + 10].try_into(),
                    self.payload[i + 10..i + 18].try_into(),
                ) {
                    let x = f64::from_le_bytes(x_bytes);
                    let y = f64::from_le_bytes(y_bytes);
                    if x.is_finite() && y.is_finite() {
                        coords.push((x, y));
                    }
                }
                i += 18;
            } else {
                i += 1;
            }
        }
        coords
    }

    /// Extract all f64 values from payload (for finding dimensions)
    pub fn doubles(&self) -> Vec<f64> {
        let mut vals = Vec::new();
        for i in 0..self.payload.len().saturating_sub(7) {
            if let Ok(bytes) = self.payload[i..i + 8].try_into() {
                let val: f64 = f64::from_le_bytes(bytes);
                if val.is_finite() && val.abs() > 1e-10 && val.abs() < 1e6 {
                    vals.push(val);
                }
            }
        }
        vals
    }

    /// Extract a dimension value from this segment's payload
    ///
    /// This is more targeted than `doubles()` - it looks for values after
    /// coordinate markers first (where dimensions are often stored), then
    /// falls back to searching for reasonable dimension values.
    pub fn dimension(&self) -> Option<f64> {
        // First try: look for f64 values after coordinate markers
        // Dimensions often follow the coord data in sgCircleDim segments
        let mut i = 0;
        while i + 18 <= self.payload.len() {
            if self.payload[i..i + 2] == COORD_MARKER {
                // Skip past the coord marker and its two f64 values
                let after_coord = i + 18;
                // Check if there's another f64 right after
                if after_coord + 8 <= self.payload.len() {
                    if let Ok(bytes) = self.payload[after_coord..after_coord + 8].try_into() {
                        let val: f64 = f64::from_le_bytes(bytes);
                        // Reasonable dimension: 0.1mm to 10m
                        if val.is_finite() && val > 0.0001 && val < 10.0 {
                            return Some(val);
                        }
                    }
                }
                i += 18;
            } else {
                i += 1;
            }
        }

        // Fallback: find first reasonable dimension value in payload
        for val in self.doubles() {
            // Reasonable dimension in meters (0.1mm to 1m for typical features)
            if val > 0.0001 && val < 1.0 {
                return Some(val);
            }
        }
        None
    }
}

/// Split binary data into segments at type markers
pub fn tokenize(data: &[u8]) -> Vec<Segment> {
    let mut segments = Vec::new();

    // Find all type marker positions
    let mut markers: Vec<usize> = Vec::new();
    for i in 0..data.len().saturating_sub(6) {
        if data[i..i + 4] == TYPE_MARKER {
            markers.push(i);
        }
    }

    // Build segments from markers
    for (idx, &start) in markers.iter().enumerate() {
        let end = markers.get(idx + 1).copied().unwrap_or(data.len());

        // Parse type name
        if start + 6 > data.len() {
            continue;
        }
        let len = u16::from_le_bytes([data[start + 4], data[start + 5]]) as usize;
        if len == 0 || len >= 100 || start + 6 + len > data.len() {
            continue;
        }
        let type_bytes = &data[start + 6..start + 6 + len];
        if !type_bytes.iter().all(|&b| b.is_ascii_graphic() || b == b' ') {
            continue;
        }
        let type_name = match std::str::from_utf8(type_bytes) {
            Ok(s) => s.to_string(),
            Err(_) => continue,
        };

        // Payload starts after type string
        let payload_start = start + 6 + len;
        let payload = data[payload_start..end].to_vec();

        // Look for name in payload
        let name = find_name_in_payload(&payload);

        segments.push(Segment {
            offset: start,
            type_name,
            name,
            payload,
        });
    }

    segments
}

/// Find a UTF-16 name marker in payload
fn find_name_in_payload(payload: &[u8]) -> Option<String> {
    for i in 0..payload.len().saturating_sub(4) {
        if payload[i..i + 3] == NAME_MARKER {
            let len = payload[i + 3] as usize;
            if len > 0 && len < 100 && i + 4 + len * 2 <= payload.len() {
                let utf16_bytes = &payload[i + 4..i + 4 + len * 2];
                let utf16: Vec<u16> = utf16_bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                if let Ok(s) = String::from_utf16(&utf16) {
                    return Some(s);
                }
            }
        }
    }
    None
}

/// Entity extractor context passed to handlers
struct ExtractContext<'a> {
    /// All segments in the stream
    segments: &'a [Segment],
    /// Index of current segment
    index: usize,
}

impl<'a> ExtractContext<'a> {
    /// Get segments following the current one
    fn following(&self) -> &'a [Segment] {
        &self.segments[self.index..]
    }

    /// Find a nearby segment by type (within N segments forward)
    fn find_following(&self, type_name: &str, max_distance: usize) -> Option<&'a Segment> {
        self.following()
            .iter()
            .take(max_distance)
            .find(|s| s.type_name == type_name)
    }

    /// Find sketch name from nearby segments (searching backward)
    fn find_sketch_name(&self) -> Option<String> {
        for seg in self.segments[..self.index].iter().rev().take(20) {
            if let Some(ref name) = seg.name {
                if name.starts_with("Sketch") {
                    return Some(name.clone());
                }
            }
        }
        None
    }
}

/// Segment type handlers for entity extraction
/// Each handler returns Some(entity) if it can extract from this segment type
mod handlers {
    use super::*;

    /// Extract circle from sgArcHandle + adjacent sgCircleDim
    pub fn arc_handle(seg: &Segment, ctx: &ExtractContext) -> Option<SketchEntity> {
        // Get center from arc segment coords
        let coords = seg.coords();
        let (cx, cy) = coords.first().copied().unwrap_or((0.0, 0.0));

        // Get diameter from adjacent sgCircleDim (should be within 5 segments)
        let diameter = ctx
            .find_following("sgCircleDim", 5)?
            .dimension()?;

        Some(SketchEntity::Circle {
            center_x: cx,
            center_y: cy,
            diameter,
        })
    }

    // Future handlers can be added here:
    // pub fn line_handle(seg: &Segment, ctx: &ExtractContext) -> Option<SketchEntity> { ... }
    // pub fn spline_handle(seg: &Segment, ctx: &ExtractContext) -> Option<SketchEntity> { ... }
    // pub fn ellipse_handle(seg: &Segment, ctx: &ExtractContext) -> Option<SketchEntity> { ... }
}

/// Try to extract an entity from a segment using the appropriate handler
fn extract_entity(seg: &Segment, ctx: &ExtractContext, warnings: &mut Vec<String>) -> Option<SketchEntity> {
    match seg.type_name.as_str() {
        "sgArcHandle" => handlers::arc_handle(seg, ctx),

        // Segment types that look like geometry but we don't handle yet
        "sgSplineHandle" | "sgEllipseHandle" | "sgParabolaHandle" => {
            warnings.push(format!(
                "Unhandled sketch entity type: {} at offset {}",
                seg.type_name, seg.offset
            ));
            None
        }

        // sgLineHandle is handled via polygon extraction from moProfileFeature_c
        // Other segment types are metadata/dimensions, not geometry
        _ => None,
    }
}

/// Result of sketch extraction
pub struct SketchExtraction {
    /// Extracted sketches
    pub sketches: Vec<SketchData>,
    /// Warnings about unhandled geometry
    pub warnings: Vec<String>,
}

/// Extract sketch geometry from segments
pub fn extract_sketches(segments: &[Segment]) -> SketchExtraction {
    let mut sketches = Vec::new();
    let mut warnings = Vec::new();
    let mut current_sketch_name: Option<String> = None;

    for (i, seg) in segments.iter().enumerate() {
        let ctx = ExtractContext { segments, index: i };

        // Track current sketch name from moProfileFeature_c
        if seg.type_name == "moProfileFeature_c" {
            if let Some(ref name) = seg.name {
                if name.starts_with("Sketch") {
                    current_sketch_name = Some(name.clone());
                }
            }

            // Try to extract polygon (rectangle) from profile feature
            if let Some(sketch) = extract_polygon_from_profile(seg, &segments[i..]) {
                sketches.push(sketch);
            }
        }

        // Try entity extraction via handlers
        if let Some(entity) = extract_entity(seg, &ctx, &mut warnings) {
            // Find sketch name for this entity
            let sketch_name = ctx
                .find_sketch_name()
                .or_else(|| current_sketch_name.clone())
                .unwrap_or_else(|| format!("Sketch@{}", seg.offset));

            // Skip if already have this sketch
            if sketches.iter().any(|s| s.name == sketch_name) {
                continue;
            }

            sketches.push(SketchData {
                name: sketch_name,
                entities: vec![entity],
            });
        }
    }

    SketchExtraction { sketches, warnings }
}

/// Extract polygon sketch from a moProfileFeature_c segment and following segments
///
/// This handles closed polygons (like rectangles) where coordinates are collected
/// from the profile feature and its following segments until a feature boundary.
fn extract_polygon_from_profile(seg: &Segment, following: &[Segment]) -> Option<SketchData> {
    let name = seg.name.as_ref()?.clone();
    if !name.starts_with("Sketch") {
        return None;
    }

    // Collect coords from this segment and following until next feature boundary
    let mut all_coords: Vec<(f64, f64)> = seg.coords();

    for next in following.iter().skip(1) {
        if is_feature_boundary(&next.type_name) {
            break;
        }
        all_coords.extend(next.coords());
    }

    // Filter to valid sketch coords and find corners
    let corners: Vec<(f64, f64)> = all_coords
        .iter()
        .filter(|(x, y)| x.abs() > 0.001 && y.abs() > 0.001 && x.abs() < 1.0 && y.abs() < 1.0)
        .copied()
        .collect();

    if corners.len() < 4 {
        return None;
    }

    // Compute bounding box
    let (min_x, max_x) = corners
        .iter()
        .map(|(x, _)| *x)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), x| {
            (min.min(x), max.max(x))
        });
    let (min_y, max_y) = corners
        .iter()
        .map(|(_, y)| *y)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), y| {
            (min.min(y), max.max(y))
        });

    // Create CCW rectangle
    let rect_corners = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ];

    let entities: Vec<SketchEntity> = (0..4)
        .map(|j| {
            let (x1, y1) = rect_corners[j];
            let (x2, y2) = rect_corners[(j + 1) % 4];
            SketchEntity::Line {
                start_x: x1,
                start_y: y1,
                end_x: x2,
                end_y: y2,
            }
        })
        .collect();

    Some(SketchData { name, entities })
}

/// Check if a type name marks a feature boundary
fn is_feature_boundary(ty: &str) -> bool {
    ty.starts_with("mo")
        && (ty.contains("Extrusion")
            || ty.contains("Region")
            || ty.contains("Chain")
            || ty == "moICE_c")
}

/// Extracted sketch data containing geometry entities
#[derive(Debug, Clone)]
pub struct SketchData {
    /// Sketch name (e.g., "Sketch1", "Sketch2")
    pub name: String,
    /// Geometry entities in this sketch
    pub entities: Vec<SketchEntity>,
}

/// A sketch geometry entity extracted from the SLDPRT file
///
/// # Adding New Geometry Types
///
/// To support additional geometry (e.g., arcs, splines):
///
/// 1. Add a new variant here with the required parameters
/// 2. Add extraction logic in [`extract_sketches`] to detect and parse
/// 3. Add conversion in `parser.rs` `build_sketch_from_geometry()`
#[derive(Debug, Clone)]
pub enum SketchEntity {
    /// Line segment from start to end point
    ///
    /// Extracted from `sgLineHandle` type markers with two `1E 00` coords.
    Line {
        start_x: f64,
        start_y: f64,
        end_x: f64,
        end_y: f64,
    },

    /// Full circle defined by center and diameter
    ///
    /// Extracted from `sgArcHandle` + `sgCircleDim` combination.
    /// Note: Partial arcs also use `sgArcHandle` but need different handling.
    Circle {
        center_x: f64,
        center_y: f64,
        /// Diameter in meters (not radius!)
        diameter: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_single_segment() {
        // FF FF 01 00 0C 00 "sgLineHandle"
        let data = [
            0xff, 0xff, 0x01, 0x00, 0x0c, 0x00, b's', b'g', b'L', b'i', b'n', b'e', b'H', b'a',
            b'n', b'd', b'l', b'e',
        ];
        let segments = tokenize(&data);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].type_name, "sgLineHandle");
        assert!(segments[0].payload.is_empty());
    }

    #[test]
    fn test_tokenize_segment_with_name() {
        // Type marker + payload containing a name marker
        let mut data = vec![
            0xff, 0xff, 0x01, 0x00, 0x04, 0x00, b't', b'e', b's', b't', // type "test"
        ];
        // Add name marker in payload: FF FE FF 07 "Sketch1" in UTF-16LE
        data.extend_from_slice(&[0xff, 0xfe, 0xff, 0x07]);
        data.extend_from_slice(&[b'S', 0x00, b'k', 0x00, b'e', 0x00, b't', 0x00]);
        data.extend_from_slice(&[b'c', 0x00, b'h', 0x00, b'1', 0x00]);

        let segments = tokenize(&data);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].type_name, "test");
        assert_eq!(segments[0].name, Some("Sketch1".to_string()));
    }

    #[test]
    fn test_tokenize_segment_with_coords() {
        // Type marker + coord in payload
        let mut data = vec![
            0xff, 0xff, 0x01, 0x00, 0x04, 0x00, b't', b'e', b's', b't',
        ];
        // Add coord marker: 1E 00 <x:f64> <y:f64>
        data.extend_from_slice(&[0x1e, 0x00]);
        data.extend_from_slice(&(-0.0127f64).to_le_bytes()); // -0.5"
        data.extend_from_slice(&(0.0254f64).to_le_bytes()); // 1.0"

        let segments = tokenize(&data);
        assert_eq!(segments.len(), 1);

        let coords = segments[0].coords();
        assert_eq!(coords.len(), 1);
        assert!((coords[0].0 / 0.0254 + 0.5).abs() < 0.001);
        assert!((coords[0].1 / 0.0254 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_tokenize_multiple_segments() {
        // Two consecutive type markers
        let mut data = vec![
            0xff, 0xff, 0x01, 0x00, 0x05, 0x00, b'f', b'i', b'r', b's', b't',
        ];
        data.extend_from_slice(&[0x01, 0x02, 0x03]); // some payload
        data.extend_from_slice(&[
            0xff, 0xff, 0x01, 0x00, 0x06, 0x00, b's', b'e', b'c', b'o', b'n', b'd',
        ]);

        let segments = tokenize(&data);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].type_name, "first");
        assert_eq!(segments[0].payload, vec![0x01, 0x02, 0x03]);
        assert_eq!(segments[1].type_name, "second");
    }

    #[test]
    fn test_segment_doubles() {
        let mut data = vec![
            0xff, 0xff, 0x01, 0x00, 0x0b, 0x00, b's', b'g', b'C', b'i', b'r', b'c', b'l', b'e',
            b'D', b'i', b'm',
        ];
        data.extend_from_slice(&[0x00; 10]); // padding
        data.extend_from_slice(&0.01905f64.to_le_bytes()); // 3/4"

        let segments = tokenize(&data);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].type_name, "sgCircleDim");

        let doubles = segments[0].doubles();
        assert!(
            doubles.iter().any(|&v| (v - 0.01905).abs() < 0.0001),
            "Expected to find 0.01905 in doubles: {:?}",
            doubles
        );
    }

    #[test]
    fn test_extract_circle_sketch() {
        // Build segments that represent a circle sketch
        let segments = vec![
            Segment {
                offset: 0,
                type_name: "moProfileFeature_c".to_string(),
                name: Some("Sketch2".to_string()),
                payload: vec![],
            },
            Segment {
                offset: 20,
                type_name: "sgArcHandle".to_string(),
                name: None,
                payload: vec![],
            },
            Segment {
                offset: 40,
                type_name: "sgCircleDim".to_string(),
                name: None,
                // Payload containing the diameter 0.0254 (1 inch)
                payload: {
                    let mut p = vec![0u8; 10];
                    p.extend_from_slice(&0.0254f64.to_le_bytes());
                    p
                },
            },
        ];

        let result = extract_sketches(&segments);
        assert_eq!(result.sketches.len(), 1);
        assert_eq!(result.sketches[0].name, "Sketch2");
        assert_eq!(result.sketches[0].entities.len(), 1);

        if let SketchEntity::Circle { diameter, .. } = &result.sketches[0].entities[0] {
            assert!((diameter - 0.0254).abs() < 0.0001);
        } else {
            panic!("Expected Circle entity");
        }
    }
}
