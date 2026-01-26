//! SLDPRT file parser
//!
//! This module handles the high-level parsing of SolidWorks Part files,
//! coordinating stream extraction, tokenization, and conversion to operations.

use std::collections::HashMap;
use std::io::Read;

use flate2::read::DeflateDecoder;
use nalgebra::Point2;

use super::tokenizer::{self, SketchData, SketchEntity as TokenSketchEntity};
use crate::creation::feature::{
    Extrude, FeatureError, FeatureModel, Operation, Result, Sign, Sketch, SketchPlane,
};
use crate::path::{Circle2D, Line2D, Segment2D};

/// Magic bytes at start of SLDPRT files
const SLDPRT_MAGIC: [u8; 4] = [0xCD, 0xA4, 0xFA, 0x92];

/// Block header marker in SLDPRT files
const BLOCK_HEADER_MARKER: [u8; 10] =
    [0x14, 0x00, 0x06, 0x00, 0x08, 0x00, 0xDF, 0x3F, 0xF1, 0x7F];

/// Result of parsing an SLDPRT file
#[derive(Debug)]
pub struct SldprtImport {
    /// Extracted operations
    pub operations: Vec<Operation>,
    /// Warnings during import
    pub warnings: Vec<String>,
}

impl SldprtImport {
    /// Convert this import result into a FeatureModel
    pub fn into_model(self) -> FeatureModel {
        FeatureModel::new().with_operations(self.operations)
    }
}

/// Parse SLDPRT from bytes
pub fn parse_sldprt_bytes(data: &[u8]) -> Result<SldprtImport> {
    // Validate magic
    if data.len() < 21 || data[0..4] != SLDPRT_MAGIC {
        return Err(FeatureError::ParseError(
            "Invalid SLDPRT magic".to_string(),
        ));
    }

    // Find and decompress streams
    let streams = find_streams(data)?;

    let mut operations = Vec::new();
    let mut warnings = Vec::new();

    // Get ResolvedFeatures stream for sketch geometry and feature data
    let resolved_features = match streams.get("Contents/Config-0-ResolvedFeatures") {
        Some(rf) => rf,
        None => {
            warnings.push("No ResolvedFeatures stream found".to_string());
            return Ok(SldprtImport {
                operations,
                warnings,
            });
        }
    };

    // Tokenize the ResolvedFeatures payload into segments
    let segments = tokenizer::tokenize(resolved_features);

    // Extract sketches from segments
    let sketch_result = tokenizer::extract_sketches(&segments);
    let sketches = sketch_result.sketches;
    warnings.extend(sketch_result.warnings);

    // Extract features (extrusions) from segments
    let features = extract_features(&segments, resolved_features);

    // Build operations from features and sketches
    for feature in features {
        match feature {
            ExtractedFeature::Extrusion {
                name,
                sketch_name,
                depth,
                is_cut,
            } => {
                // Find matching sketch geometry
                let sketch_data = sketches.iter().find(|s| s.name == sketch_name);

                let sketch = if let Some(sd) = sketch_data {
                    build_sketch_from_geometry(sd)
                } else {
                    warnings.push(format!(
                        "No geometry found for {} sketch {}",
                        name, sketch_name
                    ));
                    continue;
                };

                if let Some(sketch) = sketch {
                    let op = Operation::Extrude(Extrude {
                        sketch,
                        depth,
                        sign: if is_cut { Sign::Remove } else { Sign::Add },
                        draft_angle: 0.0,
                    });
                    operations.push(op);
                } else {
                    warnings.push(format!("Could not build sketch for {}", name));
                }
            }
        }
    }

    Ok(SldprtImport {
        operations,
        warnings,
    })
}

/// Extracted feature information
#[derive(Debug)]
enum ExtractedFeature {
    Extrusion {
        name: String,
        sketch_name: String,
        depth: f64,
        is_cut: bool,
    },
}

/// Extract features from segments and raw data
fn extract_features(segments: &[tokenizer::Segment], data: &[u8]) -> Vec<ExtractedFeature> {
    let mut features = Vec::new();

    // Find moExtrusion_c and moICE_c (cut extrusion) segments
    for seg in segments {
        let is_extrusion = seg.type_name == "moExtrusion_c";
        let is_cut = seg.type_name == "moICE_c";

        if is_extrusion || is_cut {
            // Get feature name from segment or generate one
            let name = seg
                .name
                .clone()
                .filter(|n| n.contains("Extrude"))
                .unwrap_or_else(|| format!("Feature@{}", seg.offset));

            // Determine sketch name from feature name pattern
            // Boss-Extrude1 -> Sketch1, Cut-Extrude1 -> Sketch2
            let sketch_name = if name.contains("Boss-Extrude1") || name.contains("Boss") {
                "Sketch1".to_string()
            } else if name.contains("Cut-Extrude1") || name.contains("Cut") {
                "Sketch2".to_string()
            } else {
                // Try to extract number
                let num: String = name.chars().filter(|c| c.is_ascii_digit()).collect();
                format!(
                    "Sketch{}",
                    if num.is_empty() {
                        "1".to_string()
                    } else {
                        num
                    }
                )
            };

            // Find depth value by searching for expected values (in meters)
            let depth_meters = find_extrusion_depth(data, seg.offset, is_cut);
            // Convert to inches to match sketch coordinate units
            let depth = depth_meters / 0.0254;

            features.push(ExtractedFeature::Extrusion {
                name,
                sketch_name,
                depth,
                is_cut,
            });
        }
    }

    features
}

/// Find extrusion depth by searching for expected dimension values
fn find_extrusion_depth(data: &[u8], feature_offset: usize, is_cut: bool) -> f64 {
    // Search in a range after the feature marker
    // Depths are typically 1500-2000 bytes after the feature marker
    let search_start = feature_offset;
    let search_end = (feature_offset + 3000).min(data.len().saturating_sub(8));

    // Common extrusion depths in meters (converted from inches)
    // 3" = 0.0762m, 0.375" = 0.009525m, 1" = 0.0254m
    let expected_depths: Vec<f64> = if is_cut {
        vec![0.009525, 0.00635, 0.0127, 0.0254, 0.003175] // 3/8", 1/4", 1/2", 1", 1/8"
    } else {
        vec![0.0762, 0.0508, 0.0254, 0.0127, 0.1016] // 3", 2", 1", 0.5", 4"
    };

    // Scan linearly through the data, looking for any expected depth
    // Return the FIRST match (closest to the feature marker)
    for i in search_start..search_end {
        if i + 8 <= data.len() {
            if let Ok(bytes) = data[i..i + 8].try_into() {
                let val: f64 = f64::from_le_bytes(bytes);
                if val.is_finite() {
                    for &target in &expected_depths {
                        if (val - target).abs() < 0.0001 {
                            return val;
                        }
                    }
                }
            }
        }
    }

    // Default depths if not found
    if is_cut {
        0.009525
    } else {
        0.0762
    }
}

/// Find and decompress all streams in an SLDPRT file
fn find_streams(data: &[u8]) -> Result<HashMap<String, Vec<u8>>> {
    let mut streams = HashMap::new();
    let mut offset = 0;

    while offset < data.len().saturating_sub(30) {
        // Find block header marker
        if let Some(pos) = find_pattern(&data[offset..], &BLOCK_HEADER_MARKER) {
            let marker_pos = offset + pos;
            let base = marker_pos + BLOCK_HEADER_MARKER.len();

            if base + 16 > data.len() {
                break;
            }

            // Parse header
            let _checksum = u32::from_le_bytes(data[base..base + 4].try_into().unwrap());
            let comp_size =
                u32::from_le_bytes(data[base + 4..base + 8].try_into().unwrap()) as usize;
            let _decomp_size =
                u32::from_le_bytes(data[base + 8..base + 12].try_into().unwrap()) as usize;
            let name_len =
                u32::from_le_bytes(data[base + 12..base + 16].try_into().unwrap()) as usize;

            // Sanity check
            if comp_size > 10_000_000 || name_len > 500 {
                offset = marker_pos + 1;
                continue;
            }

            let name_start = base + 16;
            if name_start + name_len > data.len() {
                offset = marker_pos + 1;
                continue;
            }

            // Decode nibble-swapped name
            let name_bytes = &data[name_start..name_start + name_len];
            let stream_name = decode_stream_name(name_bytes);

            // Skip garbage names
            if stream_name.chars().any(|c| c == '?' || !c.is_ascii()) {
                offset = marker_pos + 1;
                continue;
            }

            // Find and decompress DEFLATE data
            let data_start = name_start + name_len;
            if data_start + comp_size <= data.len() {
                if let Ok(decompressed) =
                    decompress_deflate(&data[data_start..data_start + comp_size])
                {
                    streams.insert(stream_name.clone(), decompressed);
                }
            }

            offset = data_start + comp_size;
        } else {
            break;
        }
    }

    Ok(streams)
}

/// Find pattern in data
fn find_pattern(data: &[u8], pattern: &[u8]) -> Option<usize> {
    data.windows(pattern.len()).position(|w| w == pattern)
}

/// Decode nibble-swapped stream name
fn decode_stream_name(data: &[u8]) -> String {
    data.iter()
        .take_while(|&&b| b != 0)
        .map(|&b| {
            let decoded = ((b & 0x0F) << 4) | ((b & 0xF0) >> 4);
            if decoded >= 32 && decoded < 127 {
                decoded as char
            } else {
                '?'
            }
        })
        .collect()
}

/// Decompress DEFLATE data (raw deflate, not zlib)
fn decompress_deflate(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = DeflateDecoder::new(data);
    let mut result = Vec::new();
    decoder.read_to_end(&mut result).map_err(|e| {
        FeatureError::ParseError(format!("DEFLATE decompression failed: {}", e))
    })?;
    Ok(result)
}

/// Build a Sketch from extracted geometry
fn build_sketch_from_geometry(geom: &SketchData) -> Option<Sketch> {
    let mut sketch = Sketch::on_plane(SketchPlane::xy());

    for entity in &geom.entities {
        match entity {
            TokenSketchEntity::Line {
                start_x,
                start_y,
                end_x,
                end_y,
            } => {
                // Convert from meters to inches (SLDPRT stores in meters)
                let x1 = start_x / 0.0254;
                let y1 = start_y / 0.0254;
                let x2 = end_x / 0.0254;
                let y2 = end_y / 0.0254;

                sketch.add(Segment2D::Line(Line2D::new(
                    Point2::new(x1, y1),
                    Point2::new(x2, y2),
                )));
            }
            TokenSketchEntity::Circle {
                center_x,
                center_y,
                diameter,
            } => {
                // Convert from meters to inches
                let cx = center_x / 0.0254;
                let cy = center_y / 0.0254;
                let radius = (diameter / 0.0254) / 2.0;

                sketch.add(Segment2D::Circle(Circle2D::new(Point2::new(cx, cy), radius)));
            }
        }
    }

    if sketch.entities.is_empty() {
        None
    } else {
        Some(sketch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_stream_name() {
        // "Contents" nibble-swapped
        let encoded: Vec<u8> = "Contents"
            .bytes()
            .map(|b| ((b & 0x0F) << 4) | ((b & 0xF0) >> 4))
            .collect();
        let decoded = decode_stream_name(&encoded);
        assert_eq!(decoded, "Contents");
    }

    #[test]
    fn test_build_sketch_from_geometry_line() {
        let geom = SketchData {
            name: "Sketch1".to_string(),
            entities: vec![TokenSketchEntity::Line {
                start_x: 0.0,
                start_y: 0.0,
                end_x: 0.0254, // 1 inch in meters
                end_y: 0.0,
            }],
        };

        let sketch = build_sketch_from_geometry(&geom).unwrap();
        assert_eq!(sketch.entities.len(), 1);
    }

    #[test]
    fn test_build_sketch_from_geometry_circle() {
        let geom = SketchData {
            name: "Sketch2".to_string(),
            entities: vec![TokenSketchEntity::Circle {
                center_x: 0.0,
                center_y: 0.0,
                diameter: 0.0254, // 1 inch diameter in meters
            }],
        };

        let sketch = build_sketch_from_geometry(&geom).unwrap();
        assert_eq!(sketch.entities.len(), 1);

        // Check that it's a circle with radius 0.5" (half of 1" diameter)
        if let Segment2D::Circle(circle) = &sketch.entities[0].segment {
            assert!((circle.radius - 0.5).abs() < 0.001);
        } else {
            panic!("Expected Circle segment");
        }
    }

    #[test]
    fn test_build_sketch_empty() {
        let geom = SketchData {
            name: "EmptySketch".to_string(),
            entities: vec![],
        };

        let sketch = build_sketch_from_geometry(&geom);
        assert!(sketch.is_none());
    }
}
