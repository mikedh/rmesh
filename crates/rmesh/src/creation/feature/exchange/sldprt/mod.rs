//! SLDPRT file parser
//!
//! This module handles parsing of SolidWorks Part files (.SLDPRT),
//! extracting feature operations and sketch geometry.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
//! │  SLDPRT File │────▶│   Streams    │────▶│   Tokens     │
//! │  (binary)    │     │ (DEFLATE)    │     │ (typed)      │
//! └──────────────┘     └──────────────┘     └──────────────┘
//!                                                  │
//!                                                  ▼
//! ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
//! │ FeatureModel │◀────│  Operations  │◀────│  Features &  │
//! │ (Extrude,etc)│     │ (Extrude)    │     │  Sketches    │
//! └──────────────┘     └──────────────┘     └──────────────┘
//! ```
//!
//! # Stream Extraction
//!
//! SLDPRT files contain multiple named streams. The key one is:
//! - `Contents/Config-0-ResolvedFeatures` - resolved feature tree
//!
//! Streams are found by scanning for block headers, then parsing the header
//! to get compressed size, decompressed size, and the nibble-swapped stream name.
//!
//! # Feature Extraction
//!
//! Features are extracted by scanning tokens for type markers:
//!
//! | Token Type | Maps To |
//! |------------|---------|
//! | `moExtrusion_c` | `Extrude { sign: Add }` |
//! | `moICE_c` | `Extrude { sign: Remove }` |

mod parser;
mod tokenizer;

use std::path::Path;

use super::super::{FeatureError, Result};

pub use parser::SldprtImport;

/// Parse an SLDPRT file and extract operations
///
/// # Arguments
///
/// * `path` - Path to the SLDPRT file
///
/// # Returns
///
/// An `SldprtImport` containing the extracted operations and any warnings.
pub fn read_sldprt(path: impl AsRef<Path>) -> Result<SldprtImport> {
    let data = std::fs::read(path.as_ref()).map_err(|e| {
        FeatureError::ParseError(format!("Failed to read file: {}", e))
    })?;
    parse_sldprt_bytes(&data)
}

/// Parse SLDPRT from bytes
///
/// # Arguments
///
/// * `data` - Raw bytes of the SLDPRT file
///
/// # Returns
///
/// An `SldprtImport` containing the extracted operations and any warnings.
pub fn parse_sldprt_bytes(data: &[u8]) -> Result<SldprtImport> {
    parser::parse_sldprt_bytes(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_invalid_magic() {
        let data = b"not a valid sldprt file";
        let result = parse_sldprt_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_too_short() {
        let data = [0xCD, 0xA4, 0xFA, 0x92]; // Just magic, too short
        let result = parse_sldprt_bytes(&data);
        assert!(result.is_err());
    }
}
