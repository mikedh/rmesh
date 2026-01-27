//! File exchange for feature models
//!
//! This module provides import/export functionality for feature-based CAD models.
//!
//! # Supported Formats
//!
//! - **JSON**: Native serialization format for FeatureModel
//! - **SLDPRT**: SolidWorks Part files (import only)
//!
//! # Example
//!
//! ```ignore
//! use rmesh::creation::feature::exchange::{FeatureFormat, load_feature_model};
//!
//! // Load from JSON
//! let model = load_feature_model(json_bytes, FeatureFormat::Json)?;
//!
//! // Load from SLDPRT
//! let model = load_feature_model(sldprt_bytes, FeatureFormat::Sldprt)?;
//! ```

pub mod json;
pub mod sldprt;

use super::{FeatureModel, Result};

/// Supported feature model file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureFormat {
    /// JSON format (native serialization)
    Json,
    /// SolidWorks Part file (import only)
    Sldprt,
}

impl FeatureFormat {
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "json" => Some(Self::Json),
            "sldprt" => Some(Self::Sldprt),
            _ => None,
        }
    }

    /// Get the typical file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Sldprt => "SLDPRT",
        }
    }
}

/// Load a feature model from bytes
///
/// # Arguments
///
/// * `data` - Raw bytes of the file
/// * `format` - The file format
///
/// # Returns
///
/// The parsed feature model
pub fn load_feature_model(data: &[u8], format: FeatureFormat) -> Result<FeatureModel> {
    match format {
        FeatureFormat::Json => json::from_json_bytes(data),
        FeatureFormat::Sldprt => {
            let import = sldprt::parse_sldprt_bytes(data)?;
            Ok(import.into_model())
        }
    }
}

/// Load a feature model from a file path
///
/// # Arguments
///
/// * `path` - Path to the file
///
/// # Returns
///
/// The parsed feature model. Format is detected from file extension.
pub fn load_feature_model_from_path(path: impl AsRef<std::path::Path>) -> Result<FeatureModel> {
    let path = path.as_ref();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let format = FeatureFormat::from_extension(ext).ok_or_else(|| {
        super::FeatureError::ParseError(format!("Unknown file extension: {}", ext))
    })?;

    let data = std::fs::read(path)
        .map_err(|e| super::FeatureError::ParseError(format!("Failed to read file: {}", e)))?;

    load_feature_model(&data, format)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_from_extension() {
        assert_eq!(
            FeatureFormat::from_extension("json"),
            Some(FeatureFormat::Json)
        );
        assert_eq!(
            FeatureFormat::from_extension("JSON"),
            Some(FeatureFormat::Json)
        );
        assert_eq!(
            FeatureFormat::from_extension("sldprt"),
            Some(FeatureFormat::Sldprt)
        );
        assert_eq!(
            FeatureFormat::from_extension("SLDPRT"),
            Some(FeatureFormat::Sldprt)
        );
        assert_eq!(FeatureFormat::from_extension("stl"), None);
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(FeatureFormat::Json.extension(), "json");
        assert_eq!(FeatureFormat::Sldprt.extension(), "SLDPRT");
    }
}
