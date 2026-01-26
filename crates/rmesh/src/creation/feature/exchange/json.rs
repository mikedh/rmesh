//! JSON format support for feature models
//!
//! This module provides JSON serialization and deserialization for FeatureModel.

use std::path::Path;

use super::super::{FeatureModel, Operation, Result, FeatureError};

/// Read a feature model from a JSON file
pub fn read_json(path: impl AsRef<Path>) -> Result<FeatureModel> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        FeatureError::ParseError(format!("Failed to read file: {}", e))
    })?;
    from_json(&content)
}

/// Write a feature model to a JSON file
pub fn write_json(path: impl AsRef<Path>, model: &FeatureModel) -> Result<()> {
    let content = to_json(model)?;
    std::fs::write(path, content).map_err(|e| {
        FeatureError::ParseError(format!("Failed to write file: {}", e))
    })?;
    Ok(())
}

/// Parse a feature model from a JSON string
pub fn from_json(json: &str) -> Result<FeatureModel> {
    serde_json::from_str(json).map_err(|e| {
        FeatureError::ParseError(format!("JSON parse error: {}", e))
    })
}

/// Parse a feature model from JSON bytes
pub fn from_json_bytes(data: &[u8]) -> Result<FeatureModel> {
    serde_json::from_slice(data).map_err(|e| {
        FeatureError::ParseError(format!("JSON parse error: {}", e))
    })
}

/// Serialize a feature model to a JSON string
pub fn to_json(model: &FeatureModel) -> Result<String> {
    serde_json::to_string_pretty(model).map_err(|e| {
        FeatureError::ParseError(format!("JSON serialize error: {}", e))
    })
}

/// Serialize a feature model to a compact JSON string (no whitespace)
pub fn to_json_compact(model: &FeatureModel) -> Result<String> {
    serde_json::to_string(model).map_err(|e| {
        FeatureError::ParseError(format!("JSON serialize error: {}", e))
    })
}

/// Parse a list of operations from a JSON string
///
/// This is useful for importing operations from external sources
/// like SLDPRT parsers that produce operation lists.
pub fn operations_from_json(json: &str) -> Result<Vec<Operation>> {
    serde_json::from_str(json).map_err(|e| {
        FeatureError::ParseError(format!("JSON parse error: {}", e))
    })
}

/// Serialize operations to a JSON string
pub fn operations_to_json(ops: &[Operation]) -> Result<String> {
    serde_json::to_string_pretty(ops).map_err(|e| {
        FeatureError::ParseError(format!("JSON serialize error: {}", e))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::feature::{Extrude, Sign, Sketch, Units};

    #[test]
    fn test_roundtrip_json() {
        let model = FeatureModel::new()
            .with_units(Units::Millimeters)
            .with_variable("width", 10.0)
            .with_operation(Extrude::simple(Sketch::rectangle(10.0, 10.0), 5.0));

        let json = to_json(&model).unwrap();
        let parsed = from_json(&json).unwrap();

        assert_eq!(model.environment.units, parsed.environment.units);
        assert_eq!(model.operations.len(), parsed.operations.len());
    }

    #[test]
    fn test_roundtrip_compact() {
        let model = FeatureModel::new()
            .with_operation(Extrude::simple(Sketch::circle(5.0), 10.0));

        let json = to_json_compact(&model).unwrap();
        let parsed = from_json(&json).unwrap();

        assert_eq!(model.operations.len(), parsed.operations.len());
    }

    #[test]
    fn test_operations_json() {
        let ops = vec![
            Extrude::simple(Sketch::rectangle(10.0, 5.0), 5.0).into(),
            Extrude::new(Sketch::circle(2.0), 3.0, Sign::Remove).into(),
        ];

        let json = operations_to_json(&ops).unwrap();
        let parsed = operations_from_json(&json).unwrap();

        assert_eq!(ops.len(), parsed.len());
    }

    #[test]
    fn test_from_json_bytes() {
        let model = FeatureModel::new()
            .with_operation(Extrude::simple(Sketch::rectangle(5.0, 5.0), 2.0));

        let json = to_json(&model).unwrap();
        let parsed = from_json_bytes(json.as_bytes()).unwrap();

        assert_eq!(model.operations.len(), parsed.operations.len());
    }
}
