//! Error types for the feature module

use std::fmt;

/// Errors that can occur when working with feature models
#[derive(Debug, Clone)]
pub enum FeatureError {
    /// Invalid sketch (e.g., no entities, open profile)
    InvalidSketch(String),
    /// Invalid operation parameters
    InvalidOperation(String),
    /// Variable not found in environment
    VariableNotFound(String),
    /// Expression evaluation error
    ExpressionError(String),
    /// Geometry error (e.g., degenerate geometry)
    GeometryError(String),
    /// Backend execution error
    BackendError(String),
    /// Parse error (e.g., invalid file format)
    ParseError(String),
}

impl fmt::Display for FeatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeatureError::InvalidSketch(msg) => write!(f, "Invalid sketch: {}", msg),
            FeatureError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            FeatureError::VariableNotFound(name) => write!(f, "Variable not found: {}", name),
            FeatureError::ExpressionError(msg) => write!(f, "Expression error: {}", msg),
            FeatureError::GeometryError(msg) => write!(f, "Geometry error: {}", msg),
            FeatureError::BackendError(msg) => write!(f, "Backend error: {}", msg),
            FeatureError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for FeatureError {}

/// Result type for feature operations
pub type Result<T> = std::result::Result<T, FeatureError>;
