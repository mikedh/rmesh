//! Feature backend trait for mesh generation
//!
//! Backends convert feature models to triangle meshes using different algorithms.

use super::FeatureModel;
use super::error::Result;
use crate::mesh::Trimesh;

/// Settings for mesh generation backends
pub trait BackendSettings: Default + Clone {
    /// Get the mesh resolution/quality setting
    fn resolution(&self) -> u32;
}

/// A backend that can convert feature models to triangle meshes
pub trait FeatureBackend {
    /// Settings type for this backend
    type Settings: BackendSettings;

    /// Execute the feature model and produce a mesh
    fn execute(&self, model: &FeatureModel, settings: &Self::Settings) -> Result<Trimesh>;

    /// Execute with default settings
    fn execute_default(&self, model: &FeatureModel) -> Result<Trimesh> {
        self.execute(model, &Self::Settings::default())
    }

    /// Get the name of this backend (e.g., "fidget", "manifold")
    fn name(&self) -> &'static str;

    /// Check if this backend supports a given operation type
    fn supports_operation(&self, op_type: &str) -> bool;
}

/// Default mesh settings used by multiple backends
#[derive(Debug, Clone)]
pub struct DefaultSettings {
    /// Octree depth or equivalent resolution parameter
    pub depth: u32,
    /// Optional explicit bounding box
    pub bounds: Option<([f64; 3], [f64; 3])>,
}

impl Default for DefaultSettings {
    fn default() -> Self {
        Self {
            depth: 6,
            bounds: None,
        }
    }
}

impl BackendSettings for DefaultSettings {
    fn resolution(&self) -> u32 {
        self.depth
    }
}

impl DefaultSettings {
    /// Create settings with a specific depth
    pub fn with_depth(depth: u32) -> Self {
        Self {
            depth,
            bounds: None,
        }
    }

    /// Set explicit bounds for mesh generation
    #[must_use]
    pub fn with_bounds(mut self, min: [f64; 3], max: [f64; 3]) -> Self {
        self.bounds = Some((min, max));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = DefaultSettings::default();
        assert_eq!(settings.depth, 6);
        assert!(settings.bounds.is_none());
    }

    #[test]
    fn test_settings_with_bounds() {
        let settings = DefaultSettings::with_depth(8).with_bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        assert_eq!(settings.depth, 8);
        assert!(settings.bounds.is_some());
    }
}
