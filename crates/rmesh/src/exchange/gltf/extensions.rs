//! GLTF extension handling system.

use std::collections::HashMap;

use anyhow::Result;
use serde_json::Value;

/// Extension processing scopes - when the handler is invoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scope {
    /// After parsing a material.
    Material,
    /// When resolving a texture image index.
    TextureSource,
    /// After loading a primitive.
    Primitive,
    /// Before accessor reads (e.g., Draco decompression).
    PrimitivePreprocess,
}

/// Context passed to extension handlers.
pub struct ExtensionContext<'a> {
    /// The extension data.
    pub data: &'a Value,
    /// Mutable access to mesh parameters being built.
    pub mesh_params: Option<&'a mut MeshParams>,
}

/// Parameters being collected while building a mesh.
#[derive(Debug, Default)]
pub struct MeshParams {
    /// Vertex colors from extensions.
    pub vertex_colors: Option<Vec<[f64; 4]>>,
    /// Additional texture index from extensions.
    pub texture_index: Option<usize>,
}

/// Handler function type for extensions.
pub type Handler = fn(&mut ExtensionContext) -> Result<Option<Value>>;

/// Registry of extension handlers.
#[derive(Default)]
pub struct ExtensionRegistry {
    handlers: HashMap<(Scope, String), Handler>,
}

impl ExtensionRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with built-in handlers.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();

        // Register KHR_materials_pbrSpecularGlossiness handler
        registry.register(
            "KHR_materials_pbrSpecularGlossiness",
            Scope::Material,
            handle_pbr_specular_glossiness,
        );

        // Register EXT_texture_webp handler
        registry.register(
            "EXT_texture_webp",
            Scope::TextureSource,
            handle_texture_webp,
        );

        registry
    }

    /// Register a handler for an extension at a given scope.
    pub fn register(&mut self, name: &str, scope: Scope, handler: Handler) {
        self.handlers.insert((scope, name.to_string()), handler);
    }

    /// Process extensions at a given scope.
    pub fn handle<'a>(
        &self,
        extensions: &'a Option<HashMap<String, Value>>,
        scope: Scope,
        context: &mut ExtensionContext<'a>,
    ) -> Result<()> {
        if let Some(exts) = extensions {
            for (name, data) in exts {
                if let Some(handler) = self.handlers.get(&(scope, name.clone())) {
                    context.data = data;
                    handler(context)?;
                }
            }
        }
        Ok(())
    }

    /// Check if an extension is supported.
    pub fn supports(&self, name: &str, scope: Scope) -> bool {
        self.handlers.contains_key(&(scope, name.to_string()))
    }
}

// Built-in extension handlers

/// Handle KHR_materials_pbrSpecularGlossiness extension.
/// Converts specular-glossiness to metallic-roughness approximation.
#[allow(clippy::unnecessary_wraps)] // must match Handler type signature
fn handle_pbr_specular_glossiness(_context: &mut ExtensionContext) -> Result<Option<Value>> {
    // For now, just acknowledge the extension without conversion
    // A full implementation would convert specular-glossiness parameters
    // to metallic-roughness using the standard approximation
    Ok(None)
}

/// Handle EXT_texture_webp extension.
/// Returns the WebP texture source index.
#[allow(clippy::unnecessary_wraps)] // must match Handler type signature
fn handle_texture_webp(context: &mut ExtensionContext) -> Result<Option<Value>> {
    // The extension data contains { "source": index }
    if let Some(source) = context.data.get("source") {
        Ok(Some(source.clone()))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ExtensionRegistry::with_builtins();
        assert!(registry.supports("KHR_materials_pbrSpecularGlossiness", Scope::Material));
        assert!(registry.supports("EXT_texture_webp", Scope::TextureSource));
        assert!(!registry.supports("UNKNOWN_extension", Scope::Material));
    }

    #[test]
    fn test_handle_extensions() {
        let registry = ExtensionRegistry::with_builtins();

        let mut extensions = HashMap::new();
        extensions.insert(
            "EXT_texture_webp".to_string(),
            serde_json::json!({ "source": 5 }),
        );

        let dummy_data = serde_json::json!({});
        let mut context = ExtensionContext {
            data: &dummy_data,
            mesh_params: None,
        };

        registry
            .handle(&Some(extensions), Scope::TextureSource, &mut context)
            .unwrap();
    }
}
