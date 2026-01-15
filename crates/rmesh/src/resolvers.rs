//! Resolvers for external file references (MTL files, textures, etc.)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Trait for resolving external file references.
///
/// Used by mesh loaders to resolve references to external files like
/// MTL material files and texture images.
pub trait Resolver {
    fn resolve(&self, path: &str) -> Result<Vec<u8>>;
}

/// Resolves files relative to a base directory.
pub struct FileResolver {
    base: PathBuf,
}

impl FileResolver {
    /// Create a resolver with the given base directory.
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    /// Create a resolver from a file path, using its parent directory as base.
    pub fn from_file_path(file_path: &Path) -> Self {
        Self::new(file_path.parent().unwrap_or(Path::new(".")))
    }
}

impl Resolver for FileResolver {
    fn resolve(&self, path: &str) -> Result<Vec<u8>> {
        let full_path = self.base.join(path);
        std::fs::read(&full_path)
            .with_context(|| format!("failed to read '{}'", full_path.display()))
    }
}

/// Resolves files from a pre-loaded in-memory map.
///
/// Useful for testing or when files are embedded/bundled.
#[derive(Default)]
pub struct InMemoryResolver {
    files: HashMap<String, Vec<u8>>,
}

impl InMemoryResolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, path: impl Into<String>, data: Vec<u8>) {
        self.files.insert(path.into(), data);
    }
}

impl Resolver for InMemoryResolver {
    fn resolve(&self, path: &str) -> Result<Vec<u8>> {
        self.files
            .get(path)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("file not found in resolver: '{path}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_resolver() {
        let mut resolver = InMemoryResolver::new();
        resolver.insert("test.txt", b"hello".to_vec());

        assert!(resolver.resolve("test.txt").is_ok());
        assert_eq!(resolver.resolve("test.txt").unwrap(), b"hello");
        assert!(resolver.resolve("missing.txt").is_err());
    }

    #[test]
    fn test_file_resolver_from_file_path() {
        let resolver = FileResolver::from_file_path(Path::new("/some/dir/file.obj"));
        assert_eq!(resolver.base, Path::new("/some/dir"));
    }
}
