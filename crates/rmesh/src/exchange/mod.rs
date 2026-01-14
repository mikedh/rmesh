mod mtl;
mod obj;
mod stl;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::mesh::Trimesh;

use crate::exchange::obj::ObjMesh;
use crate::exchange::stl::BinaryStl;

/// Trait for resolving external file references (e.g., MTL files, textures).
pub trait Resolver {
    fn resolve(&self, path: &str) -> Result<Vec<u8>>;
}

/// Resolver that always fails - used when no external files are expected.
pub struct NoResolver;

impl Resolver for NoResolver {
    fn resolve(&self, path: &str) -> Result<Vec<u8>> {
        anyhow::bail!("cannot resolve '{path}': no resolver provided")
    }
}

/// Resolver that reads from the filesystem relative to a base path.
pub struct FileResolver {
    pub base_path: PathBuf,
}

impl FileResolver {
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }
}

impl Resolver for FileResolver {
    fn resolve(&self, path: &str) -> Result<Vec<u8>> {
        let full_path = self.base_path.join(path);
        std::fs::read(&full_path)
            .with_context(|| format!("failed to read '{}'", full_path.display()))
    }
}

/// Resolver that looks up files in a pre-loaded HashMap.
#[derive(Default)]
pub struct InMemoryResolver {
    pub files: HashMap<String, Vec<u8>>,
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
            .ok_or_else(|| anyhow::anyhow!("file not found: '{path}'"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// An enum to represent the different mesh file formats.
pub enum MeshFormat {
    // the STL format is a binary or ASCII format with a pure triangle soup
    STL,
    // the OBJ format, an ASCII format with a lot of extra junk
    OBJ,
    // the PLY format is a binary format with an ASCII header
    PLY,
}

impl MeshFormat {
    /// Convert a string to a MeshFormat enum.
    pub fn from_string(s: &str) -> Result<Self> {
        // clean up to match 'stl', '.stl', ' .STL ', etc
        let binding = s.to_ascii_lowercase();
        let clean = binding.trim().trim_start_matches('.').trim();
        match clean {
            "stl" => Ok(MeshFormat::STL),
            "obj" => Ok(MeshFormat::OBJ),
            "ply" => Ok(MeshFormat::PLY),
            _ => Err(anyhow::anyhow!("Unsupported file type: `{}`", clean)),
        }
    }
}

/// Load a mesh from raw bytes without resolving external references.
/// For formats like OBJ that may reference external files (MTL, textures),
/// those references will be silently ignored. Use `load_mesh_with_resolver`
/// to load external files.
pub fn load_mesh(file_data: &[u8], file_type: MeshFormat) -> Result<Trimesh> {
    load_mesh_with_resolver(file_data, file_type, &NoResolver)
}

/// Load a mesh from raw bytes, using the provided resolver for external references.
pub fn load_mesh_with_resolver<R: Resolver>(
    file_data: &[u8],
    file_type: MeshFormat,
    resolver: &R,
) -> Result<Trimesh> {
    match file_type {
        MeshFormat::STL => BinaryStl::from_bytes(file_data)?.to_mesh(),
        MeshFormat::OBJ => {
            let text = String::from_utf8_lossy(file_data);
            ObjMesh::from_string_with_resolver(&text, resolver).into_mesh()
        }
        MeshFormat::PLY => todo!(),
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_mesh_format_keys() {
        // check our string cleanup logic
        assert_eq!(MeshFormat::from_string("stl").unwrap(), MeshFormat::STL);
        assert_eq!(MeshFormat::from_string("STL").unwrap(), MeshFormat::STL);
        assert_eq!(MeshFormat::from_string(".stl").unwrap(), MeshFormat::STL);
        assert_eq!(MeshFormat::from_string(".STL").unwrap(), MeshFormat::STL);
        assert_eq!(MeshFormat::from_string("  .StL ").unwrap(), MeshFormat::STL);
        assert_eq!(MeshFormat::from_string("obj").unwrap(), MeshFormat::OBJ);
        assert_eq!(MeshFormat::from_string("obj").unwrap(), MeshFormat::OBJ);
        assert_eq!(MeshFormat::from_string("ply").unwrap(), MeshFormat::PLY);
        assert_eq!(MeshFormat::from_string("PLY").unwrap(), MeshFormat::PLY);
        assert_eq!(MeshFormat::from_string(".ply").unwrap(), MeshFormat::PLY);
        assert_eq!(MeshFormat::from_string(".PLY").unwrap(), MeshFormat::PLY);
        assert_eq!(MeshFormat::from_string("  .pLy ").unwrap(), MeshFormat::PLY);

        assert!(MeshFormat::from_string("foo").is_err());
    }
}
