//! Resolvers for external file references (MTL files, textures, etc.)

use std::collections::HashMap;
use std::io::Read;
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

/// Resolver backed by a ZIP archive with lazy decompression.
///
/// Builds a filename→index map on construction but only decompresses
/// entries when they are actually requested via `resolve()`.
/// Handles URL-encoded paths like `duck%20zaeCM.png`.
pub struct ZipResolver {
    /// Raw ZIP archive bytes (shared reference-counted).
    data: Vec<u8>,
    /// Map from filename (no directory prefix) → ZIP entry index.
    index: HashMap<String, usize>,
}

impl ZipResolver {
    /// Create a resolver from raw ZIP archive bytes.
    ///
    /// Scans the ZIP central directory to build a filename index
    /// but does not decompress any entries.
    pub fn from_zip_bytes(data: &[u8]) -> Result<Self> {
        let cursor = std::io::Cursor::new(data);
        let archive = zip::ZipArchive::new(cursor).context("failed to open ZIP archive")?;

        let mut index = HashMap::new();
        for i in 0..archive.len() {
            let name = archive
                .name_for_index(i)
                .context("failed to read ZIP entry name")?;
            // Key by filename only (strip directory prefixes)
            let key = name.rsplit('/').next().unwrap_or(name);
            if !key.is_empty() {
                index.insert(key.to_string(), i);
            }
        }

        Ok(Self {
            data: data.to_vec(),
            index,
        })
    }

    /// Decompress a ZIP entry by its index.
    #[allow(clippy::cast_possible_truncation)]
    fn read_entry(&self, entry_index: usize) -> Result<Vec<u8>> {
        let cursor = std::io::Cursor::new(&self.data);
        let mut archive = zip::ZipArchive::new(cursor).context("failed to reopen ZIP archive")?;
        let mut file = archive.by_index(entry_index)?;
        let mut buf = Vec::with_capacity(file.size() as usize);
        file.read_to_end(&mut buf)?;
        Ok(buf)
    }
}

impl Resolver for ZipResolver {
    fn resolve(&self, path: &str) -> Result<Vec<u8>> {
        // Try filename only (strip directory prefixes from query too)
        let filename = path.rsplit('/').next().unwrap_or(path);

        if let Some(&idx) = self.index.get(filename) {
            return self.read_entry(idx);
        }

        // Try URL-decoded filename
        let decoded = percent_decode(filename);
        if let Some(&idx) = self.index.get(&decoded) {
            return self.read_entry(idx);
        }

        Err(anyhow::anyhow!("file not found in ZIP archive: '{path}'"))
    }
}

/// Simple percent-decoding for URL-encoded filenames (e.g., `%20` → space).
fn percent_decode(s: &str) -> String {
    let mut result = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(byte) =
                u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""), 16)
        {
            result.push(byte);
            i += 3;
            continue;
        }
        result.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&result).into_owned()
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

    #[test]
    fn test_percent_decode() {
        assert_eq!(percent_decode("hello%20world"), "hello world");
        assert_eq!(percent_decode("no%2Fslash"), "no/slash");
        assert_eq!(percent_decode("plain"), "plain");
        assert_eq!(percent_decode("%"), "%");
        assert_eq!(percent_decode("%2"), "%2");
    }

    #[test]
    fn test_zip_resolver() {
        // Build a small in-memory ZIP with one entry
        let buf = std::io::Cursor::new(Vec::new());
        let mut writer = zip::ZipWriter::new(buf);
        writer
            .start_file::<_, ()>("textures/image.png", Default::default())
            .unwrap();
        std::io::Write::write_all(&mut writer, b"png_data").unwrap();
        let buf = writer.finish().unwrap();
        let zip_bytes = buf.into_inner();

        let resolver = ZipResolver::from_zip_bytes(&zip_bytes).unwrap();

        // Resolves by filename only (lazy decompression)
        assert_eq!(resolver.resolve("image.png").unwrap(), b"png_data");
        // Strips directory from query
        assert_eq!(resolver.resolve("other/image.png").unwrap(), b"png_data");
        // Missing file
        assert!(resolver.resolve("missing.png").is_err());
    }
}
