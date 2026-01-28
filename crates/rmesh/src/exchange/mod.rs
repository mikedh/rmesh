pub mod gltf;
mod mtl;
mod obj;
mod stl;

use anyhow::Result;

use crate::creation::feature::exchange::{load_feature_model, FeatureFormat};
use crate::creation::feature::FeatureModel;
use crate::geometry::Geometry;
use crate::resolvers::Resolver;
use crate::scene::{Scene, SceneNode, SceneNodeKind};
use crate::serialize::RmeshSerializable;

use crate::exchange::obj::ObjMesh;
use crate::exchange::stl::BinaryStl;
pub use crate::exchange::gltf::GltfLoader;

// Re-export resolvers for convenience
pub use crate::resolvers::{FileResolver, InMemoryResolver};

/// Supported file types for loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// Binary or ASCII triangle soup
    STL,
    /// ASCII format with materials and groups
    OBJ,
    /// Binary format with ASCII header
    PLY,
    /// glTF 2.0 JSON
    GLTF,
    /// glTF 2.0 Binary
    GLB,
    /// SolidWorks Part file (feature-based)
    SLDPRT,
    /// rmesh CAD format (binary or JSON)
    RCAD,
}

impl FileType {
    /// Parse a file type from an extension string.
    ///
    /// Handles various formats: "stl", ".stl", "STL", " .STL ", etc.
    pub fn from_extension(s: &str) -> Result<Self> {
        let binding = s.to_ascii_lowercase();
        let clean = binding.trim().trim_start_matches('.').trim();
        match clean {
            "stl" => Ok(FileType::STL),
            "obj" => Ok(FileType::OBJ),
            "ply" => Ok(FileType::PLY),
            "gltf" => Ok(FileType::GLTF),
            "glb" => Ok(FileType::GLB),
            "sldprt" => Ok(FileType::SLDPRT),
            "rcad" => Ok(FileType::RCAD),
            _ => Err(anyhow::anyhow!("Unsupported file type: `{}`", clean)),
        }
    }

    /// Detect file type from magic bytes.
    ///
    /// Returns `None` if the format cannot be determined from bytes alone.
    /// Note: OBJ files cannot be reliably detected from magic bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }

        // GLB magic: "glTF" (little-endian: 0x46546C67)
        if data[0..4] == [0x67, 0x6C, 0x54, 0x46] {
            return Some(FileType::GLB);
        }

        // Binary STL: 80-byte header followed by 4-byte triangle count
        // Check if it's a valid binary STL by verifying the triangle count
        if data.len() >= 84 {
            // Make sure it doesn't start with "solid" (ASCII STL marker)
            let starts_with_solid = data.len() >= 5
                && (data[0..5] == *b"solid" || data[0..5] == *b"SOLID");

            if !starts_with_solid {
                let triangle_count =
                    u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
                // Each triangle is 50 bytes (12 normal + 36 vertices + 2 attribute)
                let expected_size = 84 + triangle_count * 50;
                // Allow some tolerance for padding
                if data.len() >= expected_size && data.len() <= expected_size + 100 {
                    return Some(FileType::STL);
                }
            }
        }

        // ASCII STL starts with "solid"
        if data.len() >= 5 && (data[0..5] == *b"solid" || data[0..5] == *b"SOLID") {
            return Some(FileType::STL);
        }

        // GLTF JSON starts with '{' (possibly with whitespace)
        let trimmed = data.iter().position(|&b| !b.is_ascii_whitespace());
        if let Some(pos) = trimmed {
            if data[pos] == b'{' {
                return Some(FileType::GLTF);
            }
        }

        None
    }
}

/// Load any supported file type, always returns a Scene.
///
/// File type is auto-detected from magic bytes if not specified.
/// Single-mesh types (STL, OBJ, PLY) are wrapped in a Scene with one geometry.
///
/// # Arguments
///
/// * `data` - Raw file bytes
/// * `file_type` - Optional file type hint. If `None`, auto-detection is attempted.
/// * `resolver` - Optional resolver for external references (MTL files, textures, buffers).
///
/// # Examples
///
/// ```ignore
/// // Auto-detect GLB format
/// let scene = load(&glb_bytes, None, None)?;
///
/// // Explicit OBJ format with resolver for materials
/// let scene = load(&obj_bytes, Some(FileType::OBJ), Some(&resolver))?;
/// ```
pub fn load(
    data: &[u8],
    file_type: Option<FileType>,
    resolver: Option<&dyn Resolver>,
) -> Result<Scene> {
    let file_type = match file_type {
        Some(ft) => ft,
        None => FileType::from_bytes(data)
            .ok_or_else(|| anyhow::anyhow!("Could not detect file type from bytes"))?,
    };

    let (name, geometry) = match file_type {
        FileType::STL => {
            let stl = BinaryStl::from_bytes(data)?;
            let name = stl.solid_name().to_string();
            let mesh = stl.to_mesh()?;
            (name, Geometry::Mesh(Box::new(mesh)))
        }
        FileType::OBJ => {
            let text = String::from_utf8_lossy(data);
            let obj = ObjMesh::from_string(text.as_ref(), resolver);
            let name = obj.primary_name().to_string();
            let mesh = obj.into_mesh()?;
            (name, Geometry::Mesh(Box::new(mesh)))
        }
        FileType::PLY => {
            return Err(anyhow::anyhow!("PLY format not yet implemented"));
        }
        FileType::GLB => {
            let loader = GltfLoader::from_glb(data)?;
            return loader.to_scene();
        }
        FileType::GLTF => {
            let loader = GltfLoader::from_gltf(data, resolver)?;
            return loader.to_scene();
        }
        FileType::SLDPRT => {
            let model = load_feature_model(data, FeatureFormat::Sldprt)?;
            ("feature".to_string(), Geometry::Feature(Box::new(model)))
        }
        FileType::RCAD => {
            let model = FeatureModel::from_bytes(data)?;
            ("feature".to_string(), Geometry::Feature(Box::new(model)))
        }
    };

    let mut scene = Scene::new();
    let (actual_name, geom_index) = scene.add_geometry(&name, geometry);
    let root_node = SceneNode {
        name: actual_name,
        children: Vec::new(),
        transform: None,
        index: vec![geom_index],
        kind: SceneNodeKind::Geometry,
    };
    let root_index = scene.graph.add_node(root_node);
    scene.graph.root = root_index;
    Ok(scene)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_file_type_from_extension() {
        // STL variations
        assert_eq!(FileType::from_extension("stl").unwrap(), FileType::STL);
        assert_eq!(FileType::from_extension("STL").unwrap(), FileType::STL);
        assert_eq!(FileType::from_extension(".stl").unwrap(), FileType::STL);
        assert_eq!(FileType::from_extension(".STL").unwrap(), FileType::STL);
        assert_eq!(FileType::from_extension("  .StL ").unwrap(), FileType::STL);

        // OBJ variations
        assert_eq!(FileType::from_extension("obj").unwrap(), FileType::OBJ);
        assert_eq!(FileType::from_extension("OBJ").unwrap(), FileType::OBJ);
        assert_eq!(FileType::from_extension(".obj").unwrap(), FileType::OBJ);

        // PLY variations
        assert_eq!(FileType::from_extension("ply").unwrap(), FileType::PLY);
        assert_eq!(FileType::from_extension("PLY").unwrap(), FileType::PLY);
        assert_eq!(FileType::from_extension(".ply").unwrap(), FileType::PLY);
        assert_eq!(FileType::from_extension(".PLY").unwrap(), FileType::PLY);
        assert_eq!(FileType::from_extension("  .pLy ").unwrap(), FileType::PLY);

        // GLTF/GLB variations
        assert_eq!(FileType::from_extension("glb").unwrap(), FileType::GLB);
        assert_eq!(FileType::from_extension("GLB").unwrap(), FileType::GLB);
        assert_eq!(FileType::from_extension(".glb").unwrap(), FileType::GLB);
        assert_eq!(FileType::from_extension("gltf").unwrap(), FileType::GLTF);
        assert_eq!(FileType::from_extension("GLTF").unwrap(), FileType::GLTF);
        assert_eq!(FileType::from_extension(".gltf").unwrap(), FileType::GLTF);

        // SLDPRT variations
        assert_eq!(FileType::from_extension("sldprt").unwrap(), FileType::SLDPRT);
        assert_eq!(FileType::from_extension("SLDPRT").unwrap(), FileType::SLDPRT);
        assert_eq!(FileType::from_extension(".sldprt").unwrap(), FileType::SLDPRT);

        // RCAD variations
        assert_eq!(FileType::from_extension("rcad").unwrap(), FileType::RCAD);
        assert_eq!(FileType::from_extension("RCAD").unwrap(), FileType::RCAD);
        assert_eq!(FileType::from_extension(".rcad").unwrap(), FileType::RCAD);

        // Unknown
        assert!(FileType::from_extension("foo").is_err());
    }

    #[test]
    fn test_file_type_from_bytes() {
        // GLB magic bytes
        let glb_header = [0x67, 0x6C, 0x54, 0x46, 0x02, 0x00, 0x00, 0x00];
        assert_eq!(FileType::from_bytes(&glb_header), Some(FileType::GLB));

        // GLTF JSON
        let gltf_json = b"{\"asset\": {\"version\": \"2.0\"}}";
        assert_eq!(FileType::from_bytes(gltf_json), Some(FileType::GLTF));

        // GLTF with leading whitespace
        let gltf_whitespace = b"  \n{\"asset\": {}}";
        assert_eq!(FileType::from_bytes(gltf_whitespace), Some(FileType::GLTF));

        // ASCII STL
        let ascii_stl = b"solid cube\nfacet normal 0 0 1\n";
        assert_eq!(FileType::from_bytes(ascii_stl), Some(FileType::STL));

        // Unknown
        let unknown = b"UNKNOWN";
        assert_eq!(FileType::from_bytes(unknown), None);
    }

    #[test]
    fn test_load_glb() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data/cube.glb");

        if !path.exists() {
            return;
        }

        let data = std::fs::read(&path).unwrap();

        // Auto-detect format
        let scene = load(&data, None, None).unwrap();
        assert!(!scene.geometry.is_empty());

        // Explicit format
        let scene = load(&data, Some(FileType::GLB), None).unwrap();
        assert!(!scene.geometry.is_empty());
    }

    #[test]
    fn test_load_stl() {
        // Create a minimal binary STL (one triangle)
        let mut stl_data = vec![0u8; 84 + 50]; // header + 1 triangle
        // Triangle count at offset 80
        stl_data[80] = 1;
        stl_data[81] = 0;
        stl_data[82] = 0;
        stl_data[83] = 0;
        // Normal (0, 0, 1)
        stl_data[84..88].copy_from_slice(&0.0f32.to_le_bytes());
        stl_data[88..92].copy_from_slice(&0.0f32.to_le_bytes());
        stl_data[92..96].copy_from_slice(&1.0f32.to_le_bytes());
        // Vertex 1 (0, 0, 0)
        stl_data[96..100].copy_from_slice(&0.0f32.to_le_bytes());
        stl_data[100..104].copy_from_slice(&0.0f32.to_le_bytes());
        stl_data[104..108].copy_from_slice(&0.0f32.to_le_bytes());
        // Vertex 2 (1, 0, 0)
        stl_data[108..112].copy_from_slice(&1.0f32.to_le_bytes());
        stl_data[112..116].copy_from_slice(&0.0f32.to_le_bytes());
        stl_data[116..120].copy_from_slice(&0.0f32.to_le_bytes());
        // Vertex 3 (0, 1, 0)
        stl_data[120..124].copy_from_slice(&0.0f32.to_le_bytes());
        stl_data[124..128].copy_from_slice(&1.0f32.to_le_bytes());
        stl_data[128..132].copy_from_slice(&0.0f32.to_le_bytes());

        // Auto-detect format
        let scene = load(&stl_data, None, None).unwrap();
        assert_eq!(scene.geometry.len(), 1);
        assert_eq!(scene.graph.nodes.len(), 1);

        // Explicit format
        let scene = load(&stl_data, Some(FileType::STL), None).unwrap();
        assert_eq!(scene.geometry.len(), 1);
    }

    #[test]
    fn test_load_obj() {
        let obj_data = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";

        // OBJ requires explicit type (can't be reliably detected from magic bytes)
        let scene = load(obj_data, Some(FileType::OBJ), None).unwrap();
        assert_eq!(scene.geometry.len(), 1);
        assert_eq!(scene.graph.nodes.len(), 1);
    }
}
