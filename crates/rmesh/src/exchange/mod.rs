pub mod collada;
pub mod gltf;
mod mtl;
mod obj;
mod ply;
mod stl;

use anyhow::Result;

#[cfg(feature = "cad")]
use crate::creation::feature::FeatureModel;
#[cfg(feature = "cad")]
use crate::creation::feature::exchange::{FeatureFormat, load_feature_model};
use crate::geometry::Geometry;
use crate::resolvers::Resolver;
use crate::scene::Scene;
use crate::serialize::RmeshSerializable;

pub use crate::exchange::gltf::GltfLoader;
pub use crate::exchange::ply::export_ply;
use crate::exchange::obj::ObjMesh;
use crate::exchange::ply::PlyModel;
use crate::exchange::stl::BinaryStl;

/// Export a Scene to GLB (glTF 2.0 Binary) format.
pub fn export_glb(scene: &Scene) -> Result<Vec<u8>> {
    gltf::convert::from_scene(scene)
}

// Re-export resolvers for convenience
pub use crate::resolvers::{FileResolver, InMemoryResolver, ZipResolver};

/// Supported file types for loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FileType {
    /// Binary or ASCII triangle soup
    #[serde(rename = "stl")]
    STL,
    /// ASCII format with materials and groups
    #[serde(rename = "obj")]
    OBJ,
    /// Binary format with ASCII header
    #[serde(rename = "ply")]
    PLY,
    /// glTF 2.0 JSON
    #[serde(rename = "gltf")]
    GLTF,
    /// glTF 2.0 Binary
    #[serde(rename = "glb")]
    GLB,
    /// SolidWorks Part file (feature-based)
    #[cfg(feature = "cad")]
    #[serde(rename = "sldprt")]
    SLDPRT,
    /// rmesh CAD format (binary or JSON)
    #[cfg(feature = "cad")]
    #[serde(rename = "rcad")]
    RCAD,
    /// STEP / ISO 10303-21 boundary representation
    #[serde(rename = "step")]
    STEP,
    /// Collada DAE (plain XML)
    #[serde(rename = "dae")]
    DAE,
    /// Collada ZAE (ZIP-compressed DAE)
    #[serde(rename = "zae")]
    ZAE,
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
            #[cfg(feature = "cad")]
            "sldprt" => Ok(FileType::SLDPRT),
            #[cfg(feature = "cad")]
            "rcad" => Ok(FileType::RCAD),
            "step" | "stp" => Ok(FileType::STEP),
            "dae" | "collada" => Ok(FileType::DAE),
            "zae" => Ok(FileType::ZAE),
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

        // PLY magic: "ply\n" or "ply\r\n" (case-insensitive)
        if data.len() >= 4 {
            let first3 = [
                data[0].to_ascii_lowercase(),
                data[1].to_ascii_lowercase(),
                data[2].to_ascii_lowercase(),
            ];
            if first3 == [b'p', b'l', b'y'] && (data[3] == b'\n' || data[3] == b'\r') {
                return Some(FileType::PLY);
            }
        }

        // GLB magic: "glTF" (little-endian: 0x46546C67)
        if data[0..4] == [0x67, 0x6C, 0x54, 0x46] {
            return Some(FileType::GLB);
        }

        // Binary STL: 80-byte header followed by 4-byte triangle count
        // Check if it's a valid binary STL by verifying the triangle count
        if data.len() >= 84 {
            // Make sure it doesn't start with "solid" (ASCII STL marker)
            let starts_with_solid =
                data.len() >= 5 && (data[0..5] == *b"solid" || data[0..5] == *b"SOLID");

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

        // STEP / ISO 10303-21 starts with "ISO-10303-21;"
        if data.len() >= 13 && &data[..13] == b"ISO-10303-21;" {
            return Some(FileType::STEP);
        }

        // Collada DAE: XML containing "<COLLADA" in the first 100 bytes
        if let Ok(head) = std::str::from_utf8(&data[..data.len().min(100)])
            && head.to_ascii_lowercase().contains("<collada")
        {
            return Some(FileType::DAE);
        }

        // ZIP archive (ZAE) magic: PK\x03\x04
        if data[0..4] == [0x50, 0x4B, 0x03, 0x04] {
            return Some(FileType::ZAE);
        }

        // GLTF JSON starts with '{' (possibly with whitespace)
        let trimmed = data.iter().position(|&b| !b.is_ascii_whitespace());
        if let Some(pos) = trimmed
            && data[pos] == b'{'
        {
            return Some(FileType::GLTF);
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
            let ply = PlyModel::from_bytes(data)?;
            let mesh = ply.to_mesh(resolver)?;
            ("ply".to_string(), Geometry::Mesh(Box::new(mesh)))
        }
        FileType::GLB => {
            let loader = GltfLoader::from_glb(data)?;
            return loader.to_scene();
        }
        FileType::GLTF => {
            let loader = GltfLoader::from_gltf(data, resolver)?;
            return loader.to_scene();
        }
        #[cfg(feature = "cad")]
        FileType::SLDPRT => {
            let model = load_feature_model(data, FeatureFormat::Sldprt)?;
            ("feature".to_string(), Geometry::Feature(Box::new(model)))
        }
        #[cfg(feature = "cad")]
        FileType::RCAD => {
            let model = FeatureModel::from_bytes(data)?;
            ("feature".to_string(), Geometry::Feature(Box::new(model)))
        }
        FileType::STEP => {
            return crate::boundary::step::from_step(data).map_err(|e| anyhow::anyhow!("{e}"));
        }
        FileType::DAE => {
            let collada_doc = collada::load_dae(data)?;
            return collada::convert::to_scene(&collada_doc, resolver);
        }
        FileType::ZAE => {
            let (collada_doc, zip_resolver) = collada::load_zae(data)?;
            // Use the zip resolver for embedded textures, falling back to the caller's resolver
            let r: &dyn crate::resolvers::Resolver = &zip_resolver;
            return collada::convert::to_scene(&collada_doc, Some(r));
        }
    };

    let mut scene = Scene::new();
    scene.add(&name, geometry, None);
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
        #[cfg(feature = "cad")]
        {
            assert_eq!(
                FileType::from_extension("sldprt").unwrap(),
                FileType::SLDPRT
            );
            assert_eq!(
                FileType::from_extension("SLDPRT").unwrap(),
                FileType::SLDPRT
            );
            assert_eq!(
                FileType::from_extension(".sldprt").unwrap(),
                FileType::SLDPRT
            );

            // RCAD variations
            assert_eq!(FileType::from_extension("rcad").unwrap(), FileType::RCAD);
            assert_eq!(FileType::from_extension("RCAD").unwrap(), FileType::RCAD);
            assert_eq!(FileType::from_extension(".rcad").unwrap(), FileType::RCAD);
        }

        // STEP variations
        assert_eq!(FileType::from_extension("step").unwrap(), FileType::STEP);
        assert_eq!(FileType::from_extension("stp").unwrap(), FileType::STEP);
        assert_eq!(FileType::from_extension(".STEP").unwrap(), FileType::STEP);
        assert_eq!(FileType::from_extension(".STP").unwrap(), FileType::STEP);
        assert_eq!(FileType::from_extension("  .StP ").unwrap(), FileType::STEP);

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

        // STEP header
        let step_header = b"ISO-10303-21;\nHEADER;\n";
        assert_eq!(FileType::from_bytes(step_header), Some(FileType::STEP));

        // DAE: various header forms
        let dae_space = b"<?xml version=\"1.0\"?>\n<COLLADA xmlns=\"...\" version=\"1.4.1\">";
        assert_eq!(FileType::from_bytes(dae_space), Some(FileType::DAE));
        let dae_newline = b"<?xml version=\"1.0\"?>\n<COLLADA\n  xmlns=\"...\">";
        assert_eq!(FileType::from_bytes(dae_newline), Some(FileType::DAE));
        let dae_close = b"<COLLADA>";
        assert_eq!(FileType::from_bytes(dae_close), Some(FileType::DAE));

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
        assert_eq!(scene.graph.nodes.len(), 2); // Custom root + geometry child

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
        assert_eq!(scene.graph.nodes.len(), 2); // Custom root + geometry child
    }
}
