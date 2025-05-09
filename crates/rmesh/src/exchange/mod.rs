mod obj;
mod stl;

use anyhow::Result;

use crate::mesh::Trimesh;

use crate::exchange::obj::ObjMesh;
use crate::exchange::stl::BinaryStl;

#[derive(Debug, Clone, PartialEq)]
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

pub fn load_mesh(file_data: &[u8], file_type: MeshFormat) -> Result<Trimesh> {
    match file_type {
        MeshFormat::STL => BinaryStl::from_bytes(file_data)?.to_mesh(),
        MeshFormat::OBJ => ObjMesh::from_string(&String::from_utf8_lossy(file_data))?.to_mesh(),
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
