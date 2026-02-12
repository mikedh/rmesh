pub mod convert;
#[allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    missing_docs,
    unused,
    unreachable_pub
)]
pub mod schema;

use std::io::Read;

use anyhow::{Context, Result};

use crate::resolvers::ZipResolver;

/// Load a ZAE (ZIP-compressed Collada) file.
///
/// Returns the parsed Collada document and a resolver for embedded resources
/// (textures, images, etc.).
pub fn load_zae(data: &[u8]) -> Result<(schema::Collada, ZipResolver)> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor).context("failed to open ZAE as ZIP archive")?;

    // Find and read the .dae file
    let dae_index = (0..archive.len())
        .find(|&i| {
            archive.by_index(i).is_ok_and(|f| {
                std::path::Path::new(f.name())
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("dae"))
            })
        })
        .context("no .dae file found in ZAE archive")?;

    let mut dae_xml = String::new();
    archive.by_index(dae_index)?.read_to_string(&mut dae_xml)?;

    let collada: schema::Collada =
        quick_xml::de::from_str(&dae_xml).context("failed to parse Collada XML from ZAE")?;

    let resolver = ZipResolver::from_zip_bytes(data)?;

    Ok((collada, resolver))
}

/// Load a plain DAE (Collada XML) file.
pub fn load_dae(data: &[u8]) -> Result<schema::Collada> {
    let text = std::str::from_utf8(data).context("DAE file is not valid UTF-8")?;
    let collada: schema::Collada =
        quick_xml::de::from_str(text).context("failed to parse Collada XML")?;
    Ok(collada)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolvers::Resolver;

    fn test_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test/data")
            .join(name)
    }

    #[test]
    fn test_load_duck_zae() {
        let path = test_path("duck.zae");
        if !path.exists() {
            eprintln!("Skipping test: {} not found", path.display());
            return;
        }
        let data = std::fs::read(&path).unwrap();
        let (collada, resolver) = load_zae(&data).unwrap();

        // Check that we parsed the Collada document
        assert!(
            collada.library_geometries.is_some(),
            "expected library_geometries in duck.zae"
        );

        // Check scene reference and visual scene hierarchy
        assert!(
            collada.scene.is_some(),
            "expected <scene> element in duck.zae"
        );
        assert!(
            collada.library_visual_scenes.is_some(),
            "expected library_visual_scenes in duck.zae"
        );
        let vs = collada.library_visual_scenes.as_ref().unwrap();
        assert!(
            !vs.visual_scene.is_empty(),
            "expected at least one visual_scene"
        );

        // Check that the resolver can find embedded textures
        // The duck.zae contains "duck zaeCM.png" (with space in name)
        let texture = resolver.resolve("duck zaeCM.png");
        assert!(
            texture.is_ok(),
            "expected to find 'duck zaeCM.png' in ZAE archive"
        );
    }

    #[test]
    fn test_load_all_zae() {
        let test_dir = test_path("");
        let mut zae_files: Vec<_> = std::fs::read_dir(&test_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "zae"))
            .collect();
        zae_files.sort_by_key(|e| e.path());

        if zae_files.is_empty() {
            eprintln!("Skipping test: no .zae files in {}", test_dir.display());
            return;
        }

        let mut timer = crate::timer::Timer::new("load_all_zae");
        for entry in &zae_files {
            let path = entry.path();
            let name = path.file_name().unwrap().to_string_lossy();
            let data = std::fs::read(&path).unwrap();

            let (collada, _resolver) = load_zae(&data).unwrap();
            timer.record(&name);

            assert!(
                collada.library_geometries.is_some(),
                "expected library_geometries in {name}"
            );
        }
        timer.print_conditionally();
    }

    #[test]
    fn test_convert_duck_zae() {
        let path = test_path("duck.zae");
        if !path.exists() {
            eprintln!("Skipping test: {} not found", path.display());
            return;
        }
        let data = std::fs::read(&path).unwrap();
        let (collada, resolver) = load_zae(&data).unwrap();
        let scene = convert::to_scene(&collada, Some(&resolver)).unwrap();

        assert!(!scene.geometry.is_empty(), "expected geometry in scene");
        for (name, geom) in &scene.geometry {
            if let crate::geometry::Geometry::Mesh(mesh) = geom {
                assert!(!mesh.vertices.is_empty(), "mesh {name} has no vertices");
                assert!(!mesh.faces.is_empty(), "mesh {name} has no faces");
                eprintln!(
                    "  {name}: {} vertices, {} faces",
                    mesh.vertices.len(),
                    mesh.faces.len()
                );
            }
        }
    }

    #[test]
    fn test_convert_all_zae() {
        let test_dir = test_path("");
        let mut zae_files: Vec<_> = std::fs::read_dir(&test_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "zae"))
            .collect();
        zae_files.sort_by_key(|e| e.path());

        if zae_files.is_empty() {
            eprintln!("Skipping test: no .zae files in {}", test_dir.display());
            return;
        }

        let mut timer = crate::timer::Timer::new("convert_all_zae");
        for entry in &zae_files {
            let path = entry.path();
            let name = path.file_name().unwrap().to_string_lossy();
            let data = std::fs::read(&path).unwrap();

            let (collada, resolver) = load_zae(&data).unwrap();
            let scene = convert::to_scene(&collada, Some(&resolver)).unwrap();
            timer.record(&name);

            assert!(
                !scene.geometry.is_empty(),
                "expected geometry in scene from {name}"
            );
        }
        timer.print_conditionally();
    }

    #[test]
    fn test_load_dispatch_zae() {
        let path = test_path("duck.zae");
        if !path.exists() {
            eprintln!("Skipping test: {} not found", path.display());
            return;
        }
        let data = std::fs::read(&path).unwrap();
        let scene =
            crate::exchange::load(&data, Some(crate::exchange::FileType::ZAE), None).unwrap();
        assert!(!scene.geometry.is_empty(), "expected geometry from load()");
    }

    #[test]
    fn test_export_roundtrip() {
        let path = test_path("duck.zae");
        if !path.exists() {
            eprintln!("Skipping test: {} not found", path.display());
            return;
        }
        let data = std::fs::read(&path).unwrap();
        let (collada, resolver) = load_zae(&data).unwrap();
        let scene = convert::to_scene(&collada, Some(&resolver)).unwrap();
        let geom_count = scene.geometry.len();

        // Export to XML bytes
        let xml_bytes = convert::from_scene(&scene).unwrap();
        assert!(!xml_bytes.is_empty(), "exported XML should not be empty");

        // Parse back
        let text = std::str::from_utf8(&xml_bytes).unwrap();
        let collada2: schema::Collada = quick_xml::de::from_str(text).unwrap();

        // Verify geometry count matches
        let geom_count2 = collada2
            .library_geometries
            .as_ref()
            .map(|lg| lg.geometry.len())
            .unwrap_or(0);
        assert_eq!(
            geom_count, geom_count2,
            "geometry count should match after roundtrip"
        );
    }

    #[test]
    fn test_texture_roundtrip() {
        use crate::attributes::Material;

        let path = test_path("duck.zae");
        if !path.exists() {
            eprintln!("Skipping test: {} not found", path.display());
            return;
        }
        let data = std::fs::read(&path).unwrap();
        let (collada, resolver) = load_zae(&data).unwrap();
        let scene = convert::to_scene(&collada, Some(&resolver)).unwrap();

        // Check that initial import has textures
        let mut found_texture = false;
        for (name, geom) in &scene.geometry {
            if let crate::geometry::Geometry::Mesh(mesh) = geom {
                for mat in &mesh.materials {
                    if let Material::Simple(simple) = mat {
                        if let Some(ref tex) = simple.diffuse_texture {
                            eprintln!(
                                "  {name} material '{}': diffuse_texture {} bytes",
                                simple.name,
                                tex.bytes_len()
                            );
                            assert!(tex.bytes_len() > 0, "texture should have non-zero bytes");
                            found_texture = true;
                        }
                    }
                }
            }
        }
        assert!(found_texture, "duck.zae should have at least one texture");
    }

    #[test]
    fn test_load_plain_dae() {
        // Minimal DAE XML that should round-trip through load_dae + to_scene
        let dae_xml = r##"<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <up_axis>Y_UP</up_axis>
  </asset>
  <library_geometries>
    <geometry id="tri-mesh" name="Triangle">
      <mesh>
        <source id="tri-positions">
          <float_array id="tri-positions-array" count="9">0 0 0 1 0 0 0 1 0</float_array>
          <technique_common>
            <accessor source="#tri-positions-array" count="3" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="tri-vertices">
          <input semantic="POSITION" source="#tri-positions"/>
        </vertices>
        <triangles count="1">
          <input semantic="VERTEX" source="#tri-vertices" offset="0"/>
          <p>0 1 2</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="Triangle" name="Triangle">
        <instance_geometry url="#tri-mesh"/>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene>
    <instance_visual_scene url="#Scene"/>
  </scene>
</COLLADA>"##;

        let collada = load_dae(dae_xml.as_bytes()).unwrap();
        let scene = convert::to_scene(&collada, None).unwrap();

        assert_eq!(scene.geometry.len(), 1);
        for (_name, geom) in &scene.geometry {
            if let crate::geometry::Geometry::Mesh(mesh) = geom {
                assert_eq!(mesh.vertices.len(), 3);
                assert_eq!(mesh.faces.len(), 1);
            }
        }

        // Round-trip through export and re-import
        let xml_bytes = convert::from_scene(&scene).unwrap();
        let collada2 = load_dae(&xml_bytes).unwrap();
        let scene2 = convert::to_scene(&collada2, None).unwrap();
        assert_eq!(scene2.geometry.len(), 1);
    }
}
