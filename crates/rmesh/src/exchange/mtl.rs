use nalgebra::Vector3;

use crate::attributes::{Material, SimpleMaterial};
use crate::image::LazyImage;
use crate::resolvers::Resolver;

/// Parse an MTL file and return a list of materials.
///
/// If `resolver` is `Some`, textures will be loaded as LazyImage.
/// If `resolver` is `None`, texture references are skipped.
pub fn parse_mtl(data: &str, resolver: Option<&dyn Resolver>) -> Vec<Material> {
    let mut materials = Vec::new();
    let mut current: Option<SimpleMaterial> = None;

    for line in data.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "newmtl" => {
                // Save previous material if any
                if let Some(mat) = current.take() {
                    materials.push(Material::Simple(mat));
                }
                // Start new material
                let name = parts[1..].join(" ");
                current = Some(SimpleMaterial {
                    name,
                    ..Default::default()
                });
            }
            "Kd" if parts.len() >= 4 => {
                // Diffuse color
                if let Some(ref mut mat) = current {
                    if let (Ok(r), Ok(g), Ok(b)) = (
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                    ) {
                        mat.diffuse = Some(Vector3::new(r, g, b));
                    }
                }
            }
            "Ks" if parts.len() >= 4 => {
                // Specular color
                if let Some(ref mut mat) = current {
                    if let (Ok(r), Ok(g), Ok(b)) = (
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                    ) {
                        mat.specular = Some(Vector3::new(r, g, b));
                    }
                }
            }
            "Ns" if parts.len() >= 2 => {
                // Shininess
                if let Some(ref mut mat) = current {
                    if let Ok(ns) = parts[1].parse::<f64>() {
                        mat.shininess = Some(ns);
                    }
                }
            }
            "d" if parts.len() >= 2 => {
                // Alpha/dissolve
                if let Some(ref mut mat) = current {
                    if let Ok(d) = parts[1].parse::<f64>() {
                        mat.alpha = Some(d);
                    }
                }
            }
            "Tr" if parts.len() >= 2 => {
                // Transparency (inverse of dissolve)
                if let Some(ref mut mat) = current {
                    if let Ok(tr) = parts[1].parse::<f64>() {
                        mat.alpha = Some(1.0 - tr);
                    }
                }
            }
            "map_Kd" if parts.len() >= 2 => {
                // Diffuse texture map
                if let (Some(mat), Some(res)) = (&mut current, resolver) {
                    let texture_path = parts[1..].join(" ");
                    if let Ok(bytes) = res.resolve(&texture_path) {
                        mat.diffuse_texture = Some(LazyImage::new(bytes));
                    }
                }
            }
            _ => {
                // Ignore unknown directives (Ka, illum, etc.)
            }
        }
    }

    // Don't forget the last material
    if let Some(mat) = current {
        materials.push(Material::Simple(mat));
    }

    materials
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_mtl_basic() {
        let mtl_data = r#"
# Comment
newmtl material_0
Kd 1.0 0.5 0.25
Ks 1.0 1.0 1.0
Ns 100.0
d 0.8

newmtl material_1
Kd 0.0 1.0 0.0
"#;

        let materials = parse_mtl(mtl_data, None);
        assert_eq!(materials.len(), 2);

        // Extract SimpleMaterial from Material enum
        let mat0 = match &materials[0] {
            Material::Simple(m) => m,
            _ => panic!("expected SimpleMaterial"),
        };
        let mat1 = match &materials[1] {
            Material::Simple(m) => m,
            _ => panic!("expected SimpleMaterial"),
        };

        assert_eq!(mat0.name, "material_0");
        assert_eq!(mat0.diffuse, Some(Vector3::new(1.0, 0.5, 0.25)));
        assert_eq!(mat0.specular, Some(Vector3::new(1.0, 1.0, 1.0)));
        assert_eq!(mat0.shininess, Some(100.0));
        assert_eq!(mat0.alpha, Some(0.8));

        assert_eq!(mat1.name, "material_1");
        assert_eq!(mat1.diffuse, Some(Vector3::new(0.0, 1.0, 0.0)));
    }
}
