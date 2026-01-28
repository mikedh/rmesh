#!/usr/bin/env rust-script
//! glTF JSON Schema to Rust codegen
//! Run with: cargo run --manifest-path crates/rmesh/schemas/codegen/Cargo.toml
//!
//! Generates crates/rmesh/src/schemas/gltf_2/mod.rs from JSON Schema files.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

use convert_case::{Case, Casing};
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let schemas_dir = Path::new(&manifest_dir).parent().unwrap();
    let crate_dir = schemas_dir.parent().unwrap();

    let core_schema_dir = schemas_dir.join("glTF/specification/2.0/schema");
    let ext_schema_dir = schemas_dir.join("glTF/extensions/2.0/Khronos");
    let output_file = crate_dir.join("src/schemas/gltf_2/mod.rs");

    let mut codegen = GltfCodegen::new();

    // Load core schemas
    codegen.load_schemas(&core_schema_dir)?;

    // Load extension schemas
    if ext_schema_dir.exists() {
        for entry in fs::read_dir(&ext_schema_dir)? {
            let entry = entry?;
            let schema_dir = entry.path().join("schema");
            if schema_dir.exists() {
                codegen.load_schemas(&schema_dir)?;
            }
        }
    }

    // Generate Rust code
    let code = codegen.generate()?;

    fs::create_dir_all(output_file.parent().unwrap())?;
    fs::write(&output_file, code)?;

    println!("Generated {}", output_file.display());
    Ok(())
}

struct GltfCodegen {
    schemas: HashMap<String, Value>,
    generated: HashSet<String>,
}

impl GltfCodegen {
    fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            generated: HashSet::new(),
        }
    }

    fn load_schemas(&mut self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                let content = fs::read_to_string(&path)?;
                let schema: Value = serde_json::from_str(&content)?;
                let id = path.file_name().unwrap().to_string_lossy().to_string();
                self.schemas.insert(id, schema);
            }
        }
        Ok(())
    }

    fn generate(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let mut code = String::new();

        // Header
        code.push_str("//! glTF 2.0 schema - AUTO-GENERATED from JSON Schema\n");
        code.push_str("//! Do not edit manually. Run `cargo run -p gltf-codegen` to regenerate.\n\n");
        code.push_str("#![allow(unused_imports)]\n\n");
        code.push_str("use serde::{Deserialize, Serialize};\n");
        code.push_str("use serde_json::json;\n");
        code.push_str("use std::collections::HashMap;\n\n");

        // Constants
        code.push_str("pub type GltfIndex = usize;\n\n");
        code.push_str("// Component type codes\n");
        code.push_str("pub const COMPONENT_I8: u32 = 5120;\n");
        code.push_str("pub const COMPONENT_U8: u32 = 5121;\n");
        code.push_str("pub const COMPONENT_I16: u32 = 5122;\n");
        code.push_str("pub const COMPONENT_U16: u32 = 5123;\n");
        code.push_str("pub const COMPONENT_U32: u32 = 5125;\n");
        code.push_str("pub const COMPONENT_F32: u32 = 5126;\n\n");
        code.push_str("// GL primitive modes\n");
        code.push_str("pub const GL_POINTS: u32 = 0;\n");
        code.push_str("pub const GL_LINES: u32 = 1;\n");
        code.push_str("pub const GL_LINE_LOOP: u32 = 2;\n");
        code.push_str("pub const GL_LINE_STRIP: u32 = 3;\n");
        code.push_str("pub const GL_TRIANGLES: u32 = 4;\n");
        code.push_str("pub const GL_TRIANGLE_STRIP: u32 = 5;\n");
        code.push_str("pub const GL_TRIANGLE_FAN: u32 = 6;\n\n");

        // Helper functions
        code.push_str("pub fn component_size(t: u32) -> usize {\n");
        code.push_str("    match t { COMPONENT_I8 | COMPONENT_U8 => 1, COMPONENT_I16 | COMPONENT_U16 => 2, _ => 4 }\n");
        code.push_str("}\n\n");
        code.push_str("pub fn accessor_type_count(t: &str) -> usize {\n");
        code.push_str("    match t { \"SCALAR\" => 1, \"VEC2\" => 2, \"VEC3\" => 3, \"VEC4\" => 4, \"MAT2\" => 4, \"MAT3\" => 9, \"MAT4\" => 16, _ => 1 }\n");
        code.push_str("}\n\n");

        // Collect all struct definitions (sorted for deterministic output)
        let mut structs: BTreeMap<String, String> = BTreeMap::new();

        // Process schemas in order
        let schema_ids: Vec<_> = self.schemas.keys().cloned().collect();
        for id in schema_ids {
            if let Some(struct_code) = self.process_schema(&id)? {
                let name = self.schema_id_to_struct_name(&id);
                structs.insert(name, struct_code);
            }
        }

        // Output structs
        for struct_code in structs.values() {
            code.push_str(struct_code);
            code.push_str("\n");
        }

        // Add type aliases for common extension names
        code.push_str("// Type aliases for extension convenience\n");
        code.push_str("pub type KhrLightsPunctual = GltfKhrlightsPunctual;\n");
        code.push_str("pub type GltfLight = Light;\n");
        code.push_str("pub type GltfScene = Scene;\n");
        code.push_str("pub type GltfAnimation = Animation;\n");
        code.push_str("pub type GltfCamera = Camera;\n");

        Ok(code)
    }

    fn process_schema(&mut self, id: &str) -> Result<Option<String>, Box<dyn std::error::Error>> {
        // Skip non-object schemas and already generated
        let struct_name = self.schema_id_to_struct_name(id);
        if self.generated.contains(&struct_name) {
            return Ok(None);
        }

        let schema = match self.schemas.get(id) {
            Some(s) => s.clone(),
            None => return Ok(None),
        };

        // Skip if not an object type
        if schema.get("type").and_then(|v| v.as_str()) != Some("object") {
            return Ok(None);
        }

        self.generated.insert(struct_name.clone());

        let properties = schema.get("properties").and_then(|v| v.as_object());
        let required: HashSet<_> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let mut fields = Vec::new();
        let mut has_custom_defaults = false;

        if let Some(props) = properties {
            for (prop_name, prop_schema) in props {
                // Skip extensions/extras - they're always serde_json::Value
                if prop_name == "extensions" || prop_name == "extras" {
                    continue;
                }

                let is_required = required.contains(prop_name.as_str());
                let (rust_type, default_val) = self.json_schema_to_rust_type(prop_schema, is_required);

                if default_val.is_some() {
                    has_custom_defaults = true;
                }

                let field_name = self.to_snake_case(prop_name);
                let rename = if field_name != *prop_name {
                    Some(prop_name.clone())
                } else {
                    None
                };

                fields.push(FieldDef {
                    name: field_name,
                    rust_type,
                    rename,
                    default_val,
                });
            }
        }

        // Always add extensions and extras
        fields.push(FieldDef {
            name: "extensions".to_string(),
            rust_type: "Option<HashMap<String, serde_json::Value>>".to_string(),
            rename: None,
            default_val: None,
        });
        fields.push(FieldDef {
            name: "extras".to_string(),
            rust_type: "Option<serde_json::Value>".to_string(),
            rename: None,
            default_val: None,
        });

        // Generate struct
        let mut code = String::new();

        if has_custom_defaults {
            code.push_str("#[derive(Debug, Clone, Serialize, Deserialize)]\n");
        } else {
            code.push_str("#[derive(Debug, Clone, Default, Serialize, Deserialize)]\n");
        }
        code.push_str("#[serde(default, rename_all = \"camelCase\")]\n");
        code.push_str(&format!("pub struct {} {{\n", struct_name));

        for field in &fields {
            let field_name = if field.name == "type" {
                "type_".to_string()
            } else {
                field.name.clone()
            };

            // Determine if we need a rename attribute
            let needs_rename = field.rename.is_some() || field.name == "type";
            let rename_to = if field.name == "type" {
                "type".to_string()
            } else {
                field.rename.clone().unwrap_or_default()
            };

            if needs_rename {
                code.push_str(&format!("    #[serde(rename = \"{}\")]\n", rename_to));
            }
            code.push_str(&format!("    pub {}: {},\n", field_name, field.rust_type));
        }

        code.push_str("}\n");

        // Generate Default impl if needed
        if has_custom_defaults {
            code.push_str(&format!("\nimpl Default for {} {{\n", struct_name));
            code.push_str("    fn default() -> Self {\n");
            code.push_str("        Self {\n");
            for field in &fields {
                let name = if field.name == "type" {
                    "type_".to_string()
                } else {
                    field.name.clone()
                };
                let default = field.default_val.as_deref().unwrap_or("Default::default()");
                code.push_str(&format!("            {}: {},\n", name, default));
            }
            code.push_str("        }\n");
            code.push_str("    }\n");
            code.push_str("}\n");
        }

        Ok(Some(code))
    }

    fn json_schema_to_rust_type(&self, schema: &Value, is_required: bool) -> (String, Option<String>) {
        let base_type = self.resolve_type(schema);
        let default_val = self.extract_default(schema, &base_type, is_required);

        let rust_type = if is_required {
            base_type
        } else {
            format!("Option<{}>", base_type)
        };

        (rust_type, default_val)
    }

    fn resolve_type(&self, schema: &Value) -> String {
        // Handle $ref
        if let Some(ref_path) = schema.get("$ref").and_then(|v| v.as_str()) {
            return self.ref_to_rust_type(ref_path);
        }

        // Handle allOf (inheritance)
        if let Some(all_of) = schema.get("allOf").and_then(|v| v.as_array()) {
            for sub in all_of {
                if let Some(ref_path) = sub.get("$ref").and_then(|v| v.as_str()) {
                    return self.ref_to_rust_type(ref_path);
                }
            }
        }

        // Handle type
        match schema.get("type").and_then(|v| v.as_str()) {
            Some("string") => "String".to_string(),
            Some("integer") => "i64".to_string(),
            Some("number") => "f64".to_string(),
            Some("boolean") => "bool".to_string(),
            Some("array") => {
                let items_type = schema
                    .get("items")
                    .map(|items| self.resolve_type(items))
                    .unwrap_or_else(|| "serde_json::Value".to_string());
                format!("Vec<{}>", items_type)
            }
            Some("object") => {
                // Check for additionalProperties (HashMap)
                if let Some(add_props) = schema.get("additionalProperties") {
                    let value_type = self.resolve_type(add_props);
                    format!("HashMap<String, {}>", value_type)
                } else {
                    "serde_json::Value".to_string()
                }
            }
            _ => "serde_json::Value".to_string(),
        }
    }

    fn ref_to_rust_type(&self, ref_path: &str) -> String {
        // glTFid.schema.json -> GltfIndex
        if ref_path.contains("glTFid") {
            return "GltfIndex".to_string();
        }
        // Inline definition refs like #/definitions/packet -> serde_json::Value
        if ref_path.starts_with("#/") {
            return "serde_json::Value".to_string();
        }
        // Other refs -> struct names
        self.schema_id_to_struct_name(ref_path)
    }

    fn extract_default(&self, schema: &Value, rust_type: &str, is_required: bool) -> Option<String> {
        let default = schema.get("default")?;

        // Check if the base type is serde_json::Value
        let base_is_value = rust_type.contains("serde_json::Value");

        // For optional fields, wrap the default in Some()
        let wrap = |s: String| -> String {
            if is_required {
                s
            } else {
                format!("Some({})", s)
            }
        };

        match default {
            Value::Bool(b) => {
                if base_is_value {
                    Some(wrap(format!("serde_json::Value::Bool({})", b)))
                } else {
                    Some(wrap(b.to_string()))
                }
            }
            Value::Number(n) => {
                if base_is_value {
                    // For serde_json::Value, use json! macro or construct Number
                    if let Some(i) = n.as_i64() {
                        Some(wrap(format!("serde_json::json!({})", i)))
                    } else {
                        Some(wrap(format!("serde_json::json!({})", n.as_f64().unwrap_or(0.0))))
                    }
                } else if rust_type.contains("f64") || rust_type.contains("f32") {
                    Some(wrap(format!("{:.1}", n.as_f64().unwrap_or(0.0))))
                } else {
                    Some(wrap(n.to_string()))
                }
            }
            Value::String(s) => {
                if base_is_value {
                    Some(wrap(format!("serde_json::Value::String(\"{}\".to_string())", s)))
                } else {
                    Some(wrap(format!("\"{}\".to_string()", s)))
                }
            }
            Value::Array(arr) => {
                if arr.is_empty() {
                    Some(wrap("Vec::new()".to_string()))
                } else if arr.iter().all(|v| v.is_number()) {
                    let nums: Vec<String> = arr
                        .iter()
                        .map(|v| format!("{:.1}", v.as_f64().unwrap_or(0.0)))
                        .collect();
                    Some(wrap(format!("vec![{}]", nums.join(", "))))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn schema_id_to_struct_name(&self, id: &str) -> String {
        // accessor.sparse.indices.schema.json -> AccessorSparseIndices
        // KHR_lights_punctual -> KhrLightsPunctual
        let name = id
            .trim_end_matches(".schema.json")
            .trim_end_matches(".json")
            .replace("glTF", "Gltf")
            .replace("KHR_", "Khr")
            .replace("EXT_", "Ext");

        // Split on . and _ and capitalize each part
        name.split(|c| c == '.' || c == '_')
            .map(|part| {
                let mut chars: Vec<char> = part.chars().collect();
                if !chars.is_empty() {
                    chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
                }
                chars.into_iter().collect::<String>()
            })
            .collect()
    }

    fn to_snake_case(&self, s: &str) -> String {
        s.to_case(Case::Snake)
    }
}

struct FieldDef {
    name: String,
    rust_type: String,
    rename: Option<String>,
    default_val: Option<String>,
}
