//! glTF JSON Schema → Rust codegen using `typify`
//!
//! Reads the upstream Khronos glTF schema files and generates
//! `crates/rmesh/src/exchange/gltf/schema.rs`.
//!
//! Run with: `cargo run -p codegen --bin gltf-codegen`

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use regex::Regex;

use schemars::schema::RootSchema;
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let crate_dir = Path::new(&manifest_dir);

    let core_schema_dir = crate_dir.join("schemas/gLTF/specification/2.0/schema");
    let extension_base = crate_dir.join("schemas/gLTF/extensions/2.0/Khronos");
    let output_file = crate_dir
        .parent()
        .unwrap()
        .join("rmesh/src/exchange/gltf/schema.rs");

    // Collect all schema files: core + KHR_lights_punctual extension
    let mut schema_files: BTreeMap<String, PathBuf> = BTreeMap::new();

    // Load core schemas
    for entry in fs::read_dir(&core_schema_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            schema_files.insert(name, path);
        }
    }

    // Load KHR_lights_punctual extension schemas
    let lights_dir = extension_base.join("KHR_lights_punctual/schema");
    if lights_dir.exists() {
        for entry in fs::read_dir(&lights_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                let name = path.file_name().unwrap().to_str().unwrap().to_string();
                schema_files.entry(name).or_insert(path);
            }
        }
    }

    // Parse all schemas as JSON Values
    let mut schemas: BTreeMap<String, Value> = BTreeMap::new();
    for (name, path) in &schema_files {
        let content = fs::read_to_string(path)?;
        let value: Value = serde_json::from_str(&content)?;
        schemas.insert(name.clone(), value);
    }

    // Build the merged root schema
    let merged = build_merged_schema(&schemas);
    let merged_json = serde_json::to_string_pretty(&merged)?;

    // Parse as schemars RootSchema
    let root_schema: RootSchema = serde_json::from_str(&merged_json)?;

    // Generate types with typify
    let mut settings = typify::TypeSpaceSettings::default();
    settings.with_struct_builder(false);

    let mut type_space = typify::TypeSpace::new(&settings);
    type_space.add_root_schema(root_schema)?;

    // Format the generated code
    let tokens = type_space.to_stream();
    let raw_code = tokens.to_string();

    // Parse and pretty-print with prettyplease
    let syntax_tree: syn::File = syn::parse_str(&raw_code)?;
    let formatted = prettyplease::unparse(&syntax_tree);

    // Post-process to fix type names and substitutions
    let processed = postprocess(&formatted);

    // Build final output with preamble
    let mut output = String::new();
    output.push_str(PREAMBLE);
    output.push('\n');
    output.push_str(&processed);
    output.push('\n');
    output.push_str(POSTAMBLE);

    fs::create_dir_all(output_file.parent().unwrap())?;
    fs::write(&output_file, &output)?;
    println!("Generated {}", output_file.display());
    Ok(())
}

/// Build a single merged JSON Schema document from all the individual schema files.
fn build_merged_schema(schemas: &BTreeMap<String, Value>) -> Value {
    let root_key = "glTF.schema.json";
    let mut root = schemas
        .get(root_key)
        .cloned()
        .expect("glTF.schema.json not found");

    // Build definitions map from all non-root schemas, stripping ".schema.json" suffix
    let mut definitions = serde_json::Map::new();
    for (name, schema) in schemas {
        if name == root_key {
            continue;
        }
        // Skip abstract base types and utility schemas - they'll be inlined
        if matches!(
            name.as_str(),
            "glTFProperty.schema.json"
                | "glTFChildOfRootProperty.schema.json"
                | "glTFid.schema.json"
                | "extension.schema.json"
                | "extras.schema.json"
        ) {
            continue;
        }
        let key = name.trim_end_matches(".schema.json").to_string();
        let mut def = schema.clone();
        preprocess_schema(&mut def);
        definitions.insert(key, def);
    }

    // Pre-process root schema
    preprocess_schema(&mut root);

    // Add definitions to root
    root.as_object_mut()
        .unwrap()
        .insert("definitions".to_string(), Value::Object(definitions));

    // Rewrite all $ref paths from relative filenames to #/definitions/...
    // and strip .schema.json suffix from ref targets
    rewrite_refs(&mut root);

    // Remove $schema and $id from root
    if let Some(obj) = root.as_object_mut() {
        obj.remove("$schema");
        obj.remove("$id");
    }

    root
}

/// Pre-process a schema to make it typify-friendly.
fn preprocess_schema(schema: &mut Value) {
    match schema {
        Value::Object(map) => {
            // Remove glTF-specific, validation-only, and format keys
            map.remove("gltf_detailedDescription");
            map.remove("gltf_webgl");
            map.remove("gltf_sectionDescription");
            map.remove("gltf_uriType");
            map.remove("$schema");
            map.remove("$id");
            map.remove("dependencies"); // validation-only, not structural
            map.remove("not"); // validation-only
            map.remove("format"); // avoid `regress` dependency for string patterns
            map.remove("pattern"); // avoid `regress` dependency for string patterns

            // Convert draft-04 exclusiveMinimum/Maximum (boolean) to draft-07 (number)
            if let Some(Value::Bool(true)) = map.get("exclusiveMinimum") {
                if let Some(min_val) = map.get("minimum").cloned() {
                    map.insert("exclusiveMinimum".to_string(), min_val);
                    map.remove("minimum");
                } else {
                    map.remove("exclusiveMinimum");
                }
            }
            if let Some(Value::Bool(true)) = map.get("exclusiveMaximum") {
                if let Some(max_val) = map.get("maximum").cloned() {
                    map.insert("exclusiveMaximum".to_string(), max_val);
                    map.remove("maximum");
                } else {
                    map.remove("exclusiveMaximum");
                }
            }
            if matches!(map.get("exclusiveMinimum"), Some(Value::Bool(false))) {
                map.remove("exclusiveMinimum");
            }
            if matches!(map.get("exclusiveMaximum"), Some(Value::Bool(false))) {
                map.remove("exclusiveMaximum");
            }

            // Remove minimum: 1 from integer type fields to avoid NonZeroU64.
            // glTF uses this for "count" fields but we just want u64.
            if map.get("type").and_then(|v| v.as_str()) == Some("integer")
                && let Some(Value::Number(n)) = map.get("minimum")
                && n.as_u64() == Some(1)
            {
                map.remove("minimum");
            }

            // Inline glTFid: replace allOf:[{$ref:"glTFid.schema.json"}] with type:integer
            if let Some(Value::Array(all_of)) = map.get("allOf")
                && all_of.len() == 1
                && let Some(ref_val) = all_of[0].get("$ref").and_then(|v| v.as_str())
                && ref_val.contains("glTFid")
            {
                map.remove("allOf");
                map.insert("type".to_string(), Value::String("integer".into()));
                map.insert("minimum".to_string(), serde_json::json!(0));
            }

            // Inline glTFid in direct $ref
            if let Some(Value::String(ref_val)) = map.get("$ref")
                && ref_val.contains("glTFid")
            {
                map.remove("$ref");
                let desc = map.remove("description");
                map.clear();
                map.insert("type".to_string(), Value::String("integer".into()));
                map.insert("minimum".to_string(), serde_json::json!(0));
                if let Some(d) = desc {
                    map.insert("description".to_string(), d);
                }
            }

            // Remove allOf inheritance from base types (glTFProperty, glTFChildOfRootProperty).
            // typify handles allOf by merging, but we don't want the base types as separate types.
            let base_ref_info = if let Some(Value::Array(all_of)) = map.get("allOf") {
                let refs: Vec<String> = all_of
                    .iter()
                    .filter_map(|v| v.get("$ref").and_then(|r| r.as_str()).map(String::from))
                    .collect();
                let is_base = refs
                    .iter()
                    .any(|r| r.contains("glTFProperty") || r.contains("glTFChildOfRootProperty"));
                let has_child = refs.iter().any(|r| r.contains("glTFChildOfRootProperty"));
                if is_base { Some(has_child) } else { None }
            } else {
                None
            };
            if let Some(has_child_of_root) = base_ref_info {
                map.remove("allOf");
                let props = map
                    .entry("properties".to_string())
                    .or_insert_with(|| Value::Object(serde_json::Map::new()));
                if let Value::Object(props_map) = props
                    && has_child_of_root
                    && !props_map.contains_key("name")
                {
                    props_map.insert(
                        "name".to_string(),
                        serde_json::json!({"type": "string", "description": "The user-defined name of this object."}),
                    );
                }
            }

            // Handle $ref coexisting with other keywords (draft-04 pattern).
            // In draft-07, $ref overrides - keep only $ref and description.
            if map.contains_key("$ref") && map.len() > 2 {
                let ref_val = map.get("$ref").cloned().unwrap();
                let desc = map.get("description").cloned();
                map.clear();
                map.insert("$ref".to_string(), ref_val);
                if let Some(d) = desc {
                    map.insert("description".to_string(), d);
                }
            }

            // Simplify anyOf with const values to just the base type
            if let Some(Value::Array(any_of)) = map.get("anyOf")
                && let Some(simplified) = simplify_any_of(any_of)
            {
                map.remove("anyOf");
                for (k, v) in simplified {
                    map.insert(k, v);
                }
            }

            // Remove oneOf from image schema - make all fields optional
            if map.contains_key("oneOf") {
                map.remove("oneOf");
            }

            // Replace empty {} sub-schemas in properties with proper inline types
            if let Some(Value::Object(props)) = map.get_mut("properties") {
                for (key, prop_schema) in props.iter_mut() {
                    if let Value::Object(obj) = prop_schema
                        && obj.is_empty()
                    {
                        match key.as_str() {
                            "extensions" => {
                                *prop_schema = serde_json::json!({
                                    "type": "object",
                                    "additionalProperties": true
                                });
                            }
                            "extras" => {
                                // Any type
                                *prop_schema = Value::Bool(true);
                            }
                            "name" => {
                                *prop_schema = serde_json::json!({
                                    "type": "string",
                                    "description": "The user-defined name of this object."
                                });
                            }
                            _ => {
                                *prop_schema = Value::Bool(true);
                            }
                        }
                    }
                }
                // Ensure extensions and extras exist with proper types
                if !props.contains_key("extensions") {
                    props.insert(
                        "extensions".to_string(),
                        serde_json::json!({
                            "type": "object",
                            "additionalProperties": true
                        }),
                    );
                }
                if !props.contains_key("extras") {
                    props.insert("extras".to_string(), Value::Bool(true));
                }
            }

            // Recurse into all values
            for v in map.values_mut() {
                preprocess_schema(v);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                preprocess_schema(v);
            }
        }
        _ => {}
    }
}

/// Try to simplify an anyOf array where all variants are the same base type
/// with different const values.
fn simplify_any_of(variants: &[Value]) -> Option<Vec<(String, Value)>> {
    if variants.is_empty() {
        return None;
    }

    let mut base_type: Option<&str> = None;
    for variant in variants {
        if let Some(ty) = variant.get("type").and_then(|v| v.as_str()) {
            match base_type {
                None => base_type = Some(ty),
                Some(existing) if existing != ty => return None,
                _ => {}
            }
        }
    }

    let base = base_type?;
    Some(vec![("type".to_string(), Value::String(base.to_string()))])
}

/// Recursively rewrite $ref values from relative filenames to #/definitions/...
/// and strip .schema.json suffix.
fn rewrite_refs(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(ref_path)) = map.get_mut("$ref")
                && !ref_path.starts_with('#')
                && Path::new(ref_path.as_str())
                    .extension()
                    .is_some_and(|e| e == "json")
            {
                let key = ref_path.trim_end_matches(".schema.json");
                *ref_path = format!("#/definitions/{}", key);
            }
            for v in map.values_mut() {
                rewrite_refs(v);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                rewrite_refs(v);
            }
        }
        _ => {}
    }
}

/// Post-process the generated Rust code.
/// Shortens fully-qualified paths, strips JSON schema doc comments,
/// fixes integer types, unwraps Option for spec-default fields,
/// and replaces String fields with proper enum types.
fn postprocess(code: &str) -> String {
    let mut code = code.to_string();

    // 1. Shorten fully-qualified paths (longest first to avoid partial matches)
    let path_replacements = [
        (
            "::serde_json::Map<::std::string::String, ::serde_json::Value>",
            "serde_json::Map<String, serde_json::Value>",
        ),
        (
            "::std::collections::HashMap<::std::string::String,",
            "HashMap<String,",
        ),
        ("::std::option::Option", "Option"),
        ("::std::string::String", "String"),
        ("::std::vec::Vec", "Vec"),
        ("::serde_json::Map", "serde_json::Map"),
        ("::serde_json::Value", "serde_json::Value"),
        ("::serde::Deserialize", "Deserialize"),
        ("::serde::Serialize", "Serialize"),
        ("::std::default::Default", "Default"),
        ("::std::collections::HashMap", "HashMap"),
        ("::std::borrow::Cow", "std::borrow::Cow"),
        ("::std::error::Error", "std::error::Error"),
        ("::std::fmt::Display", "std::fmt::Display"),
        ("::std::fmt::Debug", "std::fmt::Debug"),
        ("::std::fmt::Formatter", "std::fmt::Formatter"),
        ("::std::fmt::Error", "std::fmt::Error"),
        ("::std::convert::TryFrom", "std::convert::TryFrom"),
    ];
    for (from, to) in &path_replacements {
        code = code.replace(from, to);
    }

    // 2. Strip embedded JSON schema doc comments
    let schema_re = Regex::new(
        r"(?ms)^[ \t]*///[ \t]*<details><summary>JSON schema</summary>.*?^[ \t]*///[ \t]*</details>\n"
    ).unwrap();
    code = schema_re.replace_all(&code, "").to_string();

    // 3. Fix integer types — targeted field replacements
    let int_replacements = [
        ("pub component_type: i64", "pub component_type: u32"),
        ("pub count: i64", "pub count: u64"),
        ("pub byte_length: i64", "pub byte_length: u64"),
        (
            "pub byte_stride: Option<i64>",
            "pub byte_stride: Option<u32>",
        ),
        ("pub target: Option<i64>", "pub target: Option<u32>"),
        ("pub mode: i64", "pub mode: u32"),
        ("pub mag_filter: Option<i64>", "pub mag_filter: Option<u32>"),
        ("pub min_filter: Option<i64>", "pub min_filter: Option<u32>"),
        ("pub wrap_s: i64", "pub wrap_s: u32"),
        ("pub wrap_t: i64", "pub wrap_t: u32"),
    ];
    for (from, to) in &int_replacements {
        code = code.replace(from, to);
    }

    // Fix default_u64 generic parameters
    code = code.replace("default_u64::<i64, 4>", "default_u64::<u32, 4>");
    code = code.replace("default_u64::<i64, 10497>", "default_u64::<u32, 10497>");

    // 4. Fix Option wrapping for spec-default fields

    // Material.alpha_cutoff: Option<f64> → f64 with serde default
    code = replace_field_in_struct(
        &code,
        "Material",
        "alpha_cutoff",
        &[r#"    #[serde(
        rename = "alphaCutoff",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub alpha_cutoff: Option<f64>,"#],
        r#"    ///The alpha cutoff value of the material.
    #[serde(rename = "alphaCutoff", default = "defaults::material_alpha_cutoff")]
    pub alpha_cutoff: f64,"#,
    );

    // Fix Material Default impl for alpha_cutoff
    code = code.replace(
        "            alpha_cutoff: Default::default(),\n            alpha_mode: defaults::material_alpha_mode(),",
        "            alpha_cutoff: defaults::material_alpha_cutoff(),\n            alpha_mode: defaults::material_alpha_mode(),",
    );

    // MaterialPbrMetallicRoughness.metallic_factor: Option<f64> → f64
    code = replace_field_in_struct(
        &code,
        "MaterialPbrMetallicRoughness",
        "metallic_factor",
        &[r#"    #[serde(
        rename = "metallicFactor",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub metallic_factor: Option<f64>,"#],
        r#"    ///The factor for the metalness of the material.
    #[serde(rename = "metallicFactor", default = "defaults::pbr_metallic_factor")]
    pub metallic_factor: f64,"#,
    );

    // MaterialPbrMetallicRoughness.roughness_factor: Option<f64> → f64
    code = replace_field_in_struct(
        &code,
        "MaterialPbrMetallicRoughness",
        "roughness_factor",
        &[r#"    #[serde(
        rename = "roughnessFactor",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub roughness_factor: Option<f64>,"#],
        r#"    ///The factor for the roughness of the material.
    #[serde(rename = "roughnessFactor", default = "defaults::pbr_roughness_factor")]
    pub roughness_factor: f64,"#,
    );

    // Fix MaterialPbrMetallicRoughness Default impl
    code = code.replace(
        "            metallic_factor: Default::default(),\n            metallic_roughness_texture: Default::default(),\n            roughness_factor: Default::default(),",
        "            metallic_factor: defaults::pbr_metallic_factor(),\n            metallic_roughness_texture: Default::default(),\n            roughness_factor: defaults::pbr_roughness_factor(),",
    );

    // Light.intensity: Option<f64> → f64
    code = replace_field_in_struct(
        &code,
        "Light",
        "intensity",
        &[
            "    #[serde(default, skip_serializing_if = \"Option::is_none\")]\n    pub intensity: Option<f64>,",
        ],
        "    ///Intensity of the light source.\n    #[serde(default = \"defaults::light_intensity\")]\n    pub intensity: f64,",
    );

    // LightSpot.inner_cone_angle: Option<f64> → f64
    code = replace_field_in_struct(
        &code,
        "LightSpot",
        "inner_cone_angle",
        &[r#"    #[serde(
        rename = "innerConeAngle",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub inner_cone_angle: Option<f64>,"#],
        r#"    ///Angle in radians from centre of spotlight where falloff begins.
    #[serde(rename = "innerConeAngle", default = "defaults::light_spot_inner_cone_angle")]
    pub inner_cone_angle: f64,"#,
    );

    // LightSpot.outer_cone_angle: Option<f64> → f64
    code = replace_field_in_struct(
        &code,
        "LightSpot",
        "outer_cone_angle",
        &[r#"    #[serde(
        rename = "outerConeAngle",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub outer_cone_angle: Option<f64>,"#],
        r#"    ///Angle in radians from centre of spotlight where falloff ends.
    #[serde(rename = "outerConeAngle", default = "defaults::light_spot_outer_cone_angle")]
    pub outer_cone_angle: f64,"#,
    );

    // Fix LightSpot Default impl
    code = code.replace(
        "            inner_cone_angle: Default::default(),\n            outer_cone_angle: Default::default(),",
        "            inner_cone_angle: defaults::light_spot_inner_cone_angle(),\n            outer_cone_angle: defaults::light_spot_outer_cone_angle(),",
    );

    // 5. Replace type_: String with enum types (context-aware)
    code = replace_field_type_in_struct(&code, "Accessor", "type_", "String", "AccessorType");
    code = replace_field_type_in_struct(&code, "Camera", "type_", "String", "CameraType");
    code = replace_field_type_in_struct(&code, "Light", "type_", "String", "GltfLightType");

    // Material.alpha_mode: String → GltfAlphaMode
    code = replace_field_type_in_struct(&code, "Material", "alpha_mode", "String", "GltfAlphaMode");

    // AnimationSampler.interpolation: String → GltfInterpolation
    code = replace_field_type_in_struct(
        &code,
        "AnimationSampler",
        "interpolation",
        "String",
        "GltfInterpolation",
    );

    // AnimationChannelTarget.path: String → GltfAnimationPath
    code = replace_field_type_in_struct(
        &code,
        "AnimationChannelTarget",
        "path",
        "String",
        "GltfAnimationPath",
    );

    // 6. Update defaults that return String to return enums
    code = code.replace(
        "pub(super) fn animation_sampler_interpolation() -> String {\n        \"LINEAR\".to_string()\n    }",
        "pub(super) fn animation_sampler_interpolation() -> super::GltfInterpolation {\n        super::GltfInterpolation::LINEAR\n    }",
    );
    code = code.replace(
        "pub(super) fn material_alpha_mode() -> String {\n        \"OPAQUE\".to_string()\n    }",
        "pub(super) fn material_alpha_mode() -> super::GltfAlphaMode {\n        super::GltfAlphaMode::OPAQUE\n    }",
    );

    // 7. Derive Default on all structs and remove hand-written Default impls
    // Add Default to the derive list for all structs
    code = code.replace(
        "#[derive(Deserialize, Serialize, Clone, Debug)]\npub struct",
        "#[derive(Deserialize, Serialize, Clone, Debug, Default)]\npub struct",
    );

    // Remove hand-written `impl Default for ...` blocks (derive handles it now).
    // These blocks have the pattern: impl Default for X { fn default() -> Self { Self { ... } } }
    // We need to match 3 levels of nested braces, so use a non-regex approach.
    code = remove_default_impls(&code);

    // 8. Inject new default functions into the defaults module
    let new_defaults = r"
    pub(super) fn material_alpha_cutoff() -> f64 {
        0.5
    }
    pub(super) fn pbr_metallic_factor() -> f64 {
        1.0
    }
    pub(super) fn pbr_roughness_factor() -> f64 {
        1.0
    }
    pub(super) fn light_intensity() -> f64 {
        1.0
    }
    pub(super) fn light_spot_inner_cone_angle() -> f64 {
        0.0
    }
    pub(super) fn light_spot_outer_cone_angle() -> f64 {
        std::f64::consts::FRAC_PI_4
    }";

    // Find the closing brace of the defaults module and inject before it
    if let Some(pos) = code.rfind("\n}\n") {
        // Check that this is actually the defaults module closing brace
        // by looking for "pub mod defaults" before it
        let before = &code[..pos];
        if before.contains("pub mod defaults") {
            code.insert_str(pos, new_defaults);
        }
    }

    code
}

/// Replace a field declaration in a struct by matching the old serde attribute + field pattern.
/// `old_patterns` is a list of possible patterns to try (handles formatting variations).
fn replace_field_in_struct(
    code: &str,
    struct_name: &str,
    field_name: &str,
    old_patterns: &[&str],
    new_field: &str,
) -> String {
    let mut code = code.to_string();
    for pattern in old_patterns {
        if code.contains(pattern) {
            code = code.replacen(pattern, new_field, 1);
            return code;
        }
    }
    eprintln!(
        "WARNING: Could not find field {} in struct {}",
        field_name, struct_name
    );
    code
}

/// Replace just the type of a field `pub field_name: OldType` → `pub field_name: NewType`
/// within a specific struct definition.
fn replace_field_type_in_struct(
    code: &str,
    struct_name: &str,
    field_name: &str,
    old_type: &str,
    new_type: &str,
) -> String {
    // Find the struct definition
    let struct_marker = format!("pub struct {} {{", struct_name);
    let Some(struct_start) = code.find(&struct_marker) else {
        eprintln!("WARNING: Could not find struct {}", struct_name);
        return code.to_string();
    };

    // Find the closing brace of the struct
    let after_struct = &code[struct_start..];
    let mut brace_depth = 0;
    let mut struct_end = struct_start;
    for (i, ch) in after_struct.char_indices() {
        match ch {
            '{' => brace_depth += 1,
            '}' => {
                brace_depth -= 1;
                if brace_depth == 0 {
                    struct_end = struct_start + i + 1;
                    break;
                }
            }
            _ => {}
        }
    }

    let struct_body = &code[struct_start..struct_end];
    let old_field = format!("pub {}: {}", field_name, old_type);
    let new_field = format!("pub {}: {}", field_name, new_type);

    if struct_body.contains(&old_field) {
        let new_body = struct_body.replacen(&old_field, &new_field, 1);
        let mut result = String::with_capacity(code.len());
        result.push_str(&code[..struct_start]);
        result.push_str(&new_body);
        result.push_str(&code[struct_end..]);
        result
    } else {
        eprintln!(
            "WARNING: Could not find field `{}` with type `{}` in struct {}",
            field_name, old_type, struct_name
        );
        code.to_string()
    }
}

/// Remove all `impl Default for TypeName { ... }` blocks using brace-counting.
fn remove_default_impls(code: &str) -> String {
    let marker = "impl Default for ";
    let mut result = String::with_capacity(code.len());
    let mut remaining = code;

    while let Some(start) = remaining.find(marker) {
        // Push everything before this impl
        result.push_str(&remaining[..start]);

        // Find the opening brace of the impl block
        let after_marker = &remaining[start..];
        if let Some(brace_pos) = after_marker.find('{') {
            // Count braces to find the matching close
            let mut depth = 0;
            let mut end = 0;
            for (i, ch) in after_marker[brace_pos..].char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = brace_pos + i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            // Skip past the impl block + any trailing newline
            remaining = &remaining[start + end..];
            if remaining.starts_with('\n') {
                remaining = &remaining[1..];
            }
        } else {
            // No opening brace found (shouldn't happen), just keep going
            result.push_str(&remaining[start..start + marker.len()]);
            remaining = &remaining[start + marker.len()..];
        }
    }

    result.push_str(remaining);
    result
}

/// Static preamble prepended to the generated code.
const PREAMBLE: &str = r"//! glTF 2.0 schema - AUTO-GENERATED from JSON Schema
//! Do not edit manually. Run `cargo run -p codegen --bin gltf-codegen` to regenerate.

#![allow(unused_imports)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::doc_lazy_continuation)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Index into a glTF array (accessor, buffer, node, etc).
pub type GltfIndex = usize;

// Component type codes
pub const COMPONENT_I8: u32 = 5120;
pub const COMPONENT_U8: u32 = 5121;
pub const COMPONENT_I16: u32 = 5122;
pub const COMPONENT_U16: u32 = 5123;
pub const COMPONENT_U32: u32 = 5125;
pub const COMPONENT_F32: u32 = 5126;

// GL primitive modes
pub const GL_POINTS: u32 = 0;
pub const GL_LINES: u32 = 1;
pub const GL_LINE_LOOP: u32 = 2;
pub const GL_LINE_STRIP: u32 = 3;
pub const GL_TRIANGLES: u32 = 4;
pub const GL_TRIANGLE_STRIP: u32 = 5;
pub const GL_TRIANGLE_FAN: u32 = 6;

pub fn component_size(t: u32) -> usize {
    match t {
        COMPONENT_I8 | COMPONENT_U8 => 1,
        COMPONENT_I16 | COMPONENT_U16 => 2,
        _ => 4,
    }
}

pub fn accessor_type_count(t: &AccessorType) -> usize {
    t.component_count()
}
";

/// Static postamble appended to the generated code.
const POSTAMBLE: &str = r#"
// Type aliases for extension convenience
pub type KhrLightsPunctual = GlTfKhrLightsPunctual;
pub type GltfLight = Light;
pub type GltfScene = Scene;
pub type GltfAnimation = Animation;
pub type GltfCamera = Camera;

/// Accessor element type (SCALAR, VEC2, VEC3, VEC4, MAT2, MAT3, MAT4).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessorType {
    #[default]
    SCALAR,
    VEC2,
    VEC3,
    VEC4,
    MAT2,
    MAT3,
    MAT4,
}

impl AccessorType {
    /// Number of components for this accessor type.
    pub fn component_count(&self) -> usize {
        match self {
            Self::SCALAR => 1,
            Self::VEC2 => 2,
            Self::VEC3 => 3,
            Self::VEC4 => 4,
            Self::MAT2 => 4,
            Self::MAT3 => 9,
            Self::MAT4 => 16,
        }
    }
}

/// Camera projection type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CameraType {
    #[default]
    Perspective,
    Orthographic,
}

/// Light type (KHR_lights_punctual).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GltfLightType {
    #[default]
    Point,
    Directional,
    Spot,
}

/// Material alpha rendering mode.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GltfAlphaMode {
    #[default]
    OPAQUE,
    MASK,
    BLEND,
}

/// Animation sampler interpolation algorithm.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GltfInterpolation {
    #[default]
    LINEAR,
    STEP,
    CUBICSPLINE,
}

/// Animation channel target path.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GltfAnimationPath {
    #[default]
    Translation,
    Rotation,
    Scale,
    Weights,
}
"#;
