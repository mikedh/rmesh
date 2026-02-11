use std::env;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;

    let schemas_dir = Path::new(&manifest_dir).join("schemas");
    let src_schemas_dir = Path::new(&manifest_dir).join("src").join("schemas");

    // Compile all shaders (render + compute) when GPU features are enabled
    if std::env::var("CARGO_FEATURE_WGPU").is_ok() {
        compile_shaders()?;
    }

    // Skip schema generation if the mod.rs already exists
    // The schemas are committed to the repository, so we don't need to regenerate them
    if src_schemas_dir.join("mod.rs").exists() {
        println!("cargo:rerun-if-changed=schemas/");
        return Ok(());
    }

    // Create the schemas directory in src if it doesn't exist
    fs::create_dir_all(&src_schemas_dir)?;

    // Process COLLADA schemas (XSD)
    process_collada_schema(
        &schemas_dir.join("collada_schema_1_4_1.xsd.zstandard"),
        &src_schemas_dir.join("collada_1_4_1"),
        "1.4.1",
    )?;

    process_collada_schema(
        &schemas_dir.join("collada_schema_1_5.xsd.zstandard"),
        &src_schemas_dir.join("collada_1_5"),
        "1.5",
    )?;

    // Skip glTF schema processing - the schema files are already generated
    // and committed to the repository

    // Create the main schemas mod.rs
    create_schemas_mod(&src_schemas_dir)?;

    // Tell Cargo to rerun this build script if any schema files change
    println!("cargo:rerun-if-changed=schemas/");

    Ok(())
}

fn decompress_zstd_file(input_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let file = File::open(input_path)?;
    let mut decoder = zstd::Decoder::new(file)?;
    let mut contents = String::new();
    decoder.read_to_string(&mut contents)?;
    Ok(contents)
}

fn process_collada_schema(
    schema_path: &Path,
    output_dir: &Path,
    version: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create output directory
    fs::create_dir_all(output_dir)?;

    // Decompress the XSD file
    let xsd_content = decompress_zstd_file(schema_path)?;

    // Parse XSD and generate Rust code using xsd-parser
    let rust_code = parse_xsd_and_generate_rust(&xsd_content, version)?;
    let mod_file = output_dir.join("mod.rs");
    let mut file = File::create(mod_file)?;
    file.write_all(rust_code.as_bytes())?;

    println!(
        "cargo:warning=Generated COLLADA {} schema bindings",
        version
    );

    Ok(())
}

fn parse_xsd_and_generate_rust(
    xsd_content: &str,
    version: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    use convert_case::{Case, Casing};
    use quick_xml::Reader;
    use quick_xml::events::Event;
    use std::collections::HashMap;

    let mut reader = Reader::from_str(xsd_content);
    let mut buf = Vec::new();
    let mut structs = HashMap::new();
    let mut current_complex_type = None;
    let mut current_fields = Vec::new();
    let mut in_sequence = false;
    let mut in_choice = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"xs:complexType" | b"xsd:complexType" => {
                        if let Ok(name_attr) = e.try_get_attribute("name")
                            && let Some(name_bytes) = name_attr
                        {
                            let name = String::from_utf8_lossy(&name_bytes.value);
                            current_complex_type = Some(name.to_case(Case::Pascal));
                            current_fields.clear();
                        }
                    }
                    b"xs:sequence" | b"xsd:sequence" => {
                        in_sequence = true;
                    }
                    b"xs:choice" | b"xsd:choice" => {
                        in_choice = true;
                    }
                    b"xs:element" | b"xsd:element" => {
                        if in_sequence || in_choice {
                            let mut field_name = String::new();
                            let mut field_type = "String".to_string();
                            let mut is_optional = false;
                            let mut is_array = false;

                            // Extract name
                            if let Ok(name_attr) = e.try_get_attribute("name")
                                && let Some(name_bytes) = name_attr
                            {
                                field_name =
                                    String::from_utf8_lossy(&name_bytes.value).to_case(Case::Snake);
                            }

                            // Extract type
                            if let Ok(type_attr) = e.try_get_attribute("type")
                                && let Some(type_bytes) = type_attr
                            {
                                field_type = map_xsd_type_to_rust(&String::from_utf8_lossy(
                                    &type_bytes.value,
                                ));
                            }

                            // Check if optional (minOccurs="0")
                            if let Ok(min_occurs) = e.try_get_attribute("minOccurs")
                                && let Some(min_bytes) = min_occurs
                                && String::from_utf8_lossy(&min_bytes.value) == "0"
                            {
                                is_optional = true;
                            }

                            // Check if array (maxOccurs="unbounded" or > 1)
                            if let Ok(max_occurs) = e.try_get_attribute("maxOccurs")
                                && let Some(max_bytes) = max_occurs
                            {
                                let max_val = String::from_utf8_lossy(&max_bytes.value);
                                if max_val == "unbounded" || max_val.parse::<i32>().unwrap_or(1) > 1
                                {
                                    is_array = true;
                                }
                            }

                            if !field_name.is_empty() {
                                if is_array {
                                    field_type = format!("Vec<{}>", field_type);
                                }
                                if is_optional {
                                    field_type = format!("Option<{}>", field_type);
                                }

                                current_fields.push((field_name, field_type, is_optional));
                            }
                        }
                    }
                    b"xs:attribute" | b"xsd:attribute" => {
                        let mut attr_name = String::new();
                        let mut attr_type = "String".to_string();
                        let mut is_optional = true; // Attributes are optional by default

                        if let Ok(name_attr) = e.try_get_attribute("name")
                            && let Some(name_bytes) = name_attr
                        {
                            attr_name =
                                String::from_utf8_lossy(&name_bytes.value).to_case(Case::Snake);
                        }

                        if let Ok(type_attr) = e.try_get_attribute("type")
                            && let Some(type_bytes) = type_attr
                        {
                            attr_type =
                                map_xsd_type_to_rust(&String::from_utf8_lossy(&type_bytes.value));
                        }

                        // Check if required
                        if let Ok(use_attr) = e.try_get_attribute("use")
                            && let Some(use_bytes) = use_attr
                            && String::from_utf8_lossy(&use_bytes.value) == "required"
                        {
                            is_optional = false;
                        }

                        if !attr_name.is_empty() {
                            if is_optional {
                                attr_type = format!("Option<{}>", attr_type);
                            }
                            current_fields.push((
                                format!("attr_{}", attr_name),
                                attr_type,
                                is_optional,
                            ));
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"xs:complexType" | b"xsd:complexType" => {
                    if let Some(name) = current_complex_type.take() {
                        structs.insert(name, current_fields.clone());
                        current_fields.clear();
                    }
                }
                b"xs:sequence" | b"xsd:sequence" => {
                    in_sequence = false;
                }
                b"xs:choice" | b"xsd:choice" => {
                    in_choice = false;
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("Error parsing XSD: {}", e).into()),
            _ => {}
        }
        buf.clear();
    }

    let module_name = format!("collada_{}", version.replace('.', "_"));
    let mut code = format!(
        "// COLLADA {} schema bindings\n// Generated from XSD\n\nuse serde::{{Deserialize, Serialize}};\nuse quick_xml::{{de, se}};\n\npub mod {} {{\n    use super::*;\n\n",
        version, module_name
    );

    // Add root COLLADA struct
    write!(
        code,
        "    #[derive(Debug, Clone, Serialize, Deserialize)]\n    pub struct Collada {{\n        // Root COLLADA element for version {}\n    }}\n\n",
        version
    )?;

    // Add parsed structs
    for (struct_name, fields) in structs {
        code.push_str("    #[derive(Debug, Clone, Serialize, Deserialize)]\n");
        writeln!(code, "    pub struct {} {{", struct_name)?;

        for (field_name, field_type, is_optional) in fields {
            if is_optional {
                code.push_str("        #[serde(skip_serializing_if = \"Option::is_none\")]\n");
            }
            writeln!(code, "        pub {}: {},", field_name, field_type)?;
        }

        code.push_str("    }\n\n");
    }

    code.push_str("}\n");
    Ok(code)
}

fn map_xsd_type_to_rust(xsd_type: &str) -> String {
    match xsd_type {
        // Basic XSD types
        "xs:int" | "xsd:int" => "i32".to_string(),
        "xs:integer" | "xsd:integer" | "xs:long" | "xsd:long" => "i64".to_string(),
        "xs:short" | "xsd:short" => "i16".to_string(),
        "xs:byte" | "xsd:byte" => "i8".to_string(),
        "xs:unsignedInt" | "xsd:unsignedInt" => "u32".to_string(),
        "xs:unsignedLong" | "xsd:unsignedLong" => "u64".to_string(),
        "xs:unsignedShort" | "xsd:unsignedShort" => "u16".to_string(),
        "xs:unsignedByte" | "xsd:unsignedByte" => "u8".to_string(),
        "xs:float" | "xsd:float" => "f32".to_string(),
        "xs:double" | "xsd:double" | "xs:decimal" | "xsd:decimal" => "f64".to_string(),
        "xs:boolean" | "xsd:boolean" => "bool".to_string(),
        "xs:string" | "xsd:string" | "xs:dateTime" | "xsd:dateTime" | "xs:date" | "xsd:date"
        | "xs:time" | "xsd:time" | "xs:anyURI" | "xsd:anyURI" | "xs:ID" | "xsd:ID" | "xs:IDREF"
        | "xsd:IDREF" | "xs:token" | "xsd:token" | "SidType" => "String".to_string(),

        // COLLADA specific types - convert to known Rust types or custom types
        "ListOfFloats" => "Vec<f64>".to_string(),
        "ListOfInts" => "Vec<i32>".to_string(),
        "ListOfUInts" => "Vec<u32>".to_string(),
        "ListOfBools" => "Vec<bool>".to_string(),
        "ListOfTokens" | "ListOfNames" => "Vec<String>".to_string(),
        "Float2" => "[f32; 2]".to_string(),
        "Float3" => "[f32; 3]".to_string(),
        "Float4" => "[f32; 4]".to_string(),
        "Float4x4" => "[[f32; 4]; 4]".to_string(),

        // If it's a custom type (doesn't start with xs: or xsd:), assume it's another struct
        custom_type if !custom_type.starts_with("xs:") && !custom_type.starts_with("xsd:") => {
            use convert_case::{Case, Casing};
            custom_type.to_case(Case::Pascal)
        }

        // Fallback
        _ => "String".to_string(),
    }
}

fn compile_shaders() -> Result<(), Box<dyn std::error::Error>> {
    use shaderloom::Shaderloom;

    let loom_path = "shader_src/loom.lua";
    if Path::new(loom_path).exists() {
        Shaderloom::new().build_from_file(loom_path)?;
        println!("cargo:rerun-if-changed=shader_src/");
    } else {
        eprintln!("Warning: {loom_path} not found, skipping shader build");
    }
    Ok(())
}

fn create_schemas_mod(schemas_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mod_file = schemas_dir.join("mod.rs");
    let mut file = File::create(mod_file)?;

    let content = r"// Generated schema modules

pub mod collada_1_4_1;
pub mod collada_1_5; 
pub mod gltf_2;

pub use collada_1_4_1::*;
pub use collada_1_5::*;
pub use gltf_2::*;
";

    file.write_all(content.as_bytes())?;
    Ok(())
}
