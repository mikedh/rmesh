//! Collada 1.4.1 XSD Schema → Rust codegen using `xsd-parser`
//!
//! Generates `crates/rmesh/src/exchange/collada/schema.rs` from the
//! official Collada XSD, restricted to geometry-relevant types only.
//!
//! Run with: `cargo run -p codegen --bin collada-codegen`

use std::fs;
use std::io::Read;
use std::path::Path;

use xsd_parser::{
    config::{NamespaceIdent, Schema},
    generate, Config, IdentType,
};

const COLLADA_NS: &[u8] = b"http://www.collada.org/2005/11/COLLADASchema";

/// Elements to generate (geometry, scene graph, materials, transforms).
/// All are top-level `<xs:element>` in the Collada XSD.
/// Only the leaf elements — NOT COLLADA or library_* (those are hand-written
/// in the preamble to avoid pulling the entire schema transitively).
const ELEMENTS: &[&str] = &[
    // geometry data
    "mesh",
    "source",
    "float_array",
    "int_array",
    "accessor",
    "vertices",
    "triangles",
    "polylist",
    "polygons",
    "lines",
    // scene graph
    "visual_scene",
    "node",
    "instance_geometry",
    "bind_material",
    "instance_material",
    // transforms
    "matrix",
    "translate",
    "rotate",
    "scale",
    "lookat",
    // materials / images (effect omitted — pulls in entire FX pipeline)
    "material",
    "image",
    "instance_effect",
    // metadata
    "asset",
    "extra",
    // geometry container
    "geometry",
];

/// Strip `default="x y z"` attributes where the value contains spaces,
/// since xsd-parser 1.4 can't parse multi-value defaults.
fn strip_multivalue_defaults(xsd: &str) -> String {
    let mut result = String::with_capacity(xsd.len());
    let mut remaining = xsd;
    let mut count = 0;

    while let Some(pos) = remaining.find("default=\"") {
        let after = &remaining[pos + 9..];
        if let Some(end) = after.find('"') {
            let value = &after[..end];
            if value.contains(' ') {
                // Multi-value default — strip it
                result.push_str(&remaining[..pos]);
                remaining = &after[end + 1..];
                count += 1;
            } else {
                let keep = pos + 9 + end + 1;
                result.push_str(&remaining[..keep]);
                remaining = &remaining[keep..];
            }
        } else {
            result.push_str(remaining);
            remaining = "";
            break;
        }
    }
    result.push_str(remaining);
    println!("Stripped {count} multi-value defaults");
    result
}

/// Remove redundant `#[serde(rename = "field_name")]` where the rename
/// matches the field name, and clean up other verbose annotations.
fn postprocess(code: &str) -> String {
    let mut lines: Vec<String> = Vec::new();

    for line in code.lines() {
        // Check if this is a `#[serde(rename = "...")]` or similar on a field
        // We'll collect serde lines and check against the next field line
        lines.push(line.to_string());
    }

    // Second pass: remove redundant renames
    let mut result: Vec<String> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        // Pattern: #[serde(..., rename = "foo")] followed by `pub foo: ...`
        if let Some(rename) = extract_serde_rename(&lines[i]) {
            // Look ahead for the field name
            if i + 1 < lines.len() {
                if let Some(field) = extract_field_name(&lines[i + 1]) {
                    if rename == field {
                        // Redundant rename — strip it or simplify the serde attr
                        let simplified = remove_rename_from_serde(&lines[i]);
                        if let Some(s) = simplified {
                            result.push(s);
                        }
                        // else: entire serde attr was just the rename, skip it
                        i += 1;
                        continue;
                    }
                }
            }
        }
        result.push(lines[i].clone());
        i += 1;
    }

    result.join("\n")
}

/// Extract the rename value from a serde attribute line like
/// `    #[serde(rename = "foo")]` or `    #[serde(default, rename = "foo")]`
fn extract_serde_rename(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if !trimmed.starts_with("#[serde(") {
        return None;
    }
    // Find rename = "..."
    let rename_pos = trimmed.find("rename = \"")?;
    let after = &trimmed[rename_pos + 10..];
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

/// Extract the field name from a struct field line like `    pub foo: Bar,`
fn extract_field_name(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("pub ")?;
    let colon = rest.find(':')?;
    Some(rest[..colon].trim().to_string())
}

/// Remove just the `rename = "..."` part from a serde attribute.
/// Returns None if the attribute becomes empty (should be removed entirely).
fn remove_rename_from_serde(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let indent = &line[..line.len() - trimmed.len()];

    // Parse the content between #[serde( and )]
    let inner_start = trimmed.find("#[serde(")? + 8;
    let inner_end = trimmed.rfind(")]")?;
    let inner = &trimmed[inner_start..inner_end];

    // Split by comma, remove the rename part
    let parts: Vec<&str> = inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.starts_with("rename =") && !s.starts_with("rename="))
        .collect();

    if parts.is_empty() {
        None
    } else {
        Some(format!("{}#[serde({})]", indent, parts.join(", ")))
    }
}

/// Flatten `Foo { attrs, #[serde(rename = "$value")] content: FooContent }` patterns.
///
/// xsd-parser generates a two-struct pattern for elements with both attributes and content:
///   `FooElementType { @id, @name, $value: FooElementTypeContent }`
///   `FooElementTypeContent { child1, child2, ... }`
///
/// quick_xml doesn't support `$value` for struct fields, so we merge them:
///   `FooElementType { @id, @name, child1, child2, ... }`
/// and remove the `FooElementTypeContent` struct.
fn flatten_value_structs(file: &mut syn::File) {
    use std::collections::HashMap;

    // Phase 1: Find all structs and index by name
    let mut struct_map: HashMap<String, usize> = HashMap::new();
    for (i, item) in file.items.iter().enumerate() {
        if let syn::Item::Struct(s) = item {
            struct_map.insert(s.ident.to_string(), i);
        }
    }

    // Phase 2: Find structs with $value fields pointing to *Content structs
    // Collect (parent_idx, content_type_name, value_field_idx) tuples
    let mut merges: Vec<(usize, String, usize)> = Vec::new();

    for (i, item) in file.items.iter().enumerate() {
        if let syn::Item::Struct(s) = item {
            if let syn::Fields::Named(ref fields) = s.fields {
                for (fi, field) in fields.named.iter().enumerate() {
                    if has_serde_rename(field, "$value") {
                        // Get the type name of the content field
                        if let Some(type_name) = extract_type_ident(&field.ty) {
                            if struct_map.contains_key(&type_name) {
                                merges.push((i, type_name, fi));
                            }
                        }
                    }
                }
            }
        }
    }

    if merges.is_empty() {
        return;
    }

    // Phase 3: For each merge, copy content struct fields into parent and mark for removal
    let mut content_to_remove: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (parent_idx, content_name, value_field_idx) in &merges {
        let content_idx = struct_map[content_name.as_str()];

        // Extract fields from content struct
        let content_fields: Vec<syn::Field> = if let syn::Item::Struct(s) = &file.items[content_idx]
        {
            if let syn::Fields::Named(ref fields) = s.fields {
                fields.named.iter().cloned().collect()
            } else {
                continue;
            }
        } else {
            continue;
        };

        // Modify parent struct: remove $value field, add content fields
        if let syn::Item::Struct(ref mut s) = file.items[*parent_idx] {
            if let syn::Fields::Named(ref mut fields) = s.fields {
                // Rebuild named fields: keep all except the $value field, then add content fields
                let kept: Vec<syn::Field> = fields
                    .named
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i != value_field_idx)
                    .map(|(_, f)| f.clone())
                    .collect();
                fields.named.clear();
                for field in kept {
                    fields.named.push(field);
                }
                for field in content_fields {
                    fields.named.push(field);
                }
            }
        }

        content_to_remove.insert(content_name.clone());
    }

    // Phase 4: Remove content structs and type aliases pointing to them
    file.items.retain(|item| match item {
        syn::Item::Struct(s) => !content_to_remove.contains(&s.ident.to_string()),
        syn::Item::Type(t) => {
            // Remove type aliases like `type FooContent = FooElementTypeContent;`
            let target = quote::quote!(#t).to_string();
            !content_to_remove.iter().any(|name| target.contains(name))
        }
        _ => true,
    });

    println!("Flattened {} $value struct pairs", merges.len());
}

/// Check if a field has `#[serde(rename = "name")]` attribute.
fn has_serde_rename(field: &syn::Field, rename: &str) -> bool {
    for attr in &field.attrs {
        if attr.path().is_ident("serde") {
            let text = quote::quote!(#attr).to_string();
            if text.contains(&format!("rename = \"{}\"", rename)) {
                return true;
            }
        }
    }
    false
}

/// Extract the simple type name from a type (e.g. `Foo` from `Foo` or `Option<Foo>`).
fn extract_type_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(p) => {
            let last = p.path.segments.last()?;
            Some(last.ident.to_string())
        }
        _ => None,
    }
}

/// Types provided by the preamble — skip these from generated code.
const PREAMBLE_TYPES: &[&str] = &[
    "ListOfUIntsType",
    "ListOfIntsType",
    "ListOfFloatsType",
    "ListOfBoolsType",
    "ListOfNamesType",
    "IdrefsType",
    "ListOfHexBinaryType",
    "Float2Type",
    "Float3Type",
    "Float4Type",
    "Float7Type",
    "Float2X2Type",
    "Float3X3Type",
    "Float4X4Type",
];

/// Check if a type name belongs to the FX pipeline bloat that should be stripped.
fn is_fx_bloat(name: &str) -> bool {
    name.starts_with("Fx")
        || name.starts_with("InitAs")
        || matches!(
            name,
            "Bool2Type"
                | "Bool3Type"
                | "Bool4Type"
                | "Int2Type"
                | "Int3Type"
                | "Int4Type"
                | "Float2X3Type"
                | "Float2X4Type"
                | "Float3X2Type"
                | "Float3X4Type"
                | "Float4X2Type"
                | "Float4X3Type"
        )
        || name.starts_with("InstanceEffectSetparam")
        || name.starts_with("InstanceEffectTechniqueHint")
        || name == "AnyType"
}

/// Strip FX pipeline types (structs, enums, impls, type aliases) from the AST.
fn strip_fx_bloat(file: &mut syn::File) {
    let before = file.items.len();
    file.items.retain(|item| {
        let name = match item {
            syn::Item::Struct(s) => s.ident.to_string(),
            syn::Item::Enum(e) => e.ident.to_string(),
            syn::Item::Type(t) => {
                let alias_name = t.ident.to_string();
                if is_fx_bloat(&alias_name) {
                    return false;
                }
                // Also strip if the target type is FX bloat
                if let syn::Type::Path(p) = &*t.ty {
                    if let Some(seg) = p.path.segments.last() {
                        if is_fx_bloat(&seg.ident.to_string()) {
                            return false;
                        }
                    }
                }
                return true;
            }
            syn::Item::Impl(i) => {
                // Check self_ty (e.g., impl Deref for Bool2Type)
                if let syn::Type::Path(p) = &*i.self_ty {
                    if let Some(seg) = p.path.segments.last() {
                        if is_fx_bloat(&seg.ident.to_string()) {
                            return false;
                        }
                    }
                }
                // Check trait type params (e.g., impl From<Bool2Type> for Vec<bool>)
                if let Some((_, ref path, _)) = i.trait_ {
                    for seg in &path.segments {
                        if let syn::PathArguments::AngleBracketed(ref args) = seg.arguments {
                            for arg in &args.args {
                                if let syn::GenericArgument::Type(syn::Type::Path(p)) = arg {
                                    if let Some(inner) = p.path.segments.last() {
                                        if is_fx_bloat(&inner.ident.to_string()) {
                                            return false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                return true;
            }
            _ => return true,
        };
        !is_fx_bloat(&name)
    });
    println!("Stripped {} FX pipeline items", before - file.items.len());
}

/// Simplify generated structs:
/// - Remove `setparam` and `technique_hint` from InstanceEffectElementType
/// - Make `created` and `modified` optional in AssetElementType
fn simplify_structs(file: &mut syn::File) {
    for item in &mut file.items {
        if let syn::Item::Struct(s) = item {
            let name = s.ident.to_string();
            match name.as_str() {
                "InstanceEffectElementType" => {
                    if let syn::Fields::Named(ref mut fields) = s.fields {
                        let orig = fields.named.len();
                        fields.named = fields
                            .named
                            .iter()
                            .filter(|f| {
                                let fname = f
                                    .ident
                                    .as_ref()
                                    .map(|i| i.to_string())
                                    .unwrap_or_default();
                                fname != "setparam" && fname != "technique_hint"
                            })
                            .cloned()
                            .collect();
                        println!(
                            "Simplified InstanceEffectElementType: {orig} -> {} fields",
                            fields.named.len()
                        );
                    }
                }
                "AssetElementType" => {
                    if let syn::Fields::Named(ref mut fields) = s.fields {
                        for field in fields.named.iter_mut() {
                            let fname = field
                                .ident
                                .as_ref()
                                .map(|i| i.to_string())
                                .unwrap_or_default();
                            if fname == "created" || fname == "modified" {
                                let attr: syn::Attribute = syn::parse_quote!(#[serde(default)]);
                                field.attrs.push(attr);
                                let ty = &field.ty;
                                field.ty = syn::parse_quote!(Option<#ty>);
                            }
                        }
                        println!("Made AssetElementType.created/modified optional");
                    }
                }
                _ => {}
            }
        }
    }
}

/// Add `#[serde(skip_serializing_if = "Option::is_none")]` to all `Option<T>` fields.
///
/// Without this, quick_xml serializes `None` as empty attributes (e.g. `set=""`)
/// or empty elements, which then fail to deserialize back (e.g. `""` → u64).
fn add_skip_serializing_if(file: &mut syn::File) {
    let mut count = 0;
    for item in &mut file.items {
        if let syn::Item::Struct(s) = item {
            if let syn::Fields::Named(ref mut fields) = s.fields {
                for field in fields.named.iter_mut() {
                    if is_option_type(&field.ty) && !has_skip_serializing_if(field) {
                        let attr: syn::Attribute = syn::parse_quote!(
                            #[serde(skip_serializing_if = "Option::is_none")]
                        );
                        field.attrs.push(attr);
                        count += 1;
                    }
                }
            }
        }
    }
    println!("Added skip_serializing_if to {count} Option fields");
}

/// Check if a type is `Option<T>`.
fn is_option_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            return seg.ident == "Option";
        }
    }
    false
}

/// Check if a field already has `skip_serializing_if` in its serde attributes.
fn has_skip_serializing_if(field: &syn::Field) -> bool {
    for attr in &field.attrs {
        if attr.path().is_ident("serde") {
            let text = quote::quote!(#attr).to_string();
            if text.contains("skip_serializing_if") {
                return true;
            }
        }
    }
    false
}

/// Remove duplicate type definitions, use statements, and impl blocks.
/// Also strip `use serde::*` (provided by preamble).
/// xsd-parser emits duplicates when multiple elements share dependencies.
fn deduplicate_and_clean(code: &str) -> String {
    use std::collections::HashSet;

    let mut file: syn::File = match syn::parse_str(code) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Warning: failed to parse generated code: {e}");
            return code.to_string();
        }
    };

    // Flatten $value struct pairs before deduplication
    flatten_value_structs(&mut file);

    // Strip FX pipeline bloat and simplify structs
    strip_fx_bloat(&mut file);
    simplify_structs(&mut file);

    // Add skip_serializing_if for Option fields (needed for XML roundtrip)
    add_skip_serializing_if(&mut file);

    // Build set of preamble-provided type names
    let preamble_types: HashSet<&str> = PREAMBLE_TYPES.iter().copied().collect();

    let mut seen = HashSet::new();
    let mut items = Vec::new();

    for item in &file.items {
        // Skip imports provided by preamble or only needed by stripped FX types
        if let syn::Item::Use(u) = item {
            let text = quote::quote!(#u).to_string();
            if text.contains("serde")
                || text.contains("core :: ops :: Deref")
                || text.contains("xsd_parser_types")
            {
                continue;
            }
        }

        // Skip structs/impls for types provided by preamble
        match item {
            syn::Item::Struct(s) if preamble_types.contains(s.ident.to_string().as_str()) => {
                continue;
            }
            syn::Item::Impl(i) => {
                if let syn::Type::Path(p) = &*i.self_ty {
                    if let Some(seg) = p.path.segments.last() {
                        if preamble_types.contains(seg.ident.to_string().as_str()) {
                            continue;
                        }
                    }
                }
                // Also skip TryFrom impls for preamble types
                if let Some((_, ref _trait_path, _)) = i.trait_ {
                    let impl_text = quote::quote!(#i).to_string();
                    if preamble_types.iter().any(|t| impl_text.contains(t)) {
                        continue;
                    }
                }
            }
            _ => {}
        }

        // Use a compact key: item kind + name
        let key = match item {
            syn::Item::Type(t) => format!("type:{}", t.ident),
            syn::Item::Struct(s) => format!("struct:{}", s.ident),
            syn::Item::Enum(e) => format!("enum:{}", e.ident),
            syn::Item::Use(u) => format!("use:{}", quote::quote!(#u)),
            syn::Item::Impl(i) => {
                let self_ty = quote::quote!(#i);
                format!("impl:{self_ty}")
            }
            _ => format!("other:{}", quote::quote!(#item)),
        };

        if seen.insert(key) {
            items.push(item.clone());
        }
    }

    let deduped = syn::File {
        shebang: file.shebang,
        attrs: file.attrs,
        items,
    };
    prettyplease::unparse(&deduped)
}

/// Replace primitive type aliases with their concrete types and remove the definitions.
fn inline_type_aliases(code: &str) -> String {
    const ALIASES: &[(&str, &str)] = &[
        // String aliases (longest first to avoid substring issues)
        ("UriFragmentType", "String"),
        ("HexBinaryType", "String"),
        ("DateTimeType", "String"),
        ("AnyUriType", "String"),
        ("NmtokenType", "String"),
        ("NcNameType", "String"),
        ("StringType", "String"),
        ("TokenType", "String"),
        ("IdrefType", "String"),
        ("NameType", "String"),
        ("IdType", "String"),
        // Numeric aliases (longest first)
        ("UnsignedIntType", "u32"),
        ("UnsignedByteType", "u8"),
        ("IntegerType", "i32"),
        ("BooleanType", "bool"),
        ("FloatType", "f64"),
        ("ShortType", "i16"),
        ("BoolType", "bool"),
        ("IntType", "i64"),
        ("UintType", "u64"),
    ];

    let mut result = code.to_string();
    let mut removed = 0;

    for &(alias, concrete) in ALIASES {
        // Remove the alias definition line
        let def = format!("pub type {} = {};\n", alias, concrete);
        if result.contains(&def) {
            result = result.replace(&def, "");
            removed += 1;
        }

        // Replace all uses of the alias with the concrete type (word-boundary aware)
        result = replace_type_name(&result, alias, concrete);
    }

    println!("Inlined {removed} primitive type aliases");
    result
}

/// Replace a type name with another, respecting identifier word boundaries.
fn replace_type_name(code: &str, from: &str, to: &str) -> String {
    let mut result = String::with_capacity(code.len());
    let mut remaining = code;

    while let Some(pos) = remaining.find(from) {
        let before = if pos > 0 {
            remaining.as_bytes()[pos - 1]
        } else {
            b' '
        };
        let after_pos = pos + from.len();
        let after = if after_pos < remaining.len() {
            remaining.as_bytes()[after_pos]
        } else {
            b' '
        };

        let is_ident_char = |b: u8| b.is_ascii_alphanumeric() || b == b'_';

        if !is_ident_char(before) && !is_ident_char(after) {
            result.push_str(&remaining[..pos]);
            result.push_str(to);
            remaining = &remaining[after_pos..];
        } else {
            result.push_str(&remaining[..after_pos]);
            remaining = &remaining[after_pos..];
        }
    }
    result.push_str(remaining);
    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let crate_dir = Path::new(&manifest_dir);

    let xsd_compressed = crate_dir.join("schemas/collada_schema_1_4_1.xsd.zstandard");
    let output_file = crate_dir
        .parent()
        .unwrap()
        .join("rmesh/src/exchange/collada/schema.rs");

    // Decompress XSD
    let compressed = fs::read(&xsd_compressed)?;
    let mut decoder = zstd::Decoder::new(&compressed[..])?;
    let mut xsd_content = String::new();
    decoder.read_to_string(&mut xsd_content)?;

    // Strip multi-value defaults
    let xsd_content = strip_multivalue_defaults(&xsd_content);

    // Rewrite xml.xsd import to local copy
    let tmp_dir = std::env::temp_dir().join("collada_codegen");
    fs::create_dir_all(&tmp_dir)?;
    let xml_xsd_src = crate_dir.join("schemas/xml.xsd");
    let xml_xsd_dst = tmp_dir.join("xml.xsd");
    fs::copy(&xml_xsd_src, &xml_xsd_dst)?;

    let xsd_content = xsd_content.replace(
        "schemaLocation=\"http://www.w3.org/2001/03/xml.xsd\"",
        &format!("schemaLocation=\"{}\"", xml_xsd_dst.display()),
    );

    let tmp_xsd = tmp_dir.join("collada_schema_1_4_1.xsd");
    fs::write(&tmp_xsd, &xsd_content)?;

    // Build generate list: only geometry-relevant elements
    let ns = NamespaceIdent::namespace(COLLADA_NS);
    let generate_list: Vec<_> = ELEMENTS
        .iter()
        .map(|&name| (IdentType::Element, Some(ns.clone()), name.to_string()))
        .collect();

    println!(
        "Generating {} Collada elements from XSD...",
        generate_list.len()
    );

    let config = Config::default()
        .with_schema(Schema::File(tmp_xsd.clone()))
        .with_generate(generate_list)
        .with_serde_quick_xml();

    let tokens = generate(config)?;
    let raw_code = tokens.to_string();

    // Format with prettyplease
    let syntax_tree: syn::File = syn::parse_str(&raw_code)?;
    let formatted = prettyplease::unparse(&syntax_tree);

    // Post-process: strip redundant renames, deduplicate items
    let processed = postprocess(&formatted);

    // Deduplicate: the generator can emit the same type/use multiple times
    // when different elements share dependencies. Remove duplicates.
    let processed = deduplicate_and_clean(&processed);

    // Inline primitive type aliases (IdType → String, FloatType → f64, etc.)
    let processed = inline_type_aliases(&processed);

    // Build final output
    let mut output = String::new();
    output.push_str(PREAMBLE);
    output.push('\n');
    output.push_str(&processed);
    output.push('\n');
    output.push_str(POSTAMBLE);

    fs::create_dir_all(output_file.parent().unwrap())?;
    fs::write(&output_file, &output)?;

    let _ = fs::remove_file(&tmp_xsd);
    println!(
        "Generated {} ({} bytes, {} lines)",
        output_file.display(),
        output.len(),
        output.lines().count(),
    );
    Ok(())
}

const PREAMBLE: &str = r#"//! Collada 1.4.1 schema types for geometry loading.
//! AUTO-GENERATED — run `cargo run -p codegen --bin collada-codegen` to regenerate.
//! Hand-written root types at the top; generated types below.

#![allow(clippy::default_trait_access)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::match_single_binding)]
#![allow(clippy::needless_update)]
#![allow(clippy::struct_excessive_bools)]
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

use serde::{Deserialize, Serialize};

// ── Space-separated list types ──
// Collada uses space-separated text for lists (e.g. "4 4 4 4" for vcount).
// The generated code derives Deserialize for `struct Foo(Vec<T>)`, but serde
// doesn't know to split strings on whitespace. These manual impls fix that.

macro_rules! space_list {
    ($name:ident, $inner:ty) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name(pub Vec<$inner>);

        impl std::ops::Deref for $name {
            type Target = Vec<$inner>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                let text: String = self.0.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ");
                s.serialize_str(&text)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let s = String::deserialize(d)?;
                if s.is_empty() {
                    return Ok(Self(Vec::new()));
                }
                let v: Vec<$inner> = s
                    .split_whitespace()
                    .map(|t| t.parse().map_err(serde::de::Error::custom))
                    .collect::<Result<_, _>>()?;
                Ok(Self(v))
            }
        }

        impl From<$name> for Vec<$inner> {
            fn from(v: $name) -> Vec<$inner> {
                v.0
            }
        }
    };
}

space_list!(ListOfUIntsType, u64);
space_list!(ListOfIntsType, i64);
space_list!(ListOfFloatsType, f64);
space_list!(ListOfBoolsType, bool);

// String-based list types (names, IDREFs) — split on whitespace
macro_rules! space_list_string {
    ($name:ident) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name(pub Vec<String>);

        impl std::ops::Deref for $name {
            type Target = Vec<String>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(&self.0.join(" "))
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let s = String::deserialize(d)?;
                if s.is_empty() {
                    return Ok(Self(Vec::new()));
                }
                Ok(Self(s.split_whitespace().map(String::from).collect()))
            }
        }
    };
}

space_list_string!(ListOfNamesType);
space_list_string!(IdrefsType);
space_list_string!(ListOfHexBinaryType);

// Fixed-size float list types (Float2, Float3, Float4, Float7, Float2x2, Float3x3, Float4x4)
// These are all Vec<f64> with length constraints.
space_list!(Float2Type, f64);
space_list!(Float3Type, f64);
space_list!(Float4Type, f64);
space_list!(Float7Type, f64);
space_list!(Float2X2Type, f64);
space_list!(Float3X3Type, f64);
space_list!(Float4X4Type, f64);

// ── Hand-written root types (not generated — avoids pulling entire schema) ──

/// Root `<COLLADA>` element.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename = "COLLADA")]
pub struct Collada {
    #[serde(default, rename = "@version", skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset: Option<AssetElementType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library_geometries: Option<LibraryGeometries>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library_visual_scenes: Option<LibraryVisualScenes>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library_materials: Option<LibraryMaterials>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library_images: Option<LibraryImages>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library_effects: Option<LibraryEffects>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene: Option<Scene>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LibraryGeometries {
    #[serde(default)]
    pub geometry: Vec<GeometryElementType>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LibraryVisualScenes {
    #[serde(default)]
    pub visual_scene: Vec<VisualSceneElementType>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LibraryMaterials {
    #[serde(default)]
    pub material: Vec<MaterialElementType>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LibraryImages {
    #[serde(default)]
    pub image: Vec<ImageElementType>,
}

/// Stub for library_effects — we don't generate the full effect pipeline,
/// but need to accept (and skip) this element during deserialization.
#[derive(Debug, Deserialize, Serialize)]
pub struct LibraryEffects {}

/// The `<scene>` element references a visual scene by URL.
#[derive(Debug, Deserialize, Serialize)]
pub struct Scene {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_visual_scene: Option<InstanceVisualScene>,
}

/// References a visual scene by URL fragment (e.g. instance_visual_scene).
#[derive(Debug, Deserialize, Serialize)]
pub struct InstanceVisualScene {
    #[serde(default, rename = "@url", skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, rename = "@sid", skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(default, rename = "@name", skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}
"#;

const POSTAMBLE: &str = r#"
// ── Accessor helpers for content enums ──

impl MeshElementType {
    pub fn sources(&self) -> impl Iterator<Item = &SourceElementType> + '_ {
        self.content.iter().filter_map(|c| match c {
            MeshElementTypeContent::Source(s) => Some(s),
            _ => None,
        })
    }
    pub fn vertices(&self) -> Option<&VerticesElementType> {
        self.content.iter().find_map(|c| match c {
            MeshElementTypeContent::Vertices(v) => Some(v),
            _ => None,
        })
    }
    pub fn triangles(&self) -> impl Iterator<Item = &TrianglesElementType> + '_ {
        self.content.iter().filter_map(|c| match c {
            MeshElementTypeContent::Triangles(t) => Some(t),
            _ => None,
        })
    }
    pub fn polylist(&self) -> impl Iterator<Item = &PolylistElementType> + '_ {
        self.content.iter().filter_map(|c| match c {
            MeshElementTypeContent::Polylist(p) => Some(p),
            _ => None,
        })
    }
    pub fn polygons(&self) -> impl Iterator<Item = &PolygonsElementType> + '_ {
        self.content.iter().filter_map(|c| match c {
            MeshElementTypeContent::Polygons(p) => Some(p),
            _ => None,
        })
    }
    pub fn lines(&self) -> impl Iterator<Item = &LinesElementType> + '_ {
        self.content.iter().filter_map(|c| match c {
            MeshElementTypeContent::Lines(l) => Some(l),
            _ => None,
        })
    }
}

impl SourceElementType {
    pub fn float_array(&self) -> Option<&FloatArrayElementType> {
        self.content.iter().find_map(|c| match c {
            SourceElementTypeContent::FloatArray(a) => Some(a),
            _ => None,
        })
    }
    pub fn int_array(&self) -> Option<&IntArrayElementType> {
        self.content.iter().find_map(|c| match c {
            SourceElementTypeContent::IntArray(a) => Some(a),
            _ => None,
        })
    }
    pub fn technique_common(&self) -> Option<&SourceTechniqueCommonElementType> {
        self.content.iter().find_map(|c| match c {
            SourceElementTypeContent::TechniqueCommon(t) => Some(t),
            _ => None,
        })
    }
}

impl NodeElementType {
    pub fn matrices(&self) -> impl Iterator<Item = &MatrixElementType> + '_ {
        self.content.iter().filter_map(|c| match c {
            NodeElementTypeContent::Matrix(m) => Some(m),
            _ => None,
        })
    }
    pub fn translates(&self) -> impl Iterator<Item = &TargetableFloat3Type> + '_ {
        self.content.iter().filter_map(|c| match c {
            NodeElementTypeContent::Translate(t) => Some(t),
            _ => None,
        })
    }
    pub fn rotates(&self) -> impl Iterator<Item = &RotateElementType> + '_ {
        self.content.iter().filter_map(|c| match c {
            NodeElementTypeContent::Rotate(r) => Some(r),
            _ => None,
        })
    }
    pub fn scales(&self) -> impl Iterator<Item = &TargetableFloat3Type> + '_ {
        self.content.iter().filter_map(|c| match c {
            NodeElementTypeContent::Scale(s) => Some(s),
            _ => None,
        })
    }
    pub fn instance_geometries(&self) -> impl Iterator<Item = &InstanceGeometryElementType> + '_ {
        self.content.iter().filter_map(|c| match c {
            NodeElementTypeContent::InstanceGeometry(ig) => Some(ig),
            _ => None,
        })
    }
    pub fn instance_controllers(&self) -> impl Iterator<Item = &InstanceControllerElementType> + '_ {
        self.content.iter().filter_map(|c| match c {
            NodeElementTypeContent::InstanceController(ic) => Some(ic),
            _ => None,
        })
    }
    pub fn child_nodes(&self) -> impl Iterator<Item = &NodeElementType> + '_ {
        self.content.iter().filter_map(|c| match c {
            NodeElementTypeContent::Node(n) => Some(n),
            _ => None,
        })
    }
}

impl GeometryElementType {
    pub fn mesh(&self) -> Option<&MeshElementType> {
        self.content.iter().find_map(|c| match c {
            GeometryElementTypeContent::Mesh(m) => Some(m),
            _ => None,
        })
    }
}

impl ImageElementType {
    pub fn init_from(&self) -> Option<&str> {
        self.content.iter().find_map(|c| match c {
            ImageElementTypeContent::InitFrom(uri) => Some(uri.as_str()),
            _ => None,
        })
    }
}

impl PolygonsElementType {
    pub fn inputs(&self) -> impl Iterator<Item = &InputLocalOffsetType> + '_ {
        self.content.iter().filter_map(|c| match c {
            PolygonsElementTypeContent::Input(i) => Some(i),
            _ => None,
        })
    }
    pub fn ps(&self) -> impl Iterator<Item = &ListOfUIntsType> + '_ {
        self.content.iter().filter_map(|c| match c {
            PolygonsElementTypeContent::P(p) => Some(p),
            _ => None,
        })
    }
}
"#;
