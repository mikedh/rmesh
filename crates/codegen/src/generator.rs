use crate::parse::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// Entities that get full typed structs in the generated code.
/// Everything else is parsed as `Entity::Generic(&str)`.
///
/// To add a new typed entity:
/// 1. Add its EXPRESS name (lowercase) to this list
/// 2. Run codegen to regenerate ap214.rs
/// 3. Use it in mod.rs via `ap214::Entity::NewEntity(e) => { ... }`
const TYPED_ENTITIES: &[&str] = &[
    // Geometry
    "cartesian_point",
    "direction",
    "vector",
    "axis2_placement_3d",
    // Curves
    "line",
    "circle",
    "ellipse",
    "b_spline_curve",
    "b_spline_curve_with_knots",
    "rational_b_spline_curve",
    // Surfaces
    "plane",
    "cylindrical_surface",
    "conical_surface",
    "spherical_surface",
    "toroidal_surface",
    "b_spline_surface_with_knots",
    "rational_b_spline_surface",
    // Topology
    "vertex_point",
    "edge_curve",
    "oriented_edge",
    "edge_loop",
    "face_bound",
    "face_outer_bound",
    "advanced_face",
    "closed_shell",
    "manifold_solid_brep",
    // Representations
    "shape_representation",
    "advanced_brep_shape_representation",
    // Assembly / transforms
    "representation_relationship_with_transformation",
    "shape_representation_relationship",
    "item_defined_transformation",
    // Units
    "si_unit",
    "conversion_based_unit",
    "length_unit",
];

////////////////////////////////////////////////////////////////////////////////
// Helper types to use when doing code-gen
#[derive(Debug)]
enum Type<'a> {
    Entity {
        // In order, with parent attributes first
        attrs: Vec<AttributeData<'a>>,
        supertypes: Vec<&'a str>,
    },
    // These are all TYPE in EXPRESS, but we unpack them here
    Redeclared(&'a str),
    RedeclaredPrimitive(&'a str),
    Enum(Vec<&'a str>),
    Select(Vec<&'a str>),
    Aggregation {
        optional: bool,
        type_: Box<Type<'a>>,
    },

    // Direct Rust type
    Primitive(&'a str),
}
struct TypeMap<'a>(HashMap<&'a str, Type<'a>>, &'a HashMap<&'a str, Ref<'a>>);
impl<'a> TypeMap<'a> {
    fn to_rtype_build(&mut self, s: &'a str) -> String {
        if !self.0.contains_key(s) {
            self.build(s);
        }
        self.to_rtype(s)
    }
    fn is_entity(&self, s: &str) -> bool {
        let t = self.0.get(s).expect(&format!("Could not get {:?}", s));
        match &t {
            Type::Entity { .. } => true,
            Type::Select(v) => v.iter().all(|s| self.is_entity(s)),
            _ => false,
        }
    }
    fn to_rtype(&self, s: &str) -> String {
        let t = self.0.get(s).expect(&format!("Could not get {:?}", s));
        match &t {
            Type::Entity { .. }
            | Type::Redeclared(_)
            | Type::RedeclaredPrimitive(_)
            | Type::Enum(_)
            | Type::Select(_) => format!("{}<'a>", to_camel(s)),

            Type::Primitive(s) => s.to_string(),

            Type::Aggregation { optional, type_ } => {
                if *optional {
                    format!("Vec<Option<{}>>", self.to_inner_rtype(type_))
                } else {
                    format!("Vec<{}>", self.to_inner_rtype(type_))
                }
            }
        }
    }
    fn to_inner_rtype(&self, t: &Type<'a>) -> String {
        match &t {
            Type::Aggregation { optional, type_ } => {
                if *optional {
                    format!("Vec<Option<{}>>", self.to_inner_rtype(type_))
                } else {
                    format!("Vec<{}>", self.to_inner_rtype(type_))
                }
            }
            Type::Redeclared(r) => {
                format!("{}<'a>", to_camel(r))
            }
            Type::RedeclaredPrimitive(r) => r.to_string(),
            Type::Primitive(r) => r.to_string(),

            Type::Entity { .. } | Type::Enum(_) | Type::Select(_) => panic!("Invalid inner type"),
        }
    }

    /// Find the EXPRESS name (lowercase with underscores) for a CamelCase Rust type name.
    fn find_express_name(&self, camel: &str) -> Option<&'a str> {
        self.0.keys().copied().find(|&k| to_camel(k) == camel)
    }

    /// Resolve a type name to its flattened Rust type for typed entity fields.
    /// Entity refs → usize, measures → f64, labels → &'a str, etc.
    fn to_flat_type(&self, type_str: &str) -> String {
        // Already a primitive
        match type_str {
            "f64" | "i64" | "bool" | "usize" | "Logical" => return type_str.to_string(),
            "&'a str" => return type_str.to_string(),
            _ => {}
        }

        // Handle Vec<T>, Option<T>, ArrayVec<T, N>
        if let Some(inner) = type_str
            .strip_prefix("Vec<")
            .and_then(|s| s.strip_suffix('>'))
        {
            let flat_inner = self.to_flat_type(inner);
            return format!("Vec<{}>", flat_inner);
        }
        if let Some(inner) = type_str
            .strip_prefix("Option<")
            .and_then(|s| s.strip_suffix('>'))
        {
            let flat_inner = self.to_flat_type(inner);
            return format!("Option<{}>", flat_inner);
        }
        if let Some(rest) = type_str.strip_prefix("ArrayVec::<") {
            if let Some(comma_pos) = rest.rfind(',') {
                let inner = &rest[..comma_pos].trim();
                let cap = rest[comma_pos + 1..]
                    .trim()
                    .strip_suffix('>')
                    .unwrap_or("3")
                    .trim();
                let flat_inner = self.to_flat_type(inner);
                return format!("ArrayVec::<{}, {}>", flat_inner, cap);
            }
        }

        // Strip lifetime from type name to look up
        let camel = type_str.strip_suffix("<'a>").unwrap_or(type_str);

        // Resolve CamelCase Rust name back to EXPRESS name for type map lookup
        let lookup = self.find_express_name(camel).unwrap_or(camel);

        // Look up in type map
        if let Some(t) = self.0.get(lookup) {
            match t {
                // Entity ref → usize
                Type::Entity { .. } => return "usize".to_string(),
                // SELECT where all members are entities → usize
                Type::Select(members) if members.iter().all(|m| self.is_entity(m)) => {
                    return "usize".to_string();
                }
                // Redeclared (newtype wrapper) → resolve recursively
                Type::Redeclared(inner) => {
                    let inner_rtype = self.to_rtype(inner);
                    return self.to_flat_type(&inner_rtype);
                }
                // Primitive wrapper → resolve to inner primitive
                Type::RedeclaredPrimitive(prim) => {
                    return self.to_flat_type(prim);
                }
                // Enum → &'a str (parsed as enum tag string)
                Type::Enum(_) => return "&'a str".to_string(),
                // Non-entity SELECT → &'a str (rare)
                Type::Select(_) => return "&'a str".to_string(),
                // Aggregation → resolve inner
                Type::Aggregation { optional, type_ } => {
                    let inner = self.to_flat_inner_type(type_);
                    if *optional {
                        return format!("Vec<Option<{}>>", inner);
                    } else {
                        return format!("Vec<{}>", inner);
                    }
                }
                Type::Primitive(p) => return p.to_string(),
            }
        }

        // Fallback: if it looks like a CamelCase type with lifetime, try as entity ref
        if type_str.ends_with("<'a>") {
            return "usize".to_string();
        }

        type_str.to_string()
    }

    fn to_flat_inner_type(&self, t: &Type<'a>) -> String {
        match t {
            Type::Aggregation { optional, type_ } => {
                let inner = self.to_flat_inner_type(type_);
                if *optional {
                    format!("Vec<Option<{}>>", inner)
                } else {
                    format!("Vec<{}>", inner)
                }
            }
            Type::Redeclared(r) => {
                let rtype = self.to_rtype(r);
                self.to_flat_type(&rtype)
            }
            Type::RedeclaredPrimitive(r) => self.to_flat_type(r),
            Type::Primitive(r) => r.to_string(),
            Type::Entity { .. } => "usize".to_string(),
            Type::Enum(_) => "&'a str".to_string(),
            Type::Select(members) if members.iter().all(|m| self.is_entity(m)) => {
                "usize".to_string()
            }
            Type::Select(_) => "&'a str".to_string(),
        }
    }

    fn build(&mut self, s: &'a str) {
        let v = self.1.get(s).unwrap();
        let m = match v {
            Ref::Entity(e) => e.to_type(self),
            Ref::Type(t) => t.to_type(self),
        };
        self.0.insert(s, m);
    }
    fn attributes(&mut self, s: &'a str) -> Vec<AttributeData<'a>> {
        if !self.0.contains_key(s) {
            self.build(s);
        }
        let t = self.0.get(s).expect(&format!("Could not get {:?}", s));
        if let Type::Entity { attrs, .. } = &t {
            attrs.clone()
        } else {
            panic!("Cannot get attributes of a non-entity");
        }
    }
}

impl<'a> Type<'a> {
    fn is_entity(&self) -> bool {
        matches!(self, Type::Entity { .. })
    }

    fn write_supertypes<W>(&self, name: &str, buf: &mut W) -> std::fmt::Result
    where
        W: std::fmt::Write,
    {
        if let Type::Entity { supertypes, .. } = self {
            if !supertypes.is_empty() {
                write!(buf, r#"        "{}" => &["#, capitalize(name))?;
                for (i, s) in supertypes.iter().enumerate() {
                    if i == supertypes.len() - 1 {
                        writeln!(buf, r#""{}"],"#, capitalize(s))?;
                    } else {
                        write!(buf, r#""{}", "#, capitalize(s))?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct AttributeData<'a> {
    name: &'a str,         // already camel-case
    from: Option<&'a str>, // original class, or None
    type_: String,
    optional: bool,
    dupe: bool,    // inherited from different parents with the same name
    derived: bool, // marked whether this is a derived attribute
}

////////////////////////////////////////////////////////////////////////////////

// A reference into an existing `Syntax` tree, for convenient random access
enum Ref<'a> {
    Entity(&'a EntityDecl<'a>),
    Type(&'a UnderlyingType<'a>),
}

////////////////////////////////////////////////////////////////////////////////

pub fn generate(s: &mut Syntax) -> Result<String, std::fmt::Error> {
    assert!(s.0.len() == 1, "Multiple schemas are unsupported");

    // First pass: collect entity names, then convert ambiguous IDs in SELECT
    // data types into Entity or Type refs
    let mut entity_names = HashSet::new();
    s.collect_entity_names(&mut entity_names);
    s.disambiguate(&entity_names);

    // Build a map from type names to references into `s`
    let mut ref_map = HashMap::new();
    s.build_ref_map(&mut ref_map);

    // Build the type map
    let mut type_map = TypeMap(HashMap::new(), &ref_map);
    type_map.0.insert("usize", Type::Primitive("usize"));
    type_map.0.insert("bool", Type::Primitive("bool"));
    type_map.0.insert("i64", Type::Primitive("i64"));
    type_map.0.insert("f64", Type::Primitive("f64"));
    type_map.0.insert("&'a str", Type::Primitive("&'a str"));

    for k in ref_map.keys() {
        type_map.build(k);
    }

    // Sorted keys for determinism
    let mut keys: Vec<&str> = type_map.0.keys().cloned().collect();
    keys.sort_unstable();

    let mut buf = String::new();

    // ── File header ─────────────────────────────────────────────────
    writeln!(
        &mut buf,
        r#"//! Auto-generated AP214 entity definitions.
//!
//! ## Adding a new entity type
//!
//! 1. Add the entity's EXPRESS name (lowercase) to `TYPED_ENTITIES` in
//!    `crates/codegen/src/generator.rs`.
//! 2. Run: `cargo run -p codegen -- ./reference/APs/10303-214e3-aim-long.exp \
//!         -o crates/rmesh/src/boundary/step/ap214.rs`
//! 3. Use in mod.rs: `ap214::Entity::NewEntity(e) => {{ ... }}`

// Autogenerated file, do not hand-edit!
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::restriction)]
#![allow(clippy::nursery)]

use super::parse::{{
    Derived, IResult, Logical, Parse,
    param_from_chunks, parse_complex_mapping,
}};
use arrayvec::ArrayVec;
use nom::{{
    bytes::complete::tag,
    character::complete::{{alpha0, alphanumeric1}},
    combinator::recognize,
    multi::many0,
    sequence::pair,
}};"#
    )?;

    // ── Typed entity struct definitions ──────────────────────────────
    for entity_name in TYPED_ENTITIES {
        if !type_map.0.contains_key(entity_name) {
            eprintln!(
                "Warning: typed entity '{}' not found in schema",
                entity_name
            );
            continue;
        }
        let t = &type_map.0[entity_name];
        if let Type::Entity { attrs, .. } = t {
            let camel = to_camel(entity_name);

            // Struct definition with flattened types
            writeln!(buf, "\n#[derive(Debug)]")?;
            writeln!(buf, "pub struct {}_<'a> {{", camel)?;
            for a in attrs {
                if a.derived {
                    continue;
                }
                let field_name = if a.dupe {
                    format!("{}__{}", a.from.unwrap(), a.name)
                } else {
                    a.name.to_string()
                };
                let flat_type = type_map.to_flat_type(&a.type_);
                if a.optional {
                    writeln!(buf, "    pub {}: Option<{}>,", field_name, flat_type)?;
                } else {
                    writeln!(buf, "    pub {}: {},", field_name, flat_type)?;
                }
            }
            writeln!(buf, "    _p: std::marker::PhantomData<&'a ()>,")?;
            writeln!(buf, "}}")?;

            // parse_chunks method
            writeln!(buf, "impl<'a> {}_<'a> {{", camel)?;
            writeln!(
                buf,
                "    pub fn parse_chunks(strs: &[&'a str]) -> IResult<'a, Self> {{"
            )?;
            if !attrs.is_empty() {
                writeln!(buf, "        let mut i = 0;")?;
            }
            writeln!(
                buf,
                r#"        let (s, _) = tag("{}(")(strs[0])?;"#,
                capitalize(entity_name)
            )?;

            for (idx, a) in attrs.iter().enumerate() {
                let is_last = idx == attrs.len() - 1;
                if a.derived {
                    // Skip derived attribute (serialized as `*` in STEP)
                    writeln!(
                        buf,
                        "        let (s, _) = param_from_chunks::<Derived>({}, s, &mut i, strs)?;",
                        is_last
                    )?;
                } else {
                    let flat_type = type_map.to_flat_type(&a.type_);
                    let field_name = if a.dupe {
                        format!("{}__{}", a.from.unwrap(), a.name)
                    } else {
                        a.name.to_string()
                    };

                    // Determine the parse type — what parser to call
                    let parse_type = if a.optional {
                        format!("Option<{}>", flat_type)
                    } else {
                        flat_type.clone()
                    };

                    writeln!(
                        buf,
                        "        let (s, {}) = param_from_chunks::<{}>({}, s, &mut i, strs)?;",
                        field_name, parse_type, is_last
                    )?;
                }
            }

            writeln!(buf, "        Ok((s, Self {{")?;
            for a in attrs.iter().filter(|a| !a.derived) {
                let field_name = if a.dupe {
                    format!("{}__{}", a.from.unwrap(), a.name)
                } else {
                    a.name.to_string()
                };
                writeln!(buf, "            {},", field_name)?;
            }
            writeln!(
                buf,
                "            _p: std::marker::PhantomData,\n        }}))\n    }}\n}}"
            )?;
        }
    }

    // ── Entity enum ─────────────────────────────────────────────────
    writeln!(buf, "\n#[derive(Debug)]")?;
    writeln!(buf, "pub enum Entity<'a> {{")?;
    for entity_name in TYPED_ENTITIES {
        if type_map.0.contains_key(entity_name) {
            writeln!(buf, "    {0}({0}_<'a>),", to_camel(entity_name))?;
        }
    }
    writeln!(
        buf,
        "    /// Unparsed entity — tag extracted, payload left as raw str."
    )?;
    writeln!(buf, "    Generic(&'a str),")?;
    writeln!(buf, "    ComplexEntity(Vec<Entity<'a>>),")?;
    writeln!(buf, "    _FailedToParse,")?;
    writeln!(buf, "    _EmptySlot,")?;
    writeln!(buf, "}}")?;

    // ── Parse dispatch ──────────────────────────────────────────────
    writeln!(
        buf,
        r#"
impl<'a> Entity<'a> {{
    pub fn parse_chunks(strs: &[&'a str]) -> IResult<'a, Self> {{
        let (_, r) = recognize(pair(
            nom::branch::alt((alpha0, tag("_"))),
            many0(nom::branch::alt((alphanumeric1, tag("_")))),
        ))(strs[0])?;
        match r {{"#
    )?;
    for entity_name in TYPED_ENTITIES {
        if type_map.0.contains_key(entity_name) {
            writeln!(
                buf,
                r#"            "{0}" => {1}_::parse_chunks(strs).map(|(s, v)| (s, Entity::{1}(v))),"#,
                capitalize(entity_name),
                to_camel(entity_name)
            )?;
        }
    }
    writeln!(
        buf,
        r#"            "" => parse_complex_mapping(strs[0]),
            _ => Ok(("", Entity::Generic(strs[0]))),
        }}
    }}
}}

impl<'a> Parse<'a> for Entity<'a> {{
    fn parse(s: &'a str) -> IResult<'a, Self> {{
        Self::parse_chunks(&[s])
    }}
}}"#
    )?;

    // ── superclasses_of (all entities from schema) ──────────────────
    writeln!(buf, "\npub fn superclasses_of(s: &str) -> &[&str] {{")?;
    writeln!(buf, "    match s {{")?;
    for k in &keys {
        type_map.0[k].write_supertypes(k, &mut buf)?;
    }
    writeln!(buf, "        _ => &[],")?;
    writeln!(buf, "    }}\n}}")?;

    Ok(buf)
}

fn capitalize(s: &str) -> String {
    s.chars()
        .map(|c| c.to_uppercase().next().unwrap())
        .collect()
}

fn to_camel(s: &str) -> String {
    let mut out = String::new();
    let mut cap = true;
    for c in s.chars() {
        if c == '_' {
            cap = true;
        } else if cap {
            out.push(c.to_uppercase().next().unwrap());
            cap = false;
        } else {
            out.push(c);
        }
    }
    out
}

////////////////////////////////////////////////////////////////////////////////

impl<'a> Syntax<'a> {
    fn collect_entity_names(&self, entity_names: &mut HashSet<&'a str>) {
        for v in &self.0 {
            v.collect_entity_names(entity_names);
        }
    }
    fn build_ref_map(&'a self, ref_map: &mut HashMap<&'a str, Ref<'a>>) {
        for v in &self.0 {
            v.build_ref_map(ref_map);
        }
    }
    fn disambiguate(&mut self, entity_names: &HashSet<&str>) {
        for v in &mut self.0 {
            v.disambiguate(entity_names);
        }
    }
}
impl<'a> SchemaDecl<'a> {
    fn collect_entity_names(&self, entity_names: &mut HashSet<&'a str>) {
        self.body.collect_entity_names(entity_names);
    }
    fn build_ref_map(&'a self, ref_map: &mut HashMap<&'a str, Ref<'a>>) {
        self.body.build_ref_map(ref_map);
    }
    fn disambiguate(&mut self, entity_names: &HashSet<&str>) {
        self.body.disambiguate(entity_names)
    }
}
impl<'a> SchemaBody<'a> {
    fn collect_entity_names(&self, entity_names: &mut HashSet<&'a str>) {
        for d in &self.declarations {
            match d {
                DeclarationOrRuleDecl::Declaration(d) => d.collect_entity_names(entity_names),
                DeclarationOrRuleDecl::RuleDecl(_) => (),
            }
        }
    }
    fn build_ref_map(&'a self, ref_map: &mut HashMap<&'a str, Ref<'a>>) {
        for d in &self.declarations {
            match d {
                DeclarationOrRuleDecl::Declaration(d) => d.build_ref_map(ref_map),
                DeclarationOrRuleDecl::RuleDecl(_) => (),
            }
        }
    }
    fn disambiguate(&mut self, entity_names: &HashSet<&str>) {
        for d in &mut self.declarations {
            match d {
                DeclarationOrRuleDecl::Declaration(d) => d.disambiguate(entity_names),
                DeclarationOrRuleDecl::RuleDecl(_) => (),
            }
        }
    }
}
impl<'a> Declaration<'a> {
    fn collect_entity_names(&self, entity_names: &mut HashSet<&'a str>) {
        if let Declaration::Entity(d) = self {
            entity_names.insert(d.0.0.0);
        }
    }
    fn disambiguate(&mut self, entity_names: &HashSet<&str>) {
        if let Declaration::Type(d) = self {
            d.disambiguate(entity_names);
        }
    }
    fn build_ref_map(&'a self, ref_map: &mut HashMap<&'a str, Ref<'a>>) {
        match self {
            Declaration::Entity(d) => {
                ref_map.insert(d.0.0.0, Ref::Entity(d));
            }
            Declaration::Type(d) => {
                ref_map.insert(d.type_id.0, Ref::Type(&d.underlying_type));
            }
            _ => (),
        }
    }
}
impl<'a> TypeDecl<'a> {
    fn disambiguate(&mut self, entity_names: &HashSet<&str>) {
        match &mut self.underlying_type {
            UnderlyingType::Constructed(c) => {
                c.disambiguate(entity_names);
            }
            _ => (),
        }
    }
}
impl<'a> UnderlyingType<'a> {
    fn to_type(&'a self, type_map: &mut TypeMap<'a>) -> Type {
        match self {
            UnderlyingType::Concrete(c) => c.to_type(type_map),
            UnderlyingType::Constructed(c) => c.to_type(),
        }
    }
}
impl<'a> ConcreteTypes<'a> {
    fn to_type(&self, type_map: &mut TypeMap<'a>) -> Type {
        match self {
            ConcreteTypes::Aggregation(a) => a.to_type(type_map),
            ConcreteTypes::Simple(s) => s.to_type(),
            ConcreteTypes::TypeRef(t) => Type::Redeclared(t.0),
        }
    }
}
impl<'a> SimpleExpression<'a> {
    fn to_value(&self) -> Option<usize> {
        if !self.1.is_empty() {
            return None;
        }
        let term = &self.0;
        if !term.1.is_empty() {
            return None;
        }
        let factor = &term.0;
        if !factor.1.is_none() {
            return None;
        }
        let simple_factor = &factor.0;
        let exp = if let SimpleFactor::Unary(op, exp) = simple_factor {
            if op.is_some() {
                return None;
            } else {
                exp
            }
        } else {
            return None;
        };

        let primary = if let ExpressionOrPrimary::Primary(p) = exp {
            p
        } else {
            return None;
        };
        let literal = if let Primary::Literal(lit) = primary {
            lit
        } else {
            return None;
        };

        if let Literal::Real(f) = literal {
            if f.fract() == 0.0 {
                Some(*f as usize)
            } else {
                None
            }
        } else {
            None
        }
    }
}
impl<'a> AggregationTypes<'a> {
    fn to_type(&self, type_map: &mut TypeMap<'a>) -> Type {
        let (optional, instantiable) = match self {
            AggregationTypes::Array(a) => (a.optional, &a.instantiable_type),
            AggregationTypes::Bag(a) => (false, &a.1),
            AggregationTypes::List(a) => (false, &a.instantiable_type),
            AggregationTypes::Set(a) => (false, &a.instantiable_type),
        };
        match &**instantiable {
            InstantiableType::Concrete(c) => {
                let type_ = c.to_type(type_map);
                Type::Aggregation {
                    optional,
                    type_: Box::new(type_),
                }
            }
            InstantiableType::EntityRef(e) => Type::Aggregation {
                optional,
                type_: Box::new(Type::Redeclared(e.0)),
            },
        }
    }
}
impl<'a> ConstructedTypes<'a> {
    fn to_type(&'a self) -> Type {
        match self {
            ConstructedTypes::Enumeration(e) => e.to_type(),
            ConstructedTypes::Select(s) => s.to_type(),
        }
    }
}
impl<'a> EnumerationType<'a> {
    fn to_type(&self) -> Type {
        assert!(
            !self.extensible,
            "Extensible enumerations are not supported"
        );
        match self.items_or_extension.as_ref().unwrap() {
            EnumerationItemsOrExtension::Items(e) => e.to_type(),
            _ => panic!("Extensions not supported"),
        }
    }
}
impl<'a> EnumerationItems<'a> {
    fn to_type(&self) -> Type {
        let mut out = Vec::new();
        for e in &self.0 {
            out.push(e.0);
        }
        Type::Enum(out)
    }
}
impl<'a> SelectType<'a> {
    fn to_type(&'a self) -> Type {
        assert!(!self.extensible, "Cannot handle extensible lists");
        assert!(!self.generic_entity, "Cannot handle generic entity lists");
        match &self.list_or_extension {
            SelectListOrExtension::List(e) => e.to_type(),
            _ => panic!("Extensions not supported"),
        }
    }
}
impl<'a> SelectList<'a> {
    fn to_type(&'a self) -> Type {
        let mut out = Vec::new();
        for e in &self.0 {
            out.push(e.name());
        }
        Type::Select(out)
    }
}
impl<'a> EntityDecl<'a> {
    fn to_type(&'a self, type_map: &mut TypeMap<'a>) -> Type<'a> {
        let mut derived: HashSet<(&str, &str)> = HashSet::new();
        if let Some(derive) = &self.1.derive {
            for d in &derive.0 {
                match &d.0 {
                    AttributeDecl::Redeclared(r) => {
                        assert!(r.1.is_none());
                        derived.insert((r.0.0.0.0, r.0.1.0.0));
                    }
                    AttributeDecl::Id(_) => continue,
                }
            }
        }

        let mut seen: HashSet<(&str, &str)> = HashSet::new();

        let subsuper = &self.0.1;
        let mut inherited_name_count: HashMap<&str, usize> = HashMap::new();
        if let Some(subs) = &subsuper.1 {
            for sub in &subs.0 {
                for a in type_map.attributes(sub.0) {
                    *inherited_name_count.entry(a.name).or_insert(0) += 1;
                }
            }
        }
        let inherited_names: HashSet<&str> = inherited_name_count
            .into_iter()
            .filter(|a| a.1 > 1)
            .map(|a| a.0)
            .collect();

        let mut attrs = Vec::new();
        let mut supertypes = Vec::new();
        if let Some(subs) = &subsuper.1 {
            for sub in subs.0.iter() {
                supertypes.push(sub.0);

                attrs.extend(
                    type_map
                        .attributes(sub.0)
                        .into_iter()
                        .map(|mut a| {
                            if a.from.is_none() {
                                a.from = Some(sub.0);
                            }
                            AttributeData {
                                dupe: inherited_names.contains(a.name),
                                derived: derived.contains(&(a.from.unwrap(), a.name)),
                                ..a
                            }
                        })
                        .filter(|a| seen.insert((a.from.unwrap(), a.name))),
                );
            }
        }

        for attr in &self.1.explicit_attr {
            let attr_type = attr.parameter_type.to_attr_type_str(type_map);
            for a in &attr.attributes {
                if a.is_redeclared() {
                    continue;
                }
                attrs.push(AttributeData {
                    name: a.name(),
                    from: None,
                    dupe: false,
                    derived: false,
                    type_: attr_type.clone(),
                    optional: attr.optional,
                });
            }
        }
        Type::Entity { attrs, supertypes }
    }
}
impl<'a> AttributeDecl<'a> {
    fn name(&self) -> &str {
        match self {
            AttributeDecl::Id(i) => i.0,
            AttributeDecl::Redeclared(_) => panic!("No support for renamed attributes"),
        }
    }
    fn is_redeclared(&self) -> bool {
        match self {
            AttributeDecl::Id(_) => false,
            AttributeDecl::Redeclared(_) => true,
        }
    }
}
impl<'a> GeneralizedTypes<'a> {
    fn to_attr_type_str(&'a self, type_map: &mut TypeMap<'a>) -> String {
        match self {
            GeneralizedTypes::Aggregate(_) => panic!("No support for aggregate type"),
            GeneralizedTypes::GeneralAggregation(a) => a.to_attr_type_str(type_map),
            GeneralizedTypes::GenericEntity(_) => panic!("No support for generic entity type"),
            GeneralizedTypes::Generic(_) => panic!("No support for generic generalized type"),
        }
    }
}
impl<'a> ParameterType<'a> {
    fn to_attr_type_str(&'a self, type_map: &mut TypeMap<'a>) -> String {
        match self {
            ParameterType::Generalized(g) => g.to_attr_type_str(type_map),
            ParameterType::Named(e) => type_map.to_rtype_build(e.name()),
            ParameterType::Simple(e) => e.to_attr_type_str().to_owned(),
        }
    }
}
impl<'a> GeneralAggregationTypes<'a> {
    fn upper_bound(&self) -> Option<usize> {
        let upper: Option<&Bound2> = match &self {
            GeneralAggregationTypes::Array(a) => Some(&a.bounds.1),
            GeneralAggregationTypes::Bag(_) => None,
            GeneralAggregationTypes::List(a) => a.bounds.as_ref().map(|b| &b.1),
            GeneralAggregationTypes::Set(a) => a.bounds.as_ref().map(|b| &b.1),
        };
        upper.and_then(|v| v.0.0.to_value())
    }
    fn to_attr_type_str(&'a self, type_map: &mut TypeMap<'a>) -> String {
        let (optional, param_type) = match self {
            GeneralAggregationTypes::Array(a) => (a.optional, &a.parameter_type),
            GeneralAggregationTypes::Bag(a) => (false, &a.1),
            GeneralAggregationTypes::List(a) => (false, &a.parameter_type),
            GeneralAggregationTypes::Set(a) => (false, &a.parameter_type),
        };
        let t = param_type.to_attr_type_str(type_map);
        let vec_type = match self.upper_bound() {
            Some(b) => format!("ArrayVec::<{}, {}>", t, b),
            None => format!("Vec<{}>", t),
        };
        if optional {
            format!("Option<{}>", vec_type)
        } else {
            vec_type
        }
    }
}
impl<'a> SimpleTypes<'a> {
    fn to_attr_type_str(&self) -> &str {
        match self {
            SimpleTypes::Binary(_) => "usize",
            SimpleTypes::Boolean => "bool",
            SimpleTypes::Integer => "i64",
            SimpleTypes::Logical => "Logical",
            SimpleTypes::Number => "f64",
            SimpleTypes::Real(_) => "f64",
            SimpleTypes::String(_) => "&'a str",
        }
    }
    fn to_type(&self) -> Type {
        Type::RedeclaredPrimitive(self.to_attr_type_str())
    }
}
impl<'a> ConstructedTypes<'a> {
    fn disambiguate(&mut self, entity_names: &HashSet<&str>) {
        match self {
            ConstructedTypes::Select(e) => e.disambiguate(entity_names),
            _ => (),
        }
    }
}
impl<'a> SelectType<'a> {
    fn disambiguate(&mut self, entity_names: &HashSet<&str>) {
        match &mut self.list_or_extension {
            SelectListOrExtension::List(items) => {
                for t in &mut items.0 {
                    t.disambiguate(entity_names);
                }
            }
            _ => panic!("Nope nope nope"),
        }
    }
}
impl<'a> NamedTypes<'a> {
    fn disambiguate(&mut self, entity_names: &HashSet<&str>) {
        if let NamedTypes::_Ambiguous(r) = self {
            *self = if entity_names.contains(r.0) {
                NamedTypes::Entity(EntityRef(r.0))
            } else {
                NamedTypes::Type(TypeRef(r.0))
            };
        }
    }
    fn name(&self) -> &str {
        match self {
            NamedTypes::Entity(e) => e.0,
            NamedTypes::Type(e) => e.0,
            NamedTypes::_Ambiguous(e) => e.0,
        }
    }
}
