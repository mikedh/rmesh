//! Collada 1.4.1 schema types for geometry loading.
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

pub type Mesh = MeshElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct MeshElementType {
    #[serde(rename = "$value")]
    pub content: Vec<MeshElementTypeContent>,
}
#[derive(Debug, Deserialize, Serialize)]
pub enum MeshElementTypeContent {
    #[serde(rename = "source")]
    Source(Source),
    #[serde(rename = "vertices")]
    Vertices(Vertices),
    #[serde(rename = "lines")]
    Lines(Lines),
    #[serde(rename = "linestrips")]
    Linestrips(Linestrips),
    #[serde(rename = "polygons")]
    Polygons(Polygons),
    #[serde(rename = "polylist")]
    Polylist(Polylist),
    #[serde(rename = "triangles")]
    Triangles(Triangles),
    #[serde(rename = "trifans")]
    Trifans(Trifans),
    #[serde(rename = "tristrips")]
    Tristrips(Tristrips),
    #[serde(rename = "extra")]
    Extra(Extra),
}
pub type Source = SourceElementType;
pub type Vertices = VerticesElementType;
pub type Lines = LinesElementType;
pub type Linestrips = LinestripsElementType;
pub type Polygons = PolygonsElementType;
pub type Polylist = PolylistElementType;
pub type Triangles = TrianglesElementType;
pub type Trifans = TrifansElementType;
pub type Tristrips = TristripsElementType;
pub type Extra = ExtraElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct SourceElementType {
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, rename = "$value")]
    pub content: Vec<SourceElementTypeContent>,
}
#[derive(Debug, Deserialize, Serialize)]
pub enum SourceElementTypeContent {
    #[serde(rename = "asset")]
    Asset(Asset),
    #[serde(rename = "IDREF_array")]
    IdrefArray(IdrefArray),
    #[serde(rename = "Name_array")]
    NameArray(NameArray),
    #[serde(rename = "bool_array")]
    BoolArray(BoolArray),
    #[serde(rename = "float_array")]
    FloatArray(FloatArray),
    #[serde(rename = "int_array")]
    IntArray(IntArray),
    #[serde(rename = "technique_common")]
    TechniqueCommon(SourceTechniqueCommon),
    #[serde(rename = "technique")]
    Technique(Technique),
}
#[derive(Debug, Deserialize, Serialize)]
pub struct VerticesElementType {
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub input: Vec<InputLocalType>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct LinesElementType {
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(default, rename = "@material")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default)]
    pub input: Vec<InputLocalOffsetType>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p: Option<P>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct LinestripsElementType {
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(default, rename = "@material")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default)]
    pub input: Vec<InputLocalOffsetType>,
    #[serde(default)]
    pub p: Vec<P>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct PolygonsElementType {
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(default, rename = "@material")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default, rename = "$value")]
    pub content: Vec<PolygonsElementTypeContent>,
}
#[derive(Debug, Deserialize, Serialize)]
pub enum PolygonsElementTypeContent {
    #[serde(rename = "input")]
    Input(InputLocalOffsetType),
    #[serde(rename = "p")]
    P(P),
    #[serde(rename = "ph")]
    Ph(PolygonsPh),
    #[serde(rename = "extra")]
    Extra(Extra),
}
#[derive(Debug, Deserialize, Serialize)]
pub struct PolylistElementType {
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(default, rename = "@material")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default)]
    pub input: Vec<InputLocalOffsetType>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vcount: Option<ListOfUIntsType>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p: Option<P>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct TrianglesElementType {
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(default, rename = "@material")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default)]
    pub input: Vec<InputLocalOffsetType>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p: Option<P>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct TrifansElementType {
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(default, rename = "@material")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default)]
    pub input: Vec<InputLocalOffsetType>,
    #[serde(default)]
    pub p: Vec<P>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct TristripsElementType {
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(default, rename = "@material")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default)]
    pub input: Vec<InputLocalOffsetType>,
    #[serde(default)]
    pub p: Vec<P>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct ExtraElementType {
    #[serde(default, rename = "@id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, rename = "@type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asset: Option<Asset>,
    #[serde(default)]
    pub technique: Vec<Technique>,
}
pub type Asset = AssetElementType;
pub type IdrefArray = IdrefArrayElementType;
pub type NameArray = NameArrayElementType;
pub type BoolArray = BoolArrayElementType;
pub type FloatArray = FloatArrayElementType;
pub type IntArray = IntArrayElementType;
pub type SourceTechniqueCommon = SourceTechniqueCommonElementType;
pub type Technique = TechniqueElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct InputLocalType {
    #[serde(rename = "@semantic")]
    pub semantic: String,
    #[serde(rename = "@source")]
    pub source: String,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputLocalOffsetType {
    #[serde(rename = "@offset")]
    pub offset: u64,
    #[serde(rename = "@semantic")]
    pub semantic: String,
    #[serde(rename = "@source")]
    pub source: String,
    #[serde(default, rename = "@set")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub set: Option<u64>,
}
pub type P = ListOfUIntsType;
pub type PolygonsPh = PolygonsPhElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct AssetElementType {
    #[serde(default)]
    pub contributor: Vec<AssetContributor>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keywords: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<AssetUnit>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub up_axis: Option<UpAxisType>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct IdrefArrayElementType {
    #[serde(default, rename = "@id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(rename = "$text")]
    pub content: IdrefsType,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct NameArrayElementType {
    #[serde(default, rename = "@id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(rename = "$text")]
    pub content: ListOfNamesType,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct BoolArrayElementType {
    #[serde(default, rename = "@id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(rename = "$text")]
    pub content: ListOfBoolsType,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct FloatArrayElementType {
    #[serde(default, rename = "@id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(default = "FloatArrayElementType::default_digits", rename = "@digits")]
    pub digits: i16,
    #[serde(default = "FloatArrayElementType::default_magnitude", rename = "@magnitude")]
    pub magnitude: i16,
    #[serde(rename = "$text")]
    pub content: ListOfFloatsType,
}
impl FloatArrayElementType {
    #[must_use]
    pub fn default_digits() -> i16 {
        6i16
    }
    #[must_use]
    pub fn default_magnitude() -> i16 {
        38i16
    }
}
#[derive(Debug, Deserialize, Serialize)]
pub struct IntArrayElementType {
    #[serde(default, rename = "@id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(
        default = "IntArrayElementType::default_min_inclusive",
        rename = "@minInclusive"
    )]
    pub min_inclusive: i32,
    #[serde(
        default = "IntArrayElementType::default_max_inclusive",
        rename = "@maxInclusive"
    )]
    pub max_inclusive: i32,
    #[serde(rename = "$text")]
    pub content: ListOfIntsType,
}
impl IntArrayElementType {
    #[must_use]
    pub fn default_min_inclusive() -> i32 {
        -2147483648i32
    }
    #[must_use]
    pub fn default_max_inclusive() -> i32 {
        2147483647i32
    }
}
#[derive(Debug, Deserialize, Serialize)]
pub struct SourceTechniqueCommonElementType {
    pub accessor: Accessor,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct TechniqueElementType {
    #[serde(rename = "@profile")]
    pub profile: String,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct PolygonsPhElementType {
    pub p: P,
    #[serde(default)]
    pub h: Vec<ListOfUIntsType>,
}
pub type AssetContributor = AssetContributorElementType;
pub type AssetUnit = AssetUnitElementType;
#[derive(Debug, Deserialize, Serialize)]
pub enum UpAxisType {
    #[serde(rename = "X_UP")]
    XUp,
    #[serde(rename = "Y_UP")]
    YUp,
    #[serde(rename = "Z_UP")]
    ZUp,
}
pub type Accessor = AccessorElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct AssetContributorElementType {
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authoring_tool: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comments: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub copyright: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_data: Option<String>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct AssetUnitElementType {
    #[serde(default = "AssetUnitElementType::default_meter", rename = "@meter")]
    pub meter: f64,
    #[serde(default = "AssetUnitElementType::default_name", rename = "@name")]
    pub name: String,
}
impl AssetUnitElementType {
    #[must_use]
    pub fn default_meter() -> f64 {
        1f64
    }
    #[must_use]
    pub fn default_name() -> String {
        String::from("meter")
    }
}
#[derive(Debug, Deserialize, Serialize)]
pub struct AccessorElementType {
    #[serde(rename = "@count")]
    pub count: u64,
    #[serde(default = "AccessorElementType::default_offset", rename = "@offset")]
    pub offset: u64,
    #[serde(default, rename = "@source")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default = "AccessorElementType::default_stride", rename = "@stride")]
    pub stride: u64,
    #[serde(default)]
    pub param: Vec<Param>,
}
impl AccessorElementType {
    #[must_use]
    pub fn default_offset() -> u64 {
        0u64
    }
    #[must_use]
    pub fn default_stride() -> u64 {
        1u64
    }
}
pub type Param = ParamElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct ParamElementType {
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(default, rename = "@semantic")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic: Option<String>,
    #[serde(rename = "@type")]
    pub type_: String,
    #[serde(default, rename = "$text")]
    pub content: String,
}
pub type VisualScene = VisualSceneElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct VisualSceneElementType {
    #[serde(default, rename = "@id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asset: Option<Asset>,
    #[serde(default)]
    pub node: Vec<Node>,
    #[serde(default)]
    pub evaluate_scene: Vec<VisualSceneEvaluateScene>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
pub type Node = NodeElementType;
pub type VisualSceneEvaluateScene = VisualSceneEvaluateSceneElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct NodeElementType {
    #[serde(default, rename = "@id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(default = "NodeElementType::default_type_", rename = "@type")]
    pub type_: NodeType,
    #[serde(default, rename = "@layer")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<ListOfNamesType>,
    #[serde(default, rename = "$value")]
    pub content: Vec<NodeElementTypeContent>,
}
#[derive(Debug, Deserialize, Serialize)]
pub enum NodeElementTypeContent {
    #[serde(rename = "asset")]
    Asset(Asset),
    #[serde(rename = "lookat")]
    Lookat(Lookat),
    #[serde(rename = "matrix")]
    Matrix(Matrix),
    #[serde(rename = "rotate")]
    Rotate(Rotate),
    #[serde(rename = "scale")]
    Scale(Scale),
    #[serde(rename = "skew")]
    Skew(Skew),
    #[serde(rename = "translate")]
    Translate(Translate),
    #[serde(rename = "instance_camera")]
    InstanceCamera(InstanceCamera),
    #[serde(rename = "instance_controller")]
    InstanceController(InstanceController),
    #[serde(rename = "instance_geometry")]
    InstanceGeometry(InstanceGeometry),
    #[serde(rename = "instance_light")]
    InstanceLight(InstanceLight),
    #[serde(rename = "instance_node")]
    InstanceNode(InstanceNode),
    #[serde(rename = "node")]
    Node(Node),
    #[serde(rename = "extra")]
    Extra(Extra),
}
impl NodeElementType {
    #[must_use]
    pub fn default_type_() -> NodeType {
        NodeType::Node
    }
}
#[derive(Debug, Deserialize, Serialize)]
pub struct VisualSceneEvaluateSceneElementType {
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub render: Vec<VisualSceneEvaluateSceneRender>,
}
#[derive(Debug, Deserialize, Serialize)]
pub enum NodeType {
    #[serde(rename = "JOINT")]
    Joint,
    #[serde(rename = "NODE")]
    Node,
}
pub type Lookat = LookatElementType;
pub type Matrix = MatrixElementType;
pub type Rotate = RotateElementType;
pub type Scale = TargetableFloat3Type;
pub type Skew = SkewElementType;
pub type Translate = TargetableFloat3Type;
pub type InstanceCamera = InstanceWithExtraType;
pub type InstanceController = InstanceControllerElementType;
pub type InstanceGeometry = InstanceGeometryElementType;
pub type InstanceLight = InstanceWithExtraType;
pub type InstanceNode = InstanceWithExtraType;
pub type VisualSceneEvaluateSceneRender = VisualSceneEvaluateSceneRenderElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct LookatElementType {
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(rename = "$text")]
    pub content: Float3X3Type,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct MatrixElementType {
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(rename = "$text")]
    pub content: Float4X4Type,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct RotateElementType {
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(rename = "$text")]
    pub content: Float4Type,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct TargetableFloat3Type {
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(rename = "$text")]
    pub content: Float3Type,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct SkewElementType {
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(rename = "$text")]
    pub content: Float7Type,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct InstanceWithExtraType {
    #[serde(rename = "@url")]
    pub url: String,
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct InstanceControllerElementType {
    #[serde(rename = "@url")]
    pub url: String,
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub skeleton: Vec<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bind_material: Option<BindMaterial>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct InstanceGeometryElementType {
    #[serde(rename = "@url")]
    pub url: String,
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bind_material: Option<BindMaterial>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct VisualSceneEvaluateSceneRenderElementType {
    #[serde(rename = "@camera_node")]
    pub camera_node: String,
    #[serde(default)]
    pub layer: Vec<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_effect: Option<InstanceEffect>,
}
pub type BindMaterial = BindMaterialElementType;
pub type InstanceEffect = InstanceEffectElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct BindMaterialElementType {
    #[serde(default)]
    pub param: Vec<Param>,
    pub technique_common: BindMaterialTechniqueCommon,
    #[serde(default)]
    pub technique: Vec<Technique>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct InstanceEffectElementType {
    #[serde(rename = "@url")]
    pub url: String,
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
pub type BindMaterialTechniqueCommon = BindMaterialTechniqueCommonElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct BindMaterialTechniqueCommonElementType {
    #[serde(default)]
    pub instance_material: Vec<InstanceMaterial>,
}
pub type InstanceMaterial = InstanceMaterialElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct InstanceMaterialElementType {
    #[serde(rename = "@symbol")]
    pub symbol: String,
    #[serde(rename = "@target")]
    pub target: String,
    #[serde(default, rename = "@sid")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub bind: Vec<InstanceMaterialBind>,
    #[serde(default)]
    pub bind_vertex_input: Vec<InstanceMaterialBindVertexInput>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
pub type InstanceMaterialBind = InstanceMaterialBindElementType;
pub type InstanceMaterialBindVertexInput = InstanceMaterialBindVertexInputElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct InstanceMaterialBindElementType {
    #[serde(rename = "@semantic")]
    pub semantic: String,
    #[serde(rename = "@target")]
    pub target: String,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct InstanceMaterialBindVertexInputElementType {
    #[serde(rename = "@semantic")]
    pub semantic: String,
    #[serde(rename = "@input_semantic")]
    pub input_semantic: String,
    #[serde(default, rename = "@input_set")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_set: Option<u64>,
}
pub type Material = MaterialElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct MaterialElementType {
    #[serde(default, rename = "@id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asset: Option<Asset>,
    pub instance_effect: InstanceEffect,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
pub type Image = ImageElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct ImageElementType {
    #[serde(default, rename = "@id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, rename = "@format")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default, rename = "@height")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u64>,
    #[serde(default, rename = "@width")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u64>,
    #[serde(default = "ImageElementType::default_depth", rename = "@depth")]
    pub depth: u64,
    #[serde(rename = "$value")]
    pub content: Vec<ImageElementTypeContent>,
}
#[derive(Debug, Deserialize, Serialize)]
pub enum ImageElementTypeContent {
    #[serde(rename = "asset")]
    Asset(Asset),
    #[serde(rename = "data")]
    Data(ListOfHexBinaryType),
    #[serde(rename = "init_from")]
    InitFrom(String),
    #[serde(rename = "extra")]
    Extra(Extra),
}
impl ImageElementType {
    #[must_use]
    pub fn default_depth() -> u64 {
        1u64
    }
}
pub type Geometry = GeometryElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct GeometryElementType {
    #[serde(default, rename = "@id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "@name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "$value")]
    pub content: Vec<GeometryElementTypeContent>,
}
#[derive(Debug, Deserialize, Serialize)]
pub enum GeometryElementTypeContent {
    #[serde(rename = "asset")]
    Asset(Asset),
    #[serde(rename = "convex_mesh")]
    ConvexMesh(ConvexMesh),
    #[serde(rename = "mesh")]
    Mesh(Mesh),
    #[serde(rename = "spline")]
    Spline(Spline),
    #[serde(rename = "extra")]
    Extra(Extra),
}
pub type ConvexMesh = ConvexMeshElementType;
pub type Spline = SplineElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct ConvexMeshElementType {
    #[serde(default, rename = "@convex_hull_of")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub convex_hull_of: Option<String>,
    #[serde(default, rename = "$value")]
    pub content: Vec<ConvexMeshElementTypeContent>,
}
#[derive(Debug, Deserialize, Serialize)]
pub enum ConvexMeshElementTypeContent {
    #[serde(rename = "source")]
    Source(Source),
    #[serde(rename = "vertices")]
    Vertices(Vertices),
    #[serde(rename = "lines")]
    Lines(Lines),
    #[serde(rename = "linestrips")]
    Linestrips(Linestrips),
    #[serde(rename = "polygons")]
    Polygons(Polygons),
    #[serde(rename = "polylist")]
    Polylist(Polylist),
    #[serde(rename = "triangles")]
    Triangles(Triangles),
    #[serde(rename = "trifans")]
    Trifans(Trifans),
    #[serde(rename = "tristrips")]
    Tristrips(Tristrips),
    #[serde(rename = "extra")]
    Extra(Extra),
}
#[derive(Debug, Deserialize, Serialize)]
pub struct SplineElementType {
    #[serde(default = "SplineElementType::default_closed", rename = "@closed")]
    pub closed: bool,
    #[serde(default)]
    pub source: Vec<Source>,
    pub control_vertices: SplineControlVertices,
    #[serde(default)]
    pub extra: Vec<Extra>,
}
impl SplineElementType {
    #[must_use]
    pub fn default_closed() -> bool {
        false
    }
}
pub type SplineControlVertices = SplineControlVerticesElementType;
#[derive(Debug, Deserialize, Serialize)]
pub struct SplineControlVerticesElementType {
    #[serde(default)]
    pub input: Vec<InputLocalType>,
    #[serde(default)]
    pub extra: Vec<Extra>,
}


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
