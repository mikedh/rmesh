// COLLADA 1.4.1 schema bindings
// Generated from XSD

use serde::{Deserialize, Serialize};
use quick_xml::{de, se};

pub mod collada_1_4_1 {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Collada {
        // Root COLLADA element for version 1.4.1
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceWithExtra {
        pub attr_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSamplerRect {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSurfaceInitFromCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TargetableFloat {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSampler1DCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlSampler1D {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSampler3D {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InputLocalOffset {
        pub attr_offset: Uint,
        pub attr_semantic: String,
        pub attr_source: UriFragmentType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_set: Option<Uint>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxClearstencilCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InputGlobal {
        pub attr_semantic: String,
        pub attr_source: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSetparamSimple {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesNewparam {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotate: Option<Vec<FxAnnotateCommon>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub semantic: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub modifier: Option<FxModifierEnumCommon>,
        pub attr_sid: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSetarrayType {
        pub array: CgSetarrayType,
        pub usertype: CgSetuserType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_length: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlSamplerRect {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlSampler3D {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSamplerCubeCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSurfaceInitCubeCommon {
        pub all: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexcombinerArgumentAlphaType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgConnectParam {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgNewparam {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotate: Option<Vec<FxAnnotateCommon>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub semantic: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub modifier: Option<FxModifierEnumCommon>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlslSetparam {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSurfaceFormatHintCommon {
        pub channels: FxSurfaceFormatHintChannelsEnum,
        pub range: FxSurfaceFormatHintRangeEnum,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub precision: Option<FxSurfaceFormatHintPrecisionEnum>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub option: Option<Vec<FxSurfaceFormatHintOptionEnum>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSampler2DCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTextureConstantType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexcombinerArgumentRgbType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlslNewarrayType {
        pub array: GlslNewarrayType,
        pub attr_length: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSurfaceInitVolumeCommon {
        pub all: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSampler3DCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlslSurfaceType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub generator: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotate: Option<Vec<FxAnnotateCommon>>,
        pub code: FxCodeProfile,
        pub include: FxIncludeCommon,
        pub name: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxClearcolorCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxCleardepthCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxAnnotateCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxCodeProfile {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommonColorOrTextureType {
        pub color: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxIncludeCommon {
        pub attr_sid: String,
        pub attr_url: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommonTransparentType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSetparam {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSamplerDepthCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlSampler2D {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTextureUnit {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub texcoord: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSurfaceType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub generator: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotate: Option<Vec<FxAnnotateCommon>>,
        pub code: FxCodeProfile,
        pub include: FxIncludeCommon,
        pub name: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSurfaceInitPlanarCommon {
        pub all: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSampler1D {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxColortargetCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSamplerCube {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TargetableFloat3 {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSamplerRectCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommonNewparamType {
        pub attr_sid: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlslSetarrayType {
        pub array: GlslSetarrayType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_length: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlslNewparam {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSetuserType {
        pub attr_source: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSurfaceCommon {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub format: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub format_hint: Option<FxSurfaceFormatHintCommon>,
        pub size: Int3,
        pub viewport_ratio: Float2,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mip_levels: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mipmap_generate: Option<bool>,
        pub attr_type: FxSurfaceTypeEnum,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexcombinerCommandType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSampler2D {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommonFloatOrParamType {
        pub float: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexenvCommandType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexcombinerCommandRgbType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexturePipeline {
        pub texcombiner: GlesTexcombinerCommandType,
        pub texenv: GlesTexenvCommandType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlSamplerDepth {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxDepthtargetCommon {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesSamplerState {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InputLocal {
        pub attr_semantic: String,
        pub attr_source: UriFragmentType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlslSetparamSimple {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSamplerDepth {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxNewparamCommon {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotate: Option<Vec<FxAnnotateCommon>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub semantic: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub modifier: Option<FxModifierEnumCommon>,
        pub attr_sid: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexcombinerCommandAlphaType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlSamplerCube {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgNewarrayType {
        pub array: CgNewarrayType,
        pub usertype: CgSetuserType,
        pub attr_length: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxStenciltargetCommon {
    }

}
