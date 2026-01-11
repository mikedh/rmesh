// COLLADA 1.5 schema bindings
// Generated from XSD

use serde::{Deserialize, Serialize};
use quick_xml::{de, se};

pub mod collada_1_5 {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Collada {
        // Root COLLADA element for version 1.5
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BrepType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub curves: Option<CurvesType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub surfaces: Option<SurfacesType>,
        pub source: Vec<SourceType>,
        pub vertices: VerticesType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub edges: Option<EdgesType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub wires: Option<WiresType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub faces: Option<FacesType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub shells: Option<ShellsType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub solids: Option<SolidsType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct OrientType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryVisualScenesType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub visual_scene: Vec<VisualSceneType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MotionType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsTechniqueType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PhysicsSceneType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_force_field: Option<Vec<InstanceForceFieldType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_physics_model: Option<Vec<InstancePhysicsModelType>>,
        pub technique_common: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub gravity: Option<TargetableFloat3Type>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time_step: Option<TargetableFloatType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceRigidConstraintType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        pub attr_constraint: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommonSidrefOrParamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RigidBodyType {
        pub technique_common: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub dynamic: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxCommonNewparamType {
        pub attr_sid: SidType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSamplerRectType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexturePipelineType {
        pub texcombiner: GlesTexcombinerCommandType,
        pub texenv: GlesTexenvCommandType,
        pub extra: ExtraType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsFrameType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LightType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub technique_common: String,
        pub ambient: String,
        pub color: TargetableFloat3Type,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Gles2NewparamType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub semantic: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NodeType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub lookat: LookatType,
        pub matrix: MatrixType,
        pub rotate: RotateType,
        pub scale: ScaleType,
        pub skew: SkewType,
        pub translate: TranslateType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_camera: Option<Vec<InstanceCameraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_controller: Option<Vec<InstanceControllerType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_geometry: Option<Vec<InstanceGeometryType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_light: Option<Vec<InstanceLightType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_node: Option<Vec<InstanceNodeType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub node: Option<Vec<NodeType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_type: Option<NodeEnum>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_layer: Option<ListOfNamesType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxIncludeType {
        pub attr_sid: SidType,
        pub attr_url: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryNodesType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub node: Vec<NodeType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSourcesType {
        pub inline: String,
        pub import: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WiresType {
        pub input: Vec<InputLocalOffsetType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<PType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SamplerType {
        pub input: Vec<InputLocalType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_pre_behavior: Option<SamplerBehaviorEnum>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_post_behavior: Option<SamplerBehaviorEnum>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexcombinerArgumentRgbType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProfileBridgeType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_platform: Option<String>,
        pub attr_url: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlslArrayType {
        pub attr_length: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryPhysicsScenesType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub physics_scene: Vec<PhysicsSceneType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MotionAxisInfoType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AxisType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FormulaSetparamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ParamType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_semantic: Option<String>,
        pub attr_type: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommonFloat2OrParamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PolygonsType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input: Option<Vec<InputLocalOffsetType>>,
        pub p: PType,
        pub ph: String,
        pub p: PType,
        pub h: Vec<ListOfUintsType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlslProgramType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub shader: Option<Vec<GlslShaderType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub bind_attribute: Option<Vec<String>>,
        pub semantic: String,
        pub attr_symbol: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EdgesType {
        pub input: Vec<InputLocalOffsetType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<PType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ControllerType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub skin: SkinType,
        pub morph: MorphType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstancePhysicsMaterialType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexcombinerCommandRgbType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct IdrefArrayType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InputGlobalType {
        pub attr_semantic: String,
        pub attr_source: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSampler1DType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CapsuleType {
        pub height: FloatType,
        pub radius: Float3Type,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CameraType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub optics: String,
        pub technique_common: String,
        pub orthographic: String,
        pub xmag: TargetableFloatType,
        pub ymag: TargetableFloatType,
        pub aspect_ratio: TargetableFloatType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ImageSourceType {
        pub ref: String,
        pub hex: String,
        pub attr_format: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct OriginType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TristripsType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input: Option<Vec<InputLocalOffsetType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<Vec<PType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_material: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryControllersType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub controller: Vec<ControllerType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LimitsSubType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstancePhysicsModelType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_force_field: Option<Vec<InstanceForceFieldType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_rigid_body: Option<Vec<InstanceRigidBodyType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_rigid_constraint: Option<Vec<InstanceRigidConstraintType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        pub attr_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_parent: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryPhysicsModelsType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub physics_model: Vec<PhysicsModelType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AxisConstraintType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MatrixType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChannelType {
        pub attr_source: UrifragmentType,
        pub attr_target: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesSamplerType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub texcoord: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryPhysicsMaterialsType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub physics_material: Vec<PhysicsMaterialType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SweptSurfaceType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct VisualSceneType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub node: Vec<NodeType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub evaluate_scene: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub render: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub layer: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_material: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub technique_override: Option<String>,
        pub attr_ref: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_pass: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsIndexType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BindMaterialType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub param: Option<Vec<ParamType>>,
        pub technique_common: String,
        pub instance_material: Vec<InstanceMaterialType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgUserType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub setparam: Option<Vec<CgSetparamType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_source: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SurfaceType {
        pub cylinder: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LinestripsType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input: Option<Vec<InputLocalOffsetType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<Vec<PType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_material: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommonParamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProfileCommonType {
        pub technique: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub constant: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct IntArrayType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_min_inclusive: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_max_inclusive: Option<i64>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexenvCommandType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EllipsoidType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceMaterialType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub bind: Option<Vec<String>>,
        pub attr_semantic: String,
        pub attr_target: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryJointsType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryArticulatedSystemsType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EffectType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotate: Option<Vec<FxAnnotateType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub newparam: Option<Vec<FxNewparamType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        pub attr_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxDepthtargetType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxClearstencilType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SidrefArrayType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PhysicsModelType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub rigid_body: Option<Vec<RigidBodyType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub rigid_constraint: Option<Vec<RigidConstraintType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instance_physics_model: Option<Vec<InstancePhysicsModelType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxClearcolorType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TargetableFloat3Type {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgNewparamType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotate: Option<Vec<FxAnnotateType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub semantic: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub modifier: Option<FxModifierEnum>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NameArrayType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceImageType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgPassType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub states: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FacesType {
        pub input: Vec<InputLocalOffsetType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<PType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NurbsType {
        pub control_vertices: String,
        pub input: Vec<InputLocalType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryGeometriesType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub geometry: Vec<GeometryType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InputLocalOffsetType {
        pub attr_offset: UintType,
        pub attr_semantic: String,
        pub attr_source: UrifragmentType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_set: Option<UintType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceWithExtraType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        pub attr_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgArrayType {
        pub attr_length: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_resizable: Option<bool>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ParabolaType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PolylistType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input: Option<Vec<InputLocalOffsetType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub vcount: Option<ListOfUintsType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<PType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_material: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexcombinerCommandAlphaType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxCommonFloatOrParamType {
        pub float: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceJointType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsLimitsType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceKinematicsModelType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BoolArrayType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LinesType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input: Option<Vec<InputLocalOffsetType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<PType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_material: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TrianglesType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input: Option<Vec<InputLocalOffsetType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<PType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_material: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MotionEffectorInfoType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryCamerasType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub camera: Vec<CameraType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Gles2ProgramType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub shader: Option<Vec<Gles2ShaderType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub linker: Option<Vec<FxTargetType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub bind_attribute: Option<Vec<String>>,
        pub semantic: String,
        pub attr_symbol: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxNewparamType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotate: Option<Vec<FxAnnotateType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub semantic: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub modifier: Option<FxModifierEnum>,
        pub attr_sid: SidType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SphereType {
        pub radius: FloatType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MotionTechniqueType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceFormulaType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MorphType {
        pub source: Vec<SourceType>,
        pub targets: String,
        pub input: Vec<InputLocalType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LinkType {
        pub attachment_full: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryForceFieldsType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub force_field: Vec<ForceFieldType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommonBoolOrParamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ExtraType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub technique: Vec<TechniqueType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_type: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NurbsSurfaceType {
        pub control_vertices: String,
        pub input: Vec<InputLocalType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FormulaType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryKinematicsModelsType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxCleardepthType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TorusType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TargetableFloat4Type {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AccessorType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub param: Option<Vec<ParamType>>,
        pub attr_count: UintType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_offset: Option<UintType>,
        pub attr_source: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_stride: Option<UintType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SkinType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub bind_shape_matrix: Option<Float4X4Type>,
        pub source: Vec<SourceType>,
        pub joints: String,
        pub input: Vec<InputLocalType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommonIntOrParamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsAxisInfoType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ImageMipsType {
        pub attr_levels: u32,
        pub attr_auto_generate: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FormulaNewparamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EllipseType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxStenciltargetType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InputLocalType {
        pub attr_semantic: String,
        pub attr_source: UrifragmentType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TranslateType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlslNewparamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CylinderType {
        pub height: FloatType,
        pub radius: Float2Type,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProfileCgType {
        pub technique: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub pass: Vec<CgPassType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        pub attr_sid: SidType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BoxType {
        pub half_extents: Float3Type,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HyperbolaType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RotateType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ImageType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub renderable: Option<String>,
        pub attr_share: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MinmaxType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsModelTechniqueType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AnimationClipType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub instance_animation: Vec<InstanceWithExtraType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_start: Option<FloatType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_end: Option<FloatType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSampler3DType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ForceFieldType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub technique: Vec<TechniqueType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ShellsType {
        pub input: Vec<InputLocalOffsetType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<PType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TokenArrayType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsModelType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryAnimationsType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub animation: Vec<AnimationType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceForceFieldType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GeometryType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub convex_mesh: ConvexMeshType,
        pub mesh: MeshType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxTargetType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub binary: Option<String>,
        pub hex: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_format: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FloatArrayType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_digits: Option<DigitsType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_magnitude: Option<MagnitudeType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceEffectType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub technique_hint: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_platform: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_profile: Option<String>,
        pub attr_ref: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxRendertargetType {
        pub param: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlslShaderType {
        pub sources: FxSourcesType,
        pub attr_stage: FxPipelineStageEnum,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SurfaceCurvesType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceLightType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceArticulatedSystemType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JointLimitsType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryKinematicsScenesType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceCameraType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommonFloatOrParamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CgSetparamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsConnectParamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryImagesType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub image: Vec<ImageType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct VerticesType {
        pub input: Vec<InputLocalType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        pub attr_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryEffectsType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub effect: Vec<EffectType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryAnimationClipsType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub animation_clip: Vec<AnimationClipType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LookatType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FormulaTechniqueType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ArticulatedSystemType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SurfacesType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceControllerType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub skeleton: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub bind_material: Option<BindMaterialType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        pub attr_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxCommonColorOrTextureType {
        pub color: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSamplerCubeType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ScaleType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SolidsType {
        pub input: Vec<InputLocalOffsetType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<PType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JointType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BindKinematicsModelType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTextureConstantType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SkewType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TechniqueType {
        pub attr_profile: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryFormulasType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PhysicsMaterialType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub technique_common: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub dynamic_friction: Option<TargetableFloatType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub restitution: Option<TargetableFloatType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub static_friction: Option<TargetableFloatType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PcurvesType {
        pub input: Vec<InputLocalOffsetType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CircleType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TrifansType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input: Option<Vec<InputLocalOffsetType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<Vec<PType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
        pub attr_count: UintType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_material: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSampler2DType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesNewparamType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotate: Option<Vec<FxAnnotateType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub semantic: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub modifier: Option<FxModifierEnum>,
        pub attr_sid: SidType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxCodeType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PlaneType {
        pub equation: Float4Type,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProfileGlesType {
        pub technique: Vec<String>,
        pub pass: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub states: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSamplerType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TargetableFloatType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxCommonTransparentType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceGeometryType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub bind_material: Option<BindMaterialType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        pub attr_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Gles2PassType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotate: Option<Vec<FxAnnotateType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub states: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxSamplerDepthType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryLightsType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub light: Vec<LightType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CurvesType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LibraryMaterialsType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub material: Vec<MaterialType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Gles2ShaderType {
        pub sources: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_entry: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AnimationType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub source: Vec<SourceType>,
        pub sampler: Vec<SamplerType>,
        pub channel: Vec<ChannelType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub animation: Option<Vec<AnimationType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceNodeType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AssetType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub contributor: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub author: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub author_email: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub author_website: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub authoring_tool: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub comments: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub copyright: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub source_data: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LineType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceRigidBodyType {
        pub technique_common: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub angular_velocity: Option<Float3Type>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub velocity: Option<Float3Type>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub dynamic: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BindJointAxisType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxColortargetType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexcombinerArgumentAlphaType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SourceType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub token_array: TokenArrayType,
        pub idref_array: IdrefArrayType,
        pub name_array: NameArrayType,
        pub bool_array: BoolArrayType,
        pub float_array: FloatArrayType,
        pub int_array: IntArrayType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub technique_common: Option<String>,
        pub accessor: AccessorType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MeshType {
        pub source: Vec<SourceType>,
        pub vertices: VerticesType,
        pub lines: LinesType,
        pub linestrips: LinestripsType,
        pub polygons: PolygonsType,
        pub polylist: PolylistType,
        pub triangles: TrianglesType,
        pub trifans: TrifansType,
        pub tristrips: TristripsType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ConeType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsSetparamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsParamType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProfileGles2Type {
        pub include: FxIncludeType,
        pub code: FxCodeType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub newparam: Option<Vec<String>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsBindType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InstanceKinematicsSceneType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FxAnnotateType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SplineType {
        pub source: Vec<SourceType>,
        pub control_vertices: String,
        pub input: Vec<InputLocalType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RigidConstraintType {
        pub ref_attachment: String,
        pub translate: TranslateType,
        pub rotate: RotateType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_rigid_body: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ConvexMeshType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_convex_hull_of: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsSceneType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProfileGlslType {
        pub technique: Vec<String>,
        pub pass: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub states: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlesTexcombinerCommandType {
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MaterialType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub asset: Option<AssetType>,
        pub instance_effect: InstanceEffectType,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extra: Option<Vec<ExtraType>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CurveType {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_sid: Option<SidType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attr_name: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KinematicsNewparamType {
    }

}
