use nalgebra::Matrix4;
use wgpu::util::DeviceExt;
use wgpu::{
    DepthBiasState, MultisampleState, PipelineCompilationOptions, StencilState, TextureFormat,
};

use crate::gpu::DEPTH_FORMAT;
use crate::render::mat4_to_array;
use crate::render::shaders::{MaterialUniforms, ModelUniforms};
use crate::upload::{GpuMesh, MeshVertex};

/// Pre-created bind groups for a single mesh draw call.
pub struct MeshBindGroups {
    pub model_bind_group: wgpu::BindGroup,
    pub material_bind_group: wgpu::BindGroup,
    pub texture_bind_group: wgpu::BindGroup,
}

pub struct MeshRenderer {
    pipeline_fill: wgpu::RenderPipeline,
    pipeline_wire: wgpu::RenderPipeline,
    model_bind_group_layout: wgpu::BindGroupLayout,
    material_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    default_white_view: wgpu::TextureView,
    default_normal_view: wgpu::TextureView,
    default_black_view: wgpu::TextureView,
    default_sampler: wgpu::Sampler,
}

impl MeshRenderer {
    pub fn new_with_format(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_bgl: &wgpu::BindGroupLayout,
        format: TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh.wgsl").into()),
        });

        let model_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_bgl"),
            entries: &[
                // binding 0: base_color_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 1: mr_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: normal_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: occlusion_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 4: emissive_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 5: shared sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[camera_bgl, &model_bgl, &material_bgl, &texture_bgl],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Unorm8x4,
                    offset: 24,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 28,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 36,
                    shader_location: 4,
                },
            ],
        };

        let make_pipeline = |cull: Option<wgpu::Face>, poly_mode: wgpu::PolygonMode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mesh_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: std::slice::from_ref(&vertex_layout),
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: cull,
                    polygon_mode: poly_mode,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let pipeline_fill = make_pipeline(None, wgpu::PolygonMode::Fill);
        let pipeline_wire = make_pipeline(None, wgpu::PolygonMode::Line);

        // 1x1 white fallback (base color, metallic-roughness, occlusion)
        let default_white = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("default_white"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255u8, 255, 255, 255],
        );
        let default_white_view = default_white.create_view(&wgpu::TextureViewDescriptor::default());

        // 1x1 flat normal (128,128,255 = tangent-space +Z)
        let default_normal = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("default_normal"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[128u8, 128, 255, 255],
        );
        let default_normal_view =
            default_normal.create_view(&wgpu::TextureViewDescriptor::default());

        // 1x1 black (no emissive)
        let default_black = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("default_black"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[0u8, 0, 0, 255],
        );
        let default_black_view = default_black.create_view(&wgpu::TextureViewDescriptor::default());

        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("default_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline_fill,
            pipeline_wire,
            model_bind_group_layout: model_bgl,
            material_bind_group_layout: material_bgl,
            texture_bind_group_layout: texture_bgl,
            default_white_view,
            default_normal_view,
            default_black_view,
            default_sampler,
        }
    }

    /// Pre-create bind groups for all meshes (call before render pass).
    pub fn prepare_bind_groups(
        &self,
        device: &wgpu::Device,
        meshes: &[GpuMesh],
    ) -> Vec<MeshBindGroups> {
        meshes
            .iter()
            .map(|mesh| {
                let transform = mesh.transform;
                let normal_matrix = transform
                    .try_inverse()
                    .unwrap_or_else(Matrix4::identity)
                    .transpose();

                let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("model_uniform"),
                    contents: bytemuck::bytes_of(&ModelUniforms {
                        model: mat4_to_array(&transform),
                        normal_matrix: mat4_to_array(&normal_matrix),
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let has_texture = mesh.base_color_texture.is_some();
                let mat = &mesh.material;

                let material_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("material_uniform"),
                        contents: bytemuck::bytes_of(&MaterialUniforms {
                            base_color: mat.base_color,
                            metallic: mat.metallic,
                            roughness: mat.roughness,
                            use_vertex_color: u32::from(mat.use_vertex_color),
                            has_texture: u32::from(has_texture),
                            emissive_factor: mat.emissive_factor,
                            occlusion_strength: mat.occlusion_strength,
                            normal_scale: mat.normal_scale,
                            has_mr_texture: u32::from(mat.has_metallic_roughness_texture),
                            has_normal_texture: u32::from(mat.has_normal_texture),
                            has_occlusion_texture: u32::from(mat.has_occlusion_texture),
                            has_emissive_texture: u32::from(mat.has_emissive_texture),
                            _pad_0: [0; 12],
                        }),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("model_bg"),
                    layout: &self.model_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: model_buffer.as_entire_binding(),
                    }],
                });

                let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("material_bg"),
                    layout: &self.material_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: material_buffer.as_entire_binding(),
                    }],
                });

                let base_color_view = mesh
                    .base_color_texture
                    .as_ref()
                    .map_or(&self.default_white_view, |t| &t.view);
                let mr_view = mesh
                    .metallic_roughness_texture
                    .as_ref()
                    .map_or(&self.default_white_view, |t| &t.view);
                let normal_view = mesh
                    .normal_texture
                    .as_ref()
                    .map_or(&self.default_normal_view, |t| &t.view);
                let occlusion_view = mesh
                    .occlusion_texture
                    .as_ref()
                    .map_or(&self.default_white_view, |t| &t.view);
                let emissive_view = mesh
                    .emissive_texture
                    .as_ref()
                    .map_or(&self.default_black_view, |t| &t.view);

                let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("texture_bg"),
                    layout: &self.texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(base_color_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(mr_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(occlusion_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(emissive_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(&self.default_sampler),
                        },
                    ],
                });

                MeshBindGroups {
                    model_bind_group,
                    material_bind_group,
                    texture_bind_group,
                }
            })
            .collect()
    }

    pub fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        meshes: &'a [GpuMesh],
        bind_groups: &'a [MeshBindGroups],
        wireframe: bool,
    ) {
        pass.set_pipeline(if wireframe {
            &self.pipeline_wire
        } else {
            &self.pipeline_fill
        });

        for (mesh, bg) in meshes.iter().zip(bind_groups.iter()) {
            pass.set_bind_group(1, &bg.model_bind_group, &[]);
            pass.set_bind_group(2, &bg.material_bind_group, &[]);
            pass.set_bind_group(3, &bg.texture_bind_group, &[]);
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }
}
