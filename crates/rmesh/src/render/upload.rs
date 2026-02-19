use image::GenericImageView;
use nalgebra::{Matrix4, Point3, Vector4};
use wgpu::util::DeviceExt;

use crate::attributes::{DEFAULT_COLOR, Material};
use crate::bounds::Bounds3;
use crate::geometry::Geometry;
use crate::image::LazyImage;
use crate::render::ShadingMode;
use crate::scene::{Scene, SceneNodeKind};

/// GPU-ready mesh vertex: 52 bytes interleaved.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [u8; 4],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

/// GPU-ready line vertex.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

/// GPU-ready point vertex.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// Material converted for GPU use.
#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct GpuMaterial {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub use_vertex_color: bool,
    pub emissive_factor: [f32; 3],
    pub occlusion_strength: f32,
    pub normal_scale: f32,
    pub has_metallic_roughness_texture: bool,
    pub has_normal_texture: bool,
    pub has_occlusion_texture: bool,
    pub has_emissive_texture: bool,
}

impl Default for GpuMaterial {
    fn default() -> Self {
        Self {
            base_color: [0.4, 0.4, 0.4, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            use_vertex_color: false,
            emissive_factor: [0.0, 0.0, 0.0],
            occlusion_strength: 1.0,
            normal_scale: 1.0,
            has_metallic_roughness_texture: false,
            has_normal_texture: false,
            has_occlusion_texture: false,
            has_emissive_texture: false,
        }
    }
}

/// A GPU texture with its view.
pub struct GpuTexture {
    #[allow(dead_code)] // Holds ownership; GPU texture lives as long as this struct.
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

/// An uploaded mesh draw call.
pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub transform: Matrix4<f32>,
    pub material: GpuMaterial,
    pub base_color_texture: Option<GpuTexture>,
    pub metallic_roughness_texture: Option<GpuTexture>,
    pub normal_texture: Option<GpuTexture>,
    pub occlusion_texture: Option<GpuTexture>,
    pub emissive_texture: Option<GpuTexture>,
}

/// An uploaded path draw call.
pub struct GpuPath {
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
}

/// An uploaded point cloud draw call.
pub struct GpuPointCloud {
    pub vertex_buffer: wgpu::Buffer,
    pub point_count: u32,
}

/// All uploaded scene data.
pub struct SceneGpuData {
    pub meshes: Vec<GpuMesh>,
    pub paths: Vec<GpuPath>,
    pub points: Vec<GpuPointCloud>,
    pub bounds: Bounds3,
}

#[allow(clippy::cast_possible_truncation)]
fn convert_material(mat: &Material) -> GpuMaterial {
    let pbr = mat.to_pbr();
    GpuMaterial {
        base_color: [
            pbr.base_color_factor.x as f32,
            pbr.base_color_factor.y as f32,
            pbr.base_color_factor.z as f32,
            pbr.base_color_factor.w as f32,
        ],
        metallic: pbr.metallic_factor as f32,
        roughness: pbr.roughness_factor as f32,
        use_vertex_color: false,
        emissive_factor: [
            pbr.emissive_factor.x as f32,
            pbr.emissive_factor.y as f32,
            pbr.emissive_factor.z as f32,
        ],
        occlusion_strength: pbr.occlusion_strength as f32,
        normal_scale: pbr.normal_scale as f32,
        has_metallic_roughness_texture: pbr.metallic_roughness_texture.is_some(),
        has_normal_texture: pbr.normal_texture.is_some(),
        has_occlusion_texture: pbr.occlusion_texture.is_some(),
        has_emissive_texture: pbr.emissive_texture.is_some(),
    }
}

/// Upload a `LazyImage` to a GPU texture.
fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: &LazyImage,
    srgb: bool,
) -> Option<GpuTexture> {
    let decoded = img.decode()?;
    let rgba = decoded.to_rgba8();
    let (w, h) = decoded.dimensions();

    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };

    let size = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pbr_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        texture.as_image_copy(),
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: Some(h),
        },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Some(GpuTexture { texture, view })
}

#[allow(clippy::cast_possible_truncation)]
fn mat4_f64_to_f32(m: &Matrix4<f64>) -> Matrix4<f32> {
    Matrix4::new(
        m[(0, 0)] as f32,
        m[(0, 1)] as f32,
        m[(0, 2)] as f32,
        m[(0, 3)] as f32,
        m[(1, 0)] as f32,
        m[(1, 1)] as f32,
        m[(1, 2)] as f32,
        m[(1, 3)] as f32,
        m[(2, 0)] as f32,
        m[(2, 1)] as f32,
        m[(2, 2)] as f32,
        m[(2, 3)] as f32,
        m[(3, 0)] as f32,
        m[(3, 1)] as f32,
        m[(3, 2)] as f32,
        m[(3, 3)] as f32,
    )
}

/// Upload scene geometry to GPU buffers.
pub fn upload_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &Scene,
    shading_mode: ShadingMode,
) -> SceneGpuData {
    let mut meshes = Vec::new();
    let mut paths = Vec::new();
    let mut points = Vec::new();

    let mut bounds = Bounds3::empty();

    // Walk the scene graph to get geometry with world transforms
    let geometry_names: Vec<String> = scene.geometry.keys().cloned().collect();

    scene.graph.walk(|_idx, node, world_transform| {
        if node.kind != SceneNodeKind::Geometry {
            return;
        }

        for &geom_idx in &node.index {
            if geom_idx >= geometry_names.len() {
                continue;
            }
            let name = &geometry_names[geom_idx];
            let Some(geom) = scene.geometry.get(name) else {
                continue;
            };

            match geom {
                Geometry::Mesh(mesh) => {
                    upload_mesh(
                        device,
                        queue,
                        mesh,
                        world_transform,
                        shading_mode,
                        &mut meshes,
                        &mut bounds,
                    );
                }
                Geometry::Path2D(path) => {
                    upload_path2d(device, path, world_transform, &mut paths, &mut bounds);
                }
                Geometry::Path3D(path) => {
                    upload_path3d(device, path, world_transform, &mut paths, &mut bounds);
                }
                Geometry::PointCloud(pc) => {
                    upload_point_cloud(device, pc, world_transform, &mut points, &mut bounds);
                }
                #[cfg(feature = "cad")]
                Geometry::Feature(_) => {
                    // FeatureModel not directly renderable; would need meshing first.
                }
                Geometry::Brep(brep) => {
                    // Tessellate BREP to mesh for rendering
                    let params = crate::boundary::tesselate::TesselationParams::default();
                    let mesh = brep.tesselate(&params);
                    upload_mesh(
                        device,
                        queue,
                        &mesh,
                        world_transform,
                        shading_mode,
                        &mut meshes,
                        &mut bounds,
                    );
                }
            }
        }
    });

    // If the scene graph is empty but geometry exists, upload without transforms
    if meshes.is_empty() && paths.is_empty() && points.is_empty() && !scene.geometry.is_empty() {
        let identity = Matrix4::identity();
        for geom in scene.geometry.values() {
            match geom {
                Geometry::Mesh(mesh) => {
                    upload_mesh(
                        device,
                        queue,
                        mesh,
                        &identity,
                        shading_mode,
                        &mut meshes,
                        &mut bounds,
                    );
                }
                Geometry::Path2D(path) => {
                    upload_path2d(device, path, &identity, &mut paths, &mut bounds);
                }
                Geometry::Path3D(path) => {
                    upload_path3d(device, path, &identity, &mut paths, &mut bounds);
                }
                Geometry::PointCloud(pc) => {
                    upload_point_cloud(device, pc, &identity, &mut points, &mut bounds);
                }
                #[cfg(feature = "cad")]
                Geometry::Feature(_) => {}
                Geometry::Brep(brep) => {
                    // Tessellate BREP to mesh for rendering
                    let params = crate::boundary::tesselate::TesselationParams::default();
                    let mesh = brep.tesselate(&params);
                    upload_mesh(
                        device,
                        queue,
                        &mesh,
                        &identity,
                        shading_mode,
                        &mut meshes,
                        &mut bounds,
                    );
                }
            }
        }
    }

    // Ensure valid bounds
    if bounds.is_empty() {
        bounds = Bounds3::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
    }

    log::info!(
        "upload: {} meshes, {} paths, {} points; bounds: [{:.3}, {:.3}, {:.3}] to [{:.3}, {:.3}, {:.3}]",
        meshes.len(),
        paths.len(),
        points.len(),
        bounds.min.x,
        bounds.min.y,
        bounds.min.z,
        bounds.max.x,
        bounds.max.y,
        bounds.max.z,
    );

    SceneGpuData {
        meshes,
        paths,
        points,
        bounds,
    }
}

fn transform_point(p: &Point3<f64>, m: &Matrix4<f64>) -> Point3<f64> {
    let v = m * nalgebra::Vector4::new(p.x, p.y, p.z, 1.0);
    Point3::new(v.x, v.y, v.z)
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn upload_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &crate::mesh::Trimesh,
    world_transform: &Matrix4<f64>,
    shading_mode: ShadingMode,
    meshes: &mut Vec<GpuMesh>,
    bounds: &mut Bounds3,
) {
    if mesh.faces.is_empty() {
        return;
    }

    // Get per-vertex attributes if available
    let vertex_colors: Option<&[Vector4<u8>]> =
        mesh.attributes_vertex.colors.first().map(|c| c.as_slice());

    let vertex_uvs: Option<&[nalgebra::Vector2<f64>]> =
        mesh.attributes_vertex.uv.first().map(|u| u.as_slice());

    let smooth_vertex_normals: Option<&[nalgebra::Vector3<f64>]> =
        mesh.attributes_vertex.normals.first().map(|n| n.as_slice());

    let vertex_tangents: Option<&[nalgebra::Vector4<f64>]> = mesh
        .attributes_vertex
        .tangents
        .first()
        .map(|t| t.as_slice());

    let has_vertex_colors = vertex_colors.is_some();

    let default_tangent = [0.0f32, 0.0, 1.0, 1.0];

    // If mesh has file vertex normals AND we're in Smooth mode, use them as-is
    // with the efficient shared-vertex path (no duplication).
    let use_file_normals = smooth_vertex_normals.is_some() && shading_mode == ShadingMode::Smooth;

    let (vertices, indices) = if use_file_normals {
        let normals = smooth_vertex_normals.unwrap();
        let mut verts = Vec::with_capacity(mesh.vertices.len());
        for (vi, p) in mesh.vertices.iter().enumerate() {
            let wp = transform_point(p, world_transform);
            bounds.include_point(&wp);

            let n = normals
                .get(vi)
                .copied()
                .unwrap_or_else(nalgebra::Vector3::zeros);
            let color = vertex_colors
                .and_then(|c| c.get(vi))
                .copied()
                .unwrap_or(DEFAULT_COLOR);
            let uv = vertex_uvs
                .and_then(|u| u.get(vi))
                .copied()
                .unwrap_or_else(nalgebra::Vector2::zeros);
            let tangent = vertex_tangents
                .and_then(|t| t.get(vi))
                .map_or(default_tangent, |t| {
                    [t.x as f32, t.y as f32, t.z as f32, t.w as f32]
                });

            verts.push(MeshVertex {
                position: [p.x as f32, p.y as f32, p.z as f32],
                normal: [n.x as f32, n.y as f32, n.z as f32],
                color: [color.x, color.y, color.z, color.w],
                uv: [uv.x as f32, uv.y as f32],
                tangent,
            });
        }

        let idxs: Vec<u32> = mesh
            .faces
            .iter()
            .flat_map(|f| f.iter().map(|&i| i as u32))
            .collect();

        (verts, idxs)
    } else {
        // Expanded per-face-corner path (3 verts per face) with computed normals
        let corner_normals: Vec<nalgebra::Vector3<f64>> = match shading_mode {
            ShadingMode::Smooth => {
                // Prefer file-authored smooth groups (e.g. OBJ `s N` directives)
                let file_groups = mesh
                    .attributes_face
                    .groupings
                    .iter()
                    .find(|g| g.kind == crate::attributes::GroupingKind::Smoothing)
                    .map(|g| g.indices.as_slice());
                mesh.smooth_vertex_normals(std::f64::consts::FRAC_PI_6, file_groups)
            }
            ShadingMode::Flat => {
                let fn_ = mesh.face_normals();
                mesh.faces
                    .iter()
                    .enumerate()
                    .flat_map(|(fi, _)| std::iter::repeat_n(fn_[fi], 3))
                    .collect()
            }
            ShadingMode::Full => mesh.smooth_vertex_normals(std::f64::consts::PI, None),
        };

        let mut verts = Vec::with_capacity(mesh.faces.len() * 3);
        let mut idxs = Vec::with_capacity(mesh.faces.len() * 3);

        for (fi, face) in mesh.faces.iter().enumerate() {
            for (ci, &vi) in face.iter().enumerate() {
                let p = &mesh.vertices[vi];
                let wp = transform_point(p, world_transform);
                bounds.include_point(&wp);

                let n = corner_normals[fi * 3 + ci];
                let color = vertex_colors
                    .and_then(|c| c.get(vi))
                    .copied()
                    .unwrap_or(DEFAULT_COLOR);
                let uv = vertex_uvs
                    .and_then(|u| u.get(vi))
                    .copied()
                    .unwrap_or_else(nalgebra::Vector2::zeros);
                let tangent = vertex_tangents
                    .and_then(|t| t.get(vi))
                    .map_or(default_tangent, |t| {
                        [t.x as f32, t.y as f32, t.z as f32, t.w as f32]
                    });

                idxs.push(verts.len() as u32);
                verts.push(MeshVertex {
                    position: [p.x as f32, p.y as f32, p.z as f32],
                    normal: [n.x as f32, n.y as f32, n.z as f32],
                    color: [color.x, color.y, color.z, color.w],
                    uv: [uv.x as f32, uv.y as f32],
                    tangent,
                });
            }
        }

        (verts, idxs)
    };

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // Determine material and upload textures
    let first_mat = mesh.materials.first();
    let mut material = first_mat.map_or(GpuMaterial::default(), convert_material);

    if has_vertex_colors {
        material.use_vertex_color = true;
    }

    let pbr = first_mat.map(|m| m.to_pbr());

    let base_color_texture = pbr.as_ref().and_then(|p| {
        p.base_color_texture
            .as_ref()
            .and_then(|img| upload_texture(device, queue, img, true))
    });

    let metallic_roughness_texture = pbr.as_ref().and_then(|p| {
        p.metallic_roughness_texture
            .as_ref()
            .and_then(|img| upload_texture(device, queue, img, false))
    });

    let normal_texture = pbr.as_ref().and_then(|p| {
        p.normal_texture
            .as_ref()
            .and_then(|img| upload_texture(device, queue, img, false))
    });

    let occlusion_texture = pbr.as_ref().and_then(|p| {
        p.occlusion_texture
            .as_ref()
            .and_then(|img| upload_texture(device, queue, img, false))
    });

    let emissive_texture = pbr.as_ref().and_then(|p| {
        p.emissive_texture
            .as_ref()
            .and_then(|img| upload_texture(device, queue, img, true))
    });

    let transform = mat4_f64_to_f32(world_transform);

    meshes.push(GpuMesh {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        transform,
        material,
        base_color_texture,
        metallic_roughness_texture,
        normal_texture,
        occlusion_texture,
        emissive_texture,
    });
}

#[allow(clippy::cast_possible_truncation)]
fn upload_path2d(
    device: &wgpu::Device,
    path: &crate::path::Path2D,
    world_transform: &Matrix4<f64>,
    paths: &mut Vec<GpuPath>,
    bounds: &mut Bounds3,
) {
    let segments = path.to_segments();
    let color = [0.0f32, 0.8, 0.2]; // Green for 2D paths

    for segment_points in &segments {
        if segment_points.len() < 2 {
            continue;
        }
        let mut vertices = Vec::with_capacity(segment_points.len() * 2);
        for window in segment_points.windows(2) {
            let p0 = Point3::new(window[0].x, window[0].y, 0.0);
            let p1 = Point3::new(window[1].x, window[1].y, 0.0);
            let wp0 = transform_point(&p0, world_transform);
            let wp1 = transform_point(&p1, world_transform);
            bounds.include_point(&wp0);
            bounds.include_point(&wp1);
            vertices.push(LineVertex {
                position: [wp0.x as f32, wp0.y as f32, wp0.z as f32],
                color,
            });
            vertices.push(LineVertex {
                position: [wp1.x as f32, wp1.y as f32, wp1.z as f32],
                color,
            });
        }

        if vertices.is_empty() {
            continue;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("path2d_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        paths.push(GpuPath {
            vertex_buffer,
            vertex_count: vertices.len() as u32,
        });
    }
}

#[allow(clippy::cast_possible_truncation)]
fn upload_path3d(
    device: &wgpu::Device,
    path: &crate::path::Path3D,
    world_transform: &Matrix4<f64>,
    paths: &mut Vec<GpuPath>,
    bounds: &mut Bounds3,
) {
    let segments = path.to_segments();
    let color = [0.8f32, 0.6, 0.0]; // Orange for 3D paths

    for segment_points in &segments {
        if segment_points.len() < 2 {
            continue;
        }
        let mut vertices = Vec::with_capacity(segment_points.len() * 2);
        for window in segment_points.windows(2) {
            let wp0 = transform_point(&window[0], world_transform);
            let wp1 = transform_point(&window[1], world_transform);
            bounds.include_point(&wp0);
            bounds.include_point(&wp1);
            vertices.push(LineVertex {
                position: [wp0.x as f32, wp0.y as f32, wp0.z as f32],
                color,
            });
            vertices.push(LineVertex {
                position: [wp1.x as f32, wp1.y as f32, wp1.z as f32],
                color,
            });
        }

        if vertices.is_empty() {
            continue;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("path3d_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        paths.push(GpuPath {
            vertex_buffer,
            vertex_count: vertices.len() as u32,
        });
    }
}

#[allow(clippy::cast_possible_truncation)]
fn upload_point_cloud(
    device: &wgpu::Device,
    pc: &crate::geometry::PointCloud,
    world_transform: &Matrix4<f64>,
    points: &mut Vec<GpuPointCloud>,
    bounds: &mut Bounds3,
) {
    if pc.points.is_empty() {
        return;
    }

    let default_color = Vector4::new(200u8, 200, 200, 255);
    let vertices: Vec<PointVertex> = pc
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let wp = transform_point(p, world_transform);
            bounds.include_point(&wp);
            let c = pc
                .colors
                .as_ref()
                .and_then(|cs| cs.get(i))
                .copied()
                .unwrap_or(default_color);
            PointVertex {
                position: [wp.x as f32, wp.y as f32, wp.z as f32],
                color: [
                    f32::from(c.x) / 255.0,
                    f32::from(c.y) / 255.0,
                    f32::from(c.z) / 255.0,
                    f32::from(c.w) / 255.0,
                ],
            }
        })
        .collect();

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("point_cloud_vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    points.push(GpuPointCloud {
        vertex_buffer,
        point_count: vertices.len() as u32,
    });
}
