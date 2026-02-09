use std::path::Path;

use anyhow::Result;
use nalgebra::{Point3, Vector3, Vector4};
use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ndarray::Array2,
    npyffi,
};
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rmesh::attributes::{
    AlphaMode, Attributes, Grouping, GroupingKind, Material, SimpleMaterial, UNSET,
};
use rmesh::boundary::faces;
use rmesh::exchange::{FileResolver, FileType, InMemoryResolver, load};
use rmesh::geometry::Geometry;
use rmesh::mesh::Trimesh;
use rmesh::resolvers::Resolver;
use rmesh_viewer::{RenderOptions, SceneViewer, ViewerOptions};

// ============================================================================
// PyMaterial
// ============================================================================

/// A material that can be applied to mesh faces.
#[pyclass(name = "Material")]
pub struct PyMaterial {
    data: Material,
}

#[pymethods]
impl PyMaterial {
    /// The name of the material.
    #[getter]
    fn name(&self) -> &str {
        self.data.name()
    }

    /// The kind of material: "empty", "simple", or "pbr".
    #[getter]
    fn kind(&self) -> &str {
        match &self.data {
            Material::Empty(_) => "empty",
            Material::Simple(_) => "simple",
            Material::PBR(_) => "pbr",
        }
    }

    // -------------------------------------------------------------------------
    // SimpleMaterial properties
    // -------------------------------------------------------------------------

    /// Diffuse color (RGB), only for simple materials.
    #[getter]
    fn diffuse(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        if let Material::Simple(SimpleMaterial {
            diffuse: Some(d), ..
        }) = &self.data
        {
            Some(readonly_1d(py, vec![d.x, d.y, d.z]))
        } else {
            None
        }
    }

    /// Specular color (RGB), only for simple materials.
    #[getter]
    fn specular(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        if let Material::Simple(SimpleMaterial {
            specular: Some(s), ..
        }) = &self.data
        {
            Some(readonly_1d(py, vec![s.x, s.y, s.z]))
        } else {
            None
        }
    }

    /// Shininess value, only for simple materials.
    #[getter]
    fn shininess(&self) -> Option<f64> {
        if let Material::Simple(SimpleMaterial { shininess, .. }) = &self.data {
            *shininess
        } else {
            None
        }
    }

    /// Alpha (transparency) value, only for simple materials.
    #[getter]
    fn alpha(&self) -> Option<f64> {
        if let Material::Simple(SimpleMaterial { alpha, .. }) = &self.data {
            *alpha
        } else {
            None
        }
    }

    /// Whether this material has a diffuse texture, only for simple materials.
    #[getter]
    fn has_diffuse_texture(&self) -> bool {
        matches!(&self.data, Material::Simple(s) if s.diffuse_texture.is_some())
    }

    // -------------------------------------------------------------------------
    // PBRMaterial properties
    // -------------------------------------------------------------------------

    /// Base color factor (RGBA), only for PBR materials.
    #[getter]
    fn base_color_factor(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        if let Material::PBR(pbr) = &self.data {
            let c = &pbr.base_color_factor;
            Some(readonly_1d(py, vec![c.x, c.y, c.z, c.w]))
        } else {
            None
        }
    }

    /// Metallic factor (0.0 = dielectric, 1.0 = metal), only for PBR materials.
    #[getter]
    fn metallic_factor(&self) -> Option<f64> {
        if let Material::PBR(pbr) = &self.data {
            Some(pbr.metallic_factor)
        } else {
            None
        }
    }

    /// Roughness factor (0.0 = smooth, 1.0 = rough), only for PBR materials.
    #[getter]
    fn roughness_factor(&self) -> Option<f64> {
        if let Material::PBR(pbr) = &self.data {
            Some(pbr.roughness_factor)
        } else {
            None
        }
    }

    /// Normal map scale, only for PBR materials.
    #[getter]
    fn normal_scale(&self) -> Option<f64> {
        if let Material::PBR(pbr) = &self.data {
            Some(pbr.normal_scale)
        } else {
            None
        }
    }

    /// Occlusion strength, only for PBR materials.
    #[getter]
    fn occlusion_strength(&self) -> Option<f64> {
        if let Material::PBR(pbr) = &self.data {
            Some(pbr.occlusion_strength)
        } else {
            None
        }
    }

    /// Emissive color factor (RGB), only for PBR materials.
    #[getter]
    fn emissive_factor(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        if let Material::PBR(pbr) = &self.data {
            let e = &pbr.emissive_factor;
            Some(readonly_1d(py, vec![e.x, e.y, e.z]))
        } else {
            None
        }
    }

    /// Alpha blending mode: "opaque", "mask", or "blend", only for PBR materials.
    #[getter]
    fn alpha_mode(&self) -> Option<&str> {
        if let Material::PBR(pbr) = &self.data {
            Some(match pbr.alpha_mode {
                AlphaMode::Opaque => "opaque",
                AlphaMode::Mask => "mask",
                AlphaMode::Blend => "blend",
            })
        } else {
            None
        }
    }

    /// Alpha cutoff threshold for mask mode, only for PBR materials.
    #[getter]
    fn alpha_cutoff(&self) -> Option<f64> {
        if let Material::PBR(pbr) = &self.data {
            Some(pbr.alpha_cutoff)
        } else {
            None
        }
    }

    /// Whether the material is double-sided, only for PBR materials.
    #[getter]
    fn double_sided(&self) -> Option<bool> {
        if let Material::PBR(pbr) = &self.data {
            Some(pbr.double_sided)
        } else {
            None
        }
    }

    /// Whether this material has a base color texture, only for PBR materials.
    #[getter]
    fn has_base_color_texture(&self) -> bool {
        matches!(&self.data, Material::PBR(p) if p.base_color_texture.is_some())
    }

    /// Whether this material has a metallic-roughness texture, only for PBR materials.
    #[getter]
    fn has_metallic_roughness_texture(&self) -> bool {
        matches!(&self.data, Material::PBR(p) if p.metallic_roughness_texture.is_some())
    }

    /// Whether this material has a normal texture, only for PBR materials.
    #[getter]
    fn has_normal_texture(&self) -> bool {
        matches!(&self.data, Material::PBR(p) if p.normal_texture.is_some())
    }

    /// Whether this material has an occlusion texture, only for PBR materials.
    #[getter]
    fn has_occlusion_texture(&self) -> bool {
        matches!(&self.data, Material::PBR(p) if p.occlusion_texture.is_some())
    }

    /// Whether this material has an emissive texture, only for PBR materials.
    #[getter]
    fn has_emissive_texture(&self) -> bool {
        matches!(&self.data, Material::PBR(p) if p.emissive_texture.is_some())
    }

    fn __repr__(&self) -> String {
        let name = self.data.name();
        let kind = match &self.data {
            Material::Empty(_) => "empty",
            Material::Simple(_) => "simple",
            Material::PBR(_) => "pbr",
        };
        if name.is_empty() {
            format!("Material(kind='{}')", kind)
        } else {
            format!("Material(name='{}', kind='{}')", name, kind)
        }
    }
}

// ============================================================================
// Attribute Classes
// ============================================================================

/// A single grouping (e.g., material, group, smoothing) for faces.
#[pyclass(name = "Grouping")]
pub struct PyGrouping {
    kind: String,
    names: Py<pyo3::types::PyList>,
    indices: Py<PyArray1<i64>>,
}

#[pymethods]
impl PyGrouping {
    /// The kind of grouping: "material", "group", "smoothing", "object", or "unspecified".
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// List of unique names in this grouping.
    #[getter]
    fn names(&self, py: Python<'_>) -> Py<pyo3::types::PyList> {
        self.names.clone_ref(py)
    }

    /// Per-face index into names, shape (n_faces,). Value of -1 means no assignment.
    #[getter]
    fn indices(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        self.indices.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let names = self.names.bind(py);
        format!(
            "Grouping(kind='{}', names={}, n_faces={})",
            self.kind,
            names.len(),
            self.indices.bind(py).len()
        )
    }
}

impl PyGrouping {
    fn from_grouping(grouping: &rmesh::attributes::Grouping, py: Python<'_>) -> Self {
        let kind = match &grouping.kind {
            GroupingKind::Material => "material",
            GroupingKind::Group => "group",
            GroupingKind::Smoothing => "smoothing",
            GroupingKind::Object => "object",
            GroupingKind::Surface => "surface",
            GroupingKind::Unspecified => "unspecified",
        };
        let names = pyo3::types::PyList::new(py, &grouping.names)
            .unwrap()
            .unbind();
        let indices: Vec<i64> = grouping.indices.iter().map(|&i| i as i64).collect();
        Self {
            kind: kind.to_string(),
            names,
            indices: readonly_1d(py, indices),
        }
    }
}

/// Collection of groupings accessible by kind.
#[pyclass(name = "GroupingCollection")]
pub struct PyGroupingCollection {
    groupings: Vec<Py<PyGrouping>>,
}

#[pymethods]
impl PyGroupingCollection {
    /// Get grouping by material if present.
    #[getter]
    fn material(&self, py: Python<'_>) -> Option<Py<PyGrouping>> {
        self.find_by_kind(py, "material")
    }

    /// Get grouping by group if present.
    #[getter]
    fn group(&self, py: Python<'_>) -> Option<Py<PyGrouping>> {
        self.find_by_kind(py, "group")
    }

    /// Get grouping by smoothing if present.
    #[getter]
    fn smoothing(&self, py: Python<'_>) -> Option<Py<PyGrouping>> {
        self.find_by_kind(py, "smoothing")
    }

    /// Get grouping by object if present.
    #[getter]
    fn object(&self, py: Python<'_>) -> Option<Py<PyGrouping>> {
        self.find_by_kind(py, "object")
    }

    /// Get surface grouping if present.
    #[getter]
    fn surface(&self, py: Python<'_>) -> Option<Py<PyGrouping>> {
        self.find_by_kind(py, "surface")
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyGrouping>> {
        self.find_by_kind(py, key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("No grouping '{}'", key)))
    }

    fn __contains__(&self, py: Python<'_>, key: &str) -> bool {
        self.find_by_kind(py, key).is_some()
    }

    fn __len__(&self) -> usize {
        self.groupings.len()
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyGroupingIterator>> {
        let cloned: Vec<Py<PyGrouping>> = self.groupings.iter().map(|g| g.clone_ref(py)).collect();
        Py::new(
            py,
            PyGroupingIterator {
                groupings: cloned,
                index: 0,
            },
        )
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let kinds: Vec<String> = self
            .groupings
            .iter()
            .map(|g| g.borrow(py).kind.clone())
            .collect();
        format!("GroupingCollection([{}])", kinds.join(", "))
    }

    /// List all available grouping kinds.
    fn keys(&self, py: Python<'_>) -> Vec<String> {
        self.groupings
            .iter()
            .map(|g| g.borrow(py).kind.clone())
            .collect()
    }
}

impl PyGroupingCollection {
    fn find_by_kind(&self, py: Python<'_>, kind: &str) -> Option<Py<PyGrouping>> {
        self.groupings
            .iter()
            .find(|g| g.borrow(py).kind == kind)
            .map(|g| g.clone_ref(py))
    }
}

#[pyclass]
struct PyGroupingIterator {
    groupings: Vec<Py<PyGrouping>>,
    index: usize,
}

#[pymethods]
impl PyGroupingIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<Py<PyGrouping>> {
        if self.index < self.groupings.len() {
            let item = self.groupings[self.index].clone_ref(py);
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

/// Access to vertex attribute arrays (uv, normals, colors).
#[pyclass(name = "VertexAttributes")]
pub struct PyVertexAttributes {
    uv: Py<pyo3::types::PyList>,
    normals: Py<pyo3::types::PyList>,
    colors: Py<pyo3::types::PyList>,
}

#[pymethods]
impl PyVertexAttributes {
    /// List of UV coordinate arrays, each shape (n_vertices, 2).
    #[getter]
    fn uv(&self, py: Python<'_>) -> Py<pyo3::types::PyList> {
        self.uv.clone_ref(py)
    }

    /// List of normal arrays, each shape (n_vertices, 3).
    #[getter]
    fn normals(&self, py: Python<'_>) -> Py<pyo3::types::PyList> {
        self.normals.clone_ref(py)
    }

    /// List of color arrays, each shape (n_vertices, 4) as RGBA u8.
    #[getter]
    fn colors(&self, py: Python<'_>) -> Py<pyo3::types::PyList> {
        self.colors.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "VertexAttributes(uv={}, normals={}, colors={})",
            self.uv.bind(py).len(),
            self.normals.bind(py).len(),
            self.colors.bind(py).len()
        )
    }
}

/// Access to face attribute arrays (uv, normals, colors) and groupings.
#[pyclass(name = "FaceAttributes")]
pub struct PyFaceAttributes {
    uv: Py<pyo3::types::PyList>,
    normals: Py<pyo3::types::PyList>,
    colors: Py<pyo3::types::PyList>,
    groupings: Py<PyGroupingCollection>,
}

#[pymethods]
impl PyFaceAttributes {
    /// List of UV coordinate arrays, each shape (n_faces, 2).
    #[getter]
    fn uv(&self, py: Python<'_>) -> Py<pyo3::types::PyList> {
        self.uv.clone_ref(py)
    }

    /// List of normal arrays, each shape (n_faces, 3).
    #[getter]
    fn normals(&self, py: Python<'_>) -> Py<pyo3::types::PyList> {
        self.normals.clone_ref(py)
    }

    /// List of color arrays, each shape (n_faces, 4) as RGBA u8.
    #[getter]
    fn colors(&self, py: Python<'_>) -> Py<pyo3::types::PyList> {
        self.colors.clone_ref(py)
    }

    /// Collection of groupings (material, group, smoothing, object).
    #[getter]
    fn groupings(&self, py: Python<'_>) -> Py<PyGroupingCollection> {
        self.groupings.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let gc = self.groupings.borrow(py);
        format!(
            "FaceAttributes(uv={}, normals={}, colors={}, groupings={})",
            self.uv.bind(py).len(),
            self.normals.bind(py).len(),
            self.colors.bind(py).len(),
            gc.groupings.len()
        )
    }
}

/// Helper to create array lists from attribute vectors
fn create_uv_list(py: Python<'_>, uv_sets: &[rmesh::attributes::UV]) -> Py<pyo3::types::PyList> {
    let list = pyo3::types::PyList::empty(py);
    for uv in uv_sets {
        let flat: Vec<f64> = uv.iter().flat_map(|v| [v.x, v.y]).collect();
        let nd = Array2::from_shape_vec((uv.len(), 2), flat).unwrap();
        let arr = PyArray2::from_array(py, &nd);
        make_readonly(&arr);
        list.append(arr).unwrap();
    }
    list.unbind()
}

fn create_normals_list(
    py: Python<'_>,
    normal_sets: &[rmesh::attributes::Normal],
) -> Py<pyo3::types::PyList> {
    let list = pyo3::types::PyList::empty(py);
    for normals in normal_sets {
        let flat: Vec<f64> = normals.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
        let nd = Array2::from_shape_vec((normals.len(), 3), flat).unwrap();
        let arr = PyArray2::from_array(py, &nd);
        make_readonly(&arr);
        list.append(arr).unwrap();
    }
    list.unbind()
}

fn create_colors_list(
    py: Python<'_>,
    color_sets: &[rmesh::attributes::Color],
) -> Py<pyo3::types::PyList> {
    let list = pyo3::types::PyList::empty(py);
    for colors in color_sets {
        let flat: Vec<u8> = colors.iter().flat_map(|v| [v.x, v.y, v.z, v.w]).collect();
        let nd = Array2::from_shape_vec((colors.len(), 4), flat).unwrap();
        let arr = PyArray2::from_array(py, &nd);
        make_readonly(&arr);
        list.append(arr).unwrap();
    }
    list.unbind()
}

// ============================================================================
// PyPolygon2D
// ============================================================================

/// A 2D polygon with an exterior ring and optional interior holes.
#[pyclass(name = "Polygon2D")]
pub struct PyPolygon2D {
    exterior_cache: OnceCell<Py<PyArray2<f64>>>,
    interiors_cache: OnceCell<Py<pyo3::types::PyList>>,
    pub(crate) data: rmesh::path::Polygon2D,
}

#[pymethods]
impl PyPolygon2D {
    /// The exterior ring as an (N, 2) array.
    #[getter]
    fn exterior(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.exterior_cache
            .get_or_init(|| {
                let flat: Vec<f64> = self.data.exterior.iter().flat_map(|p| [p.x, p.y]).collect();
                let nd = Array2::from_shape_vec((self.data.exterior.len(), 2), flat).unwrap();
                let arr = PyArray2::from_array(py, &nd);
                make_readonly(&arr);
                arr.unbind()
            })
            .clone_ref(py)
    }

    /// Interior holes as a list of (M, 2) arrays.
    #[getter]
    fn interiors(&self, py: Python<'_>) -> Py<pyo3::types::PyList> {
        self.interiors_cache
            .get_or_init(|| {
                let list = pyo3::types::PyList::empty(py);
                for hole in &self.data.interiors {
                    let flat: Vec<f64> = hole.iter().flat_map(|p| [p.x, p.y]).collect();
                    let nd = Array2::from_shape_vec((hole.len(), 2), flat).unwrap();
                    let arr = PyArray2::from_array(py, &nd);
                    make_readonly(&arr);
                    list.append(arr).unwrap();
                }
                list.unbind()
            })
            .clone_ref(py)
    }

    /// Optional 4x4 transform from 2D polygon space back to 3D, shape (4, 4).
    #[getter]
    fn to_3d(&self, py: Python<'_>) -> Option<Py<PyArray2<f64>>> {
        self.data.to_3d.map(|m| {
            let flat: Vec<f64> = (0..4)
                .flat_map(|r| (0..4).map(move |c| m[(r, c)]))
                .collect();
            let nd = Array2::from_shape_vec((4, 4), flat).unwrap();
            let arr = PyArray2::from_array(py, &nd);
            make_readonly(&arr);
            arr.unbind()
        })
    }

    /// The signed area of this polygon (exterior minus holes).
    #[getter]
    fn area(&self) -> f64 {
        self.data.area()
    }

    /// Axis-aligned bounding box as a (2, 2) array [[min_x, min_y], [max_x, max_y]],
    /// or None if the polygon has no vertices.
    #[getter]
    fn bounds(&self, py: Python<'_>) -> Option<Py<PyArray2<f64>>> {
        self.data
            .bounds()
            .map(|(min, max)| readonly_bounds(py, vec![min.x, min.y, max.x, max.y], 2))
    }

    /// Extents of the bounding box [width, height] as a (2,) array,
    /// or None if the polygon has no vertices.
    #[getter]
    fn extents(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.data.extents().map(|e| readonly_1d(py, e.to_vec()))
    }

    /// Compute the union of this polygon with another.
    fn union(&self, py: Python<'_>, other: &PyPolygon2D) -> Vec<Py<PyPolygon2D>> {
        self.data
            .union(&other.data)
            .into_iter()
            .map(|p| wrap_polygon(py, p))
            .collect()
    }

    /// Compute the difference of this polygon minus another.
    fn difference(&self, py: Python<'_>, other: &PyPolygon2D) -> Vec<Py<PyPolygon2D>> {
        self.data
            .difference(&other.data)
            .into_iter()
            .map(|p| wrap_polygon(py, p))
            .collect()
    }

    /// Compute the intersection of this polygon with another.
    fn intersection(&self, py: Python<'_>, other: &PyPolygon2D) -> Vec<Py<PyPolygon2D>> {
        self.data
            .intersection(&other.data)
            .into_iter()
            .map(|p| wrap_polygon(py, p))
            .collect()
    }

    /// Offset the polygon boundary by `distance`.
    ///
    /// Positive = expand outward, negative = shrink inward.
    /// Returns empty list if polygon shrinks to nothing.
    fn buffer(&self, py: Python<'_>, distance: f64) -> Vec<Py<PyPolygon2D>> {
        self.data
            .buffer(distance)
            .into_iter()
            .map(|p| wrap_polygon(py, p))
            .collect()
    }

    /// Lift this 2D polygon back to 3D using the stored `to_3d` transform.
    ///
    /// Returns a `Path3D` with line segments tracing the exterior and interior
    /// rings in 3D space. Returns `None` if no `to_3d` transform is set.
    fn to_path3d(&self, py: Python<'_>) -> Option<Py<PyPath3D>> {
        self.data.to_path3d().map(|p| wrap_path3d(py, p))
    }

    /// Open an interactive 2D viewer window displaying this polygon.
    #[pyo3(signature = (*, title="rmesh 2D", width=1280, height=720, background=None))]
    fn show(
        &self,
        py: Python<'_>,
        title: &str,
        width: u32,
        height: u32,
        background: Option<[f32; 3]>,
    ) {
        use rmesh_viewer::{Viewer2D, ViewerOptions};
        let data = self.data.clone();
        let options = ViewerOptions {
            title: title.to_string(),
            width,
            height,
            background: background.unwrap_or([1.0, 1.0, 1.0]),
        };
        py.detach(|| data.show_2d_with_options(options));
    }

    fn __repr__(&self) -> String {
        format!(
            "<rmesh.Polygon2D vertices: {} holes: {}>",
            self.data.exterior.len(),
            self.data.interiors.len()
        )
    }
}

// ============================================================================
// PyPath2D
// ============================================================================

/// A 2D path made of line/arc/bezier/spline segments.
#[pyclass(name = "Path2D")]
pub struct PyPath2D {
    pub(crate) data: rmesh::path::Path2D,
    vertices_cache: OnceCell<Py<PyArray2<f64>>>,
}

#[pymethods]
impl PyPath2D {
    /// The shared vertex array as an (N, 2) array.
    #[getter]
    fn vertices(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.vertices_cache
            .get_or_init(|| {
                let flat: Vec<f64> = self.data.vertices.iter().flat_map(|p| [p.x, p.y]).collect();
                let nd = Array2::from_shape_vec((self.data.vertices.len(), 2), flat).unwrap();
                let arr = PyArray2::from_array(py, &nd);
                make_readonly(&arr);
                arr.unbind()
            })
            .clone_ref(py)
    }

    /// Extract closed polygons from this path.
    fn polygons(&self, py: Python<'_>) -> Vec<Py<PyPolygon2D>> {
        self.data
            .polygons()
            .iter()
            .map(|p| wrap_polygon(py, p.clone()))
            .collect()
    }

    /// Lift this 2D path back to 3D using the stored `to_3d` transform.
    ///
    /// Transforms all vertices through the 4x4 matrix and converts
    /// compatible segments (Line, Bezier, BSpline). Returns `None` if
    /// no `to_3d` transform is set.
    fn to_path3d(&self, py: Python<'_>) -> Option<Py<PyPath3D>> {
        self.data.to_path3d().map(|p| wrap_path3d(py, p))
    }

    /// Offset the path boundary by `distance`, preserving analytical curves.
    ///
    /// Circles and arcs from the original path are recognized in the
    /// buffered result with adjusted radii. Positive = expand outward,
    /// negative = shrink inward. Returns empty list if path shrinks to nothing.
    fn buffer(&self, py: Python<'_>, distance: f64) -> Vec<Py<PyPath2D>> {
        self.data
            .buffer(distance)
            .into_iter()
            .map(|p| wrap_path2d(py, p))
            .collect()
    }

    /// Open an interactive 2D viewer window displaying this path.
    #[pyo3(signature = (*, title="rmesh 2D", width=1280, height=720, background=None))]
    fn show(
        &self,
        py: Python<'_>,
        title: &str,
        width: u32,
        height: u32,
        background: Option<[f32; 3]>,
    ) {
        use rmesh_viewer::{Viewer2D, ViewerOptions};
        let data = self.data.clone();
        let options = ViewerOptions {
            title: title.to_string(),
            width,
            height,
            background: background.unwrap_or([1.0, 1.0, 1.0]),
        };
        py.detach(|| data.show_2d_with_options(options));
    }

    /// Segment descriptions as a list of dicts.
    ///
    /// Each dict contains at minimum a ``"kind"`` key (``"line"``, ``"arc"``,
    /// ``"circle"``, ``"ellipse"``, ``"cubic_bezier"``, ``"quadratic_bezier"``,
    /// ``"bspline"``).  Circles include ``"center"`` (2-tuple) and ``"radius"``.
    /// Arcs include ``"start"``/``"finish"`` (2-tuples), ``"center"`` (2-tuple),
    /// ``"radius"``, and ``"angle"`` (sweep in radians).  Lines include a
    /// ``"points"`` list of vertex indices.
    #[getter]
    fn segments(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        use rmesh::path::Segment2D;
        let verts = &self.data.vertices;
        let mut out = Vec::with_capacity(self.data.segments.len());
        for seg in &self.data.segments {
            let d = PyDict::new(py);
            match seg {
                Segment2D::Line(l) => {
                    d.set_item("kind", "line")?;
                    d.set_item("points", &l.points)?;
                }
                Segment2D::Circle(c) => {
                    d.set_item("kind", "circle")?;
                    let center = &verts[c.center];
                    d.set_item("center", (center.x, center.y))?;
                    d.set_item("radius", c.radius)?;
                    d.set_item("center_index", c.center)?;
                }
                Segment2D::Arc(a) => {
                    d.set_item("kind", "arc")?;
                    let s = &verts[a.start];
                    let f = &verts[a.finish];
                    d.set_item("start", (s.x, s.y))?;
                    d.set_item("finish", (f.x, f.y))?;
                    d.set_item("angle", a.sweep_angle())?;
                    if let Some(center) = a.center(verts) {
                        d.set_item("center", (center.x, center.y))?;
                    }
                    if let Some(radius) = a.radius(verts) {
                        d.set_item("radius", radius)?;
                    }
                    d.set_item("start_index", a.start)?;
                    d.set_item("finish_index", a.finish)?;
                }
                Segment2D::Ellipse(_) => {
                    d.set_item("kind", "ellipse")?;
                }
                Segment2D::CubicBezier(_) => {
                    d.set_item("kind", "cubic_bezier")?;
                }
                Segment2D::QuadraticBezier(_) => {
                    d.set_item("kind", "quadratic_bezier")?;
                }
                Segment2D::BSpline(_) => {
                    d.set_item("kind", "bspline")?;
                }
            }
            out.push(d.unbind());
        }
        Ok(out)
    }

    /// Discretize all segments into polylines.
    ///
    /// Returns a list of (M, 2) arrays, one per segment.
    fn discretize(&self, py: Python<'_>) -> Vec<Py<PyArray2<f64>>> {
        self.data
            .discretize()
            .into_iter()
            .map(|pts| {
                let flat: Vec<f64> = pts.iter().flat_map(|p| [p.x, p.y]).collect();
                let nd = Array2::from_shape_vec((pts.len(), 2), flat).unwrap();
                let arr = PyArray2::from_array(py, &nd);
                make_readonly(&arr);
                arr.unbind()
            })
            .collect()
    }

    /// Axis-aligned bounding box as a (2, 2) array [[min_x, min_y], [max_x, max_y]],
    /// or None if the path has no vertices.
    #[getter]
    fn bounds(&self, py: Python<'_>) -> Option<Py<PyArray2<f64>>> {
        self.data
            .bounds()
            .map(|(min, max)| readonly_bounds(py, vec![min.x, min.y, max.x, max.y], 2))
    }

    /// Extents of the bounding box [width, height] as a (2,) array,
    /// or None if the path has no vertices.
    #[getter]
    fn extents(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.data.extents().map(|e| readonly_1d(py, e.to_vec()))
    }

    fn __repr__(&self) -> String {
        format!(
            "<rmesh.Path2D vertices: {} segments: {}>",
            self.data.vertices.len(),
            self.data.segments.len()
        )
    }
}

// ============================================================================
// PyPath3D
// ============================================================================

/// A 3D path consisting of vertices and line segments.
#[pyclass(name = "Path3D")]
pub struct PyPath3D {
    vertices_cache: OnceCell<Py<PyArray2<f64>>>,
    pub(crate) data: rmesh::path::Path3D,
}

#[pymethods]
impl PyPath3D {
    /// The path vertices as an (N, 3) array.
    #[getter]
    fn vertices(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.vertices_cache
            .get_or_init(|| {
                let flat: Vec<f64> = self
                    .data
                    .vertices
                    .iter()
                    .flat_map(|p| [p.x, p.y, p.z])
                    .collect();
                let nd = Array2::from_shape_vec((self.data.vertices.len(), 3), flat).unwrap();
                let arr = PyArray2::from_array(py, &nd);
                make_readonly(&arr);
                arr.unbind()
            })
            .clone_ref(py)
    }

    /// Discretize the path into a list of (M, 3) polylines.
    fn discretize(&self, py: Python<'_>) -> Vec<Py<PyArray2<f64>>> {
        self.data
            .discretize()
            .into_iter()
            .map(|pts| {
                let flat: Vec<f64> = pts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
                let nd = Array2::from_shape_vec((pts.len(), 3), flat).unwrap();
                let arr = PyArray2::from_array(py, &nd);
                make_readonly(&arr);
                arr.unbind()
            })
            .collect()
    }

    /// Axis-aligned bounding box as a (2, 3) array [[min_x, min_y, min_z], [max_x, max_y, max_z]],
    /// or None if the path has no vertices.
    #[getter]
    fn bounds(&self, py: Python<'_>) -> Option<Py<PyArray2<f64>>> {
        self.data.bounds().map(|(min, max)| {
            readonly_bounds(py, vec![min.x, min.y, min.z, max.x, max.y, max.z], 3)
        })
    }

    /// Extents of the bounding box [x, y, z] as a (3,) array,
    /// or None if the path has no vertices.
    #[getter]
    fn extents(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.data.extents().map(|e| readonly_1d(py, e.to_vec()))
    }

    fn __repr__(&self) -> String {
        format!(
            "<rmesh.Path3D vertices: {} segments: {}>",
            self.data.vertices.len(),
            self.data.segments.len()
        )
    }
}

/// Wrap a `Path3D` into a Python `PyPath3D` object.
pub(crate) fn wrap_path3d(py: Python<'_>, data: rmesh::path::Path3D) -> Py<PyPath3D> {
    Py::new(
        py,
        PyPath3D {
            vertices_cache: OnceCell::new(),
            data,
        },
    )
    .unwrap()
}

// ============================================================================
// Helpers
// ============================================================================

/// Wrap a `Path2D` into a Python `PyPath2D` object.
pub(crate) fn wrap_path2d(py: Python<'_>, data: rmesh::path::Path2D) -> Py<PyPath2D> {
    Py::new(
        py,
        PyPath2D {
            data,
            vertices_cache: OnceCell::new(),
        },
    )
    .unwrap()
}

/// Wrap a `Polygon2D` into a Python `PyPolygon2D` object.
pub(crate) fn wrap_polygon(py: Python<'_>, data: rmesh::path::Polygon2D) -> Py<PyPolygon2D> {
    Py::new(
        py,
        PyPolygon2D {
            exterior_cache: OnceCell::new(),
            interiors_cache: OnceCell::new(),
            data,
        },
    )
    .unwrap()
}

/// Make a numpy array read-only by clearing the WRITEABLE flag.
pub(crate) fn make_readonly<T: numpy::Element, D: numpy::ndarray::Dimension>(
    arr: &Bound<'_, numpy::PyArray<T, D>>,
) {
    unsafe {
        (*arr.as_array_ptr()).flags &= !npyffi::flags::NPY_ARRAY_WRITEABLE;
    }
}

/// Create a read-only 1D numpy array from a Vec.
pub(crate) fn readonly_1d<T: numpy::Element>(py: Python<'_>, data: Vec<T>) -> Py<PyArray1<T>> {
    let arr = PyArray1::from_vec(py, data);
    make_readonly(&arr);
    arr.unbind()
}

/// Create a read-only (2, cols) numpy array for bounds data.
pub(crate) fn readonly_bounds(py: Python<'_>, data: Vec<f64>, cols: usize) -> Py<PyArray2<f64>> {
    let nd = Array2::from_shape_vec((2, cols), data).unwrap();
    let arr = PyArray2::from_array(py, &nd);
    make_readonly(&arr);
    arr.unbind()
}

macro_rules! cached_array {
    ($self:expr, $py:expr, $cache:ident, $data:expr, $cols:expr) => {{
        $self
            .$cache
            .get_or_init(|| {
                let flat: &[_] = bytemuck::cast_slice($data);
                let nd = Array2::from_shape_vec(($data.len(), $cols), flat.to_vec()).unwrap();
                let arr = PyArray2::from_array($py, &nd);
                make_readonly(&arr);
                arr.unbind()
            })
            .clone_ref($py)
    }};
}

// ============================================================================
// PyTrimesh
// ============================================================================

#[pyclass(name = "Trimesh")]
pub struct PyTrimesh {
    pub(crate) data: Trimesh,
    vertices_cache: OnceCell<Py<PyArray2<f64>>>,
    faces_cache: OnceCell<Py<PyArray2<i64>>>,
    face_normals_cache: OnceCell<Py<PyArray2<f64>>>,
    edges_cache: OnceCell<Py<PyArray2<i64>>>,
    face_adjacency_cache: OnceCell<Py<PyArray2<i64>>>,
    face_adjacency_angles_cache: OnceCell<Py<PyArray1<f64>>>,
    vertex_attributes_cache: OnceCell<Py<PyVertexAttributes>>,
    face_attributes_cache: OnceCell<Py<PyFaceAttributes>>,
}

impl PyTrimesh {
    pub fn new_from_trimesh(data: Trimesh) -> Self {
        Self {
            data,
            vertices_cache: OnceCell::new(),
            faces_cache: OnceCell::new(),
            face_normals_cache: OnceCell::new(),
            edges_cache: OnceCell::new(),
            face_adjacency_cache: OnceCell::new(),
            face_adjacency_angles_cache: OnceCell::new(),
            vertex_attributes_cache: OnceCell::new(),
            face_attributes_cache: OnceCell::new(),
        }
    }
}

#[pymethods]
impl PyTrimesh {
    #[new]
    #[pyo3(signature = (vertices, faces, *, vertex_normals=None, face_colors=None, face_surfaces=None))]
    pub fn new(
        _py: Python<'_>,
        vertices: PyReadonlyArray2<'_, f64>,
        faces: PyReadonlyArray2<'_, i64>,
        vertex_normals: Option<PyReadonlyArray2<'_, f64>>,
        face_colors: Option<PyReadonlyArray2<'_, u8>>,
        face_surfaces: Option<(Vec<Bound<'_, PyDict>>, PyReadonlyArray1<'_, i64>)>,
    ) -> PyResult<Self> {
        let vertices: Vec<Point3<f64>> = vertices
            .as_array()
            .rows()
            .into_iter()
            .map(|r| Point3::new(r[0], r[1], r[2]))
            .collect();
        let faces: Vec<[usize; 3]> = faces
            .as_array()
            .rows()
            .into_iter()
            .map(|r| [r[0] as usize, r[1] as usize, r[2] as usize])
            .collect();

        let mut attr_vertex = Attributes::default();
        let mut attr_face = Attributes::default();

        if let Some(n) = vertex_normals {
            attr_vertex.normals.push(
                n.as_array()
                    .rows()
                    .into_iter()
                    .map(|r| Vector3::new(r[0], r[1], r[2]))
                    .collect(),
            );
        }
        if let Some(c) = face_colors {
            attr_face.colors.push(
                c.as_array()
                    .rows()
                    .into_iter()
                    .map(|r| Vector4::new(r[0], r[1], r[2], r[3]))
                    .collect(),
            );
        }

        let mut data = Trimesh::new(vertices, faces, Some(attr_vertex), Some(attr_face))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Parse face_surfaces kwarg: (list[dict], ndarray[int64])
        if let Some((surface_dicts, face_index)) = face_surfaces {
            let (surfaces, names) = parse_surface_dicts(&surface_dicts)?;
            let indices: Vec<usize> = face_index.as_array().iter().map(|&i| i as usize).collect();

            data.face_surfaces = surfaces;
            data.attributes_face.groupings.push(Grouping {
                kind: GroupingKind::Surface,
                names,
                indices,
            });
        }

        Ok(Self::new_from_trimesh(data))
    }

    #[getter]
    fn vertices(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        cached_array!(self, py, vertices_cache, &self.data.vertices, 3)
    }

    #[getter]
    fn faces(&self, py: Python<'_>) -> Py<PyArray2<i64>> {
        // faces are [usize; 3], need to convert to i64 for numpy
        self.faces_cache
            .get_or_init(|| {
                let flat: Vec<i64> = self
                    .data
                    .faces
                    .iter()
                    .flat_map(|f| [f[0] as i64, f[1] as i64, f[2] as i64])
                    .collect();
                let nd = Array2::from_shape_vec((self.data.faces.len(), 3), flat).unwrap();
                let arr = PyArray2::from_array(py, &nd);
                make_readonly(&arr);
                arr.unbind()
            })
            .clone_ref(py)
    }

    /// Full access to vertex attributes (multiple UV sets, normals, colors).
    #[getter]
    fn vertex_attributes(&self, py: Python<'_>) -> Py<PyVertexAttributes> {
        self.vertex_attributes_cache
            .get_or_init(|| {
                Py::new(
                    py,
                    PyVertexAttributes {
                        uv: create_uv_list(py, &self.data.attributes_vertex.uv),
                        normals: create_normals_list(py, &self.data.attributes_vertex.normals),
                        colors: create_colors_list(py, &self.data.attributes_vertex.colors),
                    },
                )
                .unwrap()
            })
            .clone_ref(py)
    }

    /// Full access to face attributes (multiple UV sets, normals, colors, groupings).
    #[getter]
    fn face_attributes(&self, py: Python<'_>) -> Py<PyFaceAttributes> {
        self.face_attributes_cache
            .get_or_init(|| {
                let groupings: Vec<Py<PyGrouping>> = self
                    .data
                    .attributes_face
                    .groupings
                    .iter()
                    .map(|g| Py::new(py, PyGrouping::from_grouping(g, py)).unwrap())
                    .collect();
                let gc = Py::new(py, PyGroupingCollection { groupings }).unwrap();
                Py::new(
                    py,
                    PyFaceAttributes {
                        uv: create_uv_list(py, &self.data.attributes_face.uv),
                        normals: create_normals_list(py, &self.data.attributes_face.normals),
                        colors: create_colors_list(py, &self.data.attributes_face.colors),
                        groupings: gc,
                    },
                )
                .unwrap()
            })
            .clone_ref(py)
    }

    #[getter]
    fn face_normals(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.face_normals_cache
            .get_or_init(|| {
                let normals = self.data.face_normals();
                let flat: Vec<f64> = normals.iter().flat_map(|n| [n.x, n.y, n.z]).collect();
                let nd = Array2::from_shape_vec((normals.len(), 3), flat).unwrap();
                let arr = PyArray2::from_array(py, &nd);
                make_readonly(&arr);
                arr.unbind()
            })
            .clone_ref(py)
    }

    #[getter]
    fn edges(&self, py: Python<'_>) -> Py<PyArray2<i64>> {
        self.edges_cache
            .get_or_init(|| {
                let edges = self.data.edges();
                let flat: Vec<i64> = edges
                    .iter()
                    .flat_map(|e| [e[0] as i64, e[1] as i64])
                    .collect();
                let nd = Array2::from_shape_vec((edges.len(), 2), flat).unwrap();
                let arr = PyArray2::from_array(py, &nd);
                make_readonly(&arr);
                arr.unbind()
            })
            .clone_ref(py)
    }

    #[getter]
    fn bounds(&self, py: Python<'_>) -> Option<Py<PyArray2<f64>>> {
        self.data.bounds().map(|(min, max)| {
            readonly_bounds(py, vec![min.x, min.y, min.z, max.x, max.y, max.z], 3)
        })
    }

    #[getter]
    fn area(&self) -> f64 {
        self.data.area()
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.data.volume()
    }

    #[getter]
    fn mass(&self) -> f64 {
        self.data.mass()
    }

    #[getter]
    fn center_mass(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        let cm = self.data.center_mass();
        readonly_1d(py, vec![cm.x, cm.y, cm.z])
    }

    #[getter]
    fn moment_inertia(&self, py: Python<'_>) -> Option<Py<PyArray2<f64>>> {
        self.data.moment_inertia().map(|m| {
            let flat: Vec<f64> = vec![
                m[(0, 0)],
                m[(0, 1)],
                m[(0, 2)],
                m[(1, 0)],
                m[(1, 1)],
                m[(1, 2)],
                m[(2, 0)],
                m[(2, 1)],
                m[(2, 2)],
            ];
            let nd = Array2::from_shape_vec((3, 3), flat).unwrap();
            let arr = PyArray2::from_array(py, &nd);
            make_readonly(&arr);
            arr.unbind()
        })
    }

    #[getter]
    fn is_watertight(&self) -> bool {
        self.data.is_watertight()
    }

    #[getter]
    fn is_winding_consistent(&self) -> bool {
        self.data.is_winding_consistent()
    }

    #[getter]
    fn is_volume(&self) -> bool {
        self.data.is_volume()
    }

    #[getter]
    fn euler_number(&self) -> i64 {
        self.data.euler_number()
    }

    #[getter]
    fn edges_unique(&self, py: Python<'_>) -> Py<PyArray2<i64>> {
        // Not cached since edges_unique is already cached in Rust
        let edges = self.data.edges_unique();
        let flat: Vec<i64> = edges
            .iter()
            .flat_map(|e| [e[0] as i64, e[1] as i64])
            .collect();
        let nd = Array2::from_shape_vec((edges.len(), 2), flat).unwrap();
        let arr = PyArray2::from_array(py, &nd);
        make_readonly(&arr);
        arr.unbind()
    }

    /// All edges with each pair sorted [min, max], shape (n_faces * 3, 2).
    #[getter]
    fn edges_sorted(&self, py: Python<'_>) -> Py<PyArray2<i64>> {
        let edges = self.data.edges_sorted();
        let flat: Vec<i64> = edges
            .iter()
            .flat_map(|e| [e.edge[0] as i64, e.edge[1] as i64])
            .collect();
        let nd = Array2::from_shape_vec((edges.len(), 2), flat).unwrap();
        let arr = PyArray2::from_array(py, &nd);
        make_readonly(&arr);
        arr.unbind()
    }

    /// Geometric length of each unique edge, shape (n_edges_unique,).
    #[getter]
    fn edges_unique_length(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        readonly_1d(py, self.data.edges_unique_length())
    }

    /// Index mapping: edges_sorted[i] -> edges_unique index, shape (n_faces * 3,).
    #[getter]
    fn edges_unique_inverse(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        let flat: Vec<i64> = self
            .data
            .edges_unique_inverse()
            .iter()
            .map(|&i| i as i64)
            .collect();
        readonly_1d(py, flat)
    }

    /// Principal inertia components (eigenvalues of inertia tensor, sorted descending).
    #[getter]
    fn principal_inertia_components(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.data
            .principal_inertia_components()
            .map(|v| readonly_1d(py, vec![v.x, v.y, v.z]))
    }

    /// Principal inertia vectors (eigenvectors as matrix columns), shape (3, 3).
    #[getter]
    fn principal_inertia_vectors(&self, py: Python<'_>) -> Option<Py<PyArray2<f64>>> {
        self.data.principal_inertia_vectors().map(|m| {
            let flat: Vec<f64> = vec![
                m[(0, 0)],
                m[(0, 1)],
                m[(0, 2)],
                m[(1, 0)],
                m[(1, 1)],
                m[(1, 2)],
                m[(2, 0)],
                m[(2, 1)],
                m[(2, 2)],
            ];
            let nd = Array2::from_shape_vec((3, 3), flat).unwrap();
            let arr = PyArray2::from_array(py, &nd);
            make_readonly(&arr);
            arr.unbind()
        })
    }

    #[getter]
    fn face_adjacency(&self, py: Python<'_>) -> Py<PyArray2<i64>> {
        self.face_adjacency_cache
            .get_or_init(|| {
                let adj = self.data.face_adjacency();
                let flat: Vec<i64> = adj
                    .iter()
                    .flat_map(|(a, b)| [*a as i64, *b as i64])
                    .collect();
                let nd = Array2::from_shape_vec((adj.len(), 2), flat).unwrap();
                let arr = PyArray2::from_array(py, &nd);
                make_readonly(&arr);
                arr.unbind()
            })
            .clone_ref(py)
    }

    #[getter]
    fn face_adjacency_angles(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.face_adjacency_angles_cache
            .get_or_init(|| readonly_1d(py, self.data.face_adjacency_angles()))
            .clone_ref(py)
    }

    /// Per-face areas.
    #[getter]
    fn area_faces(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        // Copy data into numpy-owned memory to avoid use-after-free if mesh is GC'd
        readonly_1d(py, self.data.faces_area().to_vec())
    }

    /// Cross product vectors for each face (unnormalized face normals * 2 * area).
    #[getter]
    fn faces_cross(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        let crosses = self.data.faces_cross();
        let flat: Vec<f64> = crosses.iter().flat_map(|c| [c.x, c.y, c.z]).collect();
        let nd = Array2::from_shape_vec((crosses.len(), 3), flat).unwrap();
        let arr = PyArray2::from_array(py, &nd);
        make_readonly(&arr);
        arr.unbind()
    }

    /// Extents of the bounding box (max - min for each axis).
    #[getter]
    fn extents(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.data.extents().map(|e| readonly_1d(py, e.to_vec()))
    }

    /// Geometric center of the vertices (mean position).
    #[getter]
    fn centroid(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.data
            .centroid()
            .map(|c| readonly_1d(py, vec![c.x, c.y, c.z]))
    }

    /// Actual vertex positions for each face, shape (n_faces, 3, 3).
    #[getter]
    fn triangles(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        let flat = self.data.triangles_flat();
        let n_faces = self.data.faces.len();
        // Return as (n_faces * 3, 3) - caller can reshape to (n_faces, 3, 3)
        let nd = Array2::from_shape_vec((n_faces * 3, 3), flat).unwrap();
        let arr = PyArray2::from_array(py, &nd);
        make_readonly(&arr);
        arr.unbind()
    }

    /// Center of each triangle, shape (n_faces, 3).
    #[getter]
    fn triangles_center(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        let centers = self.data.triangles_center();
        let flat: Vec<f64> = centers.iter().flat_map(|c| [c.x, c.y, c.z]).collect();
        let nd = Array2::from_shape_vec((centers.len(), 3), flat).unwrap();
        let arr = PyArray2::from_array(py, &nd);
        make_readonly(&arr);
        arr.unbind()
    }

    /// For each adjacent face pair, the vertex indices not on the shared edge.
    /// Shape (n_adjacency, 2).
    #[getter]
    fn face_adjacency_unshared(&self, py: Python<'_>) -> Py<PyArray2<i64>> {
        let unshared = self.data.face_adjacency_unshared();
        let flat: Vec<i64> = unshared
            .iter()
            .flat_map(|[a, b]| [*a as i64, *b as i64])
            .collect();
        let nd = Array2::from_shape_vec((unshared.len(), 2), flat).unwrap();
        let arr = PyArray2::from_array(py, &nd);
        make_readonly(&arr);
        arr.unbind()
    }

    /// Projection of unshared vertices onto adjacent face planes.
    /// Negative values indicate locally convex geometry.
    #[getter]
    fn face_adjacency_projections(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        readonly_1d(py, self.data.face_adjacency_projections())
    }

    /// Boolean array indicating whether each adjacent face pair is locally convex.
    #[getter]
    fn face_adjacency_convex(&self, py: Python<'_>) -> Py<PyArray1<bool>> {
        readonly_1d(py, self.data.face_adjacency_convex())
    }

    /// Check if the mesh is convex.
    #[getter]
    fn is_convex(&self) -> bool {
        self.data.is_convex()
    }

    /// List of materials attached to this mesh.
    #[getter]
    fn materials(&self, py: Python<'_>) -> Vec<Py<PyMaterial>> {
        self.data
            .materials
            .iter()
            .map(|m| Py::new(py, PyMaterial { data: m.clone() }).unwrap())
            .collect()
    }

    /// The convex hull of this mesh as a new Trimesh.
    ///
    /// Parameters
    /// ----------
    /// compact : bool
    ///     If true (default) only vertices on the hull are kept and face
    ///     indices are remapped.  Set to false to preserve all referenced
    ///     source vertices. Both modes filter unreferenced vertices.
    #[pyo3(signature = (compact = true))]
    fn convex_hull(&self, compact: bool) -> Self {
        Self::new_from_trimesh(self.data.convex_hull(compact).into_owned())
    }

    /// Face surface data from BREP/CAD source, if available.
    ///
    /// Returns ``None`` if no surface data is present, otherwise a tuple of
    /// ``(surfaces, face_index)`` where ``surfaces`` is a list of dicts and
    /// ``face_index`` is a per-face int64 index array.
    #[getter]
    fn face_surfaces(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.data.face_surfaces.is_empty() {
            return None;
        }

        let surfaces = pyo3::types::PyList::empty(py);
        for s in &self.data.face_surfaces {
            let dict = PyDict::new(py);
            match s {
                rmesh::boundary::Surface::Plane(p) => {
                    dict.set_item("kind", "Plane").unwrap();
                    dict.set_item("origin", [p.origin.x, p.origin.y, p.origin.z])
                        .unwrap();
                    dict.set_item("normal", [p.normal.x, p.normal.y, p.normal.z])
                        .unwrap();
                }
                rmesh::boundary::Surface::Cylinder(c) => {
                    dict.set_item("kind", "Cylinder").unwrap();
                    dict.set_item("origin", [c.origin.x, c.origin.y, c.origin.z])
                        .unwrap();
                    dict.set_item("axis", [c.axis.x, c.axis.y, c.axis.z])
                        .unwrap();
                    dict.set_item("radius", c.radius).unwrap();
                }
                rmesh::boundary::Surface::Cone(c) => {
                    dict.set_item("kind", "Cone").unwrap();
                    dict.set_item("apex", [c.apex.x, c.apex.y, c.apex.z])
                        .unwrap();
                    dict.set_item("axis", [c.axis.x, c.axis.y, c.axis.z])
                        .unwrap();
                    dict.set_item("half_angle", c.half_angle).unwrap();
                }
                rmesh::boundary::Surface::Sphere(s) => {
                    dict.set_item("kind", "Sphere").unwrap();
                    dict.set_item("center", [s.center.x, s.center.y, s.center.z])
                        .unwrap();
                    dict.set_item("radius", s.radius).unwrap();
                }
                rmesh::boundary::Surface::Torus(t) => {
                    dict.set_item("kind", "Torus").unwrap();
                    dict.set_item("center", [t.center.x, t.center.y, t.center.z])
                        .unwrap();
                    dict.set_item("axis", [t.axis.x, t.axis.y, t.axis.z])
                        .unwrap();
                    dict.set_item("major_radius", t.major_radius).unwrap();
                    dict.set_item("minor_radius", t.minor_radius).unwrap();
                }
            }
            surfaces.append(dict).unwrap();
        }

        // Find the Surface grouping to get the face index
        let face_index = self
            .data
            .attributes_face
            .groupings
            .iter()
            .find(|g| matches!(g.kind, GroupingKind::Surface))
            .map(|g| {
                let indices: Vec<i64> = g
                    .indices
                    .iter()
                    .map(|&i| if i == UNSET { -1 } else { i as i64 })
                    .collect();
                readonly_1d(py, indices)
            });

        let result = pyo3::types::PyTuple::new(
            py,
            &[
                surfaces.into_any(),
                match face_index {
                    Some(arr) => arr.into_bound(py).into_any(),
                    None => py.None().into_bound(py),
                },
            ],
        )
        .unwrap();
        Some(result.unbind().into())
    }

    /// Project the mesh onto a plane at multiple levels.
    ///
    /// When BREP data is available (via ``face_surfaces``), circles and arcs
    /// from aligned cylinders are preserved as analytical entities in the
    /// returned ``Path2D`` objects.
    ///
    /// Parameters
    /// ----------
    /// normal : (3,) float
    ///     The plane normal direction.
    /// origin : (3,) float
    ///     A point on the plane.
    /// levels : (N,) float
    ///     Height offsets along the normal to project at.
    ///
    /// Returns
    /// -------
    /// list of (list of Path2D or None)
    ///     One entry per level. None if no geometry intersects that level.
    fn project(
        &self,
        py: Python<'_>,
        normal: PyReadonlyArray1<'_, f64>,
        origin: PyReadonlyArray1<'_, f64>,
        levels: PyReadonlyArray1<'_, f64>,
    ) -> Vec<Option<Vec<Py<PyPath2D>>>> {
        let n = normal.as_array();
        let o = origin.as_array();
        let normal = Vector3::new(n[0], n[1], n[2]);
        let origin = Point3::new(o[0], o[1], o[2]);
        let levels: Vec<f64> = levels.as_array().to_vec();

        self.data
            .project(&normal, &origin, &levels)
            .into_iter()
            .map(|opt| opt.map(|paths| paths.into_iter().map(|p| wrap_path2d(py, p)).collect()))
            .collect()
    }

    /// Decompose the mesh into approximate convex parts.
    #[pyo3(signature = (max_hulls=64, resolution=400_000))]
    fn decompose(&self, max_hulls: u32, resolution: u32) -> Vec<Self> {
        let params = rmesh::decomposition::DecompositionParams {
            max_convex_hulls: max_hulls,
            resolution,
            ..Default::default()
        };
        self.data
            .decompose(&params)
            .into_iter()
            .map(Self::new_from_trimesh)
            .collect()
    }

    /// Voxelize the mesh at the given resolution.
    #[pyo3(signature = (resolution=100_000, fill_mode="flood"))]
    fn voxelize(&self, resolution: u32, fill_mode: &str) -> PyResult<PyVoxelGrid> {
        let mode = match fill_mode {
            "flood" => rmesh::voxel::FillMode::FloodFill,
            "surface" => rmesh::voxel::FillMode::SurfaceOnly,
            "raycast" => rmesh::voxel::FillMode::Raycast,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown fill_mode: '{}'. Use 'flood', 'surface', or 'raycast'",
                    fill_mode
                )));
            }
        };
        Ok(PyVoxelGrid {
            data: self.data.voxelize(resolution, mode),
        })
    }

    /// Split the mesh into sub-meshes.
    ///
    /// Parameters
    /// ----------
    /// on : str, optional
    ///     If None (default), split by connected components.
    ///     If a string, split by the named face grouping attribute:
    ///     "material", "group", "smoothing", "object", or "surface".
    ///
    /// Returns
    /// -------
    /// list of Trimesh
    ///     The sub-meshes. Empty list for empty meshes.
    #[pyo3(signature = (*, on=None))]
    fn split(&self, on: Option<&str>) -> PyResult<Vec<Self>> {
        let kind = match on {
            None => None,
            Some(s) => {
                let k = match s {
                    "material" => GroupingKind::Material,
                    "group" => GroupingKind::Group,
                    "smoothing" => GroupingKind::Smoothing,
                    "object" => GroupingKind::Object,
                    "surface" => GroupingKind::Surface,
                    other => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Unknown grouping kind: '{}'. Use 'material', 'group', 'smoothing', 'object', or 'surface'",
                            other
                        )));
                    }
                };
                Some(k)
            }
        };
        Ok(self
            .data
            .split(kind.as_ref())
            .into_iter()
            .map(Self::new_from_trimesh)
            .collect())
    }

    /// Directed boundary edges (edges shared by exactly one face).
    ///
    /// Shape (N, 2) where each row is a directed edge [v0, v1] preserving
    /// the winding from the original face. Empty if the mesh is watertight.
    #[getter]
    fn edges_boundary(&self, py: Python<'_>) -> Py<PyArray2<i64>> {
        let edges = self.data.edges_boundary();
        let flat: Vec<i64> = edges
            .iter()
            .flat_map(|e| [e[0] as i64, e[1] as i64])
            .collect();
        let nd = Array2::from_shape_vec((edges.len(), 2), flat).unwrap();
        let arr = PyArray2::from_array(py, &nd);
        make_readonly(&arr);
        arr.unbind()
    }

    /// Fill simple holes by fan-triangulating boundary loops.
    ///
    /// Returns a new mesh with additional faces closing each hole.
    /// Check ``is_watertight`` on the result to see if all holes were filled.
    fn fill_holes(&self) -> Self {
        Self::new_from_trimesh(self.data.fill_holes())
    }

    fn __repr__(&self) -> String {
        format!(
            "<rmesh.Trimesh vertices: ({}, 3) faces: ({}, 3)>",
            self.data.vertices.len(),
            self.data.faces.len()
        )
    }

    #[pyo3(signature = (target_faces=None, aggressiveness=None))]
    fn simplify(&self, target_faces: Option<usize>, aggressiveness: Option<f64>) -> Self {
        let options = rmesh::simplify::SimplifyOptions {
            target_count: target_faces,
            aggressiveness: aggressiveness.unwrap_or(7.0),
            preserve_attributes: true,
            ..Default::default()
        };
        Self::new_from_trimesh(self.data.simplify(&options).into())
    }

    /// Clean up the mesh by merging vertices, removing degenerate faces, etc.
    ///
    /// Unlike trimesh which uses global tolerances, all options are passed explicitly.
    ///
    /// Parameters
    /// ----------
    /// merge_vertices : int, optional
    ///     If provided, merge duplicate vertices at this decimal precision.
    ///     Example: 8 means vertices within 1e-8 are considered equal.
    /// merge_tex : int, optional
    ///     If provided, also consider UV coordinates at this precision when merging.
    ///     If None, UVs are ignored (vertices merged even if UVs differ).
    /// merge_norm : int, optional
    ///     If provided, also consider vertex normals at this precision when merging.
    ///     If None, normals are ignored (vertices merged even if normals differ).
    /// remove_degenerate : int, optional
    ///     If provided, remove faces where any two vertices are equal at this
    ///     decimal precision. Example: 8 means vertices within 1e-8 are considered
    ///     equal and triangles with duplicate vertices are removed.
    /// remove_infinite : bool
    ///     Remove NaN/Inf values (default: False)
    /// remove_unreferenced : bool
    ///     Remove unreferenced vertices (default: False)
    ///
    /// Returns
    /// -------
    /// Trimesh
    ///     A new cleaned mesh
    ///
    /// Examples
    /// --------
    /// >>> mesh = rmesh.load("model.stl")
    /// >>> # Merge vertices at 1e-8 precision (like trimesh default)
    /// >>> cleaned = mesh.cleanup(merge_vertices=8)
    /// >>> # Full cleanup like trimesh.process(validate=True)
    /// >>> cleaned = mesh.cleanup(merge_vertices=8, remove_degenerate=12, remove_infinite=True)
    #[pyo3(signature = (
        merge_vertices=None,
        merge_tex=None,
        merge_norm=None,
        remove_degenerate=None,
        remove_infinite=false,
        remove_unreferenced=false
    ))]
    fn cleanup(
        &self,
        merge_vertices: Option<u32>,
        merge_tex: Option<u32>,
        merge_norm: Option<u32>,
        remove_degenerate: Option<u32>,
        remove_infinite: bool,
        remove_unreferenced: bool,
    ) -> Self {
        let options = rmesh::cleanup::CleanupOptions {
            merge_vertices,
            merge_tex,
            merge_norm,
            remove_degenerate,
            remove_infinite,
            remove_unreferenced,
        };
        Self::new_from_trimesh(self.data.cleanup(&options).into())
    }

    /// Open an interactive 3D viewer window displaying this mesh.
    #[pyo3(signature = (*, title="rmesh viewer", width=1280, height=720, background=None))]
    fn show(
        &self,
        py: Python<'_>,
        title: &str,
        width: u32,
        height: u32,
        background: Option<[f32; 3]>,
    ) {
        use rmesh::scene::Scene;
        let mut scene = Scene::new();
        scene.add_geometry("mesh", Geometry::Mesh(Box::new(self.data.clone())));
        let options = ViewerOptions {
            title: title.to_string(),
            width,
            height,
            background: background.unwrap_or([0.15, 0.15, 0.18]),
        };
        py.detach(|| scene.show_with_options(options));
    }

    /// Render this mesh to a PNG image (headless, no window).
    ///
    /// Returns PNG bytes.
    #[pyo3(signature = (*, width=1280, height=720, background=None))]
    fn to_image<'py>(
        &self,
        py: Python<'py>,
        width: u32,
        height: u32,
        background: Option<[f32; 3]>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        use rmesh::scene::Scene;
        let mut scene = Scene::new();
        scene.add_geometry("mesh", Geometry::Mesh(Box::new(self.data.clone())));
        let options = RenderOptions {
            width,
            height,
            background: background.unwrap_or([0.15, 0.15, 0.18]),
        };
        let rgba = py.detach(|| scene.render_to_image(&options));
        let img = image::RgbaImage::from_raw(width, height, rgba)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("render failed"))?;
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &buf))
    }
}

// ============================================================================
// PyVoxelGrid
// ============================================================================

#[pyclass(name = "VoxelGrid")]
pub struct PyVoxelGrid {
    data: rmesh::voxel::VoxelGrid,
}

#[pymethods]
impl PyVoxelGrid {
    #[getter]
    fn dims(&self) -> (u32, u32, u32) {
        let d = self.data.dims();
        (d[0], d[1], d[2])
    }

    #[getter]
    fn scale(&self) -> f64 {
        self.data.scale()
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.data.volume()
    }

    #[getter]
    fn area(&self) -> f64 {
        self.data.area()
    }

    fn __repr__(&self) -> String {
        let d = self.data.dims();
        format!(
            "<rmesh.VoxelGrid dims: ({}, {}, {}) scale: {:.6}>",
            d[0],
            d[1],
            d[2],
            self.data.scale()
        )
    }
}

// ============================================================================
// load
// ============================================================================

#[pyfunction(name = "load")]
#[pyo3(signature = (file_obj, file_type=None, *, resolver=None))]
pub fn py_load(
    py: Python<'_>,
    file_obj: Py<PyAny>,
    file_type: Option<&str>,
    resolver: Option<Py<PyAny>>,
) -> Result<crate::scene::PyScene> {
    // Try bytes first
    if let Ok(bytes) = file_obj.extract::<Vec<u8>>(py) {
        let ft = file_type.map(FileType::from_extension).transpose()?;
        let scene = load_with_resolver(&bytes, ft, resolver, py)?;
        return Ok(crate::scene::PyScene::from_scene(scene));
    }

    // Try path
    let path_str: String = file_obj
        .extract(py)
        .or_else(|_| {
            file_obj
                .bind(py)
                .call_method0("__fspath__")?
                .extract::<String>()
        })
        .map_err(|_| anyhow::anyhow!("file_obj must be str, PathLike, or bytes"))?;

    let path = Path::new(&path_str);
    let bytes = std::fs::read(path)?;
    let ft = match file_type {
        Some(ft_str) => Some(FileType::from_extension(ft_str)?),
        None => path
            .extension()
            .and_then(|e| e.to_str())
            .map(FileType::from_extension)
            .transpose()?,
    };

    let scene = load(&bytes, ft, Some(&FileResolver::from_file_path(path)))?;
    Ok(crate::scene::PyScene::from_scene(scene))
}

struct PyCallableResolver(Py<PyAny>);

impl Resolver for PyCallableResolver {
    fn resolve(&self, path: &str) -> anyhow::Result<Vec<u8>> {
        Python::attach(|py| {
            self.0
                .call1(py, (path,))?
                .extract::<Vec<u8>>(py)
                .map_err(|e| anyhow::anyhow!("resolver must return bytes: {e}"))
        })
    }
}

fn load_with_resolver(
    bytes: &[u8],
    file_type: Option<FileType>,
    resolver: Option<Py<PyAny>>,
    py: Python<'_>,
) -> Result<rmesh::scene::Scene> {
    match resolver {
        None => load(bytes, file_type, None),
        Some(res) => {
            let bound = res.bind(py);
            if let Ok(dict) = bound.cast::<PyDict>() {
                let mut mem = InMemoryResolver::new();
                for (k, v) in dict.iter() {
                    mem.insert(k.extract::<String>()?, v.extract::<Vec<u8>>()?);
                }
                load(bytes, file_type, Some(&mem))
            } else if bound.is_callable() {
                load(
                    bytes,
                    file_type,
                    Some(&PyCallableResolver(res.clone_ref(py))),
                )
            } else {
                Err(anyhow::anyhow!("resolver must be dict or callable"))
            }
        }
    }
}

/// Parse a list of Python dicts into `(Vec<Surface>, Vec<String>)`.
fn parse_surface_dicts(
    dicts: &[Bound<'_, PyDict>],
) -> PyResult<(Vec<rmesh::boundary::Surface>, Vec<String>)> {
    let mut surfaces = Vec::new();
    let mut names = Vec::new();

    for dict in dicts {
        let kind: String = dict
            .get_item("kind")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'kind' key"))?
            .extract()?;

        macro_rules! get_key {
            ($dict:expr, $key:expr) => {
                $dict.get_item($key)?.ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!("surface missing '{}'", $key))
                })
            };
        }

        let surface = match kind.as_str() {
            "Plane" => {
                let origin: [f64; 3] = get_key!(dict, "origin")?.extract()?;
                let normal: [f64; 3] = get_key!(dict, "normal")?.extract()?;
                rmesh::boundary::Surface::Plane(faces::SurfacePlane {
                    origin: Point3::new(origin[0], origin[1], origin[2]),
                    normal: Vector3::new(normal[0], normal[1], normal[2]),
                })
            }
            "Cylinder" => {
                let origin: [f64; 3] = get_key!(dict, "origin")?.extract()?;
                let axis: [f64; 3] = get_key!(dict, "axis")?.extract()?;
                let radius: f64 = get_key!(dict, "radius")?.extract()?;
                rmesh::boundary::Surface::Cylinder(faces::Cylinder {
                    origin: Point3::new(origin[0], origin[1], origin[2]),
                    axis: Vector3::new(axis[0], axis[1], axis[2]),
                    radius,
                })
            }
            "Cone" => {
                let apex: [f64; 3] = get_key!(dict, "apex")?.extract()?;
                let axis: [f64; 3] = get_key!(dict, "axis")?.extract()?;
                let half_angle: f64 = get_key!(dict, "half_angle")?.extract()?;
                rmesh::boundary::Surface::Cone(faces::Cone {
                    apex: Point3::new(apex[0], apex[1], apex[2]),
                    axis: Vector3::new(axis[0], axis[1], axis[2]),
                    half_angle,
                })
            }
            "Sphere" => {
                let center: [f64; 3] = get_key!(dict, "center")?.extract()?;
                let radius: f64 = get_key!(dict, "radius")?.extract()?;
                rmesh::boundary::Surface::Sphere(faces::Sphere {
                    center: Point3::new(center[0], center[1], center[2]),
                    radius,
                })
            }
            "Torus" => {
                let center: [f64; 3] = get_key!(dict, "center")?.extract()?;
                let axis: [f64; 3] = get_key!(dict, "axis")?.extract()?;
                let major_radius: f64 = get_key!(dict, "major_radius")?.extract()?;
                let minor_radius: f64 = get_key!(dict, "minor_radius")?.extract()?;
                rmesh::boundary::Surface::Torus(faces::Torus {
                    center: Point3::new(center[0], center[1], center[2]),
                    axis: Vector3::new(axis[0], axis[1], axis[2]),
                    major_radius,
                    minor_radius,
                })
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown surface kind: '{}'",
                    other
                )));
            }
        };

        names.push(kind);
        surfaces.push(surface);
    }

    Ok((surfaces, names))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmesh::creation::create_box;

    #[test]
    fn test_pytrimesh() {
        let m = PyTrimesh::new_from_trimesh(create_box(&[1.0, 1.0, 1.0]));
        assert_eq!(m.data.faces.len(), 12);
    }
}
