use std::path::Path;

use anyhow::Result;
use nalgebra::{Point3, Vector3, Vector4};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods, ndarray::Array2, npyffi};
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rmesh::attributes::{Attributes, GroupingKind, Material};
use rmesh::exchange::{FileResolver, InMemoryResolver, MeshFormat, load_mesh};
use rmesh::mesh::Trimesh;
use rmesh::resolvers::Resolver;

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
        let kind = match grouping.kind {
            GroupingKind::Material => "material",
            GroupingKind::Group => "group",
            GroupingKind::Smoothing => "smoothing",
            GroupingKind::Object => "object",
            GroupingKind::Unspecified => "unspecified",
        };
        let names = pyo3::types::PyList::new(py, &grouping.names)
            .unwrap()
            .unbind();
        let indices: Vec<i64> = grouping.indices.iter().map(|&i| i as i64).collect();
        let indices_arr = PyArray1::from_vec(py, indices);
        make_readonly(&indices_arr);
        Self {
            kind: kind.to_string(),
            names,
            indices: indices_arr.unbind(),
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
// Helpers
// ============================================================================

/// Make a numpy array read-only by clearing the WRITEABLE flag.
fn make_readonly<T: numpy::Element, D: numpy::ndarray::Dimension>(
    arr: &Bound<'_, numpy::PyArray<T, D>>,
) {
    unsafe {
        (*arr.as_array_ptr()).flags &= !npyffi::flags::NPY_ARRAY_WRITEABLE;
    }
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

macro_rules! cached_array_opt {
    ($self:expr, $py:expr, $cache:ident, $data:expr, $cols:expr) => {{
        $self
            .$cache
            .get_or_init(|| {
                $data.as_ref().map(|d| {
                    let flat: &[_] = bytemuck::cast_slice(d.as_slice());
                    let nd = Array2::from_shape_vec((d.len(), $cols), flat.to_vec()).unwrap();
                    let arr = PyArray2::from_array($py, &nd);
                    make_readonly(&arr);
                    arr.unbind()
                })
            })
            .as_ref()
            .map(|a| a.clone_ref($py))
    }};
}

// ============================================================================
// PyTrimesh
// ============================================================================

#[pyclass(name = "Trimesh")]
pub struct PyTrimesh {
    data: Trimesh,
    vertices_cache: OnceCell<Py<PyArray2<f64>>>,
    faces_cache: OnceCell<Py<PyArray2<i64>>>,
    uv_cache: OnceCell<Option<Py<PyArray2<f64>>>>,
    vertex_normals_cache: OnceCell<Option<Py<PyArray2<f64>>>>,
    face_colors_cache: OnceCell<Option<Py<PyArray2<u8>>>>,
    face_normals_cache: OnceCell<Py<PyArray2<f64>>>,
    edges_cache: OnceCell<Py<PyArray2<i64>>>,
    face_adjacency_cache: OnceCell<Py<PyArray2<i64>>>,
    face_adjacency_angles_cache: OnceCell<Py<PyArray1<f64>>>,
    vertex_attributes_cache: OnceCell<Py<PyVertexAttributes>>,
    face_attributes_cache: OnceCell<Py<PyFaceAttributes>>,
}

impl PyTrimesh {
    fn new_from_trimesh(data: Trimesh) -> Self {
        Self {
            data,
            vertices_cache: OnceCell::new(),
            faces_cache: OnceCell::new(),
            uv_cache: OnceCell::new(),
            vertex_normals_cache: OnceCell::new(),
            face_colors_cache: OnceCell::new(),
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
    pub fn new(
        vertices: PyReadonlyArray2<'_, f64>,
        faces: PyReadonlyArray2<'_, i64>,
    ) -> Result<Self> {
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
        Ok(Self::new_from_trimesh(Trimesh::new(
            vertices, faces, None, None,
        )?))
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

    #[getter]
    fn uv(&self, py: Python<'_>) -> Option<Py<PyArray2<f64>>> {
        cached_array_opt!(self, py, uv_cache, self.data.uv(), 2)
    }

    #[getter]
    fn vertex_normals(&self, py: Python<'_>) -> Option<Py<PyArray2<f64>>> {
        cached_array_opt!(
            self,
            py,
            vertex_normals_cache,
            self.data.attributes_vertex.normals.first(),
            3
        )
    }

    #[getter]
    fn face_colors(&self, py: Python<'_>) -> Option<Py<PyArray2<u8>>> {
        cached_array_opt!(
            self,
            py,
            face_colors_cache,
            self.data.attributes_face.colors.first(),
            4
        )
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
            let data = vec![min.x, min.y, min.z, max.x, max.y, max.z];
            let nd = Array2::from_shape_vec((2, 3), data).unwrap();
            let arr = PyArray2::from_array(py, &nd);
            make_readonly(&arr);
            arr.unbind()
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
    fn center_mass(&self) -> (f64, f64, f64) {
        let cm = self.data.center_mass();
        (cm.x, cm.y, cm.z)
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
            .flat_map(|e| [e[0] as i64, e[1] as i64])
            .collect();
        let nd = Array2::from_shape_vec((edges.len(), 2), flat).unwrap();
        let arr = PyArray2::from_array(py, &nd);
        make_readonly(&arr);
        arr.unbind()
    }

    /// Geometric length of each unique edge, shape (n_edges_unique,).
    #[getter]
    fn edges_unique_length(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        let lengths = self.data.edges_unique_length();
        let arr = PyArray1::from_vec(py, lengths);
        make_readonly(&arr);
        arr.unbind()
    }

    /// Index mapping: edges_sorted[i] -> edges_unique index, shape (n_faces * 3,).
    #[getter]
    fn edges_unique_inverse(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        let inverse = self.data.edges_unique_inverse();
        let flat: Vec<i64> = inverse.iter().map(|&i| i as i64).collect();
        let arr = PyArray1::from_vec(py, flat);
        make_readonly(&arr);
        arr.unbind()
    }

    /// Principal inertia components (eigenvalues of inertia tensor, sorted descending).
    #[getter]
    fn principal_inertia_components(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.data.principal_inertia_components().map(|v| {
            let arr = PyArray1::from_vec(py, vec![v.x, v.y, v.z]);
            make_readonly(&arr);
            arr.unbind()
        })
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

    fn face_adjacency_angles(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.face_adjacency_angles_cache
            .get_or_init(|| {
                let angles = self.data.face_adjacency_angles();
                let arr = PyArray1::from_vec(py, angles);
                make_readonly(&arr);
                arr.unbind()
            })
            .clone_ref(py)
    }

    /// Per-face areas.
    #[getter]
    fn area_faces(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        let areas = self.data.faces_area();
        let arr = PyArray1::from_slice(py, areas);
        make_readonly(&arr);
        arr.unbind()
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
        self.data.extents().map(|e| {
            let arr = PyArray1::from_vec(py, vec![e.x, e.y, e.z]);
            make_readonly(&arr);
            arr.unbind()
        })
    }

    /// Geometric center of the vertices (mean position).
    #[getter]
    fn centroid(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.data.centroid().map(|c| {
            let arr = PyArray1::from_vec(py, vec![c.x, c.y, c.z]);
            make_readonly(&arr);
            arr.unbind()
        })
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
        let projections = self.data.face_adjacency_projections();
        let arr = PyArray1::from_vec(py, projections);
        make_readonly(&arr);
        arr.unbind()
    }

    /// Boolean array indicating whether each adjacent face pair is locally convex.
    #[getter]
    fn face_adjacency_convex(&self, py: Python<'_>) -> Py<PyArray1<bool>> {
        let convex = self.data.face_adjacency_convex();
        let arr = PyArray1::from_vec(py, convex);
        make_readonly(&arr);
        arr.unbind()
    }

    /// Check if the mesh is convex.
    #[getter]
    fn is_convex(&self) -> bool {
        self.data.is_convex()
    }

    fn material_name(&self, index: usize) -> Option<String> {
        self.data.materials.get(index).map(|m| match m {
            Material::Simple(s) => s.name.clone(),
            _ => String::new(),
        })
    }

    fn material_has_texture(&self, index: usize) -> bool {
        matches!(
            self.data.materials.get(index),
            Some(Material::Simple(s)) if s.diffuse_texture.is_some()
        )
    }

    fn material_diffuse(&self, index: usize) -> Option<[f64; 3]> {
        match self.data.materials.get(index)? {
            Material::Simple(s) => s.diffuse.map(|d| [d.x, d.y, d.z]),
            _ => None,
        }
    }

    #[pyo3(signature = (target_faces, aggressiveness=None))]
    fn simplify(&self, target_faces: usize, aggressiveness: Option<f64>) -> Self {
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
    /// >>> mesh = rmesh.load_mesh("model.stl")
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

    #[staticmethod]
    #[pyo3(signature = (vertices, faces, vertex_normals=None, face_colors=None))]
    fn from_arrays(
        vertices: PyReadonlyArray2<'_, f64>,
        faces: PyReadonlyArray2<'_, i64>,
        vertex_normals: Option<PyReadonlyArray2<'_, f64>>,
        face_colors: Option<PyReadonlyArray2<'_, u8>>,
    ) -> Result<Self> {
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

        Ok(Self::new_from_trimesh(Trimesh::new(
            vertices,
            faces,
            Some(attr_vertex),
            Some(attr_face),
        )?))
    }
}

// ============================================================================
// load_mesh
// ============================================================================

#[pyfunction(name = "load_mesh")]
#[pyo3(signature = (file_obj, file_type=None, *, resolver=None))]
pub fn py_load_mesh(
    py: Python<'_>,
    file_obj: Py<PyAny>,
    file_type: Option<&str>,
    resolver: Option<Py<PyAny>>,
) -> Result<PyTrimesh> {
    // Try bytes first
    if let Ok(bytes) = file_obj.extract::<Vec<u8>>(py) {
        let fmt = MeshFormat::from_string(
            file_type.ok_or_else(|| anyhow::anyhow!("file_type required for bytes"))?,
        )?;
        return Ok(PyTrimesh::new_from_trimesh(load_with_resolver(
            &bytes, fmt, resolver, py,
        )?));
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
    let fmt = match file_type {
        Some(ft) => MeshFormat::from_string(ft)?,
        None => MeshFormat::from_string(
            path.extension()
                .and_then(|e| e.to_str())
                .ok_or_else(|| anyhow::anyhow!("cannot infer file type"))?,
        )?,
    };

    let data = load_mesh(&bytes, fmt, Some(&FileResolver::from_file_path(path)))?;
    Ok(PyTrimesh::new_from_trimesh(data))
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
    fmt: MeshFormat,
    resolver: Option<Py<PyAny>>,
    py: Python<'_>,
) -> Result<Trimesh> {
    match resolver {
        None => load_mesh(bytes, fmt, None),
        Some(res) => {
            let bound = res.bind(py);
            if let Ok(dict) = bound.downcast::<PyDict>() {
                let mut mem = InMemoryResolver::new();
                for (k, v) in dict.iter() {
                    mem.insert(k.extract::<String>()?, v.extract::<Vec<u8>>()?);
                }
                load_mesh(bytes, fmt, Some(&mem))
            } else if bound.is_callable() {
                load_mesh(bytes, fmt, Some(&PyCallableResolver(res.clone_ref(py))))
            } else {
                Err(anyhow::anyhow!("resolver must be dict or callable"))
            }
        }
    }
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
