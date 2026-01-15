use std::path::Path;

use anyhow::Result;
use nalgebra::{Point3, Vector3, Vector4};
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods, ndarray::Array2, npyffi};
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rmesh::attributes::{Attributes, Material};
use rmesh::exchange::{FileResolver, InMemoryResolver, MeshFormat, load_mesh};
use rmesh::mesh::Trimesh;
use rmesh::resolvers::Resolver;

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

    #[getter]
    fn face_count(&self) -> usize {
        self.data.faces.len()
    }

    #[getter]
    fn material_count(&self) -> usize {
        self.data.materials.len()
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
        Self::new_from_trimesh(
            self.data
                .simplify(target_faces, aggressiveness.unwrap_or(7.0)),
        )
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
        Python::with_gil(|py| {
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
        assert_eq!(m.face_count(), 12);
    }
}
