use anyhow::Result;
use nalgebra::{Point3, Vector3, Vector4};
use numpy::ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use numpy::{PyArray2, PyReadonlyArray2};

use rmesh::attributes::Attributes;
use rmesh::exchange::{InMemoryResolver, MeshFormat, NoResolver, Resolver, load_mesh_with_resolver};
use rmesh::mesh::Trimesh;
use rmesh::simplify::SimplifyOptions;

/// A Python callable that acts as a Resolver.
struct PyCallableResolver {
    callable: Py<PyAny>,
}

impl Resolver for PyCallableResolver {
    fn resolve(&self, path: &str) -> anyhow::Result<Vec<u8>> {
        Python::with_gil(|py| {
            let result = self
                .callable
                .call1(py, (path,))
                .map_err(|e| anyhow::anyhow!("resolver error: {e}"))?;

            result
                .extract::<Vec<u8>>(py)
                .map_err(|e| anyhow::anyhow!("resolver must return bytes: {e}"))
        })
    }
}

#[pyclass(name = "Trimesh")]
#[derive(Clone)]
pub struct PyTrimesh {
    data: Trimesh,
}

#[pymethods]
impl PyTrimesh {
    #[new]
    /// Create a new Trimesh from vertices and faces.
    pub fn new<'py>(
        vertices: PyReadonlyArray2<'py, f64>,
        faces: PyReadonlyArray2<'py, i64>,
    ) -> Result<Self> {
        let vertices: Vec<Point3<f64>> = vertices
            .as_array()
            .rows()
            .into_iter()
            .map(|x| Point3::new(x[0], x[1], x[2]))
            .collect();

        let faces: Vec<[usize; 3]> = faces
            .as_array()
            .rows()
            .into_iter()
            .map(|x| [x[0] as usize, x[1] as usize, x[2] as usize])
            .collect();

        Ok(PyTrimesh {
            data: Trimesh::new(vertices, faces, None, None)?,
        })
    }

    #[getter]
    pub fn get_vertices<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        let vertices = &self.data.vertices;
        let shape = (vertices.len(), 3);

        let arr = Array2::from_shape_vec(
            shape,
            vertices
                .iter()
                .flat_map(|p| [p.x, p.y, p.z])
                .collect(),
        )
        .unwrap();

        PyArray2::from_array(py, &arr).to_owned().into()
    }

    #[getter]
    pub fn get_faces<'py>(&self, py: Python<'py>) -> Py<PyArray2<i64>> {
        let faces = &self.data.faces;
        let shape = (faces.len(), 3);

        let arr = Array2::from_shape_vec(
            shape,
            faces
                .iter()
                .flat_map(|&[a, b, c]| [a as i64, b as i64, c as i64])
                .collect(),
        )
        .unwrap();

        PyArray2::from_array(py, &arr).to_owned().into()
    }

    #[getter]
    pub fn get_uv<'py>(&self, py: Python<'py>) -> Option<Py<PyArray2<f64>>> {
        self.data.uv().as_ref().map(|uvs| {
            let shape = (uvs.len(), 2);
            let arr =
                Array2::from_shape_vec(shape, uvs.iter().flat_map(|p| [p.x, p.y]).collect())
                    .unwrap();
            PyArray2::from_array(py, &arr).to_owned().into()
        })
    }

    pub fn py_check(&self) -> usize {
        10
    }

    /// Get vertex normals if they exist.
    #[getter]
    pub fn get_vertex_normals<'py>(&self, py: Python<'py>) -> Option<Py<PyArray2<f64>>> {
        self.data.attributes_vertex.normals.first().map(|normals| {
            let shape = (normals.len(), 3);
            let arr = Array2::from_shape_vec(
                shape,
                normals.iter().flat_map(|n| [n.x, n.y, n.z]).collect(),
            )
            .unwrap();
            PyArray2::from_array(py, &arr).to_owned().into()
        })
    }

    /// Get face colors if they exist.
    /// Returns RGBA colors as uint8 array of shape (n_faces, 4).
    #[getter]
    pub fn get_face_colors<'py>(&self, py: Python<'py>) -> Option<Py<PyArray2<u8>>> {
        self.data.attributes_face.colors.first().map(|colors| {
            let shape = (colors.len(), 4);
            let arr = Array2::from_shape_vec(
                shape,
                colors
                    .iter()
                    .flat_map(|c| [c.x, c.y, c.z, c.w])
                    .collect(),
            )
            .unwrap();
            PyArray2::from_array(py, &arr).to_owned().into()
        })
    }

    /// Simplify the mesh to a target face count.
    ///
    /// Parameters
    /// ----------
    /// target_faces : int
    ///     Target number of faces after simplification.
    /// aggressiveness : float, optional
    ///     Controls how aggressively to simplify. Default: 7.0.
    ///     Higher values mean faster but potentially lower quality.
    ///
    /// Returns
    /// -------
    /// Trimesh
    ///     Simplified mesh with preserved vertex attributes.
    #[pyo3(signature = (target_faces, aggressiveness=None))]
    pub fn simplify(&self, target_faces: usize, aggressiveness: Option<f64>) -> Self {
        let simplified = self
            .data
            .simplify(target_faces, aggressiveness.unwrap_or(7.0));
        PyTrimesh { data: simplified }
    }

    /// Simplify the mesh with quality metrics returned.
    ///
    /// Parameters
    /// ----------
    /// target_faces : int
    ///     Target number of faces after simplification.
    /// aggressiveness : float, optional
    ///     Controls how aggressively to simplify. Default: 7.0.
    ///
    /// Returns
    /// -------
    /// tuple[Trimesh, dict]
    ///     Tuple of (simplified mesh, quality metrics dict).
    #[pyo3(signature = (target_faces, aggressiveness=None))]
    pub fn simplify_with_quality<'py>(
        &self,
        py: Python<'py>,
        target_faces: usize,
        aggressiveness: Option<f64>,
    ) -> (Self, Bound<'py, PyDict>) {
        let options = SimplifyOptions {
            target_count: target_faces,
            aggressiveness: aggressiveness.unwrap_or(7.0),
            preserve_attributes: true,
            compute_quality: true,
            ..Default::default()
        };

        let result = self.data.simplify_with_options(options);

        let simplified = PyTrimesh {
            data: Trimesh::new(
                result.vertices,
                result.faces,
                Some(result.attributes_vertex),
                Some(result.attributes_face),
            )
            .unwrap(),
        };

        let quality_dict = PyDict::new(py);
        if let Some(q) = result.quality {
            quality_dict
                .set_item("volume_ratio", q.volume_ratio)
                .unwrap();
            quality_dict
                .set_item("surface_area_ratio", q.surface_area_ratio)
                .unwrap();
            quality_dict
                .set_item("face_count_ratio", q.face_count_ratio)
                .unwrap();
            quality_dict.set_item("min_angle", q.min_angle).unwrap();
            quality_dict
                .set_item("degenerate_count", q.degenerate_count)
                .unwrap();
            quality_dict
                .set_item("flipped_count", q.flipped_count)
                .unwrap();
        }

        (simplified, quality_dict)
    }

    #[getter]
    pub fn get_face_count(&self) -> usize {
        self.data.faces.len()
    }

    /// Create a mesh from arrays including vertex normals and face colors.
    ///
    /// Parameters
    /// ----------
    /// vertices : ndarray (n, 3)
    /// faces : ndarray (m, 3)
    /// vertex_normals : ndarray (n, 3), optional
    /// face_colors : ndarray (m, 4) uint8, optional
    ///     RGBA colors per face
    #[staticmethod]
    #[pyo3(signature = (vertices, faces, vertex_normals=None, face_colors=None))]
    pub fn from_arrays<'py>(
        vertices: PyReadonlyArray2<'py, f64>,
        faces: PyReadonlyArray2<'py, i64>,
        vertex_normals: Option<PyReadonlyArray2<'py, f64>>,
        face_colors: Option<PyReadonlyArray2<'py, u8>>,
    ) -> Result<Self> {
        let vertices: Vec<Point3<f64>> = vertices
            .as_array()
            .rows()
            .into_iter()
            .map(|x| Point3::new(x[0], x[1], x[2]))
            .collect();

        let faces: Vec<[usize; 3]> = faces
            .as_array()
            .rows()
            .into_iter()
            .map(|x| [x[0] as usize, x[1] as usize, x[2] as usize])
            .collect();

        let mut attributes_vertex = Attributes::default();
        let mut attributes_face = Attributes::default();

        if let Some(normals_arr) = vertex_normals {
            let normals: Vec<Vector3<f64>> = normals_arr
                .as_array()
                .rows()
                .into_iter()
                .map(|x| Vector3::new(x[0], x[1], x[2]))
                .collect();
            attributes_vertex.normals.push(normals);
        }

        if let Some(colors_arr) = face_colors {
            let colors: Vec<Vector4<u8>> = colors_arr
                .as_array()
                .rows()
                .into_iter()
                .map(|x| Vector4::new(x[0], x[1], x[2], x[3]))
                .collect();
            attributes_face.colors.push(colors);
        }

        Ok(PyTrimesh {
            data: Trimesh::new(
                vertices,
                faces,
                Some(attributes_vertex),
                Some(attributes_face),
            )?,
        })
    }
}

/// Load a mesh from bytes.
///
/// Parameters
/// ----------
/// file_data : bytes
///     The raw file data.
/// file_type : str
///     The file type (e.g., "obj", "stl").
/// resolver : dict or callable, optional
///     For resolving external file references (e.g., MTL files for OBJ).
///     - If a dict, keys are file paths and values are file bytes.
///     - If a callable, it should take a path string and return bytes.
///     - If None (default), external references are silently ignored.
#[pyfunction(name = "load_mesh")]
#[pyo3(signature = (file_data, file_type, resolver=None))]
pub fn py_load_mesh(
    file_data: Vec<u8>,
    file_type: String,
    resolver: Option<Py<PyAny>>,
) -> Result<PyTrimesh> {
    let format = MeshFormat::from_string(&file_type)?;
    let bytes = &file_data;

    let data = match resolver {
        Some(res) => Python::with_gil(|py| {
            let bound = res.bind(py);
            if bound.is_instance_of::<PyDict>() {
                #[allow(deprecated)]
                let dict: &Bound<'_, PyDict> = bound
                    .downcast()
                    .map_err(|e| anyhow::anyhow!("expected dict: {e}"))?;
                let mut mem_resolver = InMemoryResolver::new();
                for (key, value) in dict.iter() {
                    let path: String = key
                        .extract()
                        .map_err(|e| anyhow::anyhow!("dict key must be str: {e}"))?;
                    let data: Vec<u8> = value
                        .extract()
                        .map_err(|e| anyhow::anyhow!("dict value must be bytes: {e}"))?;
                    mem_resolver.insert(path, data);
                }
                load_mesh_with_resolver(bytes, format, &mem_resolver)
            } else if bound.is_callable() {
                let callable_resolver = PyCallableResolver {
                    callable: res.clone_ref(py),
                };
                load_mesh_with_resolver(bytes, format, &callable_resolver)
            } else {
                let type_name = bound
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|_| "unknown".to_string());
                Err(anyhow::anyhow!(
                    "resolver must be a dict or callable, got {type_name}"
                ))
            }
        })?,
        None => load_mesh_with_resolver(bytes, format, &NoResolver)?,
    };

    Ok(PyTrimesh { data })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmesh::creation::create_box;

    #[test]
    fn test_mesh_python() {
        let data = create_box(&[1.0, 1.0, 1.0]);
        let m = PyTrimesh { data };
        assert_eq!(m.py_check(), 10);
    }
}
