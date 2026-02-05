use nalgebra::Matrix4;
use numpy::PyReadonlyArray2;
use once_cell::sync::OnceCell;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use rmesh::geometry::Geometry;
use rmesh_viewer::{RenderOptions, SceneViewer, ViewerOptions};

use crate::mesh::{PyPath2D, PyPath3D, PyPolygon2D, PyTrimesh, readonly_1d, readonly_bounds};

// ============================================================================
// PyGeometryDict
// ============================================================================

/// A dict-like collection of geometry, keyed by name.
#[pyclass(name = "GeometryDict")]
pub struct PyGeometryDict {
    /// Stores (name, geometry) pairs, preserving insertion order
    pub(crate) items: Vec<(String, Py<PyAny>)>,
}

#[pymethods]
impl PyGeometryDict {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.items
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, obj)| obj.clone_ref(py))
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("'{}'", key)))
    }

    fn __contains__(&self, key: &str) -> bool {
        self.items.iter().any(|(name, _)| name == key)
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn __iter__(&self) -> PyGeometryDictKeysIter {
        PyGeometryDictKeysIter {
            keys: self.items.iter().map(|(k, _)| k.clone()).collect(),
            index: 0,
        }
    }

    fn keys(&self) -> Vec<String> {
        self.items.iter().map(|(k, _)| k.clone()).collect()
    }

    fn values(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.items.iter().map(|(_, v)| v.clone_ref(py)).collect()
    }

    fn items(&self, py: Python<'_>) -> Vec<(String, Py<PyAny>)> {
        self.items
            .iter()
            .map(|(k, v)| (k.clone(), v.clone_ref(py)))
            .collect()
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> Option<Py<PyAny>> {
        self.items
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, obj)| obj.clone_ref(py))
            .or(default)
    }

    fn __repr__(&self) -> String {
        let keys: Vec<_> = self.items.iter().map(|(k, _)| format!("'{}'", k)).collect();
        format!("GeometryDict({{{}}})", keys.join(", "))
    }
}

#[pyclass]
pub struct PyGeometryDictKeysIter {
    keys: Vec<String>,
    index: usize,
}

#[pymethods]
impl PyGeometryDictKeysIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<String> {
        if self.index < self.keys.len() {
            let key = self.keys[self.index].clone();
            self.index += 1;
            Some(key)
        } else {
            None
        }
    }
}

// ============================================================================
// PyScene
// ============================================================================

/// A scene containing named geometry with an optional scene graph.
#[pyclass(name = "Scene")]
pub struct PyScene {
    pub(crate) data: rmesh::scene::Scene,
    geometry_cache: OnceCell<Py<PyGeometryDict>>,
}

impl PyScene {
    pub fn from_scene(data: rmesh::scene::Scene) -> Self {
        Self {
            data,
            geometry_cache: OnceCell::new(),
        }
    }
}

#[pymethods]
impl PyScene {
    /// Create a new empty scene.
    #[new]
    fn new() -> Self {
        Self::from_scene(rmesh::scene::Scene::new())
    }

    /// Add geometry to the scene.
    ///
    /// Parameters
    /// ----------
    /// geometry : Trimesh | Path2D | Path3D | Polygon2D
    ///     The geometry to add.
    /// name : str, optional
    ///     Name for the geometry. Auto-generated if not provided.
    /// transforms : list of (4, 4) arrays, optional
    ///     One 4x4 transform per instance. If not provided a single
    ///     instance with identity transform is created.
    ///
    /// Returns
    /// -------
    /// str
    ///     The unique name assigned to the geometry.
    #[pyo3(signature = (geometry, *, name=None, transforms=None))]
    fn add(
        &mut self,
        geometry: &Bound<'_, PyAny>,
        name: Option<&str>,
        transforms: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    ) -> PyResult<String> {
        let geom = if let Ok(m) = geometry.cast::<PyTrimesh>() {
            Geometry::Mesh(Box::new(m.borrow().data.clone()))
        } else if let Ok(p) = geometry.cast::<PyPath3D>() {
            Geometry::Path3D(p.borrow().data.clone())
        } else if let Ok(p) = geometry.cast::<PyPath2D>() {
            Geometry::Path2D(Box::new(p.borrow().data.clone()))
        } else if let Ok(p) = geometry.cast::<PyPolygon2D>() {
            let poly = &p.borrow().data;
            if let Some(path3d) = poly.to_path3d() {
                Geometry::Path3D(path3d)
            } else {
                Geometry::Path2D(Box::new(poly.to_path2d()))
            }
        } else {
            return Err(PyTypeError::new_err(
                "expected Trimesh, Path2D, Path3D, or Polygon2D",
            ));
        };

        let mats: Option<Vec<Matrix4<f64>>> = transforms.map(|ts| {
            ts.iter()
                .map(|t| {
                    let a = t.as_array();
                    Matrix4::new(
                        a[[0, 0]],
                        a[[0, 1]],
                        a[[0, 2]],
                        a[[0, 3]],
                        a[[1, 0]],
                        a[[1, 1]],
                        a[[1, 2]],
                        a[[1, 3]],
                        a[[2, 0]],
                        a[[2, 1]],
                        a[[2, 2]],
                        a[[2, 3]],
                        a[[3, 0]],
                        a[[3, 1]],
                        a[[3, 2]],
                        a[[3, 3]],
                    )
                })
                .collect()
        });

        let label = name.unwrap_or("");
        let result = self.data.add(label, geom, mats.as_deref());

        // Invalidate geometry cache
        self.geometry_cache = OnceCell::new();

        Ok(result)
    }

    /// Get all geometry in the scene as a dict-like object keyed by name.
    #[getter]
    fn geometry(&self, py: Python<'_>) -> Py<PyGeometryDict> {
        self.geometry_cache
            .get_or_init(|| {
                let items: Vec<(String, Py<PyAny>)> = self
                    .data
                    .geometry
                    .iter()
                    .filter_map(|(name, geom)| {
                        let obj: Py<PyAny> = match geom {
                            Geometry::Mesh(mesh) => {
                                Py::new(py, PyTrimesh::new_from_trimesh((**mesh).clone()))
                                    .ok()?
                                    .into_any()
                            }
                            Geometry::Feature(model) => Py::new(
                                py,
                                crate::feature::PyFeatureModel {
                                    inner: (**model).clone(),
                                },
                            )
                            .ok()?
                            .into_any(),
                            // TODO: Path2D, Path3D, PointCloud bindings
                            _ => return None,
                        };
                        Some((name.clone(), obj))
                    })
                    .collect();
                Py::new(py, PyGeometryDict { items }).unwrap()
            })
            .clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!("Scene(geometry={})", self.data.geometry.len())
    }

    fn __len__(&self) -> usize {
        self.data.geometry.len()
    }

    /// Open an interactive 3D viewer window displaying this scene.
    #[pyo3(signature = (*, title="rmesh viewer", width=1280, height=720, background=None))]
    fn show(
        &self,
        py: Python<'_>,
        title: &str,
        width: u32,
        height: u32,
        background: Option<[f32; 3]>,
    ) {
        let options = ViewerOptions {
            title: title.to_string(),
            width,
            height,
            background: background.unwrap_or([0.15, 0.15, 0.18]),
        };
        let data = self.data.clone();
        py.detach(|| data.show_with_options(options));
    }

    /// Axis-aligned bounding box as a (2, 3) array [[min_x, min_y, min_z], [max_x, max_y, max_z]],
    /// or None if the scene has no geometry with valid bounds.
    #[getter]
    fn bounds(&self, py: Python<'_>) -> Option<Py<numpy::PyArray2<f64>>> {
        self.data.bounds().map(|(min, max)| {
            readonly_bounds(py, vec![min.x, min.y, min.z, max.x, max.y, max.z], 3)
        })
    }

    /// Extents of the bounding box [x, y, z] as a (3,) array,
    /// or None if the scene has no geometry with valid bounds.
    #[getter]
    fn extents(&self, py: Python<'_>) -> Option<Py<numpy::PyArray1<f64>>> {
        self.data.extents().map(|e| readonly_1d(py, e.to_vec()))
    }

    /// Render this scene to a PNG image (headless, no window).
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
        let options = RenderOptions {
            width,
            height,
            background: background.unwrap_or([0.15, 0.15, 0.18]),
        };
        let data = self.data.clone();
        let rgba = py.detach(|| data.render_to_image(&options));
        let img = image::RgbaImage::from_raw(width, height, rgba)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("render failed"))?;
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &buf))
    }
}
