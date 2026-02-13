#![allow(unsafe_code)]

use nalgebra::Point2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use rmesh::geometry::Geometry;
use rmesh::scene::Scene;

/// Build a `ViewerOptions` from Python kwargs.
fn viewer_options(
    title: &str,
    width: u32,
    height: u32,
    background: Option<[f32; 3]>,
) -> crate::ViewerOptions {
    crate::ViewerOptions {
        title: title.to_string(),
        width,
        height,
        background: background.unwrap_or([0.15, 0.15, 0.18]),
    }
}

/// Clone native data from a Python object that exposes `_data_ptr()`.
///
/// # Safety
/// The Python object must be an `rmesh.Trimesh` or `rmesh.Scene` whose
/// `_data_ptr()` method returns a valid pointer to `T`. The object is kept
/// alive by the caller for the duration of this function.
unsafe fn clone_native<T: Clone>(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<T> {
    let ptr: usize = obj.call_method0(py, "_data_ptr")?.extract(py)?;
    if ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "null data pointer",
        ));
    }
    // SAFETY: ptr points to a valid T inside the Python object which is kept
    // alive by the caller. We clone immediately while holding the GIL.
    Ok(unsafe { &*(ptr as *const T) }.clone())
}

/// Show a Trimesh in an interactive 3D viewer window.
///
/// Accepts a native `rmesh.Trimesh` object. All mesh attributes (UVs,
/// normals, materials, textures, smooth groups) are preserved.
#[pyfunction]
#[pyo3(signature = (mesh, *, title="rmesh viewer", width=1280, height=720, background=None))]
fn show_trimesh(
    py: Python<'_>,
    mesh: Py<PyAny>,
    title: &str,
    width: u32,
    height: u32,
    background: Option<[f32; 3]>,
) -> PyResult<()> {
    // Clone the native Trimesh data from the Python object.
    let trimesh: rmesh::mesh::Trimesh = unsafe { clone_native(py, &mesh)? };

    let mut scene = Scene::new();
    scene.add_geometry("mesh", Geometry::Mesh(Box::new(trimesh)));

    let options = viewer_options(title, width, height, background);

    py.detach(|| {
        use crate::SceneViewer;
        scene.show_with_options(options)
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

/// Show a Scene in an interactive 3D viewer window.
///
/// Accepts a native `rmesh.Scene` object. All geometry attributes (UVs,
/// normals, materials, textures, smooth groups) and the scene graph are
/// preserved.
#[pyfunction]
#[pyo3(signature = (scene, *, title="rmesh viewer", width=1280, height=720, background=None))]
fn show_scene(
    py: Python<'_>,
    scene: Py<PyAny>,
    title: &str,
    width: u32,
    height: u32,
    background: Option<[f32; 3]>,
) -> PyResult<()> {
    // Clone the native Scene data from the Python object.
    let data: Scene = unsafe { clone_native(py, &scene)? };

    let options = viewer_options(title, width, height, background);

    py.detach(|| {
        use crate::SceneViewer;
        data.show_with_options(options)
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

/// Show a 2D polygon in an interactive viewer window.
///
/// `exterior` is an (N×2) f64 array of the outer boundary.
/// `interiors` is an optional list of (M×2) f64 arrays for holes.
#[pyfunction]
#[pyo3(signature = (exterior, *, interiors=None, title="rmesh 2D", width=1280, height=720, background=None))]
fn show_polygon2d(
    py: Python<'_>,
    exterior: PyReadonlyArray2<'_, f64>,
    interiors: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    title: &str,
    width: u32,
    height: u32,
    background: Option<[f32; 3]>,
) -> PyResult<()> {
    use rmesh::path::polygon::Polygon2D;

    let ext: Vec<Point2<f64>> = exterior
        .as_array()
        .rows()
        .into_iter()
        .map(|r| Point2::new(r[0], r[1]))
        .collect();

    let holes: Vec<Vec<Point2<f64>>> = interiors
        .unwrap_or_default()
        .iter()
        .map(|h| {
            h.as_array()
                .rows()
                .into_iter()
                .map(|r| Point2::new(r[0], r[1]))
                .collect()
        })
        .collect();

    let poly = Polygon2D {
        exterior: ext,
        interiors: holes,
        to_3d: None,
    };

    let options = crate::ViewerOptions {
        title: title.to_string(),
        width,
        height,
        background: background.unwrap_or([1.0, 1.0, 1.0]),
    };

    py.detach(|| {
        use crate::Viewer2D;
        poly.show_2d_with_options(options)
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

/// Show 2D polylines in an interactive viewer window.
///
/// `polylines` is a list of (N×2) f64 arrays, each representing a polyline.
#[pyfunction]
#[pyo3(signature = (polylines, *, title="rmesh 2D", width=1280, height=720, background=None))]
fn show_polylines_2d(
    py: Python<'_>,
    polylines: Vec<PyReadonlyArray2<'_, f64>>,
    title: &str,
    width: u32,
    height: u32,
    background: Option<[f32; 3]>,
) -> PyResult<()> {
    let color = [0.0_f32, 0.8, 0.2, 1.0];

    let mut lines = Vec::new();
    let mut min = Point2::new(f64::INFINITY, f64::INFINITY);
    let mut max = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);

    for pl in &polylines {
        let pts: Vec<Point2<f64>> = pl
            .as_array()
            .rows()
            .into_iter()
            .map(|r| {
                let p = Point2::new(r[0], r[1]);
                min.x = min.x.min(p.x);
                min.y = min.y.min(p.y);
                max.x = max.x.max(p.x);
                max.y = max.y.max(p.y);
                p
            })
            .collect();
        if pts.len() >= 2 {
            lines.push((pts, color));
        }
    }

    if min.x > max.x {
        min = Point2::new(-1.0, -1.0);
        max = Point2::new(1.0, 1.0);
    }

    let data = crate::View2DData {
        lines,
        fills: Vec::new(),
        bounds: (min, max),
    };

    let options = crate::ViewerOptions {
        title: title.to_string(),
        width,
        height,
        background: background.unwrap_or([1.0, 1.0, 1.0]),
    };

    py.detach(|| crate::show_2d_data(&data, options))
        .map_err(|e: anyhow::Error| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

/// A Python module for interactive 3D/2D viewing of rmesh geometry.
#[pymodule]
#[pyo3(name = "rmesh_viewer")]
fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(show_trimesh, m)?)?;
    m.add_function(wrap_pyfunction!(show_scene, m)?)?;
    m.add_function(wrap_pyfunction!(show_polygon2d, m)?)?;
    m.add_function(wrap_pyfunction!(show_polylines_2d, m)?)?;
    Ok(())
}
