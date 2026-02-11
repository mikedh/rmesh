use nalgebra::{Point2, Point3, Vector3, Vector4};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use rmesh::attributes::Attributes;
use rmesh::geometry::Geometry;
use rmesh::mesh::Trimesh;
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

/// Show a triangle mesh in an interactive 3D viewer window.
///
/// Blocks until the window is closed.
#[pyfunction]
#[pyo3(signature = (vertices, faces, *, vertex_normals=None, vertex_colors=None, title="rmesh viewer", width=1280, height=720, background=None))]
#[allow(clippy::too_many_arguments)]
fn show_trimesh(
    py: Python<'_>,
    vertices: PyReadonlyArray2<'_, f64>,
    faces: PyReadonlyArray2<'_, i64>,
    vertex_normals: Option<PyReadonlyArray2<'_, f64>>,
    vertex_colors: Option<PyReadonlyArray2<'_, u8>>,
    title: &str,
    width: u32,
    height: u32,
    background: Option<[f32; 3]>,
) -> PyResult<()> {
    let verts: Vec<Point3<f64>> = vertices
        .as_array()
        .rows()
        .into_iter()
        .map(|r| Point3::new(r[0], r[1], r[2]))
        .collect();
    let tris: Vec<[usize; 3]> = faces
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0] as usize, r[1] as usize, r[2] as usize])
        .collect();

    let mut attr_vertex = Attributes::default();
    if let Some(n) = vertex_normals {
        attr_vertex.normals.push(
            n.as_array()
                .rows()
                .into_iter()
                .map(|r| Vector3::new(r[0], r[1], r[2]))
                .collect(),
        );
    }
    let mut attr_face = Attributes::default();
    if let Some(c) = vertex_colors {
        attr_face.colors.push(
            c.as_array()
                .rows()
                .into_iter()
                .map(|r| Vector4::new(r[0], r[1], r[2], r[3]))
                .collect(),
        );
    }

    let mesh = Trimesh::new(verts, tris, Some(attr_vertex), Some(attr_face))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let mut scene = Scene::new();
    scene.add_geometry("mesh", Geometry::Mesh(Box::new(mesh)));

    let options = viewer_options(title, width, height, background);

    py.detach(|| {
        use crate::SceneViewer;
        scene.show_with_options(options)
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

/// Show a scene in an interactive 3D viewer window.
///
/// Accepts a list of mesh dicts, each with `vertices` (N×3 f64) and `faces` (M×3 i64).
#[pyfunction]
#[pyo3(signature = (meshes, *, title="rmesh viewer", width=1280, height=720, background=None))]
fn show_scene(
    py: Python<'_>,
    meshes: Vec<Bound<'_, pyo3::types::PyDict>>,
    title: &str,
    width: u32,
    height: u32,
    background: Option<[f32; 3]>,
) -> PyResult<()> {
    let mut scene = Scene::new();

    for (i, d) in meshes.iter().enumerate() {
        let verts_arr: PyReadonlyArray2<'_, f64> = d
            .get_item("vertices")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'vertices'"))?
            .extract()?;
        let faces_arr: PyReadonlyArray2<'_, i64> = d
            .get_item("faces")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'faces'"))?
            .extract()?;

        let verts: Vec<Point3<f64>> = verts_arr
            .as_array()
            .rows()
            .into_iter()
            .map(|r| Point3::new(r[0], r[1], r[2]))
            .collect();
        let tris: Vec<[usize; 3]> = faces_arr
            .as_array()
            .rows()
            .into_iter()
            .map(|r| [r[0] as usize, r[1] as usize, r[2] as usize])
            .collect();

        let mesh = Trimesh::new(verts, tris, None, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let name = format!("mesh_{i}");
        scene.add_geometry(&name, Geometry::Mesh(Box::new(mesh)));
    }

    let options = viewer_options(title, width, height, background);

    py.detach(|| {
        use crate::SceneViewer;
        scene.show_with_options(options)
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
