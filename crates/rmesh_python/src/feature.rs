//! Python bindings for the feature-based CAD system

use std::collections::HashMap;

use nalgebra::{Point3, Vector3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rmesh::creation::feature::{
    backends::fidget::{FidgetBackend, FidgetSettings},
    evaluator,
    exchange::load_feature_model_from_path,
    Chamfer, EdgeSelection, Environment, Extrude, FeatureBackend, FeatureModel, Fillet, Loft,
    Operation, Revolve, Sign, Sketch, SketchPlane, Sweep, Units,
};
use rmesh::path::{Line, Path2D, Path3D, Segment3D};
use rmesh::serialize::RmeshSerializable;

use crate::mesh::PyTrimesh;

// =============================================================================
// Evaluate function (top-level utility)
// =============================================================================

/// Evaluate a Starlark expression with the given variables.
///
/// Supports full Starlark syntax including:
/// - Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
/// - Comparisons: `<`, `>`, `<=`, `>=`, `==`, `!=`
/// - Boolean: `and`, `or`, `not`
/// - Conditionals: `x if condition else y`
/// - Lists: `[1, 2, 3]`, list comprehensions
/// - Math module: `math.sqrt`, `math.sin`, `math.cos`, `math.pi`, etc.
///
/// Parameters
/// ----------
/// script : str
///     The Starlark expression to evaluate.
/// variables : dict[str, float], optional
///     Variables available in the expression.
///
/// Returns
/// -------
/// float
///     The result of the expression.
///
/// Examples
/// --------
/// >>> import rmesh
/// >>> rmesh.evaluate("2 + 3")
/// 5.0
/// >>> rmesh.evaluate("width * 2", {"width": 100.0})
/// 200.0
/// >>> rmesh.evaluate("math.sqrt(x) + math.sin(math.pi / 2)", {"x": 16.0})
/// 5.0
#[pyfunction(name = "evaluate")]
#[pyo3(signature = (script, variables=None))]
pub fn py_evaluate(script: &str, variables: Option<&Bound<'_, PyDict>>) -> PyResult<f64> {
    let vars: HashMap<String, f64> = match variables {
        Some(dict) => {
            let mut map = HashMap::new();
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                let value: f64 = v.extract()?;
                map.insert(key, value);
            }
            map
        }
        None => HashMap::new(),
    };

    evaluator::evaluate(script, &vars)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

// =============================================================================
// Units
// =============================================================================

/// Units for dimensions in a feature model.
#[pyclass(name = "Units", eq)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyUnits {
    Meters,
    Millimeters,
    Inches,
    Feet,
}

#[pymethods]
impl PyUnits {
    /// Conversion factor from this unit to meters.
    fn to_meters(&self) -> f64 {
        self.to_rust().to_meters()
    }

    /// Conversion factor from meters to this unit.
    fn from_meters(&self) -> f64 {
        self.to_rust().from_meters()
    }

    /// Convert a value from one unit to another.
    #[staticmethod]
    fn convert(value: f64, from: PyUnits, to: PyUnits) -> f64 {
        Units::convert(value, from.to_rust(), to.to_rust())
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyUnits::Meters => "Units.Meters",
            PyUnits::Millimeters => "Units.Millimeters",
            PyUnits::Inches => "Units.Inches",
            PyUnits::Feet => "Units.Feet",
        }
    }
}

impl PyUnits {
    fn to_rust(&self) -> Units {
        match self {
            PyUnits::Meters => Units::Meters,
            PyUnits::Millimeters => Units::Millimeters,
            PyUnits::Inches => Units::Inches,
            PyUnits::Feet => Units::Feet,
        }
    }

    fn from_rust(units: Units) -> Self {
        match units {
            Units::Meters => PyUnits::Meters,
            Units::Millimeters => PyUnits::Millimeters,
            Units::Inches => PyUnits::Inches,
            Units::Feet => PyUnits::Feet,
        }
    }
}

// =============================================================================
// Environment
// =============================================================================

/// Environment settings for a feature model.
///
/// Contains units, named variables, and parametric equations.
/// Variables can be referenced in dimension expressions using Starlark syntax.
///
/// Examples
/// --------
/// >>> env = rmesh.feature.Environment()
/// >>> env.set_variable("width", 100.0)
/// >>> env.set_variable("height", 50.0)
/// >>> env.evaluate("width * height")
/// 5000.0
#[pyclass(name = "Environment")]
pub struct PyEnvironment {
    inner: Environment,
}

#[pymethods]
impl PyEnvironment {
    #[new]
    fn new() -> Self {
        Self {
            inner: Environment::new(),
        }
    }

    /// Get the units for this environment.
    #[getter]
    fn units(&self) -> PyUnits {
        PyUnits::from_rust(self.inner.units)
    }

    /// Set the units for this environment.
    #[setter]
    fn set_units(&mut self, units: PyUnits) {
        self.inner.units = units.to_rust();
    }

    /// Get all variable names.
    #[getter]
    fn variable_names(&self) -> Vec<String> {
        self.inner.variable_names().cloned().collect()
    }

    /// Get all variables as a dictionary.
    #[getter]
    fn variables(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.variables {
            dict.set_item(k, v)?;
        }
        Ok(dict.unbind())
    }

    /// Get a variable value by name.
    fn get_variable(&self, name: &str) -> Option<f64> {
        self.inner.get_variable(name)
    }

    /// Set a variable value.
    fn set_variable(&mut self, name: &str, value: f64) {
        self.inner.set_variable(name, value);
    }

    /// Check if a variable exists.
    fn has_variable(&self, name: &str) -> bool {
        self.inner.has_variable(name)
    }

    /// Evaluate an expression using Starlark.
    ///
    /// All environment variables are available by name.
    fn evaluate(&self, expr: &str) -> PyResult<f64> {
        self.inner
            .evaluate(expr)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Environment(units={:?}, variables={{{}}})",
            self.inner.units,
            self.inner
                .variables
                .iter()
                .map(|(k, v)| format!("'{}': {}", k, v))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

// =============================================================================
// SketchPlane
// =============================================================================

/// A plane on which a sketch is drawn.
///
/// Use the plane helper functions (XY, XZ, YZ) to create common planes,
/// or create a custom plane from origin and normal.
#[pyclass(name = "SketchPlane")]
#[derive(Clone)]
pub struct PySketchPlane {
    inner: SketchPlane,
}

#[pymethods]
impl PySketchPlane {
    /// Create a plane at the origin with +Z normal (XY plane).
    #[staticmethod]
    fn xy() -> Self {
        Self {
            inner: SketchPlane::xy(),
        }
    }

    /// Create a plane at the origin with +Y normal (XZ plane).
    #[staticmethod]
    fn xz() -> Self {
        Self {
            inner: SketchPlane::xz(),
        }
    }

    /// Create a plane at the origin with +X normal (YZ plane).
    #[staticmethod]
    fn yz() -> Self {
        Self {
            inner: SketchPlane::yz(),
        }
    }

    /// Create a plane at a given Z height, parallel to XY.
    #[staticmethod]
    fn at_z(z: f64) -> Self {
        Self {
            inner: SketchPlane::at_z(z),
        }
    }

    /// Create a plane from origin and normal vector.
    #[staticmethod]
    #[pyo3(signature = (origin, normal))]
    fn from_normal(origin: (f64, f64, f64), normal: (f64, f64, f64)) -> Self {
        Self {
            inner: SketchPlane::from_normal(
                Point3::new(origin.0, origin.1, origin.2),
                Vector3::new(normal.0, normal.1, normal.2),
            ),
        }
    }

    /// Get the origin point of this plane.
    #[getter]
    fn origin(&self) -> (f64, f64, f64) {
        let o = self.inner.origin;
        (o.x, o.y, o.z)
    }

    /// Get the normal vector of this plane.
    #[getter]
    fn normal(&self) -> (f64, f64, f64) {
        let n = self.inner.normal();
        (n.x, n.y, n.z)
    }

    fn __repr__(&self) -> String {
        let o = self.inner.origin;
        let n = self.inner.normal();
        format!(
            "SketchPlane(origin=({:.3}, {:.3}, {:.3}), normal=({:.3}, {:.3}, {:.3}))",
            o.x, o.y, o.z, n.x, n.y, n.z
        )
    }
}

// =============================================================================
// Plane helper functions (convenience constructors)
// =============================================================================

/// Create an XY plane at a given Z height.
///
/// Parameters
/// ----------
/// z : float, optional
///     Z coordinate of the plane (default: 0).
///
/// Returns
/// -------
/// SketchPlane
///     A plane parallel to XY at the given Z height.
#[pyfunction]
#[pyo3(signature = (z=0.0))]
#[allow(non_snake_case)]
fn XY(z: f64) -> PySketchPlane {
    PySketchPlane {
        inner: SketchPlane::at_z(z),
    }
}

/// Create an XZ plane at a given Y position.
///
/// Parameters
/// ----------
/// y : float, optional
///     Y coordinate of the plane (default: 0).
///
/// Returns
/// -------
/// SketchPlane
///     A plane parallel to XZ at the given Y position.
#[pyfunction]
#[pyo3(signature = (y=0.0))]
#[allow(non_snake_case)]
fn XZ(y: f64) -> PySketchPlane {
    let mut plane = SketchPlane::xz();
    plane.origin.y = y;
    PySketchPlane { inner: plane }
}

/// Create a YZ plane at a given X position.
///
/// Parameters
/// ----------
/// x : float, optional
///     X coordinate of the plane (default: 0).
///
/// Returns
/// -------
/// SketchPlane
///     A plane parallel to YZ at the given X position.
#[pyfunction]
#[pyo3(signature = (x=0.0))]
#[allow(non_snake_case)]
fn YZ(x: f64) -> PySketchPlane {
    let mut plane = SketchPlane::yz();
    plane.origin.x = x;
    PySketchPlane { inner: plane }
}

// =============================================================================
// Sketch
// =============================================================================

/// A 2D sketch containing geometry on a plane.
///
/// Sketches are used as the basis for operations like Extrude and Revolve.
///
/// Examples
/// --------
/// >>> import rmesh
/// >>> f = rmesh.feature
/// >>> # Rectangle from SVG path
/// >>> sketch = f.Sketch.from_svg("M0,0 L10,0 L10,5 L0,5 Z")
/// >>> # Circle from SVG element
/// >>> sketch = f.Sketch.from_svg("<circle r='5'/>")
/// >>> # On a specific plane
/// >>> sketch = f.Sketch.from_svg("M0,0 L10,0 L10,5 L0,5 Z", plane=f.XY(z=5))
#[pyclass(name = "Sketch")]
#[derive(Clone)]
pub struct PySketch {
    inner: Sketch,
}

#[pymethods]
impl PySketch {
    /// Create a sketch from an SVG path d-string or element.
    ///
    /// Parameters
    /// ----------
    /// svg : str
    ///     An SVG path d-string (e.g., "M0,0 L10,0 L10,10 L0,10 Z")
    ///     or SVG element (e.g., "<circle r='5'/>").
    /// plane : SketchPlane, optional
    ///     The plane to draw the sketch on (default: XY at z=0).
    ///
    /// Returns
    /// -------
    /// Sketch
    ///     A new sketch with the geometry from the SVG.
    #[staticmethod]
    #[pyo3(signature = (svg, plane=None))]
    fn from_svg(svg: &str, plane: Option<PySketchPlane>) -> PyResult<Self> {
        let path2d = Path2D::from_svg(svg)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let plane = plane.map(|p| p.inner).unwrap_or_else(SketchPlane::xy);
        let sketch = path2d_to_sketch(&path2d, plane);

        Ok(Self { inner: sketch })
    }

    /// Get the plane this sketch is on.
    #[getter]
    fn plane(&self) -> PySketchPlane {
        PySketchPlane {
            inner: self.inner.plane.clone(),
        }
    }

    /// Get the number of entities in this sketch.
    fn __len__(&self) -> usize {
        self.inner.entities.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sketch(entities={}, vertices={})",
            self.inner.entities.len(),
            self.inner.vertices.len()
        )
    }
}

/// Convert a Path2D to a Sketch
fn path2d_to_sketch(path: &Path2D, plane: SketchPlane) -> Sketch {
    let mut sketch = Sketch::on_plane(plane);

    // Copy vertices
    for v in &path.vertices {
        sketch.add_vertex(*v);
    }

    // Copy segments as entities
    for seg in &path.segments {
        sketch.add(seg.clone());
    }

    sketch
}

// =============================================================================
// Operation wrappers
// =============================================================================

/// An extrusion operation.
///
/// Extrudes a 2D sketch along its plane normal to create a 3D solid.
///
/// Parameters
/// ----------
/// sketch : Sketch
///     The 2D sketch to extrude.
/// depth : float
///     Extrusion depth (positive = in normal direction).
/// remove : bool, optional
///     If True, subtract from existing geometry (default: False = add).
/// draft_angle : float, optional
///     Draft angle in radians (default: 0 = no draft).
///
/// Examples
/// --------
/// >>> from rmesh.feature import Sketch
/// >>> from rmesh.feature.ops import Extrude
/// >>> # Simple extrusion
/// >>> sketch = Sketch.from_svg("M0,0 L10,0 L10,5 L0,5 Z")
/// >>> op = Extrude(sketch, depth=10)
/// >>> # Cutting operation
/// >>> hole = Sketch.circle(2)
/// >>> cut = Extrude(hole, depth=5, remove=True)
#[pyclass(name = "Extrude")]
#[derive(Clone)]
pub struct PyExtrude {
    inner: Extrude,
}

#[pymethods]
impl PyExtrude {
    #[new]
    #[pyo3(signature = (sketch, depth, remove=false, draft_angle=0.0))]
    fn new(sketch: PySketch, depth: f64, remove: bool, draft_angle: f64) -> Self {
        let sign = if remove { Sign::Remove } else { Sign::Add };
        let inner = Extrude::new(sketch.inner, depth, sign).with_draft(draft_angle);
        Self { inner }
    }

    #[getter]
    fn depth(&self) -> f64 {
        self.inner.depth
    }

    #[getter]
    fn remove(&self) -> bool {
        matches!(self.inner.sign, Sign::Remove)
    }

    #[getter]
    fn draft_angle(&self) -> f64 {
        self.inner.draft_angle
    }

    fn __repr__(&self) -> String {
        format!(
            "Extrude(depth={:.3}, remove={}, draft_angle={:.3})",
            self.inner.depth,
            matches!(self.inner.sign, Sign::Remove),
            self.inner.draft_angle
        )
    }
}

/// A revolve operation.
///
/// Revolves a 2D sketch around an axis to create a solid of revolution.
///
/// Parameters
/// ----------
/// sketch : Sketch
///     The 2D sketch to revolve.
/// axis : tuple[float, float, float], optional
///     Axis direction vector (default: Y axis).
/// axis_origin : tuple[float, float, float], optional
///     Point on the axis (default: origin).
/// angle : float, optional
///     Revolution angle in radians (default: 2*PI = full revolution).
/// remove : bool, optional
///     If True, subtract from existing geometry (default: False = add).
#[pyclass(name = "Revolve")]
#[derive(Clone)]
pub struct PyRevolve {
    inner: Revolve,
}

#[pymethods]
impl PyRevolve {
    #[new]
    #[pyo3(signature = (sketch, axis=None, axis_origin=None, angle=None, remove=false))]
    fn new(
        sketch: PySketch,
        axis: Option<(f64, f64, f64)>,
        axis_origin: Option<(f64, f64, f64)>,
        angle: Option<f64>,
        remove: bool,
    ) -> Self {
        let sign = if remove { Sign::Remove } else { Sign::Add };
        let axis = axis
            .map(|(x, y, z)| Vector3::new(x, y, z))
            .unwrap_or_else(Vector3::y);
        let axis_origin = axis_origin
            .map(|(x, y, z)| Point3::new(x, y, z))
            .unwrap_or_else(Point3::origin);
        let angle = angle.unwrap_or(std::f64::consts::TAU);

        let inner = Revolve::new(sketch.inner, axis, axis_origin, angle, sign);
        Self { inner }
    }

    #[getter]
    fn angle(&self) -> f64 {
        self.inner.angle
    }

    #[getter]
    fn remove(&self) -> bool {
        matches!(self.inner.sign, Sign::Remove)
    }

    fn __repr__(&self) -> String {
        format!(
            "Revolve(angle={:.3}, remove={})",
            self.inner.angle,
            matches!(self.inner.sign, Sign::Remove)
        )
    }
}

/// A sweep operation.
///
/// Sweeps a 2D profile along a 3D path.
///
/// Parameters
/// ----------
/// profile : Sketch
///     The 2D profile sketch to sweep.
/// path : list[tuple[float, float, float]]
///     List of 3D points defining the sweep path.
/// remove : bool, optional
///     If True, subtract from existing geometry (default: False = add).
/// fixed_orientation : bool, optional
///     If True, keep profile orientation fixed (default: False = follow path).
#[pyclass(name = "Sweep")]
#[derive(Clone)]
pub struct PySweep {
    inner: Sweep,
}

#[pymethods]
impl PySweep {
    #[new]
    #[pyo3(signature = (profile, path, remove=false, fixed_orientation=false))]
    fn new(
        profile: PySketch,
        path: Vec<(f64, f64, f64)>,
        remove: bool,
        fixed_orientation: bool,
    ) -> Self {
        let sign = if remove { Sign::Remove } else { Sign::Add };

        // Convert path points to Path3D with a polyline segment
        let vertices: Vec<Point3<f64>> = path
            .into_iter()
            .map(|(x, y, z)| Point3::new(x, y, z))
            .collect();
        let indices: Vec<usize> = (0..vertices.len()).collect();
        let line_segment = Segment3D::Line(Line::from_points(indices));
        let path3d = Path3D::from_vertices_and_segments(vertices, vec![line_segment]);

        let inner = Sweep::new(profile.inner, path3d, sign).with_fixed_orientation(fixed_orientation);
        Self { inner }
    }

    #[getter]
    fn remove(&self) -> bool {
        matches!(self.inner.sign, Sign::Remove)
    }

    fn __repr__(&self) -> String {
        format!("Sweep(remove={})", matches!(self.inner.sign, Sign::Remove))
    }
}

/// A loft operation.
///
/// Creates a solid by blending between multiple profiles.
///
/// Parameters
/// ----------
/// profiles : list[Sketch]
///     List of 2D sketches to loft between (minimum 2).
/// remove : bool, optional
///     If True, subtract from existing geometry (default: False = add).
/// closed : bool, optional
///     If True, connect last profile back to first (default: False).
#[pyclass(name = "Loft")]
#[derive(Clone)]
pub struct PyLoft {
    inner: Loft,
}

#[pymethods]
impl PyLoft {
    #[new]
    #[pyo3(signature = (profiles, remove=false, closed=false))]
    fn new(profiles: Vec<PySketch>, remove: bool, closed: bool) -> PyResult<Self> {
        if profiles.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Loft requires at least 2 profiles",
            ));
        }

        let sign = if remove { Sign::Remove } else { Sign::Add };
        let profiles: Vec<Sketch> = profiles.into_iter().map(|p| p.inner).collect();
        let inner = Loft::new(profiles, sign).with_closed(closed);

        Ok(Self { inner })
    }

    #[getter]
    fn remove(&self) -> bool {
        matches!(self.inner.sign, Sign::Remove)
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.closed
    }

    fn __repr__(&self) -> String {
        format!(
            "Loft(profiles={}, remove={}, closed={})",
            self.inner.profiles.len(),
            matches!(self.inner.sign, Sign::Remove),
            self.inner.closed
        )
    }
}

/// A fillet operation.
///
/// Rounds edges with a given radius.
///
/// Parameters
/// ----------
/// radius : float
///     Fillet radius.
/// edges : list[int] | str | None, optional
///     Edge selection: list of indices, "all", or None for all edges.
#[pyclass(name = "Fillet")]
#[derive(Clone)]
pub struct PyFillet {
    inner: Fillet,
}

#[pymethods]
impl PyFillet {
    #[new]
    #[pyo3(signature = (radius, edges=None))]
    fn new(radius: f64, edges: Option<Py<PyAny>>, py: Python<'_>) -> PyResult<Self> {
        let edge_selection = parse_edge_selection(edges, py)?;
        let inner = Fillet::new(radius).with_edges(edge_selection);
        Ok(Self { inner })
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.inner.radius
    }

    fn __repr__(&self) -> String {
        format!("Fillet(radius={:.3})", self.inner.radius)
    }
}

/// A chamfer operation.
///
/// Bevels edges with a given distance.
///
/// Parameters
/// ----------
/// distance : float
///     Chamfer distance.
/// distance2 : float | None, optional
///     Second distance for asymmetric chamfer.
/// edges : list[int] | str | None, optional
///     Edge selection: list of indices, "all", or None for all edges.
#[pyclass(name = "Chamfer")]
#[derive(Clone)]
pub struct PyChamfer {
    inner: Chamfer,
}

#[pymethods]
impl PyChamfer {
    #[new]
    #[pyo3(signature = (distance, distance2=None, edges=None))]
    fn new(
        distance: f64,
        distance2: Option<f64>,
        edges: Option<Py<PyAny>>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let edge_selection = parse_edge_selection(edges, py)?;
        let inner = match distance2 {
            Some(d2) => Chamfer::asymmetric(distance, d2).with_edges(edge_selection),
            None => Chamfer::new(distance).with_edges(edge_selection),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn distance(&self) -> f64 {
        self.inner.distance
    }

    #[getter]
    fn distance2(&self) -> Option<f64> {
        self.inner.distance2
    }

    fn __repr__(&self) -> String {
        match self.inner.distance2 {
            Some(d2) => format!("Chamfer(distance={:.3}, distance2={:.3})", self.inner.distance, d2),
            None => format!("Chamfer(distance={:.3})", self.inner.distance),
        }
    }
}

/// Parse edge selection from Python object
fn parse_edge_selection(edges: Option<Py<PyAny>>, py: Python<'_>) -> PyResult<EdgeSelection> {
    match edges {
        None => Ok(EdgeSelection::All),
        Some(obj) => {
            // Try as list of integers
            if let Ok(indices) = obj.extract::<Vec<usize>>(py) {
                return Ok(EdgeSelection::Indices(indices));
            }
            // Try as string
            if let Ok(s) = obj.extract::<String>(py) {
                return match s.to_lowercase().as_str() {
                    "all" => Ok(EdgeSelection::All),
                    other => Ok(EdgeSelection::Filter(other.to_string())),
                };
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "edges must be list[int], 'all', or a filter string",
            ))
        }
    }
}

// =============================================================================
// FeatureModel
// =============================================================================

/// A feature-based CAD model.
///
/// Contains an environment with units and variables, plus an ordered list of operations.
/// Operations are executed sequentially to build the final geometry.
///
/// Parameters
/// ----------
/// units : str, optional
///     Unit system: "mm", "in", "m", or "ft" (default: "mm").
///
/// Examples
/// --------
/// >>> from rmesh.feature import FeatureModel, Sketch
/// >>> from rmesh.feature.ops import Extrude
/// >>> model = FeatureModel(units="in")
/// >>> model.add(Extrude(Sketch.from_svg("M0,0 L10,0 L10,10 L0,10 Z"), depth=5))
/// >>> mesh = model.to_mesh()
/// >>> print(model.to_json())
#[pyclass(name = "FeatureModel")]
pub struct PyFeatureModel {
    inner: FeatureModel,
}

#[pymethods]
impl PyFeatureModel {
    #[new]
    #[pyo3(signature = (*operations, units=None))]
    fn new(operations: &Bound<'_, pyo3::types::PyTuple>, units: Option<&str>) -> PyResult<Self> {
        let mut model = FeatureModel::new();

        if let Some(u) = units {
            let rust_units = match u.to_lowercase().as_str() {
                "mm" | "millimeters" | "millimeter" => Units::Millimeters,
                "in" | "inches" | "inch" => Units::Inches,
                "m" | "meters" | "meter" => Units::Meters,
                "ft" | "feet" | "foot" => Units::Feet,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown units: '{}'. Use 'mm', 'in', 'm', or 'ft'",
                        other
                    )));
                }
            };
            model.environment.units = rust_units;
        }

        // Add any operations passed to constructor
        for item in operations.iter() {
            let op = extract_operation(&item)?;
            model.operations.push(op);
        }

        Ok(Self { inner: model })
    }

    /// Add an operation to the model.
    ///
    /// Parameters
    /// ----------
    /// operation : Extrude | Revolve | Sweep | Loft | Fillet | Chamfer
    ///     The operation to add.
    ///
    /// Returns
    /// -------
    /// self
    ///     Returns the model for chaining.
    fn add(&mut self, operation: &Bound<'_, PyAny>) -> PyResult<()> {
        let op = extract_operation(operation)?;
        self.inner.operations.push(op);
        Ok(())
    }

    /// Execute the model and generate a mesh.
    ///
    /// Parameters
    /// ----------
    /// depth : int, optional
    ///     Octree depth for mesh generation (1-15, default: 6).
    ///     Higher values produce more detailed meshes but take longer.
    ///
    /// Returns
    /// -------
    /// Trimesh
    ///     The generated triangle mesh.
    #[pyo3(signature = (depth=6))]
    fn to_mesh(&self, depth: u8) -> PyResult<PyTrimesh> {
        let backend = FidgetBackend::new();
        let settings = FidgetSettings::with_depth(depth);

        let mesh = backend
            .execute(&self.inner, &settings)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyTrimesh::new_from_trimesh(mesh))
    }

    /// Get the environment for this model.
    #[getter]
    fn environment(&self) -> PyEnvironment {
        PyEnvironment {
            inner: self.inner.environment.clone(),
        }
    }

    /// Set a variable in the environment.
    fn set_variable(&mut self, name: &str, value: f64) {
        self.inner.environment.set_variable(name, value);
    }

    /// Get a variable from the environment.
    fn get_variable(&self, name: &str) -> Option<f64> {
        self.inner.environment.get_variable(name)
    }

    /// Get the number of operations.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if the model has no operations.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Serialize the model to JSON using rmesh format.
    ///
    /// The JSON includes a wrapper with `rmesh_name` and `data` fields
    /// for type identification and compatibility.
    fn to_json(&self) -> PyResult<String> {
        let bytes = self
            .inner
            .to_bytes(None, true) // as_json=true
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        String::from_utf8(bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Serialize the model to compressed binary format.
    ///
    /// Uses MessagePack + zstd compression with integrity checking.
    /// Much more compact than JSON for storage/transmission.
    ///
    /// Parameters
    /// ----------
    /// compress_level : int, optional
    ///     Zstd compression level 1-22 (default: 3).
    ///
    /// Returns
    /// -------
    /// bytes
    ///     Compressed binary data with 128-byte header.
    #[pyo3(signature = (compress_level=None))]
    fn to_bytes<'py>(&self, py: Python<'py>, compress_level: Option<i32>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self
            .inner
            .to_bytes(compress_level, false) // as_json=false
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Load a model from a file path.
    ///
    /// Format is auto-detected from file extension (.json, .sldprt).
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the file.
    ///
    /// Returns
    /// -------
    /// FeatureModel
    ///     The loaded model.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = load_feature_model_from_path(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Load a model from bytes.
    ///
    /// Auto-detects rmesh JSON vs binary format. For external formats
    /// like SLDPRT, use `load()` with a file path.
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     Raw bytes (rmesh JSON or binary format).
    ///
    /// Returns
    /// -------
    /// FeatureModel
    ///     The loaded model.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = FeatureModel::from_bytes(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "FeatureModel(operations={}, variables={})",
            self.inner.len(),
            self.inner.environment.variables.len()
        )
    }
}

/// Extract an Operation from a Python object
fn extract_operation(obj: &Bound<'_, PyAny>) -> PyResult<Operation> {
    if let Ok(extrude) = obj.extract::<PyExtrude>() {
        return Ok(Operation::Extrude(extrude.inner));
    }
    if let Ok(revolve) = obj.extract::<PyRevolve>() {
        return Ok(Operation::Revolve(revolve.inner));
    }
    if let Ok(sweep) = obj.extract::<PySweep>() {
        return Ok(Operation::Sweep(sweep.inner));
    }
    if let Ok(loft) = obj.extract::<PyLoft>() {
        return Ok(Operation::Loft(loft.inner));
    }
    if let Ok(fillet) = obj.extract::<PyFillet>() {
        return Ok(Operation::Fillet(fillet.inner));
    }
    if let Ok(chamfer) = obj.extract::<PyChamfer>() {
        return Ok(Operation::Chamfer(chamfer.inner));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected Extrude, Revolve, Sweep, Loft, Fillet, or Chamfer",
    ))
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the feature submodule with all classes flattened.
pub fn register_feature_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let feature = PyModule::new(py, "feature")?;

    // Core classes
    feature.add_class::<PyUnits>()?;
    feature.add_class::<PyEnvironment>()?;
    feature.add_class::<PyFeatureModel>()?;
    feature.add_class::<PySketch>()?;
    feature.add_class::<PySketchPlane>()?;

    // Operations (flattened, no submodule)
    feature.add_class::<PyExtrude>()?;
    feature.add_class::<PyRevolve>()?;
    feature.add_class::<PySweep>()?;
    feature.add_class::<PyLoft>()?;
    feature.add_class::<PyFillet>()?;
    feature.add_class::<PyChamfer>()?;

    // Plane helpers (flattened, no submodule)
    feature.add_function(wrap_pyfunction!(XY, &feature)?)?;
    feature.add_function(wrap_pyfunction!(XZ, &feature)?)?;
    feature.add_function(wrap_pyfunction!(YZ, &feature)?)?;

    parent.add_submodule(&feature)?;

    // Register in sys.modules for proper import support
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("rmesh.feature", &feature)?;

    Ok(())
}
