mod app;
mod app2d;
pub(crate) mod gpu;
pub(crate) mod input;
#[cfg(feature = "python")]
mod python;
mod viewer_thread;

use anyhow::Result;
use nalgebra::Point2;
use rmesh::path::Path2D;
use rmesh::path::Polygon2D;
use rmesh::scene::Scene;

/// A filled polygon: (vertices, triangle indices, RGBA color)
pub type FilledPolygon = (Vec<Point2<f64>>, Vec<[usize; 3]>, [f32; 4]);

/// Options for configuring the viewer window.
pub struct ViewerOptions {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub background: [f32; 3],
}

impl Default for ViewerOptions {
    fn default() -> Self {
        Self {
            title: "rmesh viewer".to_string(),
            width: 1280,
            height: 720,
            background: [1.0, 1.0, 1.0],
        }
    }
}

/// Trait for displaying scenes in an interactive viewer window.
pub trait SceneViewer {
    /// Open an interactive viewer window. Blocks until closed.
    fn show(&self) -> Result<()>;

    /// Open an interactive viewer window with custom options. Blocks until closed.
    fn show_with_options(&self, options: ViewerOptions) -> Result<()>;
}

/// Intermediate geometry for the 2D viewer.
#[derive(Clone)]
pub struct View2DData {
    /// Polylines: each is (points, RGBA color).
    pub lines: Vec<(Vec<Point2<f64>>, [f32; 4])>,
    /// Filled polygons: each is (vertices, triangles, RGBA color).
    pub fills: Vec<FilledPolygon>,
    /// Axis-aligned bounding box.
    pub bounds: (Point2<f64>, Point2<f64>),
}

/// Trait for displaying 2D geometry in a viewer window.
pub trait Viewer2D {
    fn view_2d_data(&self) -> View2DData;

    fn show_2d(&self) -> Result<()> {
        self.show_2d_with_options(ViewerOptions {
            title: "rmesh 2D".to_string(),
            background: [1.0, 1.0, 1.0],
            ..Default::default()
        })
    }

    fn show_2d_with_options(&self, options: ViewerOptions) -> Result<()> {
        env_logger::try_init().ok();
        let data = self.view_2d_data();
        app2d::run(&data, options)
    }
}

impl Viewer2D for Path2D {
    fn view_2d_data(&self) -> View2DData {
        let segments = self.discretize();
        let color = [0.0_f32, 0.8, 0.2, 1.0]; // green

        let mut lines = Vec::new();
        let mut min = Point2::new(f64::INFINITY, f64::INFINITY);
        let mut max = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);

        for seg in &segments {
            if seg.len() < 2 {
                continue;
            }
            for p in seg {
                min.x = min.x.min(p.x);
                min.y = min.y.min(p.y);
                max.x = max.x.max(p.x);
                max.y = max.y.max(p.y);
            }
            lines.push((seg.clone(), color));
        }

        if min.x > max.x {
            min = Point2::new(-1.0, -1.0);
            max = Point2::new(1.0, 1.0);
        }

        View2DData {
            lines,
            fills: Vec::new(),
            bounds: (min, max),
        }
    }
}

impl Viewer2D for Polygon2D {
    fn view_2d_data(&self) -> View2DData {
        let fill_color = [0.2_f32, 0.5, 0.8, 0.3];
        let outline_color = [0.2_f32, 0.4, 0.8, 1.0];

        let (bmin, bmax) = self
            .bounds()
            .unwrap_or((Point2::new(-1.0, -1.0), Point2::new(1.0, 1.0)));

        // Build lines from exterior + interiors
        let mut lines = Vec::new();
        let mut ext_line = self.exterior.clone();
        if !ext_line.is_empty() {
            ext_line.push(ext_line[0]); // close
            lines.push((ext_line, outline_color));
        }
        for hole in &self.interiors {
            let mut h = hole.clone();
            if !h.is_empty() {
                h.push(h[0]); // close
                lines.push((h, outline_color));
            }
        }

        // Triangulate for fill using earcut-style approach
        let fills = triangulate_polygon(self, fill_color);

        View2DData {
            lines,
            fills,
            bounds: (bmin, bmax),
        }
    }
}

impl Viewer2D for [Polygon2D] {
    fn view_2d_data(&self) -> View2DData {
        let fill_color = [0.2_f32, 0.5, 0.8, 0.3];
        let outline_color = [0.2_f32, 0.4, 0.8, 1.0];

        let mut lines = Vec::new();
        let mut fills = Vec::new();
        let mut min = Point2::new(f64::INFINITY, f64::INFINITY);
        let mut max = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);

        for poly in self {
            if let Some((bmin, bmax)) = poly.bounds() {
                min.x = min.x.min(bmin.x);
                min.y = min.y.min(bmin.y);
                max.x = max.x.max(bmax.x);
                max.y = max.y.max(bmax.y);
            }

            let mut ext_line = poly.exterior.clone();
            if !ext_line.is_empty() {
                ext_line.push(ext_line[0]);
                lines.push((ext_line, outline_color));
            }
            for hole in &poly.interiors {
                let mut h = hole.clone();
                if !h.is_empty() {
                    h.push(h[0]);
                    lines.push((h, outline_color));
                }
            }

            fills.extend(triangulate_polygon(poly, fill_color));
        }

        if min.x > max.x {
            min = Point2::new(-1.0, -1.0);
            max = Point2::new(1.0, 1.0);
        }

        View2DData {
            lines,
            fills,
            bounds: (min, max),
        }
    }
}

/// Triangulate a single polygon into fill data using earcut.
fn triangulate_polygon(poly: &Polygon2D, color: [f32; 4]) -> Vec<FilledPolygon> {
    // Build a flat vertex array and use earcut
    let mut flat: Vec<[f64; 2]> = Vec::new();
    let mut hole_starts = Vec::new();

    // Exterior ring
    for p in &poly.exterior {
        flat.push([p.x, p.y]);
    }

    // Interior rings
    for hole in &poly.interiors {
        hole_starts.push(flat.len());
        for p in hole {
            flat.push([p.x, p.y]);
        }
    }

    // Earcut triangulation
    let mut earcut = earcut::Earcut::new();
    let mut result: Vec<usize> = Vec::new();
    earcut.earcut(flat.iter().copied(), &hole_starts, &mut result);

    if result.is_empty() {
        return Vec::new();
    }

    let vertices: Vec<Point2<f64>> = flat.iter().map(|p| Point2::new(p[0], p[1])).collect();
    let tris: Vec<[usize; 3]> = result.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

    vec![(vertices, tris, color)]
}

/// Show 2D data directly in a viewer window. Blocks until closed.
pub fn show_2d_data(data: &View2DData, options: ViewerOptions) -> Result<()> {
    env_logger::try_init().ok();
    app2d::run(data, options)
}

impl SceneViewer for Scene {
    fn show(&self) -> Result<()> {
        self.show_with_options(ViewerOptions::default())
    }

    fn show_with_options(&self, options: ViewerOptions) -> Result<()> {
        env_logger::try_init().ok();
        app::run(self, options)
    }
}
