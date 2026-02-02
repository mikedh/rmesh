use nalgebra::{Matrix4, Point2};

/// Orthographic 2D camera.
pub struct View2D {
    /// Center of the view in data coordinates.
    pub center: Point2<f64>,
    /// Half the visible height in data coordinates; width = half_height * aspect.
    pub half_height: f64,
    /// Axis-aligned bounding box of the data.
    pub data_bounds: (Point2<f64>, Point2<f64>),
}

impl View2D {
    pub fn new(bounds_min: Point2<f64>, bounds_max: Point2<f64>) -> Self {
        let mut v = Self {
            center: Point2::origin(),
            half_height: 1.0,
            data_bounds: (bounds_min, bounds_max),
        };
        v.fit_to_data();
        v
    }

    /// Auto-fit the view to the data bounds with 10% padding.
    pub fn fit_to_data(&mut self) {
        let min = self.data_bounds.0;
        let max = self.data_bounds.1;
        self.center = Point2::new((min.x + max.x) * 0.5, (min.y + max.y) * 0.5);
        let h = (max.y - min.y) * 0.5;
        self.half_height = if h > 1e-12 { h * 1.1 } else { 1.0 };
    }

    /// Build an orthographic projection matrix for the current view.
    #[allow(clippy::cast_possible_truncation)]
    pub fn view_proj_matrix(&self, aspect: f64) -> Matrix4<f32> {
        let hh = self.half_height;
        let hw = hh * aspect;
        let left = (self.center.x - hw) as f32;
        let right = (self.center.x + hw) as f32;
        let bottom = (self.center.y - hh) as f32;
        let top = (self.center.y + hh) as f32;
        // wgpu clip space: x,y in [-1,1], z in [0,1]
        #[rustfmt::skip]
        let m = Matrix4::new(
            2.0 / (right - left), 0.0,                    0.0, -(right + left) / (right - left),
            0.0,                  2.0 / (top - bottom),    0.0, -(top + bottom) / (top - bottom),
            0.0,                  0.0,                     1.0,  0.0,
            0.0,                  0.0,                     0.0,  1.0,
        );
        m
    }

    /// Pan by a pixel delta (normalised to [0..1] of window).
    pub fn pan(&mut self, dx: f64, dy: f64, aspect: f64) {
        let hw = self.half_height * aspect;
        // dx/dy are fraction-of-window; map to data units
        self.center.x -= dx * 2.0 * hw;
        self.center.y += dy * 2.0 * self.half_height;
    }

    /// Zoom toward a cursor position (in pixels). factor < 1 zooms in.
    pub fn zoom_toward(&mut self, factor: f64, cursor_x: f64, cursor_y: f64, w: u32, h: u32) {
        let aspect = f64::from(w) / f64::from(h);
        // Convert pixel to data coords *before* zoom
        let data = self.pixel_to_data(cursor_x, cursor_y, w, h, aspect);
        self.half_height *= factor;
        // After zoom, the same pixel should map to the same data point.
        let data_after = self.pixel_to_data(cursor_x, cursor_y, w, h, aspect);
        self.center.x += data.x - data_after.x;
        self.center.y += data.y - data_after.y;
    }

    /// Fit view to a rectangle defined by two pixel corners.
    pub fn zoom_to_box(&mut self, px0: f64, py0: f64, px1: f64, py1: f64, win_w: u32, win_h: u32) {
        let aspect = f64::from(win_w) / f64::from(win_h);
        let d0 = self.pixel_to_data(px0, py0, win_w, win_h, aspect);
        let d1 = self.pixel_to_data(px1, py1, win_w, win_h, aspect);
        let cx = (d0.x + d1.x) * 0.5;
        let cy = (d0.y + d1.y) * 0.5;
        let box_hw = (d1.x - d0.x).abs() * 0.5;
        let box_hh = (d1.y - d0.y).abs() * 0.5;
        if box_hh < 1e-12 || box_hw < 1e-12 {
            return;
        }
        // Pick the larger extent so the box fits
        let hh_from_w = box_hw / aspect;
        self.half_height = box_hh.max(hh_from_w);
        self.center = Point2::new(cx, cy);
    }

    /// Convert pixel coords to data coords.
    pub fn pixel_to_data(
        &self,
        px: f64,
        py: f64,
        win_w: u32,
        win_h: u32,
        aspect: f64,
    ) -> Point2<f64> {
        let hw = self.half_height * aspect;
        let ndc_x = (px / f64::from(win_w)) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / f64::from(win_h)) * 2.0; // flip y
        Point2::new(
            self.center.x + ndc_x * hw,
            self.center.y + ndc_y * self.half_height,
        )
    }
}
