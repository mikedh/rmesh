//! Raster operations for paths: boolean image masks, raster clipping,
//! and polyline resampling.

use nalgebra::{Matrix3, Point2};

/// A simple boolean image for raster mask operations.
/// Row-major storage, nonzero = inside.
pub struct BooleanImage {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl BooleanImage {
    /// Create a new boolean image from raw data.
    pub fn new(data: Vec<u8>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height);
        Self {
            data,
            width,
            height,
        }
    }

    /// Create a boolean image filled with a single value.
    pub fn filled(width: usize, height: usize, value: bool) -> Self {
        Self {
            data: vec![if value { 255 } else { 0 }; width * height],
            width,
            height,
        }
    }

    /// Check if a single pixel coordinate is inside the mask.
    #[inline]
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    // SAFETY: px >= 0 && py >= 0 guards ensure no sign loss; pixel coords are bounded by image dimensions.
    pub fn contains_pixel(&self, px: i64, py: i64) -> bool {
        px >= 0
            && py >= 0
            && (px as usize) < self.width
            && (py as usize) < self.height
            && self.data[py as usize * self.width + px as usize] != 0
    }

    /// Batch check: transform metric points to pixel coords and check mask.
    ///
    /// Fuses the affine transform and mask lookup into one pass with no
    /// intermediate allocation beyond the output vector.
    #[allow(clippy::cast_possible_truncation)]
    pub fn contains_transformed(
        &self,
        points: &[Point2<f64>],
        to_raster: &Matrix3<f64>,
    ) -> Vec<bool> {
        // Extract transform components once
        let sx = to_raster[(0, 0)];
        let sy = to_raster[(1, 1)];
        let tx = to_raster[(0, 2)];
        let ty = to_raster[(1, 2)];

        points
            .iter()
            .map(|p| {
                let px = (p.x * sx + tx).round() as i64;
                let py = (p.y * sy + ty).round() as i64;
                self.contains_pixel(px, py)
            })
            .collect()
    }
}

/// Resample a polyline at uniform arc-length intervals.
///
/// Returns points spaced approximately `step` apart along the polyline.
/// Always includes the first and last points.
pub fn resample_polyline(points: &[Point2<f64>], step: f64) -> Vec<Point2<f64>> {
    if points.len() < 2 || step <= 0.0 {
        return points.to_vec();
    }

    let mut result = Vec::new();
    result.push(points[0]);

    let mut residual = 0.0; // distance accumulated since last emitted point

    for window in points.windows(2) {
        let a = window[0];
        let b = window[1];
        let seg_len = ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
        if seg_len < 1e-15 {
            continue;
        }

        let dx = (b.x - a.x) / seg_len;
        let dy = (b.y - a.y) / seg_len;

        let mut dist = step - residual; // distance along this segment to next sample
        while dist <= seg_len {
            result.push(Point2::new(a.x + dx * dist, a.y + dy * dist));
            dist += step;
        }
        residual = seg_len - (dist - step);
    }

    // Always include the last point
    let last = points[points.len() - 1];
    if let Some(prev) = result.last() {
        let d = ((last.x - prev.x).powi(2) + (last.y - prev.y).powi(2)).sqrt();
        if d > 1e-12 {
            result.push(last);
        }
    }

    result
}

/// Clip a polyline against a boolean image mask, returning visible fragments.
///
/// Discretizes the path, resamples at pixel-level density, batch-checks
/// against the mask, inflates transitions to close small gaps, links
/// nearby blocks, and returns contiguous visible fragments.
///
/// # Arguments
/// * `polyline` - The discretized polyline points (metric coordinates)
/// * `mask` - Boolean image mask (nonzero = inside)
/// * `to_raster` - Affine transform from metric → pixel coordinates
/// * `inflate_radius` - 1D dilation radius in samples for gap closing along curve
/// * `link_distance` - Metric distance threshold for merging nearby visible blocks
///
/// # Returns
/// Vector of polyline fragments (metric coordinates) that are inside the mask.
pub fn raster_clip(
    polyline: &[Point2<f64>],
    mask: &BooleanImage,
    to_raster: &Matrix3<f64>,
    inflate_radius: usize,
    link_distance: f64,
) -> Vec<Vec<Point2<f64>>> {
    if polyline.len() < 2 {
        return Vec::new();
    }

    // Derive pixel scale from transform to determine resample step
    let scale = to_raster[(0, 0)].abs();
    if scale < 1e-15 {
        return Vec::new();
    }
    let step = 1.0 / scale;

    // Resample at ~1 sample per pixel
    let resampled = resample_polyline(polyline, step);
    if resampled.is_empty() {
        return Vec::new();
    }

    // Batch check all samples against the mask
    let mut inside = mask.contains_transformed(&resampled, to_raster);

    // Check if the path is closed (first ≈ last point)
    let is_closed = {
        let first = resampled[0];
        let last = resampled[resampled.len() - 1];
        ((first.x - last.x).powi(2) + (first.y - last.y).powi(2)).sqrt() < step * 2.0
    };

    // Check if all points are inside
    let all_inside = inside.iter().all(|&v| v);
    if all_inside {
        // Return the original polyline, not the resampled one
        return vec![polyline.to_vec()];
    }

    // 1D inflate: for each True→False transition, extend True by inflate_radius
    if inflate_radius > 0 {
        inflate_mask_1d(&mut inside, inflate_radius);
    }

    // Find contiguous blocks of True values
    let mut blocks = find_blocks(&inside, is_closed);

    // Link nearby blocks if their endpoints are close in metric space
    if link_distance > 0.0 {
        link_blocks(&mut blocks, &resampled, link_distance);
    }

    // Extract sub-sequences from resampled points
    blocks
        .into_iter()
        .filter_map(|(start, end)| {
            if end <= start {
                return None;
            }
            let fragment: Vec<Point2<f64>> =
                resampled[start..=end.min(resampled.len() - 1)].to_vec();
            if fragment.len() >= 2 {
                Some(fragment)
            } else {
                None
            }
        })
        .collect()
}

/// 1D mask inflation: extend True regions by `radius` samples in both directions.
///
/// For each True→False transition, extends True by `radius` in the outward direction.
/// This bridges small gaps (≤2*radius samples) along the curve.
fn inflate_mask_1d(mask: &mut [bool], radius: usize) {
    let n = mask.len();
    if n == 0 || radius == 0 {
        return;
    }

    // Find transition points and inflate
    // We need a copy to avoid reading modified values
    let original = mask.to_vec();

    for i in 0..n {
        if !original[i] {
            // Check if there's a True value within `radius` in either direction
            let near_true =
                (1..=radius).any(|d| (i >= d && original[i - d]) || (i + d < n && original[i + d]));
            if near_true {
                mask[i] = true;
            }
        }
    }
}

/// Find contiguous blocks of True values in a boolean array.
///
/// Returns `(start, end)` pairs (inclusive indices).
/// For closed paths, wraps around to merge blocks that span the boundary.
fn find_blocks(mask: &[bool], is_closed: bool) -> Vec<(usize, usize)> {
    let n = mask.len();
    if n == 0 {
        return Vec::new();
    }

    let mut blocks = Vec::new();
    let mut block_start: Option<usize> = None;

    for (i, &m) in mask.iter().enumerate() {
        if m {
            if block_start.is_none() {
                block_start = Some(i);
            }
        } else if let Some(start) = block_start.take() {
            blocks.push((start, i - 1));
        }
    }
    // Close final block
    if let Some(start) = block_start {
        blocks.push((start, n - 1));
    }

    // For closed paths, merge the first and last block if they wrap around
    if is_closed && blocks.len() >= 2 {
        let last = blocks.len() - 1;
        if blocks[last].1 == n - 1 && blocks[0].0 == 0 {
            let merged_start = blocks[last].0;
            let merged_end = blocks[0].1;
            blocks[0] = (merged_start, merged_end + n); // signal wrapping
            blocks.pop();
        }
    }

    blocks
}

/// Merge nearby blocks if their endpoints are within `link_distance` in metric space.
fn link_blocks(blocks: &mut Vec<(usize, usize)>, points: &[Point2<f64>], link_distance: f64) {
    if blocks.len() < 2 {
        return;
    }

    let dist_sq = link_distance * link_distance;
    let n = points.len();

    let mut merged = true;
    while merged {
        merged = false;
        let mut i = 0;
        while i + 1 < blocks.len() {
            let end_idx = blocks[i].1.min(n - 1);
            let start_idx = blocks[i + 1].0.min(n - 1);
            let end_pt = points[end_idx];
            let start_pt = points[start_idx];
            let d = (end_pt.x - start_pt.x).powi(2) + (end_pt.y - start_pt.y).powi(2);
            if d < dist_sq {
                blocks[i].1 = blocks[i + 1].1;
                blocks.remove(i + 1);
                merged = true;
            } else {
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_polyline() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)];
        let resampled = resample_polyline(&points, 2.0);
        // Should have samples at 0, 2, 4, 6, 8, 10
        assert_eq!(resampled.len(), 6);
        assert!((resampled[0].x - 0.0).abs() < 1e-10);
        assert!((resampled[1].x - 2.0).abs() < 1e-10);
        assert!((resampled.last().unwrap().x - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_resample_polyline_multi_segment() {
        // L-shaped path: (0,0) -> (10,0) -> (10,10)
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
        ];
        let resampled = resample_polyline(&points, 5.0);
        // Total length = 20, step = 5 → samples at 0, 5, 10, 15, 20
        assert_eq!(resampled.len(), 5);
    }

    #[test]
    fn test_boolean_image_contains() {
        // 4x4 image with a 2x2 block set in the center
        let mut data = vec![0u8; 16];
        data[5] = 255; // (1,1)
        data[6] = 255; // (2,1)
        data[9] = 255; // (1,2)
        data[10] = 255; // (2,2)
        let img = BooleanImage::new(data, 4, 4);

        assert!(!img.contains_pixel(0, 0));
        assert!(img.contains_pixel(1, 1));
        assert!(img.contains_pixel(2, 2));
        assert!(!img.contains_pixel(3, 3));
        assert!(!img.contains_pixel(-1, 0));
        assert!(!img.contains_pixel(0, 4));
    }

    #[test]
    fn test_contains_transformed() {
        // 10x10 image, all set
        let img = BooleanImage::filled(10, 10, true);

        // Identity-ish transform: 1 pixel per unit, no offset
        let to_raster = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

        let points = vec![
            Point2::new(5.0, 5.0),  // inside
            Point2::new(15.0, 5.0), // outside
            Point2::new(-1.0, 0.0), // outside
        ];
        let result = img.contains_transformed(&points, &to_raster);
        assert_eq!(result, vec![true, false, false]);
    }

    #[test]
    fn test_inflate_mask_1d() {
        // Gap of 2 in the middle: ...TTT.FF.TTT...
        let mut mask = vec![true, true, true, false, false, true, true, true];
        // Inflate by 1: should NOT bridge gap of 2
        inflate_mask_1d(&mut mask, 1);
        assert_eq!(
            mask,
            vec![
                true, true, true, true, true, // inflated from neighbors
                true, true, true,
            ]
        );
    }

    #[test]
    fn test_inflate_mask_1d_small_gap() {
        // Gap of 1: ...TTT.F.TTT...
        let mut mask = vec![true, true, true, false, true, true, true];
        inflate_mask_1d(&mut mask, 1);
        assert!(mask.iter().all(|&v| v), "single gap should be bridged");
    }

    #[test]
    fn test_find_blocks() {
        let mask = vec![false, true, true, false, false, true, false];
        let blocks = find_blocks(&mask, false);
        assert_eq!(blocks, vec![(1, 2), (5, 5)]);
    }

    #[test]
    fn test_find_blocks_closed_wrap() {
        let mask = vec![true, true, false, false, true, true];
        let blocks = find_blocks(&mask, true);
        // Should merge first (0,1) and last (4,5) blocks
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, 4); // starts at 4
    }

    #[test]
    fn test_raster_clip_all_inside() {
        // Square polyline, entire mask is true
        let polyline = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let mask = BooleanImage::filled(100, 100, true);
        let to_raster = Matrix3::new(10.0, 0.0, 50.0, 0.0, -10.0, 50.0, 0.0, 0.0, 1.0);

        let result = raster_clip(&polyline, &mask, &to_raster, 0, 0.0);
        assert_eq!(result.len(), 1);
        // Should return original polyline
        assert_eq!(result[0].len(), polyline.len());
    }

    #[test]
    fn test_raster_clip_half_masked() {
        // Horizontal line from (0,0) to (10,0)
        let polyline = vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)];
        // 20x2 mask: left half = true, right half = false
        let mut data = vec![0u8; 40];
        for y in 0..2 {
            for x in 0..10 {
                data[y * 20 + x] = 255;
            }
        }
        let mask = BooleanImage::new(data, 20, 2);

        // 2 pixels per unit, offset so (0,0) maps to pixel (0,1)
        let to_raster = Matrix3::new(2.0, 0.0, 0.0, 0.0, -2.0, 1.0, 0.0, 0.0, 1.0);

        let result = raster_clip(&polyline, &mask, &to_raster, 0, 0.0);
        // Should get roughly the left half as a fragment
        assert!(!result.is_empty());
        // First fragment should start near 0 and end near 5 (half of 10)
        let frag = &result[0];
        assert!(frag[0].x < 1.0);
        assert!((frag.last().unwrap().x - 5.0).abs() < 1.5);
    }

    #[test]
    fn test_link_blocks() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.5, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(4.0, 0.0),
        ];
        // Two blocks close together
        let mut blocks = vec![(0, 1), (3, 5)];
        link_blocks(&mut blocks, &points, 2.0);
        // Should merge since gap endpoints (idx 1, idx 3) are 1.5 apart
        assert_eq!(blocks.len(), 1);
    }
}
