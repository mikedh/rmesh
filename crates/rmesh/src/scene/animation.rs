//! Animation types for scene node transforms.

/// Interpolation method for animation keyframes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Interpolation {
    /// No interpolation, jump to next value.
    Step,
    /// Linear interpolation between keyframes.
    #[default]
    Linear,
    /// Cubic spline interpolation with tangents.
    CubicSpline,
}

/// The property being animated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationPath {
    /// Translation (VEC3).
    Translation,
    /// Rotation as quaternion (VEC4).
    Rotation,
    /// Scale (VEC3).
    Scale,
    /// Morph target weights.
    Weights,
}

/// A sampler defining keyframe data for an animation.
#[derive(Debug, Clone)]
pub struct AnimationSampler {
    /// Timestamps in seconds for each keyframe.
    pub timestamps: Vec<f64>,
    /// Values at each keyframe. Shape depends on path:
    /// - Translation/Scale: VEC3 (3 elements per keyframe)
    /// - Rotation: VEC4 (4 elements per keyframe)
    /// - Weights: N elements per keyframe
    pub values: Vec<f64>,
    /// Number of components per value (3 for VEC3, 4 for VEC4, etc.).
    pub components: usize,
    /// Interpolation method.
    pub interpolation: Interpolation,
}

impl AnimationSampler {
    /// Sample the animation at time t, returning interpolated values.
    pub fn sample(&self, t: f64) -> Vec<f64> {
        if self.timestamps.is_empty() {
            return vec![0.0; self.components];
        }

        // Clamp to valid range
        let t = t.clamp(self.timestamps[0], *self.timestamps.last().unwrap());

        // Find surrounding keyframes
        let mut i = 0;
        while i < self.timestamps.len() - 1 && self.timestamps[i + 1] <= t {
            i += 1;
        }

        if i >= self.timestamps.len() - 1 {
            // At or past the last keyframe
            let start = i * self.components;
            return self.values[start..start + self.components].to_vec();
        }

        match self.interpolation {
            Interpolation::Step => {
                let start = i * self.components;
                self.values[start..start + self.components].to_vec()
            }
            Interpolation::Linear => {
                let t0 = self.timestamps[i];
                let t1 = self.timestamps[i + 1];
                let alpha = if (t1 - t0).abs() < 1e-10 {
                    0.0
                } else {
                    (t - t0) / (t1 - t0)
                };

                let start0 = i * self.components;
                let start1 = (i + 1) * self.components;

                (0..self.components)
                    .map(|j| {
                        let v0 = self.values[start0 + j];
                        let v1 = self.values[start1 + j];
                        v0 + alpha * (v1 - v0)
                    })
                    .collect()
            }
            Interpolation::CubicSpline => {
                // Cubic spline has 3 values per keyframe: in-tangent, value, out-tangent
                let t0 = self.timestamps[i];
                let t1 = self.timestamps[i + 1];
                let dt = t1 - t0;
                let alpha = if dt.abs() < 1e-10 { 0.0 } else { (t - t0) / dt };

                let stride = self.components * 3;
                let base0 = i * stride;
                let base1 = (i + 1) * stride;

                (0..self.components)
                    .map(|j| {
                        let p0 = self.values[base0 + self.components + j]; // value at i
                        let m0 = self.values[base0 + 2 * self.components + j] * dt; // out-tangent at i
                        let p1 = self.values[base1 + self.components + j]; // value at i+1
                        let m1 = self.values[base1 + j] * dt; // in-tangent at i+1

                        // Hermite interpolation
                        let t2 = alpha * alpha;
                        let t3 = t2 * alpha;
                        (2.0 * t3 - 3.0 * t2 + 1.0) * p0
                            + (t3 - 2.0 * t2 + alpha) * m0
                            + (-2.0 * t3 + 3.0 * t2) * p1
                            + (t3 - t2) * m1
                    })
                    .collect()
            }
        }
    }
}

/// A channel connecting a sampler to a specific node property.
#[derive(Debug, Clone)]
pub struct AnimationChannel {
    /// Index into the animation's samplers array.
    pub sampler: usize,
    /// Index of the target node in the scene graph.
    pub node: usize,
    /// Which property of the node is being animated.
    pub path: AnimationPath,
}

/// A complete animation with multiple channels.
#[derive(Debug, Clone, Default)]
pub struct Animation {
    /// Human-readable name.
    pub name: String,
    /// Samplers containing keyframe data.
    pub samplers: Vec<AnimationSampler>,
    /// Channels mapping samplers to node properties.
    pub channels: Vec<AnimationChannel>,
}

impl Animation {
    /// Get the total duration of the animation in seconds.
    pub fn duration(&self) -> f64 {
        self.samplers
            .iter()
            .filter_map(|s| s.timestamps.last().copied())
            .fold(0.0, f64::max)
    }

    /// Sample all channels at time t, returning (node_index, path, values) tuples.
    pub fn sample(&self, t: f64) -> impl Iterator<Item = (usize, AnimationPath, Vec<f64>)> + '_ {
        self.channels.iter().map(move |channel| {
            let values = self.samplers[channel.sampler].sample(t);
            (channel.node, channel.path, values)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let sampler = AnimationSampler {
            timestamps: vec![0.0, 1.0],
            values: vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            components: 3,
            interpolation: Interpolation::Linear,
        };

        let v = sampler.sample(0.5);
        assert!((v[0] - 0.5).abs() < 1e-10);
        assert!((v[1] - 1.0).abs() < 1e-10);
        assert!((v[2] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_step_interpolation() {
        let sampler = AnimationSampler {
            timestamps: vec![0.0, 1.0],
            values: vec![0.0, 10.0],
            components: 1,
            interpolation: Interpolation::Step,
        };

        let v = sampler.sample(0.5);
        assert!((v[0] - 0.0).abs() < 1e-10);

        let v = sampler.sample(1.0);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_animation_duration() {
        let anim = Animation {
            name: "test".to_string(),
            samplers: vec![
                AnimationSampler {
                    timestamps: vec![0.0, 2.0],
                    values: vec![0.0, 1.0],
                    components: 1,
                    interpolation: Interpolation::Linear,
                },
                AnimationSampler {
                    timestamps: vec![0.0, 3.0],
                    values: vec![0.0, 1.0],
                    components: 1,
                    interpolation: Interpolation::Linear,
                },
            ],
            channels: vec![],
        };

        assert!((anim.duration() - 3.0).abs() < 1e-10);
    }
}
