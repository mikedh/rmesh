//! Light types for scene illumination.

/// The type of light source.
#[derive(Debug, Clone, Default)]
pub enum LightType {
    /// Point light radiating in all directions.
    #[default]
    Point,
    /// Directional light (like the sun).
    Directional,
    /// Spot light with inner and outer cone angles (in radians).
    Spot {
        /// Inner cone angle where light is at full intensity.
        inner: f64,
        /// Outer cone angle where light fades to zero.
        outer: f64,
    },
    /// Ambient light affecting all surfaces equally.
    Ambient,
}

/// A light source in the scene.
#[derive(Debug, Clone)]
pub struct Light {
    /// Human-readable name.
    pub name: String,
    /// The type of light.
    pub light_type: LightType,
    /// RGB color of the light (linear, not sRGB).
    pub color: [f64; 3],
    /// Light intensity in candelas (point/spot) or lux (directional).
    pub intensity: f64,
    /// Maximum range of the light. None means infinite.
    pub range: Option<f64>,
}

impl Default for Light {
    fn default() -> Self {
        Light {
            name: String::new(),
            light_type: LightType::Point,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: None,
        }
    }
}

impl Light {
    /// Create a point light with the given color and intensity.
    pub fn point(color: [f64; 3], intensity: f64) -> Self {
        Light {
            name: String::new(),
            light_type: LightType::Point,
            color,
            intensity,
            range: None,
        }
    }

    /// Create a directional light with the given color and intensity.
    pub fn directional(color: [f64; 3], intensity: f64) -> Self {
        Light {
            name: String::new(),
            light_type: LightType::Directional,
            color,
            intensity,
            range: None,
        }
    }

    /// Create a spot light with the given parameters.
    pub fn spot(color: [f64; 3], intensity: f64, inner: f64, outer: f64) -> Self {
        Light {
            name: String::new(),
            light_type: LightType::Spot { inner, outer },
            color,
            intensity,
            range: None,
        }
    }

    /// Create an ambient light with the given color and intensity.
    pub fn ambient(color: [f64; 3], intensity: f64) -> Self {
        Light {
            name: String::new(),
            light_type: LightType::Ambient,
            color,
            intensity,
            range: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_creation() {
        let point = Light::point([1.0, 0.5, 0.0], 100.0);
        assert!(matches!(point.light_type, LightType::Point));
        assert_eq!(point.intensity, 100.0);

        let spot = Light::spot([1.0, 1.0, 1.0], 50.0, 0.3, 0.5);
        if let LightType::Spot { inner, outer } = spot.light_type {
            assert!((inner - 0.3).abs() < 1e-10);
            assert!((outer - 0.5).abs() < 1e-10);
        } else {
            panic!("Expected spot light");
        }
    }
}
