//! Backends for feature model meshing
//!
//! This module contains implementations of the [`FeatureBackend`] trait that
//! convert feature models to meshes using different approaches.
//!
//! # Available Backends
//!
//! - **Fidget**: Uses signed distance fields (SDFs) and Manifold Dual Contouring
//!   for high-quality mesh generation.
//!
//! # Example
//!
//! ```ignore
//! use rmesh::creation::feature::{FeatureModel, Extrude, Sketch};
//! use rmesh::creation::feature::backends::FidgetBackend;
//!
//! let model = FeatureModel::new()
//!     .with_operation(Extrude::simple(Sketch::rectangle(1.0, 1.0), 1.0));
//!
//! let backend = FidgetBackend::default();
//! let mesh = backend.execute(&model, &Default::default())?;
//! ```

pub mod fidget;

pub use self::fidget::{FidgetBackend, FidgetSettings};
