//! Triangle-level computations for mesh processing.

pub mod barycentric;
pub mod bvh;
pub mod closest;
pub mod inertia;

pub use inertia::{MassProperties, mass_properties, volume};
