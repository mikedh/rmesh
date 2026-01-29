//! Graph and topology operations for mesh connectivity.
//!
//! This module provides functions for computing face adjacency,
//! edge extraction, and other topological properties of meshes.

pub mod adjacency;

pub use adjacency::{EdgeGroups, ManifoldStatus, SortedEdge};
