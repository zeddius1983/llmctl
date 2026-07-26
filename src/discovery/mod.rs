//! Discovery of models from the local system. Runtime discovery lives with
//! each backend in `crate::runtime`.

pub mod catalog;
pub mod gguf;
pub mod hf;
pub mod models;
pub mod online;

pub use catalog::{ModelSource, reconcile};
pub use models::scan as scan_models;
