//! Discovery of runtimes and models from the local system.

pub mod catalog;
pub mod gguf;
pub mod models;
pub mod runtimes;

pub use catalog::{ModelSource, reconcile};
pub use models::scan as scan_models;
pub use runtimes::{discover_llama_cpp, discover_vllm};
