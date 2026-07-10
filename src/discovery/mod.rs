//! Discovery of runtimes and models from the local system.

pub mod catalog;
pub mod gguf;
pub mod hf;
pub mod models;
pub mod runtimes;

pub use catalog::{ModelSource, reconcile};
pub use hf::scan as scan_vllm_models;
pub use models::scan as scan_models;
pub use runtimes::{discover_llama_cpp, discover_vllm};
