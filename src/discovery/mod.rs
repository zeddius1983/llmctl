//! Discovery of runtimes and models from the local system.

pub mod gguf;
pub mod models;
pub mod runtimes;

pub use models::scan as scan_models;
pub use runtimes::discover_llama_cpp;
