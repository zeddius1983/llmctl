//! Core domain types shared across the app. Pure data, no I/O.
//!
//! Phase 0 populates these with static stub data so the panes render. Phases
//! 1–2 replace the stubs with real discovery and profile/option stores.

use std::path::PathBuf;

/// An inference backend (MVP: only llama.cpp).
#[derive(Debug, Clone)]
pub struct Runtime {
    pub name: String,
    pub description: String,
    pub version: Option<String>,
    pub binary_path: Option<PathBuf>,
    pub formats: Vec<String>,
}

/// A discovered GGUF model.
#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub quantization: Option<String>,
    pub architecture: Option<String>,
}

/// A reusable launch configuration.
///
/// Built-ins are global, read-only templates; editing options forks a
/// model-scoped instance (see plan: profile scoping).
#[derive(Debug, Clone)]
pub struct Profile {
    pub name: String,
    /// Built-ins are read-only templates; editing forks a model-scoped instance.
    #[allow(dead_code)] // enforced in Phase 2
    pub builtin: bool,
    pub favorite: bool,
}

/// One editable launch option, with the metadata shown in the Info pane.
#[derive(Debug, Clone)]
pub struct OptionItem {
    pub key: String,
    pub value: String,
    pub default: String,
    pub cli: String,
    pub description: String,
}

impl Runtime {
    /// Human-readable size, e.g. "23.8 GB".
    pub fn formats_label(&self) -> String {
        self.formats.join(", ")
    }
}

/// Format a byte count as a short human string (e.g. "12.3 GB").
pub fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

/// Static stub data for Phase 0 so the UI has something to render.
///
/// Child builders take their parent so the cascading dependency is real even
/// though the stub data doesn't yet vary by parent. Phases 1–2 swap these for
/// discovery (`models_for` scans the runtime's paths) and the profile store.
pub mod stubs {
    use super::*;

    pub fn runtimes() -> Vec<Runtime> {
        vec![Runtime {
            name: "llama.cpp".into(),
            description: "GGUF inference via llama-server".into(),
            version: None,
            binary_path: None,
            formats: vec!["GGUF".into()],
        }]
    }

    /// Models available for the given runtime.
    pub fn models_for(_runtime: &Runtime) -> Vec<Model> {
        models()
    }

    /// Profiles available for the given model.
    pub fn profiles_for(_model: &Model) -> Vec<Profile> {
        profiles()
    }

    /// Options resolved for the given profile.
    pub fn options_for(_profile: &Profile) -> Vec<OptionItem> {
        options()
    }

    fn models() -> Vec<Model> {
        vec![
            Model {
                name: "Qwen3-32B-Q6_K.gguf".into(),
                path: "/models/Qwen3-32B-Q6_K.gguf".into(),
                size_bytes: 27_000_000_000,
                quantization: Some("Q6_K".into()),
                architecture: Some("qwen3".into()),
            },
            Model {
                name: "Gemma-27B-Q4_K_M.gguf".into(),
                path: "/models/Gemma-27B-Q4_K_M.gguf".into(),
                size_bytes: 16_000_000_000,
                quantization: Some("Q4_K_M".into()),
                architecture: Some("gemma2".into()),
            },
            Model {
                name: "GPT-OSS-20B-Q8.gguf".into(),
                path: "/models/GPT-OSS-20B-Q8.gguf".into(),
                size_bytes: 21_000_000_000,
                quantization: Some("Q8_0".into()),
                architecture: Some("gptoss".into()),
            },
        ]
    }

    fn profiles() -> Vec<Profile> {
        ["Default", "Chat", "Coding", "Long Context", "Server"]
            .into_iter()
            .map(|name| Profile { name: name.into(), builtin: true, favorite: false })
            .collect()
    }

    fn options() -> Vec<OptionItem> {
        vec![
            OptionItem {
                key: "ctx-size".into(),
                value: "32768".into(),
                default: "4096".into(),
                cli: "--ctx-size".into(),
                description: "Maximum context window size.".into(),
            },
            OptionItem {
                key: "gpu-layers".into(),
                value: "999".into(),
                default: "0".into(),
                cli: "-ngl".into(),
                description: "Number of layers offloaded to the GPU.".into(),
            },
            OptionItem {
                key: "temperature".into(),
                value: "0.7".into(),
                default: "0.8".into(),
                cli: "--temp".into(),
                description: "Sampling temperature.".into(),
            },
            OptionItem {
                key: "top-p".into(),
                value: "0.95".into(),
                default: "0.95".into(),
                cli: "--top-p".into(),
                description: "Nucleus sampling probability.".into(),
            },
            OptionItem {
                key: "flash-attn".into(),
                value: "true".into(),
                default: "false".into(),
                cli: "--flash-attn".into(),
                description: "Enable flash attention.".into(),
            },
        ]
    }
}
