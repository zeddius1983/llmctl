//! Core domain types shared across the app. Pure data, no I/O.
//!
//! Phase 0 populates these with static stub data so the panes render. Phases
//! 1–2 replace the stubs with real discovery and profile/option stores.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// An inference backend (MVP: only llama.cpp).
#[derive(Debug, Clone)]
pub struct Runtime {
    pub name: String,
    #[allow(dead_code)] // shown in the runtime detail view (Phase 1)
    pub description: String,
    pub version: Option<String>,
    pub binary_path: Option<PathBuf>,
    pub formats: Vec<String>,
}

/// A discovered GGUF model. Serializable so the scanner can cache results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub quantization: Option<String>,
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    /// Last-modified time, seconds since the Unix epoch (cache invalidation).
    pub modified: Option<u64>,
    pub has_chat_template: bool,
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
    /// Human-readable allowed range, e.g. "0.0 – 2.0" (None for free-form).
    pub range: Option<String>,
    pub cli: String,
    pub description: String,
}

impl Runtime {
    /// Human-readable size, e.g. "23.8 GB".
    pub fn formats_label(&self) -> String {
        self.formats.join(", ")
    }
}

/// Format a Unix timestamp (seconds) as `YYYY-MM-DD` (UTC).
pub fn format_unix_date(secs: u64) -> String {
    // days since epoch → civil date (Howard Hinnant's algorithm).
    let z = (secs / 86_400) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}")
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

    /// The vLLM runtime — a stub for now (real support is a future phase), kept
    /// so multi-runtime navigation is exercisable alongside discovered llama.cpp.
    pub fn vllm_runtime() -> Runtime {
        Runtime {
            name: "vLLM".into(),
            description: "High-throughput serving with PagedAttention".into(),
            version: None,
            binary_path: None,
            formats: vec!["Safetensors".into(), "HF".into()],
        }
    }

    /// Profiles available for the given model.
    pub fn profiles_for(_model: &Model) -> Vec<Profile> {
        profiles()
    }

    /// Options resolved for the given profile.
    pub fn options_for(_profile: &Profile) -> Vec<OptionItem> {
        options()
    }

    /// Stub models for the vLLM runtime (HF/safetensors style names).
    pub fn vllm_models() -> Vec<Model> {
        let model = |name: &str, path: &str, size: u64, quant: &str, arch: &str| Model {
            name: name.into(),
            path: path.into(),
            size_bytes: size,
            quantization: Some(quant.into()),
            architecture: Some(arch.into()),
            context_length: None,
            modified: None,
            has_chat_template: true,
        };
        vec![
            model(
                "meta-llama/Llama-3.1-8B-Instruct",
                "/models/hf/Llama-3.1-8B-Instruct",
                16_100_000_000,
                "FP16",
                "llama",
            ),
            model(
                "Qwen/Qwen2.5-32B-Instruct-AWQ",
                "/models/hf/Qwen2.5-32B-Instruct-AWQ",
                19_400_000_000,
                "AWQ",
                "qwen2",
            ),
            model(
                "mistralai/Mistral-7B-Instruct-v0.3",
                "/models/hf/Mistral-7B-Instruct-v0.3",
                14_500_000_000,
                "FP16",
                "mistral",
            ),
        ]
    }

    fn profiles() -> Vec<Profile> {
        ["Default", "Chat", "Coding", "Long Context", "Server"]
            .into_iter()
            .map(|name| Profile { name: name.into(), builtin: true, favorite: false })
            .collect()
    }

    fn options() -> Vec<OptionItem> {
        let opt = |key: &str, value: &str, default: &str, cli: &str, desc: &str| OptionItem {
            key: key.into(),
            value: value.into(),
            default: default.into(),
            range: None,
            cli: cli.into(),
            description: desc.into(),
        };
        vec![
            opt("ctx-size", "32768", "4096", "--ctx-size", "Maximum context window size."),
            opt("gpu-layers", "999", "0", "-ngl", "Number of layers offloaded to the GPU."),
            opt("temperature", "0.7", "0.8", "--temp", "Sampling temperature."),
            opt("top-p", "0.95", "0.95", "--top-p", "Nucleus sampling probability."),
            opt("flash-attn", "true", "false", "--flash-attn", "Enable flash attention."),
        ]
    }
}
