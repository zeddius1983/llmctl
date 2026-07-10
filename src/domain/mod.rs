//! Core domain types shared across the app. Pure data, no I/O.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Stable identity for an inference backend. Display names are deliberately
/// separate so persistence and dispatch never depend on user-facing strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeId {
    LlamaCpp,
    Vllm,
}

impl RuntimeId {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "llama.cpp" => Some(Self::LlamaCpp),
            "vLLM" => Some(Self::Vllm),
            _ => None,
        }
    }
}

/// An installed (or configured but unavailable) inference backend.
#[derive(Debug, Clone)]
pub struct Runtime {
    pub id: RuntimeId,
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
    /// Stable catalog identity, distinct from the display filename.
    #[serde(default)]
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    /// All physical shards (one entry for a non-split model).
    #[serde(default)]
    pub shard_paths: Vec<PathBuf>,
    /// Source/provider/repository/artifact path shown by the model browser.
    #[serde(default)]
    pub catalog_path: Vec<String>,
    /// Managed catalog leaf containing the manifest, symlink, and profiles.
    #[serde(default)]
    pub catalog_dir: PathBuf,
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

impl Model {
    /// Synthetic catalog directories have no launchable source path.
    pub fn is_catalog_dir(&self) -> bool {
        self.path.as_os_str().is_empty()
    }

    pub fn is_model(&self) -> bool {
        !self.is_catalog_dir()
    }

    pub fn display_label(&self) -> &str {
        self.catalog_path.last().map(String::as_str).unwrap_or(&self.name)
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
    if unit == 0 { format!("{bytes} B") } else { format!("{size:.1} {}", UNITS[unit]) }
}
