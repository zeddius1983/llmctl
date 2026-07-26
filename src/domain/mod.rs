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
    pub bench_path: Option<PathBuf>,
    pub formats: Vec<String>,
    /// Device identifiers reported by the runtime (for example ROCm0 or Vulkan0).
    pub devices: Vec<String>,
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
    /// A same-directory `mtp-*.gguf` sidecar paired with this base model.
    /// llama.cpp loads it as a speculative draft model, never as a standalone
    /// model.
    #[serde(default)]
    pub mtp_path: Option<PathBuf>,
    /// A compatible multimodal projector sidecar. The projector encodes image
    /// or audio inputs for the base model and is never launched by itself.
    #[serde(default)]
    pub projector_path: Option<PathBuf>,
    /// Whether the base GGUF itself contains Multi-Token Prediction heads.
    #[serde(default)]
    pub has_mtp: bool,
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
    /// Hugging Face identity for a lazily-discovered online entry. A missing
    /// `file` represents a repository directory; a file makes this a
    /// launchable remote GGUF leaf even before it has a local cache path.
    #[serde(default)]
    pub remote: Option<RemoteModel>,
    /// FastFlowLM catalog identity. Present for every entry the `flm` catalog
    /// knows about, installed or not.
    #[serde(default)]
    pub flm: Option<FlmModel>,
    /// Which runtime owns this model. Determines the profile-store namespace,
    /// so profiles never leak between runtimes.
    #[serde(default = "default_runtime")]
    pub runtime: String,
}

fn default_runtime() -> String {
    crate::runtime::llama_cpp::NAME.to_string()
}

fn main_revision() -> String {
    "main".to_string()
}

/// A model in FastFlowLM's curated NPU catalog, as reported by `flm list`.
///
/// Unlike a GGUF file, this is a tag in a fixed catalog: it exists (and is
/// browsable, with full metadata) whether or not it has been downloaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlmModel {
    /// The `name:size` tag — the model's identity everywhere in `flm`.
    pub tag: String,
    /// Whether the model files are present locally.
    pub installed: bool,
    /// Hugging Face repository name, which is also the on-disk directory name
    /// under `flm`'s model root.
    pub repo: String,
    /// Repository revision this model is pinned to. Several FastFlowLM models
    /// live on a tag rather than `main`, so downloading the wrong revision
    /// yields weights the installed `flm` cannot load.
    #[serde(default = "main_revision")]
    pub revision: String,
    /// The files a model directory must contain — exactly what llmctl
    /// downloads. Deliberately narrower than the repository, which also carries
    /// a README and the `.xclbin` NPU kernels that ship with `flm` itself.
    #[serde(default)]
    pub files: Vec<String>,
    /// Capability labels (`vision`, `reasoning`, `tool-calling`, …). Drives the
    /// browser's grouping; may be empty.
    pub labels: Vec<String>,
    /// Accepts image input, so `--img-pre-resize` is meaningful.
    pub vlm: bool,
    /// Provides speech recognition, so `--asr` is meaningful.
    pub asr: bool,
    /// Upper bound for `--prefill-chunk-len`.
    pub max_prefill_len: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteModel {
    pub repo: String,
    pub revision: Option<String>,
    pub file: Option<String>,
    /// Hugging Face LFS blobs comprising this artifact (one or more GGUF
    /// shards), used to observe llama.cpp's native download progress.
    #[serde(default)]
    pub blobs: Vec<RemoteBlob>,
    /// Repository-relative companion selected for speculative MTP decoding.
    #[serde(default)]
    pub mtp_file: Option<String>,
    /// Repository-relative multimodal projector selected for this artifact.
    #[serde(default)]
    pub projector_file: Option<String>,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default)]
    pub gated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteBlob {
    pub oid: String,
    pub size_bytes: u64,
    /// Repository-relative filename for this blob. Required when llmctl
    /// downloads split GGUF artifacts directly into the Hub cache.
    #[serde(default)]
    pub file: String,
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
    ///
    /// A not-yet-downloaded model has no local path either, so the remote and
    /// FastFlowLM identities are what distinguish "a real model you don't have
    /// yet" from "a folder".
    pub fn is_catalog_dir(&self) -> bool {
        self.path.as_os_str().is_empty()
            && self.remote.as_ref().and_then(|remote| remote.file.as_ref()).is_none()
            && self.flm.is_none()
    }

    pub fn is_model(&self) -> bool {
        !self.is_catalog_dir()
    }

    /// Stable persistence identity used before and after an online model is
    /// downloaded into the Hugging Face cache.
    /// For FastFlowLM the key is the tag, which is deliberately independent of
    /// `catalog_path`: the same model is rendered under every capability label
    /// it carries, and all of those views must share one set of profiles.
    pub fn profile_key(&self) -> String {
        if let Some(flm) = &self.flm {
            return format!("flm:{}", flm.tag);
        }
        match &self.remote {
            Some(remote) => match &remote.file {
                Some(file) => format!("hf:{}/{}", remote.repo, file),
                None => format!("hf:{}", remote.repo),
            },
            None => self.path.to_string_lossy().into_owned(),
        }
    }

    pub fn display_label(&self) -> &str {
        self.catalog_path.last().map(String::as_str).unwrap_or(&self.name)
    }

    pub fn supports_mtp(&self) -> bool {
        self.has_mtp
            || self.mtp_path.is_some()
            || self.remote.as_ref().is_some_and(|remote| remote.mtp_file.is_some())
    }

    pub fn supports_multimodal(&self) -> bool {
        self.projector_path.is_some()
            || self.remote.as_ref().is_some_and(|remote| remote.projector_file.is_some())
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
