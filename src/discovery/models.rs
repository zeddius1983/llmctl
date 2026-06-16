//! GGUF model discovery: recursive scan of configured directories with a
//! mtime/size-keyed cache so re-scans (and the `F5` refresh) are cheap.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use tracing::{debug, warn};
use walkdir::WalkDir;

use crate::domain::Model;

use super::gguf;

/// On-disk scan cache: keyed by absolute path string.
#[derive(Default, Serialize, Deserialize)]
struct Cache {
    models: HashMap<String, Model>,
}

/// Scan every configured directory recursively for GGUF models.
///
/// Results are de-duplicated by first shard, sorted by name, and cached to
/// `cache_path`. Unchanged files (same size + mtime) reuse cached metadata
/// instead of re-parsing the GGUF header.
pub fn scan(paths: &[PathBuf], cache_path: &Path) -> Vec<Model> {
    let cache = load_cache(cache_path);
    let mut fresh: HashMap<String, Model> = HashMap::new();
    let mut models: Vec<Model> = Vec::new();

    for base in paths {
        if !base.exists() {
            debug!(path = %base.display(), "configured model path does not exist");
            continue;
        }
        for entry in WalkDir::new(base).follow_links(true).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if !entry.file_type().is_file()
                || !is_gguf(path)
                || !is_first_shard(path)
                || is_projector(path)
            {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            // For multi-part models, sum every shard's size, not just the first.
            let size = aggregate_size(path, meta.len());
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs());

            let key = path.to_string_lossy().into_owned();
            let model = match cache.models.get(&key) {
                Some(c) if c.size_bytes == size && c.modified == modified => c.clone(),
                _ => build_model(path, size, modified),
            };
            fresh.insert(key, model.clone());
            models.push(model);
        }
    }

    save_cache(cache_path, &Cache { models: fresh });
    models.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    models
}

/// Read a model's metadata from its GGUF header, falling back to the filename
/// for quantization when the header doesn't carry a usable file-type.
fn build_model(path: &Path, size: u64, modified: Option<u64>) -> Model {
    let raw_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let name = display_name(&raw_name);
    let info = match gguf::read_gguf_info(path) {
        Ok(info) => Some(info),
        Err(err) => {
            warn!(path = %path.display(), %err, "failed to read GGUF header");
            None
        }
    };

    // Prefer the filename label — it captures exact variants (e.g. Unsloth
    // `Q4_K_XL`, `MXFP4`) that the header's coarse `file_type` enum misses.
    let quantization =
        quant_from_filename(&name).or_else(|| info.as_ref().and_then(|i| i.file_type_label.clone()));

    Model {
        name,
        path: path.to_path_buf(),
        size_bytes: size,
        quantization,
        architecture: info.as_ref().and_then(|i| i.architecture.clone()),
        context_length: info.as_ref().and_then(|i| i.context_length),
        modified,
        has_chat_template: info.as_ref().map(|i| i.has_chat_template).unwrap_or(false),
    }
}

/// Total on-disk size of a model. For a first shard (`-00001-of-000NN.gguf`),
/// sum every shard in the same directory; otherwise return the single size.
fn aggregate_size(path: &Path, first_size: u64) -> u64 {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?i)-(\d{5})-of-(\d{5})\.gguf$").unwrap());

    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let Some(caps) = re.captures(&name) else { return first_size };
    let total: u32 = caps[2].parse().unwrap_or(1);
    let idx = caps.get(1).unwrap();
    let (head, tail) = (&name[..idx.start()], &name[idx.end()..]);
    let dir = path.parent().unwrap_or(Path::new("."));

    let mut sum = 0u64;
    for i in 1..=total {
        let shard = dir.join(format!("{head}{i:05}{tail}"));
        if let Ok(meta) = std::fs::metadata(&shard) {
            sum += meta.len();
        }
    }
    if sum == 0 { first_size } else { sum }
}

/// Drop the `-00001-of-00005` shard suffix from a multi-part model's display
/// name (the on-disk path is kept intact for launching).
fn display_name(file: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?i)-\d{5}-of-\d{5}(\.gguf)$").unwrap());
    re.replace(file, "$1").into_owned()
}

fn is_gguf(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

/// Multi-part GGUF files are named `...-00001-of-00005.gguf`; show only the
/// first shard so a split model appears once.
fn is_first_shard(path: &Path) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?i)-(\d{5})-of-\d{5}\.gguf$").unwrap());
    let name = path.to_string_lossy();
    match re.captures(&name) {
        Some(caps) => &caps[1] == "00001",
        None => true,
    }
}

/// Projector/companion files (`mmproj-*.gguf`) are not standalone models.
fn is_projector(path: &Path) -> bool {
    path.file_name()
        .map(|n| n.to_string_lossy().to_lowercase().starts_with("mmproj"))
        .unwrap_or(false)
}

/// Heuristic quantization label from the filename (e.g. `Q4_K_XL`, `IQ3_XXS`,
/// `MXFP4`). Longer/more-specific patterns are matched first.
fn quant_from_filename(name: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(IQ\d+_[A-Z0-9]+|Q\d+_K(?:_[A-Z]+)?|Q\d+_\d|Q\d+|MXFP\d+|BF16|FP16|FP8|F16|F32)\b",
        )
        .unwrap()
    });
    re.find(name).map(|m| m.as_str().to_uppercase())
}

fn load_cache(path: &Path) -> Cache {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(_) => Cache::default(),
    }
}

fn save_cache(path: &Path, cache: &Cache) {
    match serde_json::to_vec(cache) {
        Ok(bytes) => {
            if let Err(err) = std::fs::write(path, bytes) {
                warn!(path = %path.display(), %err, "failed to write model cache");
            }
        }
        Err(err) => warn!(%err, "failed to serialize model cache"),
    }
}
