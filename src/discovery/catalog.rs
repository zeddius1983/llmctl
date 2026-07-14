//! Source-aware model catalog normalization and on-disk reconciliation.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::config::ModelLayout;
use crate::domain::Model;

/// One root scanned for models and the layout used to interpret relative paths.
#[derive(Debug, Clone)]
pub struct ModelSource {
    pub name: String,
    pub root: PathBuf,
    pub layout: ModelLayout,
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    schema: u8,
    id: String,
    source: String,
    artifact: Artifact,
    #[serde(default = "available")]
    available: bool,
}

fn available() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize)]
struct Artifact {
    name: String,
    source_path: PathBuf,
    size_bytes: u64,
    modified: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    shards: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mtp_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    projector_path: Option<PathBuf>,
    #[serde(default)]
    has_mtp: bool,
}

/// Assign the logical catalog path for `path` within `source`.
pub fn catalog_path(source: &ModelSource, path: &Path, artifact: &str) -> Vec<String> {
    let relative = path.strip_prefix(&source.root).unwrap_or(path);
    let components = normal_components(relative);
    let layout = detect_layout(source, &components);
    let mut result = vec![sanitize_component(&source.name)];

    match layout {
        ModelLayout::HuggingFace => append_huggingface(&mut result, &components),
        ModelLayout::Flat => {}
        ModelLayout::LmStudio | ModelLayout::Directory | ModelLayout::Auto => {
            result.extend(
                components
                    .iter()
                    .take(components.len().saturating_sub(1))
                    .map(|s| sanitize_component(s)),
            );
        }
    }
    result.push(sanitize_component(artifact.trim_end_matches(".gguf")));
    result
}

/// Resolve Auto once from root-relative components so normalization and
/// duplicate handling cannot disagree about a source's layout.
pub fn resolved_layout(source: &ModelSource, path: &Path) -> ModelLayout {
    let relative = path.strip_prefix(&source.root).unwrap_or(path);
    detect_layout(source, &normal_components(relative))
}

fn normal_components(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect()
}

fn detect_layout(source: &ModelSource, components: &[String]) -> ModelLayout {
    if source.layout != ModelLayout::Auto {
        return source.layout;
    }
    if components.iter().any(|c| c.starts_with("models--")) {
        ModelLayout::HuggingFace
    } else if source.root.ends_with(".lmstudio/models") {
        ModelLayout::LmStudio
    } else {
        ModelLayout::Directory
    }
}

fn append_huggingface(result: &mut Vec<String>, components: &[String]) {
    let Some(model_idx) = components.iter().position(|c| c.starts_with("models--")) else {
        result.extend(
            components
                .iter()
                .take(components.len().saturating_sub(1))
                .map(|s| sanitize_component(s)),
        );
        return;
    };
    let encoded = components[model_idx].trim_start_matches("models--");
    let mut id = encoded.splitn(2, "--");
    if let Some(owner) = id.next().filter(|s| !s.is_empty()) {
        result.push(sanitize_component(owner));
    }
    if let Some(repo) = id.next().filter(|s| !s.is_empty()) {
        result.push(sanitize_component(repo));
    }

    // snapshots/<revision> is storage detail, not logical model identity.
    let tail_start = components[model_idx + 1..]
        .iter()
        .position(|c| c == "snapshots")
        .map(|i| model_idx + i + 3)
        .unwrap_or(model_idx + 1);
    result.extend(
        components[tail_start..]
            .iter()
            .take(components.len().saturating_sub(tail_start + 1))
            .map(|s| sanitize_component(s)),
    );
}

fn sanitize_component(raw: &str) -> String {
    let clean: String = raw.chars().map(|c| if c == '/' || c == '\0' { '_' } else { c }).collect();
    match clean.as_str() {
        "" | "." | ".." => "_".into(),
        _ => clean,
    }
}

/// Stable short hash used only when two source paths normalize to one leaf.
pub fn short_hash(path: &Path) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in path.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")[..8].to_string()
}

/// Materialize the managed catalog without modifying source model files.
pub fn reconcile(root: &Path, models: &mut [Model]) {
    let mut live = HashSet::new();
    for model in models.iter_mut() {
        let leaf = model.catalog_path.iter().fold(root.to_path_buf(), |p, c| p.join(c));
        live.insert(leaf.clone());
        if let Err(err) = fs::create_dir_all(leaf.join("profiles")) {
            warn!(path = %leaf.display(), %err, "failed to create model catalog leaf");
            continue;
        }
        model.catalog_dir = leaf.clone();
        reconcile_link(&leaf.join("model.gguf"), &model.path);

        let manifest = Manifest {
            schema: 1,
            id: model.id.clone(),
            source: model.catalog_path.first().cloned().unwrap_or_default(),
            artifact: Artifact {
                name: model.name.clone(),
                source_path: model.path.clone(),
                size_bytes: model.size_bytes,
                modified: model.modified,
                shards: model.shard_paths.clone(),
                mtp_path: model.mtp_path.clone(),
                projector_path: model.projector_path.clone(),
                has_mtp: model.has_mtp,
            },
            available: true,
        };
        match serde_yaml::to_string(&manifest) {
            Ok(yaml) => write_atomic(&leaf.join(".llmctl.yml"), yaml.as_bytes()),
            Err(err) => warn!(%err, "failed to serialize model manifest"),
        }
    }
    mark_stale_entries(root, &live);
}

fn mark_stale_entries(root: &Path, live: &HashSet<PathBuf>) {
    if !root.exists() {
        return;
    }
    for entry in walkdir::WalkDir::new(root).into_iter().filter_map(Result::ok) {
        if entry.file_name() != ".llmctl.yml" {
            continue;
        }
        let Some(leaf) = entry.path().parent() else { continue };
        if live.contains(leaf) {
            continue;
        }
        let Ok(bytes) = fs::read(entry.path()) else { continue };
        let Ok(mut manifest) = serde_yaml::from_slice::<Manifest>(&bytes) else { continue };
        if manifest.available {
            manifest.available = false;
            if let Ok(yaml) = serde_yaml::to_string(&manifest) {
                write_atomic(entry.path(), yaml.as_bytes());
            }
        }
    }
}

#[cfg(unix)]
fn reconcile_link(link: &Path, target: &Path) {
    if fs::read_link(link).is_ok_and(|current| current == target) {
        return;
    }
    if link.is_symlink() {
        if let Err(err) = fs::remove_file(link) {
            warn!(path = %link.display(), %err, "failed to replace catalog symlink");
            return;
        }
    } else if link.exists() {
        warn!(path = %link.display(), "refusing to replace non-symlink catalog file");
        return;
    }
    if let Err(err) = std::os::unix::fs::symlink(target, link) {
        warn!(path = %link.display(), %err, "failed to create catalog symlink");
    }
}

#[cfg(not(unix))]
fn reconcile_link(_link: &Path, _target: &Path) {}

fn write_atomic(path: &Path, bytes: &[u8]) {
    if fs::read(path).is_ok_and(|existing| existing == bytes) {
        return;
    }
    let tmp = path.with_extension("yml.tmp");
    if let Err(err) = fs::write(&tmp, bytes).and_then(|_| fs::rename(&tmp, path)) {
        warn!(path = %path.display(), %err, "failed to write model manifest");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn source(name: &str, root: &str, layout: ModelLayout) -> ModelSource {
        ModelSource { name: name.into(), root: root.into(), layout }
    }

    #[test]
    fn parses_lmstudio_relative_tree() {
        let src = source("lmstudio", "/home/u/.lmstudio/models", ModelLayout::Auto);
        let path = Path::new(
            "/home/u/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-Q8_0.gguf",
        );
        assert_eq!(
            catalog_path(&src, path, "Qwen3.6-Q8_0.gguf"),
            ["lmstudio", "unsloth", "Qwen3.6-35B-A3B-MTP-GGUF", "Qwen3.6-Q8_0"]
        );
    }

    #[test]
    fn removes_huggingface_snapshot_storage_components() {
        let src = source("huggingface", "/home/u/.cache/huggingface/hub", ModelLayout::Auto);
        let path = Path::new(
            "/home/u/.cache/huggingface/hub/models--unsloth--Qwen3-GGUF/snapshots/abc/Qwen3-Q6_K.gguf",
        );
        assert_eq!(
            catalog_path(&src, path, "Qwen3-Q6_K.gguf"),
            ["huggingface", "unsloth", "Qwen3-GGUF", "Qwen3-Q6_K"]
        );
    }

    #[test]
    fn preserves_custom_relative_layout() {
        let src = source("archive", "/data/models", ModelLayout::Directory);
        let path = Path::new("/data/models/experimental/qwen/Qwen3-Q4.gguf");
        assert_eq!(
            catalog_path(&src, path, "Qwen3-Q4.gguf"),
            ["archive", "experimental", "qwen", "Qwen3-Q4"]
        );
    }

    #[test]
    fn auto_layout_ignores_models_prefix_in_the_source_root() {
        let src = source("mirror", "/data/models--mirror/store", ModelLayout::Auto);
        let path = Path::new("/data/models--mirror/store/team/model.gguf");
        assert_eq!(resolved_layout(&src, path), ModelLayout::Directory);
        assert_eq!(catalog_path(&src, path, "model.gguf"), ["mirror", "team", "model"]);
    }

    #[test]
    fn reconciles_manifest_symlink_and_profile_directory() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-catalog-{nonce}"));
        let source = root.join("source.gguf");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source, b"GGUF").unwrap();
        let mut models = vec![Model {
            id: "local:test".into(),
            name: "Test.gguf".into(),
            path: source.clone(),
            shard_paths: vec![source.clone()],
            mtp_path: None,
            projector_path: None,
            has_mtp: false,
            catalog_path: vec!["local".into(), "Test".into()],
            catalog_dir: PathBuf::new(),
            size_bytes: 4,
            quantization: None,
            architecture: None,
            context_length: None,
            modified: None,
            has_chat_template: false,
            remote: None,
        }];
        let catalog = root.join("catalog");
        reconcile(&catalog, &mut models);
        let leaf = catalog.join("local/Test");
        assert_eq!(fs::read_link(leaf.join("model.gguf")).unwrap(), source);
        assert!(leaf.join(".llmctl.yml").is_file());
        assert!(leaf.join("profiles").is_dir());
        fs::remove_dir_all(root).unwrap();
    }
}
