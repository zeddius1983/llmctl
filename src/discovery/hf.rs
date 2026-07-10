//! Local Hugging Face model discovery for vLLM.
//!
//! A model is a directory containing `config.json` and at least one supported
//! weight file. Hugging Face cache snapshots are normalized to owner/repository
//! identity; generic configured directories preserve their relative path.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, warn};
use walkdir::WalkDir;

use crate::domain::Model;

use super::catalog::{self, ModelSource};

#[derive(Default, Serialize, Deserialize)]
struct Cache {
    models: HashMap<String, CacheEntry>,
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    config_size: u64,
    config_modified: Option<u64>,
    model: Model,
}

pub fn scan(sources: &[ModelSource], cache_path: &Path) -> Vec<Model> {
    let cache = load_cache(cache_path);
    let mut fresh = HashMap::new();
    let mut by_catalog: HashMap<Vec<String>, Model> = HashMap::new();

    for source in sources {
        if !source.root.exists() {
            continue;
        }
        for entry in
            WalkDir::new(&source.root).follow_links(true).into_iter().filter_map(Result::ok)
        {
            if !entry.file_type().is_file() || entry.file_name() != "config.json" {
                continue;
            }
            let config_path = entry.path();
            let Some(model_dir) = config_path.parent() else { continue };
            let Some((weight_size, modified)) = weight_metadata(model_dir) else { continue };
            let Ok(meta) = entry.metadata() else { continue };
            let config_modified = modified_time(&meta);
            let key = config_path.to_string_lossy().into_owned();
            let mut model = match cache.models.get(&key) {
                Some(c) if c.config_size == meta.len() && c.config_modified == config_modified => {
                    let mut model = c.model.clone();
                    model.size_bytes = weight_size;
                    model.modified = modified;
                    model
                }
                _ => match build_model(source, model_dir, config_path, weight_size, modified) {
                    Some(model) => model,
                    None => continue,
                },
            };
            model.catalog_path = catalog_path(source, model_dir);
            model.catalog_dir = PathBuf::new();

            fresh.insert(
                key,
                CacheEntry { config_size: meta.len(), config_modified, model: model.clone() },
            );
            match by_catalog.get(&model.catalog_path) {
                Some(current)
                    if !super::models::prefer_huggingface_candidate(
                        source,
                        config_path,
                        modified,
                        &current.path.join("config.json"),
                        current.modified,
                    ) => {}
                _ => {
                    by_catalog.insert(model.catalog_path.clone(), model);
                }
            }
        }
    }

    save_cache(cache_path, &Cache { models: fresh });
    let mut models: Vec<Model> = by_catalog.into_values().collect();
    models.sort_by(|a, b| a.catalog_path.cmp(&b.catalog_path));
    models
}

fn build_model(
    source: &ModelSource,
    model_dir: &Path,
    config_path: &Path,
    size_bytes: u64,
    modified: Option<u64>,
) -> Option<Model> {
    let bytes = std::fs::read(config_path).ok()?;
    let config: Value = match serde_json::from_slice(&bytes) {
        Ok(config) => config,
        Err(err) => {
            warn!(path = %config_path.display(), %err, "ignoring invalid Hugging Face config");
            return None;
        }
    };
    let catalog_path = catalog_path(source, model_dir);
    let name = huggingface_id(model_dir).unwrap_or_else(|| {
        model_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
    });
    let architecture = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .map(str::to_string);
    let context_length = ["max_position_embeddings", "model_max_length", "n_positions"]
        .into_iter()
        .find_map(|key| config.get(key).and_then(Value::as_u64));
    let quantization = config
        .get("quantization_config")
        .and_then(|q| q.get("quant_method").or_else(|| q.get("quantization_method")))
        .and_then(Value::as_str)
        .map(|s| s.to_uppercase())
        .or_else(|| config.get("torch_dtype").and_then(Value::as_str).map(str::to_uppercase));
    let has_chat_template = std::fs::read(model_dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| value.get("chat_template").cloned())
        .is_some_and(|value| !value.is_null());

    Some(Model {
        id: format!("vllm:{}:{}", source.name, catalog::short_hash(model_dir)),
        name,
        path: model_dir.to_path_buf(),
        shard_paths: weight_paths(model_dir),
        catalog_path,
        catalog_dir: PathBuf::new(),
        size_bytes,
        quantization,
        architecture,
        context_length,
        modified,
        has_chat_template,
    })
}

fn catalog_path(source: &ModelSource, model_dir: &Path) -> Vec<String> {
    let relative = model_dir.strip_prefix(&source.root).unwrap_or(model_dir);
    let parts: Vec<String> = relative
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    let mut result = vec![source.name.clone()];
    if let Some(index) = parts.iter().position(|part| part.starts_with("models--")) {
        let encoded = parts[index].trim_start_matches("models--");
        result.extend(encoded.split("--").filter(|s| !s.is_empty()).map(str::to_string));
    } else {
        result.extend(parts);
    }
    result
}

fn huggingface_id(path: &Path) -> Option<String> {
    let encoded = path
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s.to_string_lossy()),
            _ => None,
        })
        .find(|part| part.starts_with("models--"))?;
    let id = encoded.trim_start_matches("models--").replace("--", "/");
    (!id.is_empty()).then_some(id)
}

fn weight_metadata(dir: &Path) -> Option<(u64, Option<u64>)> {
    let paths = weight_paths(dir);
    if paths.is_empty() {
        return None;
    }
    let mut size = 0u64;
    let mut modified = None;
    for path in paths {
        let Ok(meta) = std::fs::metadata(path) else { continue };
        size = size.saturating_add(meta.len());
        modified = modified.max(modified_time(&meta));
    }
    Some((size, modified))
}

fn weight_paths(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "safetensors")
                || path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|name| name.starts_with("pytorch_model") && name.ends_with(".bin"))
        })
        .collect();
    paths.sort();
    paths
}

fn modified_time(meta: &std::fs::Metadata) -> Option<u64> {
    meta.modified().ok()?.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

fn load_cache(path: &Path) -> Cache {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn save_cache(path: &Path, cache: &Cache) {
    match serde_json::to_vec(cache) {
        Ok(bytes) => {
            if let Err(err) = std::fs::write(path, bytes) {
                debug!(path = %path.display(), %err, "could not save vLLM model cache");
            }
        }
        Err(err) => debug!(%err, "could not serialize vLLM model cache"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelLayout;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn scans_local_model_metadata_and_ignores_config_only_directories() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-hf-{nonce}"));
        let model = root.join("team/model");
        let ignored = root.join("not-a-model");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::create_dir_all(&ignored).unwrap();
        std::fs::write(
            model.join("config.json"),
            r#"{"architectures":["QwenForCausalLM"],"max_position_embeddings":32768,"quantization_config":{"quant_method":"awq"}}"#,
        )
        .unwrap();
        std::fs::write(model.join("model-00001-of-00002.safetensors"), vec![0; 11]).unwrap();
        std::fs::write(model.join("model-00002-of-00002.safetensors"), vec![0; 13]).unwrap();
        std::fs::write(model.join("tokenizer_config.json"), r#"{"chat_template":"{{ x }}"}"#)
            .unwrap();
        std::fs::write(ignored.join("config.json"), "{}").unwrap();
        let source = ModelSource {
            name: "models".into(),
            root: root.clone(),
            layout: ModelLayout::Directory,
        };

        let models = scan(&[source], &root.join("cache.json"));
        assert_eq!(models.len(), 1);
        let found = &models[0];
        assert_eq!(found.name, "model");
        assert_eq!(found.catalog_path, ["models", "team", "model"]);
        assert_eq!(found.size_bytes, 24);
        assert_eq!(found.architecture.as_deref(), Some("QwenForCausalLM"));
        assert_eq!(found.context_length, Some(32768));
        assert_eq!(found.quantization.as_deref(), Some("AWQ"));
        assert!(found.has_chat_template);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn huggingface_main_snapshot_wins_and_storage_path_is_hidden() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-hf-main-{nonce}"));
        let repo = root.join("models--acme--demo");
        let main = repo.join("snapshots/main-revision");
        let other = repo.join("snapshots/other-revision");
        std::fs::create_dir_all(&main).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs/main"), "main-revision\n").unwrap();
        for (dir, architecture) in [(&main, "MainModel"), (&other, "OtherModel")] {
            std::fs::write(
                dir.join("config.json"),
                format!(r#"{{"architectures":["{architecture}"]}}"#),
            )
            .unwrap();
            std::fs::write(dir.join("model.safetensors"), [0]).unwrap();
        }
        let source = ModelSource {
            name: "huggingface".into(),
            root: root.clone(),
            layout: ModelLayout::HuggingFace,
        };

        let models = scan(&[source], &root.join("cache.json"));
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "acme/demo");
        assert_eq!(models[0].catalog_path, ["huggingface", "acme", "demo"]);
        assert_eq!(models[0].architecture.as_deref(), Some("MainModel"));
        assert_eq!(models[0].path, main);
        std::fs::remove_dir_all(root).unwrap();
    }
}
