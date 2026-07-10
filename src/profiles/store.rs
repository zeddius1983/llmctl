//! Persistent, model-scoped profile instances.
//!
//! A profile is identified by (runtime, model, profile-name). Built-in
//! templates are global and read-only; the first time the user edits an option
//! (or favorites/creates a profile) for a given model, an *instance* is
//! materialized here and auto-saved. Stored as JSON under the state dir.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::domain::{Model, RuntimeId};

/// (runtime, model, profile) — uniquely identifies an instance.
type Key = (String, String, String);

/// A materialized profile instance: explicit option values plus flags.
#[derive(Debug, Clone, Default)]
pub struct Instance {
    pub values: BTreeMap<String, String>,
    pub favorite: bool,
    /// True for user-created profiles (not backed by a built-in template).
    pub custom: bool,
}

/// Flat record used for (de)serialization (JSON map keys can't be tuples).
#[derive(Serialize, Deserialize)]
struct Record {
    runtime: String,
    model: String,
    profile: String,
    #[serde(default)]
    values: BTreeMap<String, String>,
    #[serde(default)]
    favorite: bool,
    #[serde(default)]
    custom: bool,
}

#[derive(Serialize, Deserialize, Default)]
struct StoreFile {
    instances: Vec<Record>,
}

#[derive(Serialize, Deserialize)]
struct ProfileFile {
    schema: u8,
    name: String,
    #[serde(default)]
    values: BTreeMap<String, String>,
    #[serde(default)]
    favorite: bool,
    #[serde(default)]
    custom: bool,
}

pub struct ProfileStore {
    legacy_path: PathBuf,
    instances: BTreeMap<Key, Instance>,
    /// (runtime, absolute source model key) -> managed catalog leaf.
    model_dirs: BTreeMap<(String, String), PathBuf>,
    /// Instances that could not be persisted to their per-model YAML file.
    fallback: BTreeSet<Key>,
}

impl ProfileStore {
    /// Load per-model YAML profiles and import the legacy flat JSON store.
    pub fn load(legacy_path: PathBuf, models: &[Model], vllm_models: &[Model]) -> Self {
        let mut instances = match std::fs::read(&legacy_path) {
            Ok(bytes) => serde_json::from_slice::<StoreFile>(&bytes)
                .map(|f| {
                    f.instances
                        .into_iter()
                        .map(|r| {
                            (
                                (r.runtime, r.model, r.profile),
                                Instance {
                                    values: r.values,
                                    favorite: r.favorite,
                                    custom: r.custom,
                                },
                            )
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Err(_) => BTreeMap::new(),
        };
        let mut fallback: BTreeSet<Key> = instances.keys().cloned().collect();
        let model_dirs: BTreeMap<(String, String), PathBuf> =
            [(RuntimeId::LlamaCpp, models), (RuntimeId::Vllm, vllm_models)]
                .into_iter()
                .flat_map(|(runtime, models)| {
                    models.iter().filter(|m| valid_catalog_dir(&m.catalog_dir)).map(move |m| {
                        (
                            (runtime_name(runtime).to_string(), model_key(&m.path)),
                            m.catalog_dir.clone(),
                        )
                    })
                })
                .collect();

        // YAML is authoritative when both formats contain the same profile.
        for (runtime, models) in [(RuntimeId::LlamaCpp, models), (RuntimeId::Vllm, vllm_models)] {
            for model in models {
                for loaded in load_model_profiles(runtime, model, &mut instances) {
                    fallback.remove(&loaded);
                }
                for profile in super::templates::names(runtime) {
                    instances
                        .entry(key(runtime_name(runtime), &model_key(&model.path), profile))
                        .or_default();
                }
            }
        }

        let mut store = Self { legacy_path, instances, model_dirs, fallback };
        store.back_up_legacy();
        store.persist_registered();
        store
    }

    /// Register models found by a later F5 scan and load/create their files.
    pub fn sync_models(&mut self, runtime: RuntimeId, models: &[Model]) {
        for model in models {
            if !valid_catalog_dir(&model.catalog_dir) {
                continue;
            }
            self.model_dirs.insert(
                (runtime_name(runtime).into(), model_key(&model.path)),
                model.catalog_dir.clone(),
            );
            for loaded in load_model_profiles(runtime, model, &mut self.instances) {
                self.fallback.remove(&loaded);
            }
            for profile in super::templates::names(runtime) {
                self.instances
                    .entry(key(runtime_name(runtime), &model_key(&model.path), profile))
                    .or_default();
            }
        }
        self.persist_registered();
    }

    pub fn get(&self, runtime: &str, model: &str, profile: &str) -> Option<&Instance> {
        self.instances.get(&key(runtime, model, profile))
    }

    pub fn is_favorite(&self, runtime: &str, model: &str, profile: &str) -> bool {
        self.get(runtime, model, profile).map(|i| i.favorite).unwrap_or(false)
    }

    /// Create a profile instance with the given values. Used by create and
    /// duplicate; `custom` marks user-created profiles.
    pub fn create(
        &mut self,
        runtime: &str,
        model: &str,
        profile: &str,
        values: BTreeMap<String, String>,
        custom: bool,
    ) {
        let entry = key(runtime, model, profile);
        self.instances.insert(entry.clone(), Instance { values, favorite: false, custom });
        self.persist_one(&entry);
    }

    /// Rename a profile instance, preserving its values/flags.
    pub fn rename(&mut self, runtime: &str, model: &str, old: &str, new: &str) {
        if let Some(inst) = self.instances.remove(&key(runtime, model, old)) {
            self.fallback.remove(&key(runtime, model, old));
            self.remove_profile_file(runtime, model, old);
            let entry = key(runtime, model, new);
            self.instances.insert(entry.clone(), inst);
            self.persist_one(&entry);
        }
    }

    /// Remove a profile instance. For a built-in this resets it to the template
    /// defaults; for a custom profile this deletes it entirely.
    pub fn delete(&mut self, runtime: &str, model: &str, profile: &str) {
        if self.instances.remove(&key(runtime, model, profile)).is_some() {
            self.fallback.remove(&key(runtime, model, profile));
            self.remove_profile_file(runtime, model, profile);
            self.save_legacy();
        }
    }

    /// Custom (user-created) profile names for a given model.
    pub fn custom_profiles(&self, runtime: &str, model: &str) -> Vec<String> {
        self.instances
            .iter()
            .filter(|((r, m, _), inst)| r == runtime && m == model && inst.custom)
            .map(|((_, _, p), _)| p.clone())
            .collect()
    }

    /// Set one option value, materializing the instance if needed, then save.
    pub fn set_value(
        &mut self,
        runtime: &str,
        model: &str,
        profile: &str,
        option: &str,
        value: String,
        base: &BTreeMap<String, String>,
    ) {
        let inst = self.materialize(runtime, model, profile, base);
        inst.values.insert(option.to_string(), value);
        self.persist_one(&key(runtime, model, profile));
    }

    pub fn toggle_favorite(
        &mut self,
        runtime: &str,
        model: &str,
        profile: &str,
        base: &BTreeMap<String, String>,
    ) {
        let inst = self.materialize(runtime, model, profile, base);
        inst.favorite = !inst.favorite;
        self.persist_one(&key(runtime, model, profile));
    }

    /// Ensure an instance exists, seeding its values from `base` (the resolved
    /// template values) on first materialization.
    fn materialize(
        &mut self,
        runtime: &str,
        model: &str,
        profile: &str,
        base: &BTreeMap<String, String>,
    ) -> &mut Instance {
        self.instances.entry(key(runtime, model, profile)).or_insert_with(|| Instance {
            values: base.clone(),
            favorite: false,
            custom: RuntimeId::from_name(runtime)
                .is_none_or(|runtime| !super::templates::is_builtin(runtime, profile)),
        })
    }

    fn persist_registered(&mut self) {
        let keys: Vec<Key> = self
            .instances
            .keys()
            .filter(|(runtime, model, _)| {
                self.model_dirs.contains_key(&(runtime.clone(), model.clone()))
            })
            .cloned()
            .collect();
        for entry in keys {
            self.persist_yaml(&entry);
        }
        self.save_legacy();
    }

    fn persist_one(&mut self, entry: &Key) {
        self.persist_yaml(entry);
        self.save_legacy();
    }

    fn persist_yaml(&mut self, entry: &Key) {
        match self.write_profile(entry) {
            Ok(()) => {
                self.fallback.remove(entry);
            }
            Err(err) => {
                self.fallback.insert(entry.clone());
                warn!(model = %entry.1, profile = %entry.2, %err, "using legacy profile fallback");
            }
        }
    }

    fn write_profile(&self, entry: &Key) -> std::io::Result<()> {
        let (runtime, model, profile) = entry;
        let inst =
            self.instances.get(entry).ok_or_else(|| std::io::Error::other("missing instance"))?;
        let dir = self
            .model_dirs
            .get(&(runtime.clone(), model.clone()))
            .filter(|dir| valid_catalog_dir(dir))
            .ok_or_else(|| std::io::Error::other("model catalog unavailable"))?;
        let file = ProfileFile {
            schema: 1,
            name: profile.clone(),
            values: inst.values.clone(),
            favorite: inst.favorite,
            custom: inst.custom,
        };
        let yaml = serde_yaml::to_string(&file).map_err(std::io::Error::other)?;
        write_atomic_if_changed(
            &dir.join("profiles").join(profile_filename(profile)),
            yaml.as_bytes(),
        )
    }

    // Retain unavailable/failed instances in the legacy store; never discard
    // user profiles merely because the managed catalog cannot be written.
    fn save_legacy(&self) {
        let file = StoreFile {
            instances: self
                .fallback
                .iter()
                .filter_map(|(r, m, p)| {
                    self.instances.get(&(r.clone(), m.clone(), p.clone())).map(|inst| Record {
                        runtime: r.clone(),
                        model: m.clone(),
                        profile: p.clone(),
                        values: inst.values.clone(),
                        favorite: inst.favorite,
                        custom: inst.custom,
                    })
                })
                .collect(),
        };
        if file.instances.is_empty() {
            if let Err(err) = std::fs::remove_file(&self.legacy_path)
                && err.kind() != std::io::ErrorKind::NotFound
            {
                warn!(path = %self.legacy_path.display(), %err, "failed to retire legacy store");
            }
            return;
        }
        match serde_json::to_vec_pretty(&file) {
            Ok(bytes) => {
                if let Err(err) = std::fs::write(&self.legacy_path, bytes) {
                    warn!(path = %self.legacy_path.display(), %err, "failed to write profile store");
                }
            }
            Err(err) => warn!(%err, "failed to serialize profile store"),
        }
    }

    fn remove_profile_file(&self, runtime: &str, model: &str, profile: &str) {
        let Some(dir) = self.model_dirs.get(&(runtime.to_string(), model.to_string())) else {
            return;
        };
        let path = dir.join("profiles").join(profile_filename(profile));
        if let Err(err) = std::fs::remove_file(&path)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            warn!(path = %path.display(), %err, "failed to remove profile file");
        }
    }

    fn back_up_legacy(&self) {
        if !self.legacy_path.exists() {
            return;
        }
        let backup = self.legacy_path.with_extension("json.bak");
        if !backup.exists()
            && let Err(err) = std::fs::copy(&self.legacy_path, &backup)
        {
            warn!(path = %backup.display(), %err, "failed to back up legacy profile store");
        }
    }
}

fn load_model_profiles(
    runtime: RuntimeId,
    model: &Model,
    instances: &mut BTreeMap<Key, Instance>,
) -> Vec<Key> {
    let mut loaded = Vec::new();
    if !valid_catalog_dir(&model.catalog_dir) {
        return loaded;
    }
    let dir = model.catalog_dir.join("profiles");
    let Ok(entries) = std::fs::read_dir(&dir) else { return loaded };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "yml" && e != "yaml") {
            continue;
        }
        match std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_yaml::from_slice::<ProfileFile>(&bytes).ok())
        {
            Some(file) if file.schema == 1 => {
                let entry = key(runtime_name(runtime), &model_key(&model.path), &file.name);
                instances.insert(
                    entry.clone(),
                    Instance { values: file.values, favorite: file.favorite, custom: file.custom },
                );
                loaded.push(entry);
            }
            _ => warn!(path = %path.display(), "ignoring invalid profile YAML"),
        }
    }
    loaded
}

fn valid_catalog_dir(path: &Path) -> bool {
    path.is_absolute() && path.join("profiles").is_dir()
}

fn profile_filename(name: &str) -> String {
    if !name.is_empty()
        && name != "."
        && name != ".."
        && name.chars().all(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_'))
    {
        return format!("{name}.yml");
    }
    let safe: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || matches!(c, '-' | '_') { c } else { '_' })
        .collect();
    let hash = crate::discovery::catalog::short_hash(Path::new(name));
    format!("{safe}-{hash}.yml")
}

fn write_atomic_if_changed(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if std::fs::read(path).is_ok_and(|existing| existing == bytes) {
        return Ok(());
    }
    let tmp = path.with_extension("yml.tmp");
    std::fs::write(&tmp, bytes).and_then(|_| std::fs::rename(&tmp, path))
}

fn key(runtime: &str, model: &str, profile: &str) -> Key {
    (runtime.to_string(), model.to_string(), profile.to_string())
}

fn runtime_name(runtime: RuntimeId) -> &'static str {
    match runtime {
        RuntimeId::LlamaCpp => "llama.cpp",
        RuntimeId::Vllm => "vLLM",
    }
}

/// Convenience for callers that have a model path.
pub fn model_key(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn migrates_legacy_json_to_model_yaml_and_keeps_backup() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-profiles-{nonce}"));
        let catalog = root.join("catalog/model");
        std::fs::create_dir_all(catalog.join("profiles")).unwrap();
        let legacy = root.join("profiles.json");
        let model_path = root.join("source.gguf");
        let file = StoreFile {
            instances: vec![Record {
                runtime: "llama.cpp".into(),
                model: model_key(&model_path),
                profile: "Chat".into(),
                values: BTreeMap::from([("ctx-size".into(), "8192".into())]),
                favorite: true,
                custom: false,
            }],
        };
        std::fs::write(&legacy, serde_json::to_vec(&file).unwrap()).unwrap();
        let model = Model {
            id: "test".into(),
            name: "source.gguf".into(),
            path: model_path,
            shard_paths: Vec::new(),
            catalog_path: vec!["local".into(), "source".into()],
            catalog_dir: catalog.clone(),
            size_bytes: 0,
            quantization: None,
            architecture: None,
            context_length: None,
            modified: None,
            has_chat_template: false,
        };
        let store = ProfileStore::load(legacy.clone(), &[model], &[]);
        assert_eq!(
            store.get("llama.cpp", &model_key(&root.join("source.gguf")), "Chat").unwrap().values["ctx-size"],
            "8192"
        );
        assert!(catalog.join("profiles/Chat.yml").is_file());
        assert!(legacy.with_extension("json.bak").is_file());
        assert!(!legacy.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unavailable_catalog_persists_edits_in_legacy_json() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-fallback-{nonce}"));
        std::fs::create_dir_all(&root).unwrap();
        let legacy = root.join("profiles.json");
        let model_path = root.join("source.gguf");
        let model = Model {
            id: "test".into(),
            name: "source.gguf".into(),
            path: model_path.clone(),
            shard_paths: Vec::new(),
            catalog_path: vec!["local".into(), "source".into()],
            catalog_dir: PathBuf::new(), // reconcile failed
            size_bytes: 0,
            quantization: None,
            architecture: None,
            context_length: None,
            modified: None,
            has_chat_template: false,
        };
        let mut store = ProfileStore::load(legacy.clone(), &[model], &[]);
        store.set_value(
            "llama.cpp",
            &model_key(&model_path),
            "Chat",
            "ctx-size",
            "8192".into(),
            &BTreeMap::new(),
        );
        let saved: StoreFile = serde_json::from_slice(&std::fs::read(&legacy).unwrap()).unwrap();
        assert_eq!(saved.instances.len(), 1);
        assert_eq!(saved.instances[0].values["ctx-size"], "8192");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn vllm_profiles_persist_in_their_catalog_without_crossing_runtimes() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-vllm-profiles-{nonce}"));
        let catalog = root.join("catalog/vllm/huggingface/team/model");
        std::fs::create_dir_all(catalog.join("profiles")).unwrap();
        let model_path = root.join("snapshot");
        std::fs::create_dir_all(&model_path).unwrap();
        let model = Model {
            id: "vllm:test".into(),
            name: "team/model".into(),
            path: model_path.clone(),
            shard_paths: Vec::new(),
            catalog_path: vec!["huggingface".into(), "team".into(), "model".into()],
            catalog_dir: catalog.clone(),
            size_bytes: 0,
            quantization: None,
            architecture: None,
            context_length: None,
            modified: None,
            has_chat_template: true,
        };
        let legacy = root.join("profiles.json");
        let mut store = ProfileStore::load(legacy.clone(), &[], &[model.clone()]);
        store.set_value(
            "vLLM",
            &model_key(&model_path),
            "Default",
            "tensor-parallel-size",
            "2".into(),
            &BTreeMap::new(),
        );
        assert!(catalog.join("profiles/Default.yml").is_file());
        assert!(store.get("llama.cpp", &model_key(&model_path), "Default").is_none());

        let reloaded = ProfileStore::load(legacy, &[], &[model]);
        assert_eq!(
            reloaded.get("vLLM", &model_key(&model_path), "Default").unwrap().values["tensor-parallel-size"],
            "2"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
