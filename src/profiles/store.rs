//! Persistent, model-scoped profile instances.
//!
//! A profile is identified by (runtime, model, profile-name). Built-in
//! templates are global and read-only; the first time the user edits an option
//! (or favorites/creates a profile) for a given model, an *instance* is
//! materialized here and auto-saved. Stored as per-model YAML, with a legacy JSON fallback.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::domain::Model;
use crate::persistence::write_if_changed;

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
    /// Absolute source model key -> managed catalog leaf.
    model_dirs: BTreeMap<String, PathBuf>,
    /// Instances that could not be persisted to their per-model YAML file.
    fallback: BTreeSet<Key>,
}

impl ProfileStore {
    /// Load per-model YAML profiles and import the legacy flat JSON store.
    pub fn load(legacy_path: PathBuf, models: &[Model]) -> Self {
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
        let model_dirs: BTreeMap<String, PathBuf> = models
            .iter()
            .filter(|m| valid_catalog_dir(&m.catalog_dir))
            .map(|m| (m.profile_key(), m.catalog_dir.clone()))
            .collect();

        // A fallback records a failed YAML write and is newer than that YAML.
        let pending = instances.clone();
        for model in models {
            for loaded in load_model_profiles(model, &mut instances) {
                fallback.remove(&loaded);
            }
            for profile in super::templates::names(crate::runtime::templates_for(&model.runtime)) {
                instances.entry(key(&model.runtime, &model.profile_key(), profile)).or_default();
            }
        }

        instances.extend(pending.iter().map(|(key, value)| (key.clone(), value.clone())));
        fallback.extend(pending.into_keys());
        let mut store = Self { legacy_path, instances, model_dirs, fallback };
        store.back_up_legacy();
        store.persist_registered();
        store
    }

    /// Register models found by a later F5 scan and load/create their files.
    pub fn sync_models(&mut self, models: &[Model]) {
        let pending: Vec<_> = self
            .fallback
            .iter()
            .filter_map(|key| self.instances.get(key).map(|value| (key.clone(), value.clone())))
            .collect();
        for model in models {
            if !valid_catalog_dir(&model.catalog_dir) {
                continue;
            }
            self.model_dirs.insert(model.profile_key(), model.catalog_dir.clone());
            for loaded in load_model_profiles(model, &mut self.instances) {
                self.fallback.remove(&loaded);
            }
            for profile in super::templates::names(crate::runtime::templates_for(&model.runtime)) {
                self.instances
                    .entry(key(&model.runtime, &model.profile_key(), profile))
                    .or_default();
            }
        }
        for (key, value) in pending {
            self.instances.insert(key.clone(), value);
            self.fallback.insert(key);
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
    ) -> io::Result<()> {
        let entry = key(runtime, model, profile);
        self.instances.insert(entry.clone(), Instance { values, favorite: false, custom });
        self.persist_one(&entry)
    }

    /// Rename a profile instance, preserving its values/flags.
    pub fn rename(&mut self, runtime: &str, model: &str, old: &str, new: &str) -> io::Result<()> {
        let old_key = key(runtime, model, old);
        let new_key = key(runtime, model, new);
        if old == new {
            return Ok(());
        }
        if self.instances.contains_key(&new_key) {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, "profile already exists"));
        }
        let inst = self
            .instances
            .get(&old_key)
            .cloned()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "profile not found"))?;
        self.instances.insert(new_key.clone(), inst);
        if let Err(error) = self.persist_one(&new_key) {
            self.instances.remove(&new_key);
            self.fallback.remove(&new_key);
            return Err(error);
        }
        // The replacement is durable before the original is removed. If removal
        // fails, both names survive and the caller reports the failure.
        self.delete(runtime, model, old)
    }

    /// Delete on disk before removing the in-memory instance. A failed unlink
    /// must not make a profile disappear until the next reload.
    pub fn delete(&mut self, runtime: &str, model: &str, profile: &str) -> io::Result<()> {
        let entry = key(runtime, model, profile);
        self.remove_profile_file(model, profile)?;
        let previous = self.instances.remove(&entry);
        let was_fallback = self.fallback.remove(&entry);
        if let Err(error) = self.save_legacy() {
            if let Some(previous) = previous {
                self.instances.insert(entry.clone(), previous);
            }
            if was_fallback {
                self.fallback.insert(entry);
            }
            return Err(error);
        }
        Ok(())
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
    ) -> io::Result<()> {
        let inst = self.materialize(runtime, model, profile, base);
        inst.values.insert(option.to_string(), value);
        self.persist_one(&key(runtime, model, profile))
    }

    pub fn toggle_favorite(
        &mut self,
        runtime: &str,
        model: &str,
        profile: &str,
        base: &BTreeMap<String, String>,
    ) -> io::Result<()> {
        let inst = self.materialize(runtime, model, profile, base);
        inst.favorite = !inst.favorite;
        self.persist_one(&key(runtime, model, profile))
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
            custom: !super::templates::is_builtin(crate::runtime::templates_for(runtime), profile),
        })
    }

    fn persist_registered(&mut self) {
        let keys: Vec<Key> = self
            .instances
            .keys()
            .filter(|(_, model, _)| self.model_dirs.contains_key(model))
            .cloned()
            .collect();
        for entry in keys {
            self.persist_yaml(&entry);
        }
        if let Err(error) = self.save_legacy() {
            warn!(%error, "failed to persist profile fallback");
        }
    }

    fn persist_one(&mut self, entry: &Key) -> io::Result<()> {
        self.persist_yaml(entry);
        self.save_legacy()
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
        let (_, model, profile) = entry;
        let inst =
            self.instances.get(entry).ok_or_else(|| std::io::Error::other("missing instance"))?;
        let dir = self
            .model_dirs
            .get(model)
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
        write_if_changed(&dir.join("profiles").join(profile_filename(profile)), yaml.as_bytes())
    }

    // Retain unavailable/failed instances in the legacy store; never discard
    // user profiles merely because the managed catalog cannot be written.
    fn save_legacy(&self) -> io::Result<()> {
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
            return match std::fs::remove_file(&self.legacy_path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(error),
            };
        }
        let bytes = serde_json::to_vec_pretty(&file).map_err(io::Error::other)?;
        crate::persistence::write_atomic(&self.legacy_path, &bytes)
    }

    fn remove_profile_file(&self, model: &str, profile: &str) -> io::Result<()> {
        let Some(dir) = self.model_dirs.get(model) else { return Ok(()) };
        let path = dir.join("profiles").join(profile_filename(profile));
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
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

fn load_model_profiles(model: &Model, instances: &mut BTreeMap<Key, Instance>) -> Vec<Key> {
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
                let entry = key(&model.runtime, &model.profile_key(), &file.name);
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

fn key(runtime: &str, model: &str, profile: &str) -> Key {
    (runtime.to_string(), model.to_string(), profile.to_string())
}

/// Convenience for callers that have a model path.
#[cfg(test)]
pub fn model_key(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> (PathBuf, Model, ProfileStore) {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-profile-errors-{nonce}"));
        let catalog = root.join("model");
        std::fs::create_dir_all(catalog.join("profiles")).unwrap();
        let model: Model = serde_json::from_value(serde_json::json!({
            "name": "model.gguf", "path": root.join("model.gguf"),
            "catalog_dir": catalog, "size_bytes": 0, "quantization": null,
            "architecture": null, "context_length": null, "modified": null,
            "has_chat_template": false
        }))
        .unwrap();
        let store = ProfileStore::load(root.join("legacy.json"), std::slice::from_ref(&model));
        (root, model, store)
    }

    #[test]
    fn failed_rename_preserves_the_original_on_disk_and_in_memory() {
        let (root, model, mut store) = fixture();
        let key = model.profile_key();
        store.create("llama.cpp", &key, "Original", BTreeMap::new(), true).unwrap();
        let original = model.catalog_dir.join("profiles/Original.yml");
        let bytes = std::fs::read(&original).unwrap();
        std::fs::create_dir(model.catalog_dir.join("profiles/Renamed.yml")).unwrap();
        std::fs::create_dir(&store.legacy_path).unwrap();
        assert!(store.rename("llama.cpp", &key, "Original", "Renamed").is_err());
        assert_eq!(std::fs::read(&original).unwrap(), bytes);
        assert!(store.get("llama.cpp", &key, "Original").is_some());
        assert!(store.get("llama.cpp", &key, "Renamed").is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_delete_keeps_the_profile_visible() {
        let (root, model, mut store) = fixture();
        let key = model.profile_key();
        let file = model.catalog_dir.join("profiles/Chat.yml");
        std::fs::remove_file(&file).unwrap();
        std::fs::create_dir(&file).unwrap();
        assert!(store.delete("llama.cpp", &key, "Chat").is_err());
        assert!(store.get("llama.cpp", &key, "Chat").is_some());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_edit_reports_error_and_keeps_the_edit_for_retry() {
        let (root, model, mut store) = fixture();
        let key = model.profile_key();
        let file = model.catalog_dir.join("profiles/Chat.yml");
        std::fs::remove_file(&file).unwrap();
        std::fs::create_dir(&file).unwrap();
        std::fs::create_dir(&store.legacy_path).unwrap();
        assert!(
            store
                .set_value("llama.cpp", &key, "Chat", "temperature", "0.7".into(), &BTreeMap::new())
                .is_err()
        );
        assert_eq!(store.get("llama.cpp", &key, "Chat").unwrap().values["temperature"], "0.7");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fallback_edits_override_stale_yaml_on_reload_and_rescan() {
        let (root, model, mut store) = fixture();
        let entry = key("llama.cpp", &model.profile_key(), "Chat");
        store.instances.get_mut(&entry).unwrap().values.insert("temperature".into(), "0.7".into());
        store.fallback.insert(entry.clone());
        store.save_legacy().unwrap();
        let mut reloaded =
            ProfileStore::load(store.legacy_path.clone(), std::slice::from_ref(&model));
        assert_eq!(reloaded.instances[&entry].values["temperature"], "0.7");
        reloaded
            .instances
            .get_mut(&entry)
            .unwrap()
            .values
            .insert("temperature".into(), "0.8".into());
        reloaded.fallback.insert(entry.clone());
        reloaded.sync_models(&[model]);
        assert_eq!(reloaded.instances[&entry].values["temperature"], "0.8");
        std::fs::remove_dir_all(root).unwrap();
    }

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
            mtp_path: None,
            dflash_path: None,
            dflash_block_size: None,
            projector_path: None,
            has_mtp: false,
            catalog_path: vec!["local".into(), "source".into()],
            catalog_dir: catalog.clone(),
            size_bytes: 0,
            quantization: None,
            architecture: None,
            context_length: None,
            modified: None,
            has_chat_template: false,
            flm: None,
            runtime: crate::runtime::llama_cpp::NAME.into(),
            remote: None,
        };
        let store = ProfileStore::load(legacy.clone(), &[model]);
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
            mtp_path: None,
            dflash_path: None,
            dflash_block_size: None,
            projector_path: None,
            has_mtp: false,
            catalog_path: vec!["local".into(), "source".into()],
            catalog_dir: PathBuf::new(), // reconcile failed
            size_bytes: 0,
            quantization: None,
            architecture: None,
            context_length: None,
            modified: None,
            has_chat_template: false,
            flm: None,
            runtime: crate::runtime::llama_cpp::NAME.into(),
            remote: None,
        };
        let mut store = ProfileStore::load(legacy.clone(), &[model]);
        store
            .set_value(
                "llama.cpp",
                &model_key(&model_path),
                "Chat",
                "ctx-size",
                "8192".into(),
                &BTreeMap::new(),
            )
            .unwrap();
        let saved: StoreFile = serde_json::from_slice(&std::fs::read(&legacy).unwrap()).unwrap();
        assert_eq!(saved.instances.len(), 1);
        assert_eq!(saved.instances[0].values["ctx-size"], "8192");
        std::fs::remove_dir_all(root).unwrap();
    }
}
