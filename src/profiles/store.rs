//! Persistent, model-scoped profile instances.
//!
//! A profile is identified by (runtime, model, profile-name). Built-in
//! templates are global and read-only; the first time the user edits an option
//! (or favorites/creates a profile) for a given model, an *instance* is
//! materialized here and auto-saved. Stored as JSON under the state dir.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::warn;

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

pub struct ProfileStore {
    path: PathBuf,
    instances: BTreeMap<Key, Instance>,
}

impl ProfileStore {
    /// Load the store from `path`, or start empty if it doesn't exist yet.
    pub fn load(path: PathBuf) -> Self {
        let instances = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice::<StoreFile>(&bytes)
                .map(|f| {
                    f.instances
                        .into_iter()
                        .map(|r| {
                            (
                                (r.runtime, r.model, r.profile),
                                Instance { values: r.values, favorite: r.favorite, custom: r.custom },
                            )
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Err(_) => BTreeMap::new(),
        };
        Self { path, instances }
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
        self.instances
            .insert(key(runtime, model, profile), Instance { values, favorite: false, custom });
        self.save();
    }

    /// Rename a profile instance, preserving its values/flags.
    pub fn rename(&mut self, runtime: &str, model: &str, old: &str, new: &str) {
        if let Some(inst) = self.instances.remove(&key(runtime, model, old)) {
            self.instances.insert(key(runtime, model, new), inst);
            self.save();
        }
    }

    /// Remove a profile instance. For a built-in this resets it to the template
    /// defaults; for a custom profile this deletes it entirely.
    pub fn delete(&mut self, runtime: &str, model: &str, profile: &str) {
        if self.instances.remove(&key(runtime, model, profile)).is_some() {
            self.save();
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
        self.save();
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
        self.save();
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
            custom: !super::templates::is_builtin(profile),
        })
    }

    fn save(&self) {
        let file = StoreFile {
            instances: self
                .instances
                .iter()
                .map(|((r, m, p), inst)| Record {
                    runtime: r.clone(),
                    model: m.clone(),
                    profile: p.clone(),
                    values: inst.values.clone(),
                    favorite: inst.favorite,
                    custom: inst.custom,
                })
                .collect(),
        };
        match serde_json::to_vec_pretty(&file) {
            Ok(bytes) => {
                if let Err(err) = std::fs::write(&self.path, bytes) {
                    warn!(path = %self.path.display(), %err, "failed to write profile store");
                }
            }
            Err(err) => warn!(%err, "failed to serialize profile store"),
        }
    }
}

fn key(runtime: &str, model: &str, profile: &str) -> Key {
    (runtime.to_string(), model.to_string(), profile.to_string())
}

/// Convenience for callers that have a model path.
pub fn model_key(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
