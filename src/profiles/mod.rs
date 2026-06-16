//! Profile/option resolution: combine the registry, built-in templates, the
//! config defaults, and persisted instances into the lists the UI shows.

pub mod registry;
pub mod store;
pub mod templates;

use std::collections::BTreeMap;

use crate::config::Defaults;
use crate::domain::{Model, OptionItem, Profile, Runtime};
use crate::profiles::registry::OptionKind;

pub use store::ProfileStore;

/// The option's kind specialized for a given model: `ctx-size` gains an upper
/// bound equal to the model's trained context length, so `End`/`+` can target
/// "max supported context" rather than an unbounded value.
pub fn effective_kind(spec: &registry::OptionSpec, model: &Model) -> OptionKind {
    match (spec.key, model.context_length) {
        ("ctx-size", Some(ctx)) => OptionKind::Int { min: Some(0), max: Some(ctx as i64) },
        _ => spec.kind,
    }
}

/// Profiles available for a (runtime, model): built-in templates plus any
/// user-created custom profiles, with favorite flags from the store.
pub fn list_profiles(runtime: &Runtime, model: &Model, store: &ProfileStore) -> Vec<Profile> {
    let model_key = store::model_key(&model.path);
    let mut profiles: Vec<Profile> = templates::names()
        .map(|name| Profile {
            name: name.to_string(),
            builtin: true,
            favorite: store.is_favorite(&runtime.name, &model_key, name),
        })
        .collect();

    let mut custom = store.custom_profiles(&runtime.name, &model_key);
    custom.sort();
    for name in custom {
        let favorite = store.is_favorite(&runtime.name, &model_key, &name);
        profiles.push(Profile { name, builtin: false, favorite });
    }
    profiles
}

/// Resolve the option values for a (runtime, model, profile), layering:
/// instance override → template override → config default → registry default.
pub fn resolve_options(
    runtime: &Runtime,
    model: &Model,
    profile: &Profile,
    store: &ProfileStore,
    defaults: &Defaults,
) -> Vec<OptionItem> {
    let model_key = store::model_key(&model.path);
    let instance = store.get(&runtime.name, &model_key, &profile.name);
    let template = templates::find(&profile.name);

    registry::REGISTRY
        .iter()
        .map(|spec| {
            let default = config_default(spec, defaults);

            let value = instance
                .and_then(|i| i.values.get(spec.key).cloned())
                .or_else(|| template.and_then(|t| override_value(t, spec.key)))
                .unwrap_or_else(|| default.clone());
            let value = clamp_ctx_to_model(spec.key, value, model);

            OptionItem {
                key: spec.key.to_string(),
                value,
                default,
                range: effective_kind(spec, model).range_label(),
                cli: spec.cli.to_string(),
                description: spec.description.to_string(),
            }
        })
        .collect()
}

/// The fully-resolved current values for a (runtime, model, profile), including
/// any instance edits. Used to seed a duplicated/created profile.
pub fn current_values(
    runtime: &Runtime,
    model: &Model,
    profile: &Profile,
    store: &ProfileStore,
    defaults: &Defaults,
) -> BTreeMap<String, String> {
    resolve_options(runtime, model, profile, store, defaults)
        .into_iter()
        .map(|o| (o.key, o.value))
        .collect()
}

/// The resolved template values for a profile (used to seed a new instance).
pub fn resolved_values(
    profile: &Profile,
    defaults: &Defaults,
) -> BTreeMap<String, String> {
    let template = templates::find(&profile.name);
    registry::REGISTRY
        .iter()
        .map(|spec| {
            let value = template
                .and_then(|t| override_value(t, spec.key))
                .unwrap_or_else(|| config_default(spec, defaults));
            (spec.key.to_string(), value)
        })
        .collect()
}

/// Registry default, overridden by config for host/port.
fn config_default(spec: &registry::OptionSpec, defaults: &Defaults) -> String {
    match spec.key {
        "host" => defaults.host.clone(),
        "port" => defaults.port.to_string(),
        _ => spec.default.to_string(),
    }
}

fn override_value(template: &templates::Template, key: &str) -> Option<String> {
    template.overrides.iter().find(|(k, _)| *k == key).map(|(_, v)| v.to_string())
}

/// Clamp a `ctx-size` value down to the model's trained context length so the
/// default never exceeds what the model supports.
fn clamp_ctx_to_model(key: &str, value: String, model: &Model) -> String {
    if key != "ctx-size" {
        return value;
    }
    match (model.context_length, value.parse::<i64>()) {
        (Some(ctx), Ok(v)) if v > ctx as i64 => ctx.to_string(),
        _ => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> Runtime {
        Runtime {
            name: "llama.cpp".into(),
            description: String::new(),
            version: None,
            binary_path: None,
            formats: vec![],
        }
    }

    fn model() -> Model {
        Model {
            name: "x.gguf".into(),
            path: "/tmp/x.gguf".into(),
            size_bytes: 0,
            quantization: None,
            architecture: None,
            context_length: None,
            modified: None,
            has_chat_template: false,
        }
    }

    fn profile(name: &str) -> Profile {
        Profile { name: name.into(), builtin: true, favorite: false }
    }

    fn empty_store() -> ProfileStore {
        ProfileStore::load("/nonexistent/llmctl-test-store.json".into())
    }

    fn value_of(opts: &[OptionItem], key: &str) -> String {
        opts.iter().find(|o| o.key == key).unwrap().value.clone()
    }

    #[test]
    fn default_profile_uses_registry_defaults() {
        let opts =
            resolve_options(&runtime(), &model(), &profile("Default"), &empty_store(), &Defaults::default());
        assert_eq!(value_of(&opts, "ctx-size"), "4096");
        assert_eq!(value_of(&opts, "temperature"), "0.8");
    }

    #[test]
    fn template_overrides_apply() {
        let opts =
            resolve_options(&runtime(), &model(), &profile("Coding"), &empty_store(), &Defaults::default());
        assert_eq!(value_of(&opts, "ctx-size"), "16384");
        assert_eq!(value_of(&opts, "temperature"), "0.2");
    }

    fn model_with_ctx(ctx: Option<u64>) -> Model {
        Model { context_length: ctx, ..model() }
    }

    #[test]
    fn ctx_size_max_follows_model_context() {
        let m = model_with_ctx(Some(8192));
        let spec = registry::spec("ctx-size").unwrap();
        let kind = effective_kind(spec, &m);
        assert_eq!(kind.extreme(1), Some("8192".into())); // End → model max
        assert_eq!(kind.adjust("8192", 1, spec.step), Some("8192".into())); // clamps
    }

    #[test]
    fn ctx_size_default_clamped_to_small_model() {
        // Registry default is 4096; a 2048-ctx model should resolve to 2048.
        let m = model_with_ctx(Some(2048));
        let opts = resolve_options(&runtime(), &m, &profile("Default"), &empty_store(), &Defaults::default());
        assert_eq!(value_of(&opts, "ctx-size"), "2048");
    }

    #[test]
    fn host_port_come_from_config_defaults() {
        let defaults = Defaults { host: "0.0.0.0".into(), port: 9000 };
        let opts = resolve_options(&runtime(), &model(), &profile("Default"), &empty_store(), &defaults);
        assert_eq!(value_of(&opts, "host"), "0.0.0.0");
        assert_eq!(value_of(&opts, "port"), "9000");
    }
}
