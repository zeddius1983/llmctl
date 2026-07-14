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
    let model_key = model.profile_key();
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
    let model_key = model.profile_key();
    let instance = store.get(&runtime.name, &model_key, &profile.name);
    let template = templates::find(&profile.name);

    registry::REGISTRY
        .iter()
        .map(|spec| {
            let default = spec_default(spec, model, defaults);

            let value = instance
                .and_then(|i| i.values.get(spec.key).cloned())
                .or_else(|| template.and_then(|t| override_value(t, spec.key)))
                .unwrap_or_else(|| default.clone());
            let value = clamp_ctx_to_model(spec.key, value, model);
            let value = normalize_legacy(spec.key, value);

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

/// The resolved template values for a (profile, model) — the no-instance layer
/// of [`resolve_options`]. Used as the base snapshot when materializing an
/// instance on first edit/favorite; it must match what the Options pane was
/// showing, or the first edit would silently shift unedited options (e.g.
/// ctx-size from the model-aware ctx/8 back to the global 4096).
pub fn resolved_values(
    profile: &Profile,
    model: &Model,
    defaults: &Defaults,
) -> BTreeMap<String, String> {
    let template = templates::find(&profile.name);
    registry::REGISTRY
        .iter()
        .map(|spec| {
            let value = template
                .and_then(|t| override_value(t, spec.key))
                .unwrap_or_else(|| spec_default(spec, model, defaults));
            let value = clamp_ctx_to_model(spec.key, value, model);
            (spec.key.to_string(), normalize_legacy(spec.key, value))
        })
        .collect()
}

/// The model-aware default for an option: the omit token for omittable options,
/// except ctx-size, which must not *start* omitted — its 'default' means the
/// model's full trained context, which can exhaust memory — and begins at the
/// ctx/8 heuristic instead.
fn spec_default(spec: &registry::OptionSpec, model: &Model, defaults: &Defaults) -> String {
    // An integrated or sidecar MTP head is only useful when llama.cpp's MTP
    // drafter is enabled. Make that the model-aware default while still
    // allowing a saved profile value to override it.
    if spec.key == "spec-type" && model.supports_mtp() {
        return "draft-mtp".into();
    }
    match registry::omit_token(spec.key) {
        Some(token) if spec.key != "ctx-size" => token.to_string(),
        _ => model_default(spec, model, defaults),
    }
}

/// Registry default, overridden by config for host/port.
fn config_default(spec: &registry::OptionSpec, defaults: &Defaults) -> String {
    match spec.key {
        "host" => defaults.host.clone(),
        "port" => defaults.port.to_string(),
        _ => spec.default.to_string(),
    }
}

/// Default specialized for the model: `ctx-size` defaults to one eighth of the
/// model's trained context (a memory-friendly starting point that the user can
/// raise toward the model max); everything else uses the config/registry default.
fn model_default(spec: &registry::OptionSpec, model: &Model, defaults: &Defaults) -> String {
    if spec.key == "ctx-size" {
        if let Some(ctx) = model.context_length {
            if ctx >= 8 {
                return (ctx / 8).to_string();
            }
        }
    }
    config_default(spec, defaults)
}

/// Map legacy stored values for the on/off/auto enums onto the current vocabulary
/// so old profiles still launch: booleans (`true`/`false`, from when flash-attn
/// was a switch) and the old `default` sentinel (now spelled `auto`).
fn normalize_legacy(key: &str, value: String) -> String {
    if key != "flash-attn" && key != "reasoning" {
        return value;
    }
    match value.as_str() {
        "true" => "on".into(),
        "false" => "off".into(),
        "default" => "auto".into(),
        _ => value,
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
            bench_path: None,
            formats: vec![],
            devices: vec![],
        }
    }

    fn model() -> Model {
        Model {
            id: "test-model".into(),
            name: "x.gguf".into(),
            path: "/tmp/x.gguf".into(),
            shard_paths: vec!["/tmp/x.gguf".into()],
            mtp_path: None,
            projector_path: None,
            has_mtp: false,
            catalog_path: vec!["test-model".into()],
            catalog_dir: "/tmp/test-model".into(),
            size_bytes: 0,
            quantization: None,
            architecture: None,
            context_length: None,
            modified: None,
            has_chat_template: false,
            remote: None,
        }
    }

    fn profile(name: &str) -> Profile {
        Profile { name: name.into(), builtin: true, favorite: false }
    }

    fn empty_store() -> ProfileStore {
        ProfileStore::load("/nonexistent/llmctl-test-store.json".into(), &[])
    }

    fn value_of(opts: &[OptionItem], key: &str) -> String {
        opts.iter().find(|o| o.key == key).unwrap().value.clone()
    }

    #[test]
    fn default_profile_uses_registry_defaults() {
        let opts = resolve_options(
            &runtime(),
            &model(),
            &profile("Default"),
            &empty_store(),
            &Defaults::default(),
        );
        // ctx-size starts concrete (the ctx/8 heuristic / registry fallback),
        // never at 'default' (= the model's full context).
        assert_eq!(value_of(&opts, "ctx-size"), "4096");
        // Sampling params start omitted — llama.cpp's own defaults apply.
        for key in ["temperature", "top-p", "top-k", "min-p", "repeat-penalty"] {
            assert_eq!(value_of(&opts, key), registry::DEFAULT, "{key} should start at default");
        }
    }

    #[test]
    fn template_overrides_apply() {
        let opts = resolve_options(
            &runtime(),
            &model(),
            &profile("Coding"),
            &empty_store(),
            &Defaults::default(),
        );
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
    fn ctx_size_default_is_model_context_over_eight() {
        // A model with a known context defaults ctx-size to ctx / 8.
        let m = model_with_ctx(Some(32768));
        let opts = resolve_options(
            &runtime(),
            &m,
            &profile("Default"),
            &empty_store(),
            &Defaults::default(),
        );
        assert_eq!(value_of(&opts, "ctx-size"), "4096");
    }

    #[test]
    fn ctx_size_template_override_clamped_to_small_model() {
        // Long Context overrides ctx-size to 131072; a 2048-ctx model clamps it.
        let m = model_with_ctx(Some(2048));
        let opts = resolve_options(
            &runtime(),
            &m,
            &profile("Long Context"),
            &empty_store(),
            &Defaults::default(),
        );
        assert_eq!(value_of(&opts, "ctx-size"), "2048");
    }

    #[test]
    fn omittable_options_default_to_their_omit_token() {
        let opts = resolve_options(
            &runtime(),
            &model(),
            &profile("Default"),
            &empty_store(),
            &Defaults::default(),
        );
        // Enums default to "auto" (their in-band omit token)...
        for key in ["flash-attn", "reasoning"] {
            assert_eq!(value_of(&opts, key), "auto", "{key} should start at auto");
        }
        // ...numerics default to the sentinel, and reasoning-effort and
        // chat-template to their in-band "default" variant.
        for key in ["batch-size", "gpu-layers", "threads", "reasoning-effort", "chat-template"] {
            assert_eq!(value_of(&opts, key), registry::DEFAULT, "{key} should start at default");
        }
        // The valueless flags start at "on" (llama.cpp's defaults, omitted).
        for key in ["mmap", "jinja"] {
            assert_eq!(value_of(&opts, key), "on", "{key} should start at on");
        }
        // A profile that explicitly sets one still carries a concrete value.
        let server = resolve_options(
            &runtime(),
            &model(),
            &profile("Server"),
            &empty_store(),
            &Defaults::default(),
        );
        assert_eq!(value_of(&server, "flash-attn"), "on");
        assert_eq!(value_of(&server, "gpu-layers"), "999");
    }

    #[test]
    fn speculative_options_default_to_their_omit_tokens() {
        // Available for every model, defaulting to "off" (omitted from the command).
        let opts = resolve_options(
            &runtime(),
            &model(),
            &profile("Default"),
            &empty_store(),
            &Defaults::default(),
        );
        assert_eq!(value_of(&opts, "spec-type"), "none");
        assert_eq!(value_of(&opts, "spec-draft-n-max"), registry::DEFAULT);
        assert_eq!(value_of(&opts, "spec-draft-n-min"), registry::DEFAULT);
    }

    #[test]
    fn mtp_sidecar_enables_mtp_speculation_by_default() {
        let mut m = model();
        m.mtp_path = Some("/tmp/mtp-x.gguf".into());
        let opts = resolve_options(
            &runtime(),
            &m,
            &profile("Default"),
            &empty_store(),
            &Defaults::default(),
        );
        assert_eq!(value_of(&opts, "spec-type"), "draft-mtp");
    }

    #[test]
    fn integrated_mtp_head_enables_mtp_speculation_by_default() {
        let mut m = model();
        m.has_mtp = true;
        let opts = resolve_options(
            &runtime(),
            &m,
            &profile("Default"),
            &empty_store(),
            &Defaults::default(),
        );
        assert_eq!(value_of(&opts, "spec-type"), "draft-mtp");
    }

    #[test]
    fn materializing_an_instance_keeps_model_aware_defaults() {
        // Regression: editing one option materializes the instance from the
        // resolved base; that base must be model-aware, or unedited options
        // silently shift (ctx-size from ctx/8 = 32768 back to the global 4096).
        let m = model_with_ctx(Some(262144));
        let mut store = empty_store();
        let base = resolved_values(&profile("Default"), &m, &Defaults::default());
        store.set_value("llama.cpp", "/tmp/x.gguf", "Default", "temperature", "0.5".into(), &base);
        let opts =
            resolve_options(&runtime(), &m, &profile("Default"), &store, &Defaults::default());
        assert_eq!(value_of(&opts, "temperature"), "0.5");
        assert_eq!(value_of(&opts, "ctx-size"), "32768"); // still the ctx/8 default
    }

    #[test]
    fn resolved_values_clamp_template_ctx_to_the_model() {
        // The base snapshot applies the same ctx clamp as the display path:
        // Long Context's 131072 override folds to a 2048-ctx model's max.
        let m = model_with_ctx(Some(2048));
        let base = resolved_values(&profile("Long Context"), &m, &Defaults::default());
        assert_eq!(base.get("ctx-size").unwrap(), "2048");
    }

    #[test]
    fn host_port_come_from_config_defaults() {
        let defaults = Defaults { host: "0.0.0.0".into(), port: 9000 };
        let opts =
            resolve_options(&runtime(), &model(), &profile("Default"), &empty_store(), &defaults);
        assert_eq!(value_of(&opts, "host"), "0.0.0.0");
        assert_eq!(value_of(&opts, "port"), "9000");
    }
}
