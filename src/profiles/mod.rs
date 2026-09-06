//! Profile/option resolution: combine a runtime's option schema, its built-in
//! templates, the config defaults, and persisted instances into the lists the
//! UI shows.
//!
//! Everything runtime-specific — which options exist, what they default to for
//! a given model, how legacy values migrate — comes from the
//! [`RuntimeBackend`]; this module only does the layering.

pub mod registry;
pub mod store;
pub mod templates;

use std::collections::BTreeMap;

use crate::config::Defaults;
use crate::domain::{Model, OptionItem, Profile};
use crate::runtime::RuntimeBackend;

pub use store::ProfileStore;

/// Profiles available for a (runtime, model): built-in templates plus any
/// user-created custom profiles, with favorite flags from the store.
pub fn list_profiles(
    backend: &dyn RuntimeBackend,
    model: &Model,
    store: &ProfileStore,
) -> Vec<Profile> {
    let runtime = &backend.descriptor().name;
    let model_key = model.profile_key();
    let mut profiles: Vec<Profile> = templates::names(backend.templates())
        .map(|name| Profile {
            name: name.to_string(),
            builtin: true,
            favorite: store.is_favorite(runtime, &model_key, name),
        })
        .collect();

    let mut custom = store.custom_profiles(runtime, &model_key);
    custom.sort();
    for name in custom {
        let favorite = store.is_favorite(runtime, &model_key, &name);
        profiles.push(Profile { name, builtin: false, favorite });
    }
    profiles
}

/// Resolve the option values for a (runtime, model, profile), layering:
/// instance override → template override → model-aware default.
pub fn resolve_options(
    backend: &dyn RuntimeBackend,
    model: &Model,
    profile: &Profile,
    store: &ProfileStore,
    defaults: &Defaults,
) -> Vec<OptionItem> {
    let model_key = model.profile_key();
    let instance = store.get(&backend.descriptor().name, &model_key, &profile.name);
    let template = templates::find(backend.templates(), &profile.name);

    backend
        .schema()
        .specs
        .iter()
        .map(|spec| {
            let default = backend.spec_default(spec, model, defaults);

            let value = instance
                .and_then(|i| i.values.get(spec.key).cloned())
                .or_else(|| instance.and_then(|i| backend.legacy_value(spec.key, &i.values)))
                .or_else(|| template.and_then(|t| t.override_value(spec.key)))
                .unwrap_or_else(|| default.clone());
            let value = backend.clamp_to_model(spec.key, value, model);
            let value = backend.normalize_legacy(spec.key, value);
            // Keep invalid saved values editable, but canonicalize valid enums
            // and numbers exactly as interactive editing does.
            let value = validate_value(backend, model, spec, &value).unwrap_or(value);

            OptionItem {
                spec,
                value,
                default,
                range: backend.effective_kind(spec, model).range_label(),
            }
        })
        .collect()
}

/// Validate resolved options at every launch boundary. Invalid saved values
/// remain visible for correction rather than silently changing the user's run.
pub fn validate_options(
    backend: &dyn RuntimeBackend,
    model: &Model,
    options: &[OptionItem],
) -> Result<(), String> {
    for option in options {
        let spec = backend
            .schema()
            .spec(option.spec.key)
            .ok_or_else(|| format!("unknown option: {}", option.spec.key))?;
        validate_value(backend, model, spec, &option.value)
            .map_err(|error| format!("{}: {error}", option.spec.key))?;
    }
    Ok(())
}

fn validate_value(
    backend: &dyn RuntimeBackend,
    model: &Model,
    spec: &registry::OptionSpec,
    value: &str,
) -> Result<String, String> {
    if let Some(token) = backend.schema().matching_omit_token(spec.key, value) {
        return Ok(token.into());
    }
    backend.effective_kind(spec, model).validate(value)
}

/// The fully-resolved current values for a (runtime, model, profile), including
/// any instance edits. Used to seed a duplicated/created profile.
pub fn current_values(
    backend: &dyn RuntimeBackend,
    model: &Model,
    profile: &Profile,
    store: &ProfileStore,
    defaults: &Defaults,
) -> BTreeMap<String, String> {
    resolve_options(backend, model, profile, store, defaults)
        .into_iter()
        .map(|o| (o.spec.key.to_string(), o.value))
        .collect()
}

/// The resolved template values for a (profile, model) — the no-instance layer
/// of [`resolve_options`]. Used as the base snapshot when materializing an
/// instance on first edit/favorite; it must match what the Options pane was
/// showing, or the first edit would silently shift unedited options (e.g.
/// ctx-size from the model-aware ctx/8 back to the global 4096).
pub fn resolved_values(
    backend: &dyn RuntimeBackend,
    profile: &Profile,
    model: &Model,
    defaults: &Defaults,
) -> BTreeMap<String, String> {
    let template = templates::find(backend.templates(), &profile.name);
    backend
        .schema()
        .specs
        .iter()
        .map(|spec| {
            let value = template
                .and_then(|t| t.override_value(spec.key))
                .unwrap_or_else(|| backend.spec_default(spec, model, defaults));
            let value = backend.clamp_to_model(spec.key, value, model);
            (spec.key.to_string(), backend.normalize_legacy(spec.key, value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use crate::profiles::registry;
    use crate::runtime::LlamaCppBackend;

    /// A llama.cpp backend with nothing discovered — enough to exercise the
    /// resolution layering, which never touches the binary.
    fn backend() -> LlamaCppBackend {
        LlamaCppBackend::discover(
            &crate::config::LlamaCppConfig { binary: "/nonexistent/llama-server".into() },
            Path::new("/nonexistent"),
        )
    }

    fn model() -> Model {
        Model {
            entry: crate::domain::CatalogEntry::Model(crate::domain::ModelSource::Gguf {
                remote: None,
            }),
            id: "test-model".into(),
            name: "x.gguf".into(),
            path: "/tmp/x.gguf".into(),
            shard_paths: vec!["/tmp/x.gguf".into()],
            mtp_path: None,
            dflash_path: None,
            dflash_block_size: None,
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
            runtime: crate::runtime::llama_cpp::NAME.into(),
        }
    }

    fn profile(name: &str) -> Profile {
        Profile { name: name.into(), builtin: true, favorite: false }
    }

    fn empty_store() -> ProfileStore {
        ProfileStore::load("/nonexistent/llmctl-test-store.json".into(), &[])
    }

    fn value_of(opts: &[OptionItem], key: &str) -> String {
        opts.iter().find(|o| o.spec.key == key).unwrap().value.clone()
    }

    #[test]
    fn invalid_saved_values_stay_editable_but_cannot_launch() {
        let backend = backend();
        let model = model();
        for (key, value) in [
            ("temperature", "NaN"),
            ("port", "70000"),
            ("top-k", "-2"),
            ("flash-attn", "sometimes"),
        ] {
            let mut store = empty_store();
            assert!(
                store
                    .set_value(
                        "llama.cpp",
                        &model.profile_key(),
                        "Default",
                        key,
                        value.into(),
                        &BTreeMap::new(),
                    )
                    .is_err()
            );
            let options = resolve_options(
                &backend,
                &model,
                &profile("Default"),
                &store,
                &Defaults::default(),
            );
            assert_eq!(value_of(&options, key), value);
            assert!(validate_options(&backend, &model, &options).unwrap_err().starts_with(key));
        }
    }

    #[test]
    fn saved_sentinels_and_legacy_enums_are_validated_after_normalization() {
        let backend = backend();
        let model = model();
        let mut store = empty_store();
        for (key, value) in
            [("temperature", " DEFAULT "), ("flash-attn", "true"), ("reasoning", "OFF")]
        {
            assert!(
                store
                    .set_value(
                        "llama.cpp",
                        &model.profile_key(),
                        "Default",
                        key,
                        value.into(),
                        &BTreeMap::new(),
                    )
                    .is_err()
            );
        }
        let options =
            resolve_options(&backend, &model, &profile("Default"), &store, &Defaults::default());
        validate_options(&backend, &model, &options).unwrap();
        assert_eq!(value_of(&options, "temperature"), "default");
        assert_eq!(value_of(&options, "flash-attn"), "on");
        assert_eq!(value_of(&options, "reasoning"), "off");
    }

    #[test]
    fn model_context_does_not_wrap_into_a_negative_bound() {
        let backend = backend();
        let model = model_with_ctx(Some(u64::MAX));
        let kind = backend.effective_kind(backend.schema().spec("ctx-size").unwrap(), &model);
        assert!(kind.validate("4096").is_ok());
        assert_eq!(backend.clamp_to_model("ctx-size", "4096".into(), &model), "4096");
    }

    #[test]
    fn default_profile_uses_registry_defaults() {
        let opts = resolve_options(
            &backend(),
            &model(),
            &profile("Default"),
            &empty_store(),
            &Defaults::default(),
        );
        // ctx-size starts concrete (the ctx/8 heuristic / registry fallback),
        // never at 'default' (= the model's full context).
        assert_eq!(value_of(&opts, "ctx-size"), "4096");
        // Sampling params start omitted — llama.cpp's own defaults apply.
        for key in ["temperature", "top-p", "top-k", "min-p", "repeat-penalty", "presence-penalty"]
        {
            assert_eq!(value_of(&opts, key), registry::DEFAULT, "{key} should start at default");
        }
    }

    #[test]
    fn template_overrides_apply() {
        let opts = resolve_options(
            &backend(),
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
        let backend = backend();
        let spec = backend.schema().spec("ctx-size").unwrap();
        let kind = backend.effective_kind(spec, &m);
        assert_eq!(kind.extreme(1), Some("8192".into())); // End → model max
        assert_eq!(kind.adjust("8192", 1, spec.step), Some("8192".into())); // clamps
    }

    #[test]
    fn ctx_size_default_is_model_context_over_eight() {
        // A model with a known context defaults ctx-size to ctx / 8.
        let m = model_with_ctx(Some(32768));
        let opts = resolve_options(
            &backend(),
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
            &backend(),
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
            &backend(),
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
        // The valueless flag starts at "on" (llama.cpp's default, omitted), and
        // load-mode at its in-band "default" variant.
        assert_eq!(value_of(&opts, "jinja"), "on");
        assert_eq!(value_of(&opts, "load-mode"), registry::DEFAULT);
        // A profile that explicitly sets one still carries a concrete value.
        let server = resolve_options(
            &backend(),
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
            &backend(),
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
            &backend(),
            &m,
            &profile("Default"),
            &empty_store(),
            &Defaults::default(),
        );
        assert_eq!(value_of(&opts, "spec-type"), "draft-mtp");
    }

    #[test]
    fn dflash_sidecar_enables_dflash_speculation_by_default() {
        let mut m = model();
        m.dflash_path = Some("/tmp/dflash-kquant.gguf".into());

        // Only for a binary that accepts the spec type: the discovered backend
        // here has no llama-server at all, so the default stays undrafted
        // rather than resolving to a launch it could not run.
        let mut backend = backend();
        assert!(!backend.dflash_supported);
        let undrafted = resolve_options(
            &backend,
            &m,
            &profile("Default"),
            &empty_store(),
            &Defaults::default(),
        );
        assert_eq!(value_of(&undrafted, "spec-type"), "none");

        backend.dflash_supported = true;
        let opts = resolve_options(
            &backend,
            &m,
            &profile("Default"),
            &empty_store(),
            &Defaults::default(),
        );
        assert_eq!(value_of(&opts, "spec-type"), "draft-dflash");
    }

    #[test]
    fn integrated_mtp_head_enables_mtp_speculation_by_default() {
        let mut m = model();
        m.has_mtp = true;
        let opts = resolve_options(
            &backend(),
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
        let base = resolved_values(&backend(), &profile("Default"), &m, &Defaults::default());
        assert!(
            store
                .set_value(
                    "llama.cpp",
                    "/tmp/x.gguf",
                    "Default",
                    "temperature",
                    "0.5".into(),
                    &base
                )
                .is_err()
        );
        let opts =
            resolve_options(&backend(), &m, &profile("Default"), &store, &Defaults::default());
        assert_eq!(value_of(&opts, "temperature"), "0.5");
        assert_eq!(value_of(&opts, "ctx-size"), "32768"); // still the ctx/8 default
    }

    #[test]
    fn resolved_values_clamp_template_ctx_to_the_model() {
        // The base snapshot applies the same ctx clamp as the display path:
        // Long Context's 131072 override folds to a 2048-ctx model's max.
        let m = model_with_ctx(Some(2048));
        let base = resolved_values(&backend(), &profile("Long Context"), &m, &Defaults::default());
        assert_eq!(base.get("ctx-size").unwrap(), "2048");
    }

    #[test]
    fn host_port_come_from_config_defaults() {
        let defaults = Defaults { host: "0.0.0.0".into(), port: 9000 };
        let opts =
            resolve_options(&backend(), &model(), &profile("Default"), &empty_store(), &defaults);
        assert_eq!(value_of(&opts, "host"), "0.0.0.0");
        assert_eq!(value_of(&opts, "port"), "9000");
    }
}
