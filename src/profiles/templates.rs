//! Built-in, read-only profile templates. Each overrides a subset of its
//! runtime's option defaults; editing options for a model forks a model-scoped
//! instance (see store.rs) rather than mutating these.
//!
//! The tables themselves are per-runtime — a template speaks in option keys,
//! and llama.cpp's `gpu-layers`/`flash-attn` mean nothing to an NPU runtime —
//! so they live with their backends in `crate::runtime`.

/// A built-in profile template: a name plus option-value overrides.
pub struct Template {
    pub name: &'static str,
    pub overrides: &'static [(&'static str, &'static str)],
}

impl Template {
    /// This template's override for `key`, if it sets one.
    pub fn override_value(&self, key: &str) -> Option<String> {
        self.overrides.iter().find(|(k, _)| *k == key).map(|(_, v)| v.to_string())
    }
}

/// Names of the given built-in templates, in display order.
pub fn names(templates: &'static [Template]) -> impl Iterator<Item = &'static str> {
    templates.iter().map(|t| t.name)
}

pub fn is_builtin(templates: &'static [Template], name: &str) -> bool {
    templates.iter().any(|t| t.name == name)
}

pub fn find(templates: &'static [Template], name: &str) -> Option<&'static Template> {
    templates.iter().find(|t| t.name == name)
}
