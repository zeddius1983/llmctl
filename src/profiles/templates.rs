//! Built-in, global, read-only profile templates. Each overrides a subset of
//! the registry defaults; editing options for a model forks a model-scoped
//! instance (see store.rs) rather than mutating these.

/// A built-in profile template: a name plus option-value overrides.
pub struct Template {
    pub name: &'static str,
    pub overrides: &'static [(&'static str, &'static str)],
}

pub static TEMPLATES: &[Template] = &[
    Template { name: "Default", overrides: &[] },
    Template {
        name: "Chat",
        overrides: &[
            ("temperature", "0.7"),
            ("top-p", "0.9"),
            ("top-k", "40"),
            ("repeat-penalty", "1.1"),
        ],
    },
    Template {
        name: "Coding",
        overrides: &[
            ("temperature", "0.2"),
            ("top-p", "0.95"),
            ("repeat-penalty", "1.05"),
            ("ctx-size", "16384"),
        ],
    },
    Template { name: "Long Context", overrides: &[("ctx-size", "131072"), ("flash-attn", "on")] },
    Template {
        name: "Server",
        overrides: &[("host", "0.0.0.0"), ("flash-attn", "on"), ("gpu-layers", "999")],
    },
];

/// Names of all built-in templates, in display order.
pub fn names() -> impl Iterator<Item = &'static str> {
    TEMPLATES.iter().map(|t| t.name)
}

pub fn is_builtin(name: &str) -> bool {
    TEMPLATES.iter().any(|t| t.name == name)
}

pub fn find(name: &str) -> Option<&'static Template> {
    TEMPLATES.iter().find(|t| t.name == name)
}
