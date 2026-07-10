//! Built-in, global, read-only profile templates. Each overrides a subset of
//! the registry defaults; editing options for a model forks a model-scoped
//! instance (see store.rs) rather than mutating these.

use crate::domain::RuntimeId;

/// A built-in profile template: a name plus option-value overrides.
pub struct Template {
    pub name: &'static str,
    pub overrides: &'static [(&'static str, &'static str)],
}

pub static LLAMA_CPP_TEMPLATES: &[Template] = &[
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

pub static VLLM_TEMPLATES: &[Template] = &[
    Template { name: "Default", overrides: &[] },
    Template {
        name: "Throughput",
        overrides: &[("gpu-memory-utilization", "0.95"), ("enable-prefix-caching", "on")],
    },
    Template {
        name: "Low Memory",
        overrides: &[
            ("gpu-memory-utilization", "0.8"),
            ("max-model-len", "4096"),
            ("cpu-offload-gb", "4"),
        ],
    },
    Template {
        name: "Multi-GPU",
        overrides: &[("tensor-parallel-size", "2"), ("enable-prefix-caching", "on")],
    },
    Template { name: "Compatibility", overrides: &[("enforce-eager", "on")] },
];

pub fn all(runtime: RuntimeId) -> &'static [Template] {
    match runtime {
        RuntimeId::LlamaCpp => LLAMA_CPP_TEMPLATES,
        RuntimeId::Vllm => VLLM_TEMPLATES,
    }
}

/// Names of all built-in templates, in display order.
pub fn names(runtime: RuntimeId) -> impl Iterator<Item = &'static str> {
    all(runtime).iter().map(|t| t.name)
}

pub fn is_builtin(runtime: RuntimeId, name: &str) -> bool {
    all(runtime).iter().any(|t| t.name == name)
}

pub fn find(runtime: RuntimeId, name: &str) -> Option<&'static Template> {
    all(runtime).iter().find(|t| t.name == name)
}
