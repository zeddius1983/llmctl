//! FastFlowLM-specific launch options and built-in profile templates.

use super::registry::{OptionKind, OptionSpec};

use OptionKind::{Enum, Int, Str};

pub const DEFAULT: &str = "default";

pub static REGISTRY: &[OptionSpec] = &[
    OptionSpec {
        key: "ctx-len",
        cli: "--ctx-len",
        kind: Int { min: Some(512), max: None },
        default: "4096",
        step: 1024.0,
        description: "Context length in tokens. FastFlowLM requires at least 512 and rounds to a power of two; 'default' uses the catalogue default.",
    },
    OptionSpec {
        key: "pmode",
        cli: "--pmode",
        kind: Enum(&["powersaver", "balanced", "performance", "turbo"]),
        default: "performance",
        step: 1.0,
        description: "AMD NPU power mode (performance is FastFlowLM's default).",
    },
    OptionSpec {
        key: "prefill-chunk-len",
        cli: "--prefill-chunk-len",
        kind: Int { min: Some(512), max: None },
        default: "4096",
        step: 512.0,
        description: "Tokens processed per prefill chunk; 'default' lets FastFlowLM choose.",
    },
    OptionSpec {
        key: "img-pre-resize",
        cli: "--img-pre-resize",
        kind: Enum(&["default", "0", "1", "2", "3", "4"]),
        default: "default",
        step: 1.0,
        description: "Vision input resize: 0 original, 1 480p, 2 720p, 3 1080p, 4 1440p; default omits the flag.",
    },
    OptionSpec {
        key: "host",
        cli: "--host",
        kind: Str,
        default: "127.0.0.1",
        step: 0.0,
        description: "Network interface to bind the FastFlowLM server to.",
    },
    OptionSpec {
        key: "port",
        cli: "--port",
        kind: Int { min: Some(1), max: Some(65535) },
        default: "52625",
        step: 1.0,
        description: "TCP port for the OpenAI-compatible FastFlowLM server.",
    },
    OptionSpec {
        key: "q-len",
        cli: "--q-len",
        kind: Int { min: Some(1), max: None },
        default: "10",
        step: 1.0,
        description: "Maximum queued NPU requests.",
    },
    OptionSpec {
        key: "socket",
        cli: "--socket",
        kind: Int { min: Some(1), max: None },
        default: "10",
        step: 1.0,
        description: "Maximum concurrent socket connections; normally at least q-len.",
    },
    OptionSpec {
        key: "cors",
        cli: "--cors",
        kind: Enum(&["1", "0"]),
        default: "1",
        step: 1.0,
        description: "Cross-origin requests: 1 enabled, 0 disabled.",
    },
];

pub fn spec(key: &str) -> Option<&'static OptionSpec> {
    REGISTRY.iter().find(|spec| spec.key == key)
}

pub fn omit_token(key: &str) -> Option<&'static str> {
    match key {
        "ctx-len" | "prefill-chunk-len" | "img-pre-resize" => Some(DEFAULT),
        "pmode" => Some("performance"),
        "q-len" | "socket" => Some("10"),
        "cors" => Some("1"),
        _ => None,
    }
}

pub struct Template {
    pub name: &'static str,
    pub overrides: &'static [(&'static str, &'static str)],
}

pub static TEMPLATES: &[Template] = &[
    Template { name: "Default", overrides: &[] },
    Template { name: "Power Saver", overrides: &[("pmode", "powersaver")] },
    Template { name: "Balanced", overrides: &[("pmode", "balanced")] },
    Template { name: "Turbo", overrides: &[("pmode", "turbo")] },
    Template { name: "Server", overrides: &[("host", "0.0.0.0"), ("cors", "0")] },
];

pub fn names() -> impl Iterator<Item = &'static str> {
    TEMPLATES.iter().map(|template| template.name)
}

pub fn is_builtin(name: &str) -> bool {
    TEMPLATES.iter().any(|template| template.name == name)
}

pub fn override_value(profile: &str, key: &str) -> Option<String> {
    TEMPLATES
        .iter()
        .find(|template| template.name == profile)?
        .overrides
        .iter()
        .find(|(candidate, _)| *candidate == key)
        .map(|(_, value)| (*value).to_string())
}
