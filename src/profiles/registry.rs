//! Static registry of launch options: the source of truth for which options
//! exist and their type, default, range, step, CLI flag, and description. Used
//! to render the Options pane, validate edits, and drive inline adjustment.

/// The kind/domain of an option value, used for validation and adjustment.
#[derive(Debug, Clone, Copy)]
pub enum OptionKind {
    Int { min: Option<i64>, max: Option<i64> },
    Float { min: Option<f64>, max: Option<f64> },
    Enum(&'static [&'static str]),
    Str,
}

/// Metadata for a single option.
#[derive(Debug, Clone, Copy)]
pub struct OptionSpec {
    pub key: &'static str,
    pub cli: &'static str,
    pub kind: OptionKind,
    pub default: &'static str,
    /// Increment used by `+`/`-`/`[`/`]` (numeric kinds only).
    pub step: f64,
    pub description: &'static str,
}

impl OptionKind {
    /// Human-readable allowed range for the Info pane (None for free-form).
    pub fn range_label(&self) -> Option<String> {
        match self {
            OptionKind::Int { min, max } => Some(int_range(*min, *max)),
            OptionKind::Float { min, max } => Some(float_range(*min, *max)),
            OptionKind::Enum(variants) => Some(variants.join(" | ")),
            OptionKind::Str => None,
        }
    }

    /// Validate and normalize a user-entered value, or return an error message.
    pub fn validate(&self, input: &str) -> Result<String, String> {
        let input = input.trim();
        match self {
            OptionKind::Int { min, max } => {
                let v: i64 = input.parse().map_err(|_| format!("'{input}' is not an integer"))?;
                check_bound(v as f64, min.map(|m| m as f64), max.map(|m| m as f64))?;
                Ok(v.to_string())
            }
            OptionKind::Float { min, max } => {
                let v: f64 = input.parse().map_err(|_| format!("'{input}' is not a number"))?;
                check_bound(v, *min, *max)?;
                Ok(input.to_string())
            }
            OptionKind::Enum(variants) => variants
                .iter()
                .find(|v| v.eq_ignore_ascii_case(input))
                .map(|v| (*v).to_string())
                .ok_or_else(|| format!("expected one of: {}", variants.join(", "))),
            OptionKind::Str => {
                if input.is_empty() {
                    Err("value cannot be empty".into())
                } else {
                    Ok(input.to_string())
                }
            }
        }
    }

    /// Increment (`dir = +1`) or decrement (`dir = -1`) the current value.
    /// Numeric kinds clamp at their bounds; bool/enum cycle (wrap).
    pub fn adjust(&self, current: &str, dir: i32, step: f64) -> Option<String> {
        match self {
            OptionKind::Int { min, max } => {
                let cur: i64 = current.parse().ok()?;
                let mut v = cur + dir as i64 * (step.round() as i64).max(1);
                if let Some(lo) = min {
                    v = v.max(*lo);
                }
                if let Some(hi) = max {
                    v = v.min(*hi);
                }
                Some(v.to_string())
            }
            OptionKind::Float { min, max } => {
                let cur: f64 = current.parse().ok()?;
                let mut v = cur + dir as f64 * step;
                if let Some(lo) = min {
                    v = v.max(*lo);
                }
                if let Some(hi) = max {
                    v = v.min(*hi);
                }
                Some(fmt_float(v))
            }
            OptionKind::Enum(variants) => {
                let idx = variants.iter().position(|v| *v == current).unwrap_or(0) as i32;
                let n = variants.len() as i32;
                let next = (idx + dir).rem_euclid(n) as usize;
                Some(variants[next].to_string())
            }
            OptionKind::Str => None,
        }
    }

    /// Jump to the minimum (`dir = -1`) or maximum (`dir = +1`) — Home/End.
    /// Sentinel-aware stepping lives on [`OptionSpec::bump`]; resetting to the
    /// default is the `d` key (app-level), not a jump.
    pub fn extreme(&self, dir: i32) -> Option<String> {
        match self {
            OptionKind::Int { min, max } => {
                if dir < 0 { *min } else { *max }.map(|v| v.to_string())
            }
            OptionKind::Float { min, max } => if dir < 0 { *min } else { *max }.map(fmt_float),
            OptionKind::Enum(variants) => {
                if dir < 0 { variants.first() } else { variants.last() }.map(|v| (*v).to_string())
            }
            OptionKind::Str => None,
        }
    }
}

/// Sentinel value (for options with no in-band "auto") meaning "leave this flag
/// off the command line and rely on llama.cpp's own built-in default".
pub const DEFAULT: &str = "default";

/// The value at which an option is dropped from the launch command, because it
/// equals what llama.cpp would do anyway. For on/off/auto enums that's `"auto"`
/// (llama's own default); enums that carry an explicit `"default"` variant
/// (e.g. the cache types) omit at that variant; for numeric options with no
/// in-band sentinel it's the [`DEFAULT`] sentinel. `None` means always emitted.
pub fn omit_token(key: &str) -> Option<&'static str> {
    match key {
        "flash-attn" | "reasoning" => Some("auto"),
        // `mmap=on` is llama.cpp's default (omitted); `off` adds the bare
        // `--no-mmap` flag (see [`is_flag`]).
        "mmap" => Some("on"),
        // Speculative decoding is off by default.
        "spec-type" => Some("none"),
        // `jinja=on` is llama.cpp's default (omitted); `off` adds the bare
        // `--no-jinja` flag (see [`is_flag`]).
        "jinja" => Some("on"),
        "batch-size" | "device" | "gpu-layers" | "threads" | "cache-type-k" | "cache-type-v"
        | "spec-draft-n-max" | "spec-draft-n-min" | "reasoning-effort" | "chat-template"
        | "ctx-size" | "temperature" | "top-p" | "top-k" | "min-p" | "repeat-penalty" => {
            Some(DEFAULT)
        }
        // host/port are never omitted: llmctl itself needs the concrete
        // endpoint for health checks and the Session Manager display.
        _ => None,
    }
}

/// Whether the option is a valueless boolean flag (e.g. `mmap` → `--no-mmap`):
/// when not at its [`omit_token`] it emits the bare flag with no value token.
pub fn is_flag(key: &str) -> bool {
    matches!(key, "mmap" | "jinja")
}

/// The value token actually emitted on the command line. Most options pass
/// their value through verbatim; `reasoning-effort` has no native llama-server
/// flag and is delivered to the chat template as a JSON kwarg via
/// `--chat-template-kwargs` (how GPT-OSS-style templates receive it).
pub fn cli_value(key: &str, value: &str) -> String {
    match key {
        "reasoning-effort" => format!(r#"{{"reasoning_effort":"{value}"}}"#),
        _ => value.to_string(),
    }
}

/// Whether the option's omitted state is the [`DEFAULT`] sentinel (vs an in-band
/// enum variant like `"auto"` or an enum's own `"default"` choice). Only these
/// get the sentinel editing affordances (the `default` text entry); enums cycle
/// through their variants instead.
pub fn uses_sentinel(key: &str) -> bool {
    omit_token(key) == Some(DEFAULT)
        && !matches!(spec(key).map(|s| s.kind), Some(OptionKind::Enum(_)))
}

impl OptionSpec {
    /// Step the value by one increment (`dir = ±1`) for `+`/`-` and the `e`
    /// cycle. For sentinel options [`DEFAULT`] sits just below the numeric range:
    /// stepping up from it enters the concrete default; enums (whose omitted
    /// state is an ordinary `"auto"` variant) just cycle normally.
    pub fn bump(&self, kind: &OptionKind, current: &str, dir: i32) -> Option<String> {
        if uses_sentinel(self.key) && current == DEFAULT {
            return Some(if dir > 0 { self.default.to_string() } else { DEFAULT.to_string() });
        }
        kind.adjust(current, dir, self.step)
    }
}

fn check_bound(v: f64, min: Option<f64>, max: Option<f64>) -> Result<(), String> {
    if let Some(lo) = min {
        if v < lo {
            return Err(format!("must be ≥ {lo}"));
        }
    }
    if let Some(hi) = max {
        if v > hi {
            return Err(format!("must be ≤ {hi}"));
        }
    }
    Ok(())
}

fn int_range(min: Option<i64>, max: Option<i64>) -> String {
    match (min, max) {
        (Some(lo), Some(hi)) => format!("{lo} – {hi}"),
        (Some(lo), None) => format!("≥ {lo}"),
        (None, Some(hi)) => format!("≤ {hi}"),
        (None, None) => "integer".into(),
    }
}

fn float_range(min: Option<f64>, max: Option<f64>) -> String {
    match (min, max) {
        (Some(lo), Some(hi)) => format!("{lo} – {hi}"),
        (Some(lo), None) => format!("≥ {lo}"),
        (None, Some(hi)) => format!("≤ {hi}"),
        (None, None) => "number".into(),
    }
}

/// Format a float compactly: up to 3 decimals, trailing zeros trimmed.
fn fmt_float(v: f64) -> String {
    let s = format!("{v:.3}");
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string()
}

use OptionKind::{Enum, Float, Int, Str};

/// Built-in chat template names accepted by `--chat-template` (from
/// `llama-server --help`), with a leading `"default"` omit variant meaning
/// "use the template from the model's GGUF metadata".
static CHAT_TEMPLATES: &[&str] = &[
    "default",
    "bailing",
    "bailing-think",
    "bailing2",
    "chatglm3",
    "chatglm4",
    "chatml",
    "command-r",
    "deepseek",
    "deepseek-ocr",
    "deepseek2",
    "deepseek3",
    "exaone-moe",
    "exaone3",
    "exaone4",
    "falcon3",
    "gemma",
    "gigachat",
    "glmedge",
    "gpt-oss",
    "granite",
    "granite-4.0",
    "granite-4.1",
    "grok-2",
    "hunyuan-dense",
    "hunyuan-moe",
    "hunyuan-vl",
    "kimi-k2",
    "llama2",
    "llama2-sys",
    "llama2-sys-bos",
    "llama2-sys-strip",
    "llama3",
    "llama4",
    "megrez",
    "minicpm",
    "mistral-v1",
    "mistral-v3",
    "mistral-v3-tekken",
    "mistral-v7",
    "mistral-v7-tekken",
    "monarch",
    "openchat",
    "orion",
    "pangu-embedded",
    "phi3",
    "phi4",
    "rwkv-world",
    "seed_oss",
    "smolvlm",
    "solar-open",
    "vicuna",
    "vicuna-orca",
    "yandex",
    "zephyr",
];

/// The MVP option set for llama-server.
pub static REGISTRY: &[OptionSpec] = &[
    OptionSpec {
        key: "ctx-size",
        cli: "--ctx-size",
        kind: Int { min: Some(0), max: None },
        default: "4096",
        step: 1024.0,
        description: "Maximum context window size in tokens (0 or 'default' = the model's \
                      full trained context — watch your memory).",
    },
    OptionSpec {
        key: "gpu-layers",
        cli: "-ngl",
        kind: Int { min: Some(0), max: Some(999) },
        default: "999",
        step: 1.0,
        description: "Layers to offload to the GPU (999 = all; 'default' lets llama.cpp decide).",
    },
    OptionSpec {
        key: "device",
        cli: "--device",
        kind: Str,
        default: "default",
        step: 0.0,
        description: "Device to use for offloading, selected from llama-server --list-devices \
                      ('default' lets llama.cpp choose).",
    },
    OptionSpec {
        key: "temperature",
        cli: "--temp",
        kind: Float { min: Some(0.0), max: Some(2.0) },
        default: "0.8",
        step: 0.05,
        description: "Sampling temperature; lower is more deterministic \
                      ('default' = llama.cpp's 0.8).",
    },
    OptionSpec {
        key: "top-p",
        cli: "--top-p",
        kind: Float { min: Some(0.0), max: Some(1.0) },
        default: "0.95",
        step: 0.05,
        description: "Nucleus sampling: keep tokens within this cumulative probability \
                      ('default' = llama.cpp's 0.95).",
    },
    OptionSpec {
        key: "top-k",
        cli: "--top-k",
        kind: Int { min: Some(0), max: None },
        default: "40",
        step: 1.0,
        description: "Keep only the top-K most likely tokens \
                      (0 = disabled; 'default' = llama.cpp's 40).",
    },
    OptionSpec {
        key: "min-p",
        cli: "--min-p",
        kind: Float { min: Some(0.0), max: Some(1.0) },
        default: "0.05",
        step: 0.01,
        description: "Minimum token probability relative to the most likely token \
                      ('default' = llama.cpp's 0.05).",
    },
    OptionSpec {
        key: "repeat-penalty",
        cli: "--repeat-penalty",
        kind: Float { min: Some(0.0), max: Some(2.0) },
        default: "1.0",
        step: 0.05,
        description: "Penalty applied to repeated tokens \
                      (1.0 = disabled; 'default' = llama.cpp's 1.0).",
    },
    OptionSpec {
        key: "threads",
        cli: "--threads",
        kind: Int { min: Some(0), max: None },
        default: "0",
        step: 1.0,
        description: "CPU threads for generation ('default' lets llama.cpp auto-detect, i.e. -1).",
    },
    OptionSpec {
        key: "batch-size",
        cli: "--batch-size",
        kind: Int { min: Some(1), max: None },
        default: "2048",
        step: 256.0,
        description: "Logical batch size for prompt processing ('default' = llama.cpp's 2048).",
    },
    OptionSpec {
        key: "flash-attn",
        cli: "--flash-attn",
        kind: Enum(&["auto", "on", "off"]),
        default: "auto",
        step: 1.0,
        description: "Flash attention (auto = llama.cpp default; omitted from command).",
    },
    OptionSpec {
        key: "reasoning",
        cli: "--reasoning",
        kind: Enum(&["auto", "on", "off"]),
        default: "auto",
        step: 1.0,
        description: "Reasoning/thinking in chat (auto = llama.cpp default; omitted from command).",
    },
    OptionSpec {
        key: "reasoning-effort",
        cli: "--chat-template-kwargs",
        kind: Enum(&["default", "low", "medium", "high"]),
        default: "default",
        step: 1.0,
        description: "Reasoning effort passed to the chat template as \
                      {\"reasoning_effort\": …} (GPT-OSS-style models; \
                      default = omitted).",
    },
    OptionSpec {
        key: "chat-template",
        cli: "--chat-template",
        kind: Enum(CHAT_TEMPLATES),
        default: "default",
        step: 1.0,
        description: "Override the chat template with a llama.cpp built-in \
                      (default = use the template from the model's GGUF metadata).",
    },
    OptionSpec {
        key: "jinja",
        cli: "--no-jinja",
        kind: Enum(&["on", "off"]),
        default: "on",
        step: 1.0,
        description: "Jinja chat template engine (on = llama.cpp default; turn off to \
                      add --no-jinja for legacy formatting — disables tool calls and \
                      reasoning-effort).",
    },
    OptionSpec {
        key: "mmap",
        cli: "--no-mmap",
        kind: Enum(&["on", "off"]),
        default: "on",
        step: 1.0,
        description: "Memory-map the model (on = llama.cpp default; turn off to add \
                      --no-mmap for ROCm/AMD GPU compatibility).",
    },
    OptionSpec {
        key: "cache-type-k",
        cli: "--cache-type-k",
        kind: Enum(&["default", "f16", "q8_0", "q4_0"]),
        default: "default",
        step: 1.0,
        description: "KV cache data type for keys (default = llama.cpp default; \
                      lower precision = less memory).",
    },
    OptionSpec {
        key: "cache-type-v",
        cli: "--cache-type-v",
        kind: Enum(&["default", "f16", "q8_0", "q4_0"]),
        default: "default",
        step: 1.0,
        description: "KV cache data type for values (default = llama.cpp default; \
                      lower precision = less memory).",
    },
    OptionSpec {
        key: "spec-type",
        cli: "--spec-type",
        kind: Enum(&[
            "none",
            "draft-simple",
            "draft-eagle3",
            "draft-mtp",
            "ngram-simple",
            "ngram-map-k",
            "ngram-map-k4v",
            "ngram-mod",
            "ngram-cache",
        ]),
        default: "none",
        step: 1.0,
        description: "Speculative decoding type (none = disabled; draft-mtp uses the model's \
                      built-in MTP head).",
    },
    OptionSpec {
        key: "spec-draft-n-max",
        cli: "--spec-draft-n-max",
        kind: Int { min: Some(0), max: None },
        default: "3",
        step: 1.0,
        description: "Max tokens to draft per step for speculative decoding \
                      ('default' = llama.cpp's 3).",
    },
    OptionSpec {
        key: "spec-draft-n-min",
        cli: "--spec-draft-n-min",
        kind: Int { min: Some(0), max: None },
        default: "0",
        step: 1.0,
        description: "Min draft tokens for speculative decoding ('default' = llama.cpp's 0).",
    },
    OptionSpec {
        key: "host",
        cli: "--host",
        kind: Str,
        default: "127.0.0.1",
        step: 0.0,
        description: "Network interface to bind the server to.",
    },
    OptionSpec {
        key: "port",
        cli: "--port",
        kind: Int { min: Some(1), max: Some(65535) },
        default: "8000",
        step: 1.0,
        description: "TCP port the server listens on.",
    },
];

/// Look up an option spec by key.
pub fn spec(key: &str) -> Option<&'static OptionSpec> {
    REGISTRY.iter().find(|s| s.key == key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_range_is_enforced() {
        let kind = spec("gpu-layers").unwrap().kind;
        assert_eq!(kind.validate("50").unwrap(), "50");
        assert!(kind.validate("1000").is_err()); // > 999
        assert!(kind.validate("-1").is_err()); // < 0
        assert!(kind.validate("abc").is_err());
    }

    #[test]
    fn float_range_is_enforced() {
        let kind = spec("temperature").unwrap().kind;
        assert_eq!(kind.validate("0.7").unwrap(), "0.7");
        assert!(kind.validate("3.0").is_err()); // > 2.0
    }

    #[test]
    fn flash_attn_is_an_enum_dropped_when_auto() {
        let spec = spec("flash-attn").unwrap();
        assert_eq!(spec.kind.validate("OFF").unwrap(), "off");
        assert!(spec.kind.validate("true").is_err()); // legacy bool is not a variant
        // "auto" is the omitted state; it cycles like any variant (no sentinel).
        assert_eq!(omit_token("flash-attn"), Some("auto"));
        assert_eq!(spec.bump(&spec.kind, "auto", 1), Some("on".into()));
        assert_eq!(spec.kind.extreme(-1), Some("auto".into())); // Home → first variant
    }

    #[test]
    fn numeric_omittables_fold_the_default_sentinel() {
        assert_eq!(omit_token("batch-size"), Some(DEFAULT));
        assert_eq!(omit_token("threads"), Some(DEFAULT));
        // The sampling params and ctx-size are omittable too.
        for key in ["ctx-size", "temperature", "top-p", "top-k", "min-p", "repeat-penalty"] {
            assert_eq!(omit_token(key), Some(DEFAULT), "{key} should fold the sentinel");
            assert!(uses_sentinel(key), "{key} should get sentinel affordances");
        }
        // host/port stay on the command line: llmctl needs the endpoint.
        assert_eq!(omit_token("host"), None);
        assert_eq!(omit_token("port"), None);

        // Stepping up from DEFAULT enters the concrete base; stepping down stays.
        let ngl = spec("gpu-layers").unwrap();
        assert_eq!(ngl.bump(&ngl.kind, DEFAULT, 1), Some("999".into()));
        assert_eq!(ngl.bump(&ngl.kind, DEFAULT, -1), Some(DEFAULT.into()));
        // Home/End are pure min/max jumps; resetting to DEFAULT is `d` (app-level).
        assert_eq!(ngl.kind.extreme(-1), Some("0".into())); // Home → min
        assert_eq!(ngl.kind.extreme(1), Some("999".into())); // End → max
    }

    #[test]
    fn device_uses_runtime_selector_and_is_omitted_at_default() {
        let device = spec("device").unwrap();
        assert_eq!(device.cli, "--device");
        assert_eq!(omit_token("device"), Some(DEFAULT));
        assert!(uses_sentinel("device"));
    }

    #[test]
    fn adjust_clamps_numeric_and_cycles_enum() {
        let temp = spec("temperature").unwrap();
        assert_eq!(temp.kind.adjust("1.95", 1, temp.step), Some("2".into())); // clamp at 2.0
        assert_eq!(temp.kind.adjust("0.8", -1, temp.step), Some("0.75".into()));

        let cache = spec("cache-type-k").unwrap().kind;
        assert_eq!(cache.adjust("f16", 1, 1.0), Some("q8_0".into()));
        assert_eq!(cache.adjust("f16", -1, 1.0), Some("default".into())); // back toward "default"
    }

    #[test]
    fn cache_types_omit_at_their_default_variant_without_sentinel_affordances() {
        for key in ["cache-type-k", "cache-type-v"] {
            // "default" is the omitted state, but it's an in-band enum variant —
            // not the numeric sentinel — so it cycles like any other choice.
            assert_eq!(omit_token(key), Some(DEFAULT));
            assert!(!uses_sentinel(key));
            let s = spec(key).unwrap();
            assert_eq!(s.bump(&s.kind, "default", 1), Some("f16".into()));
            assert_eq!(s.kind.extreme(-1), Some("default".into())); // Home → first variant
            assert_eq!(s.kind.extreme(1), Some("q4_0".into())); // End → last
        }
    }

    #[test]
    fn speculative_options_have_proper_omit_tokens() {
        // spec-type omits at its in-band "none" variant (cycles like an enum).
        assert_eq!(omit_token("spec-type"), Some("none"));
        assert!(!uses_sentinel("spec-type"));
        let st = spec("spec-type").unwrap();
        assert_eq!(st.bump(&st.kind, "none", 1), Some("draft-simple".into()));

        // The draft-count ints fold the numeric "default" sentinel.
        let n_max = spec("spec-draft-n-max").unwrap();
        assert_eq!(omit_token("spec-draft-n-max"), Some(DEFAULT));
        assert!(uses_sentinel("spec-draft-n-max"));
        assert_eq!(n_max.bump(&n_max.kind, DEFAULT, 1), Some("3".into())); // step up enters base
        assert_eq!(spec("spec-draft-n-min").unwrap().default, "0");
    }

    #[test]
    fn reasoning_effort_is_a_json_kwarg_enum_omitted_at_default() {
        // "default" is the omitted state, an in-band enum variant (no sentinel
        // affordances) — it cycles like the cache types.
        assert_eq!(omit_token("reasoning-effort"), Some(DEFAULT));
        assert!(!uses_sentinel("reasoning-effort"));
        let s = spec("reasoning-effort").unwrap();
        assert_eq!(s.bump(&s.kind, "default", 1), Some("low".into()));
        assert_eq!(s.kind.extreme(1), Some("high".into())); // End → high

        // The emitted argv token is the chat-template kwargs JSON, not the raw value.
        assert_eq!(cli_value("reasoning-effort", "high"), r#"{"reasoning_effort":"high"}"#);
        assert_eq!(cli_value("temperature", "0.7"), "0.7"); // everything else passes through
    }

    #[test]
    fn mmap_is_a_flag_omitted_when_on() {
        assert!(is_flag("mmap"));
        assert_eq!(omit_token("mmap"), Some("on")); // on = llama default, omitted
        let s = spec("mmap").unwrap();
        assert_eq!(s.bump(&s.kind, "on", 1), Some("off".into())); // `e` toggles
    }

    #[test]
    fn jinja_is_a_flag_omitted_when_on() {
        // Same shape as mmap: on = llama.cpp's default (omitted); off emits
        // the bare --no-jinja flag.
        assert!(is_flag("jinja"));
        assert_eq!(omit_token("jinja"), Some("on"));
        let s = spec("jinja").unwrap();
        assert_eq!(s.cli, "--no-jinja");
        assert_eq!(s.bump(&s.kind, "on", 1), Some("off".into())); // `e` toggles
    }

    #[test]
    fn chat_template_is_an_enum_of_builtins_omitted_at_default() {
        assert_eq!(omit_token("chat-template"), Some(DEFAULT));
        assert!(!uses_sentinel("chat-template")); // in-band variant, cycles
        let s = spec("chat-template").unwrap();
        assert_eq!(s.kind.extreme(-1), Some("default".into())); // Home → default
        assert_eq!(s.bump(&s.kind, "default", 1), Some("bailing".into()));
        assert_eq!(s.kind.validate("LLAMA3").unwrap(), "llama3"); // case-folded
        assert!(s.kind.validate("not-a-template").is_err());
    }

    #[test]
    fn extreme_jumps_to_bounds() {
        let port = spec("port").unwrap().kind;
        assert_eq!(port.extreme(-1), Some("1".into()));
        assert_eq!(port.extreme(1), Some("65535".into()));
        let cache = spec("cache-type-k").unwrap().kind;
        assert_eq!(cache.extreme(1), Some("q4_0".into()));
    }
}
