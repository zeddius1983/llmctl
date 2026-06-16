//! Static registry of launch options: the source of truth for which options
//! exist and their type, default, range, step, CLI flag, and description. Used
//! to render the Options pane, validate edits, and drive inline adjustment.

/// The kind/domain of an option value, used for validation and adjustment.
#[derive(Debug, Clone, Copy)]
pub enum OptionKind {
    Int { min: Option<i64>, max: Option<i64> },
    Float { min: Option<f64>, max: Option<f64> },
    Bool,
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
            OptionKind::Bool => Some("true | false".into()),
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
            OptionKind::Bool => match input.to_ascii_lowercase().as_str() {
                "true" | "on" | "1" | "yes" => Ok("true".into()),
                "false" | "off" | "0" | "no" => Ok("false".into()),
                _ => Err("expected true or false".into()),
            },
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
            OptionKind::Bool => Some(if current == "true" { "false" } else { "true" }.into()),
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
    pub fn extreme(&self, dir: i32) -> Option<String> {
        match self {
            OptionKind::Int { min, max } => {
                if dir < 0 { *min } else { *max }.map(|v| v.to_string())
            }
            OptionKind::Float { min, max } => {
                if dir < 0 { *min } else { *max }.map(fmt_float)
            }
            OptionKind::Bool => Some(if dir < 0 { "false" } else { "true" }.into()),
            OptionKind::Enum(variants) => {
                if dir < 0 { variants.first() } else { variants.last() }.map(|v| (*v).to_string())
            }
            OptionKind::Str => None,
        }
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

use OptionKind::{Bool, Enum, Float, Int, Str};

/// The MVP option set for llama-server.
pub static REGISTRY: &[OptionSpec] = &[
    OptionSpec {
        key: "ctx-size",
        cli: "--ctx-size",
        kind: Int { min: Some(0), max: None },
        default: "4096",
        step: 1024.0,
        description: "Maximum context window size in tokens (0 = use model default).",
    },
    OptionSpec {
        key: "gpu-layers",
        cli: "-ngl",
        kind: Int { min: Some(0), max: Some(999) },
        default: "0",
        step: 1.0,
        description: "Number of model layers to offload to the GPU.",
    },
    OptionSpec {
        key: "temperature",
        cli: "--temp",
        kind: Float { min: Some(0.0), max: Some(2.0) },
        default: "0.8",
        step: 0.05,
        description: "Sampling temperature; lower is more deterministic.",
    },
    OptionSpec {
        key: "top-p",
        cli: "--top-p",
        kind: Float { min: Some(0.0), max: Some(1.0) },
        default: "0.95",
        step: 0.05,
        description: "Nucleus sampling: keep tokens within this cumulative probability.",
    },
    OptionSpec {
        key: "top-k",
        cli: "--top-k",
        kind: Int { min: Some(0), max: None },
        default: "40",
        step: 1.0,
        description: "Keep only the top-K most likely tokens (0 = disabled).",
    },
    OptionSpec {
        key: "min-p",
        cli: "--min-p",
        kind: Float { min: Some(0.0), max: Some(1.0) },
        default: "0.05",
        step: 0.01,
        description: "Minimum token probability relative to the most likely token.",
    },
    OptionSpec {
        key: "repeat-penalty",
        cli: "--repeat-penalty",
        kind: Float { min: Some(0.0), max: Some(2.0) },
        default: "1.0",
        step: 0.05,
        description: "Penalty applied to repeated tokens (1.0 = disabled).",
    },
    OptionSpec {
        key: "threads",
        cli: "--threads",
        kind: Int { min: Some(0), max: None },
        default: "0",
        step: 1.0,
        description: "CPU threads for generation (0 = auto-detect).",
    },
    OptionSpec {
        key: "batch-size",
        cli: "--batch-size",
        kind: Int { min: Some(1), max: None },
        default: "2048",
        step: 256.0,
        description: "Logical batch size for prompt processing.",
    },
    OptionSpec {
        key: "flash-attn",
        cli: "--flash-attn",
        kind: Bool,
        default: "false",
        step: 1.0,
        description: "Enable flash attention (faster, lower memory where supported).",
    },
    OptionSpec {
        key: "cache-type-k",
        cli: "--cache-type-k",
        kind: Enum(&["f16", "q8_0", "q4_0"]),
        default: "f16",
        step: 1.0,
        description: "KV cache data type for keys (lower = less memory).",
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
    fn bool_normalizes_synonyms() {
        let kind = spec("flash-attn").unwrap().kind;
        assert_eq!(kind.validate("on").unwrap(), "true");
        assert_eq!(kind.validate("0").unwrap(), "false");
        assert!(kind.validate("maybe").is_err());
    }

    #[test]
    fn adjust_clamps_numeric_and_cycles_enum() {
        let temp = spec("temperature").unwrap();
        assert_eq!(temp.kind.adjust("1.95", 1, temp.step), Some("2".into())); // clamp at 2.0
        assert_eq!(temp.kind.adjust("0.8", -1, temp.step), Some("0.75".into()));

        let cache = spec("cache-type-k").unwrap().kind;
        assert_eq!(cache.adjust("f16", 1, 1.0), Some("q8_0".into()));
        assert_eq!(cache.adjust("f16", -1, 1.0), Some("q4_0".into())); // wraps
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
