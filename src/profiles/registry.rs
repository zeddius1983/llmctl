//! The generic option model: the value kinds, the per-option metadata, and the
//! [`OptionSchema`] that binds a runtime's option table to its CLI-encoding
//! rules. The tables themselves live with their backends in `crate::runtime`.

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
/// off the command line and rely on the runtime's own built-in default".
pub const DEFAULT: &str = "default";

/// A runtime's option vocabulary plus the CLI-encoding rules that go with it.
///
/// Each runtime speaks a different dialect: llama.cpp has valueless `--no-*`
/// inversions and one option that is delivered as a JSON kwarg, while
/// FastFlowLM takes an explicit value for every flag. Bundling the table with
/// its rules keeps that knowledge next to the backend that owns it, and keeps
/// this type `Copy`-cheap so it can be threaded through resolution and command
/// building without borrowing the backend itself.
#[derive(Debug, Clone, Copy)]
pub struct OptionSchema {
    /// Every option this runtime exposes, in display order.
    pub specs: &'static [OptionSpec],
    /// The value at which an option is dropped from the launch command, because
    /// it equals what the runtime would do anyway. `None` means always emitted.
    pub omit_token: fn(&str) -> Option<&'static str>,
    /// Whether the option is a valueless boolean flag (e.g. `--no-mmap`).
    pub is_flag: fn(&str) -> bool,
    /// The value token actually emitted on the command line.
    pub cli_value: fn(&str, &str) -> String,
}

impl OptionSchema {
    /// Look up an option spec by key.
    pub fn spec(&self, key: &str) -> Option<&'static OptionSpec> {
        self.specs.iter().find(|s| s.key == key)
    }

    pub fn omit_token(&self, key: &str) -> Option<&'static str> {
        (self.omit_token)(key)
    }

    pub fn is_flag(&self, key: &str) -> bool {
        (self.is_flag)(key)
    }

    pub fn cli_value(&self, key: &str, value: &str) -> String {
        (self.cli_value)(key, value)
    }

    /// Whether the option's omitted state is the [`DEFAULT`] sentinel (vs an
    /// in-band enum variant like `"auto"` or an enum's own `"default"` choice).
    /// Only these get the sentinel editing affordances (the `default` text
    /// entry); enums cycle through their variants instead.
    pub fn uses_sentinel(&self, key: &str) -> bool {
        self.omit_token(key) == Some(DEFAULT)
            && !matches!(self.spec(key).map(|s| s.kind), Some(OptionKind::Enum(_)))
    }

    /// Step the value by one increment (`dir = ±1`) for `+`/`-` and the `e`
    /// cycle. For sentinel options [`DEFAULT`] sits just below the numeric
    /// range: stepping up from it enters the concrete default; enums (whose
    /// omitted state is an ordinary variant) just cycle normally.
    pub fn bump(
        &self,
        spec: &OptionSpec,
        kind: &OptionKind,
        current: &str,
        dir: i32,
    ) -> Option<String> {
        if self.uses_sentinel(spec.key) && current == DEFAULT {
            return Some(if dir > 0 { spec.default.to_string() } else { DEFAULT.to_string() });
        }
        kind.adjust(current, dir, spec.step)
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
