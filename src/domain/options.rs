//! Static option metadata and pure value validation/adjustment.

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
                check_bound(v, *min, *max)?;
                Ok(v.to_string())
            }
            OptionKind::Float { min, max } => {
                let v: f64 = input.parse().map_err(|_| format!("'{input}' is not a number"))?;
                if !v.is_finite() {
                    return Err("must be a finite number".into());
                }
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
                let mut v =
                    cur.saturating_add(i64::from(dir).saturating_mul((step.round() as i64).max(1)));
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
                if !v.is_finite() {
                    return None;
                }
                if let Some(lo) = min {
                    v = v.max(*lo);
                }
                if let Some(hi) = max {
                    v = v.min(*hi);
                }
                Some(fmt_float(v))
            }
            OptionKind::Enum(variants) => {
                if variants.is_empty() {
                    return None;
                }
                let idx = variants.iter().position(|v| *v == current).unwrap_or(0) as i32;
                let n = variants.len() as i32;
                let next = (idx + dir).rem_euclid(n) as usize;
                Some(variants[next].to_string())
            }
            OptionKind::Str => None,
        }
    }

    /// Jump to the minimum (`dir = -1`) or maximum (`dir = +1`) — Home/End.
    /// Sentinel-aware stepping lives on `OptionSchema::bump`; resetting to the
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

fn check_bound<T: PartialOrd + std::fmt::Display>(
    v: T,
    min: Option<T>,
    max: Option<T>,
) -> Result<(), String> {
    if let Some(lo) = min
        && v < lo
    {
        return Err(format!("must be ≥ {lo}"));
    }
    if let Some(hi) = max
        && v > hi
    {
        return Err(format!("must be ≤ {hi}"));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_validation_rejects_nan_and_infinity_with_or_without_bounds() {
        for kind in [
            OptionKind::Float { min: None, max: None },
            OptionKind::Float { min: Some(0.0), max: Some(2.0) },
        ] {
            for input in ["NaN", "nan", "inf", "-inf", "1e999"] {
                assert!(kind.validate(input).is_err(), "{input}");
            }
            assert_eq!(kind.validate(" 0.7 ").unwrap(), "0.7");
        }
    }

    #[test]
    fn integer_bounds_remain_exact_above_float_precision() {
        let limit = 9_007_199_254_740_992;
        let kind = OptionKind::Int { min: Some(limit), max: Some(limit) };
        assert!(kind.validate(&(limit - 1).to_string()).is_err());
        assert!(kind.validate(&(limit + 1).to_string()).is_err());
        assert!(kind.validate(&limit.to_string()).is_ok());
    }

    #[test]
    fn integer_adjustment_saturates_before_clamping() {
        let kind = OptionKind::Int { min: None, max: None };
        assert_eq!(kind.adjust(&i64::MAX.to_string(), 1, 1.0).unwrap(), i64::MAX.to_string());
        assert_eq!(kind.adjust(&i64::MIN.to_string(), -1, 1.0).unwrap(), i64::MIN.to_string());
        let bounded = OptionKind::Int { min: Some(0), max: Some(10) };
        assert_eq!(bounded.adjust("9", 1, 4.0).unwrap(), "10");
        assert_eq!(bounded.adjust("1", -1, 4.0).unwrap(), "0");
    }

    #[test]
    fn invalid_adjustments_do_not_panic_or_produce_nonfinite_values() {
        assert!(OptionKind::Enum(&[]).adjust("", 1, 1.0).is_none());
        assert!(OptionKind::Float { min: None, max: None }.adjust("NaN", 1, 1.0).is_none());
    }
}
