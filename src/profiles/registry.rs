//! The generic option model: the value kinds, the per-option metadata, and the
//! [`OptionSchema`] that binds a runtime's option table to its CLI-encoding
//! rules. The tables themselves live with their backends in `crate::runtime`.

pub use crate::domain::options::{OptionKind, OptionSpec};

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
