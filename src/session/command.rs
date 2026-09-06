//! Pure construction of a launch command from resolved options.
//!
//! No I/O: takes the runtime binary, the model, and the resolved option values,
//! and produces an argv plus shell-quoted display strings. This is the "never
//! hand-type a complex command again" core, and is unit-tested. The
//! runtime-specific builders live with their backends, which share
//! [`Command::append_options`] for the option tail.

use crate::domain::OptionItem;
use crate::profiles::registry::OptionSchema;

/// A built launch command: program + arguments, ready to spawn or display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    pub argv: Vec<String>,
}

impl Command {
    /// Append the resolved options to `argv`, in schema order.
    ///
    /// Every option is emitted as `--flag value` unless the schema says
    /// otherwise. An option sitting at its
    /// [`OptionSchema::omit_token`](crate::profiles::registry::OptionSchema::omit_token)
    /// — `flash-attn=auto`, or a numeric at the `default` sentinel — is skipped
    /// so the runtime applies its own default. Valueless boolean flags
    /// (`--no-mmap`) emit the bare flag, and values pass through
    /// [`OptionSchema::cli_value`](crate::profiles::registry::OptionSchema::cli_value),
    /// which rewrites the ones whose on-disk form isn't the literal argv token
    /// (llama.cpp's `reasoning-effort` becomes a JSON kwarg).
    pub fn append_options(argv: &mut Vec<String>, schema: &OptionSchema, options: &[OptionItem]) {
        for opt in options {
            if schema.omit_token(opt.spec.key) == Some(opt.value.as_str()) {
                continue;
            }
            argv.push(opt.spec.cli.to_string());
            if !schema.is_flag(opt.spec.key) {
                argv.push(schema.cli_value(opt.spec.key, &opt.value));
            }
        }
    }

    /// Single-line, shell-quoted command suitable for copy/paste.
    pub fn display(&self) -> String {
        self.argv.iter().map(|a| shell_quote(a)).collect::<Vec<_>>().join(" ")
    }

    /// Multi-line form with `\` continuations — one flag (and its value) per
    /// line, for the launch-preview modal.
    pub fn pretty(&self) -> String {
        if self.argv.is_empty() {
            return String::new();
        }
        let mut lines: Vec<String> = vec![shell_quote(&self.argv[0])];
        let args = &self.argv[1..];
        let mut i = 0;
        while i < args.len() {
            // Group a flag with its value (a token starting with '-' that is
            // followed by a non-flag token takes that token as its value).
            let flag = &args[i];
            if flag.starts_with('-') && i + 1 < args.len() && !args[i + 1].starts_with('-') {
                lines.push(format!("{} {}", shell_quote(flag), shell_quote(&args[i + 1])));
                i += 2;
            } else {
                lines.push(shell_quote(flag));
                i += 1;
            }
        }
        lines.join(" \\\n  ")
    }
}

/// Quote a single argument for a POSIX shell if it contains anything unsafe.
fn shell_quote(arg: &str) -> String {
    let safe = !arg.is_empty()
        && arg.chars().all(|c| {
            c.is_ascii_alphanumeric()
                || matches!(c, '_' | '.' | '/' | ':' | '=' | '@' | '%' | '+' | '-' | ',')
        });
    if safe {
        arg.to_string()
    } else {
        // Wrap in single quotes; close/escape/reopen around embedded quotes.
        format!("'{}'", arg.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_quotes_paths_with_spaces() {
        let cmd = Command {
            argv: ["server", "-m", "/m/my model.gguf", "it's", "", "$(cmd)"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        };
        assert_eq!(cmd.display(), "server -m '/m/my model.gguf' 'it'\\''s' '' '$(cmd)'");
    }

    #[test]
    fn pretty_groups_flag_and_value_per_line() {
        let cmd = Command {
            argv: ["server", "-m", "/m/x.gguf", "--ctx-size", "32768", "--flash-attn", "on"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        };
        let pretty = cmd.pretty();
        assert!(pretty.contains("-m /m/x.gguf"));
        assert!(pretty.contains("--ctx-size 32768"));
        assert!(pretty.contains("--flash-attn on"));
        assert_eq!(pretty.lines().count(), 4);
        assert!(pretty.lines().take(3).all(|line| line.ends_with('\\')));
    }
}
