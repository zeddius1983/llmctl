//! Pure construction of the `llama-server` launch command from resolved options.
//!
//! No I/O: takes the runtime binary, the model file, and the resolved option
//! values, and produces an argv plus shell-quoted display strings. This is the
//! "never hand-type a complex command again" core, and is unit-tested.

use crate::domain::OptionItem;
use crate::profiles::registry;

/// A built launch command: program + arguments, ready to spawn or display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    pub argv: Vec<String>,
}

impl Command {
    /// Build from the runtime binary, the model file path, and resolved options.
    ///
    /// Every option is emitted as `--flag value`, in registry order (current
    /// llama-server flags all take an explicit value, including
    /// `--flash-attn on|off|auto`). The model is passed via `-m`. An option left
    /// at its [`registry::omit_token`] (e.g. `flash-attn=auto`, or a numeric at
    /// the `default` sentinel) is skipped so llama.cpp applies its own default.
    /// Valueless boolean flags ([`registry::is_flag`], e.g. `--no-mmap`) emit the
    /// bare flag with no following value. Values pass through
    /// [`registry::cli_value`], which rewrites the ones whose on-disk form isn't
    /// the literal argv token (e.g. `reasoning-effort` → a JSON kwarg).
    pub fn build(binary: &str, model_path: &str, options: &[OptionItem]) -> Self {
        let mut argv = vec![binary.to_string(), "-m".to_string(), model_path.to_string()];

        Self::append_options(&mut argv, options);
        Self { argv }
    }

    /// Build a command that lets llama.cpp download/cache an exact GGUF file
    /// from Hugging Face before loading it.
    pub fn build_huggingface(binary: &str, repo: &str, file: &str, options: &[OptionItem]) -> Self {
        let mut argv = vec![
            binary.to_string(),
            "--hf-repo".into(),
            repo.to_string(),
            "--hf-file".into(),
            file.to_string(),
        ];
        Self::append_options(&mut argv, options);
        Self { argv }
    }

    fn append_options(argv: &mut Vec<String>, options: &[OptionItem]) {
        for opt in options {
            if registry::omit_token(&opt.key) == Some(opt.value.as_str()) {
                continue;
            }
            argv.push(opt.cli.clone());
            if !registry::is_flag(&opt.key) {
                argv.push(registry::cli_value(&opt.key, &opt.value));
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

    fn opt(key: &str, value: &str, cli: &str) -> OptionItem {
        OptionItem {
            key: key.into(),
            value: value.into(),
            default: String::new(),
            range: None,
            cli: cli.into(),
            description: String::new(),
        }
    }

    fn sample_options() -> Vec<OptionItem> {
        vec![
            opt("ctx-size", "32768", "--ctx-size"),
            opt("gpu-layers", "999", "-ngl"),
            opt("temperature", "0.7", "--temp"),
            opt("flash-attn", "on", "--flash-attn"),
            opt("host", "127.0.0.1", "--host"),
            opt("port", "8000", "--port"),
        ]
    }

    #[test]
    fn builds_argv_in_order_with_model_first() {
        let cmd = Command::build("llama-server", "/m/qwen.gguf", &sample_options());
        assert_eq!(
            cmd.argv,
            vec![
                "llama-server",
                "-m",
                "/m/qwen.gguf",
                "--ctx-size",
                "32768",
                "-ngl",
                "999",
                "--temp",
                "0.7",
                "--flash-attn",
                "on",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ]
        );
    }

    #[test]
    fn builds_hugging_face_repo_and_exact_file() {
        let cmd = Command::build_huggingface(
            "llama-server",
            "owner/model-GGUF",
            "model-Q4_K_M.gguf",
            &sample_options(),
        );
        assert_eq!(
            &cmd.argv[..5],
            ["llama-server", "--hf-repo", "owner/model-GGUF", "--hf-file", "model-Q4_K_M.gguf",]
        );
        assert!(!cmd.argv.iter().any(|arg| arg.starts_with("hf_")));
    }

    #[test]
    fn flash_attn_emits_its_value() {
        let cmd = Command::build("llama-server", "/m/x.gguf", &sample_options());
        let i = cmd.argv.iter().position(|a| a == "--flash-attn").unwrap();
        assert_eq!(cmd.argv[i + 1], "on");
    }

    #[test]
    fn selected_device_is_emitted_and_default_is_omitted() {
        let mut opts = sample_options();
        opts.push(opt("device", "ROCm0", "--device"));
        let cmd = Command::build("llama-server", "/m/x.gguf", &opts);
        let i = cmd.argv.iter().position(|a| a == "--device").unwrap();
        assert_eq!(cmd.argv[i + 1], "ROCm0");

        opts.pop();
        opts.push(opt("device", registry::DEFAULT, "--device"));
        let cmd = Command::build("llama-server", "/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "--device"));
    }

    #[test]
    fn omitted_values_are_skipped() {
        let mut opts = sample_options();
        opts[3] = opt("flash-attn", "auto", "--flash-attn"); // enum's omit token
        opts.push(opt("batch-size", registry::DEFAULT, "--batch-size")); // numeric sentinel
        let cmd = Command::build("llama-server", "/m/x.gguf", &opts);
        // Both flags (and their values) are absent — llama.cpp uses its defaults.
        assert!(!cmd.argv.iter().any(|a| a == "--flash-attn"));
        assert!(!cmd.argv.iter().any(|a| a == "--batch-size"));
        assert!(cmd.argv.iter().all(|a| a != registry::DEFAULT && a != "auto"));
    }

    #[test]
    fn sampling_params_at_default_are_omitted() {
        let mut opts = sample_options();
        opts[2] = opt("temperature", registry::DEFAULT, "--temp");
        opts.push(opt("top-k", registry::DEFAULT, "--top-k"));
        let cmd = Command::build("llama-server", "/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "--temp" || a == "--top-k"));
    }

    #[test]
    fn reasoning_effort_emits_chat_template_kwargs_json() {
        let mut opts = sample_options();
        opts.push(opt("reasoning-effort", "high", "--chat-template-kwargs"));
        let cmd = Command::build("llama-server", "/m/x.gguf", &opts);
        let i = cmd.argv.iter().position(|a| a == "--chat-template-kwargs").unwrap();
        assert_eq!(cmd.argv[i + 1], r#"{"reasoning_effort":"high"}"#);
        // The JSON is shell-quoted in the copy/paste form.
        assert!(cmd.display().contains(r#"'{"reasoning_effort":"high"}'"#));

        // At "default" the kwarg is dropped entirely.
        opts.pop();
        opts.push(opt("reasoning-effort", "default", "--chat-template-kwargs"));
        let cmd = Command::build("llama-server", "/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "--chat-template-kwargs"));
    }

    #[test]
    fn jinja_off_emits_bare_no_jinja_flag_and_chat_template_its_name() {
        let mut opts = sample_options();
        opts.push(opt("jinja", "off", "--no-jinja"));
        opts.push(opt("chat-template", "llama3", "--chat-template"));
        let cmd = Command::build("llama-server", "/m/x.gguf", &opts);
        assert!(cmd.argv.iter().any(|a| a == "--no-jinja"));
        let i = cmd.argv.iter().position(|a| a == "--chat-template").unwrap();
        assert_eq!(cmd.argv[i + 1], "llama3");

        // At their omit tokens both disappear.
        let opts = vec![
            opt("jinja", "on", "--no-jinja"),
            opt("chat-template", "default", "--chat-template"),
        ];
        let cmd = Command::build("llama-server", "/m/x.gguf", &opts);
        assert_eq!(cmd.argv, vec!["llama-server", "-m", "/m/x.gguf"]);
    }

    #[test]
    fn mmap_off_emits_bare_no_mmap_flag_and_on_is_omitted() {
        let mut opts = sample_options();
        opts.push(opt("mmap", "off", "--no-mmap"));
        let cmd = Command::build("llama-server", "/m/x.gguf", &opts);
        let i = cmd.argv.iter().position(|a| a == "--no-mmap").unwrap();
        // The bare flag is last (or followed by another flag) — no value token,
        // so its "off" value never reaches the command line.
        assert!(cmd.argv.get(i + 1).map(|a| a.starts_with("--")).unwrap_or(true));

        opts.pop();
        opts.push(opt("mmap", "on", "--no-mmap")); // omit token: mmap on is llama default
        let cmd = Command::build("llama-server", "/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "--no-mmap"));
    }

    #[test]
    fn display_quotes_paths_with_spaces() {
        let opts = sample_options();
        let cmd = Command::build("llama-server", "/m/my model.gguf", &opts);
        assert!(cmd.display().contains("'/m/my model.gguf'"));
        // Ordinary tokens are left unquoted.
        assert!(cmd.display().starts_with("llama-server -m '/m/my model.gguf'"));
    }

    #[test]
    fn pretty_groups_flag_and_value_per_line() {
        let cmd = Command::build("llama-server", "/m/x.gguf", &sample_options());
        let pretty = cmd.pretty();
        assert!(pretty.contains("-m /m/x.gguf")); // model flag + path grouped, not orphaned
        assert!(pretty.contains("--ctx-size 32768"));
        assert!(pretty.contains("--flash-attn on")); // flag + value grouped
        assert!(pretty.contains(" \\\n")); // line continuations
    }
}
