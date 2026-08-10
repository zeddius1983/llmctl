//! Pure construction of a launch command from resolved options.
//!
//! No I/O: takes the runtime binary, the model, and the resolved option values,
//! and produces an argv plus shell-quoted display strings. This is the "never
//! hand-type a complex command again" core, and is unit-tested. The
//! llama.cpp-shaped builders live here; each backend assembles its own prefix
//! and shares [`Command::append_options`] for the option tail.

use crate::domain::OptionItem;
use crate::profiles::registry::OptionSchema;

/// A built launch command: program + arguments, ready to spawn or display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    pub argv: Vec<String>,
}

impl Command {
    /// Build a local-model command with any selected auxiliary GGUFs. The model
    /// is passed via `-m`; options follow per [`Command::append_options`].
    pub fn build_local(
        binary: &str,
        model_path: &str,
        draft_path: Option<&str>,
        projector_path: Option<&str>,
        schema: &OptionSchema,
        options: &[OptionItem],
    ) -> Self {
        let mut argv = vec![binary.to_string(), "-m".to_string(), model_path.to_string()];

        if let Some(draft_path) = draft_path {
            argv.push("--spec-draft-model".into());
            argv.push(draft_path.to_string());
        }
        if let Some(projector_path) = projector_path {
            argv.push("--mmproj".into());
            argv.push(projector_path.to_string());
        }

        Self::append_options(&mut argv, schema, options);
        Self { argv }
    }

    /// Build a command that lets llama.cpp download/cache an exact GGUF file
    /// from Hugging Face before loading it.
    pub fn build_huggingface(
        binary: &str,
        repo: &str,
        file: &str,
        draft_path: Option<&str>,
        draft_hf: Option<&str>,
        projector_path: Option<&str>,
        projector_auto: bool,
        schema: &OptionSchema,
        options: &[OptionItem],
    ) -> Self {
        let mut argv = vec![
            binary.to_string(),
            "--hf-repo".into(),
            repo.to_string(),
            "--hf-file".into(),
            file.to_string(),
        ];
        if let Some(draft_path) = draft_path {
            argv.push("--spec-draft-model".into());
            argv.push(draft_path.to_string());
        } else if let Some(draft_hf) = draft_hf {
            argv.push("--spec-draft-hf".into());
            argv.push(draft_hf.to_string());
        }
        if let Some(projector_path) = projector_path {
            argv.push("--mmproj".into());
            argv.push(projector_path.to_string());
        } else if projector_auto {
            argv.push("--mmproj-auto".into());
        }
        Self::append_options(&mut argv, schema, options);
        Self { argv }
    }

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
            if schema.omit_token(&opt.key) == Some(opt.value.as_str()) {
                continue;
            }
            argv.push(opt.cli.clone());
            if !schema.is_flag(&opt.key) {
                argv.push(schema.cli_value(&opt.key, &opt.value));
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
    use crate::profiles::registry;
    use crate::runtime::llama_cpp::SCHEMA;

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

    fn local(model: &str, options: &[OptionItem]) -> Command {
        Command::build_local("llama-server", model, None, None, &SCHEMA, options)
    }

    #[test]
    fn builds_argv_in_order_with_model_first() {
        let cmd = local("/m/qwen.gguf", &sample_options());
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
            None,
            None,
            None,
            false,
            &SCHEMA,
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
        let cmd = local("/m/x.gguf", &sample_options());
        let i = cmd.argv.iter().position(|a| a == "--flash-attn").unwrap();
        assert_eq!(cmd.argv[i + 1], "on");
    }

    #[test]
    fn selected_device_is_emitted_and_default_is_omitted() {
        let mut opts = sample_options();
        opts.push(opt("device", "ROCm0", "--device"));
        let cmd = local("/m/x.gguf", &opts);
        let i = cmd.argv.iter().position(|a| a == "--device").unwrap();
        assert_eq!(cmd.argv[i + 1], "ROCm0");

        opts.pop();
        opts.push(opt("device", registry::DEFAULT, "--device"));
        let cmd = local("/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "--device"));
    }

    #[test]
    fn omitted_values_are_skipped() {
        let mut opts = sample_options();
        opts[3] = opt("flash-attn", "auto", "--flash-attn"); // enum's omit token
        opts.push(opt("batch-size", registry::DEFAULT, "--batch-size")); // numeric sentinel
        let cmd = local("/m/x.gguf", &opts);
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
        let cmd = local("/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "--temp" || a == "--top-k"));
    }

    #[test]
    fn local_mtp_sidecar_is_passed_as_the_draft_model() {
        let mut opts = sample_options();
        opts.push(opt("spec-type", "draft-mtp", "--spec-type"));
        let cmd = Command::build_local(
            "llama-server",
            "/m/model.gguf",
            Some("/m/mtp-model.gguf"),
            Some("/m/mmproj-BF16.gguf"),
            &SCHEMA,
            &opts,
        );
        assert_eq!(
            &cmd.argv[..9],
            [
                "llama-server",
                "-m",
                "/m/model.gguf",
                "--spec-draft-model",
                "/m/mtp-model.gguf",
                "--mmproj",
                "/m/mmproj-BF16.gguf",
                "--ctx-size",
                "32768",
            ]
        );
        let spec = cmd.argv.iter().position(|arg| arg == "--spec-type").unwrap();
        assert_eq!(cmd.argv[spec + 1], "draft-mtp");
        let projector = cmd.argv.iter().position(|arg| arg == "--mmproj").unwrap();
        assert_eq!(cmd.argv[projector + 1], "/m/mmproj-BF16.gguf");
    }

    /// The published two-GPU Muse-Glimmer configuration, built from resolved
    /// options: dFlash drafting off a `dflash-*.gguf` companion, the projector,
    /// and an even split across both cards.
    #[test]
    fn dflash_multi_gpu_launch_reproduces_the_reference_command() {
        let opts = vec![
            opt("ctx-size", "131000", "--ctx-size"),
            opt("gpu-layers", "99", "-ngl"),
            opt("split-mode", "layer", "-sm"),
            opt("tensor-split", "1,1", "-ts"),
            opt("temperature", "1", "--temp"),
            opt("top-p", "0.95", "--top-p"),
            opt("top-k", "64", "--top-k"),
            opt("spec-type", "draft-dflash", "--spec-type"),
            opt("spec-draft-n-max", "3", "--spec-draft-n-max"),
            opt("parallel", "1", "-np"),
            opt("sleep-idle-seconds", "1200", "--sleep-idle-seconds"),
            opt("host", "127.0.0.1", "--host"),
            opt("port", "9516", "--port"),
        ];
        let cmd = Command::build_local(
            "/home/m/llama.cpp/build/bin/llama-server",
            "/m/Muse-Glimmer/Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
            Some("/m/Muse-Glimmer/dflash-kquant.gguf"),
            Some("/m/Muse-Glimmer/mmproj-Muse-Glimmer-30B-Q8_0.gguf"),
            &SCHEMA,
            &opts,
        );

        for pair in [
            ["-m", "/m/Muse-Glimmer/Muse-Glimmer-30B-UD-Q8_K_XL.gguf"],
            ["--spec-draft-model", "/m/Muse-Glimmer/dflash-kquant.gguf"],
            ["--mmproj", "/m/Muse-Glimmer/mmproj-Muse-Glimmer-30B-Q8_0.gguf"],
            ["--spec-type", "draft-dflash"],
            ["--spec-draft-n-max", "3"],
            ["-sm", "layer"],
            ["-ts", "1,1"],
            ["-ngl", "99"],
            ["--ctx-size", "131000"],
            ["-np", "1"],
            ["--sleep-idle-seconds", "1200"],
            ["--port", "9516"],
        ] {
            assert!(cmd.argv.windows(2).any(|args| args == pair), "missing {pair:?}");
        }
        // Jinja templating is llama.cpp's own default, so `--jinja` is implied
        // rather than emitted.
        assert!(!cmd.argv.iter().any(|arg| arg == "--no-jinja"));
    }

    #[test]
    fn remote_companions_use_draft_hf_and_projector_auto() {
        let cmd = Command::build_huggingface(
            "llama-server",
            "owner/model-GGUF",
            "model-Q4_K_M.gguf",
            None,
            Some("owner/model-GGUF:Q4_0"),
            None,
            true,
            &SCHEMA,
            &sample_options(),
        );
        assert!(
            cmd.argv
                .windows(2)
                .any(|args| { args == ["--spec-draft-hf", "owner/model-GGUF:Q4_0"] })
        );
        assert!(cmd.argv.iter().any(|arg| arg == "--mmproj-auto"));
    }

    #[test]
    fn hybrid_hf_launch_prefers_cached_companion_paths() {
        let cmd = Command::build_huggingface(
            "llama-server",
            "owner/model-GGUF",
            "model-Q4_K_M.gguf",
            Some("/cache/mtp-model.gguf"),
            Some("owner/model-GGUF:Q4_0"),
            Some("/cache/mmproj-BF16.gguf"),
            true,
            &SCHEMA,
            &sample_options(),
        );
        assert!(
            cmd.argv
                .windows(2)
                .any(|args| { args == ["--spec-draft-model", "/cache/mtp-model.gguf"] })
        );
        assert!(
            cmd.argv.windows(2).any(|args| { args == ["--mmproj", "/cache/mmproj-BF16.gguf"] })
        );
        assert!(!cmd.argv.iter().any(|arg| arg == "--spec-draft-hf"));
        assert!(!cmd.argv.iter().any(|arg| arg == "--mmproj-auto"));
    }

    #[test]
    fn reasoning_effort_emits_chat_template_kwargs_json() {
        let mut opts = sample_options();
        opts.push(opt("reasoning-effort", "high", "--chat-template-kwargs"));
        let cmd = local("/m/x.gguf", &opts);
        let i = cmd.argv.iter().position(|a| a == "--chat-template-kwargs").unwrap();
        assert_eq!(cmd.argv[i + 1], r#"{"reasoning_effort":"high"}"#);
        // The JSON is shell-quoted in the copy/paste form.
        assert!(cmd.display().contains(r#"'{"reasoning_effort":"high"}'"#));

        // At "default" the kwarg is dropped entirely.
        opts.pop();
        opts.push(opt("reasoning-effort", "default", "--chat-template-kwargs"));
        let cmd = local("/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "--chat-template-kwargs"));
    }

    #[test]
    fn jinja_off_emits_bare_no_jinja_flag_and_chat_template_its_name() {
        let mut opts = sample_options();
        opts.push(opt("jinja", "off", "--no-jinja"));
        opts.push(opt("chat-template", "llama3", "--chat-template"));
        let cmd = local("/m/x.gguf", &opts);
        assert!(cmd.argv.iter().any(|a| a == "--no-jinja"));
        let i = cmd.argv.iter().position(|a| a == "--chat-template").unwrap();
        assert_eq!(cmd.argv[i + 1], "llama3");

        // At their omit tokens both disappear.
        let opts = vec![
            opt("jinja", "on", "--no-jinja"),
            opt("chat-template", "default", "--chat-template"),
        ];
        let cmd = local("/m/x.gguf", &opts);
        assert_eq!(cmd.argv, vec!["llama-server", "-m", "/m/x.gguf"]);
    }

    #[test]
    fn load_mode_emits_its_value_and_is_omitted_at_default() {
        let mut opts = sample_options();
        opts.push(opt("load-mode", "none", "-lm"));
        let cmd = local("/m/x.gguf", &opts);
        let i = cmd.argv.iter().position(|a| a == "-lm").unwrap();
        assert_eq!(cmd.argv[i + 1], "none"); // takes a value, unlike the old --no-mmap
        assert!(!cmd.argv.iter().any(|a| a == "--no-mmap"));

        opts.pop();
        opts.push(opt("load-mode", registry::DEFAULT, "-lm"));
        let cmd = local("/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "-lm"));
    }

    #[test]
    fn display_quotes_paths_with_spaces() {
        let opts = sample_options();
        let cmd = local("/m/my model.gguf", &opts);
        assert!(cmd.display().contains("'/m/my model.gguf'"));
        // Ordinary tokens are left unquoted.
        assert!(cmd.display().starts_with("llama-server -m '/m/my model.gguf'"));
    }

    #[test]
    fn pretty_groups_flag_and_value_per_line() {
        let cmd = local("/m/x.gguf", &sample_options());
        let pretty = cmd.pretty();
        assert!(pretty.contains("-m /m/x.gguf")); // model flag + path grouped, not orphaned
        assert!(pretty.contains("--ctx-size 32768"));
        assert!(pretty.contains("--flash-attn on")); // flag + value grouped
        assert!(pretty.contains(" \\\n")); // line continuations
    }
}
