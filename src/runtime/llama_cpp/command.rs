//! llama.cpp command prefixes with named model and companion inputs.
use super::SCHEMA;
use crate::domain::OptionItem;
use crate::session::command::Command;

pub(crate) struct LocalCommand<'a> {
    pub binary: &'a str,
    pub model_path: &'a str,
    pub draft_path: Option<&'a str>,
    pub projector_path: Option<&'a str>,
}

impl LocalCommand<'_> {
    pub fn build(self, options: &[OptionItem]) -> Command {
        let Self { binary, model_path, draft_path, projector_path } = self;
        let mut argv = vec![binary.to_string(), "-m".to_string(), model_path.to_string()];

        if let Some(draft_path) = draft_path {
            argv.push("--spec-draft-model".into());
            argv.push(draft_path.to_string());
        }
        if let Some(projector_path) = projector_path {
            argv.push("--mmproj".into());
            argv.push(projector_path.to_string());
        }

        Command::append_options(&mut argv, &SCHEMA, options);
        Command { argv }
    }
}

pub(crate) struct HuggingFaceCommand<'a> {
    pub binary: &'a str,
    pub repo: &'a str,
    pub file: &'a str,
    pub draft_path: Option<&'a str>,
    pub draft_hf: Option<&'a str>,
    pub projector_path: Option<&'a str>,
    pub projector_auto: bool,
}

impl HuggingFaceCommand<'_> {
    pub fn build(self, options: &[OptionItem]) -> Command {
        let Self { binary, repo, file, draft_path, draft_hf, projector_path, projector_auto } =
            self;
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
        Command::append_options(&mut argv, &SCHEMA, options);
        Command { argv }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::registry;

    fn opt(key: &str, value: &str) -> OptionItem {
        OptionItem {
            spec: SCHEMA.spec(key).unwrap(),
            value: value.into(),
            default: String::new(),
            range: None,
        }
    }

    fn sample_options() -> Vec<OptionItem> {
        vec![
            opt("ctx-size", "32768"),
            opt("gpu-layers", "999"),
            opt("temperature", "0.7"),
            opt("flash-attn", "on"),
            opt("host", "127.0.0.1"),
            opt("port", "8000"),
        ]
    }

    fn local(model: &str, options: &[OptionItem]) -> Command {
        LocalCommand {
            binary: "llama-server",
            model_path: model,
            draft_path: None,
            projector_path: None,
        }
        .build(options)
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
        let cmd = HuggingFaceCommand {
            binary: "llama-server",
            repo: "owner/model-GGUF",
            file: "model-Q4_K_M.gguf",
            draft_path: None,
            draft_hf: None,
            projector_path: None,
            projector_auto: false,
        }
        .build(&sample_options());
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
        opts.push(opt("device", "ROCm0"));
        let cmd = local("/m/x.gguf", &opts);
        let i = cmd.argv.iter().position(|a| a == "--device").unwrap();
        assert_eq!(cmd.argv[i + 1], "ROCm0");

        opts.pop();
        opts.push(opt("device", registry::DEFAULT));
        let cmd = local("/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "--device"));
    }

    #[test]
    fn omitted_values_are_skipped() {
        let mut opts = sample_options();
        opts[3] = opt("flash-attn", "auto"); // enum's omit token
        opts.push(opt("batch-size", registry::DEFAULT)); // numeric sentinel
        let cmd = local("/m/x.gguf", &opts);
        // Both flags (and their values) are absent — llama.cpp uses its defaults.
        assert!(!cmd.argv.iter().any(|a| a == "--flash-attn"));
        assert!(!cmd.argv.iter().any(|a| a == "--batch-size"));
        assert!(cmd.argv.iter().all(|a| a != registry::DEFAULT && a != "auto"));
    }

    #[test]
    fn sampling_params_at_default_are_omitted() {
        let mut opts = sample_options();
        opts[2] = opt("temperature", registry::DEFAULT);
        opts.push(opt("top-k", registry::DEFAULT));
        let cmd = local("/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "--temp" || a == "--top-k"));
    }

    #[test]
    fn local_mtp_sidecar_is_passed_as_the_draft_model() {
        let mut opts = sample_options();
        opts.push(opt("spec-type", "draft-mtp"));
        let cmd = LocalCommand {
            binary: "llama-server",
            model_path: "/m/model.gguf",
            draft_path: Some("/m/mtp-model.gguf"),
            projector_path: Some("/m/mmproj-BF16.gguf"),
        }
        .build(&opts);
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
            opt("ctx-size", "131000"),
            opt("gpu-layers", "99"),
            opt("split-mode", "layer"),
            opt("tensor-split", "1,1"),
            opt("temperature", "1"),
            opt("top-p", "0.95"),
            opt("top-k", "64"),
            opt("spec-type", "draft-dflash"),
            opt("spec-draft-n-max", "3"),
            opt("parallel", "1"),
            opt("sleep-idle-seconds", "1200"),
            opt("host", "127.0.0.1"),
            opt("port", "9516"),
        ];
        let cmd = LocalCommand {
            binary: "/home/m/llama.cpp/build/bin/llama-server",
            model_path: "/m/Muse-Glimmer/Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
            draft_path: Some("/m/Muse-Glimmer/dflash-kquant.gguf"),
            projector_path: Some("/m/Muse-Glimmer/mmproj-Muse-Glimmer-30B-Q8_0.gguf"),
        }
        .build(&opts);

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
        let cmd = HuggingFaceCommand {
            binary: "llama-server",
            repo: "owner/model-GGUF",
            file: "model-Q4_K_M.gguf",
            draft_path: None,
            draft_hf: Some("owner/model-GGUF:Q4_0"),
            projector_path: None,
            projector_auto: true,
        }
        .build(&sample_options());
        assert!(
            cmd.argv
                .windows(2)
                .any(|args| { args == ["--spec-draft-hf", "owner/model-GGUF:Q4_0"] })
        );
        assert!(cmd.argv.iter().any(|arg| arg == "--mmproj-auto"));
    }

    #[test]
    fn hybrid_hf_launch_prefers_cached_companion_paths() {
        let cmd = HuggingFaceCommand {
            binary: "llama-server",
            repo: "owner/model-GGUF",
            file: "model-Q4_K_M.gguf",
            draft_path: Some("/cache/mtp-model.gguf"),
            draft_hf: Some("owner/model-GGUF:Q4_0"),
            projector_path: Some("/cache/mmproj-BF16.gguf"),
            projector_auto: true,
        }
        .build(&sample_options());
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
        opts.push(opt("reasoning-effort", "high"));
        let cmd = local("/m/x.gguf", &opts);
        let i = cmd.argv.iter().position(|a| a == "--chat-template-kwargs").unwrap();
        assert_eq!(cmd.argv[i + 1], r#"{"reasoning_effort":"high"}"#);
        // The JSON is shell-quoted in the copy/paste form.
        assert!(cmd.display().contains(r#"'{"reasoning_effort":"high"}'"#));

        // At "default" the kwarg is dropped entirely.
        opts.pop();
        opts.push(opt("reasoning-effort", "default"));
        let cmd = local("/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "--chat-template-kwargs"));
    }

    #[test]
    fn jinja_off_emits_bare_no_jinja_flag_and_chat_template_its_name() {
        let mut opts = sample_options();
        opts.push(opt("jinja", "off"));
        opts.push(opt("chat-template", "llama3"));
        let cmd = local("/m/x.gguf", &opts);
        assert!(cmd.argv.iter().any(|a| a == "--no-jinja"));
        let i = cmd.argv.iter().position(|a| a == "--chat-template").unwrap();
        assert_eq!(cmd.argv[i + 1], "llama3");

        // At their omit tokens both disappear.
        let opts = vec![opt("jinja", "on"), opt("chat-template", "default")];
        let cmd = local("/m/x.gguf", &opts);
        assert_eq!(cmd.argv, vec!["llama-server", "-m", "/m/x.gguf"]);
    }

    #[test]
    fn load_mode_emits_its_value_and_is_omitted_at_default() {
        let mut opts = sample_options();
        opts.push(opt("load-mode", "none"));
        let cmd = local("/m/x.gguf", &opts);
        let i = cmd.argv.iter().position(|a| a == "-lm").unwrap();
        assert_eq!(cmd.argv[i + 1], "none"); // takes a value, unlike the old --no-mmap
        assert!(!cmd.argv.iter().any(|a| a == "--no-mmap"));

        opts.pop();
        opts.push(opt("load-mode", registry::DEFAULT));
        let cmd = local("/m/x.gguf", &opts);
        assert!(!cmd.argv.iter().any(|a| a == "-lm"));
    }
}
