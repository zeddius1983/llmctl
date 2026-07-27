//! The llama.cpp backend: `llama-server` discovery, its option vocabulary and
//! CLI dialect, command construction, and the `/health` readiness contract.

use std::path::{Path, PathBuf};
use std::process::Command as ProcCommand;

use tracing::{debug, warn};

use crate::config::{Defaults, LlamaCppConfig};
use crate::domain::{Model, OptionItem, RemoteModel, Runtime};
use crate::profiles::registry::{DEFAULT, OptionKind, OptionSchema, OptionSpec};
use crate::profiles::templates::Template;
use crate::runtime::{CatalogCtx, LaunchContext, RuntimeBackend};
use crate::session::command::Command;
use crate::session::record::{DownloadBlob, DownloadRecord};
use crate::session::supervisor;

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
pub static SPECS: &[OptionSpec] = &[
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
                      integrated or companion MTP head).",
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

/// llama.cpp's option-omission rules. For on/off/auto enums the omitted state is
/// `"auto"` (llama's own default); enums that carry an explicit `"default"`
/// variant (e.g. the cache types) omit at that variant; for numeric options with
/// no in-band sentinel it's the [`DEFAULT`] sentinel.
fn omit_token(key: &str) -> Option<&'static str> {
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
fn is_flag(key: &str) -> bool {
    matches!(key, "mmap" | "jinja")
}

/// Most options pass their value through verbatim; `reasoning-effort` has no
/// native llama-server flag and is delivered to the chat template as a JSON
/// kwarg via `--chat-template-kwargs` (how GPT-OSS-style templates receive it).
fn cli_value(key: &str, value: &str) -> String {
    match key {
        "reasoning-effort" => format!(r#"{{"reasoning_effort":"{value}"}}"#),
        _ => value.to_string(),
    }
}

/// llama.cpp's option vocabulary bound to its CLI dialect.
pub static SCHEMA: OptionSchema = OptionSchema { specs: SPECS, omit_token, is_flag, cli_value };

/// llama.cpp's built-in profile templates.
pub static TEMPLATES: &[Template] = &[
    Template { name: "Default", overrides: &[] },
    Template {
        name: "Chat",
        overrides: &[
            ("temperature", "0.7"),
            ("top-p", "0.9"),
            ("top-k", "40"),
            ("repeat-penalty", "1.1"),
        ],
    },
    Template {
        name: "Coding",
        overrides: &[
            ("temperature", "0.2"),
            ("top-p", "0.95"),
            ("repeat-penalty", "1.05"),
            ("ctx-size", "16384"),
        ],
    },
    Template { name: "Long Context", overrides: &[("ctx-size", "131072"), ("flash-attn", "on")] },
    Template {
        name: "Server",
        overrides: &[("host", "0.0.0.0"), ("flash-attn", "on"), ("gpu-layers", "999")],
    },
];

/// The llama.cpp backend: a discovered `llama-server` plus its dialect.
pub struct LlamaCppBackend {
    runtime: Runtime,
    /// Capabilities sniffed from the cached `--help`, so a launch that needs a
    /// newer flag fails with an explanation instead of an opaque server error.
    pub hf_supported: bool,
    pub draft_hf_supported: bool,
    pub mmproj_auto_supported: bool,
}

impl LlamaCppBackend {
    /// Discover llama.cpp from configuration: locate the server binary, capture
    /// its version and devices, and cache `--help` for capability sniffing.
    pub fn discover(cfg: &LlamaCppConfig, cache_dir: &Path) -> Self {
        let binary_path = super::resolve_binary(&cfg.binary);
        let bench_path = binary_path.as_deref().and_then(resolve_bench);
        let version = binary_path.as_deref().and_then(query_version);
        let devices = binary_path.as_deref().map(query_devices).unwrap_or_default();

        if let Some(path) = &binary_path {
            if let Err(err) = cache_help(path, cache_dir) {
                debug!(%err, "could not cache llama-server --help");
            }
        } else {
            warn!(binary = %cfg.binary, "llama-server binary not found");
        }

        let help = std::fs::read_to_string(cache_dir.join(HELP_CACHE)).ok();
        let advertises = |flag: &str| help.as_deref().is_some_and(|h| h.contains(flag));

        Self {
            runtime: Runtime {
                name: NAME.into(),
                description: "GGUF inference via llama-server".into(),
                version,
                binary_path,
                bench_path,
                formats: vec!["GGUF".into()],
                devices,
            },
            hf_supported: advertises("--hf-repo") && advertises("--hf-file"),
            draft_hf_supported: advertises("--spec-draft-hf"),
            mmproj_auto_supported: advertises("--mmproj-auto"),
        }
    }
}

pub const NAME: &str = "llama.cpp";
const HELP_CACHE: &str = "llama-server.help.txt";

impl RuntimeBackend for LlamaCppBackend {
    fn descriptor(&self) -> &Runtime {
        &self.runtime
    }

    fn schema(&self) -> &'static OptionSchema {
        &SCHEMA
    }

    fn templates(&self) -> &'static [Template] {
        TEMPLATES
    }

    fn models(&self, ctx: &CatalogCtx) -> Vec<Model> {
        let mut models = crate::discovery::scan_models(ctx.sources, ctx.cache_path);
        crate::discovery::reconcile(ctx.models_dir, &mut models);
        models.extend(crate::discovery::online::load_cached(ctx.models_dir));
        models
    }

    /// `ctx-size` gains an upper bound equal to the model's trained context
    /// length, so `End`/`+` target "max supported context" rather than an
    /// unbounded value.
    fn effective_kind(&self, spec: &OptionSpec, model: &Model) -> OptionKind {
        match (spec.key, model.context_length) {
            ("ctx-size", Some(ctx)) => OptionKind::Int { min: Some(0), max: Some(ctx as i64) },
            _ => spec.kind,
        }
    }

    /// Omittable options start at their omit token, except `ctx-size`: its
    /// 'default' means the model's full trained context, which can exhaust
    /// memory, so it begins at the ctx/8 heuristic instead.
    fn spec_default(&self, spec: &OptionSpec, model: &Model, defaults: &Defaults) -> String {
        // An integrated or sidecar MTP head is only useful when llama.cpp's MTP
        // drafter is enabled. Make that the model-aware default while still
        // allowing a saved profile value to override it.
        if spec.key == "spec-type" && model.supports_mtp() {
            return "draft-mtp".into();
        }
        match SCHEMA.omit_token(spec.key) {
            Some(token) if spec.key != "ctx-size" => token.to_string(),
            // ctx-size defaults to one eighth of the model's trained context: a
            // memory-friendly start the user can raise toward the model max.
            _ if spec.key == "ctx-size" => match model.context_length {
                Some(ctx) if ctx >= 8 => (ctx / 8).to_string(),
                _ => config_default(spec, defaults),
            },
            _ => config_default(spec, defaults),
        }
    }

    /// Map legacy stored values for the on/off/auto enums onto the current
    /// vocabulary so old profiles still launch: booleans (`true`/`false`, from
    /// when flash-attn was a switch) and the old `default` sentinel (now `auto`).
    fn normalize_legacy(&self, key: &str, value: String) -> String {
        if key != "flash-attn" && key != "reasoning" {
            return value;
        }
        match value.as_str() {
            "true" => "on".into(),
            "false" => "off".into(),
            "default" => "auto".into(),
            _ => value,
        }
    }

    /// Clamp `ctx-size` down to the model's trained context length so no
    /// resolved value ever exceeds what the model supports.
    fn clamp_to_model(&self, key: &str, value: String, model: &Model) -> String {
        if key != "ctx-size" {
            return value;
        }
        match (model.context_length, value.parse::<i64>()) {
            (Some(ctx), Ok(v)) if v > ctx as i64 => ctx.to_string(),
            _ => value,
        }
    }

    fn build_command(&self, ctx: &LaunchContext) -> Command {
        let model = ctx.model;
        let mtp_enabled = mtp_enabled(ctx.options);
        let mtp_path =
            mtp_enabled.then(|| model.mtp_path.as_ref().map(|p| p.display().to_string())).flatten();
        let projector_path = model.projector_path.as_ref().map(|p| p.display().to_string());

        match remote_launch(model, mtp_enabled) {
            Some(remote) => Command::build_huggingface(
                ctx.binary,
                &remote.repo,
                remote.file.as_deref().unwrap_or_default(),
                mtp_path.as_deref(),
                draft_hf(remote, mtp_enabled, &mtp_path).as_deref(),
                projector_path.as_deref(),
                projector_path.is_none() && remote.projector_file.is_some(),
                &SCHEMA,
                ctx.options,
            ),
            None => Command::build_local(
                ctx.binary,
                &model.path.display().to_string(),
                mtp_path.as_deref(),
                projector_path.as_deref(),
                &SCHEMA,
                ctx.options,
            ),
        }
    }

    /// `llama-cli` beside `llama-server`, in conversation mode. Server-only
    /// options (the endpoint) are dropped.
    fn chat_argv(&self, ctx: &LaunchContext) -> Option<Vec<String>> {
        let binary = cli_binary(ctx.binary)?;
        let options: Vec<OptionItem> =
            ctx.options.iter().filter(|o| o.key != "host" && o.key != "port").cloned().collect();
        let sub = LaunchContext { binary: &binary, model: ctx.model, options: &options };
        let mut argv = self.build_command(&sub).argv;
        argv.push("-cnv".into());
        Some(argv)
    }

    fn bench_argv(&self, ctx: &LaunchContext) -> Option<Vec<String>> {
        let bench = self.runtime.bench_path.as_ref()?.display().to_string();
        let mut argv = vec![bench, "-m".into(), ctx.model.path.display().to_string()];
        for key in ["device", "gpu-layers"] {
            if let Some(opt) = ctx.options.iter().find(|o| o.key == key && o.value != DEFAULT) {
                argv.push(if key == "device" { "--device".into() } else { "-ngl".into() });
                argv.push(opt.value.clone());
            }
        }
        Some(argv)
    }

    /// llama-server returns `200` on `GET /health` once the model is loaded and
    /// `503` while it is still loading.
    fn health_path(&self) -> &'static str {
        "/health"
    }

    /// A remote launch names the repo-relative filename that llama.cpp will
    /// fetch; a local one names the GGUF path.
    fn process_token(&self, ctx: &LaunchContext) -> String {
        match remote_launch(ctx.model, mtp_enabled(ctx.options)) {
            Some(remote) => remote.file.clone().unwrap_or_default(),
            None => ctx.model.path.display().to_string(),
        }
    }

    /// A launch that makes llama.cpp fetch its own artifacts needs flags that
    /// older builds do not have. Checking here turns an opaque server error into
    /// an explanation.
    fn launch_blocker(&self, ctx: &LaunchContext) -> Option<String> {
        let mtp_enabled = mtp_enabled(ctx.options);
        let remote = remote_launch(ctx.model, mtp_enabled)?;
        if !self.hf_supported {
            return Some(
                "this llama-server does not advertise --hf-repo/--hf-file; upgrade llama.cpp"
                    .into(),
            );
        }
        let mtp_path = mtp_enabled.then_some(ctx.model.mtp_path.as_ref()).flatten();
        if draft_hf(remote, mtp_enabled, &mtp_path.map(|p| p.display().to_string())).is_some()
            && !self.draft_hf_supported
        {
            return Some(
                "this llama-server does not advertise --spec-draft-hf; upgrade llama.cpp or download the MTP companion first"
                    .into(),
            );
        }
        if ctx.model.projector_path.is_none()
            && remote.projector_file.is_some()
            && !self.mmproj_auto_supported
        {
            return Some(
                "this llama-server does not advertise --mmproj-auto; upgrade llama.cpp or download the projector first"
                    .into(),
            );
        }
        None
    }

    /// The Hub blobs llama.cpp will pull, so the session shows download
    /// progress rather than sitting in `Starting` for minutes.
    fn launch_download(&self, ctx: &LaunchContext) -> Option<DownloadRecord> {
        let remote = remote_launch(ctx.model, mtp_enabled(ctx.options))?;
        let blobs: Vec<DownloadBlob> = remote
            .blobs
            .iter()
            .filter(|blob| {
                remote.mtp_file.as_deref() != Some(blob.file.as_str())
                    && remote.projector_file.as_deref() != Some(blob.file.as_str())
            })
            .filter_map(|blob| {
                let (incomplete_file, complete_file) =
                    crate::discovery::online::cache_blob_paths(&remote.repo, &blob.oid)?;
                Some(DownloadBlob {
                    incomplete_file,
                    complete_file,
                    expected_bytes: blob.size_bytes,
                })
            })
            .collect();
        (!blobs.is_empty()).then_some(DownloadRecord { blobs })
    }

    fn supports_online_browse(&self) -> bool {
        true
    }

    fn unavailable_reason(&self) -> Option<String> {
        self.runtime
            .binary_path
            .is_none()
            .then(|| "llama-server binary not found on PATH".to_string())
    }
}

/// Is llama.cpp's MTP drafter selected in the resolved options?
fn mtp_enabled(options: &[OptionItem]) -> bool {
    options.iter().any(|o| o.key == "spec-type" && o.value == "draft-mtp")
}

/// Registry default, overridden by config for host/port.
fn config_default(spec: &OptionSpec, defaults: &Defaults) -> String {
    match spec.key {
        "host" => defaults.host.clone(),
        "port" => defaults.port.to_string(),
        _ => spec.default.to_string(),
    }
}

/// The remote identity to launch from, when one or more required artifacts are
/// not in the local cache and llama.cpp must fetch them itself.
fn remote_launch(model: &Model, mtp_enabled: bool) -> Option<&RemoteModel> {
    let remote = model.remote.as_ref()?;
    let base_missing = model.path.as_os_str().is_empty();
    let mtp_missing = mtp_enabled && model.mtp_path.is_none() && remote.mtp_file.is_some();
    let projector_missing = model.projector_path.is_none() && remote.projector_file.is_some();
    (base_missing || mtp_missing || projector_missing).then_some(remote)
}

/// The `--spec-draft-hf` repository for an MTP companion that has to be fetched.
fn draft_hf(remote: &RemoteModel, mtp_enabled: bool, mtp_path: &Option<String>) -> Option<String> {
    if !mtp_enabled || mtp_path.is_some() {
        return None;
    }
    draft_hf_repository(&remote.repo, remote.mtp_file.as_deref()?)
}

/// Prefer `llama-bench` beside `llama-server`, then fall back to PATH for
/// installations that package the tools separately.
fn resolve_bench(server: &Path) -> Option<PathBuf> {
    let sibling = server.with_file_name("llama-bench");
    sibling.is_file().then_some(sibling).or_else(|| super::resolve_binary("llama-bench"))
}

/// `llama-cli` beside the configured `llama-server`.
fn cli_binary(server: &str) -> Option<String> {
    let path = Path::new(server);
    let file = path.file_name()?.to_string_lossy();
    if !file.contains("llama-server") {
        return None;
    }
    Some(path.with_file_name(file.replace("llama-server", "llama-cli")).display().to_string())
}

/// Run `--version` and return a short version string. llama.cpp prints version
/// info to stderr, so both streams are considered.
fn query_version(path: &Path) -> Option<String> {
    let output = supervisor::output(ProcCommand::new(path).arg("--version")).ok()?;
    let text = if output.stderr.is_empty() { &output.stdout } else { &output.stderr };
    let text = String::from_utf8_lossy(text);
    text.lines().map(str::trim).find(|l| !l.is_empty()).map(|l| l.to_string())
}

/// Run `--list-devices` and extract device identifiers from lines such as
/// `ROCm0: AMD Radeon ...`. Both streams are considered because llama.cpp's
/// informational output has moved between stdout and stderr across versions.
fn query_devices(path: &Path) -> Vec<String> {
    let Ok(output) = supervisor::output(ProcCommand::new(path).arg("--list-devices")) else {
        return Vec::new();
    };
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push('\n');
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    parse_devices(&text)
}

fn parse_devices(output: &str) -> Vec<String> {
    let mut devices = Vec::new();
    let mut in_device_list = false;
    for line in output.lines().map(str::trim) {
        if line.eq_ignore_ascii_case("Available devices:") {
            in_device_list = true;
            continue;
        }
        if !in_device_list {
            continue;
        }
        let Some((name, _description)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty()
            || name.chars().any(char::is_whitespace)
            || !name.ends_with(|c: char| c.is_ascii_digit())
        {
            continue;
        }
        if !devices.iter().any(|device| device == name) {
            devices.push(name.to_string());
        }
    }
    devices
}

/// Capture `--help` to `<cache_dir>/llama-server.help.txt`.
fn cache_help(path: &Path, cache_dir: &Path) -> std::io::Result<()> {
    let output = supervisor::output(ProcCommand::new(path).arg("--help"))?;
    let body = if output.stdout.is_empty() { output.stderr } else { output.stdout };
    std::fs::write(cache_dir.join(HELP_CACHE), body)
}

/// The `--spec-draft-hf` argument for an MTP companion in `repo`.
///
/// A quantized companion is addressed as `repo:QUANT`; one that lives in a
/// subdirectory needs the bare repo. A plain root-level `mtp-*.gguf` needs no
/// argument at all — recent llama.cpp finds it beside the `-hf` model.
fn draft_hf_repository(repo: &str, file: &str) -> Option<String> {
    let quant = crate::discovery::models::quant_from_filename(file);
    if !file.contains('/') && quant.is_none() {
        return None;
    }
    Some(quant.map(|quant| format!("{repo}:{quant}")).unwrap_or_else(|| repo.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_range_is_enforced() {
        let kind = SCHEMA.spec("gpu-layers").unwrap().kind;
        assert_eq!(kind.validate("50").unwrap(), "50");
        assert!(kind.validate("1000").is_err()); // > 999
        assert!(kind.validate("-1").is_err()); // < 0
        assert!(kind.validate("abc").is_err());
    }

    #[test]
    fn float_range_is_enforced() {
        let kind = SCHEMA.spec("temperature").unwrap().kind;
        assert_eq!(kind.validate("0.7").unwrap(), "0.7");
        assert!(kind.validate("3.0").is_err()); // > 2.0
    }

    #[test]
    fn flash_attn_is_an_enum_dropped_when_auto() {
        let spec = SCHEMA.spec("flash-attn").unwrap();
        assert_eq!(spec.kind.validate("OFF").unwrap(), "off");
        assert!(spec.kind.validate("true").is_err()); // legacy bool is not a variant
        // "auto" is the omitted state; it cycles like any variant (no sentinel).
        assert_eq!(SCHEMA.omit_token("flash-attn"), Some("auto"));
        assert_eq!(SCHEMA.bump(spec, &spec.kind, "auto", 1), Some("on".into()));
        assert_eq!(spec.kind.extreme(-1), Some("auto".into())); // Home → first variant
    }

    #[test]
    fn numeric_omittables_fold_the_default_sentinel() {
        assert_eq!(SCHEMA.omit_token("batch-size"), Some(DEFAULT));
        assert_eq!(SCHEMA.omit_token("threads"), Some(DEFAULT));
        // The sampling params and ctx-size are omittable too.
        for key in ["ctx-size", "temperature", "top-p", "top-k", "min-p", "repeat-penalty"] {
            assert_eq!(SCHEMA.omit_token(key), Some(DEFAULT), "{key} should fold the sentinel");
            assert!(SCHEMA.uses_sentinel(key), "{key} should get sentinel affordances");
        }
        // host/port stay on the command line: llmctl needs the endpoint.
        assert_eq!(SCHEMA.omit_token("host"), None);
        assert_eq!(SCHEMA.omit_token("port"), None);

        // Stepping up from DEFAULT enters the concrete base; stepping down stays.
        let ngl = SCHEMA.spec("gpu-layers").unwrap();
        assert_eq!(SCHEMA.bump(ngl, &ngl.kind, DEFAULT, 1), Some("999".into()));
        assert_eq!(SCHEMA.bump(ngl, &ngl.kind, DEFAULT, -1), Some(DEFAULT.into()));
        // Home/End are pure min/max jumps; resetting to DEFAULT is `d` (app-level).
        assert_eq!(ngl.kind.extreme(-1), Some("0".into())); // Home → min
        assert_eq!(ngl.kind.extreme(1), Some("999".into())); // End → max
    }

    #[test]
    fn device_uses_runtime_selector_and_is_omitted_at_default() {
        let device = SCHEMA.spec("device").unwrap();
        assert_eq!(device.cli, "--device");
        assert_eq!(SCHEMA.omit_token("device"), Some(DEFAULT));
        assert!(SCHEMA.uses_sentinel("device"));
    }

    #[test]
    fn adjust_clamps_numeric_and_cycles_enum() {
        let temp = SCHEMA.spec("temperature").unwrap();
        assert_eq!(temp.kind.adjust("1.95", 1, temp.step), Some("2".into())); // clamp at 2.0
        assert_eq!(temp.kind.adjust("0.8", -1, temp.step), Some("0.75".into()));

        let cache = SCHEMA.spec("cache-type-k").unwrap().kind;
        assert_eq!(cache.adjust("f16", 1, 1.0), Some("q8_0".into()));
        assert_eq!(cache.adjust("f16", -1, 1.0), Some("default".into())); // back toward "default"
    }

    #[test]
    fn cache_types_omit_at_their_default_variant_without_sentinel_affordances() {
        for key in ["cache-type-k", "cache-type-v"] {
            // "default" is the omitted state, but it's an in-band enum variant —
            // not the numeric sentinel — so it cycles like any other choice.
            assert_eq!(SCHEMA.omit_token(key), Some(DEFAULT));
            assert!(!SCHEMA.uses_sentinel(key));
            let s = SCHEMA.spec(key).unwrap();
            assert_eq!(SCHEMA.bump(s, &s.kind, "default", 1), Some("f16".into()));
            assert_eq!(s.kind.extreme(-1), Some("default".into())); // Home → first variant
            assert_eq!(s.kind.extreme(1), Some("q4_0".into())); // End → last
        }
    }

    #[test]
    fn speculative_options_have_proper_omit_tokens() {
        // spec-type omits at its in-band "none" variant (cycles like an enum).
        assert_eq!(SCHEMA.omit_token("spec-type"), Some("none"));
        assert!(!SCHEMA.uses_sentinel("spec-type"));
        let st = SCHEMA.spec("spec-type").unwrap();
        assert_eq!(SCHEMA.bump(st, &st.kind, "none", 1), Some("draft-simple".into()));

        // The draft-count ints fold the numeric "default" sentinel.
        let n_max = SCHEMA.spec("spec-draft-n-max").unwrap();
        assert_eq!(SCHEMA.omit_token("spec-draft-n-max"), Some(DEFAULT));
        assert!(SCHEMA.uses_sentinel("spec-draft-n-max"));
        assert_eq!(SCHEMA.bump(n_max, &n_max.kind, DEFAULT, 1), Some("3".into())); // step up enters base
        assert_eq!(SCHEMA.spec("spec-draft-n-min").unwrap().default, "0");
    }

    #[test]
    fn reasoning_effort_is_a_json_kwarg_enum_omitted_at_default() {
        // "default" is the omitted state, an in-band enum variant (no sentinel
        // affordances) — it cycles like the cache types.
        assert_eq!(SCHEMA.omit_token("reasoning-effort"), Some(DEFAULT));
        assert!(!SCHEMA.uses_sentinel("reasoning-effort"));
        let s = SCHEMA.spec("reasoning-effort").unwrap();
        assert_eq!(SCHEMA.bump(s, &s.kind, "default", 1), Some("low".into()));
        assert_eq!(s.kind.extreme(1), Some("high".into())); // End → high

        // The emitted argv token is the chat-template kwargs JSON, not the raw value.
        assert_eq!(SCHEMA.cli_value("reasoning-effort", "high"), r#"{"reasoning_effort":"high"}"#);
        assert_eq!(SCHEMA.cli_value("temperature", "0.7"), "0.7"); // everything else passes through
    }

    #[test]
    fn mmap_is_a_flag_omitted_when_on() {
        assert!(SCHEMA.is_flag("mmap"));
        assert_eq!(SCHEMA.omit_token("mmap"), Some("on")); // on = llama default, omitted
        let s = SCHEMA.spec("mmap").unwrap();
        assert_eq!(SCHEMA.bump(s, &s.kind, "on", 1), Some("off".into())); // `e` toggles
    }

    #[test]
    fn jinja_is_a_flag_omitted_when_on() {
        // Same shape as mmap: on = llama.cpp's default (omitted); off emits
        // the bare --no-jinja flag.
        assert!(SCHEMA.is_flag("jinja"));
        assert_eq!(SCHEMA.omit_token("jinja"), Some("on"));
        let s = SCHEMA.spec("jinja").unwrap();
        assert_eq!(s.cli, "--no-jinja");
        assert_eq!(SCHEMA.bump(s, &s.kind, "on", 1), Some("off".into())); // `e` toggles
    }

    #[test]
    fn chat_template_is_an_enum_of_builtins_omitted_at_default() {
        assert_eq!(SCHEMA.omit_token("chat-template"), Some(DEFAULT));
        assert!(!SCHEMA.uses_sentinel("chat-template")); // in-band variant, cycles
        let s = SCHEMA.spec("chat-template").unwrap();
        assert_eq!(s.kind.extreme(-1), Some("default".into())); // Home → default
        assert_eq!(SCHEMA.bump(s, &s.kind, "default", 1), Some("bailing".into()));
        assert_eq!(s.kind.validate("LLAMA3").unwrap(), "llama3"); // case-folded
        assert!(s.kind.validate("not-a-template").is_err());
    }

    #[test]
    fn extreme_jumps_to_bounds() {
        let port = SCHEMA.spec("port").unwrap().kind;
        assert_eq!(port.extreme(-1), Some("1".into()));
        assert_eq!(port.extreme(1), Some("65535".into()));
        let cache = SCHEMA.spec("cache-type-k").unwrap().kind;
        assert_eq!(cache.extreme(1), Some("q4_0".into()));
    }
}
