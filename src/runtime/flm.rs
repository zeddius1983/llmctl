//! The FastFlowLM backend: NPU inference via `flm` on AMD XDNA2 hardware.
//!
//! FastFlowLM differs from llama.cpp in two ways that shape this module:
//!
//! * **The catalog is curated, not scanned.** `flm list --json` returns every
//!   model the runtime knows about — installed or not — with its context
//!   length, quantization, footprint, and capability labels. There is no
//!   filesystem walk and no separate "online" tree; one call is the whole
//!   catalog, so a not-yet-downloaded model is fully browsable.
//! * **Models are tags, not paths.** `qwen3:4b` is the identity used to serve,
//!   pull, and remove. A model's on-disk directory is an implementation detail
//!   of `flm`, so the tag is what llmctl records and matches processes against.

use std::path::{Path, PathBuf};
use std::process::Command as ProcCommand;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;
use tracing::{debug, warn};

use crate::config::{Defaults, FastFlowLmConfig};
use crate::discovery::hf;
use crate::domain::{FlmModel, Model, OptionItem, Runtime};
use crate::profiles::registry::{OptionKind, OptionSchema, OptionSpec};
use crate::profiles::templates::Template;
use crate::runtime::{CatalogCtx, Deletion, LaunchContext, RuntimeBackend, tree_bytes};
use crate::session::command::Command;
use crate::session::record::{DownloadBlob, DownloadRecord};
use crate::session::throughput::{Phase, Sample};

pub const NAME: &str = "FastFlowLM";

/// Top-level split, mirroring llama.cpp's: what you have, and what you can get.
const LOCAL_GROUP: &str = "local";
const ONLINE_GROUP: &str = "online";
/// Fallback capability group for models that carry no labels.
const CHAT_GROUP: &str = "chat";
/// This runtime's subtree under llmctl's managed catalog, which holds the
/// per-model profile directories.
const CATALOG_ROOT: &str = "fastflowlm";

/// How the catalog is arranged, cycled with `s`. Hugging Face sort orders mean
/// nothing here — the catalog is a fixed list, not a ranked feed — so the choice
/// on offer is how to group it.
pub static VIEWS: &[&str] = &["Categories", "Flat"];
/// Index into [`VIEWS`] for the ungrouped arrangement.
const FLAT_VIEW: usize = 1;

use OptionKind::{Enum, Int, Str};

/// FastFlowLM's option set, from `flm help` (v0.9.45).
///
/// Note that `flm` takes an explicit value for every flag — there are no
/// valueless `--no-*` inversions — so the CLI dialect is a straight
/// `--flag value` mapping.
pub static SPECS: &[OptionSpec] = &[
    OptionSpec {
        key: "ctx-len",
        cli: "--ctx-len",
        kind: Int { min: Some(0), max: None },
        default: "4096",
        step: 1024.0,
        description: "Context length in tokens (rounded to the nearest power of two; \
                      capped at the model's trained context).",
    },
    OptionSpec {
        key: "prefill-chunk-len",
        cli: "--prefill-chunk-len",
        kind: Int { min: Some(0), max: None },
        default: "4096",
        step: 1024.0,
        description: "Prompt prefill chunk size ('default' uses the model's own \
                      maximum prefill length).",
    },
    OptionSpec {
        key: "pmode",
        cli: "--pmode",
        kind: Enum(&["powersaver", "balanced", "performance", "turbo"]),
        default: "performance",
        step: 1.0,
        description: "NPU power mode: trades sustained throughput against power draw.",
    },
    OptionSpec {
        key: "q-len",
        cli: "--q-len",
        kind: Int { min: Some(1), max: Some(256) },
        default: "10",
        step: 1.0,
        description: "Maximum NPU queue length — how many requests may be in flight.",
    },
    OptionSpec {
        key: "socket",
        cli: "--socket",
        kind: Int { min: Some(1), max: Some(256) },
        default: "10",
        step: 1.0,
        description: "Maximum number of concurrent socket connections.",
    },
    OptionSpec {
        key: "cors",
        cli: "--cors",
        kind: Enum(&["1", "0"]),
        default: "1",
        step: 1.0,
        description: "Cross-Origin Resource Sharing (1 = enabled, the FastFlowLM default).",
    },
    OptionSpec {
        key: "preemption",
        cli: "--preemption",
        kind: Enum(&["0", "1"]),
        default: "0",
        step: 1.0,
        description: "Allow a queued request to preempt one in flight.",
    },
    OptionSpec {
        key: "embed",
        cli: "--embed",
        kind: Enum(&["0", "1"]),
        default: "0",
        step: 1.0,
        description: "Also load the embedding model, serving /v1/embeddings.",
    },
    OptionSpec {
        key: "asr",
        cli: "--asr",
        kind: Enum(&["0", "1"]),
        default: "0",
        step: 1.0,
        description: "Also load the speech model, serving /v1/audio/transcriptions.",
    },
    OptionSpec {
        key: "img-pre-resize",
        cli: "--img-pre-resize",
        kind: Enum(&["0", "1", "2", "3", "4"]),
        default: "2",
        step: 1.0,
        description: "Pre-resize input images for vision models \
                      (0 = original, 1 = 480p, 2 = 720p, 3 = 1080p, 4 = 1440p).",
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
        default: "52625",
        step: 1.0,
        description: "TCP port the server listens on (FastFlowLM's own default is 52625).",
    },
];

/// Options `flm` applies its own default to when the flag is absent. Everything
/// else is always emitted.
fn omit_token(key: &str) -> Option<&'static str> {
    match key {
        // host/port are never omitted: llmctl needs the concrete endpoint for
        // health checks, and an explicit --port is what lets us re-acquire the
        // server process by command line (see `session::proc::find_server`).
        "host" | "port" => None,
        _ => Some(crate::profiles::registry::DEFAULT),
    }
}

/// `flm` has no valueless flags — even booleans take `0`/`1`.
fn is_flag(_key: &str) -> bool {
    false
}

fn cli_value(_key: &str, value: &str) -> String {
    value.to_string()
}

/// FastFlowLM's option vocabulary bound to its CLI dialect.
pub static SCHEMA: OptionSchema = OptionSchema { specs: SPECS, omit_token, is_flag, cli_value };

/// FastFlowLM's built-in profile templates.
///
/// These share names with llama.cpp's where the intent matches, but speak
/// FastFlowLM's option keys — there is no `gpu-layers` or `flash-attn` on an NPU.
pub static TEMPLATES: &[Template] = &[
    Template { name: "Default", overrides: &[] },
    Template { name: "Chat", overrides: &[("pmode", "performance")] },
    Template {
        name: "Long Context",
        overrides: &[("ctx-len", "131072"), ("prefill-chunk-len", "8192")],
    },
    Template {
        name: "Server",
        overrides: &[("host", "0.0.0.0"), ("q-len", "20"), ("socket", "20"), ("preemption", "1")],
    },
    Template { name: "Low Power", overrides: &[("pmode", "powersaver")] },
];

/// The FastFlowLM backend: a discovered `flm` plus the NPU stack's readiness.
pub struct FlmBackend {
    runtime: Runtime,
    /// `flm validate` reported a usable NPU stack. Discovery succeeding only
    /// means the binary exists; the driver, firmware, and memlock limits are a
    /// separate question, and getting it wrong fails at load time with an
    /// opaque device error.
    npu_ready: bool,
    /// Why the stack is not ready, when it isn't.
    npu_problem: Option<String>,
    /// The kernel's `amdxdna` driver has an NPU bound, whether or not `flm` is
    /// installed to use it.
    npu_present: bool,
    /// The last catalog read from `flm`, held so re-grouping does not re-read
    /// it. See [`FlmBackend::cached_catalog`].
    catalog_cache: Mutex<Option<Vec<Entry>>>,
}

impl FlmBackend {
    /// Discover FastFlowLM: locate `flm`, read its version, and validate the
    /// NPU stack.
    pub fn discover(cfg: &FastFlowLmConfig) -> Self {
        let binary_path = super::resolve_binary(&cfg.binary);
        if binary_path.is_none() {
            warn!(binary = %cfg.binary, "flm binary not found");
        }
        let version = binary_path.as_deref().and_then(query_version);
        let validation = binary_path.as_deref().and_then(validate);

        let devices = validation
            .as_ref()
            .map(|v| v.devices.iter().map(Device::label).collect())
            .unwrap_or_default();
        let npu_ready = validation.as_ref().is_some_and(|v| v.ready);
        let npu_problem = validation.as_ref().and_then(Validation::problem);

        Self {
            runtime: Runtime {
                name: NAME.into(),
                description: "AMD NPU inference via flm".into(),
                version: version.map(|v| format!("flm {v}")),
                // `flm bench` is a *hidden* subcommand: it is missing from
                // `flm --help`, which is what led us to believe it did not
                // exist, but v0.9.45 parses and runs it. The benchmark tool is
                // therefore `flm` itself, not a sibling binary the way
                // llama.cpp ships `llama-bench` next to `llama-server`.
                bench_path: binary_path.clone(),
                binary_path,
                formats: vec!["NPU".into()],
                devices,
            },
            npu_ready,
            npu_problem,
            npu_present: npu_present(Path::new(AMDXDNA_DRIVER)),
            catalog_cache: Mutex::new(None),
        }
    }

    /// The `flm` catalog, read fresh from the binary, or an empty list if it
    /// can't be read.
    fn catalog(&self) -> Vec<Entry> {
        let Some(binary) = &self.runtime.binary_path else { return Vec::new() };
        match list_models(binary) {
            Ok(entries) => entries,
            Err(err) => {
                warn!(%err, "could not read the flm model catalog");
                Vec::new()
            }
        }
    }

    /// The catalog, served from memory unless `ctx.reload` asks for a fresh read.
    ///
    /// `flm list --json` costs ~150 ms here: `flm` is frequently a launcher
    /// script (a distrobox entry point on this machine), so every call pays a
    /// container hop. The result is pure data that changes only when a model is
    /// installed or removed — yet cycling the catalog arrangement with `s` used
    /// to re-read it just to regroup identical entries, turning an in-memory
    /// transform into a visible hitch. Callers that *can* have missed a change
    /// — the `F5` refresh, and finishing a download — pass `reload`.
    ///
    /// Materializing the managed profile directories belongs here rather than in
    /// [`RuntimeBackend::models`] for the same reason: it is a write that has to
    /// happen once per catalog read, not once per regroup.
    fn cached_catalog(&self, ctx: &CatalogCtx) -> Vec<Entry> {
        if !ctx.reload
            && let Ok(cache) = self.catalog_cache.lock()
            && let Some(entries) = cache.as_ref()
        {
            return entries.clone();
        }

        let entries = self.catalog();
        for entry in &entries {
            // One managed leaf per tag, however many groups the model shows up
            // in. Creating it is what makes profiles persist as per-model YAML:
            // the store only adopts a model whose `profiles/` directory exists.
            // llama.cpp gets the equivalent from `catalog::reconcile`, which
            // also symlinks the artifact — meaningless for FastFlowLM, whose
            // models are multi-file directories owned by `flm`.
            let leaf = ctx.models_dir.join(CATALOG_ROOT).join(sanitize(&entry.name));
            if let Err(err) = std::fs::create_dir_all(leaf.join("profiles")) {
                debug!(%err, path = %leaf.display(), "could not create the managed profile directory");
            }
        }
        if let Ok(mut cache) = self.catalog_cache.lock() {
            *cache = Some(entries.clone());
        }
        entries
    }
}

impl RuntimeBackend for FlmBackend {
    fn id(&self) -> super::RuntimeId {
        super::RuntimeId(NAME.into())
    }

    fn download_available(&self, model: &Model) -> bool {
        model.flm().is_some_and(|flm| !flm.installed) || unfinished(model)
    }

    fn launch_blocker(&self, ctx: &super::LaunchContext) -> Option<String> {
        // `flm serve` would load whatever is under the real names, which for
        // an interrupted update is new metadata over old weights, and die.
        unfinished(ctx.model())
            .then(|| "its download did not finish; press d on the model to resume it".to_string())
    }

    fn model_transfer(&self, model: &Model) -> Option<super::ModelTransfer> {
        // `installed` is `flm`'s word, and an update llmctl did not finish can
        // leave it saying yes over a mix of two revisions; see [`unfinished`].
        if model.flm()?.installed && !unfinished(model) {
            return None;
        }
        Some(super::ModelTransfer {
            runtime: self.id(),
            model: Box::new(model.clone()),
            targets: model_dir(model).into_iter().collect(),
            run: |model, cancelled, progress| {
                download(model, cancelled, progress)
                    .map(|outcome| match outcome {
                        DownloadOutcome::Downloaded(path) => {
                            crate::discovery::online::DownloadResult::Downloaded(path)
                        }
                        DownloadOutcome::Cancelled => {
                            crate::discovery::online::DownloadResult::Cancelled
                        }
                    })
                    .map_err(anyhow::Error::msg)
            },
        })
    }

    fn descriptor(&self) -> &Runtime {
        &self.runtime
    }

    fn schema(&self) -> &'static OptionSchema {
        &SCHEMA
    }

    fn templates(&self) -> &'static [Template] {
        TEMPLATES
    }

    /// One `Model` per (group, tag) pair. A model carrying three labels is
    /// emitted three times so it appears under each; they share a
    /// `profile_key`, so they are one model as far as profiles are concerned.
    fn catalog_views(&self) -> &'static [&'static str] {
        VIEWS
    }

    /// A pure regroup of the cached catalog: switching arrangement costs the
    /// transform and nothing else.
    fn models(&self, ctx: &CatalogCtx) -> Vec<Model> {
        let root = model_root();
        let flat = ctx.view == FLAT_VIEW;
        let mut models = Vec::new();
        for entry in self.cached_catalog(ctx) {
            for group in entry.groups(flat) {
                models.push(entry.to_model(&group, &root, ctx.models_dir));
            }
        }
        models
    }

    /// FastFlowLM only ever runs on the XDNA2 NPU; that is the whole point of
    /// the runtime.
    fn device_label(&self, _options: &[OptionItem]) -> Option<String> {
        Some("NPU".into())
    }

    /// `flm` gives each model a directory of its own under its model root, so
    /// removing one is removing that directory — including any `.llmctl-part`
    /// scratch a cancelled download left in it.
    fn deletion(&self, model: &Model, catalog: &[Model]) -> Option<Deletion> {
        let dir = model_dir(model)?;
        if !dir.is_dir() {
            return None;
        }
        // A directory is named after the repository, not the tag. Two installed
        // tags resolving to one directory would make this a shared deletion;
        // the catalog has no such pair today, but do not find out the hard way.
        let tag = model.flm().map(|flm| flm.tag.as_str());
        if catalog.iter().any(|other| {
            other.flm().is_some_and(|flm| flm.installed && Some(flm.tag.as_str()) != tag)
                && model_dir(other).as_ref() == Some(&dir)
        }) {
            return None;
        }
        Some(Deletion { bytes: tree_bytes(&dir), trees: vec![dir], ..Deletion::default() })
    }

    /// `ctx-len` is bounded by the model's trained context.
    fn effective_kind(&self, spec: &OptionSpec, model: &Model) -> OptionKind {
        match (spec.key, model.context_length) {
            ("ctx-len", Some(ctx)) => {
                OptionKind::Int { min: Some(0), max: Some(i64::try_from(ctx).unwrap_or(i64::MAX)) }
            }
            _ => spec.kind,
        }
    }

    /// Unlike llama.cpp, `flm` sizes its own KV cache sensibly and the catalog
    /// states each model's intended context, so `ctx-len` starts at the model's
    /// default rather than a fraction of it.
    fn spec_default(&self, spec: &OptionSpec, model: &Model, defaults: &Defaults) -> String {
        match spec.key {
            "ctx-len" => match model.context_length {
                Some(ctx) => ctx.to_string(),
                None => spec.default.to_string(),
            },
            "prefill-chunk-len" => match model.flm().and_then(|f| f.max_prefill_len) {
                Some(len) => len.to_string(),
                None => crate::profiles::registry::DEFAULT.to_string(),
            },
            // The global default host applies, but not the global port: 8000 is
            // llama.cpp's convention and FastFlowLM has its own.
            "host" => defaults.host.clone(),
            _ => match SCHEMA.omit_token(spec.key) {
                Some(token) => token.to_string(),
                None => spec.default.to_string(),
            },
        }
    }

    fn clamp_to_model(&self, key: &str, value: String, model: &Model) -> String {
        if key != "ctx-len" {
            return value;
        }
        match (model.context_length, value.parse::<i64>()) {
            (Some(ctx), Ok(v)) if v > i64::try_from(ctx).unwrap_or(i64::MAX) => ctx.to_string(),
            _ => value,
        }
    }

    /// `flm serve <tag> [--flag value …]`.
    ///
    /// `--quiet` is always passed: the docs call it "for sub-process usages",
    /// and it keeps the detached session's log readable.
    fn build_command(&self, ctx: &LaunchContext) -> Command {
        let mut argv =
            vec![ctx.binary.to_string(), "serve".into(), tag(ctx.model()), "--quiet".into()];
        Command::append_options(&mut argv, &SCHEMA, ctx.options);
        Command { argv }
    }

    /// `flm run <tag>` is FastFlowLM's interactive mode. Server-only options
    /// (the endpoint, the queue, CORS) are meaningless there.
    fn chat_argv(&self, ctx: &LaunchContext) -> Option<Vec<String>> {
        let mut argv = vec![ctx.binary.to_string(), "run".into(), tag(ctx.model())];
        let options: Vec<OptionItem> = ctx
            .options
            .iter()
            .filter(|o| !matches!(o.spec.key, "host" | "port" | "q-len" | "socket" | "cors"))
            .cloned()
            .collect();
        Command::append_options(&mut argv, &SCHEMA, &options);
        Some(argv)
    }

    /// `flm bench <tag>` runs FastFlowLM's own multi-stage throughput
    /// benchmark on the NPU.
    ///
    /// Only `--pmode` carries over from the profile. The benchmark picks its
    /// own context lengths per stage (32k upward), so `ctx-len` and
    /// `prefill-chunk-len` would be overridden; it never opens a socket, so the
    /// server options are meaningless; and `asr`/`embed`/`img-pre-resize` only
    /// load extra models that the benchmark does not exercise.
    fn bench_argv(&self, ctx: &LaunchContext) -> Option<Vec<String>> {
        let mut argv = vec![ctx.binary.to_string(), "bench".into(), tag(ctx.model())];
        let options: Vec<OptionItem> =
            ctx.options.iter().filter(|o| o.spec.key == "pmode").cloned().collect();
        Command::append_options(&mut argv, &SCHEMA, &options);
        Some(argv)
    }

    /// FastFlowLM exposes no `/health`; `GET /v1/models` answers 200 once the
    /// server is up, which is the same signal.
    fn health_path(&self) -> &'static str {
        "/v1/models"
    }

    /// The tag, which is what appears in `flm serve <tag>`.
    fn process_token(&self, ctx: &LaunchContext) -> String {
        tag(ctx.model())
    }

    /// Launching a model that isn't downloaded lets `flm serve` fetch it
    /// itself, mirroring llama.cpp's native `-hf` launch. Tracking the expected
    /// files means the session shows `Downloading (N%)` instead of sitting in
    /// `Starting` for several minutes with nothing to show.
    ///
    /// Unlike llmctl's own downloads, `flm` writes straight to the final
    /// filenames, so there is no rename to observe. `complete_file` therefore
    /// points at a sentinel that never exists, which keeps progress tracking
    /// byte growth rather than jumping to 100% the moment a file is created.
    /// The `Downloading` state ends when the health probe reports ready, so the
    /// sentinel never strands a session.
    fn launch_download(&self, ctx: &LaunchContext) -> Option<DownloadRecord> {
        let flm = ctx.model().flm()?;
        if flm.installed {
            return None;
        }
        let blobs = expected_files(ctx.model())?
            .into_iter()
            .map(|(path, expected_bytes)| DownloadBlob {
                complete_file: path.with_extension("llmctl-never-complete"),
                incomplete_file: path,
                expected_bytes,
            })
            .collect::<Vec<_>>();
        (!blobs.is_empty()).then_some(DownloadRecord { blobs })
    }

    /// The XDNA driver hands out one hardware context at a time. A second
    /// `flm serve`/`flm run` starts, gets as far as loading the model, and dies
    /// with `DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid
    /// argument` — so llmctl declines it instead of leaving a crashed session.
    fn single_session(&self) -> bool {
        true
    }

    /// Listed where there is an NPU to use or an `flm` to use it with. A
    /// machine with neither — no XDNA hardware, nothing installed — never
    /// shows FastFlowLM; one with the NPU but no `flm` shows it dimmed, so the
    /// option is discoverable.
    fn listed(&self) -> bool {
        self.runtime.binary_path.is_some() || self.npu_present
    }

    fn unavailable_reason(&self) -> Option<String> {
        if self.runtime.binary_path.is_none() {
            return Some("flm binary not found on PATH".into());
        }
        if !self.npu_ready {
            let detail = self.npu_problem.clone().unwrap_or_else(|| "run `flm validate`".into());
            return Some(format!("the AMD NPU stack is not ready: {detail}"));
        }
        None
    }
}

/// Where the kernel lists the devices bound to the AMD XDNA (NPU) driver.
const AMDXDNA_DRIVER: &str = "/sys/bus/pci/drivers/amdxdna";

/// Whether `driver` has a device bound: an entry named like a PCI address
/// (`0000:c8:00.1`) beside the driver's own `bind`, `module` and `uevent`.
fn npu_present(driver: &Path) -> bool {
    std::fs::read_dir(driver).is_ok_and(|entries| {
        entries.filter_map(Result::ok).any(|entry| {
            entry.file_name().to_str().is_some_and(|name| name.split(':').count() == 3)
        })
    })
}

/// The tag to serve. Falls back to the display name so a malformed catalog
/// entry still produces a runnable-looking command rather than an empty one.
fn tag(model: &Model) -> String {
    model.flm().map(|f| f.tag.clone()).unwrap_or_else(|| model.name.clone())
}

/// Where `flm` keeps a model's files. `flm` names the directory after the
/// repository alone, without its owner.
pub fn model_dir(model: &Model) -> Option<PathBuf> {
    model_dir_in(&model_root(), model)
}

/// [`model_dir`] against a given root, so the guards below can be tested
/// without an environment variable the rest of the suite also reads.
fn model_dir_in(root: &Path, model: &Model) -> Option<PathBuf> {
    let flm = model.flm()?;
    let name = repo_dir_name(&flm.repo)?;
    // Only an absolute root locates storage. An empty one — `FLM_MODEL_PATH`
    // set to nothing, or no home directory to resolve — makes the join below a
    // bare relative name, and so does a relative `FLM_MODEL_PATH`: either
    // resolves against whatever working directory llmctl happens to have, so a
    // deletion would recursively remove a same-named directory sitting beside
    // it rather than the model.
    if !root.is_absolute() {
        return None;
    }
    let dir = root.join(name);
    // Never the root itself. `Entry::url` is `#[serde(default)]`, so a catalog
    // row with a missing or unparseable URL yields an empty repository, and
    // `join("")` on the root is the root — which a deletion would then remove
    // recursively, taking every installed model with it.
    (dir.parent() == Some(root)).then_some(dir)
}

/// The directory `flm` stores a repository under: its last path segment, or
/// `None` when the repository does not name one.
fn repo_dir_name(repo: &str) -> Option<&str> {
    repo.rsplit('/').next().filter(|name| !name.is_empty() && *name != "." && *name != "..")
}

/// FastFlowLM's model root: `$FLM_MODEL_PATH`, else `~/.config/flm/models`.
fn model_root() -> PathBuf {
    if let Some(path) = std::env::var_os("FLM_MODEL_PATH") {
        return PathBuf::from(path);
    }
    directories::BaseDirs::new()
        .map(|dirs| dirs.home_dir().join(".config/flm/models"))
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// `flm` CLI plumbing
// ---------------------------------------------------------------------------

/// Every `flm` invocation prints a `[FLM] Fetching models from: …` banner to
/// stdout before its real output, so JSON parsing has to skip it.
fn strip_preamble(output: &str) -> &str {
    match output.find('{') {
        Some(start) => &output[start..],
        None => "",
    }
}

/// Run `flm <args…>` and return stdout with the banner stripped.
fn run(binary: &Path, args: &[&str]) -> Option<String> {
    let output = ProcCommand::new(binary).args(args).output().ok()?;
    if !output.status.success() {
        debug!(?args, status = ?output.status, "flm invocation failed");
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[derive(Debug, Deserialize)]
struct Version {
    version: String,
}

fn query_version(binary: &Path) -> Option<String> {
    let raw = run(binary, &["version", "--json"])?;
    serde_json::from_str::<Version>(strip_preamble(&raw)).ok().map(|v| v.version)
}

/// `flm validate --json`: the NPU stack's readiness, device by device.
#[derive(Debug, Deserialize)]
struct Validation {
    #[serde(default)]
    ready: bool,
    #[serde(default)]
    devices: Vec<Device>,
    #[serde(default)]
    amd_device_found: bool,
    #[serde(default)]
    kernel_ok: bool,
    #[serde(default)]
    memlock_ok: bool,
    #[serde(default)]
    all_fw_ok: bool,
}

#[derive(Debug, Deserialize)]
struct Device {
    #[serde(default)]
    device: String,
    #[serde(default)]
    fw_major: u32,
    #[serde(default)]
    fw_minor: u32,
    #[serde(default)]
    fw_patch: u32,
}

impl Device {
    /// `accel0 (fw 1.1.2)` — the device name plus its firmware, matching the
    /// shape of llama.cpp's `ROCm0`/`Vulkan0` device identifiers.
    fn label(&self) -> String {
        let name = Path::new(&self.device)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.device.clone());
        format!("{name} (fw {}.{}.{})", self.fw_major, self.fw_minor, self.fw_patch)
    }
}

impl Validation {
    /// The first unmet precondition, phrased as something the user can act on.
    fn problem(&self) -> Option<String> {
        if self.ready {
            return None;
        }
        Some(if !self.amd_device_found {
            "no AMD XDNA2 NPU found".into()
        } else if !self.kernel_ok {
            "the kernel or amdxdna driver is too old".into()
        } else if !self.all_fw_ok {
            "the NPU firmware is out of date".into()
        } else if !self.memlock_ok {
            "the memlock limit is too low (see /etc/security/limits.conf)".into()
        } else {
            "run `flm validate` for details".into()
        })
    }
}

fn validate(binary: &Path) -> Option<Validation> {
    let raw = run(binary, &["validate", "--json"])?;
    serde_json::from_str(strip_preamble(&raw)).ok()
}

/// `flm list --json`: the full curated catalog, installed and not.
#[derive(Debug, Deserialize)]
struct Catalog {
    #[serde(default)]
    models: Vec<Entry>,
}

#[derive(Debug, Clone, Deserialize)]
struct Entry {
    name: String,
    #[serde(default)]
    installed: bool,
    #[serde(default)]
    default_context_length: Option<u64>,
    #[serde(default)]
    max_prefill_len: Option<u64>,
    /// Disk footprint in GB. Note the sibling `size` field is a parameter
    /// count, not a byte count, and must not be used for disk size.
    #[serde(default)]
    footprint: f64,
    #[serde(default)]
    label: Vec<String>,
    #[serde(default)]
    vlm: bool,
    #[serde(default)]
    asr: bool,
    #[serde(default)]
    url: String,
    /// Exactly the files a model directory needs — confirmed against an
    /// installed model. Narrower than the repository, which also holds a README
    /// and the `.xclbin` NPU kernels that ship with `flm` itself.
    #[serde(default)]
    files: Vec<String>,
    #[serde(default)]
    details: Details,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Details {
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    quantization_level: Option<String>,
}

impl Entry {
    /// Where this model sits in the browser: `local` or `online`, then — in the
    /// Categories view — each capability label it carries (`chat` if it has
    /// none).
    ///
    /// Under Categories a model with three labels yields three positions; under
    /// Flat it yields exactly one. Either way they share a tag, and identity is
    /// the tag — see [`Model::profile_key`].
    fn groups(&self, flat: bool) -> Vec<Vec<String>> {
        let top = if self.installed { LOCAL_GROUP } else { ONLINE_GROUP };
        if flat {
            return vec![vec![top.to_string()]];
        }
        let labels: Vec<&str> = if self.label.is_empty() {
            vec![CHAT_GROUP]
        } else {
            self.label.iter().map(String::as_str).collect()
        };
        labels.into_iter().map(|label| vec![top.to_string(), label.to_string()]).collect()
    }

    /// The full Hugging Face repository id (`FastFlowLM/Qwen3-0.6B-NPU2`) —
    /// everything after `huggingface.co/` and before any `/resolve/<rev>`.
    fn repo(&self) -> String {
        let base = self.url.split("/resolve/").next().unwrap_or_default();
        base.split_once("huggingface.co/")
            .map(|(_, id)| id)
            .unwrap_or(base)
            .trim_matches('/')
            .to_string()
    }

    /// The revision to download from. Most models track `main`, but several are
    /// pinned to a tag (`v0.9.22-faster-q4-1`), and fetching `main` for those
    /// produces weights the installed `flm` cannot load.
    fn revision(&self) -> String {
        self.url
            .split_once("/resolve/")
            .map(|(_, rev)| rev.trim_end_matches('/'))
            .filter(|rev| !rev.is_empty())
            .unwrap_or("main")
            .to_string()
    }

    fn to_model(&self, group: &[String], root: &Path, models_dir: &Path) -> Model {
        let repo = self.repo();
        // No usable directory name means no local path: a catalog row whose
        // URL is missing must not resolve to the model root.
        let dir = repo_dir_name(&repo).map(|name| root.join(name));
        let mut catalog_path = group.to_vec();
        catalog_path.push(self.name.clone());
        Model {
            entry: crate::domain::CatalogEntry::Model(crate::domain::ModelSource::FastFlowLm(
                FlmModel {
                    tag: self.name.clone(),
                    installed: self.installed,
                    repo,
                    revision: self.revision(),
                    files: self.files.clone(),
                    labels: self.label.clone(),
                    vlm: self.vlm,
                    asr: self.asr,
                    max_prefill_len: self.max_prefill_len,
                },
            )),
            id: format!("flm:{}", self.name),
            name: self.name.clone(),
            // Only an installed model has a local path; a catalog entry that
            // hasn't been pulled is still a real, launchable model (`flm serve`
            // downloads it), which is why `is_catalog_dir` also consults `flm`.
            path: if self.installed { dir.unwrap_or_default() } else { PathBuf::new() },
            shard_paths: Vec::new(),
            mtp_path: None,
            dflash_path: None,
            dflash_block_size: None,
            projector_path: None,
            has_mtp: false,
            catalog_path,
            // One profile directory per tag, independent of how many groups
            // the model is rendered under.
            catalog_dir: models_dir.join(CATALOG_ROOT).join(sanitize(&self.name)),
            size_bytes: (self.footprint * 1e9) as u64,
            quantization: self.details.quantization_level.clone(),
            architecture: self.details.family.clone(),
            context_length: self.default_context_length,
            modified: None,
            has_chat_template: true,
            runtime: NAME.to_string(),
        }
    }
}

/// A request's `usage` object from an `flm serve` log line, if it carries one.
///
/// Up to 1.0.9 `flm` put it on the closing chunk of a stream; 1.0.10 closes the
/// stream without it and logs it on a line of its own straight after:
///
/// ```text
/// ChatCompletionChunk: {…,"usage":{"prompt_tokens":299,"completion_tokens":10,…}}
/// Usage: {"prompt_tokens":304,"completion_tokens":1109,"active_kv_tokens":1416,…}
/// ```
fn usage(line: &str) -> Option<serde_json::Value> {
    if let Some((_, json)) = line.split_once("ChatCompletionChunk:") {
        let mut chunk = serde_json::from_str::<serde_json::Value>(json.trim()).ok()?;
        return chunk.get_mut("usage").map(serde_json::Value::take);
    }
    let json = line.strip_prefix("Usage:")?;
    serde_json::from_str::<serde_json::Value>(json.trim()).ok().filter(serde_json::Value::is_object)
}

/// Per-request timings from an `flm serve` log line, if it carries any: the
/// token counts and the durations behind them, from the request's [`usage`]:
///
/// ```text
/// "prompt_tokens":299,"completion_tokens":10,"prefill_duration_ttft":7.53,
/// "decoding_duration":0.97,"prefill_speed_tps":39.69,"decoding_speed_tps":10.36
/// ```
///
/// It reports the rates too, but the counts and durations are what get used:
/// they divide out to the same figures and, unlike a rate, can be summed across
/// requests to average a window.
pub fn parse_throughput(line: &str) -> Vec<Sample> {
    let Some(usage) = usage(line) else { return Vec::new() };

    let read = |tokens: &str, duration: &str, phase: Phase| -> Option<Sample> {
        let sample = Sample {
            phase,
            tokens: usage.get(tokens)?.as_u64()?,
            seconds: usage.get(duration)?.as_f64()?,
            // `flm serve` runs one request at a time.
            slot: None,
        };
        sample.rate().is_some().then_some(sample)
    };
    [
        read("prompt_tokens", "prefill_duration_ttft", Phase::Prefill),
        read("completion_tokens", "decoding_duration", Phase::Decode),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// A request's stage, from the lines `flm serve` prints as it reaches each:
///
/// ```text
/// [FLM]  Start prefill...
/// [FLM]  Prefill chunk 2/4 with 512 tokens
/// [FLM]  Start generating...
/// ```
///
/// No token count or rate comes until the request's [`usage`], so neither is
/// claimed. Prefill progress is the chunks already done, shown only for a
/// prompt long enough to take several. Matched anywhere in the line: `flm`
/// writes from several threads, and lines arrive spliced together.
pub fn parse_live(line: &str) -> Option<crate::session::throughput::Live> {
    use crate::session::throughput::Live;
    let stage = |phase, progress| Live {
        phase,
        tokens: None,
        rate: None,
        progress,
        slot: None,
        until_done: true,
    };
    if line.contains("[FLM]  Start generating...") {
        return Some(stage(Phase::Decode, None));
    }
    if line.contains("[FLM]  Start prefill...") {
        return Some(stage(Phase::Prefill, None));
    }
    let (_, rest) = line.split_once("[FLM]  Prefill chunk ")?;
    let (done, rest) = rest.split_once('/')?;
    let total: f64 = rest.split_whitespace().next()?.parse().ok()?;
    let chunk: f64 = done.parse().ok()?;
    let progress = (total > 1.0).then(|| (chunk - 1.0) / total);
    Some(stage(Phase::Prefill, progress))
}

/// The end of a request, finished or cancelled: `flm serve` releases the NPU
/// after every one, which a cancelled request, with no usage to close it,
/// still does.
pub fn parse_end(line: &str) -> bool {
    line.contains("NPU Lock Released!")
}

/// Context size and use from a request's [`usage`], which FastFlowLM fills
/// with the KV cache's own figures: `active_kv_tokens` of
/// `max_kv_token_capacity`.
pub fn parse_context(line: &str) -> Vec<crate::session::context::Reading> {
    use crate::session::context::Reading;
    let Some(usage) = usage(line) else { return Vec::new() };
    let field = |key: &str| usage.get(key).and_then(serde_json::Value::as_u64);
    [
        field("max_kv_token_capacity").map(Reading::Size),
        field("active_kv_tokens").map(Reading::Used),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// A tag as a single filesystem-safe path segment (`qwen3:4b` → `qwen3_4b`).
fn sanitize(tag: &str) -> String {
    tag.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' })
        .collect()
}

/// How a download ended.
pub enum DownloadOutcome {
    Downloaded(PathBuf),
    Cancelled,
}

/// Download a model's files from Hugging Face into `flm`'s model directory.
///
/// This deliberately does not use `flm pull`: that has no resume, and `flm`
/// does not reliably recognize a partially-downloaded model, so an interrupted
/// pull leaves a directory that looks installed but is not. Here every file
/// lands under a `.llmctl-part` scratch name, and none is renamed until all of
/// them are byte-complete, `config.json` last (see [`commit_order`]). An
/// interrupted transfer, update or fresh, therefore leaves the directory as it
/// was, plus scratch files that mark it [`unfinished`] and resume at two levels.
/// A half-written file continues via HTTP `Range`, and a file already present
/// at its expected size, or fully fetched but not yet moved, is not fetched
/// again.
///
/// Placing the files here is enough: `flm list` then reports the model as
/// installed and `flm check` validates it.
pub fn download(
    model: &Model,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<DownloadOutcome, String> {
    let flm = model.flm().ok_or("not a FastFlowLM model")?;
    let dir = model_dir(model).ok_or("could not resolve the FastFlowLM model directory")?;

    let sizes = file_sizes(&flm.repo, &flm.revision, &flm.files)?;
    let total: u64 = sizes.iter().map(|(_, size)| *size).sum();
    if total == 0 {
        return Err(format!("hf://{} reports no downloadable files", flm.repo));
    }

    // Bytes already on disk from an earlier attempt, so a resumed download's
    // progress starts where it left off rather than at zero.
    let done = |sizes: &[(String, u64)]| -> u64 {
        sizes
            .iter()
            .map(|(file, size)| {
                let complete = dir.join(file);
                if complete.metadata().is_ok_and(|m| m.len() == *size) {
                    return *size;
                }
                partial_path(&dir, file).metadata().map(|m| m.len().min(*size)).unwrap_or(0)
            })
            .sum()
    };

    progress(done(&sizes), total);
    // Fetch everything first, all of it under scratch names, and only then
    // move it into place. Until the last byte is in, the directory holds
    // whatever it held before — nothing for a fresh model, the previous
    // revision, still loadable, for an update — and never a mixture of the two.
    let mut fetched: Vec<&str> = Vec::new();
    for (file, size) in &sizes {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(DownloadOutcome::Cancelled);
        }
        let complete = dir.join(file);
        let partial = partial_path(&dir, file);
        if complete.metadata().is_ok_and(|m| m.len() == *size) {
            // Already have it, byte for byte. A scratch copy beside it is a
            // leftover, and would otherwise mark the model unfinished forever.
            let _ = std::fs::remove_file(&partial);
            continue;
        }
        if partial.metadata().is_ok_and(|m| m.len() == *size) {
            fetched.push(file); // fetched by an earlier attempt, never moved
            continue;
        }
        let carried = done(&sizes)
            .saturating_sub(partial.metadata().map(|m| m.len().min(*size)).unwrap_or(0));
        let transferred = hf::download_file(
            &flm.repo,
            &flm.revision,
            file,
            &partial,
            *size,
            cancelled,
            |bytes, _| progress(carried.saturating_add(bytes).min(total), total),
        )
        .map_err(|err| format!("{err:#}"))?;
        if !transferred {
            return Ok(DownloadOutcome::Cancelled);
        }
        fetched.push(file);
        progress(done(&sizes), total);
    }

    for file in commit_order(fetched) {
        let complete = dir.join(file);
        std::fs::rename(partial_path(&dir, file), &complete)
            .map_err(|err| format!("completing {}: {err}", complete.display()))?;
    }
    Ok(DownloadOutcome::Downloaded(dir))
}

/// Whether llmctl left a transfer of this model unfinished: a scratch file
/// beside one of its files.
///
/// `flm` cannot tell. It counts a model as installed once every file is present
/// under its real name and `config.json`'s `flm_version` meets the catalog's
/// `flm_min_version`. A fresh download stages everything under scratch names,
/// so it never looks installed early. An update is another matter: the old
/// revision's files already sit under the real names, so the directory looks
/// installed the moment the new `config.json` lands, whatever state the weights
/// are in.
fn unfinished(model: &Model) -> bool {
    model_dir(model).is_some_and(|dir| unfinished_in(&dir, model))
}

fn unfinished_in(dir: &Path, model: &Model) -> bool {
    model.flm().is_some_and(|flm| flm.files.iter().any(|file| partial_path(dir, file).exists()))
}

/// The order a finished transfer moves its files into place: `config.json`
/// last. Its `flm_version` is what tells `flm` the new revision is installed,
/// so it must not say so while any other file is still the old revision's.
fn commit_order<'a>(files: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    let mut files: Vec<&str> = files.into_iter().collect();
    files.sort_by_key(|file| *file == "config.json");
    files
}

/// Scratch name for an in-flight file. Kept beside the real one so a resume
/// finds it, and distinct so `flm` never mistakes it for a finished file.
fn partial_path(dir: &Path, file: &str) -> PathBuf {
    dir.join(format!("{file}.llmctl-part"))
}

/// Sizes for the model's files, from the repository tree.
fn file_sizes(repo: &str, revision: &str, files: &[String]) -> Result<Vec<(String, u64)>, String> {
    let tree = hf::tree(repo, revision).map_err(|err| format!("{err:#}"))?;
    files
        .iter()
        .map(|file| {
            tree.iter()
                .find(|entry| &entry.path == file)
                .map(|entry| (file.clone(), entry.size))
                .ok_or_else(|| format!("hf://{repo}@{revision} has no file named {file}"))
        })
        .collect()
}

/// The expected files and their sizes, for tracking a download llmctl does not
/// perform itself (see [`FlmBackend::launch_download`]).
pub fn expected_files(model: &Model) -> Option<Vec<(PathBuf, u64)>> {
    let flm = model.flm()?;
    let dir = model_dir(model)?;
    let sizes = file_sizes(&flm.repo, &flm.revision, &flm.files).ok()?;
    Some(sizes.into_iter().map(|(file, size)| (dir.join(file), size)).collect())
}

fn list_models(binary: &Path) -> Result<Vec<Entry>, String> {
    let raw = run(binary, &["list", "--json"]).ok_or("`flm list --json` failed")?;
    let catalog: Catalog =
        serde_json::from_str(strip_preamble(&raw)).map_err(|e| format!("bad catalog JSON: {e}"))?;
    Ok(catalog.models)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    /// Verbatim shape of `flm list --json` on v0.9.45, banner included.
    const SAMPLE: &str = r#"[FLM]  Fetching models from: /opt/fastflowlm/share/flm/model_list.json
{
    "models": [
        {
            "default_context_length": 16384,
            "details": { "family": "qwen3", "parameter_size": "4B", "quantization_level": "Q4_1" },
            "footprint": 2.8,
            "installed": true,
            "label": ["reasoning", "tool-calling"],
            "max_prefill_len": 4096,
            "model": "qwen3:4b",
            "name": "qwen3:4b",
            "size": 4000000000,
            "url": "https://huggingface.co/FastFlowLM/Qwen3-4B-NPU2/resolve/main"
        },
        {
            "default_context_length": 8192,
            "details": { "family": "llama3", "quantization_level": "Q4_1" },
            "footprint": 1.1,
            "installed": false,
            "label": [],
            "model": "llama3.2:1b",
            "name": "llama3.2:1b",
            "url": "https://huggingface.co/FastFlowLM/Llama-3.2-1B-NPU2/resolve/main"
        }
    ]
}"#;

    fn local(label: &str) -> Vec<String> {
        vec![LOCAL_GROUP.into(), label.into()]
    }

    fn online(label: &str) -> Vec<String> {
        vec![ONLINE_GROUP.into(), label.into()]
    }

    fn entries() -> Vec<Entry> {
        serde_json::from_str::<Catalog>(strip_preamble(SAMPLE)).unwrap().models
    }

    #[test]
    fn strips_the_banner_before_parsing() {
        assert!(strip_preamble(SAMPLE).starts_with('{'));
        // Already-clean JSON passes through untouched.
        assert_eq!(strip_preamble("{\"models\":[]}"), "{\"models\":[]}");
    }

    #[test]
    fn parses_the_catalog() {
        let entries = entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "qwen3:4b");
        assert!(entries[0].installed);
        assert_eq!(entries[0].default_context_length, Some(16384));
        assert_eq!(entries[0].details.quantization_level.as_deref(), Some("Q4_1"));
    }

    /// Switching arrangement must regroup the catalog it already has: `flm list`
    /// costs ~150 ms through a launcher script, and re-running it to reshape
    /// identical data is what made `s` hitch. A reload must still reach `flm`.
    #[test]
    #[ignore = "needs a real flm install; run with --ignored --test-threads=1"]
    fn rearranging_the_catalog_does_not_re_read_it() {
        use std::time::Instant;

        let backend = FlmBackend::discover(&FastFlowLmConfig::default());
        let dir = std::env::temp_dir().join(format!("llmctl-flm-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ctx = |view, reload| CatalogCtx {
            sources: &[],
            cache_path: Path::new("/nonexistent"),
            models_dir: &dir,
            view,
            reload,
        };

        // First read populates the cache and pays for the subprocess.
        let start = Instant::now();
        let categories = backend.models(&ctx(0, false));
        let cold = start.elapsed();
        assert!(!categories.is_empty(), "the catalog should not be empty");

        // Rearranging must not go near `flm` again.
        let start = Instant::now();
        let flat = backend.models(&ctx(FLAT_VIEW, false));
        let warm = start.elapsed();
        assert!(!flat.is_empty());
        assert!(
            warm * 10 < cold,
            "rearranging cost {warm:?} against a cold read of {cold:?} — the catalog was re-read"
        );

        // ...but the two arrangements still describe the same set of models.
        let tags = |models: &[Model]| -> std::collections::BTreeSet<String> {
            models.iter().filter_map(|m| m.flm()).map(|f| f.tag.clone()).collect()
        };
        assert_eq!(tags(&categories), tags(&flat), "an arrangement lost models");

        // A reload goes back to the binary, so it costs like the first read.
        let start = Instant::now();
        let reloaded = backend.models(&ctx(0, true));
        let refreshed = start.elapsed();
        assert_eq!(tags(&reloaded), tags(&categories));
        assert!(refreshed > warm * 10, "a reload should have re-read the catalog");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The profile directories are what make FastFlowLM profiles persist as
    /// per-model YAML, so moving that write out of `models` must not lose it.
    #[test]
    #[ignore = "needs a real flm install; run with --ignored --test-threads=1"]
    fn a_cached_read_still_materializes_the_profile_directories() {
        let backend = FlmBackend::discover(&FastFlowLmConfig::default());
        let dir = std::env::temp_dir().join(format!("llmctl-flm-leaf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ctx = CatalogCtx {
            sources: &[],
            cache_path: Path::new("/nonexistent"),
            models_dir: &dir,
            view: 0,
            reload: false,
        };

        let models = backend.models(&ctx);
        let model = models.iter().find(|m| m.is_model()).expect("a model leaf");
        assert!(
            model.catalog_dir.join("profiles").is_dir(),
            "the managed profile directory was not created"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// FastFlowLM is exclusive (one NPU hardware context) where llama.cpp is
    /// not — the distinction the launch guard rests on.
    /// Regression: `Entry::url` is optional, so a catalog row without one has
    /// an empty repository. `join("")` on the model root is the root, and the
    /// deletion plan would have removed it recursively — every installed model.
    #[test]
    fn a_catalog_row_without_a_url_resolves_to_no_directory_at_all() {
        assert_eq!(repo_dir_name("FastFlowLM/Qwen3-4B-NPU2"), Some("Qwen3-4B-NPU2"));
        assert_eq!(repo_dir_name(""), None);
        assert_eq!(repo_dir_name("owner/"), None);
        assert_eq!(repo_dir_name(".."), None);

        let mut entry = entries()[0].clone();
        entry.url = String::new();
        entry.installed = true;
        let model = entry.to_model(&local("reasoning"), Path::new("/models"), Path::new("/cfg"));
        assert_eq!(model.path, PathBuf::new(), "no directory means no local path");
        assert_eq!(model_dir(&model), None, "and nothing for a deletion to remove");
    }

    /// Regression: an unusable model root used to pass the "never the root
    /// itself" check, because the parent of a bare `Qwen3-4B-NPU2` is the empty
    /// path and so is the root. The deletion plan then held a *relative*
    /// directory, which `remove_dir_all` resolves against llmctl's own working
    /// directory — someone else's tree entirely.
    #[test]
    fn a_model_root_that_is_not_absolute_resolves_to_no_directory_at_all() {
        let mut entry = entries()[0].clone();
        entry.installed = true;
        let model = entry.to_model(&local("reasoning"), Path::new("/models"), Path::new("/cfg"));

        // `FLM_MODEL_PATH=` set to nothing, or no home directory to resolve.
        assert_eq!(model_dir_in(Path::new(""), &model), None, "an empty root names nothing");
        // A relative `FLM_MODEL_PATH`, which `flm` resolves against its own
        // working directory rather than llmctl's.
        assert_eq!(model_dir_in(Path::new("models"), &model), None, "nor does a relative one");

        let dir = model_dir_in(Path::new("/srv/flm"), &model).expect("an absolute root resolves");
        assert_eq!(dir.parent(), Some(Path::new("/srv/flm")));
    }

    /// Regression: an update interrupted after `config.json` was replaced left
    /// new metadata over the old weights. `flm` reported the model installed,
    /// so llmctl offered no download, and `flm serve` died loading it.
    #[test]
    fn a_scratch_file_marks_a_transfer_unfinished_whatever_flm_says() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0);
        let root = std::env::temp_dir().join(format!("llmctl-flm-unfinished-{nonce}"));
        let mut entry = entries()[0].clone();
        entry.installed = true;
        entry.files = vec!["config.json".into(), "model.q4nx".into()];
        let model = entry.to_model(&local("reasoning"), &root, Path::new("/cfg"));
        let dir = model_dir_in(&root, &model).expect("an absolute root resolves");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{\"flm_version\": \"1.0.3\"}").unwrap();
        std::fs::write(dir.join("model.q4nx"), b"old revision").unwrap();
        assert!(!unfinished_in(&dir, &model), "a whole model is not unfinished");
        std::fs::write(partial_path(&dir, "model.q4nx"), b"new revi").unwrap();
        assert!(unfinished_in(&dir, &model), "a scratch file means a transfer stopped part way");

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn config_json_is_moved_into_place_last() {
        assert_eq!(
            commit_order(["config.json", "model.q4nx", "tokenizer.json"]),
            ["model.q4nx", "tokenizer.json", "config.json"]
        );
        assert_eq!(commit_order(["model.q4nx"]), ["model.q4nx"]);
    }

    /// A closing chunk's usage, as a real `flm serve` log reports the KV cache.
    #[test]
    fn context_size_and_use_are_read_from_the_closing_chunk() {
        use crate::session::context::Reading;
        let line = r#"ChatCompletionChunk: {"choices":[],"usage":{"prompt_tokens":299,"completion_tokens":35,"total_tokens":334,"active_kv_tokens":334,"max_kv_token_capacity":32768,"kv_token_occupancy_rate_percentage":1.019287109375}}"#;
        assert_eq!(parse_context(line), vec![Reading::Size(32768), Reading::Used(334)]);
        assert!(parse_context(r#"ChatCompletionChunk: {"choices":[]}"#).is_empty());
        assert!(parse_context("[FLM]  Creating checkpoint at context length 299").is_empty());
    }

    /// Verbatim stage lines from real flm 1.0.10 logs, in the order a request
    /// prints them, including one spliced into another thread's output.
    #[test]
    fn a_request_in_progress_is_read_from_its_stage_lines() {
        let live = |line| parse_live(line).map(|l| (l.phase, l.progress, l.tokens, l.until_done));
        assert_eq!(live("[FLM]  Start prefill..."), Some((Phase::Prefill, None, None, true)));
        assert_eq!(
            live("[FLM]  Prefill chunk 1/1 with 13 tokens"),
            Some((Phase::Prefill, None, None, true)),
            "one chunk: no progress worth showing"
        );
        assert_eq!(
            live("[FLM]  Prefill chunk 3/4 with 512 tokens"),
            Some((Phase::Prefill, Some(0.5), None, true)),
            "two of four done"
        );
        assert_eq!(live("[FLM]  Start generating..."), Some((Phase::Decode, None, None, true)));
        assert_eq!(
            live("[LOG]  Version: 10[FLM]  Start generating..."),
            Some((Phase::Decode, None, None, true))
        );
        assert_eq!(live("[FLM]  Model loaded in 10.07 s"), None);
        assert!(parse_end("[🔵 ]  NPU Lock Released!"));
        assert!(!parse_end("[🔴 ]  Client disconnected; cancelling active request"));
        assert!(!parse_end("[🟢 ]  NPU Locked!"));
    }

    /// A whole request from a real flm 1.0.10 log, fed line by line through
    /// the same parsers the session manager runs: Reading, then Generating
    /// through the quiet of decoding, then idle with the request's rates.
    #[test]
    fn a_logged_request_reads_as_reading_then_generating_then_idle() {
        use crate::session::throughput::{Activity, Throughput};
        let parse = crate::runtime::log_parsers(NAME);
        let request = [
            "[🟢 ]  NPU Locked!",
            "[FLM]  Prompt cache miss.",
            "[FLM]  Start prefill...",
            "[FLM]  Prefill chunk 1/1 with 13 tokens",
            "\u{1b}[31m[FLM]  Creating checkpoint at context length 13\u{1b}[0m",
            "[FLM]  Start generating...",
            "[FLM]  Model RAW Output: ",
            "Okay, so I need to explain what Git rebase does in two sentences.",
            r#"Usage: {"prompt_tokens":13,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens":510,"total_tokens":523,"active_kv_tokens":523,"max_kv_token_capacity":32768,"kv_token_occupancy_rate_percentage":1.59,"load_duration":2.1e-06,"prefill_duration_ttft":0.42,"decoding_duration":34.0,"prefill_speed_tps":30.95,"decoding_speed_tps":15.0}"#,
            "[🔒 ]  Closing TCP connection (streaming, non-keep-alive)",
            "[🔵 ]  NPU Lock Released!",
        ];
        let now = std::time::Instant::now();
        let mut throughput = Throughput::default();
        let mut seen = Vec::new();
        for line in request {
            if let Some(live) = (parse.live)(line) {
                throughput.record_live(live, now);
            }
            if (parse.ends)(line) {
                throughput.record_end(None);
            }
            for sample in (parse.throughput)(line) {
                throughput.record(sample);
            }
            let state = match throughput.activity(now) {
                Some(Activity::Reading(_)) => "reading",
                Some(Activity::Generating(_)) => "generating",
                Some(Activity::Idle) => "idle",
                None => "silent",
            };
            if seen.last() != Some(&state) {
                seen.push(state);
            }
        }
        assert_eq!(seen, ["silent", "reading", "generating", "idle"]);
        assert!((throughput.rate(Phase::Decode).unwrap() - 15.0).abs() < 0.01);

        // Cancelled mid-generation: no usage, but the NPU is still released.
        throughput.record_live((parse.live)("[FLM]  Start generating...").unwrap(), now);
        assert!(matches!(throughput.activity(now), Some(Activity::Generating(_))));
        assert!((parse.ends)("[🔵 ]  NPU Lock Released!"));
        throughput.record_end(None);
        assert_eq!(throughput.activity(now), Some(Activity::Idle));
    }

    /// Verbatim from a real flm 1.0.10 log: the closing chunk has no usage any
    /// more, and the line after it does.
    #[test]
    fn flm_1_0_10_logs_usage_on_its_own_line_after_the_closing_chunk() {
        use crate::session::context::Reading;
        let closing = r#"ChatCompletionChunk: {"id":"chatcmpl-208b4c02e56a655e09cf3b0f","object":"chat.completion.chunk","created":1792967802,"model":"qwen3.6-moe:35b-a3b","system_fingerprint":"fp_19267b9e7025bb41","choices":[{"index":0,"delta":{"content":null},"finish_reason":"stop"}]}"#;
        let usage = r#"Usage: {"prompt_tokens":304,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens":1109,"total_tokens":1413,"active_kv_tokens":1416,"max_kv_token_capacity":32768,"kv_token_occupancy_rate_percentage":4.3212890625,"load_duration":2.695e-06,"prefill_duration_ttft":6.2892416,"decoding_duration":73.998832,"prefill_speed_tps":48.33651167097795,"decoding_speed_tps":14.986723033682479}"#;
        assert!(parse_throughput(closing).is_empty());
        assert!(parse_context(closing).is_empty());

        let samples = parse_throughput(usage);
        assert_eq!(samples.len(), 2);
        assert_eq!((samples[0].phase, samples[0].tokens), (Phase::Prefill, 304));
        assert_eq!((samples[1].phase, samples[1].tokens), (Phase::Decode, 1109));
        assert!((samples[1].rate().unwrap() - 14.9867).abs() < 0.001);
        assert_eq!(parse_context(usage), vec![Reading::Size(32768), Reading::Used(1416)]);
        assert!(parse_throughput("Usage: not json").is_empty());
    }

    /// A verbatim closing chunk from a real `flm serve` log.
    #[test]
    fn per_request_timings_are_read_from_the_closing_chunk() {
        let line = r#"ChatCompletionChunk: {"id":"chatcmpl-2c0e58d9ea2b7e21bd157485","object":"chat.completion.chunk","created":1785106964,"model":"qwen3.6-moe:35b-a3b","choices":[{"index":0,"delta":{"content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":299,"completion_tokens":10,"total_tokens":309,"load_duration":11.668804608,"prefill_duration_ttft":7.53399552,"decoding_duration":0.965662,"prefill_speed_tps":39.68677698390774,"decoding_speed_tps":10.355590258289132}}"#;

        let samples = parse_throughput(line);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].phase, Phase::Prefill);
        assert_eq!(samples[0].tokens, 299);
        assert_eq!(samples[1].phase, Phase::Decode);
        assert_eq!(samples[1].tokens, 10);
        // The counts over the durations reproduce the rates `flm` itself
        // reports, which is why they are used instead of those rates.
        assert!((samples[0].rate().unwrap() - 39.686_776_983_907_74).abs() < 1e-9);
        assert!((samples[1].rate().unwrap() - 10.355_590_258_289_132).abs() < 1e-9);
    }

    /// Streaming emits a chunk per token; only the closing one carries usage.
    #[test]
    fn chunks_without_usage_are_not_measurements() {
        for line in [
            r#"ChatCompletionChunk: {"id":"x","choices":[{"delta":{"content":"hi"}}]}"#,
            "ChatCompletionChunk: not json at all",
            "[FLM]  Start prefill...",
            "[FLM]  Prefill chunk 1/1 with 299 tokens",
            "",
        ] {
            assert!(parse_throughput(line).is_empty(), "should not measure: {line}");
        }
    }

    #[test]
    fn only_the_npu_runtime_is_single_session() {
        let flm = FlmBackend::discover(&FastFlowLmConfig::default());
        assert!(flm.single_session());
        let llama = crate::runtime::LlamaCppBackend::discover(
            &crate::config::LlamaCppConfig::default(),
            Path::new("/nonexistent"),
        );
        assert!(!llama.single_session());
    }

    #[test]
    fn local_and_online_split_by_installed_then_group_by_label() {
        let entries = entries();
        // Installed, so `local`, once per capability label.
        assert_eq!(entries[0].groups(false), [["local", "reasoning"], ["local", "tool-calling"]]);
        // Not installed and unlabeled: `online`, under the chat fallback.
        assert_eq!(entries[1].groups(false), [["online", "chat"]]);
    }

    #[test]
    fn the_flat_view_places_each_model_exactly_once() {
        let entries = entries();
        // Categories fans a multi-label model out across its labels...
        assert_eq!(entries[0].groups(false).len(), 2);
        // ...while Flat lists it once, directly under local/online.
        assert_eq!(entries[0].groups(true), [["local"]]);
        assert_eq!(entries[1].groups(true), [["online"]]);

        let flat = entries[0].to_model(
            &entries[0].groups(true)[0],
            Path::new("/models"),
            Path::new("/cfg"),
        );
        assert_eq!(flat.catalog_path, ["local", "qwen3:4b"]);
        // The arrangement is a view: identity and profiles are unaffected.
        assert_eq!(flat.profile_key(), "flm:qwen3:4b");
        assert_eq!(flat.catalog_dir, Path::new("/cfg/fastflowlm/qwen3_4b"));
    }

    #[test]
    fn pinned_revisions_are_honored_and_bare_urls_fall_back_to_main() {
        let pinned = |url: &str| Entry { url: url.into(), ..entries()[0].clone() }.revision();

        // Several FastFlowLM models live on a tag, not on main.
        assert_eq!(
            pinned("https://huggingface.co/FastFlowLM/Qwen3-0.6B-NPU2/resolve/v0.9.22-faster-q4-1"),
            "v0.9.22-faster-q4-1"
        );
        assert_eq!(pinned("https://huggingface.co/FastFlowLM/X-NPU2/resolve/main"), "main");
        // embed-gemma's entry carries no /resolve/ segment at all.
        assert_eq!(pinned("https://huggingface.co/FastFlowLM/Embedding-Gemma-300M-NPU2"), "main");
        assert_eq!(pinned(""), "main");
    }

    #[test]
    fn a_multi_group_model_keeps_one_profile_identity() {
        let entry = &entries()[0];
        let models: Vec<Model> = entry
            .groups(false)
            .iter()
            .map(|g| entry.to_model(g, Path::new("/models"), Path::new("/cfg")))
            .collect();
        assert_eq!(models.len(), 2);
        // Two tree positions...
        assert_eq!(models[0].catalog_path, ["local", "reasoning", "qwen3:4b"]);
        assert_eq!(models[1].catalog_path, ["local", "tool-calling", "qwen3:4b"]);
        // ...one model, so profiles and their on-disk home are shared.
        for model in &models {
            assert_eq!(model.profile_key(), "flm:qwen3:4b");
            assert_eq!(model.catalog_dir, Path::new("/cfg/fastflowlm/qwen3_4b"));
        }
    }

    #[test]
    fn disk_size_comes_from_footprint_not_the_parameter_count() {
        let entry = &entries()[0];
        let model = entry.to_model(&local("reasoning"), Path::new("/models"), Path::new("/cfg"));
        // footprint 2.8 GB, not the 4e9 `size` (which counts parameters).
        assert_eq!(model.size_bytes, 2_800_000_000);
    }

    #[test]
    fn an_installed_model_resolves_its_directory_and_an_absent_one_stays_launchable() {
        let entries = entries();
        let installed =
            entries[0].to_model(&local("reasoning"), Path::new("/models"), Path::new("/cfg"));
        assert_eq!(installed.path, Path::new("/models/Qwen3-4B-NPU2"));
        assert!(installed.is_model());

        let absent = entries[1].to_model(&online("chat"), Path::new("/models"), Path::new("/cfg"));
        assert_eq!(absent.path, Path::new(""));
        // No local path, but it is a model — not a folder — so it can be
        // selected, profiled, and launched (flm downloads it on demand).
        assert!(absent.is_model());
        assert!(!absent.is_catalog_dir());
    }

    #[test]
    fn every_flag_is_emitted_with_an_explicit_value() {
        // flm has no valueless flags, and host/port are never omitted.
        assert!(!SCHEMA.is_flag("cors"));
        assert_eq!(SCHEMA.omit_token("host"), None);
        assert_eq!(SCHEMA.omit_token("port"), None);
        assert_eq!(SCHEMA.omit_token("pmode"), Some(crate::profiles::registry::DEFAULT));
        assert!(SCHEMA.uses_sentinel("pmode"));
    }

    #[test]
    fn an_npu_is_present_when_the_xdna_driver_has_a_device_bound() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0);
        let driver = std::env::temp_dir().join(format!("llmctl-amdxdna-{nonce}"));
        // The driver loaded with nothing bound: only its own files.
        for file in ["bind", "unbind", "uevent", "new_id", "remove_id"] {
            std::fs::create_dir_all(&driver).unwrap();
            std::fs::write(driver.join(file), "").unwrap();
        }
        std::fs::create_dir_all(driver.join("module")).unwrap();
        assert!(!npu_present(&driver));
        std::fs::create_dir_all(driver.join("0000:c8:00.1")).unwrap();
        assert!(npu_present(&driver));
        assert!(!npu_present(Path::new("/nonexistent/llmctl/amdxdna")), "no driver, no NPU");
        let _ = std::fs::remove_dir_all(&driver);
    }

    #[test]
    fn fastflowlm_is_listed_with_an_npu_or_an_flm_but_not_with_neither() {
        let backend = |binary: Option<&str>, npu_present| FlmBackend {
            runtime: Runtime {
                name: NAME.into(),
                description: String::new(),
                version: None,
                binary_path: binary.map(PathBuf::from),
                bench_path: None,
                formats: vec![],
                devices: vec![],
            },
            npu_ready: false,
            npu_problem: None,
            npu_present,
            catalog_cache: Mutex::new(None),
        };
        assert!(!backend(None, false).listed(), "no NPU and no flm: nothing to offer");
        let no_flm = backend(None, true);
        assert!(no_flm.listed(), "an NPU without flm is shown, dimmed");
        assert_eq!(no_flm.unavailable_reason().as_deref(), Some("flm binary not found on PATH"));
        assert!(backend(Some("/usr/bin/flm"), false).listed(), "flm with no NPU says why");
    }

    #[test]
    fn all_resolved_defaults_pass_launch_validation() {
        let backend = FlmBackend {
            runtime: Runtime {
                name: NAME.into(),
                description: String::new(),
                version: None,
                binary_path: None,
                bench_path: None,
                formats: vec![],
                devices: vec![],
            },
            npu_ready: true,
            npu_problem: None,
            npu_present: true,
            catalog_cache: Mutex::new(None),
        };
        let model =
            entries()[0].to_model(&local("reasoning"), Path::new("/models"), Path::new("/cfg"));
        let defaults = Defaults::default();
        let options = SPECS
            .iter()
            .map(|spec| {
                let value = backend.spec_default(spec, &model, &defaults);
                OptionItem { spec, default: value.clone(), value, range: None }
            })
            .collect::<Vec<_>>();

        assert_eq!(
            options.iter().find(|option| option.spec.key == "pmode").unwrap().value,
            crate::profiles::registry::DEFAULT
        );
        crate::profiles::validate_options(&backend, &model, &options).unwrap();
    }

    fn opt(key: &str, value: &str) -> OptionItem {
        OptionItem {
            spec: SCHEMA.spec(key).unwrap(),
            value: value.into(),
            default: String::new(),
            range: None,
        }
    }

    #[test]
    fn serve_command_always_carries_the_port() {
        let backend = FlmBackend {
            runtime: Runtime {
                name: NAME.into(),
                description: String::new(),
                version: None,
                binary_path: None,
                bench_path: None,
                formats: vec![],
                devices: vec![],
            },
            npu_ready: true,
            npu_problem: None,
            npu_present: true,
            catalog_cache: Mutex::new(None),
        };
        let model =
            entries()[0].to_model(&local("reasoning"), Path::new("/models"), Path::new("/cfg"));
        let options = vec![
            opt("ctx-len", "8192"),
            opt("pmode", crate::profiles::registry::DEFAULT),
            opt("host", "127.0.0.1"),
            opt("port", "52625"),
        ];
        let ctx =
            LaunchContext::new("flm", &model, &options).expect("selected model is launchable");
        let argv = backend.build_command(&ctx).argv;

        assert_eq!(argv[..4], ["flm", "serve", "qwen3:4b", "--quiet"]);
        assert!(argv.windows(2).any(|w| w == ["--ctx-len", "8192"]));
        // An omitted option leaves the flag off entirely...
        assert!(!argv.iter().any(|a| a == "--pmode"));
        // ...but the endpoint is always explicit: health checks and process
        // re-acquisition both depend on it.
        assert!(argv.windows(2).any(|w| w == ["--port", "52625"]));
        assert!(argv.windows(2).any(|w| w == ["--host", "127.0.0.1"]));
    }

    #[test]
    fn chat_drops_server_only_options() {
        let backend = FlmBackend {
            runtime: Runtime {
                name: NAME.into(),
                description: String::new(),
                version: None,
                binary_path: None,
                bench_path: None,
                formats: vec![],
                devices: vec![],
            },
            npu_ready: true,
            npu_problem: None,
            npu_present: true,
            catalog_cache: Mutex::new(None),
        };
        let model =
            entries()[0].to_model(&local("reasoning"), Path::new("/models"), Path::new("/cfg"));
        let options = vec![opt("ctx-len", "8192"), opt("host", "127.0.0.1"), opt("port", "52625")];
        let ctx =
            LaunchContext::new("flm", &model, &options).expect("selected model is launchable");
        let argv = backend.chat_argv(&ctx).unwrap();

        assert_eq!(argv[..3], ["flm", "run", "qwen3:4b"]);
        assert!(argv.windows(2).any(|w| w == ["--ctx-len", "8192"]));
        assert!(!argv.iter().any(|a| a == "--port" || a == "--host"));
    }

    #[test]
    fn bench_keeps_only_the_power_mode() {
        let backend = FlmBackend {
            runtime: Runtime {
                name: NAME.into(),
                description: String::new(),
                version: None,
                binary_path: None,
                bench_path: None,
                formats: vec![],
                devices: vec![],
            },
            npu_ready: true,
            npu_problem: None,
            npu_present: true,
            catalog_cache: Mutex::new(None),
        };
        let model =
            entries()[0].to_model(&local("reasoning"), Path::new("/models"), Path::new("/cfg"));
        let options = vec![opt("pmode", "turbo"), opt("ctx-len", "8192"), opt("port", "52625")];
        let ctx =
            LaunchContext::new("flm", &model, &options).expect("selected model is launchable");
        let argv = backend.bench_argv(&ctx).unwrap();

        assert_eq!(argv, ["flm", "bench", "qwen3:4b", "--pmode", "turbo"]);
    }

    /// Exercises the real `flm` on this machine: discovery, NPU validation, and
    /// the live catalog. Ignored by default (needs the hardware and spawns
    /// processes); run with `--ignored`.
    #[test]
    #[ignore = "needs a real flm install and an AMD NPU; run with --ignored --test-threads=1"]
    fn discovers_the_installed_runtime_and_its_catalog() {
        let backend = FlmBackend::discover(&FastFlowLmConfig::default());
        let runtime = backend.descriptor();
        assert!(runtime.binary_path.is_some(), "flm not found on PATH");
        assert!(runtime.version.as_deref().is_some_and(|v| v.starts_with("flm ")));
        assert!(!runtime.devices.is_empty(), "flm validate reported no NPU devices");
        assert_eq!(backend.unavailable_reason(), None, "NPU stack is not ready");

        let catalog = backend.catalog();
        assert!(catalog.len() > 10, "expected a populated catalog, got {}", catalog.len());
        assert!(catalog.iter().any(|e| e.installed), "no installed models found");
        // Every entry must yield a tag and at least one browser group.
        for entry in &catalog {
            assert!(entry.name.contains(':'), "{} is not a name:size tag", entry.name);
            assert!(!entry.groups(false).is_empty(), "{} landed in no group", entry.name);
        }
    }

    /// Full launch lifecycle against the real NPU: serve an installed model,
    /// wait for `/v1/models` to answer, confirm llmctl tracked the *server*
    /// process rather than any launcher wrapper in front of it, then stop it and
    /// confirm the process is really gone.
    /// Run single-threaded because the NPU admits only one client.
    #[test]
    #[ignore = "launches a real NPU server; run with --ignored --test-threads=1"]
    fn launch_lifecycle_on_the_npu() {
        use std::thread::sleep;

        use crate::session::{LaunchRequest, SessionManager, SessionStatus};

        let backend = FlmBackend::discover(&FastFlowLmConfig::default());
        assert_eq!(backend.unavailable_reason(), None, "FastFlowLM is not usable here");
        let binary = backend.descriptor().binary_path.clone().expect("flm binary");

        // Whichever installed model is smallest, to keep load time down.
        let entry = backend
            .catalog()
            .into_iter()
            .filter(|e| e.installed && !e.asr && e.label.iter().all(|l| l != "embeddings"))
            .min_by(|a, b| a.footprint.total_cmp(&b.footprint))
            .expect("this test needs at least one installed chat model");
        let model = entry.to_model(&local("chat"), &model_root(), Path::new("/tmp"));

        let port = 52999;
        let options =
            vec![opt("host", "127.0.0.1"), opt("port", &port.to_string()), opt("ctx-len", "2048")];
        let binary = binary.display().to_string();
        let ctx =
            LaunchContext::new(&binary, &model, &options).expect("selected model is launchable");

        let base = std::env::temp_dir().join(format!("llmctl-flm-{}", std::process::id()));
        let sessions_dir = base.join("sessions");
        let log_dir = base.join("logs");
        std::fs::create_dir_all(&sessions_dir).unwrap();
        std::fs::create_dir_all(&log_dir).unwrap();

        let mut mgr = SessionManager::new(sessions_dir, log_dir);
        let idx = mgr
            .launch(LaunchRequest {
                runtime: NAME.into(),
                model: model.name.clone(),
                model_path: backend.process_token(&ctx),
                command: backend.build_command(&ctx),
                health_path: backend.health_path().into(),
                download: None,
                profile: "Default".into(),
                size_bytes: None,
                device: None,
                host: "127.0.0.1".into(),
                port,
            })
            .expect("launch");

        let mut running = false;
        for _ in 0..120 {
            mgr.refresh();
            if mgr.sessions[idx].status == SessionStatus::Running {
                running = true;
                break;
            }
            sleep(Duration::from_millis(500));
        }
        assert!(running, "server never became Ready on /v1/models");

        // The spawned process may be a wrapper (a container entry point, say);
        // llmctl must have re-acquired the real `flm` behind it.
        let pid = mgr.sessions[idx].record.pid;
        assert_eq!(crate::session::proc::comm(pid).as_deref(), Some("flm"));
        let argv = crate::session::proc::cmdline(pid);
        assert!(argv.iter().any(|a| a == &model.flm().unwrap().tag));

        mgr.stop(idx).expect("stop");
        let mut stopped = false;
        for _ in 0..40 {
            mgr.refresh();
            if mgr.sessions[idx].status == SessionStatus::Stopped {
                stopped = true;
                break;
            }
            sleep(Duration::from_millis(500));
        }
        assert!(stopped, "session did not reach Stopped");
        assert!(!crate::session::proc::is_alive(pid), "the server process outlived the stop");

        let _ = std::fs::remove_dir_all(&base);
    }

    /// Enumerating models must materialize one managed leaf per *tag* — the
    /// profile store only adopts a model whose `profiles/` directory exists, so
    /// without this FastFlowLM profiles would silently never persist as YAML.
    #[test]
    #[ignore = "needs a real flm install; run with --ignored --test-threads=1"]
    fn enumeration_materializes_one_profile_directory_per_tag() {
        let backend = FlmBackend::discover(&FastFlowLmConfig::default());
        let models_dir =
            std::env::temp_dir().join(format!("llmctl-flm-leaves-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&models_dir);
        std::fs::create_dir_all(&models_dir).unwrap();

        let ctx = CatalogCtx {
            sources: &[],
            cache_path: Path::new("/nonexistent"),
            models_dir: &models_dir,
            view: 0,
            reload: false,
        };
        let models = backend.models(&ctx);
        assert!(!models.is_empty());

        let tags: std::collections::BTreeSet<_> =
            models.iter().map(|m| m.flm().unwrap().tag.clone()).collect();
        let leaves = std::fs::read_dir(models_dir.join(CATALOG_ROOT)).unwrap().count();
        // One directory per tag, not per (tag, group) pair — several models are
        // rendered under multiple capability labels.
        assert!(models.len() > tags.len(), "expected multi-group models in the catalog");
        assert_eq!(leaves, tags.len());

        for model in &models {
            assert!(model.catalog_dir.join("profiles").is_dir(), "{:?}", model.catalog_dir);
        }
        let _ = std::fs::remove_dir_all(&models_dir);
    }

    /// The point of replacing `flm pull`: an interrupted download must continue
    /// rather than start over. Downloads a real model, cancels partway, then
    /// resumes and checks that `flm` accepts the result.
    ///
    /// The fixture is whichever catalog entry is not yet installed and smallest
    /// on disk, rather than a fixed tag: running the test *installs* its model,
    /// so pinning one makes the test pass once and then fail on the machine that
    /// ran it.
    #[test]
    #[ignore = "downloads ~1 GB from Hugging Face; run with --ignored --test-threads=1"]
    fn an_interrupted_download_resumes_instead_of_restarting() {
        use std::sync::atomic::AtomicBool;

        let backend = FlmBackend::discover(&FastFlowLmConfig::default());
        let entry = backend
            .catalog()
            .into_iter()
            .filter(|e| !e.installed && e.footprint > 0.0 && !e.files.is_empty())
            .min_by(|a, b| a.footprint.total_cmp(&b.footprint))
            .expect("the catalog has no model left to download; uninstall one to run this test");
        let tag = entry.name.clone();
        let weights = entry
            .files
            .iter()
            .find(|f| f.ends_with(".q4nx"))
            .cloned()
            .expect("a weights file to interrupt");
        eprintln!("resume fixture: {tag} ({:.2} GB)", entry.footprint);

        let model = entry.to_model(&online("chat"), &model_root(), Path::new("/tmp"));
        let dir = model_dir(&model).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        // Cancel once a little of the weights file has landed.
        let cancelled = Arc::new(AtomicBool::new(false));
        let flag = cancelled.clone();
        let outcome = download(&model, &cancelled, move |done, _| {
            if done > 40 * 1024 * 1024 {
                flag.store(true, Ordering::Relaxed);
            }
        })
        .expect("first pass");
        assert!(matches!(outcome, DownloadOutcome::Cancelled), "expected to cancel mid-transfer");

        // A partial file must survive, under a name `flm` will not mistake for
        // a finished download.
        let partial = partial_path(&dir, &weights);
        let carried = partial.metadata().expect("partial file kept").len();
        assert!(carried > 0, "nothing was kept to resume from");
        assert!(!dir.join(&weights).exists(), "an incomplete file was published");

        // Resume: progress must start from what is already on disk, not zero.
        let cancelled = Arc::new(AtomicBool::new(false));
        let first_report = Arc::new(std::sync::Mutex::new(None));
        let seen = first_report.clone();
        let outcome = download(&model, &cancelled, move |done, _| {
            let mut seen = seen.lock().unwrap();
            if seen.is_none() {
                *seen = Some(done);
            }
        })
        .expect("resume");
        assert!(matches!(outcome, DownloadOutcome::Downloaded(_)));
        assert!(
            first_report.lock().unwrap().unwrap_or(0) >= carried,
            "resume restarted from zero instead of continuing"
        );

        // The scratch file is gone and flm accepts the model.
        assert!(!partial.exists(), "the partial file outlived the download");
        let binary = backend.descriptor().binary_path.clone().unwrap();
        let check = ProcCommand::new(&binary).args(["check", &tag]).output().unwrap();
        assert!(check.status.success(), "flm check rejected the downloaded model");
        assert!(
            backend.catalog().iter().any(|e| e.name == tag && e.installed),
            "flm does not report the model as installed"
        );

        // Put the machine back as it was found. Without this the test installs a
        // model every run, and each run has to reach further down the catalog for
        // a bigger one — this is the only directory the test created, and it was
        // verified not-installed before it started.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repo_ids_keep_their_owner_but_directories_do_not() {
        let entry = &entries()[0];
        // The Hub needs the full id...
        assert_eq!(entry.repo(), "FastFlowLM/Qwen3-4B-NPU2");
        // ...while flm names the directory after the repository alone.
        assert_eq!(repo_dir_name(&entry.repo()), Some("Qwen3-4B-NPU2"));
        let model = entry.to_model(&local("reasoning"), Path::new("/models"), Path::new("/cfg"));
        assert_eq!(model.path, Path::new("/models/Qwen3-4B-NPU2"));
    }

    /// The two arrangements must cover the same models: Flat lists each once,
    /// Categories fans multi-label models across their labels.
    #[test]
    #[ignore = "needs a real flm install; run with --ignored --test-threads=1"]
    fn both_catalog_views_cover_the_whole_catalog() {
        let backend = FlmBackend::discover(&FastFlowLmConfig::default());
        let dir = std::env::temp_dir().join(format!("llmctl-flm-views-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ctx = |view| CatalogCtx {
            sources: &[],
            cache_path: Path::new("/nonexistent"),
            models_dir: &dir,
            view,
            // The second arrangement must come from the memoized catalog.
            reload: false,
        };

        let tags = |models: &[Model]| -> std::collections::BTreeSet<String> {
            models.iter().map(|m| m.flm().unwrap().tag.clone()).collect()
        };
        let categories = backend.models(&ctx(0));
        let flat = backend.models(&ctx(FLAT_VIEW));

        assert_eq!(tags(&categories), tags(&flat), "the views disagree about the catalog");
        // Flat is one row per model; Categories repeats the multi-label ones.
        assert_eq!(flat.len(), tags(&flat).len());
        assert!(categories.len() > flat.len());
        // Flat sits directly under local/online, with no capability level.
        assert!(flat.iter().all(|m| m.catalog_path.len() == 2));
        assert!(categories.iter().all(|m| m.catalog_path.len() == 3));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tags_sanitize_into_one_path_segment() {
        assert_eq!(sanitize("qwen3:4b"), "qwen3_4b");
        assert_eq!(sanitize("qwen3.6-moe:35b-a3b"), "qwen3.6-moe_35b-a3b");
        assert!(!sanitize("a/b:c").contains('/'));
    }
}
