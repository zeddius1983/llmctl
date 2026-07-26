//! Inference runtimes behind a common [`RuntimeBackend`] interface.
//!
//! Everything that differs between runtimes lives behind this trait: how the
//! binary is discovered, which options exist and how they are spelled on the
//! command line, how models are enumerated, how a launch command is built, and
//! how readiness is probed. The browser, profile store, and session manager
//! above it are runtime-agnostic and dispatch through `&dyn RuntimeBackend`.
//!
//! Adding a runtime means adding a module here and one entry in [`discover`] —
//! not another branch in the app.

pub mod flm;
pub mod llama_cpp;

use std::path::Path;

use crate::config::{Config, Paths};
use crate::discovery::ModelSource;
use crate::domain::{Model, OptionItem, Runtime};
use crate::profiles::registry::{OptionKind, OptionSchema, OptionSpec};
use crate::profiles::templates::Template;
use crate::session::command::Command;
use crate::session::record::DownloadRecord;

pub use flm::FlmBackend;
pub use llama_cpp::LlamaCppBackend;

/// The filesystem context a backend needs to enumerate its models.
pub struct CatalogCtx<'a> {
    /// Configured model roots (llama.cpp scans these; FastFlowLM ignores them
    /// and asks `flm` for its own curated catalog).
    pub sources: &'a [ModelSource],
    /// `~/.cache/llmctl/models.json` — the scan cache.
    pub cache_path: &'a Path,
    /// `~/.config/llmctl/models/` — the managed catalog root holding per-model
    /// profiles.
    pub models_dir: &'a Path,
    /// Which of the runtime's [`RuntimeBackend::catalog_views`] to build, as an
    /// index into that list. Runtimes offering a single arrangement ignore it.
    pub view: usize,
}

/// Everything needed to build a launch/chat command for a selected model.
///
/// This is the runtime-agnostic slice of the selection; backends reach into
/// their own model metadata (`Model::flm`, `Model::remote`, …) for the rest.
pub struct LaunchContext<'a> {
    pub binary: &'a str,
    pub model: &'a Model,
    pub options: &'a [OptionItem],
}

/// One inference runtime: discovery result plus its dialect and behavior.
pub trait RuntimeBackend: Send + Sync {
    /// Identity and probe result, rendered in the Runtime column.
    fn descriptor(&self) -> &Runtime;

    /// This runtime's option vocabulary and CLI-encoding rules.
    fn schema(&self) -> &'static OptionSchema;

    /// Built-in profile templates, expressed in this runtime's option keys.
    fn templates(&self) -> &'static [Template];

    /// Flat model list for this runtime's subtree. Each `Model::catalog_path`
    /// places it in the browser tree; a model may appear at several paths (see
    /// the FastFlowLM label groups), in which case `profile_key` keeps them one
    /// logical model.
    fn models(&self, ctx: &CatalogCtx) -> Vec<Model>;

    /// A spec's kind specialized for a model — used to bound context length by
    /// what the model was trained for. Most options are model-independent.
    fn effective_kind(&self, spec: &OptionSpec, _model: &Model) -> OptionKind {
        spec.kind
    }

    /// The model-aware starting value for an option, before template and
    /// instance layers are applied.
    fn spec_default(
        &self,
        spec: &OptionSpec,
        model: &Model,
        defaults: &crate::config::Defaults,
    ) -> String;

    /// Map a stored value from an older llmctl onto the current vocabulary, so
    /// saved profiles keep launching. Most runtimes need no migration.
    fn normalize_legacy(&self, _key: &str, value: String) -> String {
        value
    }

    /// Clamp a resolved value into what this model actually supports.
    fn clamp_to_model(&self, _key: &str, value: String, _model: &Model) -> String {
        value
    }

    /// The server launch command.
    fn build_command(&self, ctx: &LaunchContext) -> Command;

    /// Argv for an interactive terminal chat, or `None` if unsupported.
    fn chat_argv(&self, ctx: &LaunchContext) -> Option<Vec<String>>;

    /// Argv for a throughput benchmark, or `None` if unsupported.
    fn bench_argv(&self, ctx: &LaunchContext) -> Option<Vec<String>>;

    /// HTTP path whose `200` means "the model is loaded and serving".
    fn health_path(&self) -> &'static str;

    /// The token identifying this model in the launched process's own argv — a
    /// GGUF path for llama.cpp, a `name:size` tag for FastFlowLM. llmctl
    /// records it to re-acquire the process from `/proc` (see
    /// [`crate::session::proc::find_server`]), so it must appear in the argv
    /// the backend builds.
    fn process_token(&self, ctx: &LaunchContext) -> String;

    /// Why this particular launch cannot proceed — a missing server capability,
    /// say — or `None` if it is fine. Distinct from
    /// [`RuntimeBackend::unavailable_reason`], which is about the runtime
    /// itself rather than the selected model and options.
    fn launch_blocker(&self, _ctx: &LaunchContext) -> Option<String> {
        None
    }

    /// Artifacts this launch will download before serving, so the session shows
    /// progress instead of appearing hung.
    fn launch_download(&self, _ctx: &LaunchContext) -> Option<DownloadRecord> {
        None
    }

    /// Alternative arrangements of this runtime's catalog, cycled with `s` and
    /// named in the pane title. Empty when the catalog has only one shape.
    ///
    /// This is the counterpart to the Hub's sort orders, which only mean
    /// something for a runtime whose catalog is fetched from Hugging Face —
    /// hence [`RuntimeBackend::supports_online_browse`] governing that instead.
    fn catalog_views(&self) -> &'static [&'static str] {
        &[]
    }

    /// Whether this runtime offers the lazy online (Hugging Face) subtree with
    /// its repository/artifact drill-down, Hub-wide search, and sort orders.
    ///
    /// This is what separates llama.cpp's `online` subtree — a live view of the
    /// whole Hub — from FastFlowLM's, which is a fixed catalog that happens to
    /// be hosted there. Both use the name `online`, so the path alone cannot
    /// tell them apart.
    fn supports_online_browse(&self) -> bool {
        false
    }

    /// Why this runtime cannot launch right now, or `None` if it is usable.
    /// Rendered in the status line and returned when a launch is attempted.
    fn unavailable_reason(&self) -> Option<String> {
        self.descriptor()
            .binary_path
            .is_none()
            .then(|| format!("{} binary not found on PATH", self.descriptor().name))
    }
}

/// The built-in templates of a runtime identified only by name.
///
/// The profile store keys everything by runtime *name*, because that is what it
/// reads back off disk long before any backend has been probed. This is the one
/// place a name has to be resolved without a `&dyn RuntimeBackend` in hand —
/// everywhere else, dispatch goes through the trait.
pub fn templates_for(runtime: &str) -> &'static [Template] {
    match runtime {
        flm::NAME => flm::TEMPLATES,
        _ => llama_cpp::TEMPLATES,
    }
}

/// Resolve a binary to an absolute path: honor an explicit path, else search
/// `$PATH`.
pub(crate) fn resolve_binary(binary: &str) -> Option<std::path::PathBuf> {
    let candidate = Path::new(binary);
    if candidate.is_absolute() || binary.contains('/') {
        return candidate.exists().then(|| candidate.to_path_buf());
    }
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var).map(|dir| dir.join(binary)).find(|p| p.is_file())
}

/// Probe every known runtime, in display order.
pub fn discover(config: &Config, paths: &Paths) -> Vec<Box<dyn RuntimeBackend>> {
    vec![
        Box::new(LlamaCppBackend::discover(&config.runtime.llama_cpp, &paths.cache_dir)),
        Box::new(FlmBackend::discover(&config.runtime.fastflowlm)),
    ]
}
