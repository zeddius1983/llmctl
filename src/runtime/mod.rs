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

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

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
    /// Discard any memoized enumeration and read the source again.
    ///
    /// Enumeration can be expensive — FastFlowLM shells out to `flm list`, which
    /// costs ~150 ms through a launcher script — so a backend is free to serve
    /// repeat calls from memory. This is how a caller says it has reason to
    /// believe the source changed: the `F5` refresh, or a download finishing.
    /// Rearranging the same catalog does not set it.
    pub reload: bool,
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

/// A directory that is worth nothing once the directories it depends on are
/// empty, and can then be removed whole.
///
/// This exists for the Hugging Face cache: removing the last artifact of a
/// repository empties `blobs/` and `snapshots/` but leaves a `refs/main`
/// pointer naming a revision whose files are gone. Pruning empty directories
/// alone would never clear it, and only the runtime knows the combination is a
/// husk rather than someone else's data.
pub struct Husk {
    /// The directory to remove recursively.
    pub dir: PathBuf,
    /// Removal happens only if each of these is absent or empty.
    pub empty_first: Vec<PathBuf>,
}

/// What removing a model from local storage would delete, computed before
/// anything is unlinked.
///
/// Nothing is unlinked until the user agrees to the plan, and then it is this
/// plan that runs rather than a fresh one — so what happens is what was
/// confirmed. `bytes` is the net figure the prompt quotes: files held back for
/// another model are excluded, because they are not what the user gets back.
#[derive(Default)]
pub struct Deletion {
    /// Files and symlinks to unlink.
    pub files: Vec<PathBuf>,
    /// Directory trees to remove wholesale (a FastFlowLM model directory).
    pub trees: Vec<PathBuf>,
    /// Directories to remove afterwards, deepest first, and only if unlinking
    /// left them empty. Never recursive: anything still in them is not ours.
    pub prune: Vec<PathBuf>,
    /// A directory that becomes meaningless once the deletion lands.
    pub husk: Option<Husk>,
    /// Bytes the deletion frees. Excludes anything left behind for another
    /// model, so it is what the user actually gets back.
    pub bytes: u64,
}

impl Deletion {
    /// Whether this plan would remove nothing, i.e. the model is not stored.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty() && self.trees.is_empty()
    }

    /// Unlink everything, then clear the directories that emptied out.
    ///
    /// An already-missing path is not an error: the plan is a snapshot, and
    /// something else having removed a file first is the outcome asked for.
    pub fn execute(&self) -> Result<()> {
        for path in &self.files {
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(err) if err.kind() == ErrorKind::NotFound => {}
                Err(err) => {
                    return Err(err).with_context(|| format!("removing {}", path.display()));
                }
            }
        }
        for tree in &self.trees {
            match std::fs::remove_dir_all(tree) {
                Ok(()) => {}
                Err(err) if err.kind() == ErrorKind::NotFound => {}
                Err(err) => {
                    return Err(err).with_context(|| format!("removing {}", tree.display()));
                }
            }
        }
        // Best-effort tidying: `remove_dir` fails on a non-empty directory,
        // which is exactly the "leave what is not ours" rule.
        for dir in &self.prune {
            let _ = std::fs::remove_dir(dir);
        }
        if let Some(husk) = &self.husk
            && husk.empty_first.iter().all(|dir| is_empty_dir(dir))
        {
            let _ = std::fs::remove_dir_all(&husk.dir);
        }
        Ok(())
    }
}

/// Whether `dir` is absent or holds no entries.
fn is_empty_dir(dir: &Path) -> bool {
    match std::fs::read_dir(dir) {
        Ok(mut entries) => entries.next().is_none(),
        Err(err) => err.kind() == ErrorKind::NotFound,
    }
}

/// Size on disk of one path, without following symlinks: a Hugging Face
/// snapshot link costs nothing, and the blob it points at is counted once in
/// its own right.
pub(crate) fn file_bytes(path: &Path) -> u64 {
    std::fs::symlink_metadata(path)
        .map(|meta| if meta.is_symlink() { 0 } else { meta.len() })
        .unwrap_or(0)
}

/// What a symlink points at, resolved against the link's own directory:
/// `huggingface_hub` writes relative links, llmctl absolute ones, and the same
/// Hub cache holds both. `None` for anything that is not a symlink.
pub(crate) fn link_target(link: &Path) -> Option<PathBuf> {
    let target = std::fs::read_link(link).ok()?;
    if target.is_absolute() { Some(target) } else { Some(link.parent()?.join(target)) }
}

/// A path with every symlink resolved, or the path itself if it cannot be.
/// Used as file identity: two catalog entries that resolve here are one model.
pub(crate) fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Total size of every regular file under `dir`.
pub(crate) fn tree_bytes(dir: &Path) -> u64 {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| entry.metadata().ok())
        .map(|meta| meta.len())
        .sum()
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

    /// The on-disk footprint of `model`, so the user can reclaim it. `None`
    /// when this runtime stores nothing for the model — it was never
    /// downloaded, or the runtime has no notion of local storage at all.
    ///
    /// `catalog` is this runtime's full model list. A companion file shared
    /// with another stored model — the projector or dFlash drafter paired with
    /// every quantization of a repository — has to survive the deletion, and
    /// nothing but the catalog can say whether it is still spoken for.
    fn deletion(&self, _model: &Model, _catalog: &[Model]) -> Option<Deletion> {
        None
    }

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

    /// Recover `key`'s value from a superseded option name in a stored profile,
    /// for the case where a renamed flag would otherwise drop a user's setting
    /// on load. Consulted only when the profile has no value under `key`.
    fn legacy_value(
        &self,
        _key: &str,
        _stored: &std::collections::BTreeMap<String, String>,
    ) -> Option<String> {
        None
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

    /// Whether this runtime can only have one model loaded at a time, so llmctl
    /// declines a second launch rather than spawning one that dies on load.
    ///
    /// How many servers to run is normally the user's call, not llmctl's: two
    /// llama.cpp servers that between them overcommit VRAM still start, and may
    /// well be what was wanted. This is for the narrower case where the device
    /// admits exactly one client and a second launch *cannot* succeed.
    fn single_session(&self) -> bool {
        false
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
