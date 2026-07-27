//! Application state and the input/event loop.
//!
//! Navigation follows Yazi's miller-columns: child panes are derived from the
//! parent's selection and only revealed one level ahead of focus (see
//! `IMPLEMENTATION_PLAN.md` → Navigation model).

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::DefaultTerminal;
use ratatui::widgets::ListState;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use crate::config::{Config, ModelLayout, ModelSourceConfig, Paths};
use crate::discovery;
use crate::discovery::ModelSource;
use crate::domain::{Model, OptionItem, Profile, format_unix_date, human_size};
use crate::profiles::{self, ProfileStore};
use crate::runtime::{CatalogCtx, LaunchContext, RuntimeBackend};
use crate::session::{self, LaunchRequest, SessionManager};
use crate::ui;

/// What a modal text prompt is collecting.
#[derive(Clone)]
pub enum PromptKind {
    EditOption { key: String },
    NewProfile,
    RenameProfile { old: String },
    DuplicateProfile { src: String },
}

/// A modal text input (option editing or profile naming).
pub struct Prompt {
    pub kind: PromptKind,
    pub title: String,
    pub buffer: String,
    pub error: Option<String>,
}

/// A read-only modal message (launch-command preview, copy confirmation,
/// errors). Dismissed by any key.
pub struct Message {
    pub title: String,
    pub lines: Vec<String>,
}

/// Enums with more variants than this open a [`Selector`] popup on `e`/Enter
/// instead of cycling in place.
const SELECTOR_THRESHOLD: usize = 8;

/// A modal single-select list (combo box) for large enums like chat-template:
/// type to filter, arrows to move, Enter to pick — instead of blind cycling.
pub struct Selector {
    /// Option key the picked value applies to.
    pub key: String,
    pub title: String,
    /// All enum variants, in registry order.
    pub variants: Vec<String>,
    /// Case-insensitive substring filter typed so far.
    pub filter: String,
    /// Cursor index into [`Self::filtered`].
    pub cursor: usize,
}

pub struct ModelSearch {
    pub query: String,
    pub cursor: usize,
    result_indices: Vec<usize>,
    pub online: bool,
    pub scope: Vec<String>,
}

#[derive(Debug)]
pub enum ModelDownloadStatus {
    Downloading,
    Cancelling,
    Downloaded(PathBuf),
    Cancelled,
    Interrupted,
    Failed(String),
}

/// Where a tracked download's bytes come from, and how to fetch them.
#[derive(Clone)]
enum DownloadSource {
    /// Hugging Face blobs that llmctl fetches into the Hub cache itself.
    Hub(Box<crate::domain::RemoteModel>),
    /// A FastFlowLM model, fetched from its Hugging Face repository into
    /// `flm`'s model directory. Deliberately not `flm pull` — that cannot
    /// resume and mis-detects partial downloads.
    Flm(Box<crate::domain::Model>),
}

pub struct ModelDownload {
    id: u64,
    pub model_id: String,
    pub model: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub status: ModelDownloadStatus,
    source: DownloadSource,
    cancelled: Arc<AtomicBool>,
}

impl ModelDownload {
    pub fn percent(&self) -> u8 {
        transfer_percent(self.downloaded_bytes, self.total_bytes)
    }
}

fn transfer_percent(downloaded_bytes: u64, total_bytes: u64) -> u8 {
    if total_bytes == 0 {
        return 0;
    }
    ((downloaded_bytes as u128 * 100 / total_bytes as u128).min(100)) as u8
}

fn restore_model_downloads(models_dir: &std::path::Path) -> (Vec<ModelDownload>, u64) {
    let mut downloads = Vec::new();
    let mut next_id = 1_u64;
    for record in discovery::online::load_download_records(models_dir) {
        let total_bytes = record.remote.blobs.iter().map(|blob| blob.size_bytes).sum();
        let downloaded_bytes = discovery::online::cached_downloaded_bytes(&record.remote);
        if total_bytes > 0
            && downloaded_bytes >= total_bytes
            && discovery::online::finalize_cached_download(&record.remote).is_ok()
        {
            if let Err(error) =
                discovery::online::delete_download_record(models_dir, &record.model_id)
            {
                tracing::warn!(%error, model = %record.model_id, "failed to remove completed download record");
            }
            continue;
        }
        let status = if total_bytes == 0 {
            ModelDownloadStatus::Failed("persisted download has no blob size metadata".into())
        } else {
            ModelDownloadStatus::Interrupted
        };
        downloads.push(ModelDownload {
            id: next_id,
            model_id: record.model_id,
            model: record.model,
            downloaded_bytes,
            total_bytes,
            status,
            source: DownloadSource::Hub(Box::new(record.remote)),
            cancelled: Arc::new(AtomicBool::new(false)),
        });
        next_id = next_id.wrapping_add(1).max(1);
    }
    (downloads, next_id)
}

enum ModelDownloadEvent {
    Progress { id: u64, downloaded_bytes: u64, total_bytes: u64 },
    Finished { id: u64, result: std::result::Result<discovery::online::DownloadResult, String> },
}

struct CatalogRoute {
    items: Vec<Model>,
    selected: usize,
    prefix: Vec<String>,
    history: Vec<(Vec<Model>, Option<usize>, Vec<String>)>,
}

impl Selector {
    /// Variants matching the current filter (case-insensitive substring).
    pub fn filtered(&self) -> Vec<&str> {
        let needle = self.filter.to_lowercase();
        self.variants
            .iter()
            .filter(|v| v.to_lowercase().contains(&needle))
            .map(String::as_str)
            .collect()
    }

    /// The variant under the cursor, if any survives the filter.
    pub fn selected(&self) -> Option<&str> {
        self.filtered().get(self.cursor).copied()
    }
}

/// The top-level screen the UI is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    /// The Yazi-style runtime/model/profile/options browser.
    Browser,
    /// The Session Manager (running servers).
    Sessions,
    /// A session's log tail.
    Logs,
}

/// The four navigable panes. The Info pane is always visible and never focused;
/// it previews whatever is selected in the focused pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pane {
    Runtime,
    Model,
    Profile,
    Options,
}

impl Pane {
    /// Navigation moves strictly left→right: Runtime → Model → Profile → Options.
    pub fn next(self) -> Self {
        match self {
            Pane::Runtime => Pane::Model,
            Pane::Model => Pane::Profile,
            Pane::Profile => Pane::Options,
            Pane::Options => Pane::Options,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Pane::Runtime => Pane::Runtime,
            Pane::Model => Pane::Runtime,
            Pane::Profile => Pane::Model,
            Pane::Options => Pane::Profile,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Pane::Runtime => 0,
            Pane::Model => 1,
            Pane::Profile => 2,
            Pane::Options => 3,
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Pane::Runtime => "Runtime",
            Pane::Model => "Model",
            Pane::Profile => "Profile",
            Pane::Options => "Options",
        }
    }
}

/// A list of items plus its selection cursor.
pub struct PaneList<T> {
    pub items: Vec<T>,
    pub state: ListState,
}

impl<T> PaneList<T> {
    fn new(items: Vec<T>) -> Self {
        let mut list = Self { items, state: ListState::default() };
        list.select_first();
        list
    }

    pub fn selected(&self) -> Option<&T> {
        self.state.selected().and_then(|i| self.items.get(i))
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Replace contents and reset the cursor to the top (new subtree).
    fn replace(&mut self, items: Vec<T>) {
        self.items = items;
        self.select_first();
    }

    fn move_by(&mut self, delta: isize) {
        if self.items.is_empty() {
            return;
        }
        let len = self.items.len() as isize;
        let current = self.state.selected().unwrap_or(0) as isize;
        let next = (current + delta).clamp(0, len - 1);
        self.state.select(Some(next as usize));
    }

    fn select_first(&mut self) {
        self.state.select(if self.items.is_empty() { None } else { Some(0) });
    }

    fn select_last(&mut self) {
        if !self.items.is_empty() {
            self.state.select(Some(self.items.len() - 1));
        }
    }
}

pub struct App {
    #[allow(dead_code)] // retained for Phase 2+ (profiles, defaults)
    pub config: Config,
    pub focus: Pane,
    pub runtimes: PaneList<Box<dyn RuntimeBackend>>,
    pub models: PaneList<Model>,
    /// Child nodes of the selected catalog directory (empty for a model leaf).
    pub catalog_preview: Vec<Model>,
    pub profiles: PaneList<Profile>,
    pub options: PaneList<OptionItem>,
    pub show_help: bool,
    pub prompt: Option<Prompt>,
    /// A modal enum-variant selector (combo box), if open.
    pub selector: Option<Selector>,
    pub model_search: Option<ModelSearch>,
    /// A read-only modal message overlay, if any.
    pub message: Option<Message>,
    /// Which top-level screen is active.
    pub screen: Screen,
    /// Running/known inference sessions.
    pub sessions: SessionManager,
    /// Selection cursor in the Session Manager list.
    pub session_sel: ListState,
    /// Loaded log lines for the Logs screen.
    pub log_lines: Vec<String>,
    /// Whether the log view tails the bottom of the file.
    pub log_follow: bool,
    /// Scroll offset (lines from the top) for the log view when not following.
    pub log_scroll: u16,
    should_quit: bool,
    /// Discovered GGUF models for the llama.cpp runtime, plus its cached online
    /// tree. The online browsing machinery below is llama.cpp-specific, so this
    /// list stays llama.cpp's alone.
    scanned_models: Vec<Model>,
    /// FastFlowLM's catalog, from `flm list`. Unlike the GGUF scan this is a
    /// single curated list covering installed and available models alike.
    flm_models: Vec<Model>,
    catalog_prefix: Vec<String>,
    catalog_history: Vec<(Vec<Model>, Option<usize>, Vec<String>)>,
    /// Expanded, absolute model search directories.
    model_sources: Vec<ModelSource>,
    model_cache: PathBuf,
    models_dir: PathBuf,
    /// Persisted, model-scoped profile instances.
    store: ProfileStore,
    /// Last time live session status was refreshed.
    last_tick: Instant,
    /// A foreground interactive chat (`llama-cli`) to run on the next loop turn,
    /// suspending the TUI while it owns the terminal.
    pending_chat: Option<Vec<String>>,
    /// A foreground `llama-bench` invocation for the selected model.
    pending_benchmark: Option<Vec<String>>,
    online_tx: Sender<discovery::online::Response>,
    online_rx: Receiver<discovery::online::Response>,
    online_pending: Option<discovery::online::Request>,
    online_search_due: Option<(Instant, String)>,
    online_sort: discovery::online::Sort,
    /// Which of the selected runtime's `catalog_views` is active (cycled with
    /// `s`). Runtimes offering a single arrangement ignore it.
    catalog_view: usize,
    online_epoch: u64,
    online_reload_deferred: bool,
    online_restore_models: bool,
    online_search_results: Vec<String>,
    download_tx: Sender<ModelDownloadEvent>,
    download_rx: Receiver<ModelDownloadEvent>,
    pub model_downloads: Vec<ModelDownload>,
    next_download_id: u64,
}

impl App {
    pub fn new(config: Config, paths: Paths) -> Self {
        let runtimes = crate::runtime::discover(&config, &paths);
        let model_sources = resolve_model_sources(&config.models.paths, &config.models.sources);
        let model_cache = paths.cache_dir.join("models.json");
        let mut scanned_models = discovery::scan_models(&model_sources, &model_cache);
        discovery::reconcile(&paths.models_dir, &mut scanned_models);
        let online_sort = discovery::online::cached_sort(&paths.models_dir);
        let (model_downloads, next_download_id) = restore_model_downloads(&paths.models_dir);
        scanned_models.extend(discovery::online::load_cached(&paths.models_dir));

        let catalog_ctx = CatalogCtx {
            sources: &model_sources,
            cache_path: &model_cache,
            models_dir: &paths.models_dir,
            view: 0,
            // Nothing is memoized yet, so this reads from `flm` either way.
            reload: false,
        };
        let flm_models = runtimes
            .iter()
            .find(|backend| !backend.supports_online_browse())
            .map(|backend| backend.models(&catalog_ctx))
            .unwrap_or_default();

        let mut all_models = scanned_models.clone();
        all_models.extend(flm_models.iter().cloned());
        let store = ProfileStore::load(paths.state_dir.join("profiles.json"), &all_models);
        // Built after discovery's one-shot `Command`s: the supervisor ignores
        // SIGCHLD, which would otherwise prevent reaping those probe processes.
        let sessions = SessionManager::new(paths.sessions_dir.clone(), paths.log_dir.clone());

        let (online_tx, online_rx) = mpsc::channel();
        let (download_tx, download_rx) = mpsc::channel();
        let mut app = Self {
            config,
            focus: Pane::Runtime,
            runtimes: PaneList::new(runtimes),
            models: PaneList::new(Vec::new()),
            catalog_preview: Vec::new(),
            profiles: PaneList::new(Vec::new()),
            options: PaneList::new(Vec::new()),
            show_help: false,
            prompt: None,
            selector: None,
            model_search: None,
            message: None,
            screen: Screen::Browser,
            sessions,
            session_sel: ListState::default(),
            log_lines: Vec::new(),
            log_follow: true,
            log_scroll: 0,
            should_quit: false,
            scanned_models,
            flm_models,
            catalog_prefix: Vec::new(),
            catalog_history: Vec::new(),
            model_sources,
            model_cache,
            models_dir: paths.models_dir,
            store,
            last_tick: Instant::now(),
            pending_chat: None,
            pending_benchmark: None,
            online_tx,
            online_rx,
            online_pending: None,
            online_search_due: None,
            online_sort,
            catalog_view: 0,
            online_epoch: 0,
            online_reload_deferred: false,
            online_restore_models: false,
            online_search_results: Vec::new(),
            download_tx,
            download_rx,
            model_downloads,
            next_download_id,
        };
        app.sync_session_selection();
        // Derive the whole chain from the initially-selected runtime.
        app.rebuild_below(Pane::Runtime);
        app
    }

    /// Run the draw/input loop until the user quits. A short poll timeout drives
    /// a periodic tick so live session status/resources stay current without
    /// blocking on input (no async runtime needed — see ADR-007).
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.should_quit {
            self.poll_online();
            self.poll_online_search();
            self.poll_model_download();
            self.poll_restarts();
            if self.last_tick.elapsed() >= Duration::from_secs(1) {
                self.tick();
            }
            terminal.draw(|frame| ui::draw(frame, self))?;
            if event::poll(Duration::from_millis(250))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.on_key(key);
                    }
                }
            }
            // A chat request hands the terminal to llama-cli, then we re-enter.
            if let Some(argv) = self.pending_chat.take() {
                run_foreground(terminal, &argv, "chat")?;
            }
            if let Some(argv) = self.pending_benchmark.take() {
                run_foreground(terminal, &argv, "benchmark")?;
            }
        }
        Ok(())
    }

    /// Finish any restart whose old process has exited. Polled on the input
    /// loop rather than the one-second tick so a restart comes back promptly.
    fn poll_restarts(&mut self) {
        let errors = self.sessions.poll_restarts();
        if !errors.is_empty() {
            self.message = Some(Message { title: "Restart failed".into(), lines: errors });
        }
    }

    /// Periodic refresh: update live session status/resources, and reload the
    /// log tail when the Logs screen is open.
    fn tick(&mut self) {
        self.sessions.refresh();
        self.sync_session_selection();
        self.reconcile_downloaded_online_models();
        if self.screen == Screen::Logs {
            self.reload_logs();
        }
        self.last_tick = Instant::now();
    }

    fn reconcile_downloaded_online_models(&mut self) {
        let has_remote_session =
            self.sessions.sessions.iter().any(|session| {
                session.record.command.iter().any(|argument| argument == "--hf-repo")
            });
        let has_incomplete_remote = self.scanned_models.iter().any(|model| {
            model.remote.as_ref().is_some_and(|remote| {
                remote.file.is_some()
                    && (model.path.as_os_str().is_empty()
                        || (remote.mtp_file.is_some() && model.mtp_path.is_none())
                        || (remote.projector_file.is_some() && model.projector_path.is_none()))
            })
        });
        if !has_remote_session || !has_incomplete_remote {
            return;
        }
        let models = discovery::online::load_cached(&self.models_dir);
        let newly_cached = models.iter().any(|fresh| {
            self.scanned_models.iter().find(|old| old.id == fresh.id).is_some_and(|old| {
                (old.path.as_os_str().is_empty() && !fresh.path.as_os_str().is_empty())
                    || (old.mtp_path.is_none() && fresh.mtp_path.is_some())
                    || (old.projector_path.is_none() && fresh.projector_path.is_some())
            })
        });
        if newly_cached {
            self.scanned_models
                .retain(|model| !discovery::online::is_online_path(&model.catalog_path));
            self.scanned_models.extend(models);
            self.store.sync_models(&self.scanned_models);
            self.rebuild_below(Pane::Model);
        }
    }

    fn poll_online(&mut self) {
        while let Ok(response) = self.online_rx.try_recv() {
            if response.epoch != self.online_epoch {
                if self.online_reload_deferred {
                    self.online_pending = None;
                    self.online_reload_deferred = false;
                    self.perform_online_reload();
                }
                continue;
            }
            if self.online_pending.as_ref() == Some(&response.request) {
                self.online_pending = None;
            }
            let search_query = match &response.request {
                discovery::online::Request::Search { query, .. } => Some(query.clone()),
                _ => None,
            };
            match response.result {
                Ok(models) if search_query.is_some() => {
                    let query = search_query.as_deref().unwrap_or_default();
                    let current = self
                        .model_search
                        .as_ref()
                        .is_some_and(|search| search.online && search.query == query);
                    if current {
                        self.replace_online_search_results(models);
                        self.refresh_model_search();
                    }
                }
                Ok(models) => {
                    self.scanned_models
                        .retain(|model| !discovery::online::is_online_path(&model.catalog_path));
                    self.scanned_models.extend(models);
                    self.store.sync_models(&self.scanned_models);
                    if self.online_restore_models {
                        self.show_online_models_root();
                        self.online_restore_models = false;
                    } else {
                        self.rebuild_below(Pane::Model);
                    }
                    self.refresh_model_search();
                }
                Err(error) => {
                    self.online_restore_models = false;
                    self.message = Some(Message {
                        title: "Hugging Face unavailable".into(),
                        lines: vec![error, "Cached entries remain available.".into()],
                    });
                }
            }
            if self.focus == Pane::Model {
                self.maybe_fetch_online(false);
            }
        }
    }

    fn clear_online_search_results(&mut self) {
        if self.online_search_results.is_empty() {
            return;
        }
        self.scanned_models
            .retain(|model| !self.online_search_results.iter().any(|id| id == &model.id));
        self.online_search_results.clear();
    }

    fn replace_online_search_results(&mut self, models: Vec<Model>) {
        self.clear_online_search_results();
        for model in models {
            if self.scanned_models.iter().any(|cached| cached.id == model.id) {
                continue;
            }
            self.online_search_results.push(model.id.clone());
            self.scanned_models.push(model);
        }
    }

    fn save_online_search_selection(&mut self, model: &Model) -> std::result::Result<(), String> {
        discovery::online::save_selected_repository(&self.models_dir, model, self.online_sort)
            .map_err(|error| error.to_string())?;
        self.clear_online_search_results();
        self.scanned_models
            .retain(|cached| !discovery::online::is_online_path(&cached.catalog_path));
        self.scanned_models.extend(discovery::online::load_cached(&self.models_dir));
        self.store.sync_models(&self.scanned_models);
        Ok(())
    }

    fn fetch_online(&mut self, request: discovery::online::Request) {
        if self.online_pending.is_some() {
            return;
        }
        self.online_pending = Some(request.clone());
        let root = self.models_dir.clone();
        let tx = self.online_tx.clone();
        let epoch = self.online_epoch;
        std::thread::spawn(move || {
            let result =
                discovery::online::fetch(&root, &request).map_err(|error| error.to_string());
            let _ = tx.send(discovery::online::Response { epoch, request, result });
        });
    }

    fn poll_online_search(&mut self) {
        let Some((changed, query)) = self.online_search_due.clone() else { return };
        if changed.elapsed() < Duration::from_millis(400) || self.online_pending.is_some() {
            return;
        }
        self.online_search_due = None;
        if !query.trim().is_empty() {
            self.fetch_online(discovery::online::Request::Search {
                query,
                author: None,
                sort: self.online_sort,
            });
        }
    }

    fn maybe_fetch_online(&mut self, force: bool) {
        // FastFlowLM names its remote group `online` too, and `request_for_path`
        // only inspects the path — without this, browsing its catalog would fire
        // Hugging Face requests for repositories that do not exist.
        if !self.runtimes.selected().is_some_and(|backend| backend.supports_online_browse()) {
            return;
        }
        let Some(selected) = self.models.selected() else { return };
        let Some(request) =
            discovery::online::request_for_path(&selected.catalog_path, self.online_sort)
        else {
            return;
        };
        let cached = match &request {
            discovery::online::Request::Repositories(_) => {
                self.scanned_models.iter().any(|model| {
                    model
                        .remote
                        .as_ref()
                        .is_some_and(|remote| remote.file.is_none() && !remote.repo.is_empty())
                })
            }
            discovery::online::Request::Repository(repo) => {
                let artifacts = self.scanned_models.iter().filter(|model| {
                    model
                        .remote
                        .as_ref()
                        .is_some_and(|remote| remote.repo == *repo && remote.file.is_some())
                });
                let mut found = false;
                let complete = artifacts.inspect(|_| found = true).all(|model| {
                    !model.path.as_os_str().is_empty()
                        || !model.remote.as_ref().unwrap().blobs.is_empty()
                });
                found && complete
            }
            discovery::online::Request::Search { .. } => true,
        };
        if force || !cached {
            self.fetch_online(request);
        }
    }

    /// Is the Hugging Face browsing surface — Hub-wide search, sort orders,
    /// lazy repository fetches — active right now?
    ///
    /// The path alone is not enough: FastFlowLM also names its remote group
    /// `online`, but that is a fixed catalog rather than a live view of the Hub,
    /// so the runtime has to agree.
    pub fn online_view_active(&self) -> bool {
        self.runtimes.selected().is_some_and(|backend| backend.supports_online_browse())
            && self.focus >= Pane::Model
            && (discovery::online::is_online_path(&self.catalog_prefix)
                || self
                    .models
                    .selected()
                    .is_some_and(|model| discovery::online::is_online_path(&model.catalog_path)))
    }

    /// Title for a catalog pane showing `prefix`.
    ///
    /// A runtime with alternative arrangements names the active one; otherwise
    /// the Hub subtree names its sort order, and everything else is "Model".
    fn catalog_title(&self, prefix: &[String]) -> String {
        match self.catalog_view_label() {
            Some(view) => view.to_string(),
            None => model_catalog_title(prefix, self.online_sort),
        }
    }

    pub fn model_pane_title(&self) -> String {
        self.catalog_title(&self.catalog_prefix)
    }

    pub fn catalog_parent_title(&self) -> String {
        match self.catalog_history.last() {
            Some((_, _, prefix)) => self.catalog_title(prefix),
            None => "Model".into(),
        }
    }

    pub fn catalog_preview_title(&self) -> String {
        self.models
            .selected()
            .filter(|model| model.is_catalog_dir())
            .map(|model| self.catalog_title(&model.catalog_path))
            .unwrap_or_else(|| self.model_pane_title())
    }

    /// How many arrangements the selected runtime offers for its catalog.
    fn catalog_view_count(&self) -> usize {
        self.runtimes.selected().map(|backend| backend.catalog_views().len()).unwrap_or(0)
    }

    /// The name of the active arrangement, for the pane title.
    pub fn catalog_view_label(&self) -> Option<&'static str> {
        let views = self.runtimes.selected()?.catalog_views();
        if views.len() < 2 {
            return None;
        }
        views.get(self.catalog_view % views.len()).copied()
    }

    /// Switch to the next catalog arrangement (`s`), rebuilding the tree.
    fn cycle_catalog_view(&mut self) {
        let count = self.catalog_view_count();
        if count < 2 {
            return;
        }
        // Remember where we are browsing, and what is selected if it is a
        // model, so the new arrangement can restore both.
        let prefix = self.catalog_prefix.clone();
        let selection = self.selected_model().map(|model| model.id.clone());

        self.catalog_view = (self.catalog_view + 1) % count;
        self.catalog_history.clear();
        self.catalog_prefix.clear();
        // Same catalog, arranged differently — no reason to ask `flm` again.
        self.refresh_flm_models(false);
        self.rebuild_below(Pane::Runtime);

        self.restore_catalog_position(selection.as_deref(), &prefix);
    }

    fn cycle_online_sort(&mut self) {
        if !self.online_view_active() {
            return;
        }
        self.online_sort = self.online_sort.next();
        self.reload_online_layout();
    }

    fn reload_online_layout(&mut self) {
        self.online_search_due = None;
        self.model_search = None;
        self.clear_online_search_results();
        if self.online_pending.is_some() {
            // Let the sole writer finish, then clear what it wrote before
            // starting the replacement request. This prevents a stale worker
            // from repopulating the cache after a view switch.
            self.online_epoch = self.online_epoch.wrapping_add(1);
            self.online_reload_deferred = true;
            return;
        }
        self.perform_online_reload();
    }

    fn perform_online_reload(&mut self) {
        self.online_epoch = self.online_epoch.wrapping_add(1);
        self.online_pending = None;
        if let Err(error) = discovery::online::clear_cached_layout(&self.models_dir) {
            self.message = Some(Message {
                title: "Cannot reset online catalog".into(),
                lines: vec![error.to_string()],
            });
            return;
        }

        self.scanned_models.retain(|model| !discovery::online::is_online_path(&model.catalog_path));
        self.scanned_models.extend(discovery::online::load_cached(&self.models_dir));
        self.store.sync_models(&self.scanned_models);

        self.online_restore_models = true;
        self.show_online_models_root();
        self.fetch_online(discovery::online::Request::Repositories(self.online_sort));
    }

    fn show_online_models_root(&mut self) {
        if let Some(runtime) =
            self.runtimes.items.iter().position(|backend| backend.supports_online_browse())
        {
            self.runtimes.state.select(Some(runtime));
        }
        let online = vec!["online".to_string()];
        let huggingface = vec!["online".to_string(), "huggingface".to_string()];
        let root_items = self.catalog_children(&[]);
        let Some(online_selected) =
            root_items.iter().position(|model| model.catalog_path == online)
        else {
            return;
        };
        let online_items = self.catalog_children(&online);
        let Some(huggingface_selected) =
            online_items.iter().position(|model| model.catalog_path == huggingface)
        else {
            return;
        };
        self.catalog_history = vec![
            (root_items, Some(online_selected), Vec::new()),
            (online_items, Some(huggingface_selected), online),
        ];
        self.catalog_prefix = huggingface.clone();
        self.models.replace(self.catalog_children(&huggingface));
        self.focus = Pane::Model;
        self.rebuild_below(Pane::Model);
    }

    fn on_key(&mut self, key: KeyEvent) {
        // A read-only message overlay is dismissed by any key.
        if self.message.is_some() {
            self.message = None;
            return;
        }
        // A text prompt is modal: it consumes all input until closed.
        if self.prompt.is_some() {
            self.prompt_key(key);
            return;
        }
        // So is the enum-variant selector.
        if self.selector.is_some() {
            self.selector_key(key);
            return;
        }
        if self.model_search.is_some() {
            self.model_search_key(key);
            return;
        }
        // Help overlay swallows input apart from its own dismissal keys.
        if self.show_help {
            match key.code {
                KeyCode::Char('?') | KeyCode::Esc | KeyCode::Char('q') => self.show_help = false,
                _ => {}
            }
            return;
        }

        match self.screen {
            Screen::Browser => self.on_key_browser(key),
            Screen::Sessions => self.on_key_sessions(key),
            Screen::Logs => self.on_key_logs(key),
        }
    }

    /// Key handling for the Yazi-style browser screen.
    fn on_key_browser(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('?') => self.show_help = true,
            KeyCode::Char('/') => {
                let selected_dir = self
                    .models
                    .selected()
                    .filter(|model| model.is_catalog_dir())
                    .map(|model| model.catalog_path.as_slice());
                let hub = self
                    .runtimes
                    .selected()
                    .is_some_and(|backend| backend.supports_online_browse());
                let scope =
                    normalized_search_scope(self.focus, selected_dir, &self.catalog_prefix, hub);
                // FastFlowLM's `online` group is a local catalog, so `/` filters
                // it in place rather than querying the Hub.
                let online = hub && discovery::online::is_online_path(&scope);
                self.model_search = Some(ModelSearch {
                    query: String::new(),
                    cursor: 0,
                    result_indices: self.ranked_model_indices("", &scope, online),
                    online,
                    scope,
                })
            }
            KeyCode::Char('t') => self.open_sessions(),
            KeyCode::Char('y') => self.yank_command(),
            KeyCode::Char('s') if self.focus == Pane::Model && self.online_view_active() => {
                self.cycle_online_sort()
            }
            KeyCode::Char('s') if self.focus == Pane::Model && self.catalog_view_count() > 1 => {
                self.cycle_catalog_view()
            }
            KeyCode::Char('s') => self.start_session(),
            KeyCode::Char('C') => self.start_chat(),
            KeyCode::Char('b') => self.start_benchmark(),
            // Move focus across panes. In Options (the leaf) Enter edits the
            // selected value instead; `l`/Right stay pure navigation.
            KeyCode::Enter if self.focus == Pane::Options => self.open_editor(),
            KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => self.enter(),
            KeyCode::Char('h') | KeyCode::Left => self.go_back(),

            // Move selection within the focused pane.
            KeyCode::Char('j') | KeyCode::Down => self.move_selection(1),
            KeyCode::Char('k') | KeyCode::Up => self.move_selection(-1),
            KeyCode::Char('g') => self.select_first(),
            KeyCode::Char('G') => self.select_last(),

            // In Options, Home/End jump an option to its min/max; elsewhere
            // they move to the first/last list item.
            KeyCode::Home if self.focus == Pane::Options => self.set_option_extreme(-1),
            KeyCode::End if self.focus == Pane::Options => self.set_option_extreme(1),
            KeyCode::Home => self.select_first(),
            KeyCode::End => self.select_last(),

            // Inline option adjustment (Options pane).
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::Char(']') => self.adjust_option(1),
            KeyCode::Char('-') | KeyCode::Char('[') => self.adjust_option(-1),

            // Edit the selected option / toggle the selected profile favorite.
            KeyCode::Char('e') => self.open_editor(),
            KeyCode::Char('f') => self.toggle_favorite(),

            // Profile management (Profile pane); in Options, `d` resets the
            // selected option to its resolved default instead.
            KeyCode::Char('a') => self.prompt_new_profile(),
            KeyCode::Char('r') => self.prompt_rename_profile(),
            KeyCode::Char('D') => self.prompt_duplicate_profile(),
            KeyCode::Char('d') if self.focus == Pane::Model && self.download_available() => {
                self.download_selected_model()
            }
            KeyCode::Char('d') if self.focus == Pane::Options => self.reset_option_default(),
            KeyCode::Char('d') => self.delete_profile(),

            // Re-scan model directories.
            KeyCode::F(5) => self.refresh_models(),

            _ => {}
        }
    }

    fn model_search_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.model_search = None;
                self.online_search_due = None;
                self.clear_online_search_results();
            }
            KeyCode::Enter => {
                let target = self
                    .model_search
                    .as_ref()
                    .and_then(|s| s.result_indices.get(s.cursor))
                    .and_then(|i| self.catalog_source().get(*i))
                    .cloned();
                let promote = self.model_search.as_ref().is_some_and(|search| search.online)
                    && target.as_ref().is_some_and(|model| {
                        model.remote.as_ref().is_some_and(|remote| remote.file.is_none())
                    });
                self.model_search = None;
                self.online_search_due = None;
                if let Some(target) = target {
                    if promote && let Err(error) = self.save_online_search_selection(&target) {
                        self.message = Some(Message {
                            title: "Cannot save Hugging Face model".into(),
                            lines: vec![error],
                        });
                        return;
                    }
                    self.clear_online_search_results();
                    self.jump_to_model(&target.id);
                } else {
                    self.clear_online_search_results();
                }
            }
            KeyCode::Up => {
                if let Some(search) = self.model_search.as_mut() {
                    search.cursor = search.cursor.saturating_sub(1);
                }
            }
            KeyCode::Down => {
                if let Some(search) = self.model_search.as_mut() {
                    let max = search.result_indices.len().saturating_sub(1);
                    search.cursor = (search.cursor + 1).min(max);
                }
            }
            KeyCode::Backspace => {
                if let Some(search) = self.model_search.as_mut() {
                    search.query.pop();
                    search.cursor = 0;
                }
                self.refresh_model_search();
                self.schedule_online_search();
            }
            KeyCode::Char(c) => {
                if let Some(search) = self.model_search.as_mut() {
                    search.query.push(c);
                    search.cursor = 0;
                }
                self.refresh_model_search();
                self.schedule_online_search();
            }
            _ => {}
        }
    }

    pub fn search_results(&self) -> Vec<&Model> {
        let Some(search) = &self.model_search else { return Vec::new() };
        search.result_indices.iter().filter_map(|i| self.catalog_source().get(*i)).collect()
    }

    fn refresh_model_search(&mut self) {
        let Some((query, scope, online)) =
            self.model_search.as_ref().map(|s| (s.query.clone(), s.scope.clone(), s.online))
        else {
            return;
        };
        let results = self.ranked_model_indices(&query, &scope, online);
        if let Some(search) = self.model_search.as_mut() {
            search.result_indices = results;
            search.cursor = search.cursor.min(search.result_indices.len().saturating_sub(1));
        }
    }

    fn schedule_online_search(&mut self) {
        let Some(search) = &self.model_search else { return };
        if search.online && search.scope.len() <= 2 {
            self.online_search_due = Some((Instant::now(), search.query.clone()));
        }
    }

    fn ranked_model_indices(
        &self,
        raw_query: &str,
        scope: &[String],
        online_only: bool,
    ) -> Vec<usize> {
        rank_models(self.catalog_source(), raw_query, scope, online_only)
    }

    /// Navigate to a model by id, within the selected runtime. Returns whether
    /// the model was found and the browser moved.
    fn jump_to_model(&mut self, id: &str) -> bool {
        let Some(path) =
            self.catalog_source().iter().find(|m| m.id == id).map(|m| m.catalog_path.clone())
        else {
            return false;
        };
        let Some(route) = self.catalog_route(&path) else { return false };
        self.focus = Pane::Model;
        self.apply_catalog_route(route);
        true
    }

    /// Commit a resolved route — only ever called once the whole route exists,
    /// so the browser never lands half-way.
    fn apply_catalog_route(&mut self, route: CatalogRoute) {
        self.catalog_prefix = route.prefix;
        self.catalog_history = route.history;
        self.models.items = route.items;
        self.models.state.select(Some(route.selected));
        self.rebuild_below(Pane::Model);
        self.maybe_fetch_online(false);
    }

    /// Put the browser back where it was after the catalog was rebuilt in a
    /// different arrangement.
    ///
    /// The anchor is the directory being browsed, not the highlighted row: a
    /// row may be a folder that the new arrangement does not have, and landing
    /// *on* a folder is not the same as being *inside* one. Browsing
    /// `online ▸ chat` in Categories should leave you inside `online` in Flat,
    /// looking at models — not back at the group list with `online` highlighted.
    ///
    /// A selected model takes precedence, since it is identified by tag and so
    /// survives regrouping wherever it lands.
    fn restore_catalog_position(&mut self, selection: Option<&str>, prefix: &[String]) {
        if let Some(id) = selection
            && self.jump_to_model(id)
        {
            return;
        }
        self.descend_to_prefix(prefix);
    }

    /// Move the browser inside `prefix`, or the deepest part of it that still
    /// exists, listing that directory's contents.
    fn descend_to_prefix(&mut self, prefix: &[String]) {
        for depth in (1..=prefix.len()).rev() {
            let Some(route) = self.catalog_route(&prefix[..depth]) else { continue };
            self.apply_catalog_route(route);
            // `catalog_route` lands *on* the directory; step into it so the
            // pane shows its contents.
            if self.models.selected().is_some_and(|model| model.is_catalog_dir()) {
                self.enter();
            }
            return;
        }
    }

    fn catalog_route(&self, path: &[String]) -> Option<CatalogRoute> {
        let mut items = self.catalog_children(&[]);
        let mut prefix = Vec::new();
        let mut history = Vec::new();
        for (depth, component) in path.iter().enumerate() {
            let selected = items.iter().position(|m| m.display_label() == component)?;
            let node = &items[selected];
            let last = depth + 1 == path.len();
            if node.is_catalog_dir() {
                if last {
                    return Some(CatalogRoute { items, selected, prefix, history });
                }
                history.push((items.clone(), Some(selected), prefix.clone()));
                prefix = node.catalog_path.clone();
                items = self.catalog_children(&prefix);
            } else if last {
                return Some(CatalogRoute { items, selected, prefix, history });
            } else {
                return None;
            }
        }
        None
    }

    /// Key handling for the Session Manager screen.
    fn on_key_sessions(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('?') => self.show_help = true,
            KeyCode::Esc | KeyCode::Char('t') => self.screen = Screen::Browser,
            KeyCode::Char('j') | KeyCode::Down => self.move_session(1),
            KeyCode::Char('k') | KeyCode::Up => self.move_session(-1),
            KeyCode::Char('g') | KeyCode::Home => {
                let any = self.async_job_count() > 0;
                self.session_sel.select(any.then_some(0));
            }
            KeyCode::Char('G') | KeyCode::End => {
                let len = self.async_job_count();
                self.session_sel.select((len > 0).then_some(len - 1));
            }
            KeyCode::Char('x') => self.stop_async_job(false),
            KeyCode::Char('K') => self.stop_async_job(true),
            KeyCode::Char('R') => self.restart_async_job(),
            KeyCode::Char('d') => self.remove_async_job(),
            KeyCode::Char('c') => self.copy_endpoint(),
            KeyCode::Char('y') => self.yank_session_command(),
            KeyCode::Char('L') | KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => {
                self.open_logs()
            }
            KeyCode::F(5) => {
                self.sessions.rediscover();
                self.sync_session_selection();
            }
            _ => {}
        }
    }

    /// Key handling for the log-tail screen.
    fn on_key_logs(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Esc | KeyCode::Char('L') | KeyCode::Char('h') | KeyCode::Left => {
                self.screen = Screen::Sessions
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.log_follow = false;
                self.log_scroll = self.log_scroll.saturating_add(1);
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.log_follow = false;
                self.log_scroll = self.log_scroll.saturating_sub(1);
            }
            KeyCode::PageDown => {
                self.log_follow = false;
                self.log_scroll = self.log_scroll.saturating_add(10);
            }
            KeyCode::PageUp => {
                self.log_follow = false;
                self.log_scroll = self.log_scroll.saturating_sub(10);
            }
            KeyCode::Char('g') | KeyCode::Home => {
                self.log_follow = false;
                self.log_scroll = 0;
            }
            KeyCode::Char('G') | KeyCode::End => self.log_follow = true,
            KeyCode::F(5) => self.reload_logs(),
            _ => {}
        }
    }

    // --- Session manager / launch ------------------------------------------

    /// Switch to the Session Manager screen, refreshing live status first.
    fn open_sessions(&mut self) {
        self.screen = Screen::Sessions;
        self.sessions.refresh();
        self.sync_session_selection();
    }

    /// Keep the session selection cursor within the bounds of the session list.
    fn sync_session_selection(&mut self) {
        let len = self.async_job_count();
        if len == 0 {
            self.session_sel.select(None);
        } else {
            let i = self.session_sel.selected().unwrap_or(0).min(len - 1);
            self.session_sel.select(Some(i));
        }
    }

    fn move_session(&mut self, delta: isize) {
        let len = self.async_job_count();
        if len == 0 {
            return;
        }
        let cur = self.session_sel.selected().unwrap_or(0) as isize;
        let next = (cur + delta).clamp(0, len as isize - 1);
        self.session_sel.select(Some(next as usize));
    }

    /// Build a launch request from the current selection and resolved options.
    ///
    /// Everything runtime-specific — the argv, the readiness path, the process
    /// token, any pre-flight capability check — comes from the backend.
    fn build_launch_request(&self) -> Result<LaunchRequest, String> {
        let backend = self.runtimes.selected().ok_or("no runtime selected")?;
        if let Some(reason) = backend.unavailable_reason() {
            return Err(reason);
        }
        let model = self.selected_model().ok_or("no model selected")?;
        let profile = self.profiles.selected().ok_or("no profile selected")?;
        let options = self.options.items.clone();
        let binary = backend
            .descriptor()
            .binary_path
            .as_ref()
            .ok_or("runtime binary not found on PATH")?
            .display()
            .to_string();

        let ctx = LaunchContext { binary: &binary, model, options: &options };
        if let Some(blocker) = backend.launch_blocker(&ctx) {
            return Err(blocker);
        }

        let host = option_value(&options, "host").unwrap_or_else(|| "127.0.0.1".into());
        let port = option_value(&options, "port")
            .and_then(|v| v.parse().ok())
            .or_else(|| backend.schema().spec("port")?.default.parse().ok())
            .unwrap_or(8000);

        Ok(LaunchRequest {
            runtime: backend.descriptor().name.clone(),
            model: model.name.clone(),
            model_path: backend.process_token(&ctx),
            command: backend.build_command(&ctx),
            health_path: backend.health_path().to_string(),
            download: backend.launch_download(&ctx),
            profile: profile.name.clone(),
            host,
            port,
        })
    }

    /// Why a second model cannot be started on `runtime` right now, as message
    /// lines — or `None` if the runtime is happy to run several at once, or has
    /// nothing live.
    ///
    /// Kept out of [`App::build_launch_request`] on purpose: this blocks
    /// *starting* a model, not building its command, so `y` still previews and
    /// copies the launch line while a session holds the device.
    fn single_session_conflict(&self, runtime: &str) -> Option<Vec<String>> {
        let backend = self.runtimes.items.iter().find(|b| b.descriptor().name == runtime)?;
        if !backend.single_session() {
            return None;
        }
        let live = self.sessions.active_for_runtime(runtime)?;
        Some(vec![
            format!("{runtime} can only run one model at a time."),
            format!("{} is {}.", live.record.name, live.status_label()),
            "Stop it in the Session Manager (x), then start this one.".into(),
        ])
    }

    /// Preview the generated command and copy it to the clipboard (`y`).
    fn yank_command(&mut self) {
        if self.focus != Pane::Profile && self.focus != Pane::Options {
            return;
        }
        match self.build_launch_request() {
            Ok(req) => {
                copy_to_clipboard(&req.command.display());
                self.message = Some(Message {
                    title: "Launch command".into(),
                    lines: command_message_lines(&req.command),
                });
            }
            Err(e) => {
                self.message =
                    Some(Message { title: "Cannot build command".into(), lines: vec![e] })
            }
        }
    }

    /// Launch a server for the current selection and jump to the manager (`s`).
    fn start_session(&mut self) {
        if self.focus != Pane::Profile && self.focus != Pane::Options {
            return;
        }
        let req = match self.build_launch_request() {
            Ok(req) => req,
            Err(e) => {
                self.message = Some(Message { title: "Cannot launch".into(), lines: vec![e] });
                return;
            }
        };
        if let Some(lines) = self.single_session_conflict(&req.runtime) {
            self.message = Some(Message { title: "Cannot launch".into(), lines });
            return;
        }
        match self.sessions.launch(req) {
            Ok(idx) => {
                let endpoint = self.sessions.sessions[idx].record.endpoint();
                let name = self.sessions.sessions[idx].record.name.clone();
                let status = self.sessions.sessions[idx].status_label();
                self.screen = Screen::Sessions;
                self.session_sel.select(Some(idx));
                self.message = Some(Message {
                    title: "Launched".into(),
                    lines: vec![name, format!("{status} — {endpoint}")],
                });
            }
            Err(e) => {
                self.message =
                    Some(Message { title: "Launch failed".into(), lines: vec![e.to_string()] })
            }
        }
    }

    /// Start (or reveal) a download for the selected model — the `d` key.
    fn download_selected_model(&mut self) {
        let Some(model) = self.selected_model().cloned() else { return };
        if model.flm.is_some() {
            self.download_flm_model(&model);
            return;
        }
        let Some(remote) = model.remote.clone() else { return };
        let total_bytes = remote.blobs.iter().map(|blob| blob.size_bytes).sum();
        let downloaded_bytes = discovery::online::cached_downloaded_bytes(&remote);
        if remote.file.is_none() || (total_bytes > 0 && downloaded_bytes >= total_bytes) {
            return;
        }
        if remote.gated && std::env::var_os("HF_TOKEN").is_none() {
            self.message = Some(Message {
                title: "Cannot download".into(),
                lines: vec!["This gated model requires HF_TOKEN.".into()],
            });
            return;
        }
        if remote.blobs.is_empty() || total_bytes == 0 {
            self.message = Some(Message {
                title: "Cannot download".into(),
                lines: vec![
                    "Download metadata is unavailable; press F5 in this repository and retry."
                        .into(),
                ],
            });
            return;
        }

        if let Some(index) =
            self.model_downloads.iter().position(|download| download.model_id == model.id)
        {
            if matches!(
                self.model_downloads[index].status,
                ModelDownloadStatus::Cancelled
                    | ModelDownloadStatus::Interrupted
                    | ModelDownloadStatus::Failed(_)
            ) {
                self.resume_model_download(index);
            }
            self.screen = Screen::Sessions;
            self.session_sel.select(Some(self.sessions.sessions.len() + index));
            return;
        }

        let id = self.next_download_id();
        let cancelled = Arc::new(AtomicBool::new(false));
        let display =
            format!("{}/{}", remote.repo, remote.file.as_deref().unwrap_or(model.name.as_str()));
        let record = discovery::online::DownloadJobRecord::new(
            model.id.clone(),
            display.clone(),
            remote.clone(),
        );
        if let Err(error) = discovery::online::save_download_record(&self.models_dir, &record) {
            self.message = Some(Message {
                title: "Cannot download".into(),
                lines: vec![format!("Cannot persist download state: {error}")],
            });
            return;
        }
        self.model_downloads.push(ModelDownload {
            id,
            model_id: model.id,
            model: display,
            downloaded_bytes: 0,
            total_bytes,
            status: ModelDownloadStatus::Downloading,
            source: DownloadSource::Hub(Box::new(remote.clone())),
            cancelled: cancelled.clone(),
        });
        self.reveal_latest_download();
        self.spawn_model_download(id, DownloadSource::Hub(Box::new(remote)), cancelled);
    }

    /// `flm pull <tag>`. There is no persisted resume record: `flm` keeps its
    /// own partial state and a re-run picks up where it left off, so an
    /// interrupted pull is simply started again.
    fn download_flm_model(&mut self, model: &Model) {
        let Some(flm) = model.flm.as_ref() else { return };
        if flm.installed {
            return;
        }
        // Already tracked: jump to it rather than starting a second transfer.
        if let Some(index) =
            self.model_downloads.iter().position(|download| download.model_id == model.id)
        {
            let restartable = matches!(
                self.model_downloads[index].status,
                ModelDownloadStatus::Cancelled
                    | ModelDownloadStatus::Interrupted
                    | ModelDownloadStatus::Failed(_)
            );
            if restartable {
                let id = self.next_download_id();
                let cancelled = Arc::new(AtomicBool::new(false));
                let source = self.model_downloads[index].source.clone();
                let download = &mut self.model_downloads[index];
                download.id = id;
                download.status = ModelDownloadStatus::Downloading;
                download.cancelled = cancelled.clone();
                self.spawn_model_download(id, source, cancelled);
            }
            self.screen = Screen::Sessions;
            self.session_sel.select(Some(self.sessions.sessions.len() + index));
            return;
        }

        let id = self.next_download_id();
        let cancelled = Arc::new(AtomicBool::new(false));
        let source = DownloadSource::Flm(Box::new(model.clone()));
        self.model_downloads.push(ModelDownload {
            id,
            model_id: model.id.clone(),
            model: flm.tag.clone(),
            downloaded_bytes: 0,
            total_bytes: model.size_bytes,
            status: ModelDownloadStatus::Downloading,
            source: source.clone(),
            cancelled: cancelled.clone(),
        });
        self.reveal_latest_download();
        self.spawn_model_download(id, source, cancelled);
    }

    /// Switch to the Session Manager with the newest download selected.
    fn reveal_latest_download(&mut self) {
        self.screen = Screen::Sessions;
        self.session_sel
            .select(Some(self.sessions.sessions.len() + self.model_downloads.len() - 1));
    }

    fn next_download_id(&mut self) -> u64 {
        let id = self.next_download_id;
        self.next_download_id = self.next_download_id.wrapping_add(1).max(1);
        id
    }

    fn spawn_model_download(&self, id: u64, source: DownloadSource, cancelled: Arc<AtomicBool>) {
        let tx = self.download_tx.clone();
        std::thread::spawn(move || {
            let progress = |downloaded_bytes, total_bytes| {
                let _ = tx.send(ModelDownloadEvent::Progress { id, downloaded_bytes, total_bytes });
            };
            let result = match source {
                DownloadSource::Hub(remote) => {
                    discovery::online::download_model(&remote, &cancelled, progress)
                        .map_err(|error| error.to_string())
                }
                DownloadSource::Flm(model) => {
                    crate::runtime::flm::download(&model, &cancelled, progress).map(|outcome| {
                        match outcome {
                            crate::runtime::flm::DownloadOutcome::Downloaded(path) => {
                                discovery::online::DownloadResult::Downloaded(path)
                            }
                            crate::runtime::flm::DownloadOutcome::Cancelled => {
                                discovery::online::DownloadResult::Cancelled
                            }
                        }
                    })
                }
            };
            let _ = tx.send(ModelDownloadEvent::Finished { id, result });
        });
    }

    fn poll_model_download(&mut self) {
        let mut refresh_models = false;
        let mut refresh_flm = false;
        let mut completed_records = Vec::new();
        while let Ok(event) = self.download_rx.try_recv() {
            match event {
                ModelDownloadEvent::Progress { id, downloaded_bytes, total_bytes } => {
                    let Some(download) =
                        self.model_downloads.iter_mut().find(|download| download.id == id)
                    else {
                        continue;
                    };
                    if !matches!(download.status, ModelDownloadStatus::Downloading) {
                        continue;
                    }
                    download.downloaded_bytes = downloaded_bytes.min(total_bytes);
                    download.total_bytes = total_bytes;
                }
                ModelDownloadEvent::Finished { id, result } => {
                    let Some(download) =
                        self.model_downloads.iter_mut().find(|download| download.id == id)
                    else {
                        continue;
                    };
                    match result {
                        Ok(discovery::online::DownloadResult::Downloaded(path)) => {
                            download.downloaded_bytes = download.total_bytes;
                            download.status = ModelDownloadStatus::Downloaded(path);
                            match download.source {
                                DownloadSource::Flm { .. } => refresh_flm = true,
                                DownloadSource::Hub(_) => {
                                    completed_records.push(download.model_id.clone());
                                    refresh_models = true;
                                }
                            }
                        }
                        Ok(discovery::online::DownloadResult::Cancelled) => {
                            download.status = ModelDownloadStatus::Cancelled;
                        }
                        Err(_) if download.cancelled.load(Ordering::Relaxed) => {
                            download.status = ModelDownloadStatus::Cancelled;
                        }
                        Err(error) => download.status = ModelDownloadStatus::Failed(error),
                    }
                }
            }
        }
        for model_id in completed_records {
            if let Err(error) =
                discovery::online::delete_download_record(&self.models_dir, &model_id)
            {
                tracing::warn!(%error, model = %model_id, "failed to remove completed download record");
            }
        }
        if refresh_flm {
            // A download just finished: `flm` now reports the model installed,
            // which is exactly the change the cached catalog cannot know about.
            self.refresh_flm_models(true);
            self.reselect_current_catalog();
        }
        if refresh_models {
            self.refresh_downloaded_online_models();
        }
    }

    /// Rebuild the Model pane in place, keeping the cursor on the same entry.
    fn reselect_current_catalog(&mut self) {
        let selected = self.models.selected().map(|model| model.id.clone());
        let items = self.catalog_children(&self.catalog_prefix);
        let index = selected
            .as_ref()
            .and_then(|id| items.iter().position(|model| &model.id == id))
            .unwrap_or(0);
        self.models.items = items;
        self.models.state.select((!self.models.items.is_empty()).then_some(index));
        self.rebuild_below(Pane::Model);
    }

    fn refresh_downloaded_online_models(&mut self) {
        let selected = self.models.selected().map(|model| model.id.clone());
        self.scanned_models.retain(|model| !discovery::online::is_online_path(&model.catalog_path));
        self.scanned_models.extend(discovery::online::load_cached(&self.models_dir));
        self.store.sync_models(&self.scanned_models);
        if self.runtimes.selected().is_some_and(|backend| backend.supports_online_browse()) {
            let items = self.catalog_children(&self.catalog_prefix);
            let index = selected
                .as_ref()
                .and_then(|id| items.iter().position(|model| &model.id == id))
                .unwrap_or(0);
            self.models.items = items;
            self.models.state.select((!self.models.items.is_empty()).then_some(index));
            self.rebuild_below(Pane::Model);
        }
    }

    fn resume_model_download(&mut self, index: usize) {
        let Some(download) = self.model_downloads.get(index) else { return };
        if matches!(
            download.status,
            ModelDownloadStatus::Downloading
                | ModelDownloadStatus::Cancelling
                | ModelDownloadStatus::Downloaded(_)
        ) {
            return;
        }
        let id = self.next_download_id();
        let cancelled = Arc::new(AtomicBool::new(false));
        let source = self.model_downloads[index].source.clone();
        // Only Hub downloads persist a resume record; `flm` tracks its own.
        if let DownloadSource::Hub(remote) = &source {
            let record = discovery::online::DownloadJobRecord::new(
                self.model_downloads[index].model_id.clone(),
                self.model_downloads[index].model.clone(),
                (**remote).clone(),
            );
            if let Err(error) = discovery::online::save_download_record(&self.models_dir, &record) {
                self.message = Some(Message {
                    title: "Cannot resume".into(),
                    lines: vec![format!("Cannot persist download state: {error}")],
                });
                return;
            }
        }
        let download = &mut self.model_downloads[index];
        download.id = id;
        download.status = ModelDownloadStatus::Downloading;
        download.cancelled = cancelled.clone();
        self.spawn_model_download(id, source, cancelled);
    }

    /// Run the runtime's interactive client in the foreground (`C`), suspending
    /// the TUI while it owns the terminal. Server-only flags are dropped by the
    /// backend.
    fn start_chat(&mut self) {
        if self.focus != Pane::Profile && self.focus != Pane::Options {
            return;
        }
        let Some(backend) = self.runtimes.selected() else { return };
        if let Some(reason) = backend.unavailable_reason() {
            self.message = Some(Message { title: "Cannot start chat".into(), lines: vec![reason] });
            return;
        }
        // An interactive client loads the model too, so it collides with a live
        // server on a single-session runtime exactly as a second launch would.
        let runtime = backend.descriptor().name.clone();
        if let Some(lines) = self.single_session_conflict(&runtime) {
            self.message = Some(Message { title: "Cannot start chat".into(), lines });
            return;
        }
        let (Some(model), Some(binary)) =
            (self.selected_model(), backend.descriptor().binary_path.as_ref())
        else {
            return;
        };
        let binary = binary.display().to_string();
        let ctx = LaunchContext { binary: &binary, model, options: &self.options.items };
        match backend.chat_argv(&ctx) {
            Some(argv) => self.pending_chat = Some(argv),
            None => {
                self.message = Some(Message {
                    title: "Chat unavailable".into(),
                    lines: vec![format!("{runtime} has no interactive client on this system.")],
                });
            }
        }
    }

    /// Run the runtime's benchmark tool in the foreground (`b`). Runtimes
    /// without one leave `pending_benchmark` unset, so the key is inert.
    fn start_benchmark(&mut self) {
        let (Some(backend), Some(model)) = (self.runtimes.selected(), self.selected_model()) else {
            return;
        };
        // A benchmark loads the model just as a server or an interactive client
        // does, so it collides with a live session on a single-session runtime.
        // Only reachable since FastFlowLM gained a benchmark: `llama-bench`
        // belongs to a runtime that is happy to run several at once.
        let runtime = backend.descriptor().name.clone();
        if let Some(lines) = self.single_session_conflict(&runtime) {
            self.message = Some(Message { title: "Cannot start benchmark".into(), lines });
            return;
        }
        let Some(binary) = backend.descriptor().binary_path.as_ref() else { return };
        let binary = binary.display().to_string();
        let ctx = LaunchContext { binary: &binary, model, options: &self.options.items };
        self.pending_benchmark = backend.bench_argv(&ctx);
    }

    pub fn async_job_count(&self) -> usize {
        self.sessions.sessions.len() + self.model_downloads.len()
    }

    fn selected_server_index(&self) -> Option<usize> {
        self.session_sel.selected().filter(|index| *index < self.sessions.sessions.len())
    }

    fn selected_download_index(&self) -> Option<usize> {
        self.session_sel
            .selected()?
            .checked_sub(self.sessions.sessions.len())
            .filter(|index| *index < self.model_downloads.len())
    }

    pub fn selected_server_session(&self) -> Option<&session::Session> {
        self.selected_server_index().and_then(|index| self.sessions.sessions.get(index))
    }

    pub fn selected_model_download(&self) -> Option<&ModelDownload> {
        self.selected_download_index().and_then(|index| self.model_downloads.get(index))
    }

    /// Apply a fallible supervisor action to the selected server session.
    fn session_action(&mut self, f: impl Fn(&mut SessionManager, usize) -> Result<()>, verb: &str) {
        let Some(i) = self.selected_server_index() else { return };
        if let Err(e) = f(&mut self.sessions, i) {
            self.message =
                Some(Message { title: format!("Failed to {verb}"), lines: vec![e.to_string()] });
        }
    }

    fn stop_async_job(&mut self, force: bool) {
        if let Some(index) = self.selected_download_index() {
            let download = &mut self.model_downloads[index];
            if matches!(download.status, ModelDownloadStatus::Downloading) {
                download.cancelled.store(true, Ordering::Relaxed);
                download.status = ModelDownloadStatus::Cancelling;
            }
            return;
        }
        if force {
            self.session_action(|manager, index| manager.kill(index), "kill");
        } else {
            self.session_action(|manager, index| manager.stop(index), "stop");
        }
    }

    fn restart_async_job(&mut self) {
        if let Some(index) = self.selected_download_index() {
            self.resume_model_download(index);
        } else {
            self.session_action(|manager, index| manager.restart(index), "restart");
        }
    }

    /// Remove a terminated server or inactive download record (`d`).
    fn remove_async_job(&mut self) {
        if let Some(index) = self.selected_download_index() {
            if matches!(
                self.model_downloads[index].status,
                ModelDownloadStatus::Downloading | ModelDownloadStatus::Cancelling
            ) {
                self.message = Some(Message {
                    title: "Cannot remove".into(),
                    lines: vec!["Cancel the download before removing it.".into()],
                });
                return;
            }
            let model_id = self.model_downloads[index].model_id.clone();
            if let Err(error) =
                discovery::online::delete_download_record(&self.models_dir, &model_id)
            {
                self.message = Some(Message {
                    title: "Cannot remove".into(),
                    lines: vec![format!("Cannot remove persisted download state: {error}")],
                });
                return;
            }
            self.model_downloads.remove(index);
            self.sync_session_selection();
            return;
        }
        let Some(i) = self.selected_server_index() else { return };
        if self.sessions.remove(i) {
            self.sync_session_selection();
        } else {
            self.message = Some(Message {
                title: "Cannot remove".into(),
                lines: vec![
                    "Only Stopped or Crashed sessions can be removed; stop it first.".into(),
                ],
            });
        }
    }

    /// Copy the selected session's endpoint URL to the clipboard (`c`).
    fn copy_endpoint(&mut self) {
        let Some(i) = self.selected_server_index() else { return };
        let endpoint = self.sessions.sessions[i].record.endpoint();
        copy_to_clipboard(&endpoint);
        self.message = Some(Message { title: "Endpoint copied".into(), lines: vec![endpoint] });
    }

    /// Show + copy the selected session's stored launch command (`y`).
    fn yank_session_command(&mut self) {
        let Some(i) = self.selected_server_index() else { return };
        let argv = self.sessions.sessions[i].record.command.clone();
        let cmd = session::command::Command { argv };
        copy_to_clipboard(&cmd.display());
        self.message =
            Some(Message { title: "Session command".into(), lines: command_message_lines(&cmd) });
    }

    /// Open the log-tail screen for the selected session (`L`).
    fn open_logs(&mut self) {
        if self.selected_server_index().is_none() {
            return;
        }
        self.screen = Screen::Logs;
        self.log_follow = true;
        self.log_scroll = 0;
        self.reload_logs();
    }

    /// Reload the tail of the selected session's log file.
    fn reload_logs(&mut self) {
        let lines = self
            .session_sel
            .selected()
            .and_then(|i| self.sessions.sessions.get(i))
            .map(|s| read_log_tail(&s.record.log_file, 1000))
            .unwrap_or_default();
        self.log_lines = lines;
    }

    /// Handle a keystroke while a modal text prompt is open.
    fn prompt_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => self.prompt = None,
            KeyCode::Enter => self.commit_prompt(),
            KeyCode::Backspace => {
                if let Some(p) = self.prompt.as_mut() {
                    p.buffer.pop();
                    p.error = None;
                }
            }
            KeyCode::Char(c) => {
                if let Some(p) = self.prompt.as_mut() {
                    p.buffer.push(c);
                    p.error = None;
                }
            }
            _ => {}
        }
    }

    /// Handle a keystroke while the enum-variant selector is open: printable
    /// keys narrow the filter, arrows/Home/End move, Enter picks, Esc cancels.
    fn selector_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => self.selector = None,
            KeyCode::Enter => {
                if let Some(sel) = self.selector.take() {
                    if let Some(value) = sel.selected().map(str::to_string) {
                        self.apply_option_value(&sel.key, value);
                    }
                }
            }
            _ => {
                let Some(sel) = self.selector.as_mut() else {
                    return;
                };
                match key.code {
                    KeyCode::Up => sel.cursor = sel.cursor.saturating_sub(1),
                    KeyCode::Down => {
                        sel.cursor = (sel.cursor + 1).min(sel.filtered().len().saturating_sub(1))
                    }
                    KeyCode::Home => sel.cursor = 0,
                    KeyCode::End => sel.cursor = sel.filtered().len().saturating_sub(1),
                    KeyCode::Backspace => {
                        sel.filter.pop();
                        sel.cursor = 0;
                    }
                    KeyCode::Char(c) => {
                        sel.filter.push(c);
                        sel.cursor = 0;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Open the option editor. Small enums cycle in place, large ones
    /// ([`SELECTOR_THRESHOLD`]) open the filterable selector popup; numeric/
    /// string open a text prompt. Applies only to real (non-stub) runtimes.
    fn open_editor(&mut self) {
        if self.focus != Pane::Options {
            return;
        }
        let Some(option) = self.options.selected() else {
            return;
        };
        let key = option.key.clone();
        let current = option.value.clone();

        if key == "device" {
            let mut variants = vec![profiles::registry::DEFAULT.to_string()];
            if let Some(runtime) = self.runtimes.selected() {
                variants.extend(runtime.descriptor().devices.iter().cloned());
            }
            self.selector = Some(Selector {
                title: "Select device".into(),
                key,
                cursor: variants.iter().position(|v| *v == current).unwrap_or(0),
                variants,
                filter: String::new(),
            });
            return;
        }

        let Some(backend) = self.runtimes.selected() else { return };
        if let Some(spec) = backend.schema().spec(&key) {
            use profiles::registry::OptionKind;
            if let OptionKind::Enum(variants) = spec.kind {
                if variants.len() > SELECTOR_THRESHOLD {
                    self.selector = Some(Selector {
                        title: format!("Select {key}"),
                        key,
                        variants: variants.iter().map(|v| (*v).to_string()).collect(),
                        filter: String::new(),
                        // Start on the current value.
                        cursor: variants.iter().position(|v| *v == current).unwrap_or(0),
                    });
                    return;
                }
                // Small enums don't need a popup — `e` advances to the next
                // state (which, for omittable options, cycles "default" too).
                if let Some(next) = backend.schema().bump(spec, &spec.kind, &current, 1) {
                    self.apply_option_value(&key, next);
                }
                return;
            }
        }
        let title = if backend.schema().uses_sentinel(&key) {
            format!("Edit {key} (number or 'default')")
        } else {
            format!("Edit {key}")
        };
        self.prompt = Some(Prompt {
            kind: PromptKind::EditOption { key: key.clone() },
            title,
            buffer: current,
            error: None,
        });
    }

    /// Reset the selected option to its resolved default (`d` in Options).
    /// Unlike `Home`, this restores the *resolved* default — the omit token for
    /// omittable options, but e.g. ctx/8 for ctx-size or the config host/port.
    fn reset_option_default(&mut self) {
        if self.focus != Pane::Options {
            return;
        }
        let Some(option) = self.options.selected() else {
            return;
        };
        let key = option.key.clone();
        let default = option.default.clone();
        self.apply_option_value(&key, default);
    }

    /// Increment/decrement the selected option in place (auto-saves).
    fn adjust_option(&mut self, dir: i32) {
        if let Some(option) = self.options.selected() {
            if option.key == "device" {
                let next = self
                    .runtimes
                    .selected()
                    .map(|runtime| cycle_device(&runtime.descriptor().devices, &option.value, dir));
                if let Some(next) = next {
                    self.apply_option_value("device", next);
                }
                return;
            }
        }
        let schema = self.runtimes.selected().map(|b| b.schema());
        self.transform_option(move |spec, kind, current| {
            schema.and_then(|schema| schema.bump(spec, kind, current, dir))
        });
    }

    /// Set the selected option to its min (`dir < 0`) or max (`dir > 0`).
    fn set_option_extreme(&mut self, dir: i32) {
        self.transform_option(|_spec, kind, _current| kind.extreme(dir));
    }

    /// Shared helper: compute a new value for the selected option and apply it.
    fn transform_option(
        &mut self,
        f: impl Fn(
            &profiles::registry::OptionSpec,
            &profiles::registry::OptionKind,
            &str,
        ) -> Option<String>,
    ) {
        if self.focus != Pane::Options {
            return;
        }
        let Some(option) = self.options.selected() else {
            return;
        };
        let key = option.key.clone();
        let current = option.value.clone();
        let Some(backend) = self.runtimes.selected() else { return };
        let Some(spec) = backend.schema().spec(&key) else {
            return;
        };
        // Use the model-aware kind so ctx-size respects the model's max context.
        let kind = match self.selected_model() {
            Some(m) => backend.effective_kind(spec, m),
            None => spec.kind,
        };
        if let Some(value) = f(spec, &kind, &current) {
            self.apply_option_value(&key, value);
        }
    }

    /// Validate and commit the open prompt; dispatch by its kind.
    fn commit_prompt(&mut self) {
        let Some(prompt) = self.prompt.as_ref() else {
            return;
        };
        let input = prompt.buffer.trim().to_string();
        let kind = prompt.kind.clone(); // release the borrow before dispatching
        let result = match kind {
            PromptKind::EditOption { key } => self.commit_option_edit(&key, &input),
            PromptKind::NewProfile => self.commit_new_profile(&input),
            PromptKind::RenameProfile { old } => self.commit_rename_profile(&old, &input),
            PromptKind::DuplicateProfile { src } => self.commit_duplicate_profile(&src, &input),
        };
        match result {
            Ok(()) => self.prompt = None,
            Err(message) => {
                if let Some(p) = self.prompt.as_mut() {
                    p.error = Some(message);
                }
            }
        }
    }

    fn commit_option_edit(&mut self, key: &str, input: &str) -> Result<(), String> {
        let backend = self.runtimes.selected().ok_or("no runtime selected")?;
        let spec = backend.schema().spec(key).ok_or("unknown option")?;
        // Sentinel options accept "default" (or an empty entry) to drop the flag.
        if backend.schema().uses_sentinel(key)
            && (input.is_empty() || input.eq_ignore_ascii_case(profiles::registry::DEFAULT))
        {
            self.apply_option_value(key, profiles::registry::DEFAULT.to_string());
            return Ok(());
        }
        let kind = match self.selected_model() {
            Some(m) => backend.effective_kind(spec, m),
            None => spec.kind,
        };
        let value = kind.validate(input)?;
        self.apply_option_value(key, value);
        Ok(())
    }

    /// Persist an option value to the model-scoped instance (auto-saves) and
    /// refresh the Options pane while preserving the cursor position.
    fn apply_option_value(&mut self, key: &str, value: String) {
        let (Some(backend), Some(m), Some(p)) =
            (self.runtimes.selected(), self.selected_model(), self.profiles.selected())
        else {
            return;
        };
        let runtime = backend.descriptor().name.clone();
        let model = m.profile_key();
        let profile = p.clone();
        let base = profiles::resolved_values(backend.as_ref(), &profile, m, &self.config.defaults);

        let cursor = self.options.state.selected();
        self.store.set_value(&runtime, &model, &profile.name, key, value, &base);
        self.rebuild_below(Pane::Profile);
        self.options.state.select(cursor);
    }

    /// Toggle the favorite flag on the selected profile (real runtimes only).
    fn toggle_favorite(&mut self) {
        if self.focus != Pane::Profile {
            return;
        }
        let (Some(backend), Some(m), Some(p)) =
            (self.runtimes.selected(), self.selected_model(), self.profiles.selected())
        else {
            return;
        };
        let runtime = backend.descriptor().name.clone();
        let model = m.profile_key();
        let profile = p.clone();
        let base = profiles::resolved_values(backend.as_ref(), &profile, m, &self.config.defaults);

        let cursor = self.profiles.state.selected();
        self.store.toggle_favorite(&runtime, &model, &profile.name, &base);
        self.rebuild_below(Pane::Model);
        self.profiles.state.select(cursor);
    }

    // --- profile management (Profile pane) ---------------------------------

    fn prompt_new_profile(&mut self) {
        if self.focus != Pane::Profile {
            return;
        }
        self.prompt = Some(Prompt {
            kind: PromptKind::NewProfile,
            title: "New profile".into(),
            buffer: String::new(),
            error: None,
        });
    }

    fn prompt_rename_profile(&mut self) {
        if self.focus != Pane::Profile {
            return;
        }
        let Some(p) = self.profiles.selected() else {
            return;
        };
        if p.builtin {
            return; // built-in templates are read-only
        }
        let old = p.name.clone();
        self.prompt = Some(Prompt {
            kind: PromptKind::RenameProfile { old: old.clone() },
            title: format!("Rename {old}"),
            buffer: old,
            error: None,
        });
    }

    fn prompt_duplicate_profile(&mut self) {
        if self.focus != Pane::Profile {
            return;
        }
        let Some(p) = self.profiles.selected() else {
            return;
        };
        let src = p.name.clone();
        self.prompt = Some(Prompt {
            kind: PromptKind::DuplicateProfile { src: src.clone() },
            title: format!("Duplicate {src}"),
            buffer: format!("{src} copy"),
            error: None,
        });
    }

    /// Delete a custom profile, or reset a built-in to its template defaults.
    fn delete_profile(&mut self) {
        if self.focus != Pane::Profile {
            return;
        }
        let (Some(backend), Some(m), Some(p)) =
            (self.runtimes.selected(), self.selected_model(), self.profiles.selected())
        else {
            return;
        };
        let runtime = backend.descriptor().name.clone();
        let model = m.profile_key();
        let name = p.name.clone();

        let cursor = self.profiles.state.selected().unwrap_or(0);
        self.store.delete(&runtime, &model, &name);
        self.rebuild_below(Pane::Model);
        let len = self.profiles.items.len();
        if len > 0 {
            self.profiles.state.select(Some(cursor.min(len - 1)));
            self.rebuild_below(Pane::Profile);
        }
    }

    fn commit_new_profile(&mut self, name: &str) -> Result<(), String> {
        self.validate_new_name(name)?;
        let (runtime, model) = self.current_runtime_model().ok_or("no model selected")?;
        let backend = self.runtimes.selected().ok_or("no runtime selected")?;
        let m = self.selected_model().ok_or("no model selected")?;
        // Seed from the Default template's resolved values for this model.
        let default = Profile { name: "Default".into(), builtin: true, favorite: false };
        let values =
            profiles::resolved_values(backend.as_ref(), &default, m, &self.config.defaults);
        self.store.create(&runtime, &model, name, values, true);
        self.refresh_profiles(Some(name));
        Ok(())
    }

    fn commit_rename_profile(&mut self, old: &str, name: &str) -> Result<(), String> {
        if name.eq_ignore_ascii_case(old) {
            return Ok(()); // no change
        }
        self.validate_new_name(name)?;
        let (runtime, model) = self.current_runtime_model().ok_or("no model selected")?;
        self.store.rename(&runtime, &model, old, name);
        self.refresh_profiles(Some(name));
        Ok(())
    }

    fn commit_duplicate_profile(&mut self, src: &str, name: &str) -> Result<(), String> {
        self.validate_new_name(name)?;
        let (Some(backend), Some(m)) = (self.runtimes.selected(), self.selected_model()) else {
            return Err("no model selected".into());
        };
        let runtime = backend.descriptor().name.clone();
        let model = m.profile_key();
        let src_profile = Profile {
            name: src.to_string(),
            builtin: profiles::templates::is_builtin(backend.templates(), src),
            favorite: false,
        };
        // Copy the source's *current* values (including any instance edits).
        let values = profiles::current_values(
            backend.as_ref(),
            m,
            &src_profile,
            &self.store,
            &self.config.defaults,
        );
        self.store.create(&runtime, &model, name, values, true);
        self.refresh_profiles(Some(name));
        Ok(())
    }

    fn validate_new_name(&self, name: &str) -> Result<(), String> {
        if name.is_empty() {
            return Err("name cannot be empty".into());
        }
        if self.profiles.items.iter().any(|p| p.name.eq_ignore_ascii_case(name)) {
            return Err(format!("'{name}' already exists"));
        }
        Ok(())
    }

    fn current_runtime_model(&self) -> Option<(String, String)> {
        let backend = self.runtimes.selected()?;
        let m = self.selected_model()?;
        Some((backend.descriptor().name.clone(), m.profile_key()))
    }

    /// Rebuild the profile list, then optionally select a profile by name and
    /// refresh its options.
    fn refresh_profiles(&mut self, select: Option<&str>) {
        self.rebuild_below(Pane::Model);
        if let Some(name) = select {
            if let Some(i) = self.profiles.items.iter().position(|p| p.name == name) {
                self.profiles.state.select(Some(i));
                self.rebuild_below(Pane::Profile);
            }
        }
    }

    /// The selected catalog leaf. Directory nodes intentionally have no path.
    pub fn selected_model(&self) -> Option<&Model> {
        self.models.selected().filter(|m| m.is_model())
    }

    /// Whether `d` can start a download for the selection: an online GGUF that
    /// is not cached yet, or a FastFlowLM catalog entry that is not installed.
    pub fn download_available(&self) -> bool {
        self.focus == Pane::Model
            && self.selected_model().is_some_and(|model| match &model.flm {
                Some(flm) => !flm.installed,
                None => {
                    model.path.as_os_str().is_empty()
                        && model.remote.as_ref().is_some_and(|remote| remote.file.is_some())
                }
            })
    }

    /// Whether the selected runtime ships a benchmark tool for this model.
    /// The model must be present locally: llama.cpp needs a GGUF to point
    /// `llama-bench` at, and `flm bench` on an un-pulled tag would turn a
    /// benchmark into a multi-gigabyte download.
    pub fn benchmark_available(&self) -> bool {
        self.selected_model().is_some_and(|model| !model.path.as_os_str().is_empty())
            && self
                .runtimes
                .selected()
                .is_some_and(|backend| backend.descriptor().bench_path.is_some())
    }

    pub fn catalog_parent(&self) -> Option<(&[Model], Option<usize>)> {
        self.catalog_history.last().map(|(items, selected, _)| (items.as_slice(), *selected))
    }

    /// The flat model list backing the browser tree for the selected runtime.
    fn catalog_source(&self) -> &[Model] {
        // The online (Hugging Face) subtree and its blob downloads only exist
        // for llama.cpp, so the two catalogs are stored separately rather than
        // merged behind one list.
        match self.runtimes.selected() {
            Some(backend) if !backend.supports_online_browse() => &self.flm_models,
            _ => &self.scanned_models,
        }
    }

    fn catalog_children(&self, prefix: &[String]) -> Vec<Model> {
        let source = self.catalog_source();
        if let Some(repositories) = online_repository_children(source, prefix) {
            return repositories;
        }
        if let Some(artifacts) = online_artifact_children(source, prefix) {
            return artifacts;
        }
        catalog_children_of(source, prefix)
    }

    /// Rebuild the FastFlowLM model list for the current arrangement.
    ///
    /// `reload` decides whether `flm list` runs again or the backend serves the
    /// catalog it already has. Pass it when the catalog itself may have changed
    /// — a manual refresh, or a download that just installed a model — and not
    /// when only the arrangement did.
    fn refresh_flm_models(&mut self, reload: bool) {
        let Some(backend) = self.runtimes.items.iter().find(|b| !b.supports_online_browse()) else {
            return;
        };
        let ctx = CatalogCtx {
            sources: &self.model_sources,
            cache_path: &self.model_cache,
            models_dir: &self.models_dir,
            view: self.catalog_view,
            reload,
        };
        self.flm_models = backend.models(&ctx);
        self.store.sync_models(&self.flm_models);
    }

    /// Re-scan configured model directories (the `F5` refresh).
    fn refresh_models(&mut self) {
        if self.online_view_active() {
            self.reload_online_layout();
            return;
        }
        // The user asked for fresh data, so go back to `flm` — this is how a
        // model installed outside llmctl shows up.
        self.refresh_flm_models(true);
        self.scanned_models = discovery::scan_models(&self.model_sources, &self.model_cache);
        discovery::reconcile(&self.models_dir, &mut self.scanned_models);
        self.scanned_models.extend(discovery::online::load_cached(&self.models_dir));
        self.store.sync_models(&self.scanned_models);
        self.catalog_history.clear();
        self.catalog_prefix.clear();
        // Models or anything downstream may have changed; rebuild from runtime.
        self.rebuild_below(Pane::Runtime);
    }

    /// Drill into the preview pane, but only if it actually has items.
    fn enter(&mut self) {
        if self.focus == Pane::Model {
            let Some(selected) = self.models.selected() else { return };
            if selected.is_catalog_dir() {
                if self.catalog_preview.is_empty() {
                    return;
                }
                let previous = (
                    self.models.items.clone(),
                    self.models.state.selected(),
                    self.catalog_prefix.clone(),
                );
                self.catalog_history.push(previous);
                self.catalog_prefix = selected.catalog_path.clone();
                self.models.replace(self.catalog_preview.clone());
                self.rebuild_below(Pane::Model);
                self.maybe_fetch_online(false);
            } else if !self.profiles.is_empty() {
                self.focus = Pane::Profile;
            }
        } else if self.focus != Pane::Options && !self.preview_is_empty() {
            self.focus = self.focus.next();
            if self.focus == Pane::Model {
                self.maybe_fetch_online(false);
            }
        }
    }

    fn go_back(&mut self) {
        if self.focus == Pane::Model {
            if let Some((items, selected, prefix)) = self.catalog_history.pop() {
                self.catalog_prefix = prefix;
                self.models.items = items;
                self.models.state.select(selected);
                self.rebuild_below(Pane::Model);
            } else {
                self.focus = Pane::Runtime;
            }
        } else {
            self.focus = self.focus.prev();
        }
    }

    /// Is the pane immediately right of focus (the preview) empty?
    fn preview_is_empty(&self) -> bool {
        match self.focus {
            Pane::Runtime => self.models.is_empty(),
            Pane::Model => {
                if self.selected_model().is_some() {
                    self.profiles.is_empty()
                } else {
                    self.catalog_preview.is_empty()
                }
            }
            Pane::Profile => self.options.is_empty(),
            Pane::Options => true,
        }
    }

    fn move_selection(&mut self, delta: isize) {
        match self.focus {
            Pane::Runtime => self.runtimes.move_by(delta),
            Pane::Model => self.models.move_by(delta),
            Pane::Profile => self.profiles.move_by(delta),
            Pane::Options => self.options.move_by(delta),
        }
        self.rebuild_below(self.focus);
        if self.focus == Pane::Model {
            self.maybe_fetch_online(false);
        }
    }

    fn select_first(&mut self) {
        match self.focus {
            Pane::Runtime => self.runtimes.select_first(),
            Pane::Model => self.models.select_first(),
            Pane::Profile => self.profiles.select_first(),
            Pane::Options => self.options.select_first(),
        }
        self.rebuild_below(self.focus);
        if self.focus == Pane::Model {
            self.maybe_fetch_online(false);
        }
    }

    fn select_last(&mut self) {
        match self.focus {
            Pane::Runtime => self.runtimes.select_last(),
            Pane::Model => self.models.select_last(),
            Pane::Profile => self.profiles.select_last(),
            Pane::Options => self.options.select_last(),
        }
        self.rebuild_below(self.focus);
        if self.focus == Pane::Model {
            self.maybe_fetch_online(false);
        }
    }

    /// Rebuild every pane below `changed` from the current selection chain,
    /// cascading top-down so each child sees its freshly-reset parent.
    fn rebuild_below(&mut self, changed: Pane) {
        let level = changed.index();
        if level < Pane::Model.index() {
            self.catalog_history.clear();
            self.catalog_prefix.clear();
            let models = if self.runtimes.selected().is_some() {
                self.catalog_children(&[])
            } else {
                vec![]
            };
            self.models.replace(models);
        }
        if level < Pane::Profile.index() {
            self.catalog_preview = match self.models.selected() {
                Some(m) if m.is_catalog_dir() => self.catalog_children(&m.catalog_path),
                _ => Vec::new(),
            };
            let profiles = match (self.runtimes.selected(), self.selected_model()) {
                (Some(backend), Some(m)) => {
                    profiles::list_profiles(backend.as_ref(), m, &self.store)
                }
                _ => Vec::new(),
            };
            self.profiles.replace(profiles);
        }
        if level < Pane::Options.index() {
            let options =
                match (self.runtimes.selected(), self.selected_model(), self.profiles.selected()) {
                    (Some(backend), Some(m), Some(p)) => profiles::resolve_options(
                        backend.as_ref(),
                        m,
                        p,
                        &self.store,
                        &self.config.defaults,
                    ),
                    _ => Vec::new(),
                };
            self.options.replace(options);
        }
    }

    /// Two-line status bar content for the hovered item: a primary locator
    /// (line 1 — a path) and a secondary metadata summary (line 2).
    pub fn status(&self) -> (String, String) {
        match self.focus {
            Pane::Runtime => self.runtimes.selected().map(|backend| {
                let runtime = backend.descriptor();
                let primary = runtime
                    .binary_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(binary not found)".into());
                let mut meta = Vec::new();
                if let Some(v) = &runtime.version {
                    meta.push(v.clone());
                }
                meta.push(runtime.formats_label());
                if !runtime.devices.is_empty() {
                    meta.push(runtime.devices.join(", "));
                }
                // Surface an unusable runtime here rather than at launch time.
                if let Some(reason) = backend.unavailable_reason() {
                    meta.push(reason);
                }
                (primary, meta.join(" · "))
            }),
            Pane::Model => self.models.selected().map(|m| {
                if let Some(remote) = &m.remote {
                    let primary = match &remote.file {
                        Some(file) => format!("hf://{}/{file}", remote.repo),
                        None => format!("hf://{}", remote.repo),
                    };
                    let mut meta = vec![format!("{} downloads", remote.downloads)];
                    meta.push(format!("{} likes", remote.likes));
                    if remote.gated {
                        meta.push("gated".into());
                    }
                    if remote.file.is_some() {
                        meta.push(if m.path.as_os_str().is_empty() {
                            "remote".into()
                        } else {
                            "cached".into()
                        });
                        meta.push(human_size(m.size_bytes));
                        if let Some(quantization) = &m.quantization {
                            meta.push(quantization.clone());
                        }
                        if remote.mtp_file.is_some() {
                            meta.push("MTP".into());
                        } else if m.has_mtp {
                            meta.push("MTP integrated".into());
                        }
                        if m.supports_multimodal() {
                            meta.push("multimodal".into());
                        }
                    }
                    return (primary, meta.join(" · "));
                }
                if m.is_catalog_dir() {
                    let metadata = if discovery::online::is_online_path(&m.catalog_path)
                        && self.online_pending.is_some()
                    {
                        "loading Hugging Face…"
                    } else if discovery::online::is_online_path(&m.catalog_path) {
                        "online catalog · F5 refresh"
                    } else {
                        "catalog directory"
                    };
                    return (m.catalog_path.join(" / "), metadata.into());
                }
                let primary = m.path.display().to_string();
                let mut meta = vec![human_size(m.size_bytes)];
                if let Some(q) = &m.quantization {
                    meta.push(q.clone());
                }
                if let Some(a) = &m.architecture {
                    meta.push(a.clone());
                }
                if let Some(ctx) = m.context_length {
                    meta.push(format!("ctx {ctx}"));
                }
                if m.has_chat_template {
                    meta.push("chat-template".into());
                }
                if let Some(mtp) = &m.mtp_path {
                    let name = mtp.file_name().unwrap_or_default().to_string_lossy();
                    meta.push(format!("MTP {name}"));
                } else if m.has_mtp {
                    meta.push("MTP integrated".into());
                }
                if let Some(projector) = &m.projector_path {
                    let name = projector.file_name().unwrap_or_default().to_string_lossy();
                    meta.push(format!("projector {name}"));
                }
                if let Some(secs) = m.modified {
                    meta.push(format_unix_date(secs));
                }
                (primary, meta.join(" · "))
            }),
            Pane::Profile => self.profiles.selected().map(|p| {
                let kind = if p.builtin { "built-in template" } else { "custom profile" };
                let fav = if p.favorite { " · ★" } else { "" };
                (p.name.clone(), format!("{kind}{fav}"))
            }),
            Pane::Options => self.options.selected().map(|o| {
                (o.key.clone(), format!("current {} · default {} · {}", o.value, o.default, o.cli))
            }),
        }
        .unwrap_or_default()
    }

    /// The committed path (Runtime ▸ Model ▸ …) up to and including focus,
    /// for the footer breadcrumb.
    pub fn breadcrumb(&self) -> Vec<String> {
        let mut crumbs = Vec::new();
        if let Some(r) = self.runtimes.selected() {
            crumbs.push(r.descriptor().name.clone());
        }
        if self.focus >= Pane::Model {
            crumbs.extend(self.catalog_prefix.iter().cloned());
            if let Some(m) = self.models.selected()
                && let Some(name) = m.catalog_path.last()
            {
                crumbs.push(name.clone());
            }
        }
        if self.focus >= Pane::Profile {
            if let Some(p) = self.profiles.selected() {
                crumbs.push(p.name.clone());
            }
        }
        if self.focus >= Pane::Options {
            if let Some(o) = self.options.selected() {
                crumbs.push(o.key.clone());
            }
        }
        crumbs
    }
}

/// Hub repository lists are already ranked by the requested API sort. Preserve
/// that order at the virtual repository root instead of applying the local
/// catalog's alphabetical directory ordering.
fn online_repository_children(models: &[Model], prefix: &[String]) -> Option<Vec<Model>> {
    if prefix != ["online", "huggingface"] {
        return None;
    }
    Some(
        models
            .iter()
            .filter(|model| {
                model.catalog_path.starts_with(prefix)
                    && model.catalog_path.len() == prefix.len() + 1
                    && model.remote.as_ref().is_some_and(|remote| remote.file.is_none())
            })
            .cloned()
            .collect(),
    )
}

/// Quantized files within a Hub repository are easiest to choose when ordered
/// from the smallest download to the largest. Unknown sizes sort last.
fn online_artifact_children(models: &[Model], prefix: &[String]) -> Option<Vec<Model>> {
    if prefix.len() != 3 || prefix[..2] != ["online", "huggingface"] {
        return None;
    }
    let mut artifacts: Vec<Model> = models
        .iter()
        .filter(|model| {
            model.catalog_path.starts_with(prefix)
                && model.catalog_path.len() == prefix.len() + 1
                && model.remote.as_ref().is_some_and(|remote| remote.file.is_some())
        })
        .cloned()
        .collect();
    artifacts.sort_by(|a, b| {
        (a.size_bytes == 0, a.size_bytes, a.name.to_ascii_lowercase()).cmp(&(
            b.size_bytes == 0,
            b.size_bytes,
            b.name.to_ascii_lowercase(),
        ))
    });
    Some(artifacts)
}

fn model_catalog_title(prefix: &[String], sort: discovery::online::Sort) -> String {
    if prefix.len() == 3 && prefix[..2] == ["online", "huggingface"] {
        return "Model".into();
    }
    if discovery::online::is_online_path(prefix) { sort.label().into() } else { "Model".into() }
}

/// Model search follows file-manager semantics: local queries recurse from the
/// directory currently being displayed (`catalog_prefix`), not from all model
/// sources. Hovering the virtual `online` source from the runtime root enters
/// the Hugging Face search scope; entering a flat repository row narrows the
/// scope to its cached artifacts.
///
/// `hub` says whether the selected runtime's `online` group really is the
/// Hugging Face browser. FastFlowLM uses the same group name for a fixed local
/// catalog, and expanding its scope to `online/huggingface` would search a
/// subtree it does not have.
fn normalized_search_scope(
    focus: Pane,
    selected_dir: Option<&[String]>,
    catalog_prefix: &[String],
    hub: bool,
) -> Vec<String> {
    if discovery::online::is_online_path(catalog_prefix) {
        if hub && catalog_prefix == ["online"] {
            return vec!["online".into(), "huggingface".into()];
        }
        return catalog_prefix.to_vec();
    }
    if hub && selected_dir.is_some_and(discovery::online::is_online_path) {
        return vec!["online".into(), "huggingface".into()];
    }
    match focus {
        Pane::Runtime => Vec::new(),
        Pane::Model | Pane::Profile | Pane::Options => catalog_prefix.to_vec(),
    }
}

/// Direct children of `prefix` in a flat model list: leaves are the models
/// themselves, and deeper paths collapse into one synthetic folder each.
fn catalog_children_of(source: &[Model], prefix: &[String]) -> Vec<Model> {
    use std::collections::BTreeMap;
    let mut children: BTreeMap<String, Model> = BTreeMap::new();
    for model in source {
        if !model.catalog_path.starts_with(prefix) || model.catalog_path.len() <= prefix.len() {
            continue;
        }
        let name = model.catalog_path[prefix.len()].clone();
        let is_leaf = model.catalog_path.len() == prefix.len() + 1;
        children.entry(name.clone()).or_insert_with(|| {
            if is_leaf {
                model.clone()
            } else {
                Model {
                    id: String::new(),
                    name,
                    path: PathBuf::new(),
                    shard_paths: Vec::new(),
                    mtp_path: None,
                    projector_path: None,
                    has_mtp: false,
                    catalog_path: model.catalog_path[..=prefix.len()].to_vec(),
                    catalog_dir: PathBuf::new(),
                    size_bytes: 0,
                    quantization: None,
                    architecture: None,
                    context_length: None,
                    modified: None,
                    has_chat_template: false,
                    remote: None,
                    flm: None,
                    runtime: model.runtime.clone(),
                }
            }
        });
    }
    children.into_values().collect()
}

/// Rank `models` against a search query, returning indices **into `models`**.
///
/// Taking the slice rather than reading it from `App` is deliberate: the
/// returned indices are only meaningful against the list they came from, and
/// llmctl has more than one (llama.cpp's scanned catalog and FastFlowLM's).
/// Ranking against one list and resolving against another panicked; a single
/// slice parameter makes that mismatch unrepresentable.
fn rank_models(
    models: &[Model],
    raw_query: &str,
    scope: &[String],
    online_only: bool,
) -> Vec<usize> {
    let query = raw_query.to_lowercase();
    let tokens: Vec<&str> = query.split_whitespace().collect();
    let mut matches: Vec<(i32, usize)> = models
        .iter()
        .enumerate()
        .filter_map(|(index, m)| {
            if !catalog_entry_in_search_scope(
                &m.catalog_path,
                m.remote.is_some(),
                scope,
                online_only,
            ) {
                return None;
            }
            let artifact = m.name.to_lowercase();
            let path = m.catalog_path.join(" ").to_lowercase();
            if !tokens.iter().all(|t| artifact.contains(t) || path.contains(t)) {
                return None;
            }
            let mut score = 0;
            if artifact == query || artifact.trim_end_matches(".gguf") == query {
                score += 1000;
            } else if artifact.starts_with(&query) {
                score += 500;
            }
            score += tokens.iter().filter(|t| artifact.contains(**t)).count() as i32 * 100;
            Some((score, index))
        })
        .collect();
    matches.sort_by(|(sa, a), (sb, b)| {
        sb.cmp(sa).then_with(|| models[*a].catalog_path.cmp(&models[*b].catalog_path))
    });
    matches.into_iter().map(|(_, index)| index).collect()
}

fn catalog_entry_in_search_scope(
    catalog_path: &[String],
    remote: bool,
    scope: &[String],
    online: bool,
) -> bool {
    remote == online && catalog_path.starts_with(scope) && catalog_path.len() > scope.len()
}

/// Resolve the directories to scan for models.
///
/// When `config.models.paths` is set we honor it (expanding `~`); otherwise we
/// fall back to the well-known runtime model locations. We never scan `$HOME`
/// itself, only specific subdirectories (per the requirements).
fn resolve_model_sources(configured: &[PathBuf], named: &[ModelSourceConfig]) -> Vec<ModelSource> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let expand = |p: &PathBuf| match (p.strip_prefix("~"), &home) {
        (Ok(rest), Some(home)) => home.join(rest),
        _ => p.clone(),
    };
    let mut sources: Vec<ModelSource> = if configured.is_empty() && named.is_empty() {
        default_model_sources(home.as_deref())
    } else {
        configured
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let root = expand(p);
                let name = root
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| format!("local-{}", i + 1));
                ModelSource { name, root, layout: ModelLayout::Auto }
            })
            .collect()
    };
    sources.extend(named.iter().map(|s| ModelSource {
        name: s.name.clone(),
        root: expand(&s.path),
        layout: s.layout,
    }));

    // De-duplicate roots (e.g. LLAMA_CACHE may equal ~/.cache/llama.cpp).
    sources.sort_by(|a, b| a.root.cmp(&b.root));
    sources.dedup_by(|a, b| a.root == b.root);
    sources
}

/// Well-known directories where local runtimes keep models, including
/// env-var-configured caches. Only existing dirs matter; the scanner skips the
/// rest.
fn default_model_sources(home: Option<&std::path::Path>) -> Vec<ModelSource> {
    use std::env::var_os;
    let mut dirs: Vec<ModelSource> = Vec::new();
    let source =
        |name: &str, root: PathBuf, layout| ModelSource { name: name.into(), root, layout };

    // llama.cpp download cache (LLAMA_CACHE overrides the default location).
    if let Some(cache) = var_os("LLAMA_CACHE") {
        dirs.push(source("llama-cache", PathBuf::from(cache), ModelLayout::Directory));
    } else if let Some(home) = home {
        dirs.push(source("llama-cache", home.join(".cache/llama.cpp"), ModelLayout::Directory));
    }

    // HuggingFace hub cache (used by `llama-server -hf` and others).
    if let Some(hf) = var_os("HUGGINGFACE_HUB_CACHE") {
        dirs.push(source("huggingface", PathBuf::from(hf), ModelLayout::HuggingFace));
    } else if let Some(hf) = var_os("HF_HOME") {
        dirs.push(source("huggingface", PathBuf::from(hf).join("hub"), ModelLayout::HuggingFace));
    } else if let Some(home) = home {
        dirs.push(source(
            "huggingface",
            home.join(".cache/huggingface/hub"),
            ModelLayout::HuggingFace,
        ));
    }

    if let Some(home) = home {
        dirs.push(source("lmstudio", home.join(".lmstudio/models"), ModelLayout::LmStudio));
        dirs.push(source("models", home.join("models"), ModelLayout::Directory));
    }

    dirs
}

/// Look up a resolved option's value by key.
fn option_value(options: &[OptionItem], key: &str) -> Option<String> {
    options.iter().find(|o| o.key == key).map(|o| o.value.clone())
}

/// Hand the terminal to a foreground tool, then re-enter the TUI. The detached
/// session supervisor sets `SIGCHLD` to `SIG_IGN`, which would make `wait()`
/// fail, so default disposition is restored while the tool runs.
fn run_foreground(terminal: &mut DefaultTerminal, argv: &[String], label: &str) -> Result<()> {
    use std::process::Command as StdCommand;
    let Some((prog, args)) = argv.split_first() else {
        return Ok(());
    };

    ratatui::restore(); // leave the alternate screen + raw mode
    // SAFETY: setting a signal disposition is async-signal-safe and unconditional.
    unsafe { libc::signal(libc::SIGCHLD, libc::SIG_DFL) };
    let status = StdCommand::new(prog).args(args).status();
    unsafe { libc::signal(libc::SIGCHLD, libc::SIG_IGN) };

    if let Err(e) = &status {
        eprintln!("\n[llmctl] failed to start {label}: {e}");
    }
    eprintln!("\n[llmctl] {label} ended — press Enter to return to llmctl.");
    let _ = std::io::stdin().read_line(&mut String::new());

    *terminal = ratatui::init();
    terminal.clear()?;
    Ok(())
}

/// The body lines for a command-preview message: the pretty command plus a copy
/// confirmation.
fn command_message_lines(cmd: &session::command::Command) -> Vec<String> {
    let mut lines: Vec<String> = cmd.pretty().lines().map(String::from).collect();
    lines.push(String::new());
    lines.push("(copied to clipboard)".into());
    lines
}

/// Copy text to the system clipboard via the OSC 52 terminal escape. Works over
/// SSH and needs no external tool; terminals without support silently ignore it.
fn copy_to_clipboard(text: &str) {
    use std::io::Write;
    let payload = session::supervisor::base64(text.as_bytes());
    let seq = format!("\x1b]52;c;{payload}\x07");
    let mut out = std::io::stdout();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}

/// Read up to the last `max_lines` lines of a (possibly large) log file.
///
/// Servers write to this file as if it were a terminal — carriage returns to
/// redraw a progress line in place, ANSI sequences to erase it and hide the
/// cursor. Those bytes must not reach our own terminal, so each line is reduced
/// to the text a terminal would finally have displayed. Invalid UTF-8 is
/// replaced rather than discarding the whole file.
fn read_log_tail(path: &std::path::Path, max_lines: usize) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap_or_default();
    let content = String::from_utf8_lossy(&bytes);
    let mut lines: Vec<String> = content.lines().map(visible_line).collect();
    if lines.len() > max_lines {
        lines = lines.split_off(lines.len() - max_lines);
    }
    lines
}

/// Variation selectors (VS1–VS16 and the supplementary block) change how the
/// preceding character is drawn without being drawn themselves — which is
/// exactly what makes their width unmeasurable.
fn is_variation_selector(c: char) -> bool {
    matches!(c, '\u{FE00}'..='\u{FE0F}' | '\u{E0100}'..='\u{E01EF}')
}

/// What a terminal would show for one log line.
///
/// A carriage return rewrites the row from column 0, so a progress bar that
/// ticked a hundred times arrives as one line holding a hundred states. Only the
/// last one was ever visible, and it is the only one worth showing in a log
/// tail — the rest would be a wall of `Downloading: 0.3% … 0.5% …`.
fn visible_line(raw: &str) -> String {
    raw.split('\r')
        .map(strip_control)
        .filter(|segment| !segment.trim().is_empty())
        .last()
        .unwrap_or_default()
}

/// Drop ANSI escape sequences, stray control bytes, and variation selectors,
/// keeping printable text.
///
/// Left in place, `ESC[K` (erase to end of line) would wipe the rest of the row
/// including the log pane's border, and `ESC[?25l` would hide the cursor for the
/// rest of the session.
///
/// Variation selectors go for a subtler reason. `⬇️` is `U+2B07 U+FE0F`, and the
/// selector asks for emoji presentation, which a terminal draws two columns
/// wide — but `unicode-width` still measures the pair as one. The renderer then
/// lays the row out one cell narrower than it actually paints, and everything to
/// its right, border included, is overwritten. Dropping the selector leaves a
/// bare `U+2B07`, which measures and draws as one column. Characters that are
/// emoji by default (`🔗`, `🔒`) carry no selector and already measure correctly.
fn strip_control(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            if is_variation_selector(c) {
                continue;
            }
            if c == '\t' || !c.is_control() {
                out.push(c);
            }
            continue;
        }
        match chars.next() {
            // CSI: parameter bytes, then a final byte in @..~ .
            Some('[') => {
                for c in chars.by_ref() {
                    if ('\x40'..='\x7e').contains(&c) {
                        break;
                    }
                }
            }
            // OSC: runs until BEL or a String Terminator.
            Some(']') => {
                while let Some(c) = chars.next() {
                    if c == '\x07' {
                        break;
                    }
                    if c == '\x1b' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            // Any other escape is two characters; both are already consumed.
            _ => {}
        }
    }
    out
}

/// Cycle through automatic device selection and the devices discovered from
/// `llama-server --list-devices`, wrapping in either direction.
fn cycle_device(devices: &[String], current: &str, dir: i32) -> String {
    let variants = std::iter::once(profiles::registry::DEFAULT)
        .chain(devices.iter().map(String::as_str))
        .collect::<Vec<_>>();
    let current = variants.iter().position(|value| *value == current).unwrap_or(0) as i32;
    let next = (current + dir.signum()).rem_euclid(variants.len() as i32) as usize;
    variants[next].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selector() -> Selector {
        Selector {
            key: "chat-template".into(),
            title: "Select chat-template".into(),
            variants: ["default", "chatml", "llama2", "llama3", "mistral-v1", "zephyr"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            filter: String::new(),
            cursor: 0,
        }
    }

    /// Regression: `flm` writes progress to its log the way it would to a
    /// terminal. A bare carriage return sent the cursor back to column 0 and
    /// `ESC[K` erased to end of line, so those rows overwrote the log pane's
    /// borders and the text beside them.
    #[test]
    fn log_lines_are_reduced_to_what_a_terminal_would_show() {
        // Verbatim bytes from a real FastFlowLM session log.
        let overall = "\r[FLM]  Overall progress:  1/6 files";
        assert_eq!(visible_line(overall), "[FLM]  Overall progress:  1/6 files");

        // A progress bar redrawn in place: only the final state was ever visible.
        let progress = "\u{1b}[?25l\r\u{1b}[K[FLM]  Downloading: 0.0% (0.0MB / 2340.0MB)\
                        \r\u{1b}[K[FLM]  Downloading: 0.3% (6.0MB / 2340.0MB)\
                        \r\u{1b}[K[FLM]  Downloading: 100.0% (2340.0MB / 2340.0MB)\u{1b}[?25h";
        assert_eq!(visible_line(progress), "[FLM]  Downloading: 100.0% (2340.0MB / 2340.0MB)");

        // Cursor show/hide around plain text leaves just the text.
        assert_eq!(
            visible_line("\u{1b}[?25l\u{1b}[?25h[FLM]  Checking Hash..."),
            "[FLM]  Checking Hash..."
        );

        // Nothing survives a line that was only control bytes.
        assert_eq!(visible_line("\u{1b}[?25l\u{1b}[?25h"), "");

        // Ordinary lines pass through untouched, including colour codes.
        assert_eq!(visible_line("plain server line"), "plain server line");
        assert_eq!(visible_line("\u{1b}[31mred\u{1b}[0m text"), "red text");
    }

    /// Regression: a variation selector makes a character draw two columns wide
    /// while `unicode-width` still measures one, so the log pane laid out rows
    /// narrower than it painted them and clobbered its own border.
    #[test]
    fn rendered_log_width_matches_what_the_terminal_draws() {
        use unicode_width::UnicodeWidthStr;

        // Verbatim from a FastFlowLM session log: U+2B07 followed by U+FE0F.
        let arrow = visible_line("[\u{2B07}\u{FE0F} ]  Incoming Request: GET");
        assert_eq!(arrow, "[\u{2B07} ]  Incoming Request: GET");
        // The selector is gone, so the measured width is now the drawn width.
        assert!(!arrow.chars().any(is_variation_selector));
        assert_eq!(arrow.width(), arrow.chars().count());

        // Characters that are emoji by default carry no selector and already
        // measure correctly at two columns; they must survive untouched.
        let link = visible_line("[\u{1F517} ]  TCP connection established");
        assert!(link.starts_with("[\u{1F517}"));
        assert_eq!(link.width(), link.chars().count() + 1);
    }

    #[test]
    fn no_rendered_log_line_can_carry_control_bytes() {
        // Whatever a server writes, nothing that could move the cursor or erase
        // the frame may reach the terminal.
        let nasty = "\u{1b}[2J\u{1b}]0;title\u{7}\rone\u{1b}[Ktwo\u{0}\u{8}";
        let rendered = visible_line(nasty);
        assert!(
            !rendered.chars().any(|c| c.is_control() && c != '\t'),
            "control byte survived: {rendered:?}"
        );
        assert_eq!(rendered, "onetwo");
    }

    #[test]
    fn selector_filters_case_insensitive_substring() {
        let mut sel = selector();
        sel.filter = "LLaMA".into();
        assert_eq!(sel.filtered(), vec!["llama2", "llama3"]);
        sel.filter = "tral".into(); // substring, not just prefix
        assert_eq!(sel.filtered(), vec!["mistral-v1"]);
        sel.filter = "nope".into();
        assert!(sel.filtered().is_empty());
        assert_eq!(sel.selected(), None);
    }

    #[test]
    fn model_download_percentage_is_bounded() {
        assert_eq!(transfer_percent(0, 300), 0);
        assert_eq!(transfer_percent(201, 300), 67);
        assert_eq!(transfer_percent(400, 300), 100);
    }

    #[test]
    fn persisted_download_is_restored_as_interrupted() {
        let stamp =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-app-download-{stamp}"));
        let model_id = format!("online:huggingface:llmctl-tests/restart-{stamp}:model.gguf");
        let remote = crate::domain::RemoteModel {
            repo: format!("llmctl-tests/restart-{stamp}"),
            revision: Some("revision".into()),
            file: Some("model.gguf".into()),
            blobs: vec![crate::domain::RemoteBlob {
                oid: "cd".repeat(32),
                size_bytes: 42,
                file: "model.gguf".into(),
            }],
            mtp_file: None,
            projector_file: None,
            downloads: 0,
            likes: 0,
            gated: false,
        };
        let record = discovery::online::DownloadJobRecord::new(
            model_id.clone(),
            format!("{}/model.gguf", remote.repo),
            remote,
        );
        discovery::online::save_download_record(&root, &record).unwrap();

        let (downloads, next_id) = restore_model_downloads(&root);
        assert_eq!(downloads.len(), 1);
        assert_eq!(downloads[0].model_id, model_id);
        assert_eq!(downloads[0].downloaded_bytes, 0);
        assert_eq!(downloads[0].total_bytes, 42);
        assert!(matches!(downloads[0].status, ModelDownloadStatus::Interrupted));
        assert_eq!(next_id, 2);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn selector_selection_tracks_the_filtered_list() {
        let mut sel = selector();
        assert_eq!(sel.selected(), Some("default")); // cursor 0, no filter
        sel.filter = "llama".into();
        sel.cursor = 1;
        assert_eq!(sel.selected(), Some("llama3"));
        sel.cursor = 5; // beyond the filtered list
        assert_eq!(sel.selected(), None);
    }

    #[test]
    fn chat_template_enum_exceeds_the_selector_threshold() {
        use crate::profiles::registry::OptionKind;
        use crate::runtime::llama_cpp::SCHEMA;
        let spec = SCHEMA.spec("chat-template").unwrap();
        let OptionKind::Enum(variants) = spec.kind else {
            panic!("chat-template should be an enum");
        };
        assert!(variants.len() > SELECTOR_THRESHOLD);
        // The small on/off/auto enums keep cycling in place.
        let flash = SCHEMA.spec("flash-attn").unwrap();
        let OptionKind::Enum(variants) = flash.kind else {
            panic!("flash-attn should be an enum");
        };
        assert!(variants.len() <= SELECTOR_THRESHOLD);
    }

    #[test]
    fn device_hotkeys_cycle_and_wrap_in_both_directions() {
        let devices = vec!["ROCm0".into(), "Vulkan0".into()];
        assert_eq!(cycle_device(&devices, "default", 1), "ROCm0");
        assert_eq!(cycle_device(&devices, "ROCm0", 1), "Vulkan0");
        assert_eq!(cycle_device(&devices, "Vulkan0", 1), "default");
        assert_eq!(cycle_device(&devices, "default", -1), "Vulkan0");
        assert_eq!(cycle_device(&devices, "ROCm0", -1), "default");
    }

    #[test]
    fn device_hotkeys_stay_at_default_when_no_devices_are_discovered() {
        assert_eq!(cycle_device(&[], "default", 1), "default");
        assert_eq!(cycle_device(&[], "stale-device", -1), "default");
    }

    fn flm_catalog_entry(tag: &str, label: &str) -> Model {
        let mut model = crate::domain::Model {
            id: format!("flm:{tag}"),
            name: tag.into(),
            path: PathBuf::new(),
            shard_paths: Vec::new(),
            mtp_path: None,
            projector_path: None,
            has_mtp: false,
            catalog_path: vec!["online".into(), label.into(), tag.into()],
            catalog_dir: PathBuf::new(),
            size_bytes: 0,
            quantization: None,
            architecture: None,
            context_length: None,
            modified: None,
            has_chat_template: true,
            remote: None,
            flm: None,
            runtime: crate::runtime::flm::NAME.into(),
        };
        model.flm = Some(crate::domain::FlmModel {
            tag: tag.into(),
            installed: false,
            repo: format!("FastFlowLM/{tag}"),
            revision: "main".into(),
            files: vec!["model.q4nx".into()],
            labels: vec![label.into()],
            vlm: false,
            asr: false,
            max_prefill_len: None,
        });
        model
    }

    /// Drives the real app: selecting FastFlowLM and pressing `s` must leave a
    /// populated Model pane in every arrangement, and must keep the user on the
    /// model they were looking at rather than resetting to the group list.
    #[test]
    #[ignore = "needs a real flm install; run with --ignored --test-threads=1"]
    fn cycling_the_catalog_view_keeps_the_pane_populated_and_the_selection() {
        let stamp =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-view-{stamp}"));
        let paths = Paths {
            config_file: root.join("config.toml"),
            models_dir: root.join("models"),
            state_dir: root.join("state"),
            cache_dir: root.join("cache"),
            log_dir: root.join("logs"),
            sessions_dir: root.join("sessions"),
        };
        paths.ensure_dirs().unwrap();

        let mut app = App::new(Config::default(), paths);
        let flm = app
            .runtimes
            .items
            .iter()
            .position(|b| !b.supports_online_browse())
            .expect("FastFlowLM backend");
        app.runtimes.state.select(Some(flm));
        app.rebuild_below(Pane::Runtime);
        app.focus = Pane::Model;

        assert_eq!(app.catalog_view_label(), Some("Categories"));
        let categories = app.models.items.len();
        assert!(categories > 0, "Categories view showed nothing");

        app.on_key(KeyEvent::from(KeyCode::Char('s')));

        assert_eq!(app.catalog_view_label(), Some("Flat"));
        assert!(!app.models.items.is_empty(), "Flat view showed nothing");
        // Both arrangements start at the same local/online groups.
        assert_eq!(app.models.items.len(), categories);

        // And drilling into a group reaches the models themselves.
        app.enter();
        assert!(app.models.items.iter().any(|m| m.is_model()), "no models under the group");

        // Select a model, then switch arrangement: the browser must stay on it
        // rather than dropping back to the group list.
        let index = app.models.items.iter().position(|m| m.is_model()).unwrap();
        app.models.state.select(Some(index));
        app.rebuild_below(Pane::Model);
        let tag = app.selected_model().unwrap().flm.as_ref().unwrap().tag.clone();

        app.on_key(KeyEvent::from(KeyCode::Char('s')));
        assert_eq!(app.catalog_view_label(), Some("Categories"));
        assert_eq!(app.focus, Pane::Model, "arrangement switch moved the focus");
        assert_eq!(
            app.selected_model().and_then(|m| m.flm.as_ref()).map(|f| f.tag.clone()),
            Some(tag.clone()),
            "arrangement switch lost the selected model"
        );
        // Still inside a group, not back at the top of the tree.
        assert!(!app.catalog_prefix.is_empty(), "arrangement switch reset to the group list");

        // And back again, from the deeper arrangement to the flatter one.
        app.on_key(KeyEvent::from(KeyCode::Char('s')));
        assert_eq!(app.catalog_view_label(), Some("Flat"));
        assert_eq!(
            app.selected_model().and_then(|m| m.flm.as_ref()).map(|f| f.tag.clone()),
            Some(tag),
            "switching back lost the selected model"
        );

        // F5 re-reads the catalog through the same subprocess path, so it fails
        // the same way if the SIGCHLD disposition is not handled.
        app.refresh_models();
        assert!(!app.flm_models.is_empty(), "F5 emptied the FastFlowLM catalog");
        assert!(!app.models.items.is_empty(), "F5 emptied the model pane");

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Browsing a capability folder in Categories and switching to Flat must
    /// land inside the group, looking at models — not back at the group list.
    #[test]
    #[ignore = "needs a real flm install; run with --ignored --test-threads=1"]
    fn switching_arrangement_from_a_capability_folder_lands_on_models() {
        let stamp =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-view-folder-{stamp}"));
        let paths = Paths {
            config_file: root.join("config.toml"),
            models_dir: root.join("models"),
            state_dir: root.join("state"),
            cache_dir: root.join("cache"),
            log_dir: root.join("logs"),
            sessions_dir: root.join("sessions"),
        };
        paths.ensure_dirs().unwrap();

        let mut app = App::new(Config::default(), paths);
        let flm = app.runtimes.items.iter().position(|b| !b.supports_online_browse()).unwrap();
        app.runtimes.state.select(Some(flm));
        app.rebuild_below(Pane::Runtime);
        app.focus = Pane::Model;
        assert_eq!(app.catalog_view_label(), Some("Categories"));

        // Enter `online`, then the `chat` capability folder inside it.
        let online = app.models.items.iter().position(|m| m.name == "online").unwrap();
        app.models.state.select(Some(online));
        app.rebuild_below(Pane::Model);
        app.enter();
        let chat = app.models.items.iter().position(|m| m.name == "chat").unwrap();
        app.models.state.select(Some(chat));
        app.rebuild_below(Pane::Model);
        app.enter();
        assert_eq!(app.catalog_prefix, ["online", "chat"]);

        // Deselect, so the anchor is the directory rather than a model.
        app.models.state.select(None);
        app.rebuild_below(Pane::Model);
        assert!(app.selected_model().is_none());

        app.on_key(KeyEvent::from(KeyCode::Char('s')));

        assert_eq!(app.catalog_view_label(), Some("Flat"));
        // Inside `online`, showing models — not back at the group list.
        assert_eq!(app.catalog_prefix, ["online"]);
        assert!(
            app.models.items.iter().all(|m| m.is_model()),
            "expected the flat model list, got folders"
        );
        assert!(app.models.items.len() > 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The AMD NPU takes one client at a time, so a live FastFlowLM session has
    /// to block the next launch — while leaving llama.cpp free to run several.
    /// Ignored by default: it spawns a stand-in server process.
    #[test]
    #[ignore = "spawns a real process; run with --ignored --test-threads=1"]
    fn a_live_flm_session_blocks_a_second_launch() {
        use crate::session::LaunchRequest;
        use crate::session::command::Command;

        let stamp =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-exclusive-{stamp}"));
        let paths = Paths {
            config_file: root.join("config.toml"),
            models_dir: root.join("models"),
            state_dir: root.join("state"),
            cache_dir: root.join("cache"),
            log_dir: root.join("logs"),
            sessions_dir: root.join("sessions"),
        };
        paths.ensure_dirs().unwrap();

        let mut app = App::new(Config::default(), paths);
        let flm = crate::runtime::flm::NAME;
        assert!(app.single_session_conflict(flm).is_none(), "nothing running yet");

        // A stand-in for `flm serve`: long-lived, and never answering /health,
        // so the session stays non-terminal for the duration of the test.
        let req = LaunchRequest {
            runtime: flm.into(),
            model: "qwen3:0.6b".into(),
            model_path: "qwen3:0.6b".into(),
            command: Command { argv: vec!["sleep".into(), "30".into()] },
            health_path: "/v1/models".into(),
            download: None,
            profile: "Default".into(),
            host: "127.0.0.1".into(),
            port: 52625,
        };
        app.sessions.launch(req).expect("launch stand-in");

        let lines = app.single_session_conflict(flm).expect("a live flm session must block");
        assert!(lines[0].contains("one model at a time"), "{lines:?}");
        assert!(lines[1].contains("qwen3"), "the blocking session is named: {lines:?}");
        // llama.cpp shares no such constraint.
        assert!(app.single_session_conflict("llama.cpp").is_none());

        // Once it is gone, the guard lifts.
        app.sessions.stop(0).expect("stop");
        for _ in 0..50 {
            app.sessions.refresh();
            if app.single_session_conflict(flm).is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(app.single_session_conflict(flm).is_none(), "guard stuck after the session ended");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn flat_catalog_walks_from_the_group_straight_to_the_models() {
        let mut flat = flm_catalog_entry("qwen3:4b", "reasoning");
        flat.catalog_path = vec!["online".into(), "qwen3:4b".into()];
        let mut other = flm_catalog_entry("gpt-oss:20b", "chat");
        other.catalog_path = vec!["online".into(), "gpt-oss:20b".into()];
        let mut local = flm_catalog_entry("qwen3:0.6b", "chat");
        local.catalog_path = vec!["local".into(), "qwen3:0.6b".into()];
        let source = vec![flat, other, local];

        // Top level: the two groups, as folders.
        let top = catalog_children_of(&source, &[]);
        assert_eq!(top.iter().map(|m| m.name.clone()).collect::<Vec<_>>(), ["local", "online"]);
        assert!(top.iter().all(|m| m.is_catalog_dir()));

        // Entering a group lists the models directly — no capability level.
        let online = catalog_children_of(&source, &["online".into()]);
        assert_eq!(online.len(), 2);
        assert!(online.iter().all(|m| m.is_model()), "flat entries must be leaves");
        assert!(online.iter().any(|m| m.name == "qwen3:4b"));
    }

    /// Regression: search ranked one runtime's catalog and then resolved the
    /// resulting indices against another, panicking with an out-of-bounds index
    /// as soon as the ranked list was the longer of the two.
    #[test]
    fn search_indices_address_the_list_they_were_ranked_against() {
        // A FastFlowLM catalog that is longer than the llama.cpp one, which is
        // what turned the mismatch into a panic rather than a wrong result.
        let flm: Vec<Model> =
            (0..54).map(|i| flm_catalog_entry(&format!("qwen3:{i}b"), "reasoning")).collect();

        let scope = vec!["online".into(), "reasoning".into()];
        let ranked = rank_models(&flm, "qwen3", &scope, false);
        assert!(!ranked.is_empty());
        // Every index must address the slice that was ranked.
        for index in &ranked {
            assert!(*index < flm.len(), "index {index} escapes a {}-entry list", flm.len());
            assert_eq!(flm[*index].runtime, crate::runtime::flm::NAME);
        }

        // A FastFlowLM entry has no `remote`, so a Hub-scoped search must not
        // return it — that is what keeps `/` from querying Hugging Face here.
        assert!(rank_models(&flm, "qwen3", &scope, true).is_empty());
    }

    #[test]
    fn local_search_scope_is_the_current_directory_not_the_hovered_child() {
        let prefix = vec!["models".into(), "team".into()];
        let hovered = vec!["models".into(), "team".into(), "project".into()];
        assert_eq!(normalized_search_scope(Pane::Model, Some(&hovered), &prefix, true), prefix);
        assert!(normalized_search_scope(Pane::Runtime, Some(&hovered), &prefix, true).is_empty());
    }

    #[test]
    fn online_search_scope_tracks_the_flat_repository_folder() {
        let online = vec!["online".into()];
        assert_eq!(
            normalized_search_scope(Pane::Model, Some(&online), &[], true),
            vec!["online", "huggingface"]
        );

        // FastFlowLM names its remote group `online` too, but it is a fixed
        // local catalog — the scope must stay put rather than expanding into a
        // Hugging Face subtree it does not have.
        assert_eq!(
            normalized_search_scope(Pane::Model, Some(&online), &[], false),
            Vec::<String>::new()
        );
        let flm = vec!["online".into(), "reasoning".into()];
        assert_eq!(normalized_search_scope(Pane::Model, None, &flm, false), flm);

        let repository = vec!["online".into(), "huggingface".into(), "unsloth/model".into()];
        assert_eq!(normalized_search_scope(Pane::Profile, None, &repository, true), repository);
    }

    #[test]
    fn online_artifact_pane_uses_the_standard_model_title() {
        assert_eq!(
            model_catalog_title(
                &["online".into(), "huggingface".into(), "DreamFast/gemma-3-12b".into()],
                discovery::online::Sort::Trending,
            ),
            "Model"
        );
    }

    #[test]
    fn recursive_scope_excludes_siblings_and_the_other_source_kind() {
        let scope = vec!["models".into(), "team".into()];
        let nested = vec!["models".into(), "team".into(), "repo".into(), "model".into()];
        let sibling = vec!["models".into(), "other".into(), "model".into()];
        let online = vec!["online".into(), "huggingface".into(), "owner/repo".into()];

        assert!(catalog_entry_in_search_scope(&nested, false, &scope, false));
        assert!(!catalog_entry_in_search_scope(&scope, false, &scope, false));
        assert!(!catalog_entry_in_search_scope(&sibling, false, &scope, false));
        assert!(!catalog_entry_in_search_scope(&online, true, &scope, false));
        assert!(catalog_entry_in_search_scope(
            &online,
            true,
            &["online".into(), "huggingface".into()],
            true
        ));
    }
}
