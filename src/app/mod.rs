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

use crate::config::{Config, Paths};
use crate::discovery;
use crate::domain::{Model, OptionItem, Profile, Runtime, format_unix_date, human_size, stubs};
use crate::profiles::{self, ProfileStore, store};
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
    pub runtimes: PaneList<Runtime>,
    pub models: PaneList<Model>,
    pub profiles: PaneList<Profile>,
    pub options: PaneList<OptionItem>,
    pub show_help: bool,
    pub prompt: Option<Prompt>,
    should_quit: bool,
    /// Discovered GGUF models for the llama.cpp runtime.
    scanned_models: Vec<Model>,
    /// Expanded, absolute model search directories.
    model_paths: Vec<PathBuf>,
    model_cache: PathBuf,
    /// Persisted, model-scoped profile instances.
    store: ProfileStore,
}

impl App {
    pub fn new(config: Config, paths: Paths) -> Self {
        // Discover the real llama.cpp runtime; keep vLLM as a demo stub.
        let llama = discovery::discover_llama_cpp(&config.runtime.llama_cpp, &paths.cache_dir);
        let model_paths = expand_model_paths(&config.models.paths);
        let model_cache = paths.cache_dir.join("models.json");
        let scanned_models = discovery::scan_models(&model_paths, &model_cache);
        let store = ProfileStore::load(paths.state_dir.join("profiles.json"));

        let mut app = Self {
            config,
            focus: Pane::Runtime,
            runtimes: PaneList::new(vec![llama, stubs::vllm_runtime()]),
            models: PaneList::new(Vec::new()),
            profiles: PaneList::new(Vec::new()),
            options: PaneList::new(Vec::new()),
            show_help: false,
            prompt: None,
            should_quit: false,
            scanned_models,
            model_paths,
            model_cache,
            store,
        };
        // Derive the whole chain from the initially-selected runtime.
        app.rebuild_below(Pane::Runtime);
        app
    }

    /// Run the draw/input loop until the user quits.
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.should_quit {
            terminal.draw(|frame| ui::draw(frame, self))?;
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    self.on_key(key);
                }
            }
        }
        Ok(())
    }

    fn on_key(&mut self, key: KeyEvent) {
        // A text prompt is modal: it consumes all input until closed.
        if self.prompt.is_some() {
            self.prompt_key(key);
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

        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('?') => self.show_help = true,

            // Move focus across panes.
            KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => self.enter(),
            KeyCode::Char('h') | KeyCode::Left => self.focus = self.focus.prev(),

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

            // Profile management (Profile pane).
            KeyCode::Char('a') => self.prompt_new_profile(),
            KeyCode::Char('r') => self.prompt_rename_profile(),
            KeyCode::Char('D') => self.prompt_duplicate_profile(),
            KeyCode::Char('d') => self.delete_profile(),

            // Re-scan model directories.
            KeyCode::F(5) => self.refresh_models(),

            _ => {}
        }
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

    /// Open the option editor. Bool/Enum cycle in place; numeric/string open a
    /// text prompt. Applies only to real (non-stub) runtimes.
    fn open_editor(&mut self) {
        if self.focus != Pane::Options || self.is_stub_runtime() {
            return;
        }
        let Some(option) = self.options.selected() else { return };
        let key = option.key.clone();
        let current = option.value.clone();

        if let Some(spec) = profiles::registry::spec(&key) {
            use profiles::registry::OptionKind;
            // Bool and Enum don't need a text prompt — `e` advances them.
            if matches!(spec.kind, OptionKind::Bool | OptionKind::Enum(_)) {
                if let Some(next) = spec.kind.adjust(&current, 1, spec.step) {
                    self.apply_option_value(&key, next);
                }
                return;
            }
        }
        self.prompt = Some(Prompt {
            kind: PromptKind::EditOption { key: key.clone() },
            title: format!("Edit {key}"),
            buffer: current,
            error: None,
        });
    }

    /// Increment/decrement the selected option in place (auto-saves).
    fn adjust_option(&mut self, dir: i32) {
        self.transform_option(|kind, current, step| kind.adjust(current, dir, step));
    }

    /// Set the selected option to its min (`dir < 0`) or max (`dir > 0`).
    fn set_option_extreme(&mut self, dir: i32) {
        self.transform_option(|kind, _current, _step| kind.extreme(dir));
    }

    /// Shared helper: compute a new value for the selected option and apply it.
    fn transform_option(
        &mut self,
        f: impl Fn(&profiles::registry::OptionKind, &str, f64) -> Option<String>,
    ) {
        if self.focus != Pane::Options || self.is_stub_runtime() {
            return;
        }
        let Some(option) = self.options.selected() else { return };
        let key = option.key.clone();
        let current = option.value.clone();
        let Some(spec) = profiles::registry::spec(&key) else { return };
        // Use the model-aware kind so ctx-size respects the model's max context.
        let kind = match self.models.selected() {
            Some(m) => profiles::effective_kind(spec, m),
            None => spec.kind,
        };
        if let Some(value) = f(&kind, &current, spec.step) {
            self.apply_option_value(&key, value);
        }
    }

    /// Validate and commit the open prompt; dispatch by its kind.
    fn commit_prompt(&mut self) {
        let Some(prompt) = self.prompt.as_ref() else { return };
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
        let spec = profiles::registry::spec(key).ok_or("unknown option")?;
        let kind = match self.models.selected() {
            Some(m) => profiles::effective_kind(spec, m),
            None => spec.kind,
        };
        let value = kind.validate(input)?;
        self.apply_option_value(key, value);
        Ok(())
    }

    /// Persist an option value to the model-scoped instance (auto-saves) and
    /// refresh the Options pane while preserving the cursor position.
    fn apply_option_value(&mut self, key: &str, value: String) {
        let (Some(rt), Some(m), Some(p)) =
            (self.runtimes.selected(), self.models.selected(), self.profiles.selected())
        else {
            return;
        };
        let runtime = rt.name.clone();
        let model = store::model_key(&m.path);
        let profile = p.clone();
        let base = profiles::resolved_values(&profile, &self.config.defaults);

        let cursor = self.options.state.selected();
        self.store.set_value(&runtime, &model, &profile.name, key, value, &base);
        self.rebuild_below(Pane::Profile);
        self.options.state.select(cursor);
    }

    /// Toggle the favorite flag on the selected profile (real runtimes only).
    fn toggle_favorite(&mut self) {
        if self.focus != Pane::Profile || self.is_stub_runtime() {
            return;
        }
        let (Some(rt), Some(m), Some(p)) =
            (self.runtimes.selected(), self.models.selected(), self.profiles.selected())
        else {
            return;
        };
        let runtime = rt.name.clone();
        let model = store::model_key(&m.path);
        let profile = p.clone();
        let base = profiles::resolved_values(&profile, &self.config.defaults);

        let cursor = self.profiles.state.selected();
        self.store.toggle_favorite(&runtime, &model, &profile.name, &base);
        self.rebuild_below(Pane::Model);
        self.profiles.state.select(cursor);
    }

    // --- profile management (Profile pane) ---------------------------------

    fn prompt_new_profile(&mut self) {
        if self.focus != Pane::Profile || self.is_stub_runtime() {
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
        if self.focus != Pane::Profile || self.is_stub_runtime() {
            return;
        }
        let Some(p) = self.profiles.selected() else { return };
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
        if self.focus != Pane::Profile || self.is_stub_runtime() {
            return;
        }
        let Some(p) = self.profiles.selected() else { return };
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
        if self.focus != Pane::Profile || self.is_stub_runtime() {
            return;
        }
        let (Some(rt), Some(m), Some(p)) =
            (self.runtimes.selected(), self.models.selected(), self.profiles.selected())
        else {
            return;
        };
        let runtime = rt.name.clone();
        let model = store::model_key(&m.path);
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
        // Seed from the Default template's resolved values.
        let default = Profile { name: "Default".into(), builtin: true, favorite: false };
        let values = profiles::resolved_values(&default, &self.config.defaults);
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
        let (Some(rt), Some(m)) = (self.runtimes.selected(), self.models.selected()) else {
            return Err("no model selected".into());
        };
        let runtime = rt.name.clone();
        let model = store::model_key(&m.path);
        let src_profile = Profile {
            name: src.to_string(),
            builtin: profiles::templates::is_builtin(src),
            favorite: false,
        };
        // Copy the source's *current* values (including any instance edits).
        let values = profiles::current_values(rt, m, &src_profile, &self.store, &self.config.defaults);
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
        let rt = self.runtimes.selected()?;
        let m = self.models.selected()?;
        Some((rt.name.clone(), store::model_key(&m.path)))
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

    /// True when the selected runtime is the vLLM stub (no editing/persistence).
    fn is_stub_runtime(&self) -> bool {
        self.runtimes.selected().map(|r| r.name == "vLLM").unwrap_or(true)
    }

    /// Re-scan configured model directories (the `F5` refresh).
    fn refresh_models(&mut self) {
        self.scanned_models = discovery::scan_models(&self.model_paths, &self.model_cache);
        // Models or anything downstream may have changed; rebuild from runtime.
        self.rebuild_below(Pane::Runtime);
    }

    /// Drill into the preview pane, but only if it actually has items.
    fn enter(&mut self) {
        if self.focus != Pane::Options && !self.preview_is_empty() {
            self.focus = self.focus.next();
        }
    }

    /// Is the pane immediately right of focus (the preview) empty?
    fn preview_is_empty(&self) -> bool {
        match self.focus {
            Pane::Runtime => self.models.is_empty(),
            Pane::Model => self.profiles.is_empty(),
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
    }

    fn select_first(&mut self) {
        match self.focus {
            Pane::Runtime => self.runtimes.select_first(),
            Pane::Model => self.models.select_first(),
            Pane::Profile => self.profiles.select_first(),
            Pane::Options => self.options.select_first(),
        }
        self.rebuild_below(self.focus);
    }

    fn select_last(&mut self) {
        match self.focus {
            Pane::Runtime => self.runtimes.select_last(),
            Pane::Model => self.models.select_last(),
            Pane::Profile => self.profiles.select_last(),
            Pane::Options => self.options.select_last(),
        }
        self.rebuild_below(self.focus);
    }

    /// Rebuild every pane below `changed` from the current selection chain,
    /// cascading top-down so each child sees its freshly-reset parent.
    fn rebuild_below(&mut self, changed: Pane) {
        let level = changed.index();
        if level < Pane::Model.index() {
            let models = match self.runtimes.selected() {
                // vLLM is a stub; llama.cpp uses the discovered GGUF models.
                Some(rt) if rt.name == "vLLM" => stubs::vllm_models(),
                Some(_) => self.scanned_models.clone(),
                None => Vec::new(),
            };
            self.models.replace(models);
        }
        if level < Pane::Profile.index() {
            let profiles = match (self.runtimes.selected(), self.models.selected()) {
                (Some(rt), Some(m)) if rt.name == "vLLM" => stubs::profiles_for(m),
                (Some(rt), Some(m)) => profiles::list_profiles(rt, m, &self.store),
                _ => Vec::new(),
            };
            self.profiles.replace(profiles);
        }
        if level < Pane::Options.index() {
            let options =
                match (self.runtimes.selected(), self.models.selected(), self.profiles.selected()) {
                    (Some(rt), _, Some(p)) if rt.name == "vLLM" => stubs::options_for(p),
                    (Some(rt), Some(m), Some(p)) => {
                        profiles::resolve_options(rt, m, p, &self.store, &self.config.defaults)
                    }
                    _ => Vec::new(),
                };
            self.options.replace(options);
        }
    }

    /// Two-line status bar content for the hovered item: a primary locator
    /// (line 1 — a path) and a secondary metadata summary (line 2).
    pub fn status(&self) -> (String, String) {
        match self.focus {
            Pane::Runtime => self.runtimes.selected().map(|r| {
                let primary = r
                    .binary_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(binary not found)".into());
                let mut meta = Vec::new();
                if let Some(v) = &r.version {
                    meta.push(v.clone());
                }
                meta.push(r.formats_label());
                (primary, meta.join(" · "))
            }),
            Pane::Model => self.models.selected().map(|m| {
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
            crumbs.push(r.name.clone());
        }
        if self.focus >= Pane::Model {
            if let Some(m) = self.models.selected() {
                crumbs.push(m.name.clone());
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

/// Resolve the directories to scan for models.
///
/// When `config.models.paths` is set we honor it (expanding `~`); otherwise we
/// fall back to the well-known runtime model locations. We never scan `$HOME`
/// itself, only specific subdirectories (per the requirements).
fn expand_model_paths(configured: &[PathBuf]) -> Vec<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut paths: Vec<PathBuf> = if configured.is_empty() {
        default_model_dirs(home.as_deref())
    } else {
        configured
            .iter()
            .map(|p| match (p.strip_prefix("~"), &home) {
                (Ok(rest), Some(home)) => home.join(rest),
                _ => p.clone(),
            })
            .collect()
    };

    // De-duplicate (e.g. LLAMA_CACHE may equal ~/.cache/llama.cpp).
    paths.sort();
    paths.dedup();
    paths
}

/// Well-known directories where local runtimes keep models, including
/// env-var-configured caches. Only existing dirs matter; the scanner skips the
/// rest.
fn default_model_dirs(home: Option<&std::path::Path>) -> Vec<PathBuf> {
    use std::env::var_os;
    let mut dirs: Vec<PathBuf> = Vec::new();

    // llama.cpp download cache (LLAMA_CACHE overrides the default location).
    if let Some(cache) = var_os("LLAMA_CACHE") {
        dirs.push(PathBuf::from(cache));
    } else if let Some(home) = home {
        dirs.push(home.join(".cache/llama.cpp"));
    }

    // HuggingFace hub cache (used by `llama-server -hf` and others).
    if let Some(hf) = var_os("HUGGINGFACE_HUB_CACHE") {
        dirs.push(PathBuf::from(hf));
    } else if let Some(hf) = var_os("HF_HOME") {
        dirs.push(PathBuf::from(hf).join("hub"));
    } else if let Some(home) = home {
        dirs.push(home.join(".cache/huggingface/hub"));
    }

    if let Some(home) = home {
        dirs.push(home.join(".lmstudio/models")); // LM Studio
        dirs.push(home.join("models")); // generic convention
    }

    dirs
}
