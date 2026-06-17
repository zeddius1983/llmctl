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
use std::time::{Duration, Instant};

use crate::config::{Config, Paths};
use crate::discovery;
use crate::domain::{Model, OptionItem, Profile, Runtime, format_unix_date, human_size, stubs};
use crate::profiles::{self, ProfileStore, store};
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
    pub runtimes: PaneList<Runtime>,
    pub models: PaneList<Model>,
    pub profiles: PaneList<Profile>,
    pub options: PaneList<OptionItem>,
    pub show_help: bool,
    pub prompt: Option<Prompt>,
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
    /// Discovered GGUF models for the llama.cpp runtime.
    scanned_models: Vec<Model>,
    /// Expanded, absolute model search directories.
    model_paths: Vec<PathBuf>,
    model_cache: PathBuf,
    /// Persisted, model-scoped profile instances.
    store: ProfileStore,
    /// Last time live session status was refreshed.
    last_tick: Instant,
    /// A foreground interactive chat (`llama-cli`) to run on the next loop turn,
    /// suspending the TUI while it owns the terminal.
    pending_chat: Option<Vec<String>>,
}

impl App {
    pub fn new(config: Config, paths: Paths) -> Self {
        // Discover the real llama.cpp runtime; keep vLLM as a demo stub.
        let llama = discovery::discover_llama_cpp(&config.runtime.llama_cpp, &paths.cache_dir);
        let model_paths = expand_model_paths(&config.models.paths);
        let model_cache = paths.cache_dir.join("models.json");
        let scanned_models = discovery::scan_models(&model_paths, &model_cache);
        let store = ProfileStore::load(paths.state_dir.join("profiles.json"));
        // Built after discovery's one-shot `Command`s: the supervisor ignores
        // SIGCHLD, which would otherwise prevent reaping those probe processes.
        let sessions = SessionManager::new(paths.sessions_dir.clone(), paths.log_dir.clone());

        let mut app = Self {
            config,
            focus: Pane::Runtime,
            runtimes: PaneList::new(vec![llama, stubs::vllm_runtime()]),
            models: PaneList::new(Vec::new()),
            profiles: PaneList::new(Vec::new()),
            options: PaneList::new(Vec::new()),
            show_help: false,
            prompt: None,
            message: None,
            screen: Screen::Browser,
            sessions,
            session_sel: ListState::default(),
            log_lines: Vec::new(),
            log_follow: true,
            log_scroll: 0,
            should_quit: false,
            scanned_models,
            model_paths,
            model_cache,
            store,
            last_tick: Instant::now(),
            pending_chat: None,
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
                run_chat(terminal, &argv)?;
            }
        }
        Ok(())
    }

    /// Periodic refresh: update live session status/resources, and reload the
    /// log tail when the Logs screen is open.
    fn tick(&mut self) {
        self.sessions.refresh();
        self.sync_session_selection();
        if self.screen == Screen::Logs {
            self.reload_logs();
        }
        self.last_tick = Instant::now();
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
            KeyCode::Char('t') => self.open_sessions(),
            KeyCode::Char('y') => self.yank_command(),
            KeyCode::Char('s') => self.start_session(),
            KeyCode::Char('C') => self.start_chat(),

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

    /// Key handling for the Session Manager screen.
    fn on_key_sessions(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('?') => self.show_help = true,
            KeyCode::Esc | KeyCode::Char('t') => self.screen = Screen::Browser,
            KeyCode::Char('j') | KeyCode::Down => self.move_session(1),
            KeyCode::Char('k') | KeyCode::Up => self.move_session(-1),
            KeyCode::Char('g') | KeyCode::Home => {
                let any = !self.sessions.sessions.is_empty();
                self.session_sel.select(any.then_some(0));
            }
            KeyCode::Char('G') | KeyCode::End => {
                let len = self.sessions.sessions.len();
                self.session_sel.select((len > 0).then_some(len - 1));
            }
            KeyCode::Char('x') => self.session_action(|m, i| m.stop(i), "stop"),
            KeyCode::Char('K') => self.session_action(|m, i| m.kill(i), "kill"),
            KeyCode::Char('R') => self.session_action(|m, i| m.restart(i), "restart"),
            KeyCode::Char('d') => self.remove_session(),
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
        let len = self.sessions.sessions.len();
        if len == 0 {
            self.session_sel.select(None);
        } else {
            let i = self.session_sel.selected().unwrap_or(0).min(len - 1);
            self.session_sel.select(Some(i));
        }
    }

    fn move_session(&mut self, delta: isize) {
        let len = self.sessions.sessions.len();
        if len == 0 {
            return;
        }
        let cur = self.session_sel.selected().unwrap_or(0) as isize;
        let next = (cur + delta).clamp(0, len as isize - 1);
        self.session_sel.select(Some(next as usize));
    }

    /// Build a launch request from the current selection and resolved options.
    fn build_launch_request(&self) -> Result<LaunchRequest, String> {
        if self.is_stub_runtime() {
            return Err("vLLM is a navigation-only stub (not launchable)".into());
        }
        let rt = self.runtimes.selected().ok_or("no runtime selected")?;
        let model = self.models.selected().ok_or("no model selected")?;
        let profile = self.profiles.selected().ok_or("no profile selected")?;
        let binary = rt
            .binary_path
            .as_ref()
            .ok_or("llama-server binary not found on PATH")?
            .display()
            .to_string();
        let options = self.options.items.clone();
        let host = option_value(&options, "host").unwrap_or_else(|| "127.0.0.1".into());
        let port = option_value(&options, "port").and_then(|v| v.parse().ok()).unwrap_or(8000);
        Ok(LaunchRequest {
            runtime: rt.name.clone(),
            binary,
            model: model.name.clone(),
            model_path: model.path.display().to_string(),
            profile: profile.name.clone(),
            host,
            port,
            options,
        })
    }

    /// Preview the generated command and copy it to the clipboard (`y`).
    fn yank_command(&mut self) {
        if self.focus != Pane::Profile && self.focus != Pane::Options {
            return;
        }
        match self.build_launch_request() {
            Ok(req) => {
                let cmd =
                    session::command::Command::build(&req.binary, &req.model_path, &req.options);
                copy_to_clipboard(&cmd.display());
                self.message = Some(Message {
                    title: "Launch command".into(),
                    lines: command_message_lines(&cmd),
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
        match self.sessions.launch(req) {
            Ok(idx) => {
                let endpoint = self.sessions.sessions[idx].record.endpoint();
                let name = self.sessions.sessions[idx].record.name.clone();
                self.screen = Screen::Sessions;
                self.session_sel.select(Some(idx));
                self.message = Some(Message {
                    title: "Launched".into(),
                    lines: vec![name, format!("Starting — {endpoint}")],
                });
            }
            Err(e) => {
                self.message =
                    Some(Message { title: "Launch failed".into(), lines: vec![e.to_string()] })
            }
        }
    }

    /// Launch an interactive `llama-cli` chat for the current selection in the
    /// foreground (`C`). Server-only flags (host/port) are dropped and
    /// conversation mode is forced; the TUI is suspended while it runs.
    fn start_chat(&mut self) {
        if self.focus != Pane::Profile && self.focus != Pane::Options {
            return;
        }
        let req = match self.build_launch_request() {
            Ok(req) => req,
            Err(e) => {
                self.message = Some(Message { title: "Cannot start chat".into(), lines: vec![e] });
                return;
            }
        };
        let Some(cli) = cli_binary(&req.binary) else {
            self.message = Some(Message {
                title: "llama-cli not found".into(),
                lines: vec![
                    "Expected a 'llama-cli' binary next to llama-server.".into(),
                    "Chat mode needs the interactive llama.cpp client.".into(),
                ],
            });
            return;
        };
        // Drop server-only flags; keep the model plus sampling/runtime options.
        let opts: Vec<OptionItem> =
            req.options.into_iter().filter(|o| o.key != "host" && o.key != "port").collect();
        let cmd =
            session::command::Command::build(&cli.display().to_string(), &req.model_path, &opts);
        let mut argv = cmd.argv;
        argv.push("-cnv".into()); // conversation/chat mode
        self.pending_chat = Some(argv);
    }

    /// Apply a fallible supervisor action to the selected session.
    fn session_action(&mut self, f: impl Fn(&mut SessionManager, usize) -> Result<()>, verb: &str) {
        let Some(i) = self.session_sel.selected() else {
            return;
        };
        if let Err(e) = f(&mut self.sessions, i) {
            self.message =
                Some(Message { title: format!("Failed to {verb}"), lines: vec![e.to_string()] });
        }
    }

    /// Remove a terminated session record (`d`).
    fn remove_session(&mut self) {
        let Some(i) = self.session_sel.selected() else {
            return;
        };
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
        let Some(i) = self.session_sel.selected() else {
            return;
        };
        let endpoint = self.sessions.sessions[i].record.endpoint();
        copy_to_clipboard(&endpoint);
        self.message = Some(Message { title: "Endpoint copied".into(), lines: vec![endpoint] });
    }

    /// Show + copy the selected session's stored launch command (`y`).
    fn yank_session_command(&mut self) {
        let Some(i) = self.session_sel.selected() else {
            return;
        };
        let argv = self.sessions.sessions[i].record.command.clone();
        let cmd = session::command::Command { argv };
        copy_to_clipboard(&cmd.display());
        self.message =
            Some(Message { title: "Session command".into(), lines: command_message_lines(&cmd) });
    }

    /// Open the log-tail screen for the selected session (`L`).
    fn open_logs(&mut self) {
        if self.session_sel.selected().is_none() {
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

    /// Open the option editor. Enums cycle in place; numeric/string open a
    /// text prompt. Applies only to real (non-stub) runtimes.
    fn open_editor(&mut self) {
        if self.focus != Pane::Options || self.is_stub_runtime() {
            return;
        }
        let Some(option) = self.options.selected() else {
            return;
        };
        let key = option.key.clone();
        let current = option.value.clone();

        if let Some(spec) = profiles::registry::spec(&key) {
            use profiles::registry::OptionKind;
            // Enums don't need a text prompt — `e` advances to the next state
            // (which, for omittable options, cycles through "default" too).
            if matches!(spec.kind, OptionKind::Enum(_)) {
                if let Some(next) = spec.bump(&spec.kind, &current, 1) {
                    self.apply_option_value(&key, next);
                }
                return;
            }
        }
        let title = if profiles::registry::uses_sentinel(&key) {
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

    /// Increment/decrement the selected option in place (auto-saves).
    fn adjust_option(&mut self, dir: i32) {
        self.transform_option(|spec, kind, current| spec.bump(kind, current, dir));
    }

    /// Set the selected option to its min/`default` (`dir < 0`) or max (`dir > 0`).
    fn set_option_extreme(&mut self, dir: i32) {
        self.transform_option(|spec, kind, _current| spec.jump(kind, dir));
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
        if self.focus != Pane::Options || self.is_stub_runtime() {
            return;
        }
        let Some(option) = self.options.selected() else {
            return;
        };
        let key = option.key.clone();
        let current = option.value.clone();
        let Some(spec) = profiles::registry::spec(&key) else {
            return;
        };
        // Use the model-aware kind so ctx-size respects the model's max context.
        let kind = match self.models.selected() {
            Some(m) => profiles::effective_kind(spec, m),
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
        let spec = profiles::registry::spec(key).ok_or("unknown option")?;
        // Sentinel options accept "default" (or an empty entry) to drop the flag.
        if profiles::registry::uses_sentinel(key)
            && (input.is_empty() || input.eq_ignore_ascii_case(profiles::registry::DEFAULT))
        {
            self.apply_option_value(key, profiles::registry::DEFAULT.to_string());
            return Ok(());
        }
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
        if self.focus != Pane::Profile || self.is_stub_runtime() {
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
        let values =
            profiles::current_values(rt, m, &src_profile, &self.store, &self.config.defaults);
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
            let options = match (
                self.runtimes.selected(),
                self.models.selected(),
                self.profiles.selected(),
            ) {
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

/// Look up a resolved option's value by key.
fn option_value(options: &[OptionItem], key: &str) -> Option<String> {
    options.iter().find(|o| o.key == key).map(|o| o.value.clone())
}

/// Resolve the interactive `llama-cli` binary sitting next to `llama-server`.
fn cli_binary(server_binary: &str) -> Option<PathBuf> {
    let p = std::path::Path::new(server_binary);
    let file = p.file_name()?.to_string_lossy().into_owned();
    let cli_name = file.replace("llama-server", "llama-cli");
    if cli_name == file {
        return None; // not a llama-server-style binary name
    }
    let cli = p.with_file_name(cli_name);
    cli.exists().then_some(cli)
}

/// Hand the terminal to a foreground process (interactive chat), then re-enter
/// the TUI. The detached-session supervisor sets `SIGCHLD` to `SIG_IGN`, which
/// would make `wait()` fail, so default disposition is restored while it runs.
fn run_chat(terminal: &mut DefaultTerminal, argv: &[String]) -> Result<()> {
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
        eprintln!("\n[llmctl] failed to start chat: {e}");
    }
    eprintln!("\n[llmctl] chat ended — press Enter to return to llmctl.");
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
fn read_log_tail(path: &std::path::Path, max_lines: usize) -> Vec<String> {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut lines: Vec<String> = content.lines().map(String::from).collect();
    if lines.len() > max_lines {
        lines = lines.split_off(lines.len() - max_lines);
    }
    lines
}
