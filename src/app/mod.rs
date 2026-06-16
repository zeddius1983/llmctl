//! Application state and the input/event loop.
//!
//! Navigation follows Yazi's miller-columns: child panes are derived from the
//! parent's selection and only revealed one level ahead of focus (see
//! `IMPLEMENTATION_PLAN.md` → Navigation model).

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::DefaultTerminal;
use ratatui::widgets::ListState;

use crate::config::Config;
use crate::domain::{Model, OptionItem, Profile, Runtime, human_size, stubs};
use crate::ui;

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
    #[allow(dead_code)] // wired up in Phase 1+ (discovery uses config paths)
    pub config: Config,
    pub focus: Pane,
    pub runtimes: PaneList<Runtime>,
    pub models: PaneList<Model>,
    pub profiles: PaneList<Profile>,
    pub options: PaneList<OptionItem>,
    pub show_help: bool,
    should_quit: bool,
}

impl App {
    pub fn new(config: Config) -> Self {
        let mut app = Self {
            config,
            focus: Pane::Runtime,
            runtimes: PaneList::new(stubs::runtimes()),
            models: PaneList::new(Vec::new()),
            profiles: PaneList::new(Vec::new()),
            options: PaneList::new(Vec::new()),
            show_help: false,
            should_quit: false,
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
            KeyCode::Char('g') | KeyCode::Home => self.select_first(),
            KeyCode::Char('G') | KeyCode::End => self.select_last(),

            _ => {}
        }
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
            let models = self.runtimes.selected().map(stubs::models_for).unwrap_or_default();
            self.models.replace(models);
        }
        if level < Pane::Profile.index() {
            let profiles = self.models.selected().map(stubs::profiles_for).unwrap_or_default();
            self.profiles.replace(profiles);
        }
        if level < Pane::Options.index() {
            let options = self.profiles.selected().map(stubs::options_for).unwrap_or_default();
            self.options.replace(options);
        }
    }

    /// A one-line metadata summary of the hovered item in the current (middle)
    /// column, shown in the status bar — Yazi shows hovered file info this way.
    pub fn hovered_detail(&self) -> String {
        match self.focus {
            Pane::Runtime => self.runtimes.selected().map(|r| {
                let ver = r.version.clone().unwrap_or_else(|| "version n/a".into());
                let path = r
                    .binary_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(not found)".into());
                format!("{ver} · {path} · {}", r.formats_label())
            }),
            Pane::Model => self.models.selected().map(|m| {
                format!(
                    "{} · {} · {}",
                    human_size(m.size_bytes),
                    m.quantization.as_deref().unwrap_or("?"),
                    m.architecture.as_deref().unwrap_or("?"),
                )
            }),
            Pane::Profile => self.profiles.selected().map(|p| {
                let kind = if p.builtin { "built-in template" } else { "custom profile" };
                let fav = if p.favorite { " · ★" } else { "" };
                format!("{kind}{fav}")
            }),
            Pane::Options => self.options.selected().map(|o| {
                format!("current {} · default {} · {}", o.value, o.default, o.cli)
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
