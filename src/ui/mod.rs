//! ratatui rendering — Yazi-style sliding three-column miller view.
//!
//! Layout per frame:
//! ```text
//!  header: breadcrumb path
//!  ┌ Parent ─┬ Current ───┬ Preview ──────┐
//!  │ ancestor│ focused    │ children, or  │
//!  │ list    │ list       │ leaf detail   │
//!  └─────────┴────────────┴───────────────┘
//!  footer: hovered-item metadata            keys
//! ```
//! Columns slide left as the user drills in (`l`/`→`) and right on `h`/`←`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap,
};

use crate::app::{
    App, Message, ModelDownload, ModelDownloadStatus, ModelSearch, Pane, Prompt, Screen, Selector,
};
use crate::domain::human_size;
use crate::session::{Session, SessionStatus, format_uptime};

const ACCENT: Color = Color::Yellow;

// Nerd-font glyphs (Yazi-style), written as escapes so the codepoints survive
// in source regardless of editor/transport. Require a Nerd Font in the terminal.
const ICON_RUNTIME: &str = "\u{f085}"; // cogs
const ICON_MODEL: &str = "\u{f1b2}"; // cube
const ICON_PROFILE: &str = "\u{f02e}"; // bookmark
const ICON_OPTION: &str = "\u{f1de}"; // sliders
const ICON_ROOT: &str = "\u{f015}"; // home
const ICON_SESSION: &str = "\u{f233}"; // server
const ICON_LOG: &str = "\u{f15c}"; // file-text
const ICON_DIRECTORY: &str = "\u{f07b}"; // folder
const ICON_CLOUD: &str = "\u{f0c2}"; // cloud

fn level_icon(level: Pane) -> &'static str {
    match level {
        Pane::Runtime => ICON_RUNTIME,
        Pane::Model => ICON_MODEL,
        Pane::Profile => ICON_PROFILE,
        Pane::Options => ICON_OPTION,
    }
}

/// Which slot a column occupies in the sliding window.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Parent,
    Current,
    Preview,
}

pub fn draw(frame: &mut Frame, app: &mut App) {
    match app.screen {
        Screen::Browser => draw_browser(frame, app),
        Screen::Sessions => draw_sessions(frame, app),
        Screen::Logs => draw_logs(frame, app),
    }

    if app.show_help {
        render_help(frame, frame.area());
    }
    if let Some(prompt) = &app.prompt {
        render_prompt(frame, frame.area(), prompt);
    }
    if let Some(selector) = &app.selector {
        render_selector(frame, frame.area(), selector);
    }
    if let Some(search) = &app.model_search {
        render_model_search(frame, frame.area(), app, search);
    }
    if let Some(message) = &app.message {
        render_message(frame, frame.area(), message);
    }
}

/// The Yazi-style three-column browser.
fn draw_browser(frame: &mut Frame, app: &mut App) {
    let [header, body, footer] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(0), Constraint::Length(3)])
            .areas(frame.area());

    // Parent | Current | Preview.
    let [parent, current, preview] = Layout::horizontal([
        Constraint::Percentage(18),
        Constraint::Percentage(48),
        Constraint::Percentage(34),
    ])
    .areas(body);

    render_header(frame, header, app);

    // Parent column: the level above the current one (root is virtual).
    match app.focus {
        Pane::Runtime => render_root(frame, parent),
        Pane::Model if app.catalog_parent().is_some() => render_catalog_parent(frame, parent, app),
        other => render_list(frame, parent, app, other.prev(), Role::Parent),
    }

    // Current column: the focused level.
    render_list(frame, current, app, app.focus, Role::Current);

    // Preview column: children of the hovered item, or the leaf detail.
    match app.focus {
        Pane::Runtime => render_list(frame, preview, app, Pane::Model, Role::Preview),
        Pane::Model if app.selected_model().is_none() => {
            render_catalog_preview(frame, preview, app)
        }
        Pane::Model => render_list(frame, preview, app, Pane::Profile, Role::Preview),
        Pane::Profile => render_list(frame, preview, app, Pane::Options, Role::Preview),
        Pane::Options => render_option_detail(frame, preview, app),
    }

    render_footer(frame, footer, app);
}

/// Render one level's list into a column, styled for its role.
fn render_list(frame: &mut Frame, area: Rect, app: &mut App, level: Pane, role: Role) {
    let focused = role == Role::Current;
    let title = if level == Pane::Model { app.model_pane_title() } else { level.title().into() };
    let block = pane_block(&title, focused);

    // Build owned items first so the immutable borrow ends before we take the
    // mutable state borrow below.
    let icon = level_icon(level);
    let items: Vec<ListItem> = match level {
        Pane::Runtime => app
            .runtimes
            .items
            .iter()
            .map(|r| ListItem::new(format!("{icon}  {}", r.name)))
            .collect(),
        Pane::Model => app
            .models
            .items
            .iter()
            .map(|m| {
                let label = m.display_label();
                if let Some(remote) = &m.remote
                    && remote.file.is_none()
                {
                    return ListItem::new(Line::from(vec![
                        Span::raw(format!("{ICON_CLOUD}  {label}  ")),
                        Span::styled(
                            format!(
                                "♥{} ⇩{}",
                                compact_count(remote.likes),
                                compact_count(remote.downloads)
                            ),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]));
                }
                if let Some((metadata, filename)) = model_artifact_columns(m) {
                    return ListItem::new(Line::from(vec![
                        Span::styled(metadata, Style::default().fg(Color::DarkGray)),
                        Span::raw(filename),
                    ]));
                }
                let item_icon = model_icon(m);
                ListItem::new(format!("{item_icon}  {label}"))
            })
            .collect(),
        Pane::Profile => app
            .profiles
            .items
            .iter()
            .map(|p| {
                let star = if p.favorite { " ★" } else { "" };
                ListItem::new(format!("{icon}  {}{star}", p.name))
            })
            .collect(),
        Pane::Options => app
            .options
            .items
            .iter()
            .map(|o| {
                ListItem::new(Line::from(vec![
                    Span::raw(format!("{icon}  {}: ", o.key)),
                    Span::styled(o.value.clone(), Style::default().fg(ACCENT)),
                ]))
            })
            .collect(),
    };

    if items.is_empty() {
        frame.render_widget(block, area);
        return;
    }

    // Preview columns are read-only: render plainly, no cursor.
    if role == Role::Preview {
        let list =
            List::new(items).block(block).style(Style::default().add_modifier(Modifier::DIM));
        frame.render_widget(list, area);
        return;
    }

    let highlight = match role {
        Role::Current => Style::default().fg(Color::Black).bg(ACCENT).add_modifier(Modifier::BOLD),
        // Parent: show which item we descended through, but muted.
        _ => Style::default().fg(ACCENT).add_modifier(Modifier::DIM),
    };
    let symbol = if focused { "▌ " } else { "  " };

    let state = match level {
        Pane::Runtime => &mut app.runtimes.state,
        Pane::Model => &mut app.models.state,
        Pane::Profile => &mut app.profiles.state,
        Pane::Options => &mut app.options.state,
    };

    let list = List::new(items).block(block).highlight_style(highlight).highlight_symbol(symbol);
    frame.render_stateful_widget(list, area, state);
}

fn render_catalog_parent(frame: &mut Frame, area: Rect, app: &App) {
    let Some((models, selected)) = app.catalog_parent() else { return };
    let items: Vec<ListItem> = models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let label = m.display_label();
            let marker = if Some(i) == selected { "▸" } else { " " };
            ListItem::new(format!("{marker}  {}  {label}", model_icon(m)))
        })
        .collect();
    frame.render_widget(
        List::new(items)
            .block(pane_block(&app.catalog_parent_title(), false))
            .style(Style::default().dim()),
        area,
    );
}

fn render_catalog_preview(frame: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = app
        .catalog_preview
        .iter()
        .map(|m| {
            let label = m.display_label();
            if let Some((metadata, filename)) = model_artifact_columns(m) {
                return ListItem::new(Line::from(vec![
                    Span::styled(metadata, Style::default().fg(Color::DarkGray)),
                    Span::raw(filename),
                ]));
            }
            let icon = model_icon(m);
            ListItem::new(format!("{icon}  {label}"))
        })
        .collect();
    frame.render_widget(
        List::new(items)
            .block(pane_block(&app.catalog_preview_title(), false))
            .style(Style::default().dim()),
        area,
    );
}

/// The virtual root shown left of the Runtime column.
fn render_root(frame: &mut Frame, area: Rect) {
    let block = pane_block("/", false);
    let inner = Paragraph::new(Line::from(format!("{ICON_ROOT}  llmctl").dim())).block(block);
    frame.render_widget(inner, area);
}

/// Leaf detail shown in the preview column when the Options level is current:
/// the editable option's current/default/CLI/description (spec's Option Preview).
fn render_option_detail(frame: &mut Frame, area: Rect, app: &App) {
    let block = pane_block("Detail", false);
    let text = app
        .options
        .selected()
        .map(|o| {
            Text::from(vec![
                Line::from(o.key.clone().bold().fg(ACCENT)),
                Line::raw(""),
                kv("Current", &o.value),
                kv("Default", &o.default),
                kv("Range", o.range.as_deref().unwrap_or("free-form")),
                Line::raw(""),
                Line::from("CLI".bold()),
                Line::from(o.cli.clone()),
                Line::raw(""),
                Line::from(o.description.clone()),
            ])
        })
        .unwrap_or_else(|| Text::from(Line::from("(no option selected)".dim())));
    frame.render_widget(Paragraph::new(text).block(block).wrap(Wrap { trim: false }), area);
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let crumbs = app.breadcrumb().join(" / ");
    let breadcrumb = Line::from(vec![
        Span::styled(" / ", Style::default().fg(Color::DarkGray)),
        Span::styled(crumbs, Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
    ]);
    frame.render_widget(Paragraph::new(breadcrumb), area);
}

/// All status lives at the bottom: path, then metadata, then context hotkeys.
fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let (primary, metadata) = app.status();
    let [l1, l2, l3] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)])
            .areas(area);

    // Line 1: the locator/path. Left-truncate (keeping the tail) if too wide.
    let path = truncate_left(&primary, l1.width.saturating_sub(1) as usize);
    frame.render_widget(Paragraph::new(Line::from(format!(" {path}")).dim()), l1);

    // Line 2: hovered-item metadata.
    frame.render_widget(Paragraph::new(Line::from(format!(" {metadata}")).dim()), l2);

    // Line 3: context-sensitive hotkeys for the focused pane.
    let mut spans = vec![Span::raw(" ")];
    for (k, label) in hotkeys(app) {
        spans.push(Span::styled(k, Style::default().fg(ACCENT)));
        spans.push(Span::raw(format!(" {label}   ")));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), l3);
}

/// The hotkeys relevant to the current focus, shown in the footer.
fn hotkeys(app: &App) -> Vec<(&'static str, &'static str)> {
    let mut keys: Vec<(&str, &str)> = vec![("j/k", "move")];
    match app.focus {
        Pane::Runtime => {
            keys.push(("l", "enter"));
            keys.push(("/", "search models"));
        }
        Pane::Model => {
            keys.push(("h/l", "back/enter"));
            keys.push(("/", "search models"));
            keys.push(("F5", "rescan"));
            if app.online_view_active() {
                keys.push(("s", "sort"));
            }
            if app.download_available() {
                keys.push(("d", "download"));
            }
            if app.benchmark_available() {
                keys.push(("b", "benchmark"));
            }
        }
        Pane::Profile => {
            // Built-ins are read-only templates: no rename, and `d` resets
            // (drops model-scoped edits) rather than deleting.
            let builtin = app.profiles.selected().map(|p| p.builtin).unwrap_or(false);
            keys.push(("h/l", "back/enter"));
            keys.push(("a", "new"));
            if !builtin {
                keys.push(("r", "rename"));
            }
            keys.push(("D", "dup"));
            keys.push(("d", if builtin { "reset" } else { "del" }));
            keys.push(("f", "fav"));
            keys.push(("s", "start"));
            keys.push(("C", "chat"));
            if app.benchmark_available() {
                keys.push(("b", "benchmark"));
            }
            keys.push(("y", "yank"));
        }
        Pane::Options => {
            keys.push(("h", "back"));
            keys.push(("e", "edit"));
            keys.push(("-/+", "adjust"));
            keys.push(("d", "default"));
            keys.push(("Home/End", "min/max"));
            keys.push(("s", "start"));
            keys.push(("C", "chat"));
            if app.benchmark_available() {
                keys.push(("b", "benchmark"));
            }
            keys.push(("y", "yank"));
        }
    }
    keys.push(("t", "sessions"));
    keys.push(("?", "help"));
    keys.push(("q", "quit"));
    keys
}

/// Truncate from the left, keeping the rightmost characters with a leading `…`.
fn truncate_left(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    if max <= 1 {
        return "…".repeat(max);
    }
    let tail: String = s.chars().skip(count - (max - 1)).collect();
    format!("…{tail}")
}

fn compact_count(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}m", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn model_icon(model: &crate::domain::Model) -> &'static str {
    if crate::discovery::online::is_online_path(&model.catalog_path) {
        ICON_CLOUD
    } else if model.is_catalog_dir() {
        ICON_DIRECTORY
    } else {
        ICON_MODEL
    }
}

fn model_artifact_columns(model: &crate::domain::Model) -> Option<(String, String)> {
    if !model.is_model() {
        return None;
    }
    let quantization = model.quantization.as_deref().unwrap_or("-");
    Some((
        format!("{quantization:<12}{:>7}  ", download_size(model.size_bytes)),
        model.display_label().into(),
    ))
}

fn download_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1_000.0 && unit < UNITS.len() - 1 {
        size /= 1_000.0;
        unit += 1;
    }
    if unit == 0 { format!("{bytes} B") } else { format!("{size:.1} {}", UNITS[unit]) }
}

fn download_progress(downloaded: u64, total: u64, percent: u8) -> String {
    format!("{} / {} ({percent}%)", download_size(downloaded), download_size(total))
}

fn truncate_download_name(name: &str, metadata: &str, row_width: usize) -> String {
    truncate_left(name, row_width.saturating_sub(metadata.chars().count()))
}

// --- Session Manager screen ------------------------------------------------

/// Colour for a session status indicator.
fn status_color(status: SessionStatus) -> Color {
    match status {
        SessionStatus::Downloading => Color::Cyan,
        SessionStatus::Running => Color::Green,
        SessionStatus::Starting => ACCENT,
        SessionStatus::Crashed => Color::Red,
        SessionStatus::Stopped => Color::DarkGray,
        SessionStatus::Unknown => Color::DarkGray,
    }
}

/// The Session Manager: list of servers on the left, detail on the right.
fn draw_sessions(frame: &mut Frame, app: &mut App) {
    let [header, body, footer] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)])
            .areas(frame.area());

    let title = Line::from(vec![
        Span::styled(format!(" {ICON_SESSION}  Sessions "), Style::default().fg(ACCENT).bold()),
        Span::styled(
            format!("({} jobs)", app.async_job_count()),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    frame.render_widget(Paragraph::new(title), header);

    let [jobs, detail] =
        Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)]).areas(body);
    let [sessions, downloads] =
        Layout::vertical([Constraint::Percentage(70), Constraint::Percentage(30)]).areas(jobs);
    render_session_list(frame, sessions, app);
    render_download_list(frame, downloads, app);
    render_session_detail(frame, detail, app);

    let keys = if let Some(download) = app.selected_model_download() {
        match &download.status {
            ModelDownloadStatus::Downloading => {
                vec![("x", "cancel"), ("Esc", "back"), ("q", "quit")]
            }
            ModelDownloadStatus::Cancelling => vec![("Esc", "back"), ("q", "quit")],
            ModelDownloadStatus::Cancelled
            | ModelDownloadStatus::Interrupted
            | ModelDownloadStatus::Failed(_) => {
                vec![("R", "resume"), ("d", "remove"), ("Esc", "back"), ("q", "quit")]
            }
            ModelDownloadStatus::Downloaded(_) => {
                vec![("d", "remove"), ("Esc", "back"), ("q", "quit")]
            }
        }
    } else {
        vec![
            ("x", "stop"),
            ("K", "kill"),
            ("R", "restart"),
            ("L", "logs"),
            ("c", "copy url"),
            ("y", "yank cmd"),
            ("d", "remove"),
            ("Esc", "back"),
            ("q", "quit"),
        ]
    };
    render_keyline(frame, footer, &keys);
}

fn render_session_list(frame: &mut Frame, area: Rect, app: &mut App) {
    let items: Vec<ListItem> = app
        .sessions
        .sessions
        .iter()
        .map(|s| {
            let color = status_color(s.status);
            let uptime = s.uptime_secs().map(format_uptime).unwrap_or_else(|| "—".into());
            ListItem::new(Line::from(vec![
                Span::styled(format!("{} ", s.status.glyph()), Style::default().fg(color)),
                Span::raw(s.record.name.clone()),
                Span::styled(
                    format!("   port:{}  {}", s.record.port, uptime),
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
        })
        .collect();
    let focused = app.selected_server_session().is_some() || app.async_job_count() == 0;
    let mut state = ListState::default();
    state.select(app.session_sel.selected().filter(|index| *index < app.sessions.sessions.len()));
    let list = List::new(items)
        .block(pane_block("Sessions", focused))
        .highlight_style(Style::default().fg(Color::Black).bg(ACCENT).add_modifier(Modifier::BOLD))
        .highlight_symbol("▌ ");
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_download_list(frame: &mut Frame, area: Rect, app: &App) {
    // Borders consume two columns and the selected-row marker consumes two
    // more. Reserve the progress suffix, then retain the filename tail.
    let row_width = area.width.saturating_sub(4) as usize;
    let items = app.model_downloads.iter().map(|download| {
        let suffix = match &download.status {
            ModelDownloadStatus::Downloading => "",
            ModelDownloadStatus::Cancelling => "  cancelling",
            ModelDownloadStatus::Downloaded(_) => "  downloaded",
            ModelDownloadStatus::Cancelled => "  cancelled",
            ModelDownloadStatus::Interrupted => "  interrupted",
            ModelDownloadStatus::Failed(_) => "  failed",
        };
        let metadata = format!(
            " ⇣ {}{suffix}",
            download_progress(download.downloaded_bytes, download.total_bytes, download.percent())
        );
        let name = truncate_download_name(&download.model, &metadata, row_width);
        ListItem::new(Line::from(vec![
            Span::raw(name),
            Span::styled(metadata, Style::default().fg(Color::DarkGray)),
        ]))
    });

    let selected = app
        .session_sel
        .selected()
        .and_then(|index| index.checked_sub(app.sessions.sessions.len()))
        .filter(|index| *index < app.model_downloads.len());
    let mut state = ListState::default();
    state.select(selected);
    let list = List::new(items)
        .block(pane_block("Downloads", selected.is_some()))
        .highlight_style(Style::default().fg(Color::Black).bg(ACCENT).add_modifier(Modifier::BOLD))
        .highlight_symbol("▌ ");
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_session_detail(frame: &mut Frame, area: Rect, app: &App) {
    let block = pane_block("Detail", false);
    let text = if let Some(session) = app.selected_server_session() {
        session_detail_lines(session)
    } else if let Some(download) = app.selected_model_download() {
        download_detail_lines(download)
    } else {
        frame.render_widget(block, area);
        return;
    };
    frame.render_widget(Paragraph::new(text).block(block).wrap(Wrap { trim: false }), area);
}

fn download_detail_lines(download: &ModelDownload) -> Text<'static> {
    let (status, color, detail) = match &download.status {
        ModelDownloadStatus::Downloading => ("Downloading", Color::Cyan, String::new()),
        ModelDownloadStatus::Cancelling => (
            "Cancelling",
            Color::DarkGray,
            "Waiting for the transfer worker to preserve its partial file.".into(),
        ),
        ModelDownloadStatus::Downloaded(path) => {
            ("Downloaded", Color::Green, path.display().to_string())
        }
        ModelDownloadStatus::Cancelled => {
            ("Cancelled", Color::DarkGray, "Press R to resume the partial download.".into())
        }
        ModelDownloadStatus::Interrupted => (
            "Interrupted",
            Color::DarkGray,
            "The previous llmctl process stopped. Press R to resume the partial download.".into(),
        ),
        ModelDownloadStatus::Failed(error) => ("Failed", Color::Red, error.clone()),
    };
    Text::from(vec![
        Line::from(download.model.clone().bold().fg(ACCENT)),
        Line::raw(""),
        Line::from(vec![
            Span::styled("Status: ", Style::default().fg(Color::DarkGray)),
            Span::styled(status.to_string(), Style::default().fg(color)),
        ]),
        kv(
            "Progress",
            &download_progress(download.downloaded_bytes, download.total_bytes, download.percent()),
        ),
        Line::raw(""),
        Line::from(detail),
    ])
}

fn session_detail_lines(session: &Session) -> Text<'static> {
    let r = &session.record;
    let color = status_color(session.status);
    let uptime = session.uptime_secs().map(format_uptime).unwrap_or_else(|| "—".into());
    let mem = session.rss_bytes.map(human_size).unwrap_or_else(|| "—".into());
    let cpu = session.cpu_percent.map(|c| format!("{c:.0}%")).unwrap_or_else(|| "—".into());

    Text::from(vec![
        Line::from(r.name.clone().bold().fg(ACCENT)),
        Line::raw(""),
        Line::from(vec![
            Span::styled("Status: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{} {}", session.status.glyph(), session.status_label()),
                Style::default().fg(color),
            ),
        ]),
        kv("Runtime", &r.runtime),
        kv("Model", &r.model),
        kv("Profile", &r.profile),
        Line::raw(""),
        kv("PID", &r.pid.to_string()),
        kv("Port", &r.port.to_string()),
        kv("Uptime", &uptime),
        kv("Memory", &mem),
        kv("CPU", &cpu),
        Line::raw(""),
        kv("Endpoint", &r.endpoint()),
        kv("Log", &r.log_file.display().to_string()),
        Line::raw(""),
        Line::from("Command".bold()),
        Line::from(crate::session::command::Command { argv: r.command.clone() }.display()),
    ])
}

/// The log-tail screen for a session.
fn draw_logs(frame: &mut Frame, app: &mut App) {
    let [header, body, footer] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)])
            .areas(frame.area());

    let name = app
        .session_sel
        .selected()
        .and_then(|i| app.sessions.sessions.get(i))
        .map(|s| s.record.name.clone())
        .unwrap_or_default();
    let follow = if app.log_follow { "  [tailing]" } else { "" };
    let title = Line::from(vec![
        Span::styled(format!(" {ICON_LOG}  Logs — {name}"), Style::default().fg(ACCENT).bold()),
        Span::styled(follow.to_string(), Style::default().fg(Color::Green)),
    ]);
    frame.render_widget(Paragraph::new(title), header);

    let block = pane_block("Output", true);
    let inner_height = body.height.saturating_sub(2); // borders
    let total = app.log_lines.len() as u16;
    let max_scroll = total.saturating_sub(inner_height);
    let scroll = if app.log_follow { max_scroll } else { app.log_scroll.min(max_scroll) };
    app.log_scroll = scroll; // keep state clamped/in-sync

    let text = if app.log_lines.is_empty() {
        Text::from(Line::from("(log is empty)".dim()))
    } else {
        Text::from(app.log_lines.iter().map(|l| Line::raw(l.clone())).collect::<Vec<_>>())
    };
    frame.render_widget(Paragraph::new(text).block(block).scroll((scroll, 0)), body);

    let keys =
        [("j/k", "scroll"), ("g/G", "top/tail"), ("F5", "reload"), ("Esc", "back"), ("q", "quit")];
    render_keyline(frame, footer, &keys);
}

/// Render a single-line key hint row (used by the Session/Logs screens).
fn render_keyline(frame: &mut Frame, area: Rect, keys: &[(&str, &str)]) {
    let mut spans = vec![Span::raw(" ")];
    for (k, label) in keys {
        spans.push(Span::styled(k.to_string(), Style::default().fg(ACCENT)));
        spans.push(Span::raw(format!(" {label}   ")));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// A read-only message modal (launch preview, copy confirmation, errors).
fn render_message(frame: &mut Frame, area: Rect, message: &Message) {
    let mut lines: Vec<Line> = message.lines.iter().map(|l| Line::raw(l.clone())).collect();
    lines.push(Line::raw(""));
    lines.push(Line::from("press any key to dismiss".dim().italic()));

    let width = message.lines.iter().map(|l| l.chars().count()).max().unwrap_or(20).clamp(24, 88)
        as u16
        + 4;
    let height = lines.len() as u16 + 2;
    let popup = center(area, Constraint::Length(width), Constraint::Length(height));
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(ACCENT))
        .title(format!(" {} ", message.title));

    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(lines).block(block).wrap(Wrap { trim: false }), popup);
}

fn render_help(frame: &mut Frame, area: Rect) {
    let lines = vec![
        Line::from("llmctl — keybindings".bold().fg(ACCENT)),
        Line::raw(""),
        Line::from("Navigation".bold()),
        help_row("j / k", "move down / up"),
        help_row("l / →", "drill into selection"),
        help_row("h / ←", "back up a level"),
        help_row("g / G", "first / last item"),
        help_row("/", "search models"),
        help_row("s", "sort online models"),
        help_row("d", "download selected online file"),
        Line::raw(""),
        Line::from("Profiles".bold()),
        help_row("a", "create profile"),
        help_row("r", "rename (custom only)"),
        help_row("D", "duplicate profile"),
        help_row("d", "delete / reset profile"),
        help_row("f", "toggle favorite"),
        Line::raw(""),
        Line::from("Options".bold()),
        help_row("e / Enter", "edit / cycle / pick value"),
        help_row("- / +", "decrement / increment"),
        help_row("[ / ]", "decrement / increment"),
        help_row("d", "reset to default"),
        help_row("Home/End", "min / max"),
        Line::raw(""),
        Line::from("Launch & sessions".bold()),
        help_row("s", "start server (profile/options)"),
        help_row("C", "chat in terminal (llama-cli)"),
        help_row("b", "benchmark selected model (llama-bench)"),
        help_row("y", "yank command"),
        help_row("t", "session manager"),
        help_row("x / K", "stop / kill / cancel"),
        help_row("R", "restart / resume"),
        help_row("L", "view logs"),
        help_row("c", "copy endpoint"),
        Line::raw(""),
        Line::from("General".bold()),
        help_row("F5", "rescan / reload"),
        help_row("? / q", "help / quit"),
        Line::raw(""),
        Line::from("press ? or Esc to close".dim().italic()),
    ];

    let height = lines.len() as u16 + 2;
    let popup = center(area, Constraint::Length(44), Constraint::Length(height));
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(ACCENT))
        .title(" Help ");

    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(lines).block(block), popup);
}

fn render_model_search(frame: &mut Frame, area: Rect, app: &App, search: &ModelSearch) {
    let results = app.search_results();
    let visible = results.len().min(12);
    let popup = center(area, Constraint::Percentage(72), Constraint::Length(visible as u16 + 4));
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(ACCENT))
        .title(if search.online {
            let suffix = search.scope.get(2..).unwrap_or_default().join(" / ");
            if suffix.is_empty() {
                " Search Hugging Face ".to_string()
            } else {
                format!(" Search Hugging Face / {suffix} ")
            }
        } else {
            format!(" Search {} ", search.scope.last().map(String::as_str).unwrap_or("models"))
        });
    let mut lines = vec![Line::from(vec![
        Span::styled("❯ ", Style::default().fg(ACCENT)),
        Span::raw(search.query.clone()),
        Span::styled("▏", Style::default().add_modifier(Modifier::SLOW_BLINK)),
    ])];
    if results.is_empty() {
        lines.push(Line::from("  No matching models".dim()));
    } else {
        let start = search.cursor.saturating_sub(visible.saturating_sub(1));
        for (index, model) in results.iter().enumerate().skip(start).take(visible) {
            let selected = index == search.cursor;
            let label = model.display_label();
            let context =
                model.catalog_path[..model.catalog_path.len().saturating_sub(1)].join(" / ");
            let line = format!("{} {}  ·  {}", if selected { "▸" } else { " " }, label, context);
            lines.push(if selected {
                Line::from(line).fg(Color::Black).bg(ACCENT).bold()
            } else {
                Line::from(line)
            });
        }
    }
    lines.push(Line::from(" Enter jump  ·  Esc close".dim()));
    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(lines).block(block), popup);
}

/// Modal enum-variant selector (combo box): a filter line above the variant
/// list, scrolled to the cursor.
fn render_selector(frame: &mut Frame, area: Rect, selector: &Selector) {
    let filtered = selector.filtered();

    // Filter input line, styled like the text prompt.
    let filter = Line::from(vec![
        Span::styled("❯ ", Style::default().fg(ACCENT)),
        Span::raw(selector.filter.clone()),
        Span::styled("▏", Style::default().add_modifier(Modifier::SLOW_BLINK)),
    ]);

    let list_height = filtered.len().clamp(1, 12) as u16;
    let height = list_height + 4; // borders + filter + hint
    let popup = center(area, Constraint::Length(40), Constraint::Length(height));
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(ACCENT))
        .title(format!(" {} ", selector.title));
    let inner = block.inner(popup);

    frame.render_widget(Clear, popup);
    frame.render_widget(block, popup);

    let [filter_area, list_area, hint_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(1), Constraint::Length(1)])
            .areas(inner);

    frame.render_widget(Paragraph::new(filter), filter_area);

    if filtered.is_empty() {
        frame.render_widget(Paragraph::new(Line::from("(no matches)".dim().italic())), list_area);
    } else {
        let items: Vec<ListItem> = filtered.iter().map(|v| ListItem::new(*v)).collect();
        let list = List::new(items).highlight_style(
            Style::default().fg(Color::Black).bg(ACCENT).add_modifier(Modifier::BOLD),
        );
        let mut state = ListState::default();
        state.select(Some(selector.cursor.min(filtered.len() - 1)));
        frame.render_stateful_widget(list, list_area, &mut state);
    }

    frame.render_widget(
        Paragraph::new(Line::from("type to filter · ↑/↓ · Enter pick · Esc".dim().italic())),
        hint_area,
    );
}

/// Modal text input for editing an option value or naming a profile.
fn render_prompt(frame: &mut Frame, area: Rect, prompt: &Prompt) {
    let mut lines = vec![Line::from(vec![
        Span::styled("❯ ", Style::default().fg(ACCENT)),
        Span::raw(prompt.buffer.clone()),
        Span::styled("▏", Style::default().add_modifier(Modifier::SLOW_BLINK)),
    ])];
    if let Some(err) = &prompt.error {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(err.clone(), Style::default().fg(Color::Red))));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from("Enter save · Esc cancel".dim().italic()));

    let height = lines.len() as u16 + 2;
    let popup = center(area, Constraint::Length(54), Constraint::Length(height));
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(ACCENT))
        .title(format!(" {} ", prompt.title));

    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(lines).block(block), popup);
}

// --- helpers ---------------------------------------------------------------

fn pane_block(title: &str, focused: bool) -> Block<'static> {
    let border_style =
        if focused { Style::default().fg(ACCENT) } else { Style::default().fg(Color::DarkGray) };
    let title = if focused {
        Span::styled(format!(" {title} "), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
    } else {
        Span::styled(format!(" {title} "), Style::default().fg(Color::DarkGray))
    };
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style)
        .title(title)
}

/// A "key: value" line where the key is dimmed.
fn kv<'a>(key: &str, value: &str) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{key}: "), Style::default().fg(Color::DarkGray)),
        Span::raw(value.to_string()),
    ])
}

fn help_row<'a>(keys: &str, desc: &str) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("  {keys:<8}"), Style::default().fg(ACCENT)),
        Span::raw(desc.to_string()),
    ])
}

/// Center a rect of the given width/height constraints within `area`.
fn center(area: Rect, horizontal: Constraint, vertical: Constraint) -> Rect {
    let [h] = Layout::horizontal([horizontal]).flex(Flex::Center).areas(area);
    let [v] = Layout::vertical([vertical]).flex(Flex::Center).areas(h);
    v
}

#[cfg(test)]
mod tests {
    use super::{
        ICON_CLOUD, ICON_DIRECTORY, compact_count, download_progress, model_artifact_columns,
        model_icon, truncate_download_name,
    };

    #[test]
    fn counts_use_compact_repository_badges() {
        assert_eq!(compact_count(999), "999");
        assert_eq!(compact_count(1_200), "1.2k");
        assert_eq!(compact_count(445_400), "445.4k");
        assert_eq!(compact_count(1_250_000), "1.2m");
    }

    #[test]
    fn download_progress_is_rendered_on_one_compact_line() {
        assert_eq!(download_progress(400_000_000, 1_000_000_000, 40), "400.0 MB / 1.0 GB (40%)");
    }

    #[test]
    fn download_name_is_truncated_from_the_left_to_preserve_progress() {
        let name = "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf";
        let metadata = " ⇣ 9.0 GB / 20.0 GB (45%)";
        let truncated = truncate_download_name(name, metadata, 72);

        assert!(truncated.starts_with('…'));
        assert!(truncated.ends_with("Aggressive-IQ2_M.gguf"));
        assert!(truncated.chars().count() + metadata.chars().count() <= 72);
    }

    #[test]
    fn artifact_columns_show_quant_size_and_filename() {
        let mut model = crate::domain::stubs::vllm_models().remove(0);
        model.name = "Qwen-AgentWorld-35B-A3B-UD-Q4_K_M.gguf".into();
        model.catalog_path = vec![model.name.clone()];
        model.size_bytes = 20_600_000_000;
        model.quantization = Some("Q4_K_M".into());
        model.remote = Some(crate::domain::RemoteModel {
            repo: "owner/repo".into(),
            revision: None,
            file: Some(model.name.clone()),
            blobs: Vec::new(),
            mtp_file: None,
            projector_file: None,
            downloads: 0,
            likes: 0,
            gated: false,
        });

        assert_eq!(
            model_artifact_columns(&model).unwrap(),
            ("Q4_K_M      20.6 GB  ".into(), "Qwen-AgentWorld-35B-A3B-UD-Q4_K_M.gguf".into())
        );

        model.remote = None;
        assert_eq!(
            model_artifact_columns(&model).unwrap(),
            ("Q4_K_M      20.6 GB  ".into(), "Qwen-AgentWorld-35B-A3B-UD-Q4_K_M.gguf".into())
        );
    }

    #[test]
    fn online_catalog_nodes_use_cloud_icons() {
        let mut model = crate::domain::stubs::vllm_models().remove(0);
        model.path = std::path::PathBuf::new();

        model.catalog_path = vec!["online".into()];
        assert_eq!(model_icon(&model), ICON_CLOUD);

        model.catalog_path = vec!["online".into(), "huggingface".into()];
        assert_eq!(model_icon(&model), ICON_CLOUD);

        model.catalog_path = vec!["local-models".into()];
        assert_eq!(model_icon(&model), ICON_DIRECTORY);
    }
}
