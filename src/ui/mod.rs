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
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::app::{
    App, Confirm, Message, ModelDownload, ModelDownloadStatus, ModelSearch, Pane, Prompt, Screen,
    Selector, SessionPane,
};
use crate::domain::human_size;
use crate::session::throughput::{Phase, format_rate};
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

/// The smallest terminal llmctl draws its interface in.
///
/// The classic terminal size, and about where the three-column browser stops
/// being three columns of anything: narrower or shorter than this, panes are a
/// few characters wide and the popups have nowhere to open. The app degrades
/// down to here — shedding session columns, then the pane beside the list — and
/// says what it needs below it rather than drawing a frame nobody can read.
const MIN_COLS: u16 = 80;
const MIN_ROWS: u16 = 24;

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    if area.width < MIN_COLS || area.height < MIN_ROWS {
        render_too_small(frame, area);
        return;
    }
    match app.screen {
        Screen::Browser => draw_browser(frame, app),
        Screen::Sessions => draw_sessions(frame, app),
        Screen::Logs => draw_logs(frame, app),
    }

    if app.modals.help() {
        render_help(frame, frame.area());
    }
    if let Some(prompt) = app.modals.prompt() {
        render_prompt(frame, frame.area(), prompt);
    }
    if let Some(selector) = app.modals.selector() {
        render_selector(frame, frame.area(), selector);
    }
    if let Some(search) = app.modals.search() {
        render_model_search(frame, frame.area(), app, search);
    }
    if let Some(confirm) = app.modals.confirm() {
        render_confirm(frame, frame.area(), confirm);
    }
    if let Some(message) = &app.modals.message {
        render_message(frame, frame.area(), message);
    }
}

/// What llmctl shows instead of an interface it has no room for.
///
/// Plain centred lines, no border: the message has to survive sizes where a
/// bordered popup would have nothing left inside it, and it names both what is
/// needed and what there is, so the fix is a drag of the window edge.
fn render_too_small(frame: &mut Frame, area: Rect) {
    let lines = [
        "terminal too small".to_string(),
        format!("{MIN_COLS}\u{d7}{MIN_ROWS} needed \u{b7} {}\u{d7}{} now", area.width, area.height),
        "q quit".to_string(),
    ];
    let rows = (lines.len() as u16).min(area.height);
    let top = area.y + (area.height - rows) / 2;
    for (i, line) in lines.iter().take(rows as usize).enumerate() {
        let text = truncate_right(line, area.width as usize);
        let left = area.x + (area.width.saturating_sub(text.width() as u16)) / 2;
        let row = Rect::new(left, top + i as u16, area.width - (left - area.x), 1);
        let style = if i == 0 {
            Style::default().fg(ACCENT).bold()
        } else {
            Style::default().fg(Color::DarkGray)
        };
        frame.render_widget(Paragraph::new(Line::styled(text, style)), row);
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
    match app.browser.focus {
        Pane::Runtime => render_root(frame, parent),
        Pane::Model if app.catalog_parent().is_some() => render_catalog_parent(frame, parent, app),
        other => render_list(frame, parent, app, other.prev(), Role::Parent),
    }

    // Current column: the focused level.
    render_list(frame, current, app, app.browser.focus, Role::Current);

    // Preview column: children of the hovered item, or the leaf detail.
    match app.browser.focus {
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
            .browser
            .runtimes
            .items
            .iter()
            .map(|r| ListItem::new(format!("{icon}  {}", r.descriptor().name)))
            .collect(),
        Pane::Model => app
            .browser
            .models
            .items
            .iter()
            .map(|m| {
                let label = m.display_label();
                if let Some(remote) = m.remote()
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
            .browser
            .profiles
            .items
            .iter()
            .map(|p| {
                let star = if p.favorite { " ★" } else { "" };
                ListItem::new(format!("{icon}  {}{star}", p.name))
            })
            .collect(),
        Pane::Options => app
            .browser
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
        Pane::Runtime => &mut app.browser.runtimes.state,
        Pane::Model => &mut app.browser.models.state,
        Pane::Profile => &mut app.browser.profiles.state,
        Pane::Options => &mut app.browser.options.state,
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
        .browser
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
        .browser
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
    match app.browser.focus {
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
            } else if app.catalog_view_label().is_some() {
                keys.push(("s", "group"));
            }
            if app.download_available() {
                keys.push(("d", "download"));
            }
            if app.delete_available() {
                keys.push(("D", "delete"));
            }
            if app.benchmark_available() {
                keys.push(("b", "benchmark"));
            }
        }
        Pane::Profile => {
            // Built-ins are read-only templates: no rename, and `d` resets
            // (drops model-scoped edits) rather than deleting.
            let builtin = app.browser.profiles.selected().map(|p| p.builtin).unwrap_or(false);
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
        format!("{quantization:<12}{:>SIZE_WIDTH$}  ", download_size(model.size_bytes)),
        model.display_label().into(),
    ))
}

/// Width of the size column. `download_size` is widest at three mantissa digits
/// plus a two-letter unit ("960.0 MB"), so anything narrower lets a sub-gigabyte
/// model overflow and push the name column out of alignment.
const SIZE_WIDTH: usize = 8;

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
        SessionStatus::Restarting => Color::Yellow,
        SessionStatus::Crashed => Color::Red,
        SessionStatus::Stopped => Color::DarkGray,
        SessionStatus::Unknown => Color::DarkGray,
    }
}

/// The Session Manager: list of servers on the left, detail on the right.
/// The narrowest a right-hand column is worth drawing.
///
/// Under this every Detail line wraps into fragments and a log tail shows a few
/// characters per row — the pane stops answering anything and only takes width
/// from the list, which is the one thing on this screen that must stay legible.
const MIN_DETAIL_WIDTH: u16 = 44;

/// Split the Session Manager's body into the jobs column and the pane beside
/// it, which a narrow terminal does without entirely.
fn session_panes(body: Rect) -> (Rect, Option<Rect>) {
    let [jobs, detail] =
        Layout::horizontal([Constraint::Percentage(57), Constraint::Percentage(43)]).areas(body);
    if detail.width < MIN_DETAIL_WIDTH { (body, None) } else { (jobs, Some(detail)) }
}

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

    let (jobs, detail) = session_panes(body);
    // The key handler reads this: with no pane to swap, `l` opens the log full
    // screen rather than doing nothing.
    app.session_view.detail_pane_visible = detail.is_some();
    let [sessions, downloads] =
        Layout::vertical([Constraint::Percentage(70), Constraint::Percentage(30)]).areas(jobs);
    let focused = app.selected_server_session().is_some() || app.async_job_count() == 0;
    render_session_list(
        frame,
        sessions,
        &app.sessions.sessions,
        app.session_view.selection.selected(),
        focused,
    );
    render_download_list(frame, downloads, app);
    if let Some(detail) = detail {
        match app.session_view.pane {
            // A download has no log, so its facts hold the column either way.
            SessionPane::Log if app.selected_server_session().is_some() => {
                render_session_log(frame, detail, app.selected_server_session())
            }
            _ => render_session_detail(frame, detail, app),
        }
    }

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
        let pane = match app.session_view.pane {
            SessionPane::Detail => ("l", "log"),
            SessionPane::Log => ("l", "detail"),
        };
        vec![
            ("x", "stop"),
            ("K", "kill"),
            ("R", "restart"),
            pane,
            ("L", "full log"),
            ("c", "copy url"),
            ("y", "yank cmd"),
            ("d", "remove"),
            ("Esc", "back"),
            ("q", "quit"),
        ]
    };
    render_keyline(frame, footer, &keys);
}

/// Session-row columns, in display order, with the width each is padded to.
///
/// `Model` takes whatever is left. The rest are fixed so that they line up down
/// the pane — that alignment is the point, and it is why every cell is padded
/// even when its content is shorter.
const COL_PROFILE: usize = 12; // "[inquisitor]"
const COL_PORT: usize = 6; // ":65535"
const COL_SIZE: usize = 8; // "11.3 GB"
const COL_DEVICE: usize = 6; // "Vulkan"
const COL_RATE: usize = 12; // "tg 67.73 t/s"
const COL_UPTIME: usize = 8; // "2h 34m"; four-digit hours push the row, not wrap it
/// Width of the numeric field inside a rate cell, so `67.73` and `1425` line
/// their digits up against each other. Taken from the formatter, which
/// guarantees no figure exceeds it — padding cannot shrink an oversized one.
const RATE_FIGURE: usize = crate::session::throughput::RATE_WIDTH;
/// Shown where a session has no value for a column — an older record with no
/// size or backend, or a stopped session with no uptime.
const MISSING: &str = "—";
/// Gap between columns. Wide enough that the eye reads the row as columns
/// rather than as one run of text — the rate cells especially, which already
/// carry a space inside them.
const COL_GAP: usize = 4;
/// Below this much room for the model name, a column is not worth its space.
const MIN_NAME: usize = 12;
/// The widest the name column grows to.
///
/// Names are what the rows are about, so they get columns dropped for them —
/// but only up to here. Past it the surplus stops seating a name and starts
/// opening a gulf between the name and everything that describes it.
const MAX_NAME: usize = 32;

/// Which column to give up first when the pane cannot hold them all.
///
/// Deliberately not right-to-left: the uptime sits at the far right but is
/// worth more than the size or the backend, which never change once a session
/// is up. The Detail pane carries every one of them regardless.
const DROP_ORDER: [Column; 6] = [
    Column::Size,
    Column::Device,
    Column::Profile,
    Column::Prefill,
    Column::Uptime,
    Column::Decode,
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Column {
    Profile,
    Port,
    Size,
    Device,
    Decode,
    Prefill,
    Uptime,
}

impl Column {
    /// This column's text for a session.
    ///
    /// Always something: a column that vanished when a session had no figure
    /// for it would shift every column after it out of line with the rows above
    /// and below, which is the whole point of having columns.
    fn cell(self, session: &Session) -> String {
        match self {
            // Truncated rather than allowed to widen the column: a long profile
            // name must not shift every row below it.
            Column::Profile => {
                format!("[{}]", truncate_right(&session.record.profile, COL_PROFILE - 2))
            }
            Column::Port => format!(":{}", session.record.port),
            Column::Size => session.record.size_bytes.map(human_size).unwrap_or(MISSING.into()),
            Column::Device => session.record.device.clone().unwrap_or(MISSING.into()),
            Column::Decode => rate_cell("tg", session, Phase::Decode),
            Column::Prefill => rate_cell("pp", session, Phase::Prefill),
            Column::Uptime => session.uptime_secs().map(format_uptime).unwrap_or(MISSING.into()),
        }
    }

    fn width(self) -> usize {
        match self {
            Column::Profile => COL_PROFILE,
            Column::Port => COL_PORT,
            Column::Size => COL_SIZE,
            Column::Device => COL_DEVICE,
            Column::Decode | Column::Prefill => COL_RATE,
            Column::Uptime => COL_UPTIME,
        }
    }

    /// Numbers read better right-aligned against the column that follows them.
    /// The rate cells align internally instead, so they are left as they are.
    fn pad(self, text: &str) -> String {
        let width = self.width();
        match self {
            Column::Size | Column::Uptime => pad_left(text, width),
            Column::Decode | Column::Prefill => pad_right(text, width),
            _ => pad_right(text, width),
        }
    }
}

/// A `tg`/`pp` cell: label, the figure right-aligned in a fixed field so the
/// digits line up down the pane, then the unit.
///
/// A session that has served nothing yet shows the shape of the number rather
/// than a blank, so a fresh server sits in the same columns as a busy one.
fn rate_cell(label: &str, session: &Session, phase: Phase) -> String {
    let value = session.throughput.rate(phase).map(format_rate).unwrap_or_else(|| {
        match phase {
            // Decode is reported to two decimals and prefill whole, so their
            // placeholders differ too — each is the shape of what will replace it.
            Phase::Decode => "--.--".into(),
            Phase::Prefill => "---".into(),
        }
    });
    format!("{label} {value:>RATE_FIGURE$} t/s")
}

/// How wide the name column is, and which columns follow it, in a pane of
/// `width` holding names up to `longest` terminal columns wide.
///
/// One answer for the whole list rather than one per row: rows that sized their
/// own names would stop lining up, and the alignment is the point.
fn row_plan(width: usize, longest: usize) -> (usize, Vec<Column>) {
    const ORDER: [Column; 7] = [
        Column::Profile,
        Column::Port,
        Column::Size,
        Column::Device,
        Column::Decode,
        Column::Prefill,
        Column::Uptime,
    ];
    let cost = |columns: &[Column]| -> usize {
        columns.iter().map(|column| column.width() + COL_GAP).sum()
    };

    let room = width.saturating_sub(2); // status glyph and its space
    // What the names in this pane actually need, within reason: giving up a
    // column to seat a name whole is worth it, and `MAX_NAME` is where it stops
    // being worth it.
    let wanted = longest.clamp(MIN_NAME, MAX_NAME);
    let mut columns = ORDER.to_vec();
    for dropped in DROP_ORDER {
        if room >= cost(&columns) + wanted + COL_GAP {
            break;
        }
        columns.retain(|column| *column != dropped);
    }

    // Capped at what the names need. The leftover used to go here, which on a
    // wide pane put sixty blank columns between the name and the profile and
    // left the row reading as two unrelated halves.
    let name = room.saturating_sub(cost(&columns) + COL_GAP).clamp(1, wanted.max(1));
    (name, columns)
}

/// One session row: status, model, then the columns the pane agreed on.
fn session_row(session: &Session, name_width: usize, columns: &[Column]) -> Line<'static> {
    let mut spans = vec![
        Span::styled(
            format!("{} ", session.status.glyph()),
            Style::default().fg(status_color(session.status)),
        ),
        Span::raw(pad_right(&truncate_right(&session.record.name, name_width), name_width)),
    ];
    for column in columns {
        spans.push(Span::styled(
            format!("{}{}", " ".repeat(COL_GAP), column.pad(&column.cell(session))),
            Style::default().fg(Color::DarkGray),
        ));
    }
    Line::from(spans)
}

/// Truncate to `max` terminal columns, keeping the head and marking the cut.
///
/// Columns, not `char`s: a CJK glyph or an emoji occupies two of them, so ten
/// of either would draw twice as wide as the field reserved for them and shove
/// every column after it out of line with the rows above and below.
fn truncate_right(text: &str, max: usize) -> String {
    if text.width() <= max {
        return text.to_string();
    }
    if max <= 1 {
        return "…".repeat(max);
    }
    // The ellipsis costs one column, and a wide glyph may not fit the last one
    // left — hence the running total rather than a `take`.
    let mut kept = String::new();
    let mut used = 0;
    for c in text.chars() {
        let width = c.width().unwrap_or(0);
        if used + width > max - 1 {
            break;
        }
        kept.push(c);
        used += width;
    }
    kept.push('…');
    kept
}

/// Pad `text` out to `width` terminal columns. `{:<width$}` counts `char`s,
/// which is the same mistake [`truncate_right`] used to make.
fn pad_right(text: &str, width: usize) -> String {
    format!("{text}{}", " ".repeat(width.saturating_sub(text.width())))
}

/// [`pad_right`] the other way round, for the columns that read better with
/// their digits against the column that follows them.
fn pad_left(text: &str, width: usize) -> String {
    format!("{}{text}", " ".repeat(width.saturating_sub(text.width())))
}

/// How far a session sits in from the runtime heading above it.
const SESSION_INDENT: usize = 2;

/// One drawn row of the session list.
#[derive(Debug, PartialEq)]
enum SessionListRow<'a> {
    /// A runtime's name, heading the sessions running under it. The cursor
    /// never lands here: it indexes sessions, not rows.
    Runtime(&'a str),
    /// The session at this index in the manager's list.
    Session(usize),
}

/// The session list as it is drawn: each runtime's name, then its sessions.
///
/// A heading is emitted wherever the runtime changes rather than being planned
/// from a set of runtimes, so a list that somehow arrived ungrouped still draws
/// truthfully — it repeats a heading instead of filing a session under the
/// wrong runtime. [`SessionManager`](crate::session::SessionManager) keeps the
/// list grouped so that does not arise.
fn session_list_rows(sessions: &[Session]) -> Vec<SessionListRow<'_>> {
    let mut rows = Vec::new();
    let mut runtime: Option<&str> = None;
    for (index, session) in sessions.iter().enumerate() {
        if runtime != Some(session.record.runtime.as_str()) {
            runtime = Some(&session.record.runtime);
            rows.push(SessionListRow::Runtime(&session.record.runtime));
        }
        rows.push(SessionListRow::Session(index));
    }
    rows
}

/// The drawn rows, and the one the selected session landed on.
fn session_items(
    sessions: &[Session],
    selected: Option<usize>,
    width: usize,
) -> (Vec<ListItem<'static>>, Option<usize>) {
    let selected = selected.filter(|index| *index < sessions.len());
    let rows = session_list_rows(sessions);
    let selected_row = selected
        .and_then(|index| rows.iter().position(|row| *row == SessionListRow::Session(index)));
    let longest = sessions.iter().map(|session| session.record.name.width()).max().unwrap_or(0);
    let (name_width, columns) = row_plan(width, longest);
    let indent = " ".repeat(SESSION_INDENT);
    let items = rows
        .iter()
        .map(|row| match row {
            SessionListRow::Runtime(name) => ListItem::new(Line::from(Span::styled(
                name.to_string(),
                Style::default().fg(ACCENT).bold(),
            ))),
            SessionListRow::Session(index) => {
                let mut spans = vec![Span::raw(indent.clone())];
                spans.extend(session_row(&sessions[*index], name_width, &columns).spans);
                ListItem::new(Line::from(spans))
            }
        })
        .collect();
    (items, selected_row)
}

fn render_session_list(
    frame: &mut Frame,
    area: Rect,
    sessions: &[Session],
    selected: Option<usize>,
    focused: bool,
) {
    // Two columns of border, two of selection marker, and the indent that sets
    // a session apart from the runtime heading it sits under.
    let width = area.width.saturating_sub(4 + SESSION_INDENT as u16) as usize;
    let (items, selected_row) = session_items(sessions, selected, width);
    let mut state = ListState::default();
    state.select(selected_row);
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
    let items = app.downloads.jobs.iter().map(|download| {
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
        .session_view
        .selection
        .selected()
        .and_then(|index| index.checked_sub(app.sessions.sessions.len()))
        .filter(|index| *index < app.downloads.jobs.len());
    let mut state = ListState::default();
    state.select(selected);
    let list = List::new(items)
        .block(pane_block("Downloads", selected.is_some()))
        .highlight_style(Style::default().fg(Color::Black).bg(ACCENT).add_modifier(Modifier::BOLD))
        .highlight_symbol("▌ ");
    frame.render_stateful_widget(list, area, &mut state);
}

/// A live tail of the selected session's log, in the column the Detail pane
/// otherwise holds.
///
/// The lines come from the ring buffer [`crate::session::SessionManager::refresh`]
/// already fills every tick, so the pane costs no reading of its own — the log
/// files reach tens of megabytes, and re-reading one each frame is what the
/// full-screen view does and can afford to, because nothing else is on screen.
///
/// It always shows the end of the log. `j`/`k` still move between sessions, so
/// there is no cursor to scroll it with; `L` opens the log where there is.
fn render_session_log(frame: &mut Frame, area: Rect, session: Option<&Session>) {
    let block = pane_block("Log", false);
    let Some(session) = session else {
        frame.render_widget(block, area);
        return;
    };
    let width = area.width.saturating_sub(2) as usize;
    let height = area.height.saturating_sub(2) as usize;

    // Wrapped here rather than by `Wrap`, because the pane shows the *last*
    // rows and only a row count we did ourselves says which those are.
    let mut rows: Vec<String> = Vec::new();
    for line in session.recent_log(height) {
        rows.extend(wrap_hard(line, width));
    }
    if rows.len() > height {
        rows = rows.split_off(rows.len() - height);
    }

    let text = if rows.is_empty() {
        Text::from(Line::from("(nothing logged yet)".dim()))
    } else {
        Text::from(rows.into_iter().map(Line::raw).collect::<Vec<_>>())
    };
    frame.render_widget(Paragraph::new(text).block(block), area);
}

/// Break `text` into rows of at most `width` terminal columns.
///
/// Hard-wrapped mid-word on purpose: a log line is mostly paths, flags, and
/// figures, and breaking those at spaces leaves a ragged column with no more
/// text on it.
fn wrap_hard(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }
    if text.width() <= width {
        return vec![text.to_string()];
    }
    let mut rows = Vec::new();
    let mut row = String::new();
    let mut used = 0;
    for c in text.chars() {
        let cell = c.width().unwrap_or(0);
        if used + cell > width {
            rows.push(std::mem::take(&mut row));
            used = 0;
        }
        row.push(c);
        used += cell;
    }
    if !row.is_empty() {
        rows.push(row);
    }
    rows
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

/// One `pp`/`tg` line for the Detail pane: the rate, and the request it came
/// from — the server's own summary of the last thing it did.
fn rate_line(label: &str, session: &Session, phase: Phase) -> Line<'static> {
    let Some(last) = session.throughput.last(phase) else {
        return kv(label, "—");
    };
    let Some(rate) = last.rate() else { return kv(label, "—") };
    Line::from(vec![
        Span::styled(format!("{label}: "), Style::default().fg(Color::DarkGray)),
        Span::raw(format!(
            "{} t/s  ({} tok in {:.2}s)",
            format_rate(rate),
            last.tokens,
            last.seconds
        )),
    ])
}

fn session_detail_lines(session: &Session) -> Text<'static> {
    let r = &session.record;
    let color = status_color(session.status);
    let uptime = session.uptime_secs().map(format_uptime).unwrap_or_else(|| "—".into());
    let mem = session.rss_bytes.map(human_size).unwrap_or_else(|| "—".into());
    let cpu = session.cpu_percent.map(|c| format!("{c:.0}%")).unwrap_or_else(|| "—".into());
    // The row sheds these two first when the pane narrows, so Detail is where
    // they have to remain readable.
    let size = r.size_bytes.map(human_size).unwrap_or_else(|| "—".into());
    let device = r.device.clone().unwrap_or_else(|| "—".into());

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
        kv("Size", &size),
        kv("Backend", &device),
        Line::raw(""),
        kv("PID", &r.pid.to_string()),
        kv("Port", &r.port.to_string()),
        kv("Uptime", &uptime),
        kv("Memory", &mem),
        kv("CPU", &cpu),
        Line::raw(""),
        Line::from("Throughput".bold()),
        rate_line("Prefill", session, Phase::Prefill),
        rate_line("Decode", session, Phase::Decode),
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
        .session_view
        .selection
        .selected()
        .and_then(|i| app.sessions.sessions.get(i))
        .map(|s| s.record.name.clone())
        .unwrap_or_default();
    let follow = if app.session_view.log_follow { "  [tailing]" } else { "" };
    let title = Line::from(vec![
        Span::styled(format!(" {ICON_LOG}  Logs — {name}"), Style::default().fg(ACCENT).bold()),
        Span::styled(follow.to_string(), Style::default().fg(Color::Green)),
    ]);
    frame.render_widget(Paragraph::new(title), header);

    let block = pane_block("Output", true);
    let inner_height = body.height.saturating_sub(2); // borders
    let total = app.session_view.log_lines.len() as u16;
    let max_scroll = total.saturating_sub(inner_height);
    let scroll = if app.session_view.log_follow {
        max_scroll
    } else {
        app.session_view.log_scroll.min(max_scroll)
    };
    app.session_view.log_scroll = scroll; // keep state clamped/in-sync

    let text = if app.session_view.log_lines.is_empty() {
        Text::from(Line::from("(log is empty)".dim()))
    } else {
        Text::from(
            app.session_view.log_lines.iter().map(|l| Line::raw(l.clone())).collect::<Vec<_>>(),
        )
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

    let width =
        message.lines.iter().map(|l| l.width()).max().unwrap_or(20).clamp(24, 88) as u16 + 4;
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

/// A destructive action's confirmation. Red-bordered rather than accented, and
/// the footer names the two answers explicitly: this is the one dialog where
/// dismissing by reflex must not be the same as agreeing.
fn render_confirm(frame: &mut Frame, area: Rect, confirm: &Confirm) {
    let mut lines: Vec<Line> = confirm.lines.iter().map(|line| Line::raw(line.clone())).collect();
    lines.push(Line::raw(""));
    lines.push(Line::from("y / Enter delete · any other key cancel".dim().italic()));

    // Terminal columns throughout, as ratatui's own wrapping measures them: a
    // model name in CJK is twice as wide as it is long, and counting `char`s
    // would size the popup to half the rows the question actually takes —
    // cutting off its tail, or the footer naming the two answers.
    let longest = confirm.lines.iter().map(|line| line.width()).max().unwrap_or(40);
    // Never wider than the terminal: a long model name wraps instead. The
    // preferred minimum yields to that ceiling rather than being clamped
    // against it — `clamp` panics when its floor exceeds its ceiling, which a
    // terminal under 48 columns would otherwise do on every `D`.
    let ceiling = area.width.saturating_sub(4).max(1);
    let width = (longest as u16 + 4).clamp(44.min(ceiling), ceiling);
    let inner = width.saturating_sub(4).max(1) as usize;
    let wrapped: usize =
        // `saturating_sub`: a blank spacer line occupies no rows beyond its
        // own, and `0 - 1` would underflow.
        confirm.lines.iter().map(|line| line.width().div_ceil(inner).saturating_sub(1)).sum();
    let height = ((lines.len() + wrapped) as u16 + 2).min(area.height);
    let popup = center(area, Constraint::Length(width), Constraint::Length(height));
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Red))
        .title(format!(" {} ", confirm.title));

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
        help_row("s", "sort online models / switch catalog grouping"),
        help_row("d", "download the selected model"),
        help_row("D", "delete the selected model from disk"),
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
        help_row("C", "chat in terminal (the runtime's interactive client)"),
        help_row("b", "benchmark selected model (runtimes that ship one)"),
        help_row("y", "yank command"),
        help_row("t", "session manager"),
        help_row("x / K", "stop / kill / cancel"),
        help_row("R", "restart / resume"),
        help_row("l / →", "session log beside the list"),
        help_row("L", "log full screen"),
        help_row("c", "copy endpoint"),
        Line::raw(""),
        Line::from("General".bold()),
        help_row("F5", "rescan / reload"),
        help_row("? / q", "help / quit"),
        Line::raw(""),
        Line::from("press ? or Esc to close".dim().italic()),
    ];

    let height = lines.len() as u16 + 2;
    // Sized to the widest row it actually has, rather than to a number that was
    // right when the rows were shorter: the longest description used to be cut
    // mid-word. Never wider than the terminal, which may be narrower still.
    let width = lines.iter().map(|line| line.width() as u16).max().unwrap_or(0);
    let popup =
        center(area, Constraint::Length((width + 4).min(area.width)), Constraint::Length(height));
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

/// Columns the help overlay's key column reserves.
///
/// The longest chord plus a gap. `{keys:<8}` left none at all for `e / Enter`
/// and `Home/End`, which ran straight into their own descriptions — and padded
/// in `char`s, which is not what `l / \u{2192}` occupies.
const HELP_KEY_WIDTH: usize = 11;

fn help_row<'a>(keys: &str, desc: &str) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("  {}", pad_right(keys, HELP_KEY_WIDTH)), Style::default().fg(ACCENT)),
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
        COL_GAP, ICON_CLOUD, ICON_DIRECTORY, MAX_NAME, MIN_COLS, MIN_ROWS, compact_count,
        download_progress, model_artifact_columns, model_icon, render_confirm, render_help,
        render_session_log, render_too_small, session_panes, truncate_download_name, wrap_hard,
    };
    use crate::app::Confirm;

    /// Regression: the confirmation sized itself to its content and ran off
    /// the side of the terminal.
    #[test]
    fn a_delete_confirmation_stays_inside_the_terminal() {
        let confirm = Confirm::preview(
            "Delete model",
            vec!["Remove Muse-Glimmer-30B-UD-Q4_K_XL.gguf (15.1 GB) from disk?".into()],
        );
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(60, 24)).unwrap();
        terminal.draw(|frame| render_confirm(frame, frame.area(), &confirm)).unwrap();
        let buffer = terminal.backend().buffer().clone();

        let rows: Vec<String> = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect();
        // Narrower than the question, so it wraps rather than overflowing, and
        // both halves are on screen.
        assert!(rows.iter().any(|row| row.contains("Remove Muse-Glimmer-30B-UD")));
        assert!(rows.iter().any(|row| row.contains("disk?")), "the wrapped tail stays on screen");
        assert!(rows.iter().any(|row| row.contains("y / Enter delete")));
    }

    /// Regression: the preferred 44-column minimum was clamped against a
    /// narrower ceiling, and `clamp` panics when its floor exceeds its ceiling
    /// — so `D` in a terminal under 48 columns crashed llmctl.
    #[test]
    fn a_delete_confirmation_survives_a_very_narrow_terminal() {
        let confirm = Confirm::preview(
            "Delete model",
            vec![
                "Remove Muse-Glimmer-30B-UD-Q4_K_XL.gguf (15.1 GB) from disk?".into(),
                String::new(),
            ],
        );
        for width in [1_u16, 8, 20, 40, 47, 48] {
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, 12)).unwrap();
            terminal
                .draw(|frame| render_confirm(frame, frame.area(), &confirm))
                .unwrap_or_else(|err| panic!("{width} columns: {err}"));
        }
    }

    /// Regression: the popup was sized in `char`s while ratatui wraps in
    /// terminal columns, so a model name in CJK took twice the rows the height
    /// allowed for — and the footer naming the two answers fell off the bottom
    /// of the one dialog where knowing the answers matters.
    #[test]
    fn a_confirmation_leaves_room_for_a_question_in_wide_characters() {
        let confirm = Confirm::preview(
            "Delete model",
            vec!["移除 通義千問-32B-指令-深度思考-Q4_K_XL.gguf (15.1 GB) 從磁碟?".into()],
        );
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(60, 16)).unwrap();
        terminal.draw(|frame| render_confirm(frame, frame.area(), &confirm)).unwrap();

        let rows: Vec<String> = terminal
            .backend()
            .buffer()
            .content
            .chunks(60)
            .map(|row| row.iter().map(|cell| cell.symbol()).collect())
            .collect();
        let screen = rows.join("\n");
        // A wide glyph occupies two cells, the second of which holds nothing —
        // so the tail of the question is looked for with the gaps taken out.
        let dense: String = screen.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(screen.contains("y / Enter delete"), "the footer was cut off:\n{screen}");
        assert!(dense.contains("從磁碟?"), "the question was cut off:\n{screen}");
    }

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

    /// A minimal local model leaf that tests reshape as they need.
    fn sample_model() -> crate::domain::Model {
        crate::domain::Model {
            entry: crate::domain::CatalogEntry::Model(crate::domain::ModelSource::Gguf {
                remote: None,
            }),
            id: String::new(),
            name: String::new(),
            // Non-empty so this reads as a model leaf, not a catalog folder.
            path: std::path::PathBuf::from("/models/sample.gguf"),
            shard_paths: Vec::new(),
            mtp_path: None,
            dflash_path: None,
            dflash_block_size: None,
            projector_path: None,
            has_mtp: false,
            catalog_path: Vec::new(),
            catalog_dir: std::path::PathBuf::new(),
            size_bytes: 0,
            quantization: None,
            architecture: None,
            context_length: None,
            modified: None,
            has_chat_template: false,
            runtime: crate::runtime::llama_cpp::NAME.into(),
        }
    }

    #[test]
    fn artifact_columns_show_quant_size_and_filename() {
        let mut model = sample_model();
        model.name = "Qwen-AgentWorld-35B-A3B-UD-Q4_K_M.gguf".into();
        model.catalog_path = vec![model.name.clone()];
        model.size_bytes = 20_600_000_000;
        model.quantization = Some("Q4_K_M".into());
        model.entry = crate::domain::CatalogEntry::Model(crate::domain::ModelSource::Gguf {
            remote: Some(crate::domain::RemoteModel {
                repo: "owner/repo".into(),
                revision: None,
                file: Some(model.name.clone()),
                blobs: Vec::new(),
                mtp_file: None,
                dflash_file: None,
                projector_file: None,
                downloads: 0,
                likes: 0,
                gated: false,
            }),
        });

        assert_eq!(
            model_artifact_columns(&model).unwrap(),
            ("Q4_K_M       20.6 GB  ".into(), "Qwen-AgentWorld-35B-A3B-UD-Q4_K_M.gguf".into())
        );

        model.entry =
            crate::domain::CatalogEntry::Model(crate::domain::ModelSource::Gguf { remote: None });
        assert_eq!(
            model_artifact_columns(&model).unwrap(),
            ("Q4_K_M       20.6 GB  ".into(), "Qwen-AgentWorld-35B-A3B-UD-Q4_K_M.gguf".into())
        );
    }

    /// Regression: a megabyte-scale size is one character wider than a
    /// gigabyte-scale one, so too narrow a size column pushed the name of any
    /// sub-gigabyte model out of alignment with its neighbours.
    #[test]
    fn model_names_stay_aligned_across_size_units() {
        let row = |bytes: u64, name: &str| {
            let mut model = sample_model();
            model.name = name.into();
            model.catalog_path = vec![name.into()];
            model.size_bytes = bytes;
            model.quantization = Some("Q4_1".into());
            model_artifact_columns(&model).unwrap()
        };

        let rows = [
            row(14_000_000_000, "gpt-oss:20b"),
            row(960_000_000, "lfm2.5-tk:1.2b"),
            row(3_100_000_000, "nanbeige4.1:3b"),
            row(999, "tiny.gguf"),
        ];

        // Every name starts at the same column, whatever the unit.
        let widths: Vec<usize> = rows.iter().map(|(meta, _)| meta.chars().count()).collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "size column widths disagree: {widths:?}");
        assert_eq!(rows[1].0, "Q4_1        960.0 MB  ");
        assert_eq!(rows[0].0, "Q4_1         14.0 GB  ");
    }

    #[test]
    fn online_catalog_nodes_use_cloud_icons() {
        let mut model = sample_model();
        model.path = std::path::PathBuf::new();
        model.entry = crate::domain::CatalogEntry::Directory { repository: None };

        model.catalog_path = vec!["online".into()];
        assert_eq!(model_icon(&model), ICON_CLOUD);

        model.catalog_path = vec!["online".into(), "huggingface".into()];
        assert_eq!(model_icon(&model), ICON_CLOUD);

        model.catalog_path = vec!["local-models".into()];
        assert_eq!(model_icon(&model), ICON_DIRECTORY);
    }

    fn probe_session(name: &str, rates: bool) -> crate::session::Session {
        use crate::session::record::SessionRecord;
        use crate::session::throughput::{Phase, Sample, Throughput};
        let mut throughput = Throughput::default();
        if rates {
            throughput.record(Sample { phase: Phase::Prefill, tokens: 3266, seconds: 2.292 });
            throughput.record(Sample { phase: Phase::Decode, tokens: 174, seconds: 2.569 });
        }
        crate::session::Session::probe(
            SessionRecord {
                id: "1".into(),
                name: name.into(),
                runtime: "llama.cpp".into(),
                model: "m".into(),
                model_path: "m".into(),
                profile: "inquisitor".into(),
                size_bytes: Some(12_100_000_000),
                device: Some("ROCm".into()),
                pid: 1,
                host: "127.0.0.1".into(),
                port: 8001,
                command: vec![],
                health_path: "/health".into(),
                log_file: Default::default(),
                download: None,
                started_unix: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    - 9_240,
            },
            crate::session::SessionStatus::Running,
            throughput,
        )
    }

    fn probe_session_of(runtime: &str, name: &str) -> crate::session::Session {
        let mut session = probe_session(name, true);
        session.record.runtime = runtime.into();
        session
    }

    /// Each runtime is named once, with its sessions beneath it, and the
    /// headings are rows the cursor's session indices have to be mapped past.
    #[test]
    fn sessions_are_drawn_under_their_runtime() {
        use super::SessionListRow::{Runtime, Session};

        let sessions = vec![
            probe_session_of("llama.cpp", "muse-glimmer-30b-q8_0"),
            probe_session_of("llama.cpp", "qwen3-8b-q4_k_m"),
            probe_session_of("FastFlowLM", "gpt-oss-20b"),
        ];
        assert_eq!(
            super::session_list_rows(&sessions),
            vec![Runtime("llama.cpp"), Session(0), Session(1), Runtime("FastFlowLM"), Session(2),]
        );

        // An empty list heads nothing.
        assert!(super::session_list_rows(&[]).is_empty());
    }

    /// The drawn pane: a heading at the left margin, its sessions stepped in
    /// under it, and the cursor on the session it was pointed at rather than on
    /// whatever row that index happens to be.
    #[test]
    fn the_session_pane_heads_each_group_and_indents_its_sessions() {
        let sessions = vec![
            probe_session_of("llama.cpp", "muse-glimmer-30b-q8_0"),
            probe_session_of("FastFlowLM", "gpt-oss-20b"),
        ];
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(70, 8)).unwrap();
        terminal
            .draw(|frame| super::render_session_list(frame, frame.area(), &sessions, Some(1), true))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let rows: Vec<String> = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect();

        // Inside the border, the heading starts where the selection marker
        // would; the session under it starts two further in.
        let heading = rows.iter().position(|row| row.contains("llama.cpp")).expect("heading");
        let session = rows.iter().position(|row| row.contains("muse-glimmer")).expect("session");
        assert_eq!(session, heading + 1);
        // In columns, not bytes: the border and the status glyph are multibyte.
        let column = |row: &str, text: &str| row[..row.find(text).unwrap()].chars().count();
        assert_eq!(
            column(&rows[session], "muse-glimmer"),
            column(&rows[heading], "llama.cpp") + super::SESSION_INDENT + 2, // + status glyph
        );
        // Both runtimes are named, each once.
        assert_eq!(rows.iter().filter(|row| row.contains("llama.cpp")).count(), 1);
        assert_eq!(rows.iter().filter(|row| row.contains("FastFlowLM")).count(), 1);
        // The cursor is on the second session, three rows below the first
        // heading — the row index the naive mapping would have got wrong.
        assert!(rows[heading + 3].contains("▌"), "cursor row: {:?}", rows[heading + 3]);
    }

    /// The pane is a tail: it shows the end of the log, and a line too long for
    /// the column wraps inside it rather than over the border beside it.
    #[test]
    fn the_log_pane_shows_the_end_of_the_log_wrapped_to_its_column() {
        let session = probe_session("muse", true).with_log(&[
            "oldest line, long gone",
            "slot launch_slot_: id  3 | task 0 | processing task",
            "slot print_timing: id  3 | task 0 | prompt eval time = 712.25 ms / 67 tokens",
            "slot      release: id  3 | task 0 | stop processing",
        ]);
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(40, 8)).unwrap();
        terminal.draw(|frame| render_session_log(frame, frame.area(), Some(&session))).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let rows: Vec<String> = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect();

        assert!(rows[0].contains("Log"), "the pane is titled");
        // Wrapping breaks lines mid-word, so read the pane as the text it
        // flows rather than row by row.
        let flowed: String =
            rows[1..rows.len() - 1].iter().map(|row| row.trim_matches('\u{2502}')).collect();
        // The newest line is on screen and the oldest has scrolled off.
        assert!(flowed.contains("stop processing"), "{rows:#?}");
        assert!(!flowed.contains("oldest line"), "{rows:#?}");
        // Every row still ends in the border it started with.
        for row in &rows[1..rows.len() - 1] {
            assert!(row.starts_with('\u{2502}') && row.ends_with('\u{2502}'), "{row:?}");
        }
    }

    #[test]
    fn an_unlogged_selection_leaves_an_empty_pane() {
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(30, 5)).unwrap();
        terminal.draw(|frame| render_session_log(frame, frame.area(), None)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let screen: String = (0..buffer.area.height)
            .flat_map(|y| (0..buffer.area.width).map(move |x| (x, y)))
            .map(|cell| buffer[cell].symbol().to_string())
            .collect();
        assert!(screen.contains("Log"), "the pane keeps its frame");
    }

    /// Wrapping is measured in terminal columns, so a row of wide characters
    /// stops at the column the pane ends at rather than one past it.
    #[test]
    fn wrapping_counts_columns_and_never_overruns_the_width() {
        use unicode_width::UnicodeWidthStr;

        assert_eq!(wrap_hard("short", 10), vec!["short"]);
        assert_eq!(wrap_hard("exactly-10", 10), vec!["exactly-10"]);
        assert_eq!(wrap_hard("abcdefghijk", 10), vec!["abcdefghij", "k"]);
        assert!(wrap_hard("anything", 0).is_empty(), "a pane with no room shows nothing");

        // Two columns per glyph: five fit in ten, and the odd width leaves one
        // column unused rather than splitting a character across rows.
        for width in [4, 5, 10, 11] {
            for row in wrap_hard("從磁碟移除模型檔案嗎", width) {
                assert!(row.width() <= width, "{row:?} overran {width}");
            }
        }
    }

    /// A pane too narrow to read is worse than no pane: it takes width from the
    /// list, which is the one thing on this screen that has to stay legible.
    #[test]
    fn a_narrow_terminal_gives_the_whole_width_to_the_jobs_list() {
        use ratatui::layout::Rect;
        let body = |width| Rect::new(0, 1, width, 20);

        for width in [40_u16, 60, 80, 100] {
            let (jobs, detail) = session_panes(body(width));
            assert!(detail.is_none(), "{width} columns left room for a pane");
            assert_eq!(jobs.width, width, "the list takes what the pane gave up");
        }

        for width in [120_u16, 150, 200] {
            let (jobs, detail) = session_panes(body(width));
            let detail = detail.expect("a pane at {width} columns");
            assert!(detail.width >= super::MIN_DETAIL_WIDTH, "{width}: {detail:?}");
            assert_eq!(jobs.width + detail.width, width, "the split spends every column");
        }

        // Whatever width the pane first pays for itself at, it appears once
        // and stays: no width below it has a pane, none above it lacks one.
        let first =
            (1_u16..240).find(|w| session_panes(body(*w)).1.is_some()).expect("a wide enough body");
        assert!((1..first).all(|w| session_panes(body(w)).1.is_none()), "a pane below {first}");
        assert!(
            (first..240).all(|w| session_panes(body(w)).1.is_some()),
            "the pane came and went above {first}"
        );
    }

    /// Below the floor the interface is replaced by what it needs, and the
    /// replacement has to survive sizes where even a bordered popup would not.
    #[test]
    fn a_terminal_under_the_floor_is_told_what_it_needs() {
        use ratatui::layout::Rect;

        let screen = |width: u16, height: u16| {
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| render_too_small(frame, Rect::new(0, 0, width, height))).unwrap();
            let buffer = terminal.backend().buffer().clone();
            (0..buffer.area.height)
                .map(|y| {
                    (0..buffer.area.width)
                        .map(|x| buffer[(x, y)].symbol().to_string())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        // Just under the floor, both figures are on screen: what is needed and
        // what there is, so the fix is a drag of the window edge.
        let just_under = screen(MIN_COLS - 1, MIN_ROWS);
        assert!(just_under.contains("terminal too small"), "{just_under}");
        assert!(just_under.contains(&format!("{MIN_COLS}\u{d7}{MIN_ROWS} needed")), "{just_under}");
        assert!(just_under.contains(&format!("{}\u{d7}{MIN_ROWS} now", MIN_COLS - 1)));
        assert!(just_under.contains("q quit"), "the way out is on screen");

        // Down to a terminal with room for nothing, drawing it is still safe
        // and what fits is still the message.
        for (width, height) in [(40, 10), (20, 3), (12, 2), (4, 1), (1, 1)] {
            let tiny = screen(width, height);
            assert!(
                tiny.lines().all(|row| row.chars().count() == width as usize),
                "{width}x{height} drew outside itself: {tiny:?}"
            );
        }
        assert!(screen(20, 3).contains("terminal too"), "the first line survives a squeeze");
    }

    /// Regression: the key column was padded to eight `char`s, so the two
    /// chords that reach it — `e / Enter` and `Home/End` — were printed with no
    /// gap at all and ran into their own descriptions.
    #[test]
    fn every_help_key_is_separated_from_what_it_does() {
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(110, 46)).unwrap();
        terminal.draw(|frame| render_help(frame, frame.area())).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let rows: Vec<String> = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect();
        let screen = rows.join("\n");

        assert!(screen.contains("e / Enter  edit / cycle / pick value"), "{screen}");
        assert!(screen.contains("Home/End   min / max"), "{screen}");
        // The longest description used to be cut mid-word by a fixed width.
        assert!(screen.contains("sort online models / switch catalog grouping"), "{screen}");
    }

    /// Regression: the name column took every column the row had left over, so
    /// a wide pane put sixty blanks between the name and the profile — and a
    /// middling one truncated a name it had columns to spare for, because the
    /// drop rule only ever asked for `MIN_NAME`.
    #[test]
    fn the_name_column_takes_what_the_names_need_and_no_more() {
        use unicode_width::UnicodeWidthStr;

        let sessions = [probe_session("muse-glimmer-30b-q8_0", true)];
        let name = "muse-glimmer-30b-q8_0";

        // Wide pane: the name is whole and the columns follow it directly,
        // rather than after a run of padding.
        let row = &rows_text(&sessions, 200)[0];
        let gap = row.find("[inquisitor]").expect("the profile column") - row.find(name).unwrap();
        assert_eq!(gap, name.width() + COL_GAP, "a gulf opened up: {row:?}");

        // Middling pane: a column is given up to seat the name whole. It used
        // to keep every column and cut the name to twenty.
        let row = &rows_text(&sessions, 106)[0];
        assert!(row.contains(name), "the name was cut with columns to spare: {row:?}");

        // The cap holds: a name past `MAX_NAME` is truncated rather than
        // pushing every column off the row.
        let long = "a-very-long-model-name-that-keeps-going-and-going";
        let row = &rows_text(&[probe_session(long, true)], 200)[0];
        assert!(row.contains('…'), "{row:?}");
        // In columns, not bytes: the status glyph and the ellipsis are both
        // multibyte and neither is more than one column wide.
        let at = row[..row.find("[inquisitor]").expect("the profile column")].width();
        assert_eq!(at, 2 + MAX_NAME + COL_GAP, "the name column outgrew its cap: {row:?}");

        // Narrow panes still shed columns, and the name still gets `MIN_NAME`.
        let row = &rows_text(&sessions, 60)[0];
        assert!(!row.contains("[inquisitor]"), "a narrow pane kept every column: {row:?}");
        assert!(row.width() <= 60, "{row:?}");
    }

    /// The rows a pane of `width` draws for these sessions.
    ///
    /// One plan for the set, the way the list does it: rows that sized their
    /// own name columns would not line up, which several tests here are about.
    fn rows_text(sessions: &[crate::session::Session], width: usize) -> Vec<String> {
        use unicode_width::UnicodeWidthStr;
        let longest = sessions.iter().map(|session| session.record.name.width()).max().unwrap_or(0);
        let (name_width, columns) = super::row_plan(width, longest);
        sessions
            .iter()
            .map(|session| {
                super::session_row(session, name_width, &columns)
                    .spans
                    .iter()
                    .map(|span| span.content.to_string())
                    .collect()
            })
            .collect()
    }

    fn row_text(session: &crate::session::Session, width: usize) -> String {
        rows_text(std::slice::from_ref(session), width).remove(0)
    }

    /// The point of the columns: they line up regardless of name length.
    #[test]
    fn session_columns_align_across_rows() {
        // A pane wide enough for every column. The last row has served
        // nothing: a session without figures must still line up with one that
        // has them, which is why no column is ever omitted.
        let width = 120;
        let sessions: Vec<_> = [("a", true), ("gpt-oss-20b-q8_0", true), ("fresh", false)]
            .iter()
            .map(|(name, rates)| probe_session(name, *rates))
            .collect();
        let rows = rows_text(&sessions, width);

        for row in &rows {
            assert_eq!(row.chars().count(), rows[0].chars().count(), "ragged row: {row:?}");
        }
        // Same starting column for every cell, whatever the name did. Counted
        // in characters, not bytes: the glyph and the ellipsis are multibyte.
        // Anchored on the labels, since the last row's figures are placeholders
        // — that they still start in the same column is exactly the point.
        for cell in ["[inquisitor]", ":8001", "11.3 GB", "ROCm", "tg ", "pp ", "2h 34m"] {
            let columns: Vec<Option<usize>> = rows
                .iter()
                .map(|row| row.find(cell).map(|byte| row[..byte].chars().count()))
                .collect();
            assert!(columns.iter().all(|at| at.is_some()), "{cell} missing: {rows:#?}");
            assert!(columns.iter().all(|at| *at == columns[0]), "{cell} drifts: {columns:?}");
        }
    }

    /// Regression: cells were measured and padded in `char`s. A CJK profile
    /// name of ten characters occupies twenty terminal columns, so it drew at
    /// twice the width of the field reserved for it and shoved every column
    /// after it out of line with the rows above and below.
    #[test]
    fn wide_characters_are_measured_in_terminal_columns() {
        use unicode_width::UnicodeWidthStr;

        let width = 120;
        let mut session = probe_session("通義千問-32B-指令", true);
        // Ten characters, and twice that many columns.
        session.record.profile = "深度思考模式一二三四".into();
        let rows = rows_text(&[probe_session("gpt-oss-20b-q8_0", true), session], width);
        let (plain, wide) = (rows[0].clone(), rows[1].clone());

        assert_eq!(wide.width(), plain.width(), "the wide row draws a different width");
        // The profile was cut down to its field rather than allowed to overrun.
        assert!(wide.contains('…'), "{wide:?}");
        // And every column after it starts where it does on an ASCII row.
        for cell in [":8001", "11.3 GB", "ROCm", "tg ", "pp ", "2h 34m"] {
            let at = |row: &str| row.find(cell).map(|byte| row[..byte].width());
            assert_eq!(at(&wide), at(&plain), "{cell} drifts: {wide:?}");
        }
    }

    /// The requested order, left to right.
    #[test]
    fn session_columns_are_in_the_documented_order() {
        let row = row_text(&probe_session("gpt-oss-20b-q8_0", true), 120);
        let order = [
            "gpt-oss-20b-q8_0",
            "[inquisitor]",
            ":8001",
            "11.3 GB",
            "ROCm",
            "tg ",
            "pp ",
            "2h 34m",
        ];
        let mut at = 0;
        for cell in order {
            let found =
                row[at..].find(cell).unwrap_or_else(|| panic!("{cell} out of order in {row:?}"));
            at += found + cell.len();
        }
    }

    /// A narrow pane sheds columns by worth, not by position: the size and the
    /// backend go before the uptime, and the decode rate outlives them all.
    #[test]
    fn narrow_panes_shed_the_least_useful_columns_first() {
        let session = probe_session("gpt-oss-20b-q8_0", true);
        let kept = |width: usize| {
            let row = row_text(&session, width);
            assert!(row.chars().count() <= width, "{width}: overflowed with {row:?}");
            row
        };
        assert!(kept(120).contains("11.3 GB"), "everything fits on a wide pane");
        // Narrower: the size and the backend go before the uptime does.
        let tight = kept(80);
        assert!(!tight.contains("11.3 GB") && !tight.contains("ROCm"), "{tight}");
        assert!(tight.contains("2h 34m") && tight.contains("pp "), "{tight}");
        // Narrower still: only the decode rate survives beside port and uptime.
        let tighter = kept(52);
        assert!(!tighter.contains("pp "), "{tighter}");
        assert!(tighter.contains("tg ") && tighter.contains(":8001"), "{tighter}");
        // Even absurdly narrow it renders something rather than panicking.
        assert!(!row_text(&session, 4).is_empty());
    }

    /// A session that has served nothing shows the shape of the figures rather
    /// than nothing at all, so its row occupies the same columns as the others.
    #[test]
    fn a_session_with_no_requests_shows_placeholder_rates() {
        let row = row_text(&probe_session("fresh", false), 120);
        assert!(row.contains("tg --.-- t/s"), "{row}");
        assert!(row.contains("pp   --- t/s"), "{row}");
    }

    /// An older session record carries no size or backend; those cells hold a
    /// dash rather than collapsing and dragging the row out of line.
    #[test]
    fn a_record_without_size_or_backend_keeps_its_columns() {
        let mut session = probe_session("legacy", true);
        session.record.size_bytes = None;
        session.record.device = None;
        let row = row_text(&session, 120);
        let full = row_text(&probe_session("legacy", true), 120);
        assert_eq!(row.chars().count(), full.chars().count(), "{row:?}");
        assert_eq!(row.matches('—').count(), 2, "one dash for each missing cell: {row}");
    }
}
