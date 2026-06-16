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
use ratatui::widgets::{Block, BorderType, Borders, Clear, List, ListItem, Paragraph, Wrap};

use crate::app::{App, Pane};

const ACCENT: Color = Color::Yellow;

// Nerd-font glyphs (Yazi-style), written as escapes so the codepoints survive
// in source regardless of editor/transport. Require a Nerd Font in the terminal.
const ICON_RUNTIME: &str = "\u{f085}"; // cogs
const ICON_MODEL: &str = "\u{f1b2}"; // cube
const ICON_PROFILE: &str = "\u{f02e}"; // bookmark
const ICON_OPTION: &str = "\u{f1de}"; // sliders
const ICON_ROOT: &str = "\u{f015}"; // home

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
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(2),
    ])
    .areas(frame.area());

    // Parent | Current | Preview.
    let [parent, current, preview] = Layout::horizontal([
        Constraint::Percentage(22),
        Constraint::Percentage(38),
        Constraint::Percentage(40),
    ])
    .areas(body);

    render_header(frame, header, app);

    // Parent column: the level above the current one (root is virtual).
    match app.focus {
        Pane::Runtime => render_root(frame, parent),
        other => render_list(frame, parent, app, other.prev(), Role::Parent),
    }

    // Current column: the focused level.
    render_list(frame, current, app, app.focus, Role::Current);

    // Preview column: children of the hovered item, or the leaf detail.
    match app.focus {
        Pane::Runtime => render_list(frame, preview, app, Pane::Model, Role::Preview),
        Pane::Model => render_list(frame, preview, app, Pane::Profile, Role::Preview),
        Pane::Profile => render_list(frame, preview, app, Pane::Options, Role::Preview),
        Pane::Options => render_option_detail(frame, preview, app),
    }

    render_footer(frame, footer, app);

    if app.show_help {
        render_help(frame, frame.area());
    }
}

/// Render one level's list into a column, styled for its role.
fn render_list(frame: &mut Frame, area: Rect, app: &mut App, level: Pane, role: Role) {
    let focused = role == Role::Current;
    let block = pane_block(level.title(), focused);

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
            .map(|m| ListItem::new(format!("{icon}  {}", m.name)))
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
        let list = List::new(items).block(block).style(Style::default().add_modifier(Modifier::DIM));
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
    let line = Line::from(vec![
        Span::styled(" / ", Style::default().fg(Color::DarkGray)),
        Span::styled(crumbs, Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let (primary, secondary) = app.status();
    let [top, bottom] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);

    // Line 1: the locator/path. Left-truncate (keeping the tail) if too wide.
    let path = truncate_left(&primary, top.width.saturating_sub(1) as usize);
    frame.render_widget(Paragraph::new(Line::from(format!(" {path}")).dim()), top);

    // Line 2: metadata on the left, key hints on the right.
    let keys = Line::from(vec![
        Span::styled("hjkl ", Style::default().fg(ACCENT)),
        Span::raw("nav  "),
        Span::styled("? ", Style::default().fg(ACCENT)),
        Span::raw("help  "),
        Span::styled("q ", Style::default().fg(ACCENT)),
        Span::raw("quit "),
    ])
    .right_aligned();
    let [left, right] =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(26)]).areas(bottom);
    frame.render_widget(Paragraph::new(Line::from(format!(" {secondary}")).dim()), left);
    frame.render_widget(Paragraph::new(keys), right);
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

fn render_help(frame: &mut Frame, area: Rect) {
    let lines = vec![
        Line::from("llmctl — keybindings".bold().fg(ACCENT)),
        Line::raw(""),
        help_row("j / k", "move down / up"),
        help_row("l / →", "drill into selection"),
        help_row("h / ←", "back up a level"),
        help_row("g / G", "first / last"),
        help_row("F5", "rescan models"),
        Line::raw(""),
        help_row("?", "toggle this help"),
        help_row("q", "quit"),
        Line::raw(""),
        Line::from("(more shortcuts arrive in later phases)".dim()),
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

// --- helpers ---------------------------------------------------------------

fn pane_block(title: &str, focused: bool) -> Block<'static> {
    let border_style = if focused {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let title = if focused {
        Span::styled(
            format!(" {title} "),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )
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
