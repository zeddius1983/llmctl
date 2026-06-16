//! ratatui rendering for the five-pane main screen and the help overlay.

use ratatui::Frame;
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Clear, List, ListItem, Paragraph, Wrap};

use crate::app::{App, Pane};
use crate::domain::human_size;

const ACCENT: Color = Color::Yellow;

pub fn draw(frame: &mut Frame, app: &mut App) {
    let [main, footer] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(frame.area());

    // Runtime · Model · Profile · Options · Info (Info widest, far right).
    let cols = Layout::horizontal([
        Constraint::Percentage(15),
        Constraint::Percentage(20),
        Constraint::Percentage(18),
        Constraint::Percentage(17),
        Constraint::Percentage(30),
    ])
    .split(main);

    render_list_pane(frame, cols[0], Pane::Runtime, app);
    render_list_pane(frame, cols[1], Pane::Model, app);
    render_list_pane(frame, cols[2], Pane::Profile, app);
    render_list_pane(frame, cols[3], Pane::Options, app);
    render_info_pane(frame, cols[4], app);

    render_footer(frame, footer, app);

    if app.show_help {
        render_help(frame, frame.area());
    }
}

/// A bordered, focus-aware list. The focused pane gets an accent border and a
/// reversed selection; unfocused panes keep a dim selection marker so the
/// cursor position stays visible.
fn render_list_pane(frame: &mut Frame, area: Rect, pane: Pane, app: &mut App) {
    let focused = app.focus == pane;

    // Miller-columns: panes beyond the preview level are hidden until drilled in.
    if !app.is_revealed(pane) {
        frame.render_widget(pane_block(pane.title(), false), area);
        return;
    }

    let (items, state): (Vec<ListItem>, _) = match pane {
        Pane::Runtime => (
            app.runtimes.items.iter().map(|r| ListItem::new(r.name.clone())).collect(),
            &mut app.runtimes.state,
        ),
        Pane::Model => (
            app.models.items.iter().map(|m| ListItem::new(m.name.clone())).collect(),
            &mut app.models.state,
        ),
        Pane::Profile => (
            app.profiles
                .items
                .iter()
                .map(|p| {
                    let star = if p.favorite { "★ " } else { "" };
                    ListItem::new(format!("{star}{}", p.name))
                })
                .collect(),
            &mut app.profiles.state,
        ),
        Pane::Options => (
            app.options
                .items
                .iter()
                .map(|o| ListItem::new(format!("{}: {}", o.key, o.value)))
                .collect(),
            &mut app.options.state,
        ),
    };

    let block = pane_block(pane.title(), focused);
    let highlight = if focused {
        Style::default().fg(Color::Black).bg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().add_modifier(Modifier::DIM)
    };

    let list = List::new(items)
        .block(block)
        .highlight_style(highlight)
        .highlight_symbol(if focused { "▌ " } else { "  " });

    frame.render_stateful_widget(list, area, state);
}

/// The always-visible Info pane previews the focused pane's selection.
fn render_info_pane(frame: &mut Frame, area: Rect, app: &App) {
    let block = pane_block("Info", false);
    let text = match app.focus {
        Pane::Runtime => app
            .runtimes
            .selected()
            .map(|r| {
                Text::from(vec![
                    kv("Name", &r.name),
                    kv("Description", &r.description),
                    kv("Version", r.version.as_deref().unwrap_or("(not detected)")),
                    kv(
                        "Executable",
                        &r.binary_path
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "(not detected)".into()),
                    ),
                    kv("Formats", &r.formats_label()),
                    Line::raw(""),
                    Line::from("llama-server --help".italic().dim()),
                    Line::from("(captured & cached — Phase 1)".dim()),
                ])
            })
            .unwrap_or_else(empty_preview),
        Pane::Model => app
            .models
            .selected()
            .map(|m| {
                Text::from(vec![
                    kv("Name", &m.name),
                    kv("Path", &m.path.display().to_string()),
                    kv("Size", &human_size(m.size_bytes)),
                    kv("Architecture", m.architecture.as_deref().unwrap_or("(unknown)")),
                    kv("Quantization", m.quantization.as_deref().unwrap_or("(unknown)")),
                ])
            })
            .unwrap_or_else(empty_preview),
        Pane::Profile => {
            // Profile preview shows resolved option values (stubbed in Phase 0).
            let mut lines = Vec::new();
            if let Some(p) = app.profiles.selected() {
                lines.push(kv("Profile", &p.name));
                lines.push(Line::raw(""));
            }
            for o in &app.options.items {
                lines.push(kv(&o.key, &o.value));
            }
            Text::from(lines)
        }
        Pane::Options => app
            .options
            .selected()
            .map(|o| {
                Text::from(vec![
                    Line::from(o.key.clone().bold()),
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
            .unwrap_or_else(empty_preview),
    };

    let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
    frame.render_widget(para, area);
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let hint = Line::from(vec![
        Span::styled(" hjkl ", Style::default().fg(ACCENT)),
        Span::raw("nav  "),
        Span::styled("g/G ", Style::default().fg(ACCENT)),
        Span::raw("first/last  "),
        Span::styled("? ", Style::default().fg(ACCENT)),
        Span::raw("help  "),
        Span::styled("q ", Style::default().fg(ACCENT)),
        Span::raw("quit"),
    ]);
    // Breadcrumb of the committed path (Runtime ▸ Model ▸ …).
    let crumbs = app.breadcrumb().join(" ▸ ");
    let path = Line::from(format!(" {crumbs} "))
        .right_aligned()
        .style(Style::default().fg(ACCENT));

    let crumb_width = (crumbs.chars().count() as u16 + 2).min(area.width.saturating_sub(20));
    let [left, right] =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(crumb_width)]).areas(area);
    frame.render_widget(Paragraph::new(hint), left);
    frame.render_widget(Paragraph::new(path), right);
}

fn render_help(frame: &mut Frame, area: Rect) {
    let lines = vec![
        Line::from("llmctl — keybindings".bold().fg(ACCENT)),
        Line::raw(""),
        help_row("j / k", "move down / up"),
        help_row("h / l", "back / enter pane"),
        help_row("g / G", "first / last"),
        help_row("Enter", "enter next pane"),
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
        Span::raw(format!(" {title} "))
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

fn empty_preview<'a>() -> Text<'a> {
    Text::from(Line::from("(nothing selected)".dim()))
}

/// Center a rect of the given width/height constraints within `area`.
fn center(area: Rect, horizontal: Constraint, vertical: Constraint) -> Rect {
    let [h] = Layout::horizontal([horizontal]).flex(Flex::Center).areas(area);
    let [v] = Layout::vertical([vertical]).flex(Flex::Center).areas(h);
    v
}
