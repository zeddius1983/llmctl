//! The Session Manager's Monitor: the selected session's live figures beside
//! the list, in the column the Detail pane used to hold.
//!
//! Tiles pair a current value with a minute of history, then bars show how
//! full the machine is, then the requests the server last reported. Everything
//! is built as text — sparklines and bars are block characters — so the layout
//! is a pure function of the session and the host reading, testable without a
//! terminal.

use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span, Text};

use super::{ACCENT, pad_left, pad_right, status_color, truncate_right};
use crate::session::host::HostMonitor;
use crate::session::throughput::{Phase, RATE_WIDTH, Sample, format_rate};
use crate::session::{Session, format_uptime};

/// Eighths of a cell, lowest first.
const SPARK: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
/// The narrowest a tile is worth drawing: a label, a value, and enough history
/// to see a trend in.
const MIN_TILE: usize = 18;
/// Space between two tiles on one row.
const TILE_GAP: usize = 2;
/// The least of the model name the header keeps, wrapping rather than
/// truncating it to nothing.
const MIN_NAME: usize = 8;
/// Requests listed under "Recent".
const RECENT_ROWS: usize = 5;
const GIB: f64 = (1u64 << 30) as f64;
/// Columns one phase of a request row takes after its label: six digits of
/// tokens, ` @ `, and the rate.
const REQUEST_HALF: usize = 6 + 3 + RATE_WIDTH;

/// One tile: what it measures, its current value, and its history scaled
/// against `ceiling`.
struct Tile {
    label: &'static str,
    value: String,
    history: Vec<Option<f64>>,
    ceiling: Option<f64>,
}

pub(super) fn monitor_lines(session: &Session, host: &HostMonitor, width: usize) -> Text<'static> {
    let mut lines = vec![header(session, width), Line::raw("")];

    let decode = session.throughput.rates(Phase::Decode);
    let prefill = session.throughput.rates(Phase::Prefill);
    lines.push(section("Session"));
    lines.extend(tiles(
        &[
            Tile {
                label: "Decode",
                value: rate(session.throughput.rate(Phase::Decode)),
                ceiling: peak(&decode),
                history: decode,
            },
            Tile {
                label: "Prefill",
                value: rate(session.throughput.rate(Phase::Prefill)),
                ceiling: peak(&prefill),
                history: prefill,
            },
            Tile {
                label: "CPU",
                value: session.cpu_percent.map_or("—".into(), |c| format!("{c:.0}%")),
                history: session.cpu_history.values(),
                // A server on many cores goes past 100%; the scale follows it.
                ceiling: Some(session.cpu_history.max().unwrap_or(0.0).max(100.0)),
            },
            Tile {
                label: "Memory",
                value: session
                    .rss_bytes
                    .map_or("—".into(), |b| format!("{:.1} GiB", b as f64 / GIB)),
                history: session.rss_history.values(),
                ceiling: session.rss_history.max(),
            },
        ],
        width,
        Color::Cyan,
    ));
    lines.push(Line::raw(""));
    lines.extend(context_fill(session, width));

    let machine = machine_lines(host, width);
    if !machine.is_empty() {
        lines.push(Line::raw(""));
        lines.extend(machine);
    }

    let requests = session.throughput.requests();
    if !requests.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            "Recent requests".bold(),
            Span::styled("  tokens @ t/s", Style::default().fg(Color::DarkGray)),
        ]));
        for request in requests.iter().take(RECENT_ROWS) {
            lines.push(request_line(request.prefill, request.decode));
        }
    }
    Text::from(lines)
}

/// `● Running · http://127.0.0.1:8000/v1 · gemma-4-e4b-it 16K · 38m 32s`
///
/// Everything identifying the session on one line, as `strata monitor` heads
/// its view. The compute backend and the profile are columns of the row beside
/// it; the context size appears once a log line has stated it, and the uptime
/// while there is a process to have one.
///
/// The model name gives way when the pane is narrow — it is the one part that
/// is long, and the list beside it carries it in full.
fn header(session: &Session, width: usize) -> Line<'static> {
    let r = &session.record;
    let dot = || Span::styled(" · ", Style::default().fg(Color::DarkGray));
    let mut before = vec![
        Span::styled(
            format!("{} {}", session.status.glyph(), session.status_label()),
            Style::default().fg(status_color(session.status)),
        ),
        dot(),
        Span::styled(r.endpoint(), Style::default().fg(Color::Cyan).bold()),
        dot(),
    ];
    let mut after = Vec::new();
    if let Some(size) = session.context.size {
        after.push(Span::styled(format!(" {}", tokens(size)), Style::default().fg(Color::Gray)));
    }
    if let Some(uptime) = session.uptime_secs() {
        after.push(dot());
        after.push(Span::styled(format_uptime(uptime), Style::default().fg(Color::Gray)));
    }
    let taken: usize = before.iter().chain(&after).map(Span::width).sum();
    let name = truncate_right(&r.name, width.saturating_sub(taken).max(MIN_NAME));
    before.push(Span::styled(name, Style::default().fg(ACCENT).bold()));
    before.extend(after);
    Line::from(before)
}

/// `Context fill          9.8K / 16K  61%` over a bar the pane's width: how
/// much of its context the last finished request occupied. Nothing until the
/// server has said how large the context is.
fn context_fill(session: &Session, width: usize) -> Vec<Line<'static>> {
    let fill = session.context;
    let (Some(size), Some(fraction)) = (fill.size, fill.fraction()) else { return Vec::new() };
    let figure = match fill.used {
        Some(used) => format!("{} / {}  {:.0}%", tokens(used), tokens(size), fraction * 100.0),
        None => format!("— / {}", tokens(size)),
    };
    let label = "Context fill";
    let (filled, empty) = bar(fraction, width);
    let color = if fraction >= 0.9 { Color::Red } else { Color::Cyan };
    vec![
        Line::from(vec![
            Span::styled(
                pad_right(label, width.saturating_sub(figure.chars().count())),
                Style::default().fg(Color::Gray),
            ),
            Span::raw(figure).bold(),
        ]),
        Line::from(vec![
            Span::styled(filled, Style::default().fg(color)),
            Span::styled(empty, Style::default().fg(Color::DarkGray)),
        ]),
    ]
}

/// A token count the way context sizes are spoken of: `97`, `9.8K`, `128K`.
/// Context sizes are powers of two, so a K is 1024 of them.
fn tokens(count: u64) -> String {
    if count < 1024 {
        count.to_string()
    } else if count.is_multiple_of(1024) {
        format!("{}K", count / 1024)
    } else {
        format!("{:.1}K", count as f64 / 1024.0)
    }
}

/// The GPU and RAM the sessions share. Empty when there is nothing to show —
/// no sample yet, or no live session to have taken one.
fn machine_lines(host: &HostMonitor, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if let Some(gpu) = &host.gpu {
        let more = match host.gpu_count {
            0 | 1 => String::new(),
            n => format!(" (1 of {n})"),
        };
        lines.push(Line::from(vec![
            "GPU".bold(),
            Span::styled(
                format!(" · {}{more} · system-wide", gpu.name),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        lines.extend(tiles(
            &[
                Tile {
                    label: "Busy",
                    value: gpu.busy_percent.map_or("—".into(), |b| format!("{b:.0}%")),
                    history: host.gpu_busy.values(),
                    ceiling: Some(100.0),
                },
                Tile {
                    label: "Power",
                    value: gpu.power_w.map_or("—".into(), |w| format!("{w:.0} W")),
                    history: host.gpu_power.values(),
                    ceiling: host.gpu_power.max(),
                },
                Tile {
                    label: "Temp",
                    value: gpu.temp_c.map_or("—".into(), |t| format!("{t:.0} °C")),
                    history: host.gpu_temp.values(),
                    ceiling: Some(100.0),
                },
            ],
            width,
            Color::LightMagenta,
        ));
        lines.push(Line::raw(""));
        if let Some((used, total)) = gpu.vram {
            lines.push(gauge("VRAM", used, total, width));
        }
        // On a discrete card GTT is a spill area that sits near empty; on an
        // APU it is where the model lives. Shown once it holds anything.
        if let Some((used, total)) = gpu.gtt.filter(|(used, _)| *used >= 1 << 30) {
            lines.push(gauge("GTT", used, total, width));
        }
    }
    if let Some(memory) = host.memory {
        if lines.is_empty() {
            lines.push(section("Machine"));
        }
        lines.push(gauge("RAM", memory.used(), memory.total, width));
    }
    lines
}

fn section(title: &str) -> Line<'static> {
    Line::from(title.to_string().bold())
}

/// Lay tiles out two to a row when the pane has room, else one, with a blank
/// row between rows so one tile's sparkline does not run into the next tile's
/// heading. `color` draws the sparklines; each section has its own, so the
/// session's history and the machine's read as separate groups.
fn tiles(tiles: &[Tile], width: usize, color: Color) -> Vec<Line<'static>> {
    let per_row = if width >= 2 * MIN_TILE + TILE_GAP { 2 } else { 1 };
    let tile_width = if per_row == 2 { (width - TILE_GAP) / 2 } else { width };
    let mut lines = Vec::new();
    for (n, row) in tiles.chunks(per_row).enumerate() {
        if n > 0 {
            lines.push(Line::raw(""));
        }
        let mut heading = Vec::new();
        let mut history = Vec::new();
        for (i, tile) in row.iter().enumerate() {
            if i > 0 {
                heading.push(Span::raw(" ".repeat(TILE_GAP)));
                history.push(Span::raw(" ".repeat(TILE_GAP)));
            }
            let value =
                truncate_right(&tile.value, tile_width.saturating_sub(tile.label.len() + 1));
            let label_width = tile_width.saturating_sub(value.chars().count());
            heading.push(Span::styled(
                pad_right(tile.label, label_width),
                Style::default().fg(Color::Gray),
            ));
            heading.push(Span::raw(value).bold());
            history.push(Span::styled(
                spark(&tile.history, tile_width, tile.ceiling),
                Style::default().fg(color),
            ));
        }
        lines.push(Line::from(heading));
        lines.push(Line::from(history));
    }
    lines
}

/// `VRAM ████████░░░░  40.4/124.0 GiB`
fn gauge(label: &str, used: u64, total: u64, width: usize) -> Line<'static> {
    let figure = format!(" {:.1}/{:.1} GiB", used as f64 / GIB, total as f64 / GIB);
    let label = pad_right(label, 5);
    let bar_width = width.saturating_sub(label.len() + figure.len());
    let fraction = if total == 0 { 0.0 } else { used as f64 / total as f64 };
    let (filled, empty) = bar(fraction, bar_width);
    let color = if fraction >= 0.9 { Color::Red } else { Color::Green };
    Line::from(vec![
        Span::styled(label, Style::default().fg(Color::Gray)),
        Span::styled(filled, Style::default().fg(color)),
        Span::styled(empty, Style::default().fg(Color::DarkGray)),
        Span::raw(figure),
    ])
}

/// `pp    339 @   211   tg     53 @ 39.10` — tokens at tokens per second, the
/// units in the section heading so a row fits the narrowest pane.
fn request_line(prefill: Option<Sample>, decode: Option<Sample>) -> Line<'static> {
    let half = |label: &str, sample: Option<Sample>| -> Vec<Span<'static>> {
        let label = Span::styled(format!("{label} "), Style::default().fg(Color::DarkGray));
        let Some(sample) = sample else {
            return vec![
                label,
                Span::styled(pad_left("—", REQUEST_HALF), Style::default().fg(Color::DarkGray)),
            ];
        };
        let rate = sample.rate().map_or("—".into(), format_rate);
        vec![
            label,
            Span::raw(pad_left(&sample.tokens.to_string(), 6)),
            Span::styled(" @ ", Style::default().fg(Color::DarkGray)),
            Span::raw(pad_left(&rate, RATE_WIDTH)),
        ]
    };
    let mut spans = half("pp", prefill);
    spans.push(Span::raw("   "));
    spans.extend(half("tg", decode));
    Line::from(spans)
}

fn rate(rate: Option<f64>) -> String {
    rate.map_or("—".into(), |r| format!("{} t/s", format_rate(r)))
}

fn peak(values: &[Option<f64>]) -> Option<f64> {
    values.iter().flatten().copied().reduce(f64::max)
}

/// The newest `width` samples as a sparkline, right-aligned so the latest is
/// always at the edge. A gap draws as a space; with no ceiling every measured
/// sample draws at the floor rather than inventing a scale.
fn spark(values: &[Option<f64>], width: usize, ceiling: Option<f64>) -> String {
    let shown = &values[values.len().saturating_sub(width)..];
    let mut out = " ".repeat(width - shown.len());
    for value in shown {
        out.push(match (value, ceiling) {
            (None, _) => ' ',
            (Some(v), Some(top)) if top > 0.0 => {
                let level = (v / top * (SPARK.len() - 1) as f64).round();
                SPARK[level.clamp(0.0, (SPARK.len() - 1) as f64) as usize]
            }
            (Some(_), _) => SPARK[0],
        });
    }
    out
}

/// A `width`-cell bar `fraction` full, as its filled and empty runs.
fn bar(fraction: f64, width: usize) -> (String, String) {
    let filled = ((fraction.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    ("█".repeat(filled), "░".repeat(width - filled))
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_width::UnicodeWidthStr;

    #[test]
    fn a_sparkline_keeps_the_newest_samples_at_the_right_edge() {
        let values: Vec<Option<f64>> = (0..=7).map(|v| Some(v as f64)).collect();
        assert_eq!(spark(&values, 8, Some(7.0)), "▁▂▃▄▅▆▇█");
        assert_eq!(spark(&values, 3, Some(7.0)), "▆▇█", "older samples fall off the left");
        assert_eq!(spark(&values[..2], 4, Some(7.0)), "  ▁▂", "a short history is right-aligned");
        assert_eq!(
            spark(&[Some(3.0), None, Some(9.0)], 3, Some(3.0)),
            "█ █",
            "gaps stay gaps; overshoot clamps"
        );
        assert_eq!(
            spark(&[Some(0.0), Some(0.0)], 2, Some(0.0)),
            "▁▁",
            "a zero ceiling draws the floor"
        );
    }

    #[test]
    fn a_bar_spends_exactly_its_width() {
        for width in [0, 1, 7, 20] {
            for fraction in [-0.5, 0.0, 0.33, 0.5, 1.0, 1.7] {
                let (filled, empty) = bar(fraction, width);
                assert_eq!(
                    filled.chars().count() + empty.chars().count(),
                    width,
                    "{fraction} of {width}"
                );
            }
        }
        assert_eq!(bar(0.5, 10).0.chars().count(), 5);
    }

    #[test]
    fn tiles_pair_up_when_there_is_room_and_never_overrun_the_pane() {
        let tile = |label| Tile {
            label,
            value: "211.40 t/s".into(),
            history: vec![Some(1.0); 80],
            ceiling: Some(2.0),
        };
        let four = [tile("Decode"), tile("Prefill"), tile("CPU"), tile("Memory")];
        for width in [MIN_TILE, 30, 2 * MIN_TILE + TILE_GAP, 42, 60, 90] {
            let lines = tiles(&four, width, Color::Cyan);
            let paired = width >= 2 * MIN_TILE + TILE_GAP;
            // Two lines a row, and a blank between rows.
            assert_eq!(lines.len(), if paired { 5 } else { 11 }, "{width}");
            for line in &lines {
                let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
                assert!(text.width() <= width, "{width}: {text:?} is {} wide", text.width());
            }
        }
    }

    /// The pane is never narrower than this many columns inside its borders.
    const NARROWEST: usize = super::super::MIN_DETAIL_WIDTH as usize - 2;

    #[test]
    fn a_request_row_fits_the_narrowest_pane() {
        let big = |phase, tokens| Some(Sample { phase, tokens, seconds: 0.001 });
        for (prefill, decode) in [
            (big(Phase::Prefill, 999_999), big(Phase::Decode, 999_999)),
            (big(Phase::Prefill, 12), None),
            (None, big(Phase::Decode, 7)),
        ] {
            let line = request_line(prefill, decode);
            let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
            assert!(text.width() <= NARROWEST, "{text:?} is {} wide", text.width());
        }
    }

    #[test]
    fn token_counts_read_as_context_sizes_are_spoken_of() {
        assert_eq!(tokens(97), "97");
        assert_eq!(tokens(16384), "16K");
        assert_eq!(tokens(131072), "128K");
        assert_eq!(tokens(10035), "9.8K");
    }

    #[test]
    fn a_gauge_fits_its_width() {
        for width in [30, 44, 60] {
            let line = gauge("GTT", 43_423_322_112, 133_143_986_176, width);
            let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
            assert_eq!(text.width(), width, "{text:?}");
            assert!(text.ends_with(" 40.4/124.0 GiB"));
        }
    }
}
