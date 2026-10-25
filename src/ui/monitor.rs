//! The Session Manager's Monitor: the selected session's live figures beside
//! the list, in the column the Detail pane used to hold.
//!
//! A header names the session and says what it is doing this second, boxed
//! tiles pair a current value with its recent history, and bars show how full
//! the context and the machine are. Everything is built as text —
//! borders, sparklines and bars are box-drawing and block characters — so the
//! layout is a pure function of the session and the host reading, testable
//! without a terminal.

use std::borrow::Cow;
use std::time::Instant;

use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span, Text};

use super::{ACCENT, pad_right, status_color, truncate_right};
use crate::session::host::HostMonitor;
use crate::session::throughput::{Activity, Phase, format_rate};
use crate::session::{Session, SessionStatus, format_uptime};

/// Eighths of a cell, lowest first.
const SPARK: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
/// Rows a sparkline is tall. One row has only eight levels, which flattens a
/// history into a strip; two give sixteen and room to see a trend.
const SPARK_ROWS: usize = 2;
/// The narrowest a tile is worth drawing: its border, a value, and enough
/// history to see a trend in.
const MIN_TILE: usize = 18;
/// Space between two tiles on one row.
const TILE_GAP: usize = 1;
/// The least of the model name the header keeps; with less room than this it
/// leaves the name out.
const MIN_NAME: usize = 8;
const GIB: f64 = (1u64 << 30) as f64;
const BORDER: Color = Color::DarkGray;

/// One tile: what it measures, its current value, and its history scaled
/// against `ceiling`. A live tile is showing a request still in progress.
struct Tile {
    label: Cow<'static, str>,
    value: String,
    history: Vec<Option<f64>>,
    ceiling: Option<f64>,
    live: bool,
}

impl Tile {
    fn new(
        label: impl Into<Cow<'static, str>>,
        value: String,
        history: Vec<Option<f64>>,
        ceiling: Option<f64>,
    ) -> Self {
        Self { label: label.into(), value, history, ceiling, live: false }
    }
}

pub(super) fn monitor_lines(
    session: &Session,
    host: &HostMonitor,
    width: usize,
    now: Instant,
) -> Text<'static> {
    let mut lines = vec![header(session, width, now), Line::raw("")];

    let speed = |label: &'static str, phase| {
        let history = session.throughput.rates(phase);
        let current = session.throughput.current(phase, now);
        Tile {
            label: label.into(),
            value: current.map_or("—".into(), |(r, _)| format!("{} t/s", format_rate(r))),
            ceiling: peak(&history),
            history,
            live: current.is_some_and(|(_, live)| live),
        }
    };
    lines.extend(tiles(
        &[
            speed("Decode", Phase::Decode),
            speed("Prefill", Phase::Prefill),
            Tile::new(
                "CPU",
                session.cpu_percent.map_or("—".into(), |c| format!("{c:.0}%")),
                session.cpu_history.values(),
                // A server on many cores goes past 100%; the scale follows it.
                Some(session.cpu_history.max().unwrap_or(0.0).max(100.0)),
            ),
            Tile::new(
                "Memory",
                session.rss_bytes.map_or("—".into(), |b| format!("{:.1} GiB", b as f64 / GIB)),
                session.rss_history.values(),
                session.rss_history.max(),
            ),
        ],
        width,
        Color::Cyan,
    ));
    let context = context_fill(session, width);
    if !context.is_empty() {
        lines.push(Line::raw(""));
        lines.extend(context);
    }

    let machine = machine_lines(host, width);
    if !machine.is_empty() {
        lines.push(Line::raw(""));
        lines.extend(machine);
    }
    Text::from(lines)
}

/// `● Running · http://127.0.0.1:8000/v1 · gemma-4-e4b-it 16K · 38m 32s`
///
/// Everything identifying the session on one line, as `strata monitor` heads
/// its view. While a request runs, its progress stands in for the status — see
/// [`status`]. The compute backend and the profile are columns of the row beside
/// it; the context size appears once a log line has stated it, and the uptime
/// while there is a process to have one.
///
/// The model name gives way when the pane is narrow — it is the one part that
/// is long, and the list beside it carries it in full.
fn header(session: &Session, width: usize, now: Instant) -> Line<'static> {
    let r = &session.record;
    let dot = || Span::styled(" · ", Style::default().fg(Color::DarkGray));
    let lead = vec![
        status(session, now),
        dot(),
        Span::styled(r.endpoint(), Style::default().fg(Color::Cyan).bold()),
    ];
    let size = session
        .context
        .size
        .map(|size| Span::styled(tokens(size), Style::default().fg(Color::Gray)));
    let uptime = session
        .uptime_secs()
        .map(|uptime| Span::styled(format_uptime(uptime), Style::default().fg(Color::Gray)));

    // Narrow panes shed what the list beside them already shows: first room
    // for the name, then the uptime, then the name itself. Only a pane too
    // narrow even for that wraps.
    for (keep_uptime, keep_name) in [(true, true), (false, true), (false, false)] {
        let mut tail = Vec::new();
        if let Some(size) = &size {
            if keep_name {
                tail.push(Span::raw(" "));
            } else {
                tail.push(dot());
            }
            tail.push(size.clone());
        }
        if keep_uptime && let Some(uptime) = &uptime {
            tail.push(dot());
            tail.push(uptime.clone());
        }
        let taken: usize = lead.iter().chain(&tail).map(Span::width).sum::<usize>()
            + if keep_name { dot().width() } else { 0 };
        let room = width.saturating_sub(taken);
        if keep_name && room < MIN_NAME {
            continue;
        }
        let mut line = lead.clone();
        if keep_name {
            line.push(dot());
            line.push(Span::styled(
                truncate_right(&r.name, room.max(MIN_NAME)),
                Style::default().fg(ACCENT).bold(),
            ));
        }
        line.extend(tail);
        return Line::from(line);
    }
    unreachable!("the last arrangement always returns")
}

/// The session's status, or — while a request runs — what it is doing, from
/// the server's own progress reports:
///
/// - `▶ Reading 76%` while it processes a prompt;
/// - `▶ Generating 781 tokens` while it generates;
/// - `● Running` between requests, as for every other status.
///
/// No rate: the Decode and Prefill tiles beside it carry that, live.
fn status(session: &Session, now: Instant) -> Span<'static> {
    let activity = match session.status {
        SessionStatus::Running => session.throughput.activity(now),
        _ => None,
    };
    match activity {
        Some(Activity::Generating(live)) => Span::styled(
            format!("▶ Generating {} tokens", live.tokens),
            Style::default().fg(Color::Green).bold(),
        ),
        Some(Activity::Reading(live)) => Span::styled(
            match live.progress {
                Some(progress) => format!("▶ Reading {:.0}%", progress * 100.0),
                None => "▶ Reading".into(),
            },
            Style::default().fg(Color::Cyan).bold(),
        ),
        Some(Activity::Idle) | None => Span::styled(
            format!("{} {}", session.status.glyph(), session.status_label()),
            Style::default().fg(status_color(session.status)),
        ),
    }
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
            Span::styled(empty, Style::default().fg(BORDER)),
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
        // Where throughput is measured the link heads the PCIe tiles instead.
        let pcie_tiles = host.pcie_rx.max().is_some() || host.pcie_tx.max().is_some();
        let link = match gpu.pcie_link {
            Some((generation, lanes)) if !pcie_tiles => format!(" · PCIe Gen{generation} x{lanes}"),
            _ => String::new(),
        };
        lines.push(Line::from(vec![
            "GPU".bold(),
            Span::styled(format!(" · {}{more}{link}", gpu.name), Style::default().fg(Color::Gray)),
        ]));
        lines.extend(tiles(
            &[
                Tile::new(
                    "Busy",
                    gpu.busy_percent.map_or("—".into(), |b| format!("{b:.0}%")),
                    host.gpu_busy.values(),
                    Some(100.0),
                ),
                Tile::new(
                    "Clock",
                    gpu.clock_mhz.map_or("—".into(), |mhz| format!("{mhz:.0} MHz")),
                    host.gpu_clock.values(),
                    host.gpu_clock.max(),
                ),
                Tile::new(
                    "Power",
                    gpu.power_w.map_or("—".into(), |w| format!("{w:.0} W")),
                    host.gpu_power.values(),
                    host.gpu_power.max(),
                ),
                Tile::new(
                    "Temp",
                    gpu.temp_c.map_or("—".into(), |t| format!("{t:.0} °C")),
                    host.gpu_temp.values(),
                    Some(100.0),
                ),
            ],
            width,
            Color::LightMagenta,
        ));
        // Only where throughput is measured, which today is NVIDIA. The two
        // directions share a scale, so their heights compare.
        if pcie_tiles {
            let ceiling = host.pcie_rx.max().into_iter().chain(host.pcie_tx.max()).reduce(f64::max);
            let link = gpu.pcie_link.map_or_else(String::new, |(generation, lanes)| {
                format!(" Gen{generation} x{lanes}")
            });
            lines.extend(tiles(
                &[
                    Tile::new(
                        format!("PCIe{link}"),
                        format!("to GPU {}", gpu.pcie_rx.map_or("—".into(), megabytes_per_second)),
                        host.pcie_rx.values(),
                        ceiling,
                    ),
                    Tile::new(
                        "PCIe",
                        format!(
                            "from GPU {}",
                            gpu.pcie_tx.map_or("—".into(), megabytes_per_second)
                        ),
                        host.pcie_tx.values(),
                        ceiling,
                    ),
                ],
                width,
                Color::LightMagenta,
            ));
        }
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
        lines.push(gauge("RAM", memory.used(), memory.total, width));
    }
    lines
}

/// Lay tiles out as bordered boxes, two to a row when the pane has room, else
/// one:
///
/// ```text
/// ╭ Decode ● ──────╮ ╭ Prefill ───────╮
/// │ 29.75 t/s      │ │ 58.95 t/s      │
/// │      ▃▅▆█▇▆▅▆▇ │ │       ▂▃▅▃▂▆▇▅ │
/// ╰────────────────╯ ╰────────────────╯
/// ```
///
/// `color` draws the sparklines; each section has its own, so the session's
/// history and the machine's read as separate groups. A live tile — a request
/// in progress — marks its title and shows its value in green.
fn tiles(tiles: &[Tile], width: usize, color: Color) -> Vec<Line<'static>> {
    let per_row = if width >= 2 * MIN_TILE + TILE_GAP { 2 } else { 1 };
    let tile_width = if per_row == 2 { (width - TILE_GAP) / 2 } else { width };
    // Inside the border and a space of padding either side.
    let inner = tile_width.saturating_sub(4);
    let border = Style::default().fg(BORDER);
    let mut lines = Vec::new();
    for row in tiles.chunks(per_row) {
        // Top border, value, the sparkline's rows, bottom border.
        let mut rows: Vec<Vec<Span<'static>>> = vec![Vec::new(); 3 + SPARK_ROWS];
        for (i, tile) in row.iter().enumerate() {
            if i > 0 {
                for spans in &mut rows {
                    spans.push(Span::raw(" ".repeat(TILE_GAP)));
                }
            }
            let mut title =
                vec![Span::styled(format!(" {} ", tile.label), Style::default().fg(Color::Gray))];
            if tile.live {
                title.push(Span::styled("● ", Style::default().fg(Color::Green)));
            }
            let title_width: usize = title.iter().map(Span::width).sum();
            rows[0].push(Span::styled("╭", border));
            rows[0].extend(title);
            rows[0].push(Span::styled(
                format!("{}╮", "─".repeat(tile_width.saturating_sub(2 + title_width))),
                border,
            ));

            let value = truncate_right(&tile.value, inner);
            let value_style =
                if tile.live { Style::default().fg(Color::Green) } else { Style::default() };
            rows[1].push(Span::styled("│ ", border));
            rows[1].push(Span::styled(pad_right(&value, inner), value_style.bold()));
            rows[1].push(Span::styled(" │", border));

            for (n, line) in
                spark(&tile.history, inner, tile.ceiling, SPARK_ROWS).into_iter().enumerate()
            {
                let spans = &mut rows[2 + n];
                spans.push(Span::styled("│ ", border));
                spans.push(Span::styled(line, Style::default().fg(color)));
                spans.push(Span::styled(" │", border));
            }

            rows[2 + SPARK_ROWS].push(Span::styled(
                format!("╰{}╯", "─".repeat(tile_width.saturating_sub(2))),
                border,
            ));
        }
        lines.extend(rows.into_iter().map(Line::from));
    }
    lines
}

/// `578 MB/s`, or `2.5 GB/s` from a thousand up.
fn megabytes_per_second(rate: f64) -> String {
    if rate >= 1000.0 { format!("{:.1} GB/s", rate / 1000.0) } else { format!("{rate:.0} MB/s") }
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
        Span::styled(empty, Style::default().fg(BORDER)),
        Span::raw(figure),
    ])
}

fn peak(values: &[Option<f64>]) -> Option<f64> {
    values.iter().flatten().copied().reduce(f64::max)
}

/// The newest `width` samples as a sparkline `rows` tall, top row first,
/// right-aligned so the latest is always at the edge.
///
/// Each sample fills its column in eighths of a cell from the bottom up, so
/// `rows` rows give `8 × rows` levels. A measured sample is never less than
/// the floor (`▁`), so a zero still reads as a reading rather than a gap; a
/// gap draws as blank. With no ceiling, every measured sample draws at the
/// floor rather than inventing a scale.
fn spark(values: &[Option<f64>], width: usize, ceiling: Option<f64>, rows: usize) -> Vec<String> {
    let shown = &values[values.len().saturating_sub(width)..];
    let levels = 8 * rows;
    let mut out = vec![" ".repeat(width - shown.len()); rows];
    for value in shown {
        let eighths = match (value, ceiling) {
            (None, _) => 0,
            (Some(v), Some(top)) if top > 0.0 => {
                (v / top * levels as f64).round().clamp(1.0, levels as f64) as usize
            }
            (Some(_), _) => 1,
        };
        for (row, line) in out.iter_mut().enumerate() {
            // Eighths held by the rows beneath this one.
            let below = 8 * (rows - 1 - row);
            line.push(match eighths.saturating_sub(below).min(8) {
                0 => ' ',
                fill => SPARK[fill - 1],
            });
        }
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
        let values: Vec<Option<f64>> = (1..=8).map(|v| Some(v as f64)).collect();
        assert_eq!(spark(&values, 8, Some(8.0), 1), ["▁▂▃▄▅▆▇█"]);
        assert_eq!(spark(&values, 3, Some(8.0), 1), ["▆▇█"], "older samples fall off the left");
        assert_eq!(
            spark(&values[..2], 4, Some(8.0), 1),
            ["  ▁▂"],
            "a short history is right-aligned"
        );
        assert_eq!(
            spark(&[Some(3.0), None, Some(9.0)], 3, Some(3.0), 1),
            ["█ █"],
            "gaps stay gaps; overshoot clamps"
        );
        assert_eq!(
            spark(&[Some(0.0), Some(0.0)], 2, Some(0.0), 1),
            ["▁▁"],
            "a zero ceiling draws the floor"
        );
    }

    #[test]
    fn a_taller_sparkline_stacks_its_levels_from_the_bottom_row_up() {
        // Sixteen levels over two rows: a quarter, a half, three quarters, full.
        let values = [Some(0.0), Some(4.0), Some(8.0), Some(12.0), Some(16.0)];
        assert_eq!(spark(&values, 5, Some(16.0), 2), ["   ▄█", "▁▄███"]);
        assert_eq!(
            spark(&[None, Some(1.0)], 3, Some(16.0), 2),
            ["   ", "  ▁"],
            "a gap is blank in every row, a small reading keeps the floor"
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

    fn text_of(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn boxed_tiles_pair_up_when_there_is_room_and_keep_their_borders_square() {
        let tile = |label| Tile {
            label: Cow::Borrowed(label),
            value: "211.40 t/s".into(),
            history: vec![Some(1.0); 80],
            ceiling: Some(2.0),
            live: label == "Decode",
        };
        let four = [tile("Decode"), tile("Prefill"), tile("CPU"), tile("Memory")];
        for width in [MIN_TILE, 30, 2 * MIN_TILE + TILE_GAP, 42, 60, 90] {
            let lines = tiles(&four, width, Color::Cyan);
            let paired = width >= 2 * MIN_TILE + TILE_GAP;
            // Borders, value, and the sparkline's rows, per row of boxes.
            let tall = 3 + SPARK_ROWS;
            assert_eq!(lines.len(), if paired { 2 * tall } else { 4 * tall }, "{width}");
            for row in lines.chunks(tall) {
                let widths: Vec<usize> = row.iter().map(|l| text_of(l).width()).collect();
                assert!(widths.iter().all(|w| *w == widths[0]), "{width}: ragged box {widths:?}");
                assert!(widths[0] <= width, "{width}: {:?} overruns", text_of(&row[0]));
                assert!(
                    text_of(&row[0]).starts_with('╭') && text_of(&row[tall - 1]).ends_with('╯')
                );
            }
        }
    }

    fn running_session() -> Session {
        use crate::session::record::SessionRecord;
        Session::probe(
            SessionRecord {
                id: "1".into(),
                name: "gemma".into(),
                runtime: "llama.cpp".into(),
                model: "m".into(),
                model_path: "m".into(),
                profile: "Default".into(),
                size_bytes: None,
                device: None,
                pid: 1,
                host: "127.0.0.1".into(),
                port: 8000,
                command: Vec::new(),
                health_path: "/health".into(),
                log_file: "/dev/null".into(),
                download: None,
                started_unix: 0,
            },
            SessionStatus::Running,
            Default::default(),
        )
    }

    #[test]
    fn the_header_sheds_what_the_list_shows_rather_than_wrap() {
        use crate::session::context::Reading;
        use crate::session::throughput::Live;
        let now = Instant::now();
        let mut session = running_session();
        session.record.name = "nvidia-nemotron-3-5-bolt-74b-a9b-ud-q3_k_xl".into();
        session.context.record(Reading::Size(64_512));
        let generating =
            Live { phase: Phase::Decode, tokens: 215, rate: 35.0, progress: None, slot: None };
        session.throughput.record_live(generating, now);

        let wide = text_of(&header(&session, 120, now));
        assert!(wide.contains("nvidia-nemotron-3-5-bolt"), "a wide pane keeps the name: {wide}");
        // From where status, endpoint and size fit at all: below that the
        // line wraps, which a pane that narrow cannot avoid.
        for width in [56, 62, 70, 90, 120] {
            let line = text_of(&header(&session, width, now));
            assert!(line.width() <= width, "{width}: {line:?} is {} wide", line.width());
            assert!(
                line.starts_with("▶ Generating 215 tokens · http://127.0.0.1:8000/v1"),
                "{line}"
            );
            assert!(line.contains("63K"), "the context size stays: {line}");
        }
    }

    #[test]
    fn a_request_in_progress_stands_in_for_the_status() {
        use crate::session::throughput::{Live, Sample};
        let now = Instant::now();
        let mut session = running_session();
        assert_eq!(
            status(&session, now).content,
            "● Running",
            "a runtime that never reports progress"
        );

        let reading = Live {
            phase: Phase::Prefill,
            tokens: 2048,
            rate: 274.0,
            progress: Some(0.76),
            slot: None,
        };
        session.throughput.record_live(reading, now);
        assert_eq!(status(&session, now).content, "▶ Reading 76%");

        let generating =
            Live { phase: Phase::Decode, tokens: 781, rate: 29.75, progress: None, slot: None };
        session.throughput.record_live(generating, now);
        assert_eq!(status(&session, now).content, "▶ Generating 781 tokens", "and no rate");

        session.throughput.record(Sample {
            phase: Phase::Decode,
            tokens: 931,
            seconds: 26.9,
            slot: None,
        });
        assert_eq!(status(&session, now).content, "● Running", "between requests");

        session.throughput.record_live(generating, now);
        session.status = SessionStatus::Stopped;
        assert!(
            !status(&session, now).content.contains("Generating"),
            "only a running session is busy"
        );
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

    #[test]
    fn the_pcie_link_heads_the_tiles_or_else_the_gpu_line() {
        use crate::session::host::GpuSample;
        let mut host = HostMonitor::default();
        host.gpu = Some(GpuSample { name: "NVIDIA GeForce RTX 3080".into(), ..Default::default() });
        let text = |host: &HostMonitor| -> String {
            machine_lines(host, 60).iter().map(|line| text_of(line) + "\n").collect()
        };
        assert!(!text(&host).contains("PCIe"), "no throughput, no PCIe row");
        host.gpu = Some(GpuSample {
            name: "amdgpu 1002:744c".into(),
            pcie_link: Some((4, 16)),
            ..Default::default()
        });
        let shown = text(&host);
        assert!(shown.starts_with("GPU · amdgpu 1002:744c · PCIe Gen4 x16\n"), "{shown}");
        assert!(!shown.contains("╭ PCIe"), "a link alone goes on the header, no tiles");

        host.gpu = Some(GpuSample {
            name: "NVIDIA GeForce RTX 3080".into(),
            pcie_link: Some((4, 16)),
            pcie_rx: Some(2522.0),
            pcie_tx: Some(42.0),
            ..Default::default()
        });
        host.pcie_rx.push(Some(2522.0));
        host.pcie_tx.push(Some(42.0));
        let shown = text(&host);
        assert!(shown.contains("╭ PCIe Gen4 x16 "), "{shown}");
        assert!(shown.starts_with("GPU · NVIDIA GeForce RTX 3080\n"), "the tiles carry the link");
        assert!(shown.contains("to GPU 2.5 GB/s"), "{shown}");
        assert!(shown.contains("from GPU 42 MB/s"), "{shown}");
    }
}
