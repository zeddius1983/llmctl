//! Prompt-processing (`pp`) and token-generation (`tg`) rates for a session,
//! read out of the log the server is already writing.
//!
//! Both runtimes print their own per-request timings when a request finishes —
//! exact tokens over the exact seconds spent producing them — and what is shown
//! is the latest of those, unaveraged. It is the runtime's own summary of the
//! last thing it did, which is what the log view would tell you and what a
//! benchmark would report.
//!
//! llmctl times nothing itself. It could not usefully: wall-clock timing cannot
//! tell generating apart from idling, so a server that produced 20 tokens in a
//! second and then sat quiet would appear to slow down while doing nothing.
//!
//! llama.cpp also reports a request while it runs — prompt progress, and the
//! generation rate over its last few seconds — which is what [`Live`] carries
//! and what tells the Monitor a session is busy.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Which half of a request a measurement describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Prompt processing / prefill — `pp`.
    Prefill,
    /// Token generation / decode — `tg`.
    Decode,
}

/// One finished measurement, as the runtime reported it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub phase: Phase,
    pub tokens: u64,
    /// Seconds of work, not of wall clock.
    pub seconds: f64,
    /// The server slot that ran the request, where the runtime numbers them
    /// (llama.cpp's `id N`), so its summary ends that request and no other.
    pub slot: Option<u32>,
}

impl Sample {
    pub fn rate(&self) -> Option<f64> {
        (self.seconds > 0.0 && self.tokens > 0).then(|| self.tokens as f64 / self.seconds)
    }
}

/// A request still in progress, as the runtime reports it every few seconds:
/// how far it has got and how fast it is going right now.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Live {
    pub phase: Phase,
    /// Tokens processed (prefill) or generated (decode) so far.
    pub tokens: u64,
    /// The runtime's current rate — over its last few seconds where it reports
    /// one, else since the phase began.
    pub rate: f64,
    /// How much of the prompt is processed, 0–1. Prefill only.
    pub progress: Option<f64>,
    /// The server slot running it; see [`Sample::slot`].
    pub slot: Option<u32>,
}

/// What a session is doing, as far as its log says.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activity {
    /// Processing a prompt.
    Reading(Live),
    /// Generating.
    Generating(Live),
    /// No request in progress.
    Idle,
}

/// How many rates a session keeps for the Monitor's sparklines.
const HISTORY_LEN: usize = 60;

/// How long a live reading stands without another. llama.cpp reports every
/// three seconds; a request that ends without a summary — cancelled, or a
/// crash — must not leave the session looking busy forever.
const LIVE_FOR: Duration = Duration::from_secs(10);

/// The latest rates a session has reported, and the rates before them.
#[derive(Default)]
pub struct Throughput {
    last_prefill: Option<Sample>,
    last_decode: Option<Sample>,
    /// Oldest first: every rate reported, live and final, by phase.
    history: VecDeque<(Phase, f64)>,
    /// Requests in progress, one per slot, least recently reported first.
    /// Parallel slots run requests side by side, and each one's summary ends
    /// only its own.
    live: Vec<(Live, Instant)>,
    /// Whether this runtime reports requests in progress at all. One that does
    /// not (FastFlowLM) cannot be said to be idle, only silent.
    reports_live: bool,
}

impl Throughput {
    /// A request's summary, once it is finished.
    pub fn record(&mut self, sample: Sample) {
        // A measurement that divides to nothing is not one; keeping it would
        // replace a real figure with a blank.
        let Some(rate) = sample.rate() else { return };
        match sample.phase {
            Phase::Prefill => self.last_prefill = Some(sample),
            Phase::Decode => self.last_decode = Some(sample),
        }
        self.push(sample.phase, rate);
        // The summary ends the phase it describes in its own slot; a decode
        // summary ends the request, since it is printed after the prefill's.
        self.live.retain(|(live, _)| {
            live.slot != sample.slot
                || (sample.phase == Phase::Prefill && live.phase == Phase::Decode)
        });
    }

    /// A progress report from a request still running, received at `now`.
    pub fn record_live(&mut self, live: Live, now: Instant) {
        if !(live.rate.is_finite() && live.rate > 0.0) {
            return;
        }
        self.reports_live = true;
        self.push(live.phase, live.rate);
        // A slot runs one request at a time, so its newest report replaces
        // its last; reports gone stale are dropped on the way.
        self.live.retain(|(other, at)| {
            other.slot != live.slot && now.saturating_duration_since(*at) < LIVE_FOR
        });
        self.live.push((live, now));
    }

    fn push(&mut self, phase: Phase, rate: f64) {
        if self.history.len() == HISTORY_LEN {
            self.history.pop_front();
        }
        self.history.push_back((phase, rate));
    }

    /// The rates reported for `phase`, oldest first. Each is a report rather
    /// than a tick of the clock, so an idle server's history holds still
    /// instead of decaying.
    pub fn rates(&self, phase: Phase) -> Vec<Option<f64>> {
        self.history.iter().filter(|(p, _)| *p == phase).map(|(_, rate)| Some(*rate)).collect()
    }

    /// What the session is doing at `now`, or `None` for a runtime that never
    /// says.
    pub fn activity(&self, now: Instant) -> Option<Activity> {
        if !self.reports_live {
            return None;
        }
        // With several requests running, the one that reported last speaks
        // for the session.
        Some(match self.live_at(now).last() {
            Some(live) if live.phase == Phase::Prefill => Activity::Reading(*live),
            Some(live) => Activity::Generating(*live),
            None => Activity::Idle,
        })
    }

    /// The requests still in progress at `now`, least recently reported first.
    fn live_at(&self, now: Instant) -> Vec<Live> {
        self.live
            .iter()
            .filter(|(_, at)| now.saturating_duration_since(*at) < LIVE_FOR)
            .map(|(live, _)| *live)
            .collect()
    }

    /// The rate to show for `phase` at `now`: the request in progress if it is
    /// in that phase, else the last finished one. The second is whether it is
    /// live.
    pub fn current(&self, phase: Phase, now: Instant) -> Option<(f64, bool)> {
        match self.live_at(now).into_iter().rev().find(|live| live.phase == phase) {
            Some(live) => Some((live.rate, true)),
            None => self.rate(phase).map(|rate| (rate, false)),
        }
    }

    /// The last finished measurement for `phase`.
    pub fn last(&self, phase: Phase) -> Option<Sample> {
        match phase {
            Phase::Prefill => self.last_prefill,
            Phase::Decode => self.last_decode,
        }
    }

    /// The last rate for `phase`, in tokens per second.
    pub fn rate(&self, phase: Phase) -> Option<f64> {
        self.last(phase)?.rate()
    }
}

/// Columns a formatted rate is guaranteed to occupy at most.
///
/// The session pane right-aligns the figure in a field this wide and pads it to
/// fit; padding cannot shrink what it is handed, so anything wider would shift
/// every column after it and, at the pane's edge, wrap the row. Formatting owes
/// the layout this bound, which is why the constant lives with the formatter.
pub const RATE_WIDTH: usize = 5;

/// Format a rate the way the runtimes do: two decimals while they carry
/// information, whole tokens once they do not — and never wider than
/// [`RATE_WIDTH`].
///
/// The thresholds are the *rounded* values, not the raw ones: `{:.2}` turns
/// 99.999 into `100.00`, which is six columns from a figure that tested as
/// under a hundred. Prefill on a small model runs to five digits and beyond, so
/// past that the figure goes to thousands rather than out of its field.
pub fn format_rate(tokens_per_second: f64) -> String {
    let rate = tokens_per_second.max(0.0);
    if rate < 99.995 {
        format!("{rate:.2}")
    } else if rate < 99_999.5 {
        format!("{rate:.0}")
    } else if rate < 999_500.0 {
        format!("{:.0}k", rate / 1_000.0)
    } else if rate < 999_500_000.0 {
        format!("{:.0}M", rate / 1_000_000.0)
    } else {
        // Nothing a runtime can report from a real request. The field's
        // guarantee is a guarantee regardless, so the top end saturates rather
        // than growing another tier for every order of magnitude.
        ">1G".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(phase: Phase, tokens: u64, seconds: f64) -> Sample {
        Sample { phase, tokens, seconds, slot: None }
    }

    fn live(phase: Phase, tokens: u64, rate: f64) -> Live {
        Live { phase, tokens, rate, progress: None, slot: None }
    }

    #[test]
    fn a_running_request_is_shown_live_until_its_summary_arrives() {
        let t0 = Instant::now();
        let mut throughput = Throughput::default();
        assert_eq!(throughput.activity(t0), None, "a runtime that never reports live says nothing");
        throughput.record(sample(Phase::Decode, 300, 10.0));

        throughput
            .record_live(Live { progress: Some(0.76), ..live(Phase::Prefill, 2048, 274.0) }, t0);
        assert!(
            matches!(throughput.activity(t0), Some(Activity::Reading(l)) if l.progress == Some(0.76))
        );
        assert_eq!(
            throughput.current(Phase::Decode, t0),
            Some((30.0, false)),
            "decode is still the last one"
        );

        throughput.record_live(live(Phase::Decode, 100, 28.0), t0);
        assert!(matches!(throughput.activity(t0), Some(Activity::Generating(_))));
        assert_eq!(throughput.current(Phase::Decode, t0), Some((28.0, true)));

        throughput.record(sample(Phase::Prefill, 19, 0.5));
        assert!(
            matches!(throughput.activity(t0), Some(Activity::Generating(_))),
            "prefill's summary leaves decode running"
        );
        throughput.record(sample(Phase::Decode, 931, 26.9));
        assert_eq!(throughput.activity(t0), Some(Activity::Idle));
        assert_eq!(
            throughput.rates(Phase::Decode),
            vec![Some(30.0), Some(28.0), Some(931.0 / 26.9)]
        );
    }

    #[test]
    fn a_request_that_goes_quiet_stops_counting_as_live() {
        let t0 = Instant::now();
        let mut throughput = Throughput::default();
        throughput.record_live(live(Phase::Decode, 100, 28.0), t0);
        assert_eq!(throughput.current(Phase::Decode, t0 + LIVE_FOR / 2), Some((28.0, true)));
        assert_eq!(throughput.activity(t0 + LIVE_FOR), Some(Activity::Idle));
        assert_eq!(
            throughput.current(Phase::Decode, t0 + LIVE_FOR),
            None,
            "nothing finished to fall back on"
        );
    }

    /// Regression: with two slots busy, slot B's summary cleared slot A's live
    /// reading, so the session read as idle while A was still generating.
    #[test]
    fn a_summary_ends_only_the_request_in_its_own_slot() {
        let t0 = Instant::now();
        let in_slot = |slot, rate| Live { slot: Some(slot), ..live(Phase::Decode, 100, rate) };
        let mut throughput = Throughput::default();
        throughput.record_live(in_slot(0, 28.0), t0);
        throughput.record_live(in_slot(1, 40.0), t0);
        assert_eq!(throughput.current(Phase::Decode, t0), Some((40.0, true)), "the latest report");

        throughput.record(Sample { slot: Some(1), ..sample(Phase::Decode, 931, 26.9) });
        assert!(
            matches!(throughput.activity(t0), Some(Activity::Generating(l)) if l.slot == Some(0)),
            "slot 0 is still generating"
        );
        assert_eq!(throughput.current(Phase::Decode, t0), Some((28.0, true)));

        // A slot's prefill summary leaves its own decode running.
        throughput.record(Sample { slot: Some(0), ..sample(Phase::Prefill, 19, 0.3) });
        assert_eq!(throughput.current(Phase::Decode, t0), Some((28.0, true)));
        throughput.record(Sample { slot: Some(0), ..sample(Phase::Decode, 400, 14.0) });
        assert_eq!(throughput.activity(t0), Some(Activity::Idle));
    }

    #[test]
    fn a_slot_reports_one_request_at_a_time() {
        let t0 = Instant::now();
        let mut throughput = Throughput::default();
        throughput.record_live(Live { slot: Some(2), ..live(Phase::Prefill, 512, 300.0) }, t0);
        throughput.record_live(Live { slot: Some(2), ..live(Phase::Decode, 10, 30.0) }, t0);
        assert_eq!(throughput.live.len(), 1, "the slot moved on from prefill to decode");
        assert_eq!(throughput.current(Phase::Prefill, t0), None, "no prefill left in progress");
    }

    #[test]
    fn the_history_is_bounded() {
        let mut throughput = Throughput::default();
        for _ in 0..HISTORY_LEN + 10 {
            throughput.record(sample(Phase::Decode, 10, 1.0));
        }
        assert_eq!(throughput.rates(Phase::Decode).len(), HISTORY_LEN);
    }

    #[test]
    fn the_latest_measurement_replaces_the_one_before_it() {
        let mut throughput = Throughput::default();
        throughput.record(sample(Phase::Decode, 300, 10.0));
        assert_eq!(throughput.rate(Phase::Decode), Some(30.0));

        throughput.record(sample(Phase::Decode, 10, 1.0));
        assert_eq!(throughput.rate(Phase::Decode), Some(10.0), "the newest one wins");
    }

    #[test]
    fn phases_are_kept_apart() {
        let mut throughput = Throughput::default();
        throughput.record(sample(Phase::Prefill, 3266, 10.436));
        assert!(throughput.rate(Phase::Decode).is_none());
        let pp = throughput.rate(Phase::Prefill).expect("a prefill rate");
        assert_eq!(format_rate(pp), "313");
    }

    /// A zero-token or zero-second line is not a measurement, and must not
    /// erase the last real one.
    #[test]
    fn a_measurement_that_divides_to_nothing_is_ignored() {
        let mut throughput = Throughput::default();
        throughput.record(sample(Phase::Decode, 174, 9.38));
        throughput.record(sample(Phase::Decode, 0, 1.0));
        throughput.record(sample(Phase::Decode, 5, 0.0));
        assert_eq!(throughput.last(Phase::Decode).map(|s| s.tokens), Some(174));
    }

    #[test]
    fn rates_are_formatted_at_the_precision_that_carries_information() {
        assert_eq!(format_rate(17.573), "17.57");
        assert_eq!(format_rate(99.994), "99.99");
        assert_eq!(format_rate(1425.4), "1425");
    }

    /// Regression: the session pane pads the figure into a five-column field
    /// and padding cannot shrink what it is handed. Two rates escaped it — one
    /// just under a hundred, which `{:.2}` rounds up to `100.00`, and a prefill
    /// rate past five digits. Either shifted every column after it.
    #[test]
    fn no_rate_outgrows_the_field_reserved_for_it() {
        // The rounding boundary: under a hundred by the test, over it once
        // formatted.
        assert_eq!(format_rate(99.999), "100");
        assert_eq!(format_rate(99.995), "100");
        assert_eq!(format_rate(99.9949), "99.99");
        // Prefill on a small model reaches five digits and beyond.
        assert_eq!(format_rate(99_999.0), "99999");
        assert_eq!(format_rate(123_456.0), "123k");
        assert_eq!(format_rate(9_900_000.0), "10M");

        for rate in [
            0.0,
            0.004,
            1.0,
            17.573,
            99.994,
            99.995,
            99.999,
            100.0,
            999.5,
            1425.4,
            99_999.4,
            99_999.5,
            123_456.0,
            999_499.0,
            999_500.0,
            12_345_678.0,
            f64::MAX,
            -1.0,
        ] {
            let figure = format_rate(rate);
            assert!(figure.len() <= RATE_WIDTH, "{rate} formatted as {figure:?}");
        }
    }
}
