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
}

impl Sample {
    pub fn rate(&self) -> Option<f64> {
        (self.seconds > 0.0 && self.tokens > 0).then(|| self.tokens as f64 / self.seconds)
    }
}

/// How many finished measurements a session keeps for the Monitor.
const RECENT_LEN: usize = 60;

/// The latest rates a session has reported, and the measurements before them.
#[derive(Default)]
pub struct Throughput {
    last_prefill: Option<Sample>,
    last_decode: Option<Sample>,
    /// Oldest first, both phases interleaved as the server reported them.
    recent: std::collections::VecDeque<Sample>,
}

/// One request as the server summarized it: its prompt processing and its
/// generation, either of which a runtime may not have reported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Request {
    pub prefill: Option<Sample>,
    pub decode: Option<Sample>,
}

impl Throughput {
    pub fn record(&mut self, sample: Sample) {
        // A measurement that divides to nothing is not one; keeping it would
        // replace a real figure with a blank.
        if sample.rate().is_none() {
            return;
        }
        match sample.phase {
            Phase::Prefill => self.last_prefill = Some(sample),
            Phase::Decode => self.last_decode = Some(sample),
        }
        if self.recent.len() == RECENT_LEN {
            self.recent.pop_front();
        }
        self.recent.push_back(sample);
    }

    /// The rates of the recent measurements for `phase`, oldest first — one per
    /// request, not per second, so an idle server's history does not decay.
    pub fn rates(&self, phase: Phase) -> Vec<Option<f64>> {
        self.recent.iter().filter(|s| s.phase == phase).map(Sample::rate).collect()
    }

    /// The recent requests, newest first.
    ///
    /// Both runtimes report a request's prefill before its decode, so a
    /// prefill followed by a decode is one request. Anything else stands alone
    /// rather than being paired with a neighbour it did not belong to.
    pub fn requests(&self) -> Vec<Request> {
        let mut requests = Vec::new();
        let mut pending: Option<Sample> = None;
        for &sample in &self.recent {
            match sample.phase {
                Phase::Prefill => {
                    if let Some(prefill) = pending.replace(sample) {
                        requests.push(Request { prefill: Some(prefill), decode: None });
                    }
                }
                Phase::Decode => {
                    requests.push(Request { prefill: pending.take(), decode: Some(sample) })
                }
            }
        }
        if let Some(prefill) = pending {
            requests.push(Request { prefill: Some(prefill), decode: None });
        }
        requests.reverse();
        requests
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
        Sample { phase, tokens, seconds }
    }

    #[test]
    fn recent_measurements_pair_into_requests_newest_first() {
        let mut throughput = Throughput::default();
        throughput.record(sample(Phase::Prefill, 339, 1.6));
        throughput.record(sample(Phase::Decode, 53, 1.35));
        // A prefill whose decode never came (a cancelled request) stands alone.
        throughput.record(sample(Phase::Prefill, 18, 0.2));
        throughput.record(sample(Phase::Prefill, 40, 0.4));
        throughput.record(sample(Phase::Decode, 6, 0.2));

        let requests = throughput.requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0].prefill.map(|s| s.tokens), Some(40));
        assert_eq!(requests[0].decode.map(|s| s.tokens), Some(6));
        assert_eq!(
            requests[1],
            Request { prefill: Some(sample(Phase::Prefill, 18, 0.2)), decode: None }
        );
        assert_eq!(requests[2].decode.map(|s| s.tokens), Some(53));

        assert_eq!(throughput.rates(Phase::Decode), vec![Some(53.0 / 1.35), Some(30.0)]);
    }

    #[test]
    fn the_history_is_bounded() {
        let mut throughput = Throughput::default();
        for _ in 0..RECENT_LEN + 10 {
            throughput.record(sample(Phase::Decode, 10, 1.0));
        }
        assert_eq!(throughput.rates(Phase::Decode).len(), RECENT_LEN);
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
