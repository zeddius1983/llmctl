//! Prompt-processing (`pp`) and token-generation (`tg`) rates for a session,
//! read out of the log the server is already writing.
//!
//! Both runtimes print their own per-request timings when a request finishes:
//! exact tokens over the exact seconds spent producing them. llmctl times
//! nothing itself, which matters because wall-clock timing cannot tell
//! generating apart from idling — a server that produced 20 tokens in a second
//! and then sat quiet for half a minute is still running at 20 t/s.
//!
//! Averaging over the window therefore sums tokens and *active* seconds across
//! the requests that finished in it, rather than dividing by the window.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// How far back a displayed average reaches.
pub const WINDOW: Duration = Duration::from_secs(30);

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

/// A rate to display, and how much to trust it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rate {
    pub tokens_per_second: f64,
    /// Requests behind the figure.
    pub requests: usize,
    /// Nothing finished inside the window: this is the last known figure rather
    /// than a current one, and is rendered dimmed.
    pub stale: bool,
}

/// The rolling rates for one session.
#[derive(Default)]
pub struct Throughput {
    samples: VecDeque<(Instant, Sample)>,
    /// The last measurement of each phase, kept once the window empties so an
    /// idle session still shows what it managed last.
    last_prefill: Option<Sample>,
    last_decode: Option<Sample>,
}

impl Throughput {
    pub fn record(&mut self, sample: Sample) {
        self.record_at(sample, Instant::now());
    }

    fn record_at(&mut self, sample: Sample, now: Instant) {
        match sample.phase {
            Phase::Prefill => self.last_prefill = Some(sample),
            Phase::Decode => self.last_decode = Some(sample),
        }
        self.samples.push_back((now, sample));
        while self.samples.front().is_some_and(|(at, _)| now.duration_since(*at) > WINDOW) {
            self.samples.pop_front();
        }
    }

    /// The rate to show for `phase`: the window average, else the last thing
    /// that happened.
    pub fn rate(&self, phase: Phase) -> Option<Rate> {
        self.rate_at(phase, Instant::now())
    }

    fn rate_at(&self, phase: Phase, now: Instant) -> Option<Rate> {
        // Token-weighted: a long request counts for more than a two-token one,
        // which is what summing both sides before dividing does.
        let (tokens, seconds, requests) = self
            .samples
            .iter()
            .filter(|(at, sample)| sample.phase == phase && now.duration_since(*at) <= WINDOW)
            .fold((0_u64, 0.0_f64, 0_usize), |(tokens, seconds, count), (_, sample)| {
                (tokens + sample.tokens, seconds + sample.seconds, count + 1)
            });
        if requests > 0 && seconds > 0.0 && tokens > 0 {
            return Some(Rate {
                tokens_per_second: tokens as f64 / seconds,
                requests,
                stale: false,
            });
        }

        let last = self.last(phase)?;
        Some(Rate { tokens_per_second: last.rate()?, requests: 1, stale: true })
    }

    /// The last finished measurement for `phase`, for the detail pane.
    pub fn last(&self, phase: Phase) -> Option<Sample> {
        match phase {
            Phase::Prefill => self.last_prefill,
            Phase::Decode => self.last_decode,
        }
    }
}

/// Format a rate the way the runtimes do: two decimals while they carry
/// information, whole tokens once they do not.
pub fn format_rate(tokens_per_second: f64) -> String {
    if tokens_per_second < 100.0 {
        format!("{tokens_per_second:.2}")
    } else {
        format!("{tokens_per_second:.0}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(phase: Phase, tokens: u64, seconds: f64) -> Sample {
        Sample { phase, tokens, seconds }
    }

    #[test]
    fn the_window_average_is_token_weighted_not_per_request() {
        let mut throughput = Throughput::default();
        let now = Instant::now();
        // 300 tokens at 30 t/s, then 10 tokens at 10 t/s. Averaging the two
        // rates would say 20; the honest answer is 310 tokens over 11 seconds.
        throughput.record_at(sample(Phase::Decode, 300, 10.0), now);
        throughput.record_at(sample(Phase::Decode, 10, 1.0), now);

        let rate = throughput.rate_at(Phase::Decode, now).expect("a decode rate");
        assert!((rate.tokens_per_second - 310.0 / 11.0).abs() < 1e-9, "{rate:?}");
        assert_eq!(rate.requests, 2);
        assert!(!rate.stale);
    }

    /// The reason the runtimes' own active seconds are used rather than the
    /// window's span: a server that generated for one second and idled for
    /// twenty-five is still generating at its own speed.
    #[test]
    fn an_idle_gap_does_not_dilute_the_rate() {
        let mut throughput = Throughput::default();
        let start = Instant::now();
        throughput.record_at(sample(Phase::Decode, 20, 1.0), start);

        let later = start + Duration::from_secs(25);
        let rate = throughput.rate_at(Phase::Decode, later).expect("a decode rate");
        assert!((rate.tokens_per_second - 20.0).abs() < 1e-9, "{rate:?}");
        assert!(!rate.stale, "still inside the window");
    }

    #[test]
    fn a_measurement_older_than_the_window_survives_as_the_last_known_figure() {
        let mut throughput = Throughput::default();
        let start = Instant::now();
        throughput.record_at(sample(Phase::Decode, 20, 1.0), start);

        let later = start + WINDOW + Duration::from_secs(1);
        let rate = throughput.rate_at(Phase::Decode, later).expect("the last known rate");
        assert!((rate.tokens_per_second - 20.0).abs() < 1e-9);
        assert!(rate.stale, "outside the window it is history, and is shown as such");
    }

    #[test]
    fn phases_are_kept_apart() {
        let mut throughput = Throughput::default();
        let now = Instant::now();
        throughput.record_at(sample(Phase::Prefill, 3266, 10.436), now);
        assert!(throughput.rate_at(Phase::Decode, now).is_none());
        let pp = throughput.rate_at(Phase::Prefill, now).expect("a prefill rate");
        assert_eq!(format_rate(pp.tokens_per_second), "313");
    }

    #[test]
    fn rates_are_formatted_at_the_precision_that_carries_information() {
        assert_eq!(format_rate(17.573), "17.57");
        assert_eq!(format_rate(99.994), "99.99");
        assert_eq!(format_rate(481.2), "481");
    }
}
