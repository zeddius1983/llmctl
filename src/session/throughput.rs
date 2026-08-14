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

/// The latest rates a session has reported.
#[derive(Default)]
pub struct Throughput {
    last_prefill: Option<Sample>,
    last_decode: Option<Sample>,
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
}
