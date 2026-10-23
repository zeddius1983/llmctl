//! A short, fixed-length history of one measured quantity, for the Monitor's
//! sparklines.
//!
//! Gaps are kept as gaps. A sample the source could not produce — a `/proc`
//! read that raced the process exiting, a sysfs file that vanished — is stored
//! as `None` and drawn as blank, never as a zero that would read as a real dip.

use std::collections::VecDeque;

/// How many samples a series keeps: a minute at the Monitor's one-second
/// cadence, and more than the widest pane draws.
pub const HISTORY_LEN: usize = 60;

#[derive(Debug, Clone, Default)]
pub struct Series {
    samples: VecDeque<Option<f64>>,
}

impl Series {
    pub fn push(&mut self, value: Option<f64>) {
        if self.samples.len() == HISTORY_LEN {
            self.samples.pop_front();
        }
        self.samples.push_back(value);
    }

    /// Oldest first.
    pub fn values(&self) -> Vec<Option<f64>> {
        self.samples.iter().copied().collect()
    }

    /// The newest sample, if it was measured.
    #[cfg(test)]
    pub fn last(&self) -> Option<f64> {
        self.samples.back().copied().flatten()
    }

    /// The largest measured sample, for scaling a sparkline.
    pub fn max(&self) -> Option<f64> {
        self.samples.iter().flatten().copied().reduce(f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_series_keeps_the_newest_minute_and_its_gaps() {
        let mut series = Series::default();
        for i in 0..HISTORY_LEN + 5 {
            series.push(Some(i as f64));
        }
        let values = series.values();
        assert_eq!(values.len(), HISTORY_LEN);
        assert_eq!(values[0], Some(5.0), "the oldest samples fall off the front");
        assert_eq!(series.last(), Some((HISTORY_LEN + 4) as f64));

        series.push(None);
        assert_eq!(series.last(), None, "an unmeasured sample is not the previous one");
        assert_eq!(series.max(), Some((HISTORY_LEN + 4) as f64), "and does not count as zero");
    }
}
