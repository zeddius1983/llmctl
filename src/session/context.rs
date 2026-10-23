//! How much of its context a session's last request used, read out of the log
//! the server is already writing, like its throughput (ADR-016).
//!
//! Both figures come from the server. llama.cpp prints its per-slot context at
//! load and a slot's token count when a request releases it; FastFlowLM reports
//! both in every closing usage chunk. The token count is the slot's whole
//! context, cached prefix included, which the prompt-eval count is not.

/// One fact a log line states about the context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reading {
    /// Tokens one request can hold.
    Size(u64),
    /// Tokens the last finished request occupied.
    Used(u64),
}

/// The latest of each reading.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContextFill {
    pub size: Option<u64>,
    pub used: Option<u64>,
}

impl ContextFill {
    pub fn record(&mut self, reading: Reading) {
        match reading {
            Reading::Size(size) if size > 0 => self.size = Some(size),
            Reading::Size(_) => {}
            Reading::Used(used) => self.used = Some(used),
        }
    }

    /// Used over size, once both are known.
    pub fn fraction(&self) -> Option<f64> {
        let size = self.size?;
        Some(self.used.unwrap_or(0) as f64 / size as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_latest_readings_win_and_a_zero_size_is_ignored() {
        let mut fill = ContextFill::default();
        assert_eq!(fill.fraction(), None, "nothing to measure against");
        fill.record(Reading::Size(16384));
        assert_eq!(fill.fraction(), Some(0.0), "a known size with no request yet is empty");
        fill.record(Reading::Used(8192));
        fill.record(Reading::Size(0));
        assert_eq!(fill.fraction(), Some(0.5));
        fill.record(Reading::Used(97));
        assert_eq!(fill.used, Some(97));
    }
}
