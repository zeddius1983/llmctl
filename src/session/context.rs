//! How much of its context a session's last request used, read out of the log
//! the server is already writing, like its throughput (ADR-016).
//!
//! Both figures come from the server. llama.cpp prints its per-slot context at
//! load and a slot's token count when a request releases it; FastFlowLM reports
//! both in every closing usage chunk. The token count is the slot's whole
//! context, cached prefix included, which the prompt-eval count is not.

use std::io::Read;
use std::path::Path;

/// How much of a log's start is searched for its capacity. llama.cpp states it
/// a few dozen lines in, after the model's metadata; even a verbose load
/// leaves it well inside this.
const HEAD_BYTES: u64 = 256 * 1024;

/// The context capacity a server stated at load, read from the start of its
/// log.
///
/// A rediscovered session's log tail resumes near the end of the file, and
/// llama.cpp states the capacity only once, at load. Past a few requests that
/// line is out of the tail's reach, and without this the session would never
/// learn its context size again. FastFlowLM restates its capacity with every
/// request, so the tail recovers it unaided; this finds nothing for it and
/// costs one bounded read.
pub fn capacity_from_head(log: &Path, parse: fn(&str) -> Vec<Reading>) -> Option<u64> {
    let mut head = Vec::new();
    std::fs::File::open(log).ok()?.take(HEAD_BYTES).read_to_end(&mut head).ok()?;
    String::from_utf8_lossy(&head)
        .lines()
        .flat_map(parse)
        .filter_map(|reading| match reading {
            Reading::Size(size) if size > 0 => Some(size),
            _ => None,
        })
        .next_back()
}

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

    /// Regression: a session rediscovered with a log past the tail's priming
    /// window lost its context size for good, since llama.cpp states it once.
    #[test]
    fn capacity_is_recovered_from_the_head_of_a_long_log() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0);
        let log = std::env::temp_dir().join(format!("llmctl-context-head-{nonce}.log"));
        let mut text = String::from(
            "0.09.324.355 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 16384, kv_unified = 'true'\n",
        );
        let release = "20.20.352.451 I slot      release: id  0 | task 2 | stop processing: n_tokens = 48, truncated = 0\n";
        while text.len() < 512 * 1024 {
            text.push_str(release);
        }
        std::fs::write(&log, text).unwrap();

        let parse = crate::runtime::log_parsers(crate::runtime::llama_cpp::NAME).context;
        assert_eq!(capacity_from_head(&log, parse), Some(16384));
        assert_eq!(capacity_from_head(&log.with_extension("missing"), parse), None);

        let _ = std::fs::remove_file(log);
    }

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
