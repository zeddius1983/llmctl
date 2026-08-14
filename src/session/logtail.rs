//! Incremental reading of a session's log file.
//!
//! The Session Manager polls every tick, so re-reading the whole log each time
//! is not an option: these files reach tens of megabytes over a long session.
//! A [`LogTail`] remembers where it stopped and returns only what was appended
//! since.

use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

/// How far back the first poll reaches. A session rediscovered after an llmctl
/// restart has a log full of history and no new lines until the next request,
/// so priming from the tail is what lets it show a rate immediately.
const PRIME_BYTES: u64 = 64 * 1024;

/// Ceiling on one poll's read, so a burst of server chatter cannot stall a
/// frame. Anything beyond it is picked up on the next poll.
const MAX_POLL_BYTES: u64 = 256 * 1024;

/// A position in a log file that only ever moves forward, except when the file
/// is replaced underneath it.
#[derive(Default)]
pub struct LogTail {
    offset: u64,
    started: bool,
}

impl LogTail {
    /// Lines appended since the last poll.
    ///
    /// Only whole lines are returned: a partial trailing write stays unconsumed
    /// and completes on a later poll, so a rate is never parsed out of half a
    /// line.
    pub fn poll(&mut self, path: &Path) -> Vec<String> {
        let Ok(file) = File::open(path) else { return Vec::new() };
        let Ok(len) = file.metadata().map(|meta| meta.len()) else { return Vec::new() };

        // Priming lands mid-line; that fragment is dropped below.
        let mut skip_first = false;
        if !self.started {
            self.started = true;
            self.offset = len.saturating_sub(PRIME_BYTES);
            skip_first = self.offset > 0;
        } else if len < self.offset {
            // Truncated or replaced — a restart writing a fresh log.
            self.offset = 0;
        }
        if len <= self.offset {
            return Vec::new();
        }

        let mut reader = BufReader::new(file);
        if reader.seek(SeekFrom::Start(self.offset)).is_err() {
            return Vec::new();
        }

        let budget = (len - self.offset).min(MAX_POLL_BYTES);
        let mut lines = Vec::new();
        let mut consumed = 0_u64;
        let mut buffer = Vec::new();
        while consumed < budget {
            buffer.clear();
            let Ok(read) = reader.read_until(b'\n', &mut buffer) else { break };
            if read == 0 {
                break;
            }
            if !buffer.ends_with(b"\n") {
                // A line still being written: leave it for next time.
                break;
            }
            consumed += read as u64;
            lines.push(String::from_utf8_lossy(&buffer).trim_end().to_string());
        }
        self.offset += consumed;
        if skip_first && !lines.is_empty() {
            lines.remove(0);
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn scratch(name: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("llmctl-{name}-{nonce}.log"))
    }

    fn append(path: &Path, text: &str) {
        let mut file =
            std::fs::OpenOptions::new().create(true).append(true).open(path).expect("open");
        file.write_all(text.as_bytes()).expect("write");
    }

    #[test]
    fn each_poll_returns_only_what_was_appended() {
        let path = scratch("tail");
        append(&path, "one\ntwo\n");

        let mut tail = LogTail::default();
        // A short log is read whole: priming cannot reach further back than it.
        assert_eq!(tail.poll(&path), vec!["one", "two"]);
        assert!(tail.poll(&path).is_empty(), "nothing new");

        append(&path, "three\n");
        assert_eq!(tail.poll(&path), vec!["three"]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_half_written_line_waits_for_the_rest_of_itself() {
        let path = scratch("partial");
        append(&path, "complete\npar");

        let mut tail = LogTail::default();
        assert_eq!(tail.poll(&path), vec!["complete"]);

        append(&path, "tial\n");
        assert_eq!(tail.poll(&path), vec!["partial"], "the line arrives once, whole");
        let _ = std::fs::remove_file(path);
    }

    /// A restart writes a fresh log at the same path; the old offset would then
    /// point past the end and skip everything the new server said.
    #[test]
    fn a_replaced_log_is_read_from_the_beginning() {
        let path = scratch("rotate");
        append(&path, "old and long enough to matter\n");

        let mut tail = LogTail::default();
        assert_eq!(tail.poll(&path).len(), 1);

        std::fs::write(&path, "new\n").expect("replace");
        assert_eq!(tail.poll(&path), vec!["new"]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_missing_log_is_not_an_error() {
        let mut tail = LogTail::default();
        assert!(tail.poll(Path::new("/nonexistent/llmctl-session.log")).is_empty());
    }
}
