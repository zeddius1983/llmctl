//! Incremental reading of a session's log file.
//!
//! The Session Manager polls every tick, so re-reading the whole log each time
//! is not an option: these files reach tens of megabytes over a long session.
//! A [`LogTail`] remembers where it stopped and returns only what was appended
//! since.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
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
    partial: Vec<u8>,
    discard_line: bool,
    identity: Option<(u64, u64)>,
}

impl LogTail {
    /// Lines appended since the last poll.
    ///
    /// Only whole lines are returned. Partial lines are buffered up to 64 KiB;
    /// oversized lines are discarded. Each poll reads at most 256 KiB.
    pub fn poll(&mut self, path: &Path) -> Vec<String> {
        let Ok(mut file) = File::open(path) else { return Vec::new() };
        let Ok(metadata) = file.metadata() else { return Vec::new() };
        let len = metadata.len();
        let identity = (metadata.dev(), metadata.ino());
        if !self.started {
            self.started = true;
            self.offset = len.saturating_sub(PRIME_BYTES);
            self.discard_line = self.offset > 0;
        } else if self.identity != Some(identity) || len < self.offset {
            self.offset = 0;
            self.partial.clear();
            self.discard_line = false;
        }
        self.identity = Some(identity);
        if len <= self.offset || file.seek(SeekFrom::Start(self.offset)).is_err() {
            return Vec::new();
        }
        let budget = (len - self.offset).min(MAX_POLL_BYTES);
        let mut bytes = Vec::with_capacity(budget as usize);
        // The reader, rather than the outer line loop, enforces the byte budget.
        if file.take(budget).read_to_end(&mut bytes).is_err() {
            return Vec::new();
        }
        self.offset += bytes.len() as u64;
        let mut lines = Vec::new();
        for segment in bytes.split_inclusive(|byte| *byte == b'\n') {
            let complete = segment.ends_with(b"\n");
            if self.discard_line || self.partial.len() + segment.len() > PRIME_BYTES as usize {
                // Oversized lines are skipped in bounded chunks. Retaining an
                // arbitrary suffix could turn it into a false timing sample.
                self.partial.clear();
                self.discard_line = !complete;
                continue;
            }
            self.partial.extend_from_slice(segment);
            if complete {
                lines.push(String::from_utf8_lossy(&self.partial).trim_end().to_string());
                self.partial.clear();
            }
        }
        lines
    }
}

/// Variation selectors (VS1–VS16 and the supplementary block) change how the
/// preceding character is drawn without being drawn themselves — which is
/// exactly what makes their width unmeasurable.
pub(crate) fn is_variation_selector(c: char) -> bool {
    matches!(c, '\u{FE00}'..='\u{FE0F}' | '\u{E0100}'..='\u{E01EF}')
}

/// What a terminal would show for one log line.
///
/// A carriage return rewrites the row from column 0, so a progress bar that
/// ticked a hundred times arrives as one line holding a hundred states. Only the
/// last one was ever visible, and it is the only one worth showing in a log
/// tail — the rest would be a wall of `Downloading: 0.3% … 0.5% …`.
pub(crate) fn visible_line(raw: &str) -> String {
    raw.split('\r')
        .map(strip_control)
        .filter(|segment| !segment.trim().is_empty())
        .last()
        .unwrap_or_default()
}

/// Drop ANSI escape sequences, stray control bytes, and variation selectors,
/// keeping printable text.
///
/// Left in place, `ESC[K` (erase to end of line) would wipe the rest of the row
/// including the log pane's border, and `ESC[?25l` would hide the cursor for the
/// rest of the session.
///
/// Variation selectors go for a subtler reason. `⬇️` is `U+2B07 U+FE0F`, and the
/// selector asks for emoji presentation, which a terminal draws two columns
/// wide — but `unicode-width` still measures the pair as one. The renderer then
/// lays the row out one cell narrower than it actually paints, and everything to
/// its right, border included, is overwritten. Dropping the selector leaves a
/// bare `U+2B07`, which measures and draws as one column. Characters that are
/// emoji by default (`🔗`, `🔒`) carry no selector and already measure correctly.
fn strip_control(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            if is_variation_selector(c) {
                continue;
            }
            if c == '\t' || !c.is_control() {
                out.push(c);
            }
            continue;
        }
        match chars.next() {
            // CSI: parameter bytes, then a final byte in @..~ .
            Some('[') => {
                for c in chars.by_ref() {
                    if ('\x40'..='\x7e').contains(&c) {
                        break;
                    }
                }
            }
            // OSC: runs until BEL or a String Terminator.
            Some(']') => {
                while let Some(c) = chars.next() {
                    if c == '\x07' {
                        break;
                    }
                    if c == '\x1b' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            // Any other escape is two characters; both are already consumed.
            _ => {}
        }
    }
    out
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
    fn oversized_lines_obey_the_byte_budget_and_do_not_hide_following_lines() {
        let path = scratch("oversized");
        append(&path, "");
        let mut tail = LogTail::default();
        assert!(tail.poll(&path).is_empty());
        append(&path, &"x".repeat(MAX_POLL_BYTES as usize * 2 + 10));
        append(&path, "\nvalid sample\n");
        for expected in [MAX_POLL_BYTES, MAX_POLL_BYTES * 2] {
            assert!(tail.poll(&path).is_empty());
            assert_eq!(tail.offset, expected);
            assert!(tail.partial.len() <= PRIME_BYTES as usize);
        }
        assert_eq!(tail.poll(&path), ["valid sample"]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_larger_replacement_resets_the_offset_and_partial_line() {
        let path = scratch("larger-replacement");
        append(&path, "partial");
        let mut tail = LogTail::default();
        assert!(tail.poll(&path).is_empty());
        let replacement = scratch("new-inode");
        append(&replacement, "new complete line\n");
        std::fs::rename(replacement, &path).unwrap();
        assert_eq!(tail.poll(&path), ["new complete line"]);
        std::fs::remove_file(path).unwrap();
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

    /// Regression: `flm` writes progress to its log the way it would to a
    /// terminal. A bare carriage return sent the cursor back to column 0 and
    /// `ESC[K` erased to end of line, so those rows overwrote the log pane's
    /// borders and the text beside them.
    #[test]
    fn log_lines_are_reduced_to_what_a_terminal_would_show() {
        // Verbatim bytes from a real FastFlowLM session log.
        let overall = "\r[FLM]  Overall progress:  1/6 files";
        assert_eq!(visible_line(overall), "[FLM]  Overall progress:  1/6 files");

        // A progress bar redrawn in place: only the final state was ever visible.
        let progress = "\u{1b}[?25l\r\u{1b}[K[FLM]  Downloading: 0.0% (0.0MB / 2340.0MB)\
                        \r\u{1b}[K[FLM]  Downloading: 0.3% (6.0MB / 2340.0MB)\
                        \r\u{1b}[K[FLM]  Downloading: 100.0% (2340.0MB / 2340.0MB)\u{1b}[?25h";
        assert_eq!(visible_line(progress), "[FLM]  Downloading: 100.0% (2340.0MB / 2340.0MB)");

        // Cursor show/hide around plain text leaves just the text.
        assert_eq!(
            visible_line("\u{1b}[?25l\u{1b}[?25h[FLM]  Checking Hash..."),
            "[FLM]  Checking Hash..."
        );

        // Nothing survives a line that was only control bytes.
        assert_eq!(visible_line("\u{1b}[?25l\u{1b}[?25h"), "");

        // Ordinary lines pass through untouched, including colour codes.
        assert_eq!(visible_line("plain server line"), "plain server line");
        assert_eq!(visible_line("\u{1b}[31mred\u{1b}[0m text"), "red text");
    }

    /// Regression: a variation selector makes a character draw two columns wide
    /// while `unicode-width` still measures one, so the log pane laid out rows
    /// narrower than it painted them and clobbered its own border.
    #[test]
    fn rendered_log_width_matches_what_the_terminal_draws() {
        use unicode_width::UnicodeWidthStr;

        // Verbatim from a FastFlowLM session log: U+2B07 followed by U+FE0F.
        let arrow = visible_line("[\u{2B07}\u{FE0F} ]  Incoming Request: GET");
        assert_eq!(arrow, "[\u{2B07} ]  Incoming Request: GET");
        // The selector is gone, so the measured width is now the drawn width.
        assert!(!arrow.chars().any(is_variation_selector));
        assert_eq!(arrow.width(), arrow.chars().count());

        // Characters that are emoji by default carry no selector and already
        // measure correctly at two columns; they must survive untouched.
        let link = visible_line("[\u{1F517} ]  TCP connection established");
        assert!(link.starts_with("[\u{1F517}"));
        assert_eq!(link.width(), link.chars().count() + 1);
    }

    #[test]
    fn no_rendered_log_line_can_carry_control_bytes() {
        // Whatever a server writes, nothing that could move the cursor or erase
        // the frame may reach the terminal.
        let nasty = "\u{1b}[2J\u{1b}]0;title\u{7}\rone\u{1b}[Ktwo\u{0}\u{8}";
        let rendered = visible_line(nasty);
        assert!(
            !rendered.chars().any(|c| c.is_control() && c != '\t'),
            "control byte survived: {rendered:?}"
        );
        assert_eq!(rendered, "onetwo");
    }
}
