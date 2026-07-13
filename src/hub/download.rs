//! Streaming file downloads with resume support.
//!
//! Each artifact part streams into a `<file>.part` sibling and is renamed into
//! place once complete, so the model scanner never sees half-written GGUF
//! files. An interrupted or cancelled download keeps its `.part` file and
//! resumes from that offset (HTTP `Range`) on the next attempt.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::api::{self, FilePart};

/// How a download ended when it didn't fail.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Done,
    Cancelled,
}

const CHUNK: usize = 128 * 1024;

/// Fetch every part of an artifact into `dest_dir`, updating `received` with
/// cumulative byte progress and honouring `cancel` between chunks.
pub fn download_parts(
    repo: &str,
    parts: &[FilePart],
    dest_dir: &Path,
    received: &AtomicU64,
    cancel: &AtomicBool,
) -> Result<Outcome, String> {
    let mut base = 0u64;
    for part in parts {
        let dest = dest_dir.join(&part.rfilename);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("creating {}: {e}", parent.display()))?;
        }
        if let Ok(meta) = fs::metadata(&dest) {
            if part.size == 0 || meta.len() == part.size {
                base += meta.len();
                received.store(base, Ordering::Relaxed);
                continue; // already downloaded
            }
        }
        match fetch_part(repo, part, &dest, base, received, cancel)? {
            Outcome::Cancelled => return Ok(Outcome::Cancelled),
            Outcome::Done => base = received.load(Ordering::Relaxed),
        }
    }
    Ok(Outcome::Done)
}

/// Stream one part into `<dest>.part`, then rename it into place.
fn fetch_part(
    repo: &str,
    part: &FilePart,
    dest: &Path,
    base: u64,
    received: &AtomicU64,
    cancel: &AtomicBool,
) -> Result<Outcome, String> {
    let tmp = part_path(dest);
    let mut offset = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
    if part.size > 0 && offset >= part.size {
        offset = 0; // stale/bogus partial — restart
    }

    let url = api::resolve_url(repo, &part.rfilename);
    let request = api::prepare(agent().get(&url));
    let request =
        if offset > 0 { request.set("Range", &format!("bytes={offset}-")) } else { request };
    let response = match request.call() {
        Ok(response) => response,
        // 416: the server rejected our resume offset — restart from scratch.
        Err(ureq::Error::Status(416, _)) => {
            offset = 0;
            api::prepare(agent().get(&url)).call().map_err(api::describe)?
        }
        Err(err) => return Err(api::describe(err)),
    };
    if offset > 0 && response.status() != 206 {
        offset = 0; // server ignored the Range header; it sent the whole file
    }

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(offset == 0)
        .append(offset > 0)
        .open(&tmp)
        .map_err(|e| format!("opening {}: {e}", tmp.display()))?;
    received.store(base + offset, Ordering::Relaxed);

    let mut reader = response.into_reader();
    let mut buf = vec![0u8; CHUNK];
    let mut written = offset;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(Outcome::Cancelled); // keep .part for a later resume
        }
        let n =
            reader.read(&mut buf).map_err(|e| format!("downloading {}: {e}", part.rfilename))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| format!("writing {}: {e}", tmp.display()))?;
        written += n as u64;
        received.store(base + written, Ordering::Relaxed);
    }
    drop(file);

    if part.size > 0 && written != part.size {
        // Keep the .part so a retry resumes instead of starting over.
        return Err(format!(
            "{}: got {written} of {} bytes — connection dropped? retry to resume",
            part.rfilename, part.size
        ));
    }
    fs::rename(&tmp, dest).map_err(|e| format!("finalizing {}: {e}", dest.display()))?;
    Ok(Outcome::Done)
}

/// The in-progress sibling of a destination file (`model.gguf` → `model.gguf.part`).
fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".part");
    dest.with_file_name(name)
}

/// Downloads reuse the API agent's connect timeout but keep a generous read
/// timeout — large-model transfers stall briefly on slow disks/CDN shifts.
fn agent() -> &'static ureq::Agent {
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(15))
            .timeout_read(std::time::Duration::from_secs(60))
            .build()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_path_appends_the_part_suffix() {
        assert_eq!(
            part_path(Path::new("/dl/org/repo/model-Q4.gguf")),
            PathBuf::from("/dl/org/repo/model-Q4.gguf.part")
        );
    }

    #[test]
    fn already_complete_parts_are_skipped_and_counted() {
        let nonce =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("llmctl-dl-{nonce}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("done.gguf"), b"GGUF").unwrap();

        let received = AtomicU64::new(0);
        let cancel = AtomicBool::new(false);
        let parts = vec![FilePart { rfilename: "done.gguf".into(), size: 4 }];
        let outcome = download_parts("org/repo", &parts, &dir, &received, &cancel).unwrap();
        assert_eq!(outcome, Outcome::Done);
        assert_eq!(received.load(Ordering::Relaxed), 4);
        fs::remove_dir_all(dir).unwrap();
    }
}
