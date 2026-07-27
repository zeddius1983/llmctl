//! Shared Hugging Face plumbing: the resumable file transfer, the repository
//! tree API, and URL construction.
//!
//! Two very different consumers download from the Hub. `discovery::online`
//! fetches GGUF blobs into the standard Hugging Face cache for llama.cpp, and
//! `runtime::flm` fetches a FastFlowLM model's files into `flm`'s own model
//! directory. They disagree about *where* bytes land and how completion is
//! recorded, but the transfer itself — `Range` resume, `HF_TOKEN` auth,
//! cancellation, size verification — is the same job, and lives here.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use serde::Deserialize;

const API: &str = "https://huggingface.co/api/models";

/// Report progress at most every 8 MB: often enough to look live, rarely enough
/// not to flood the UI channel.
const PROGRESS_INTERVAL: u64 = 8 * 1024 * 1024;

pub fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(10))
        .timeout_read(std::time::Duration::from_secs(30))
        .build()
}

/// One file in a repository tree, as reported by the Hub API.
#[derive(Debug, Clone, Deserialize)]
pub struct TreeEntry {
    pub path: String,
    /// Byte size. Present for plain and LFS files alike, which is why the tree
    /// endpoint is preferred over the LFS metadata on the model detail.
    #[serde(default)]
    pub size: u64,
}

/// `GET /api/models/<repo>/tree/<revision>` — every file in the repository.
pub fn tree(repo: &str, revision: &str) -> Result<Vec<TreeEntry>> {
    let url = format!("{API}/{}/tree/{}", encode_url_path(repo), encode_url_path(revision));
    let mut request = agent().get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    request
        .call()
        .with_context(|| format!("listing hf://{repo}@{revision}"))?
        .into_json()
        .with_context(|| format!("parsing the file list for hf://{repo}@{revision}"))
}

/// Download `file` from `repo` at `revision` into `dest`, resuming if `dest`
/// already holds a partial body.
///
/// `dest` is expected to be a scratch path that the caller renames into place
/// once this returns `Ok(true)`; that rename is what makes completion atomic and
/// observable. Returns `Ok(false)` if `cancelled` was set mid-transfer, leaving
/// the partial file intact for a later resume.
pub fn download_file(
    repo: &str,
    revision: &str,
    file: &str,
    dest: &Path,
    expected_bytes: u64,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<bool> {
    if cancelled.load(Ordering::Relaxed) {
        return Ok(false);
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating download directory {}", parent.display()))?;
    }

    let existing = dest.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    let url = resolve_url(repo, revision, file);
    let mut request = agent().get(&url);
    if existing > 0 {
        request = request.set("Range", &format!("bytes={existing}-"));
    }
    if let Ok(token) = std::env::var("HF_TOKEN") {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = request.call().with_context(|| format!("downloading hf://{repo}/{file}"))?;

    // A server that ignores the Range header answers 200 with the whole body,
    // so the partial has to be discarded rather than appended to.
    let resumed = existing > 0 && response.status() == 206;
    let mut options = OpenOptions::new();
    options.create(true).write(true);
    if resumed {
        options.append(true);
    } else {
        options.truncate(true);
    }
    let mut output = options
        .open(dest)
        .with_context(|| format!("opening partial download {}", dest.display()))?;

    let mut reader = response.into_reader();
    let mut downloaded = if resumed { existing } else { 0 };
    let mut reported = downloaded;
    let mut buffer = [0_u8; 256 * 1024];
    loop {
        if cancelled.load(Ordering::Relaxed) {
            output.flush().context("flushing cancelled Hugging Face download")?;
            return Ok(false);
        }
        let read = reader.read(&mut buffer).context("reading Hugging Face response")?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read]).context("writing Hugging Face download")?;
        downloaded = downloaded.saturating_add(read as u64);
        if downloaded.saturating_sub(reported) >= PROGRESS_INTERVAL {
            progress(downloaded.min(expected_bytes), expected_bytes);
            reported = downloaded;
        }
    }
    output.flush().context("flushing Hugging Face download")?;

    if downloaded != expected_bytes {
        anyhow::bail!(
            "incomplete Hugging Face download for {file}: received {downloaded} of {expected_bytes} bytes"
        );
    }
    progress(downloaded, expected_bytes);
    Ok(true)
}

pub fn resolve_url(repo: &str, revision: &str, file: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        encode_url_path(repo),
        encode_url_path(revision),
        encode_url_path(file)
    )
}

pub fn encode_url_path(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_encode_only_what_needs_it() {
        assert_eq!(
            resolve_url("owner/repo", "main", "dir/model.gguf"),
            "https://huggingface.co/owner/repo/resolve/main/dir/model.gguf"
        );
        // A pinned revision keeps its dots and dashes verbatim.
        assert_eq!(
            resolve_url("FastFlowLM/Qwen3-0.6B-NPU2", "v0.9.22-faster-q4-1", "model.q4nx"),
            "https://huggingface.co/FastFlowLM/Qwen3-0.6B-NPU2/resolve/v0.9.22-faster-q4-1/model.q4nx"
        );
        // Path separators survive; anything else unsafe is percent-encoded.
        assert_eq!(
            resolve_url("owner/model", "main", "nested/model Q4_K_M.gguf"),
            "https://huggingface.co/owner/model/resolve/main/nested/model%20Q4_K_M.gguf"
        );
    }

    /// Live check against the real Hub: the tree endpoint must report a size for
    /// plain files (`config.json`) as well as LFS ones (`model.q4nx`), since the
    /// FastFlowLM downloader needs both. Ignored by default.
    #[test]
    #[ignore = "hits the network; run with --ignored"]
    fn tree_reports_sizes_for_plain_and_lfs_files() {
        let files = tree("FastFlowLM/Qwen3-0.6B-NPU2", "v0.9.22-faster-q4-1").unwrap();
        let by_name = |name: &str| files.iter().find(|f| f.path == name).cloned();

        let config = by_name("config.json").expect("config.json");
        assert!(config.size > 0, "plain file reported no size");

        let weights = by_name("model.q4nx").expect("model.q4nx");
        assert!(weights.size > 100_000_000, "LFS weights reported {} bytes", weights.size);
    }
}
