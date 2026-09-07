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
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

const API: &str = "https://huggingface.co/api/models";

/// Report progress at most every 8 MB: often enough to look live, rarely enough
/// not to flood the UI channel.
const PROGRESS_INTERVAL: u64 = 8 * 1024 * 1024;
const DOWNLOAD_ATTEMPTS: u32 = 5;
const INITIAL_RETRY_DELAY: Duration = Duration::from_secs(1);
const API_READ_TIMEOUT: Duration = Duration::from_secs(30);
const DOWNLOAD_READ_TIMEOUT: Duration = Duration::from_secs(120);

pub fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(API_READ_TIMEOUT)
        .build()
}

fn download_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(DOWNLOAD_READ_TIMEOUT)
        .build()
}

#[derive(Clone, Copy)]
struct RetryPolicy {
    attempts: u32,
    initial_delay: Duration,
}

impl RetryPolicy {
    const DOWNLOAD: Self = Self { attempts: DOWNLOAD_ATTEMPTS, initial_delay: INITIAL_RETRY_DELAY };

    fn delay_after(self, attempt: u32) -> Duration {
        self.initial_delay.saturating_mul(1_u32 << attempt.saturating_sub(1).min(4))
    }
}

enum AttemptError {
    Retryable(anyhow::Error),
    Fatal(anyhow::Error),
}

struct DownloadRequest<'a> {
    repo: &'a str,
    revision: &'a str,
    file: &'a str,
    url: &'a str,
    dest: &'a Path,
    expected_bytes: u64,
    cancelled: &'a AtomicBool,
    retry: RetryPolicy,
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
    progress: impl FnMut(u64, u64),
) -> Result<bool> {
    let url = resolve_url(repo, revision, file);
    download_url(
        DownloadRequest {
            repo,
            revision,
            file,
            url: &url,
            dest,
            expected_bytes,
            cancelled,
            retry: RetryPolicy::DOWNLOAD,
        },
        progress,
    )
}

fn download_url(request: DownloadRequest<'_>, mut progress: impl FnMut(u64, u64)) -> Result<bool> {
    let DownloadRequest { repo, revision, file, url, dest, expected_bytes, cancelled, retry } =
        request;
    if cancelled.load(Ordering::Relaxed) {
        return Ok(false);
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating download directory {}", parent.display()))?;
    }

    for attempt in 1..=retry.attempts.max(1) {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(false);
        }

        let offset = dest.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        if offset == expected_bytes {
            progress(expected_bytes, expected_bytes);
            return Ok(true);
        }

        match download_attempt(url, repo, file, dest, expected_bytes, cancelled, &mut progress) {
            Ok(completed) => {
                if completed && attempt > 1 {
                    tracing::info!(
                        repository = repo,
                        revision,
                        file,
                        attempt,
                        "Hugging Face download recovered after retry"
                    );
                }
                return Ok(completed);
            }
            Err(AttemptError::Fatal(error)) => return Err(error),
            Err(AttemptError::Retryable(error)) if attempt == retry.attempts.max(1) => {
                tracing::error!(
                    repository = repo,
                    revision,
                    file,
                    attempt,
                    error = %format_args!("{error:#}"),
                    "Hugging Face download exhausted its retries"
                );
                return Err(error.context(format!(
                    "Hugging Face download failed after {} attempts",
                    retry.attempts.max(1)
                )));
            }
            Err(AttemptError::Retryable(error)) => {
                let delay = retry.delay_after(attempt);
                let resume_offset = dest.metadata().map(|metadata| metadata.len()).unwrap_or(0);
                tracing::warn!(
                    repository = repo,
                    revision,
                    file,
                    attempt,
                    next_attempt = attempt + 1,
                    resume_offset,
                    delay_ms = delay.as_millis(),
                    error = %format_args!("{error:#}"),
                    "retrying interrupted Hugging Face download"
                );
                if !wait_for_retry(delay, cancelled) {
                    return Ok(false);
                }
            }
        }
    }

    unreachable!("the retry loop always returns")
}

fn download_attempt(
    url: &str,
    repo: &str,
    file: &str,
    dest: &Path,
    expected_bytes: u64,
    cancelled: &AtomicBool,
    progress: &mut impl FnMut(u64, u64),
) -> std::result::Result<bool, AttemptError> {
    let existing = dest.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    let mut request = download_agent().get(url);
    if existing > 0 {
        request = request.set("Range", &format!("bytes={existing}-"));
    }
    if let Ok(token) = std::env::var("HF_TOKEN") {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = match request.call() {
        Ok(response) => response,
        Err(ureq::Error::Status(status, _)) => {
            let error = anyhow!("Hugging Face returned HTTP {status} for hf://{repo}/{file}");
            if retryable_http_status(status) {
                return Err(AttemptError::Retryable(error));
            }
            return Err(AttemptError::Fatal(error));
        }
        Err(ureq::Error::Transport(error)) => {
            return Err(AttemptError::Retryable(
                anyhow!(error).context(format!("downloading hf://{repo}/{file}")),
            ));
        }
    };

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
        .with_context(|| format!("opening partial download {}", dest.display()))
        .map_err(AttemptError::Fatal)?;

    let mut reader = response.into_reader();
    let mut downloaded = if resumed { existing } else { 0 };
    let mut reported = downloaded;
    let mut buffer = [0_u8; 256 * 1024];
    loop {
        if cancelled.load(Ordering::Relaxed) {
            output
                .flush()
                .context("flushing cancelled Hugging Face download")
                .map_err(AttemptError::Fatal)?;
            return Ok(false);
        }
        let read = reader
            .read(&mut buffer)
            .context("reading Hugging Face response")
            .map_err(AttemptError::Retryable)?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .context("writing Hugging Face download")
            .map_err(AttemptError::Fatal)?;
        downloaded = downloaded.saturating_add(read as u64);
        if downloaded.saturating_sub(reported) >= PROGRESS_INTERVAL {
            progress(downloaded.min(expected_bytes), expected_bytes);
            reported = downloaded;
        }
    }
    output.flush().context("flushing Hugging Face download").map_err(AttemptError::Fatal)?;

    if downloaded != expected_bytes {
        return Err(AttemptError::Retryable(anyhow!(
            "incomplete Hugging Face download for {file}: received {downloaded} of {expected_bytes} bytes"
        )));
    }
    progress(downloaded, expected_bytes);
    Ok(true)
}

fn retryable_http_status(status: u16) -> bool {
    status == 408 || status == 429 || (500..=599).contains(&status)
}

fn wait_for_retry(delay: Duration, cancelled: &AtomicBool) -> bool {
    const POLL_INTERVAL: Duration = Duration::from_millis(100);
    let mut remaining = delay;
    while !remaining.is_zero() {
        if cancelled.load(Ordering::Relaxed) {
            return false;
        }
        let sleep = remaining.min(POLL_INTERVAL);
        std::thread::sleep(sleep);
        remaining = remaining.saturating_sub(sleep);
    }
    !cancelled.load(Ordering::Relaxed)
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
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::thread;

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

    #[test]
    fn interrupted_transfer_retries_with_the_partial_file_range() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model.gguf", listener.local_addr().unwrap());
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let server_requests = Arc::clone(&requests);
        let server = thread::spawn(move || {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 1024];
                while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                    let read = stream.read(&mut buffer).unwrap();
                    assert!(read > 0);
                    request.extend_from_slice(&buffer[..read]);
                }
                server_requests.lock().unwrap().push(String::from_utf8(request).unwrap());

                if attempt == 0 {
                    stream
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\nhello")
                        .unwrap();
                } else {
                    stream
                        .write_all(b"HTTP/1.1 206 Partial Content\r\nContent-Length: 7\r\nContent-Range: bytes 5-11/12\r\nConnection: close\r\n\r\n world!")
                        .unwrap();
                }
            }
        });

        let root = std::env::temp_dir().join(format!(
            "llmctl-hf-retry-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let dest = root.join("model.gguf.part");
        let cancelled = AtomicBool::new(false);
        let completed = download_url(
            DownloadRequest {
                repo: "owner/repo",
                revision: "main",
                file: "model.gguf",
                url: &url,
                dest: &dest,
                expected_bytes: 12,
                cancelled: &cancelled,
                retry: RetryPolicy { attempts: 2, initial_delay: Duration::ZERO },
            },
            |_, _| {},
        )
        .unwrap();

        server.join().unwrap();
        assert!(completed);
        assert_eq!(std::fs::read(&dest).unwrap(), b"hello world!");
        let requests = requests.lock().unwrap();
        assert!(!requests[0].to_ascii_lowercase().contains("range:"));
        assert!(requests[1].to_ascii_lowercase().contains("range: bytes=5-"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn only_transient_http_statuses_are_retried() {
        for status in [408, 429, 500, 502, 503, 504, 599] {
            assert!(retryable_http_status(status), "HTTP {status}");
        }
        for status in [400, 401, 403, 404, 416] {
            assert!(!retryable_http_status(status), "HTTP {status}");
        }
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
