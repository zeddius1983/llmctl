//! Hugging Face hub integration: online model search, GGUF file listing, and
//! background downloads into the configured download directory.
//!
//! No async runtime (ADR-007): every network call runs on a short-lived worker
//! thread that reports back over an `mpsc` channel drained by the app's poll
//! loop. Download progress is shared through atomics so the UI can render it
//! every frame without extra messages.

pub mod api;
pub mod download;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, mpsc};

pub use api::{FilePart, HubFile, HubModel, HubRepo};

/// A completed background operation, delivered to the app's event loop.
pub enum HubEvent {
    /// Search results for the given request epoch (stale epochs are dropped).
    SearchResults { epoch: u64, result: Result<Vec<HubModel>, String> },
    /// The GGUF file listing for a repository.
    RepoFiles { repo: String, result: Result<HubRepo, String> },
    /// A download finished (successfully, cancelled, or failed).
    DownloadDone { id: u64, result: Result<download::Outcome, String> },
}

pub fn spawn_search(tx: mpsc::Sender<HubEvent>, epoch: u64, query: String) {
    std::thread::spawn(move || {
        let result = api::search_models(&query);
        let _ = tx.send(HubEvent::SearchResults { epoch, result });
    });
}

pub fn spawn_repo_files(tx: mpsc::Sender<HubEvent>, repo: String) {
    std::thread::spawn(move || {
        let result = api::repo_files(&repo);
        let _ = tx.send(HubEvent::RepoFiles { repo, result });
    });
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_download(
    tx: mpsc::Sender<HubEvent>,
    id: u64,
    repo: String,
    parts: Vec<FilePart>,
    dest_dir: PathBuf,
    received: Arc<AtomicU64>,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let result = download::download_parts(&repo, &parts, &dest_dir, &received, &cancel);
        let _ = tx.send(HubEvent::DownloadDone { id, result });
    });
}

/// The on-disk directory a repository downloads into:
/// `<root>/<owner>/<repo>`, with path-hostile characters replaced.
pub fn repo_dir(root: &Path, repo_id: &str) -> PathBuf {
    let mut dir = root.to_path_buf();
    for component in repo_id.split('/').filter(|c| !c.is_empty()) {
        dir.push(sanitize(component));
    }
    dir
}

fn sanitize(raw: &str) -> String {
    let clean: String =
        raw.chars().map(|c| if c == '/' || c == '\\' || c == '\0' { '_' } else { c }).collect();
    match clean.as_str() {
        "" | "." | ".." => "_".into(),
        _ => clean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_dir_nests_owner_and_repo_under_the_root() {
        assert_eq!(
            repo_dir(Path::new("/dl"), "unsloth/Qwen3-4B-GGUF"),
            PathBuf::from("/dl/unsloth/Qwen3-4B-GGUF")
        );
    }

    #[test]
    fn repo_dir_neutralizes_traversal_components() {
        assert_eq!(repo_dir(Path::new("/dl"), "../evil"), PathBuf::from("/dl/_/evil"));
        assert_eq!(repo_dir(Path::new("/dl"), "a//b"), PathBuf::from("/dl/a/b"));
    }
}
