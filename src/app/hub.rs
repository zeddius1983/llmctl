//! The online Hugging Face subtree of the model browser.
//!
//! The hub is browsed like a local folder: a virtual `online ▸ huggingface`
//! directory whose children are repositories (trending, or the last committed
//! `/` search) and whose repository folders list downloadable GGUF artifacts.
//! This module owns that virtual tree: node synthesis, fetch triggers, the
//! download lifecycle, and the event drain that applies worker-thread results.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::domain::{Model, RemoteKind, human_size};
use crate::hub::{self, HubEvent, HubFile, HubModel, HubRepo, download};

use super::{App, Pane, SearchMode};

/// Top-level virtual folder holding online sources.
pub const ONLINE: &str = "online";
/// The Hugging Face folder under [`ONLINE`].
pub const HUGGINGFACE: &str = "huggingface";

/// Lifecycle of one artifact download.
pub enum DownloadState {
    Running,
    Done,
    Cancelled,
    Failed(String),
}

/// A tracked download; progress is read from the shared counter every frame.
pub struct Download {
    pub id: u64,
    pub repo: String,
    /// Logical artifact name (shard suffix stripped).
    pub file: String,
    /// Expected total bytes (0 when the API did not report sizes).
    pub total: u64,
    pub received: Arc<AtomicU64>,
    pub cancel: Arc<AtomicBool>,
    pub state: DownloadState,
}

impl Download {
    pub fn received_bytes(&self) -> u64 {
        self.received.load(Ordering::Relaxed)
    }

    pub fn cancelling(&self) -> bool {
        matches!(self.state, DownloadState::Running) && self.cancel.load(Ordering::Relaxed)
    }

    /// `1.2 GB / 4.5 GB (27%)`, degrading gracefully when the total is unknown.
    pub fn progress_label(&self) -> String {
        let received = self.received_bytes();
        if self.total == 0 {
            return human_size(received);
        }
        let percent = (received.saturating_mul(100) / self.total).min(100);
        format!("{} / {} ({percent}%)", human_size(received), human_size(self.total))
    }
}

pub struct HubState {
    /// Repositories listed under `online/huggingface` — trending by default,
    /// or the results of the last committed `/` search.
    pub results: Vec<HubModel>,
    /// The committed query behind `results` (empty = trending).
    pub query: String,
    /// Cached GGUF listings per repository id.
    pub listings: HashMap<String, HubRepo>,
    /// A short in-flight notice shown in the footer, cleared when work lands.
    pub loading: Option<String>,
    pub error: Option<String>,
    pub downloads: Vec<Download>,
    /// Monotonic search id so stale responses are dropped.
    pub(super) epoch: u64,
    /// Repository listings currently being fetched.
    pending_repos: HashSet<String>,
    /// Whether the initial trending fetch has been kicked off.
    fetched: bool,
    next_download_id: u64,
}

impl HubState {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            query: String::new(),
            listings: HashMap::new(),
            loading: None,
            error: None,
            downloads: Vec::new(),
            epoch: 0,
            pending_repos: HashSet::new(),
            fetched: false,
            next_download_id: 1,
        }
    }

    /// The download tracking `file` within `repo`, if any.
    pub fn download_for(&self, repo: &str, file: &str) -> Option<&Download> {
        self.downloads.iter().find(|d| d.repo == repo && d.file == file)
    }
}

/// Does this catalog prefix lie inside the online subtree?
pub fn in_hub_tree(prefix: &[String]) -> bool {
    prefix.first().map(String::as_str) == Some(ONLINE)
}

/// The repository id a hub prefix points into (`online/huggingface/<repo>`).
fn repo_of(prefix: &[String]) -> Option<&str> {
    (prefix.len() >= 3 && prefix[1] == HUGGINGFACE).then(|| prefix[2].as_str())
}

fn dir_node(catalog_path: Vec<String>, remote: Option<RemoteKind>) -> Model {
    Model {
        id: format!("hub:{}", catalog_path.join("/")),
        name: catalog_path.last().cloned().unwrap_or_default(),
        path: PathBuf::new(),
        shard_paths: Vec::new(),
        catalog_path,
        catalog_dir: PathBuf::new(),
        size_bytes: 0,
        quantization: None,
        architecture: None,
        context_length: None,
        modified: None,
        has_chat_template: false,
        remote,
    }
}

/// The `online` folder shown at the catalog root.
pub fn online_root_node() -> Model {
    dir_node(vec![ONLINE.into()], None)
}

fn repo_node(repo: &HubModel) -> Model {
    dir_node(vec![ONLINE.into(), HUGGINGFACE.into(), repo.id.clone()], Some(RemoteKind::Repo))
}

fn file_node(repo_id: &str, file: &HubFile) -> Model {
    let mut node = dir_node(
        vec![ONLINE.into(), HUGGINGFACE.into(), repo_id.into(), file.name.clone()],
        Some(RemoteKind::File),
    );
    node.size_bytes = file.size_bytes;
    node.quantization = file.quant.clone();
    // Carry the shard names so the row can show a "(N parts)" hint; the node
    // still has no local path, so it stays non-launchable.
    node.shard_paths = file.parts.iter().map(|p| PathBuf::from(&p.rfilename)).collect();
    node
}

impl App {
    /// Children of a prefix inside the online subtree; `None` for local paths.
    pub(super) fn hub_children(&self, prefix: &[String]) -> Option<Vec<Model>> {
        if !in_hub_tree(prefix) {
            return None;
        }
        Some(match prefix.len() {
            1 => vec![dir_node(vec![ONLINE.into(), HUGGINGFACE.into()], None)],
            2 => self.hub.results.iter().map(repo_node).collect(),
            3 => {
                let repo_id = &prefix[2];
                self.hub
                    .listings
                    .get(repo_id)
                    .map(|listing| listing.files.iter().map(|f| file_node(repo_id, f)).collect())
                    .unwrap_or_default()
            }
            _ => Vec::new(),
        })
    }

    /// Kick off whatever fetch the current hub location still needs. Safe to
    /// call repeatedly — in-flight and completed work is skipped.
    pub(super) fn hub_ensure_loaded(&mut self) {
        let prefix = self.catalog_prefix.clone();
        if !in_hub_tree(&prefix) {
            return;
        }
        if let Some(repo_id) = repo_of(&prefix) {
            self.hub_fetch_listing(repo_id.to_string());
        } else if !self.hub.fetched {
            self.hub_search(String::new());
        }
    }

    /// Fetch whatever the hovered hub node needs so the preview column fills
    /// itself: the trending list for the hub folders, a repo's file listing
    /// when a repository is selected. Duplicate fetches are skipped.
    pub(super) fn hub_prefetch_on_preview(&mut self) {
        let Some(node) = self.models.selected() else { return };
        if node.remote == Some(RemoteKind::Repo) {
            if let Some(repo_id) = node.catalog_path.get(2).cloned() {
                self.hub_fetch_listing(repo_id);
            }
            return;
        }
        if node.is_catalog_dir() && in_hub_tree(&node.catalog_path) && !self.hub.fetched {
            self.hub_search(String::new());
        }
    }

    /// Title of the model column: repository file listings keep the classic
    /// "Files" tab name; everything else stays "Model".
    pub fn model_pane_title(&self) -> String {
        match repo_of(&self.catalog_prefix) {
            Some(repo_id) => format!("Files — {repo_id}"),
            None => "Model".into(),
        }
    }

    /// `F5` inside the hub subtree: refetch whatever the current folder shows.
    pub(super) fn hub_refresh_current(&mut self) {
        let prefix = self.catalog_prefix.clone();
        if let Some(repo_id) = repo_of(&prefix).map(str::to_string) {
            self.hub.listings.remove(&repo_id);
            self.hub_fetch_listing(repo_id);
        } else {
            self.hub_search(self.hub.query.clone());
        }
    }

    /// Whether the browser is currently inside the online subtree (drives the
    /// footer hotkey hints).
    pub fn browsing_hub(&self) -> bool {
        in_hub_tree(&self.catalog_prefix)
    }

    /// Header-line indicator for background hub work: an error, an in-flight
    /// fetch, or the number of running downloads. `(text, is_error)`.
    pub fn hub_activity(&self) -> Option<(String, bool)> {
        if let Some(error) = &self.hub.error {
            return Some((error.clone(), true));
        }
        if let Some(loading) = &self.hub.loading {
            return Some((loading.clone(), false));
        }
        let active =
            self.hub.downloads.iter().filter(|d| matches!(d.state, DownloadState::Running)).count();
        match active {
            0 => None,
            1 => {
                let d = self
                    .hub
                    .downloads
                    .iter()
                    .find(|d| matches!(d.state, DownloadState::Running))?;
                Some((format!("⇣ {} {}", d.file, d.progress_label()), false))
            }
            n => Some((format!("⇣ {n} downloads running"), false)),
        }
    }

    /// Run an online repository search; an empty query lists trending models.
    /// Results arrive as [`HubEvent::SearchResults`].
    pub(super) fn hub_search(&mut self, query: String) {
        self.hub.fetched = true;
        self.hub.epoch += 1;
        self.hub.error = None;
        self.hub.loading = Some(if query.trim().is_empty() {
            "Loading trending models…".into()
        } else {
            format!("Searching \u{201c}{}\u{201d}…", query.trim())
        });
        hub::spawn_search(self.hub_tx.clone(), self.hub.epoch, query);
    }

    fn hub_fetch_listing(&mut self, repo_id: String) {
        if self.hub.listings.contains_key(&repo_id) || self.hub.pending_repos.contains(&repo_id) {
            return;
        }
        self.hub.pending_repos.insert(repo_id.clone());
        self.hub.error = None;
        self.hub.loading = Some(format!("Loading files for {repo_id}…"));
        hub::spawn_repo_files(self.hub_tx.clone(), repo_id);
    }

    /// Enter on a remote file: jump to it in the local catalog when it is
    /// already on disk, otherwise start (or retry) its download.
    pub(super) fn hub_download_or_open(&mut self) {
        let Some((repo_id, file)) = self.hub_selected_artifact() else {
            return;
        };
        if self.hub_file_downloaded(&repo_id, &file) {
            self.hub_jump_to_downloaded(&repo_id, &file);
            return;
        }
        if self
            .hub
            .download_for(&repo_id, &file.name)
            .is_some_and(|d| matches!(d.state, DownloadState::Running))
        {
            return; // already in flight
        }

        // Replace any finished/failed record for this artifact with a fresh
        // attempt; an interrupted download resumes from its .part files.
        self.hub.downloads.retain(|d| !(d.repo == repo_id && d.file == file.name));
        let received = Arc::new(AtomicU64::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        let id = self.hub.next_download_id;
        self.hub.next_download_id += 1;
        self.hub.downloads.push(Download {
            id,
            repo: repo_id.clone(),
            file: file.name.clone(),
            total: file.size_bytes,
            received: received.clone(),
            cancel: cancel.clone(),
            state: DownloadState::Running,
        });
        hub::spawn_download(
            self.hub_tx.clone(),
            id,
            repo_id.clone(),
            file.parts.clone(),
            hub::repo_dir(&self.download_dir, &repo_id),
            received,
            cancel,
        );
    }

    /// Request cancellation of the selected remote file's download (`x`).
    /// The worker acknowledges between chunks; the .part file is kept so a
    /// later attempt resumes.
    pub(super) fn hub_cancel_selected(&mut self) {
        let Some((repo_id, file)) = self.hub_selected_artifact() else {
            return;
        };
        if let Some(download) = self
            .hub
            .download_for(&repo_id, &file.name)
            .filter(|d| matches!(d.state, DownloadState::Running))
        {
            download.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// The hub artifact behind the selected model-pane node, if any.
    fn hub_selected_artifact(&self) -> Option<(String, HubFile)> {
        let node = self.models.selected().filter(|m| m.is_remote_file())?;
        let repo_id = node.catalog_path.get(2)?.clone();
        let file = self
            .hub
            .listings
            .get(&repo_id)?
            .files
            .iter()
            .find(|f| Some(&f.name) == node.catalog_path.last())?
            .clone();
        Some((repo_id, file))
    }

    /// Every artifact part exists on disk with its expected size.
    pub fn hub_file_downloaded(&self, repo_id: &str, file: &HubFile) -> bool {
        let dir = hub::repo_dir(&self.download_dir, repo_id);
        file.parts.iter().all(|p| {
            std::fs::metadata(dir.join(&p.rfilename))
                .is_ok_and(|m| p.size == 0 || m.len() == p.size)
        })
    }

    /// A short state marker for a remote file row / footer: on disk, in
    /// flight, or how the last attempt ended. `None` when never touched.
    pub fn hub_file_marker(&self, node: &Model) -> Option<String> {
        let repo_id = node.catalog_path.get(2)?;
        let name = node.catalog_path.last()?;
        if let Some(download) = self.hub.download_for(repo_id, name) {
            return Some(match &download.state {
                DownloadState::Running if download.cancelling() => "cancelling…".into(),
                DownloadState::Running => format!("⇣ {}", download.progress_label()),
                DownloadState::Done => "✓ downloaded".into(),
                DownloadState::Cancelled => "cancelled — Enter resumes".into(),
                DownloadState::Failed(e) => format!("✗ {e}"),
            });
        }
        let file = self.hub.listings.get(repo_id)?.files.iter().find(|f| &f.name == name)?;
        self.hub_file_downloaded(repo_id, file).then(|| "✓ downloaded".into())
    }

    /// The hub search-result entry a repo node refers to.
    pub fn hub_repo_meta(&self, node: &Model) -> Option<&HubModel> {
        let repo_id = node.catalog_path.get(2)?;
        self.hub.results.iter().find(|m| &m.id == repo_id)
    }

    /// Leave the hub subtree and select the downloaded model in the browser.
    fn hub_jump_to_downloaded(&mut self, repo_id: &str, file: &HubFile) {
        let first = hub::repo_dir(&self.download_dir, repo_id).join(&file.parts[0].rfilename);
        let find = |models: &[Model]| {
            models.iter().find(|m| m.shard_paths.contains(&first)).map(|m| m.id.clone())
        };
        // The scanner may not have seen a just-finished download yet.
        let id = match find(&self.scanned_models) {
            Some(id) => Some(id),
            None => {
                self.rescan_models_quiet();
                find(&self.scanned_models)
            }
        };
        if let Some(id) = id {
            self.jump_to_model(&id);
        }
    }

    /// Apply finished background work. Called every loop turn so results and
    /// download completions appear without waiting for the 1s tick.
    pub(super) fn drain_hub_events(&mut self) {
        while let Ok(event) = self.hub_events.try_recv() {
            match event {
                HubEvent::SearchResults { epoch, result } => {
                    if epoch != self.hub.epoch {
                        continue; // superseded by a newer search
                    }
                    self.hub.loading = None;
                    let results = match result {
                        Ok(models) => models,
                        Err(e) => {
                            self.hub.error = Some(e);
                            continue;
                        }
                    };
                    // An open `/` hub search consumes results live; otherwise
                    // they are the folder listing itself.
                    if let Some(search) =
                        self.model_search.as_mut().filter(|s| s.mode == SearchMode::HubRepos)
                    {
                        search.online_results = results;
                        search.cursor =
                            search.cursor.min(search.online_results.len().saturating_sub(1));
                    } else {
                        self.hub.results = results;
                        self.hub_refresh_pane();
                    }
                }
                HubEvent::RepoFiles { repo, result } => {
                    self.hub.pending_repos.remove(&repo);
                    self.hub.loading = None;
                    match result {
                        Ok(listing) => {
                            self.hub.listings.insert(repo, listing);
                            self.hub_refresh_pane();
                        }
                        Err(e) => self.hub.error = Some(e),
                    }
                }
                HubEvent::DownloadDone { id, result } => {
                    if let Some(dl) = self.hub.downloads.iter_mut().find(|d| d.id == id) {
                        dl.state = match result {
                            Ok(download::Outcome::Done) => DownloadState::Done,
                            Ok(download::Outcome::Cancelled) => DownloadState::Cancelled,
                            Err(e) => DownloadState::Failed(e),
                        };
                        // Surface the new model in the local catalog without
                        // yanking the user out of wherever they are browsing.
                        if matches!(dl.state, DownloadState::Done) {
                            self.rescan_models_quiet();
                        }
                    }
                }
            }
        }
    }

    /// Re-derive the model pane after hub data arrived, keeping the cursor on
    /// the same row where possible.
    fn hub_refresh_pane(&mut self) {
        if !in_hub_tree(&self.catalog_prefix) {
            // Not inside the hub — at most the preview of a hovered hub dir
            // needs recomputing.
            self.rebuild_below(self.focus);
            return;
        }
        let selected = self.models.state.selected();
        let items = self.catalog_children(&self.catalog_prefix);
        let cursor = match items.is_empty() {
            true => None,
            false => Some(selected.unwrap_or(0).min(items.len() - 1)),
        };
        self.models.items = items;
        self.models.state.select(cursor);
        self.rebuild_below(Pane::Model);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hub_prefixes_are_recognized() {
        let deep = ["online".to_string(), "huggingface".into(), "org/repo".into()];
        assert!(in_hub_tree(&["online".into()]));
        assert!(in_hub_tree(&deep));
        assert!(!in_hub_tree(&["models".into(), "online".into()]));
        assert_eq!(repo_of(&deep), Some("org/repo"));
        assert_eq!(repo_of(&deep[..2]), None);
    }

    #[test]
    fn remote_nodes_have_expected_shape() {
        let repo =
            repo_node(&HubModel { id: "org/repo".into(), likes: 1, downloads: 2, created: None });
        assert!(repo.is_catalog_dir());
        assert!(!repo.is_model());
        assert_eq!(repo.catalog_path, ["online", "huggingface", "org/repo"]);

        let file = file_node(
            "org/repo",
            &HubFile {
                name: "m-Q4_K_M.gguf".into(),
                quant: Some("Q4_K_M".into()),
                size_bytes: 9,
                parts: Vec::new(),
            },
        );
        assert!(file.is_remote_file());
        assert!(!file.is_catalog_dir());
        assert!(!file.is_model());
        assert_eq!(file.display_label(), "m-Q4_K_M.gguf");
        assert_eq!(file.size_bytes, 9);
        assert_eq!(file.quantization.as_deref(), Some("Q4_K_M"));
    }
}
