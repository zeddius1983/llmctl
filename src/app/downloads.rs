//! Transfer lifecycle and worker events, independent of navigation/rendering.
use crate::discovery;
use crate::runtime::{ModelTransfer, RuntimeId};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{
    Arc,
    mpsc::{self, Receiver, Sender},
};

#[derive(Debug)]
pub enum ModelDownloadStatus {
    Downloading,
    Cancelling,
    Downloaded(PathBuf),
    Cancelled,
    Interrupted,
    Failed(String),
}

/// Where a tracked download's bytes come from, and how to fetch them.
#[derive(Clone)]
pub(super) enum DownloadSource {
    /// Hugging Face blobs that llmctl fetches into the Hub cache itself.
    Hub(Box<crate::domain::RemoteModel>),
    /// A transfer whose layout and implementation belong to its backend.
    Backend(ModelTransfer),
}

pub struct ModelDownload {
    pub(super) id: u64,
    pub model_id: String,
    pub model: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub status: ModelDownloadStatus,
    pub(super) source: DownloadSource,
    pub(super) cancelled: Arc<AtomicBool>,
}

impl ModelDownload {
    pub fn percent(&self) -> u8 {
        transfer_percent(self.downloaded_bytes, self.total_bytes)
    }

    /// The paths this transfer is writing, so a deletion can tell whether it
    /// would remove files still being fetched — whatever the model is called
    /// in the catalog the user is deleting from.
    pub(super) fn targets(&self) -> Vec<PathBuf> {
        match &self.source {
            DownloadSource::Hub(remote) => remote
                .blobs
                .iter()
                .filter_map(|blob| discovery::online::cache_blob_paths(&remote.repo, &blob.oid))
                .flat_map(|(incomplete, complete)| [incomplete, complete])
                .collect(),
            DownloadSource::Backend(transfer) => transfer.targets.clone(),
        }
    }
}

pub(super) fn transfer_percent(downloaded_bytes: u64, total_bytes: u64) -> u8 {
    if total_bytes == 0 {
        return 0;
    }
    ((downloaded_bytes as u128 * 100 / total_bytes as u128).min(100)) as u8
}

pub(super) fn restore_model_downloads(models_dir: &std::path::Path) -> (Vec<ModelDownload>, u64) {
    let mut downloads = Vec::new();
    let mut next_id = 1_u64;
    for record in discovery::online::load_download_records(models_dir) {
        let total_bytes = record.remote.blobs.iter().map(|blob| blob.size_bytes).sum();
        let downloaded_bytes = discovery::online::cached_downloaded_bytes(&record.remote);
        if total_bytes > 0
            && downloaded_bytes >= total_bytes
            && discovery::online::finalize_cached_download(&record.remote).is_ok()
        {
            if let Err(error) =
                discovery::online::delete_download_record(models_dir, &record.model_id)
            {
                tracing::warn!(%error, model = %record.model_id, "failed to remove completed download record");
            }
            continue;
        }
        let status = if total_bytes == 0 {
            ModelDownloadStatus::Failed("persisted download has no blob size metadata".into())
        } else {
            ModelDownloadStatus::Interrupted
        };
        downloads.push(ModelDownload {
            id: next_id,
            model_id: record.model_id,
            model: record.model,
            downloaded_bytes,
            total_bytes,
            status,
            source: DownloadSource::Hub(Box::new(record.remote)),
            cancelled: Arc::new(AtomicBool::new(false)),
        });
        next_id = next_id.wrapping_add(1).max(1);
    }
    (downloads, next_id)
}

enum ModelDownloadEvent {
    Progress { id: u64, downloaded_bytes: u64, total_bytes: u64 },
    Finished { id: u64, result: std::result::Result<discovery::online::DownloadResult, String> },
}

pub(super) struct DownloadChanges {
    pub online: bool,
    pub runtimes: HashSet<RuntimeId>,
    pub records: Vec<String>,
}

pub struct DownloadManager {
    pub jobs: Vec<ModelDownload>,
    tx: Sender<ModelDownloadEvent>,
    rx: Receiver<ModelDownloadEvent>,
    next_id: u64,
}

impl DownloadManager {
    pub(super) fn load(root: &std::path::Path) -> Self {
        let (jobs, next_id) = restore_model_downloads(root);
        let (tx, rx) = mpsc::channel();
        Self { jobs, next_id, tx, rx }
    }

    pub(super) fn cancel(&mut self, index: usize) {
        if let Some(job) = self.jobs.get_mut(index) {
            if matches!(job.status, ModelDownloadStatus::Downloading) {
                job.cancelled.store(true, Ordering::Relaxed);
                job.status = ModelDownloadStatus::Cancelling;
            }
        }
    }

    pub(super) fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);
        id
    }

    pub(super) fn spawn(&self, id: u64, source: DownloadSource, cancelled: Arc<AtomicBool>) {
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let progress = |downloaded_bytes, total_bytes| {
                let _ = tx.send(ModelDownloadEvent::Progress { id, downloaded_bytes, total_bytes });
            };
            let result = match source {
                DownloadSource::Hub(remote) => {
                    discovery::online::download_model(&remote, &cancelled, progress)
                        .map_err(|error| error.to_string())
                }
                DownloadSource::Backend(transfer) => {
                    (transfer.run)(&transfer.model, &cancelled, &mut { progress })
                        .map_err(|error| error.to_string())
                }
            };
            let _ = tx.send(ModelDownloadEvent::Finished { id, result });
        });
    }

    pub(super) fn poll(&mut self) -> DownloadChanges {
        let mut refresh_models = false;
        let mut refresh_runtimes = HashSet::new();
        let mut completed_records = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            match event {
                ModelDownloadEvent::Progress { id, downloaded_bytes, total_bytes } => {
                    let Some(download) = self.jobs.iter_mut().find(|download| download.id == id)
                    else {
                        continue;
                    };
                    if !matches!(download.status, ModelDownloadStatus::Downloading) {
                        continue;
                    }
                    download.downloaded_bytes = downloaded_bytes.min(total_bytes);
                    download.total_bytes = total_bytes;
                }
                ModelDownloadEvent::Finished { id, result } => {
                    let Some(download) = self.jobs.iter_mut().find(|download| download.id == id)
                    else {
                        continue;
                    };
                    match result {
                        Ok(discovery::online::DownloadResult::Downloaded(path)) => {
                            download.downloaded_bytes = download.total_bytes;
                            download.status = ModelDownloadStatus::Downloaded(path);
                            match &download.source {
                                DownloadSource::Backend(transfer) => {
                                    refresh_runtimes.insert(transfer.runtime.clone());
                                }
                                DownloadSource::Hub(_) => {
                                    completed_records.push(download.model_id.clone());
                                    refresh_models = true;
                                }
                            }
                        }
                        Ok(discovery::online::DownloadResult::Cancelled) => {
                            download.status = ModelDownloadStatus::Cancelled;
                        }
                        Err(_) if download.cancelled.load(Ordering::Relaxed) => {
                            download.status = ModelDownloadStatus::Cancelled;
                        }
                        Err(error) => download.status = ModelDownloadStatus::Failed(error),
                    }
                }
            }
        }
        DownloadChanges {
            online: refresh_models,
            runtimes: refresh_runtimes,
            records: completed_records,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_attempts_and_late_progress_cannot_overwrite_a_cancelled_job() {
        let (tx, rx) = mpsc::channel();
        let remote =
            serde_json::from_value(serde_json::json!({"repo": "org/repo", "file": "model.gguf"}))
                .unwrap();
        let mut manager = DownloadManager {
            jobs: vec![ModelDownload {
                id: 2,
                model_id: "model".into(),
                model: "model".into(),
                downloaded_bytes: 0,
                total_bytes: 100,
                status: ModelDownloadStatus::Downloading,
                source: DownloadSource::Hub(Box::new(remote)),
                cancelled: Arc::new(AtomicBool::new(false)),
            }],
            tx,
            rx,
            next_id: 3,
        };
        manager
            .tx
            .send(ModelDownloadEvent::Progress { id: 1, downloaded_bytes: 99, total_bytes: 100 })
            .unwrap();
        manager.poll();
        assert_eq!(manager.jobs[0].downloaded_bytes, 0);
        manager
            .tx
            .send(ModelDownloadEvent::Progress { id: 2, downloaded_bytes: 50, total_bytes: 100 })
            .unwrap();
        manager.poll();
        manager.cancel(0);
        manager
            .tx
            .send(ModelDownloadEvent::Progress { id: 2, downloaded_bytes: 98, total_bytes: 100 })
            .unwrap();
        manager
            .tx
            .send(ModelDownloadEvent::Finished { id: 2, result: Err("cancelled read".into()) })
            .unwrap();
        let changes = manager.poll();
        assert_eq!(manager.jobs[0].downloaded_bytes, 50);
        assert!(matches!(manager.jobs[0].status, ModelDownloadStatus::Cancelled));
        assert!(changes.runtimes.is_empty() && changes.records.is_empty() && !changes.online);
    }
}
