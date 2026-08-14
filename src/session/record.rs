//! On-disk session metadata, persisted as `session-<id>.json` under the state
//! sessions dir so running servers can be rediscovered after an llmctl restart.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::warn;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadBlob {
    pub incomplete_file: PathBuf,
    pub complete_file: PathBuf,
    pub expected_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRecord {
    pub blobs: Vec<DownloadBlob>,
}

/// Everything we persist about a launched server, enough to rediscover it,
/// show it in the manager, copy its endpoint, and restart it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub id: String,
    pub name: String,
    pub runtime: String,
    pub model: String,
    pub model_path: String,
    pub profile: String,
    /// Total size of the model's files, recorded at launch. Sessions written by
    /// an older llmctl have none, and show nothing rather than a wrong figure.
    #[serde(default)]
    pub size_bytes: Option<u64>,
    /// The compute backend this server runs on (`ROCm`, `Vulkan`, `NPU`, …).
    #[serde(default)]
    pub device: Option<String>,
    pub pid: i32,
    pub host: String,
    pub port: u16,
    pub command: Vec<String>,
    /// HTTP path whose `200` means "ready" for the runtime that owns this
    /// session. Persisted so a rediscovered session probes the right endpoint
    /// without llmctl having to re-resolve its backend.
    #[serde(default = "default_health_path")]
    pub health_path: String,
    pub log_file: PathBuf,
    /// Cache blobs observed while llama.cpp downloads a remote Hub artifact.
    #[serde(default)]
    pub download: Option<DownloadRecord>,
    /// Process start time, seconds since the Unix epoch (for uptime).
    pub started_unix: u64,
}

/// Records written before llmctl knew about multiple runtimes are all
/// llama.cpp sessions.
fn default_health_path() -> String {
    "/health".to_string()
}

impl SessionRecord {
    /// JSON path for this record under `dir`.
    pub fn file_in(&self, dir: &Path) -> PathBuf {
        dir.join(format!("session-{}.json", self.id))
    }

    /// Write (or overwrite) the record's JSON file under `dir`.
    pub fn save(&self, dir: &Path) {
        let path = self.file_in(dir);
        match serde_json::to_vec_pretty(self) {
            Ok(bytes) => {
                if let Err(err) = std::fs::write(&path, bytes) {
                    warn!(path = %path.display(), %err, "failed to write session record");
                }
            }
            Err(err) => warn!(%err, "failed to serialize session record"),
        }
    }

    /// Remove the record's JSON file (session ended / pruned).
    pub fn delete(&self, dir: &Path) {
        let path = self.file_in(dir);
        if let Err(err) = std::fs::remove_file(&path) {
            if err.kind() != std::io::ErrorKind::NotFound {
                warn!(path = %path.display(), %err, "failed to remove session record");
            }
        }
    }

    /// The connectable OpenAI-compatible endpoint URL.
    pub fn endpoint(&self) -> String {
        let host = match self.host.as_str() {
            "0.0.0.0" | "::" | "" => "127.0.0.1",
            other => other,
        };
        format!("http://{host}:{}/v1", self.port)
    }
}

/// Load every `session-*.json` record under `dir`.
pub fn load_all(dir: &Path) -> Vec<SessionRecord> {
    let mut records = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return records;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_session = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("session-") && n.ends_with(".json"));
        if !is_session {
            continue;
        }
        match std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()) {
            Some(record) => records.push(record),
            None => warn!(path = %path.display(), "skipping unreadable session record"),
        }
    }
    records.sort_by_key(|r: &SessionRecord| r.started_unix);
    records
}
