//! Session lifecycle: launching servers, persisting and rediscovering them,
//! and tracking their live status/resource usage for the Session Manager.
//!
//! Process spawning/signalling is delegated to a [`SessionSupervisor`]
//! (ADR-005); this module owns the policy: port-conflict resolution, status
//! derivation (`Starting`/`Running`/`Stopped`/`Crashed`), and `/proc` sampling.

pub mod command;
pub mod health;
pub mod proc;
pub mod record;
pub mod supervisor;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};

use crate::domain::OptionItem;
use command::Command;
use health::Health;
use proc::CpuSample;
use record::{DownloadRecord, SessionRecord};
use supervisor::{DetachedSupervisor, LaunchSpec, SessionSupervisor};

/// Observable lifecycle state of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    /// The runtime is downloading model artifacts before it can load them.
    Downloading,
    /// Process is up but `/health` isn't ready yet (model still loading).
    Starting,
    /// `/health` returned 200 (or it was previously Running and is still alive).
    Running,
    /// We asked it to stop and the process is gone.
    Stopped,
    /// The process exited without us asking it to.
    Crashed,
    /// Alive but state can't be determined. Part of the documented state set
    /// (requirements §Session State Detection); reserved for richer health
    /// classification in Phase 4.
    #[allow(dead_code)]
    Unknown,
}

impl SessionStatus {
    /// Status glyph (matches the requirements' indicators).
    pub fn glyph(self) -> &'static str {
        match self {
            SessionStatus::Downloading => "⇩",
            SessionStatus::Running => "●",
            SessionStatus::Starting => "◐",
            SessionStatus::Crashed => "✖",
            SessionStatus::Stopped => "■",
            SessionStatus::Unknown => "?",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SessionStatus::Downloading => "Downloading",
            SessionStatus::Running => "Running",
            SessionStatus::Starting => "Starting",
            SessionStatus::Crashed => "Crashed",
            SessionStatus::Stopped => "Stopped",
            SessionStatus::Unknown => "Unknown",
        }
    }

    /// Terminal states are not re-evaluated once reached.
    fn is_terminal(self) -> bool {
        matches!(self, SessionStatus::Stopped | SessionStatus::Crashed)
    }
}

/// One tracked session: its persisted record plus live, in-memory status.
pub struct Session {
    pub record: SessionRecord,
    pub status: SessionStatus,
    pub cpu_percent: Option<f64>,
    pub rss_bytes: Option<u64>,
    pub download_percent: Option<u8>,
    /// True once the user requested a stop/kill — distinguishes Stopped vs Crashed.
    requested_stop: bool,
    /// Previous CPU sample for delta-based percentage.
    last_cpu: Option<CpuSample>,
}

impl Session {
    fn new(record: SessionRecord, status: SessionStatus) -> Self {
        let download_percent = download_percent(&record);
        Self {
            record,
            status,
            cpu_percent: None,
            rss_bytes: None,
            download_percent,
            requested_stop: false,
            last_cpu: None,
        }
    }

    /// Seconds the process has been alive (None for terminal states).
    pub fn uptime_secs(&self) -> Option<u64> {
        if self.status.is_terminal() {
            return None;
        }
        now_unix().checked_sub(self.record.started_unix)
    }

    pub fn status_label(&self) -> String {
        session_status_label(self.status, self.download_percent)
    }
}

fn session_status_label(status: SessionStatus, download_percent: Option<u8>) -> String {
    match download_percent {
        Some(percent) if status == SessionStatus::Downloading => {
            format!("Downloading ({percent}%)")
        }
        _ => status.label().into(),
    }
}

fn download_percent(record: &SessionRecord) -> Option<u8> {
    if let Some(download) = record.download.as_ref() {
        return download_record_percent(download);
    }
    if record.runtime != "FastFlowLM" {
        return None;
    }
    match fastflow_download_state(&record.log_file) {
        FastFlowDownloadState::Downloading(percent) => Some(percent),
        FastFlowDownloadState::NotStarted if record.fastflow_download => Some(0),
        FastFlowDownloadState::NotStarted | FastFlowDownloadState::Complete => None,
    }
}

fn download_record_percent(download: &DownloadRecord) -> Option<u8> {
    let expected: u128 = download.blobs.iter().map(|blob| blob.expected_bytes as u128).sum();
    if expected == 0 || download.blobs.is_empty() {
        return None;
    }
    let mut downloaded = 0_u128;
    let mut complete = true;
    for blob in &download.blobs {
        if blob.complete_file.is_file() {
            downloaded += blob.expected_bytes as u128;
        } else {
            complete = false;
            downloaded += std::fs::metadata(&blob.incomplete_file)
                .map(|metadata| metadata.len().min(blob.expected_bytes) as u128)
                .unwrap_or(0);
        }
    }
    if complete { None } else { Some(((downloaded.saturating_mul(100) / expected).min(99)) as u8) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FastFlowDownloadState {
    NotStarted,
    Downloading(u8),
    Complete,
}

/// Derive aggregate first-use download progress from FLM's server log. FLM
/// prints per-file progress, so completed file sizes must be added to the
/// current file's downloaded bytes to avoid resetting the displayed percent.
fn fastflow_download_state(log_file: &Path) -> FastFlowDownloadState {
    let Ok(log) = std::fs::read(log_file) else {
        return FastFlowDownloadState::NotStarted;
    };
    fastflow_download_state_from_text(&String::from_utf8_lossy(&log))
}

fn fastflow_download_state_from_text(log: &str) -> FastFlowDownloadState {
    if log.contains("Model downloaded successfully!")
        || log.contains("All downloads completed successfully!")
    {
        return FastFlowDownloadState::Complete;
    }

    let mut started = false;
    let mut total_mb = None;
    let mut file_sizes_mb = Vec::new();
    let mut current_file = None;
    let mut current_downloaded_mb = None;
    let mut current_percent = None;

    // Progress updates use carriage returns and may be prefixed with ANSI
    // clearing sequences, so treat both CR and LF as logical separators and
    // search within each fragment instead of assuming clean lines.
    for fragment in log.split(['\n', '\r']) {
        if fragment.contains("Missing files (")
            || fragment.contains("Downloading")
            || fragment.contains("Files to download (")
        {
            started = true;
        }

        if let Some(rest) = fragment.split_once("Files to download (").map(|(_, rest)| rest)
            && let Some(size) = rest.split_once(')').and_then(|(size, _)| parse_size_mb(size))
        {
            total_mb = Some(size);
            file_sizes_mb.clear();
            current_file = None;
            current_downloaded_mb = None;
            current_percent = None;
            continue;
        }

        if total_mb.is_some()
            && current_file.is_none()
            && fragment.contains("- ")
            && let Some(open) = fragment.rfind('(')
            && let Some(close) = fragment[open + 1..].find(')')
            && let Some(size) = parse_size_mb(&fragment[open + 1..open + 1 + close])
        {
            file_sizes_mb.push(size);
        }

        if let Some(rest) = fragment.split_once("Downloading ").map(|(_, rest)| rest)
            && !rest.starts_with(':')
            && let Some((position, _)) = rest.split_once(':')
            && let Some((index, count)) = position.split_once('/')
            && count.trim().parse::<usize>().is_ok()
            && let Ok(index) = index.trim().parse::<usize>()
        {
            current_file = Some(index);
            current_downloaded_mb = None;
            current_percent = None;
            continue;
        }

        if let Some(rest) = fragment.split_once("Downloading:").map(|(_, rest)| rest) {
            current_percent = rest
                .trim_start()
                .split_once('%')
                .and_then(|(percent, _)| percent.trim().parse::<f64>().ok());
            current_downloaded_mb = rest
                .split_once('(')
                .and_then(|(_, sizes)| sizes.split_once('/'))
                .and_then(|(downloaded, _)| parse_size_mb(downloaded.trim()));
        }
    }

    if let (Some(total), Some(index), Some(downloaded)) =
        (total_mb, current_file, current_downloaded_mb)
        && total > 0.0
    {
        let completed: f64 = file_sizes_mb.iter().take(index.saturating_sub(1)).sum();
        let percent = ((completed + downloaded) * 100.0 / total).round().clamp(0.0, 99.0) as u8;
        return FastFlowDownloadState::Downloading(percent);
    }
    if let Some(percent) = current_percent {
        return FastFlowDownloadState::Downloading(percent.round().clamp(0.0, 99.0) as u8);
    }
    if started { FastFlowDownloadState::Downloading(0) } else { FastFlowDownloadState::NotStarted }
}

fn parse_size_mb(text: &str) -> Option<f64> {
    let text = text.trim();
    for (unit, multiplier) in [("GB", 1024.0), ("MB", 1.0), ("KB", 1.0 / 1024.0)] {
        if let Some(number) = text.strip_suffix(unit) {
            return number.trim().parse::<f64>().ok().map(|value| value * multiplier);
        }
    }
    None
}

/// Everything the manager needs to launch a server. Built by the app from the
/// current runtime/model/profile selection and resolved options.
pub struct LaunchRequest {
    pub runtime: String,
    pub binary: String,
    pub command_prefix: Vec<String>,
    pub fastflow: bool,
    pub fastflow_download: bool,
    pub model: String,
    pub model_path: String,
    pub mtp_path: Option<String>,
    pub projector_path: Option<String>,
    pub hf_repo: Option<String>,
    pub draft_hf: Option<String>,
    pub projector_auto: bool,
    pub download: Option<DownloadRecord>,
    pub profile: String,
    pub host: String,
    pub port: u16,
    pub health_path: String,
    pub options: Vec<OptionItem>,
}

/// Owns the supervisor and the set of tracked sessions.
pub struct SessionManager {
    dir: PathBuf,
    log_dir: PathBuf,
    supervisor: Box<dyn SessionSupervisor>,
    pub sessions: Vec<Session>,
}

static SEQ: AtomicU64 = AtomicU64::new(0);

impl SessionManager {
    /// Construct the manager, then rediscover sessions left running by a
    /// previous llmctl run (pruning any that are no longer alive).
    pub fn new(dir: PathBuf, log_dir: PathBuf) -> Self {
        let mut mgr = Self {
            dir,
            log_dir,
            supervisor: Box::new(DetachedSupervisor::new()),
            sessions: Vec::new(),
        };
        mgr.rediscover();
        mgr
    }

    /// Reload persisted records; keep those whose process is still alive and
    /// matches, delete the JSON for the rest (the spec's "stale records removed").
    pub fn rediscover(&mut self) {
        self.sessions.clear();
        for mut record in record::load_all(&self.dir) {
            let alive =
                proc::is_alive(record.pid) && proc::cmdline_matches(record.pid, &record.model_path);
            if !alive
                && let Some(binary) = record.command.first()
                && let Some(pid) = proc::find_server(binary, &record.model_path, record.port)
            {
                record.pid = pid;
                record.save(&self.dir);
            }
            let alive =
                proc::is_alive(record.pid) && proc::cmdline_matches(record.pid, &record.model_path);
            if alive {
                let health = health::probe_path(&record.host, record.port, &record.health_path);
                let status = match health {
                    Health::Ready => SessionStatus::Running,
                    _ if download_percent(&record).is_some() => SessionStatus::Downloading,
                    _ => SessionStatus::Starting,
                };
                if record.fastflow_download
                    && (health == Health::Ready
                        || fastflow_download_state(&record.log_file)
                            == FastFlowDownloadState::Complete)
                {
                    record.fastflow_download = false;
                    record.save(&self.dir);
                }
                self.sessions.push(Session::new(record, status));
            } else {
                record.delete(&self.dir);
            }
        }
    }

    /// Launch a server from `req`, resolving a free port if the preferred one is
    /// taken. Returns the index of the new session.
    pub fn launch(&mut self, req: LaunchRequest) -> Result<usize> {
        if req.fastflow
            && self.sessions.iter().any(|session| {
                session.record.runtime == "FastFlowLM" && !session.status.is_terminal()
            })
        {
            anyhow::bail!("FastFlowLM can keep only one NPU LLM loaded at a time");
        }
        let port = self.resolve_port(req.port, None);

        // Reflect the resolved port in the options we render into the command.
        let mut options = req.options;
        if let Some(opt) = options.iter_mut().find(|o| o.key == "port") {
            opt.value = port.to_string();
        }
        let command = if req.fastflow {
            Command::build_fastflow(
                &req.command_prefix,
                command::FastFlowMode::Serve,
                &req.model_path,
                &options,
            )
        } else {
            match &req.hf_repo {
                Some(repo) => Command::build_huggingface(
                    &req.binary,
                    repo,
                    &req.model_path,
                    req.mtp_path.as_deref(),
                    req.draft_hf.as_deref(),
                    req.projector_path.as_deref(),
                    req.projector_auto,
                    &options,
                ),
                None => Command::build_local(
                    &req.binary,
                    &req.model_path,
                    req.mtp_path.as_deref(),
                    req.projector_path.as_deref(),
                    &options,
                ),
            }
        };

        let id = next_id();
        let log_file = supervisor::log_path(&self.log_dir, &id);
        let spec = LaunchSpec { argv: command.argv.clone(), log_file: log_file.clone() };
        let spawned = self.supervisor.spawn(&spec)?;

        let record = SessionRecord {
            id,
            name: session_name(&req.model, &req.profile),
            runtime: req.runtime,
            model: req.model,
            model_path: req.model_path,
            profile: req.profile,
            pid: spawned.pid,
            host: req.host,
            port,
            health_path: req.health_path,
            command: command.argv,
            log_file,
            download: req.download,
            fastflow_download: req.fastflow_download,
            started_unix: now_unix(),
        };
        record.save(&self.dir);
        let status = if download_percent(&record).is_some() {
            SessionStatus::Downloading
        } else {
            SessionStatus::Starting
        };
        self.sessions.push(Session::new(record, status));
        Ok(self.sessions.len() - 1)
    }

    /// The live pid that actually backs a session, re-acquiring the real server
    /// if a launcher wrapper re-exec'd or daemonized it under a different pid
    /// (and possibly its own session). Persists the record when the pid changes.
    /// Returns `None` if no live matching process exists.
    fn live_pid(&mut self, idx: usize) -> Option<i32> {
        let (pid, binary, model_path, port) = {
            let s = self.sessions.get(idx)?;
            let binary = s.record.command.first().cloned().unwrap_or_default();
            (s.record.pid, binary, s.record.model_path.clone(), s.record.port)
        };
        let exe = std::path::Path::new(&binary)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        // The recorded pid is authoritative only while it is the server binary.
        if proc::is_alive(pid)
            && proc::comm(pid).as_deref() == Some(&exe[..exe.len().min(15)])
            && proc::cmdline_matches(pid, &model_path)
        {
            return Some(pid);
        }
        // Otherwise the recorded pid is a launcher (or gone); find the real one.
        if let Some(real) = proc::find_server(&binary, &model_path, port) {
            if real != pid {
                let s = &mut self.sessions[idx];
                s.record.pid = real;
                s.record.save(&self.dir);
            }
            return Some(real);
        }
        // Last resort: an unclassified but still-matching process is better than
        // signalling nothing.
        (proc::is_alive(pid) && proc::cmdline_matches(pid, &model_path)).then_some(pid)
    }

    /// Refresh live status and resource usage for every session. Cheap enough
    /// to call on the periodic UI tick.
    pub fn refresh(&mut self) {
        for idx in 0..self.sessions.len() {
            if self.sessions[idx].status.is_terminal() {
                self.sessions[idx].cpu_percent = None;
                self.sessions[idx].rss_bytes = None;
                continue;
            }
            let Some(pid) = self.live_pid(idx) else {
                let s = &mut self.sessions[idx];
                s.status =
                    if s.requested_stop { SessionStatus::Stopped } else { SessionStatus::Crashed };
                s.cpu_percent = None;
                s.rss_bytes = None;
                s.download_percent = None;
                s.last_cpu = None;
                s.download_percent = None;
                continue;
            };

            let host = self.sessions[idx].record.host.clone();
            let port = self.sessions[idx].record.port;
            let prev = self.sessions[idx].last_cpu;
            let status = self.sessions[idx].status;
            let progress = download_percent(&self.sessions[idx].record);

            let rss = proc::rss_bytes(pid);
            let sample = proc::cpu_sample(pid);
            let health_path = self.sessions[idx].record.health_path.clone();
            // Health requests only promote a loading process to Running. Once
            // promoted, process liveness is authoritative; probing forever
            // provides no state change and FastFlowLM logs every request.
            let health = if needs_health_probe(status) {
                health::probe_path(&host, port, &health_path)
            } else {
                Health::Ready
            };

            let s = &mut self.sessions[idx];
            s.rss_bytes = rss;
            s.download_percent = if health == Health::Ready { None } else { progress };
            if let Some(now) = sample {
                if let Some(prev) = prev {
                    s.cpu_percent = proc::cpu_percent(prev, now);
                }
                s.last_cpu = Some(now);
            }
            // Ready promotes to Running; otherwise remain in the appropriate
            // loading state until a later probe succeeds.
            s.status = match health {
                Health::Ready => SessionStatus::Running,
                _ if progress.is_some() => SessionStatus::Downloading,
                _ => SessionStatus::Starting,
            };
            if s.record.fastflow_download
                && (health == Health::Ready
                    || fastflow_download_state(&s.record.log_file)
                        == FastFlowDownloadState::Complete)
            {
                s.record.fastflow_download = false;
                s.record.save(&self.dir);
            }
        }
    }

    /// SIGTERM the server (re-acquiring the real pid behind a launcher wrapper).
    pub fn stop(&mut self, idx: usize) -> Result<()> {
        self.sessions.get_mut(idx).ok_or_else(|| anyhow!("no such session"))?.requested_stop = true;
        match self.live_pid(idx) {
            Some(pid) => self.supervisor.stop(pid),
            None => Ok(()), // already gone
        }
    }

    /// SIGKILL the server (re-acquiring the real pid behind a launcher wrapper).
    pub fn kill(&mut self, idx: usize) -> Result<()> {
        self.sessions.get_mut(idx).ok_or_else(|| anyhow!("no such session"))?.requested_stop = true;
        match self.live_pid(idx) {
            Some(pid) => self.supervisor.kill(pid),
            None => Ok(()),
        }
    }

    /// Stop the running process and relaunch with the stored command.
    pub fn restart(&mut self, idx: usize) -> Result<()> {
        let live = self.live_pid(idx);
        let (mut command, preferred) = {
            let s = self.sessions.get(idx).ok_or_else(|| anyhow!("no such session"))?;
            (s.record.command.clone(), s.record.port)
        };
        // Stop the old process; allow reusing its own port by excluding it.
        if let Some(pid) = live {
            let _ = self.supervisor.stop(pid);
        }
        let port = self.resolve_port(preferred, Some(idx));
        set_port_arg(&mut command, port);

        let id = next_id();
        let log_file = supervisor::log_path(&self.log_dir, &id);
        let spec = LaunchSpec { argv: command.clone(), log_file: log_file.clone() };
        let spawned = self.supervisor.spawn(&spec)?;

        let session = &mut self.sessions[idx];
        session.record.delete(&self.dir); // remove the old id's file
        session.record.id = id;
        session.record.pid = spawned.pid;
        session.record.port = port;
        session.record.command = command;
        session.record.log_file = log_file;
        session.record.started_unix = now_unix();
        session.record.save(&self.dir);
        session.download_percent = download_percent(&session.record);
        session.status = if session.download_percent.is_some() {
            SessionStatus::Downloading
        } else {
            SessionStatus::Starting
        };
        session.requested_stop = false;
        session.last_cpu = None;
        session.cpu_percent = None;
        session.rss_bytes = None;
        Ok(())
    }

    /// Drop a terminated session (deletes its JSON). No-op if still alive.
    pub fn remove(&mut self, idx: usize) -> bool {
        let Some(session) = self.sessions.get(idx) else {
            return false;
        };
        if !session.status.is_terminal() {
            return false;
        }
        session.record.delete(&self.dir);
        self.sessions.remove(idx);
        true
    }

    /// Choose a bindable port at or after `preferred`, skipping ports already
    /// used by other live sessions (`except` is excluded, e.g. during restart).
    fn resolve_port(&self, preferred: u16, except: Option<usize>) -> u16 {
        let in_use: Vec<u16> = self
            .sessions
            .iter()
            .enumerate()
            .filter(|(i, s)| Some(*i) != except && !s.status.is_terminal())
            .map(|(_, s)| s.record.port)
            .collect();

        let mut port = preferred.max(1);
        for _ in 0..256 {
            if !in_use.contains(&port) && health::port_is_free(port) {
                return port;
            }
            port = port.saturating_add(1);
        }
        preferred
    }
}

/// Seconds since the Unix epoch.
fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// A unique-ish session id: `<unix-seconds>-<counter>`.
fn next_id() -> String {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{}-{}", now_unix(), seq)
}

/// Derive a session name like `qwen3-32b-q6_k-coding` from model + profile.
fn session_name(model: &str, profile: &str) -> String {
    let model = model.strip_suffix(".gguf").unwrap_or(model);
    format!("{}-{}", slug(model), slug(profile))
}

fn needs_health_probe(status: SessionStatus) -> bool {
    matches!(status, SessionStatus::Starting | SessionStatus::Downloading)
}

/// Lowercase, replacing runs of non-alphanumeric characters with a single dash.
fn slug(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Replace the value following `--port` in an argv (used on restart).
fn set_port_arg(argv: &mut [String], port: u16) {
    if let Some(i) = argv.iter().position(|a| a == "--port") {
        if let Some(v) = argv.get_mut(i + 1) {
            *v = port.to_string();
        }
    }
}

/// Format an uptime in seconds compactly, e.g. `2h 17m`, `3m`, `12s`.
pub fn format_uptime(secs: u64) -> String {
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}h {m}m")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_and_session_name() {
        assert_eq!(slug("Qwen3-32B-Q6_K"), "qwen3-32b-q6_k");
        assert_eq!(slug("Long Context"), "long-context");
        assert_eq!(session_name("Gemma-27B-Q4_K_M.gguf", "Coding"), "gemma-27b-q4_k_m-coding");
    }

    #[test]
    fn uptime_formats_by_magnitude() {
        assert_eq!(format_uptime(45), "45s");
        assert_eq!(format_uptime(125), "2m 5s");
        assert_eq!(format_uptime(8225), "2h 17m");
    }

    #[test]
    fn download_progress_sums_complete_and_partial_shards() {
        use crate::session::record::DownloadBlob;

        let root = std::env::temp_dir().join(format!("llmctl-progress-{}", now_unix()));
        std::fs::create_dir_all(&root).unwrap();
        let first_complete = root.join("first");
        let second_incomplete = root.join("second.incomplete");
        let second_complete = root.join("second");
        std::fs::write(&first_complete, vec![0; 100]).unwrap();
        std::fs::write(&second_incomplete, vec![0; 34]).unwrap();
        let download = DownloadRecord {
            blobs: vec![
                DownloadBlob {
                    incomplete_file: root.join("first.incomplete"),
                    complete_file: first_complete,
                    expected_bytes: 100,
                },
                DownloadBlob {
                    incomplete_file: second_incomplete,
                    complete_file: second_complete.clone(),
                    expected_bytes: 100,
                },
            ],
        };

        assert_eq!(download_record_percent(&download), Some(67));
        std::fs::write(second_complete, vec![0; 100]).unwrap();
        assert_eq!(download_record_percent(&download), None);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn downloading_status_includes_percentage() {
        assert_eq!(session_status_label(SessionStatus::Downloading, Some(67)), "Downloading (67%)");
        assert_eq!(session_status_label(SessionStatus::Starting, None), "Starting");
    }

    #[test]
    fn fastflow_progress_is_aggregate_across_catalogue_files() {
        let log = "[FLM]  Missing files (4):\n\
                   [FLM]  Files to download (663.05 MB):\n\
                     - config.json (0.00 MB)\n\
                     - model.q4nx (652.14 MB)\n\
                     - tokenizer.json (10.89 MB)\n\
                     - tokenizer_config.json (0.01 MB)\n\
                   [FLM]  Downloading 1/4: config.json\n\
                   [FLM]  Overall progress:  1/4 files\n\
                   [FLM]  Downloading 2/4: model.q4nx\r\x1b[K\
                   [FLM]  Downloading: 61.7% (402.2MB / 652.1MB)";

        assert_eq!(fastflow_download_state_from_text(log), FastFlowDownloadState::Downloading(61));

        let tokenizer = format!(
            "{log}\n[FLM]  Overall progress:  2/4 files\n\
             [FLM]  Downloading 3/4: tokenizer.json\r\x1b[K\
             [FLM]  Downloading: 3.9% (0.4MB / 10.9MB)"
        );
        assert_eq!(
            fastflow_download_state_from_text(&tokenizer),
            FastFlowDownloadState::Downloading(98)
        );
    }

    #[test]
    fn fastflow_progress_ends_before_model_loading() {
        assert_eq!(
            fastflow_download_state_from_text(
                "[FLM]  Downloading: 99.8% (651.0MB / 652.1MB)\n\
                 [FLM]  All downloads completed successfully!\n\
                 [FLM]  Model downloaded successfully!\n\
                 [FLM]  Loading model: /home/user/.config/flm/models/example"
            ),
            FastFlowDownloadState::Complete
        );
    }

    #[test]
    fn health_polling_stops_after_session_reaches_running() {
        assert!(needs_health_probe(SessionStatus::Starting));
        assert!(needs_health_probe(SessionStatus::Downloading));
        assert!(!needs_health_probe(SessionStatus::Running));
    }

    fn opt(key: &str, value: &str, cli: &str) -> OptionItem {
        OptionItem {
            key: key.into(),
            value: value.into(),
            default: String::new(),
            range: None,
            cli: cli.into(),
            description: String::new(),
        }
    }

    /// Full pipeline against a real HTTP server that answers `/health` with 200:
    /// launch → Starting/Running → rediscover (new manager) → stop → Stopped →
    /// remove. Ignored by default (spawns processes); run with `--ignored`.
    #[test]
    #[ignore = "spawns real processes; run with --ignored"]
    fn launch_lifecycle_with_fake_server() {
        use std::thread::sleep;
        use std::time::Duration;

        let base = std::env::temp_dir().join(format!("llmctl-life-{}", std::process::id()));
        let sess_dir = base.join("sessions");
        let log_dir = base.join("logs");
        std::fs::create_dir_all(&sess_dir).unwrap();
        std::fs::create_dir_all(&log_dir).unwrap();

        // A standalone executable that ignores llama flags and serves /health.
        let server = base.join("fake-server");
        std::fs::write(
            &server,
            "#!/usr/bin/env python3\n\
             import sys, http.server\n\
             port = 0\n\
             a = sys.argv\n\
             for i, x in enumerate(a):\n\
             \x20   if x == '--port':\n\
             \x20       port = int(a[i + 1])\n\
             class H(http.server.BaseHTTPRequestHandler):\n\
             \x20   def do_GET(self):\n\
             \x20       self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n\
             \x20   def log_message(self, *a):\n\
             \x20       pass\n\
             http.server.HTTPServer(('127.0.0.1', port), H).serve_forever()\n",
        )
        .unwrap();
        std::fs::set_permissions(&server, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .unwrap();

        let req = LaunchRequest {
            runtime: "llama.cpp".into(),
            binary: server.display().to_string(),
            command_prefix: vec![server.display().to_string()],
            fastflow: false,
            fastflow_download: false,
            model: "fake.gguf".into(),
            model_path: "/models/fake.gguf".into(),
            mtp_path: None,
            projector_path: None,
            hf_repo: None,
            draft_hf: None,
            projector_auto: false,
            download: None,
            profile: "Default".into(),
            host: "127.0.0.1".into(),
            port: 18900,
            health_path: "/health".into(),
            options: vec![opt("host", "127.0.0.1", "--host"), opt("port", "18900", "--port")],
        };

        let mut mgr = SessionManager::new(sess_dir.clone(), log_dir.clone());
        let idx = mgr.launch(req).expect("launch");
        let pid = mgr.sessions[idx].record.pid;
        let port = mgr.sessions[idx].record.port;

        // Wait until /health reports Running.
        let mut running = false;
        for _ in 0..50 {
            mgr.refresh();
            if mgr.sessions[idx].status == SessionStatus::Running {
                running = true;
                break;
            }
            sleep(Duration::from_millis(100));
        }
        assert!(running, "session should reach Running via /health");
        assert!(mgr.sessions[idx].record.file_in(&sess_dir).exists(), "json persisted");

        // A fresh manager rediscovers the live session.
        let rediscovered = SessionManager::new(sess_dir.clone(), log_dir.clone());
        assert_eq!(rediscovered.sessions.len(), 1, "rediscovered the running session");
        assert_eq!(rediscovered.sessions[0].record.port, port);

        // Stop it; it should become Stopped (we requested it).
        mgr.stop(idx).expect("stop");
        let mut stopped = false;
        for _ in 0..50 {
            mgr.refresh();
            if mgr.sessions[idx].status == SessionStatus::Stopped {
                stopped = true;
                break;
            }
            sleep(Duration::from_millis(100));
        }
        assert!(stopped, "session should be Stopped after SIGTERM");
        // `Stopped` can latch a moment before the process fully exits (its
        // /proc cmdline empties during teardown), so poll for it to disappear.
        let mut gone = false;
        for _ in 0..50 {
            if !proc::is_alive(pid) {
                gone = true;
                break;
            }
            sleep(Duration::from_millis(100));
        }
        assert!(gone, "process gone after SIGTERM");

        // Remove the terminated record.
        assert!(mgr.remove(idx), "terminated session removable");
        assert!(mgr.sessions.is_empty());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_port_skips_a_bound_port() {
        let dir = std::env::temp_dir().join(format!("llmctl-mgr-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mgr = SessionManager::new(dir.clone(), dir);
        // Bind an ephemeral port so it is guaranteed in use, then confirm the
        // resolver moves past it to a free, higher port.
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let busy = listener.local_addr().unwrap().port();
        let got = mgr.resolve_port(busy, None);
        assert_ne!(got, busy);
        assert!(got > busy);
    }

    #[test]
    fn set_port_arg_rewrites_value() {
        let mut argv = vec![
            "llama-server".into(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            "8000".into(),
        ];
        set_port_arg(&mut argv, 8042);
        assert_eq!(argv.last().unwrap(), "8042");
    }
}
