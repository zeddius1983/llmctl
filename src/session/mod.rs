//! Session lifecycle: launching servers, persisting and rediscovering them,
//! and tracking their live status/resource usage for the Session Manager.
//!
//! Process spawning/signalling is delegated to a [`SessionSupervisor`]
//! (ADR-005); this module owns the policy: port-conflict resolution, status
//! derivation (`Starting`/`Running`/`Stopped`/`Crashed`), and `/proc` sampling.

pub mod command;
pub mod health;
pub mod logtail;
pub mod proc;
pub mod record;
pub mod supervisor;
pub mod throughput;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};

use command::Command;
use health::Health;
use logtail::LogTail;
use proc::CpuSample;
use record::{DownloadRecord, SessionRecord};
use supervisor::{DetachedSupervisor, LaunchSpec, SessionSupervisor};
use throughput::Throughput;

/// Observable lifecycle state of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    /// llama.cpp is downloading one or more GGUF blobs into the Hub cache.
    Downloading,
    /// Process is up but `/health` isn't ready yet (model still loading).
    Starting,
    /// `/health` returned 200 (or it was previously Running and is still alive).
    Running,
    /// Stopped and waiting for the old process to exit before respawning.
    Restarting,
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
            SessionStatus::Restarting => "↻",
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
            SessionStatus::Restarting => "Restarting",
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

/// A restart that has signalled the old process and is waiting for it to exit
/// before the replacement is spawned.
///
/// Respawning immediately is a race: the old server still holds its port, its
/// GPU memory, or — on the AMD NPU, which grants exactly one hardware context —
/// the device itself, and the replacement loses. Waiting is what makes a restart
/// land on the same port with the same resources.
struct PendingRestart {
    argv: Vec<String>,
    preferred_port: u16,
    /// The process being waited on: the *real* server, already re-acquired
    /// through any launcher wrapper. `None` if it was already gone.
    old_pid: Option<i32>,
    /// When politeness runs out and the old process gets SIGKILL.
    kill_at: std::time::Instant,
    /// Whether that escalation has happened, so it happens only once.
    escalated: bool,
}

/// How long a restart waits for a graceful exit before escalating to SIGKILL.
const RESTART_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

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
    /// Prompt-processing and token-generation rates, and the position in the
    /// server's log they are read from.
    pub throughput: Throughput,
    log_tail: LogTail,
    /// Set while a restart is waiting out the old process; see [`PendingRestart`].
    restart_pending: Option<PendingRestart>,
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
            throughput: Throughput::default(),
            log_tail: LogTail::default(),
            restart_pending: None,
        }
    }

    /// A session with pre-seeded rates, for rendering tests.
    #[cfg(test)]
    pub fn probe(record: SessionRecord, status: SessionStatus, throughput: Throughput) -> Self {
        Self { throughput, ..Self::new(record, status) }
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
    download_record_percent(record.download.as_ref()?)
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

/// Everything the manager needs to launch a server. Built by the app from the
/// current runtime/model/profile selection and resolved options.
///
/// The argv is assembled by the runtime backend before it gets here — the
/// manager stays out of the business of knowing anyone's flags; it only patches
/// the `--port` value if the preferred port turns out to be taken.
pub struct LaunchRequest {
    pub runtime: String,
    pub model: String,
    /// The token that identifies this model in the server's own command line: a
    /// GGUF path for llama.cpp, a `name:size` tag for FastFlowLM. Used to
    /// re-acquire the process from `/proc`, so it must appear in the argv.
    pub model_path: String,
    /// The launch command, already built by the backend.
    pub command: Command,
    /// HTTP path whose `200` means "ready" for this runtime.
    pub health_path: String,
    pub download: Option<DownloadRecord>,
    pub profile: String,
    /// Total size of the model's files, for the Session Manager.
    pub size_bytes: Option<u64>,
    /// The compute backend the server will use, as the runtime resolved it.
    pub device: Option<String>,
    pub host: String,
    pub port: u16,
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
        for record in record::load_all(&self.dir) {
            let alive =
                proc::is_alive(record.pid) && proc::cmdline_matches(record.pid, &record.model_path);
            if alive {
                let status = match health::probe(&record.host, record.port, &record.health_path) {
                    Health::Ready => SessionStatus::Running,
                    _ if download_percent(&record).is_some() => SessionStatus::Downloading,
                    _ => SessionStatus::Starting,
                };
                self.sessions.push(Session::new(record, status));
            } else {
                record.delete(&self.dir);
            }
        }
    }

    /// Launch the already-built command in `req`, moving to a free port if the
    /// preferred one is taken. Returns the index of the new session.
    pub fn launch(&mut self, req: LaunchRequest) -> Result<usize> {
        let port = self.resolve_port(req.port, None);

        // Every backend emits an explicit `--port`, so the resolved port can be
        // patched into the finished argv rather than rebuilding it.
        let mut command = req.command;
        set_port_arg(&mut command.argv, port);

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
            size_bytes: req.size_bytes,
            device: req.device,
            profile: req.profile,
            pid: spawned.pid,
            host: req.host,
            port,
            command: command.argv,
            health_path: req.health_path,
            log_file,
            download: req.download,
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

    /// The first session of `runtime` that still holds its resources — anything
    /// not yet Stopped or Crashed, including one still downloading, since that
    /// process is already up and will claim the device the moment it serves.
    ///
    /// Used to enforce [`crate::runtime::RuntimeBackend::single_session`].
    pub fn active_for_runtime(&self, runtime: &str) -> Option<&Session> {
        self.sessions.iter().find(|s| !s.status.is_terminal() && s.record.runtime == runtime)
    }

    /// A live session of `runtime` currently serving `model`, if any. Used to
    /// refuse deleting the files a running server has open.
    pub fn active_for_model(&self, runtime: &str, model: &str) -> Option<&Session> {
        self.sessions.iter().find(|s| {
            !s.status.is_terminal() && s.record.runtime == runtime && s.record.model == model
        })
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
            // A restart in flight owns this session until its replacement is up:
            // the old pid is on its way out and the new one does not exist yet,
            // so sampling it would only misread the gap as a crash.
            if self.sessions[idx].restart_pending.is_some() {
                continue;
            }
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
            let was_running = self.sessions[idx].status == SessionStatus::Running;
            let progress = download_percent(&self.sessions[idx].record);
            let health_path = self.sessions[idx].record.health_path.clone();

            let rss = proc::rss_bytes(pid);
            let sample = proc::cpu_sample(pid);
            // The server already timed its own requests; read what it wrote
            // rather than timing anything here (see `session::throughput`).
            let parse = crate::runtime::throughput_parser(&self.sessions[idx].record.runtime);
            let log_file = self.sessions[idx].record.log_file.clone();
            let appended = self.sessions[idx].log_tail.poll(&log_file);
            // Once a session is Running its status no longer depends on the
            // probe — the `was_running` arm below keeps it Running for as long
            // as the process lives — so re-probing every tick buys nothing. It
            // does cost something: servers that log each connection (FastFlowLM
            // logs four lines per request) fill their own log with llmctl's
            // health checks. Probe only while readiness is still in question.
            let health =
                if was_running { Health::Ready } else { health::probe(&host, port, &health_path) };

            let s = &mut self.sessions[idx];
            s.rss_bytes = rss;
            s.download_percent = progress;
            for measurement in appended.iter().flat_map(|line| parse(line)) {
                s.throughput.record(measurement);
            }
            if let Some(now) = sample {
                if let Some(prev) = prev {
                    s.cpu_percent = proc::cpu_percent(prev, now);
                }
                s.last_cpu = Some(now);
            }
            // Ready promotes to Running; otherwise keep Running if we were already
            // there (tolerate transient probe failures), else Starting.
            s.status = match health {
                Health::Ready => SessionStatus::Running,
                _ if was_running => SessionStatus::Running,
                _ if progress.is_some() => SessionStatus::Downloading,
                _ => SessionStatus::Starting,
            };
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

    /// Ask the running process to stop, and arrange for the replacement to be
    /// spawned once it is actually gone.
    ///
    /// The respawn is deferred rather than immediate because SIGTERM only
    /// *starts* a shutdown: until the old server exits it still holds its port
    /// and its device, and a replacement launched in that window either lands on
    /// a different port or, on a single-session runtime, fails outright.
    /// [`SessionManager::poll_restarts`] finishes the job on the caller's tick,
    /// so nothing blocks the UI while the old process winds down.
    pub fn restart(&mut self, idx: usize) -> Result<()> {
        if self.sessions.get(idx).is_some_and(|s| s.restart_pending.is_some()) {
            return Err(anyhow!("already restarting"));
        }
        let live = self.live_pid(idx);
        let (argv, preferred_port) = {
            let s = self.sessions.get(idx).ok_or_else(|| anyhow!("no such session"))?;
            (s.record.command.clone(), s.record.port)
        };
        if let Some(pid) = live {
            let _ = self.supervisor.stop(pid);
        }

        let session = &mut self.sessions[idx];
        session.restart_pending = Some(PendingRestart {
            argv,
            preferred_port,
            old_pid: live,
            kill_at: std::time::Instant::now() + RESTART_GRACE,
            escalated: false,
        });
        // Not a user-requested stop: the session is on its way back up, and must
        // not latch as Stopped if the old process disappears before the tick.
        session.requested_stop = false;
        session.status = SessionStatus::Restarting;
        session.cpu_percent = None;
        session.rss_bytes = None;
        session.last_cpu = None;
        Ok(())
    }

    /// Advance every pending restart: spawn the replacement for any whose old
    /// process has exited, and escalate to SIGKILL for any that has outstayed
    /// [`RESTART_GRACE`]. Cheap enough for the poll loop.
    ///
    /// Returns one message per restart whose replacement failed to spawn — the
    /// error the immediate-respawn version used to return from `restart` itself.
    /// A session that stays `Restarting` is one whose old process will not die;
    /// that is reported by showing it, not by inventing an outcome.
    pub fn poll_restarts(&mut self) -> Vec<String> {
        let mut errors = Vec::new();
        for idx in 0..self.sessions.len() {
            let Some((old_pid, kill_at, escalated)) = self.sessions[idx]
                .restart_pending
                .as_ref()
                .map(|p| (p.old_pid, p.kill_at, p.escalated))
            else {
                continue;
            };

            if !self.old_process_gone(idx, old_pid) {
                if !escalated && std::time::Instant::now() >= kill_at {
                    if let Some(pid) = old_pid {
                        let _ = self.supervisor.kill(pid);
                    }
                    if let Some(pending) = self.sessions[idx].restart_pending.as_mut() {
                        pending.escalated = true;
                    }
                }
                continue;
            }

            let Some(pending) = self.sessions[idx].restart_pending.take() else { continue };
            if let Err(error) = self.spawn_replacement(idx, pending) {
                self.sessions[idx].status = SessionStatus::Crashed;
                errors.push(format!("{}: {error}", self.sessions[idx].record.name));
            }
        }
        errors
    }

    /// Whether the process a restart is waiting on has exited. The argv check
    /// guards against a recycled pid looking like the old server.
    fn old_process_gone(&self, idx: usize, old_pid: Option<i32>) -> bool {
        let Some(pid) = old_pid else { return true };
        let Some(session) = self.sessions.get(idx) else { return true };
        !(proc::is_alive(pid) && proc::cmdline_matches(pid, &session.record.model_path))
    }

    /// Launch the replacement for a restart whose old process is gone, reusing
    /// the session slot. The preferred port is now genuinely free, so a restart
    /// normally comes back on the port it left.
    fn spawn_replacement(&mut self, idx: usize, pending: PendingRestart) -> Result<()> {
        let mut command = pending.argv;
        let port = self.resolve_port(pending.preferred_port, Some(idx));
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

    /// The single-session guard must see a session that is still starting or
    /// downloading (the device is spoken for) but ignore one that has ended.
    #[test]
    fn active_for_runtime_matches_only_live_sessions_of_that_runtime() {
        fn record(runtime: &str) -> SessionRecord {
            SessionRecord {
                id: "0-0".into(),
                name: format!("{runtime}-model"),
                runtime: runtime.into(),
                model: "m".into(),
                model_path: "m".into(),
                profile: "Default".into(),
                size_bytes: None,
                device: None,
                pid: 1,
                host: "127.0.0.1".into(),
                port: 1,
                command: vec![],
                health_path: "/health".into(),
                log_file: PathBuf::new(),
                download: None,
                started_unix: 0,
            }
        }

        let dir = std::env::temp_dir().join(format!("llmctl-active-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut mgr = SessionManager::new(dir.clone(), dir);
        mgr.sessions = vec![
            Session::new(record("FastFlowLM"), SessionStatus::Crashed),
            Session::new(record("llama.cpp"), SessionStatus::Running),
        ];
        assert!(mgr.active_for_runtime("FastFlowLM").is_none(), "a crashed session is not live");
        assert!(mgr.active_for_runtime("vLLM").is_none(), "unknown runtime");
        assert!(mgr.active_for_runtime("llama.cpp").is_some());

        // Downloading counts: the process is up and will claim the device.
        mgr.sessions[0].status = SessionStatus::Downloading;
        assert!(mgr.active_for_runtime("FastFlowLM").is_some());
    }

    #[test]
    fn downloading_status_includes_percentage() {
        assert_eq!(session_status_label(SessionStatus::Downloading, Some(67)), "Downloading (67%)");
        assert_eq!(session_status_label(SessionStatus::Starting, None), "Starting");
    }

    fn opt(key: &str, value: &str, cli: &str) -> crate::domain::OptionItem {
        crate::domain::OptionItem {
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

        let command = Command::build_local(
            &server.display().to_string(),
            "/models/fake.gguf",
            None,
            None,
            &crate::runtime::llama_cpp::SCHEMA,
            &[opt("host", "127.0.0.1", "--host"), opt("port", "18900", "--port")],
        );
        let req = LaunchRequest {
            runtime: "llama.cpp".into(),
            model: "fake.gguf".into(),
            model_path: "/models/fake.gguf".into(),
            command,
            health_path: "/health".into(),
            download: None,
            profile: "Default".into(),
            size_bytes: None,
            device: None,
            host: "127.0.0.1".into(),
            port: 18900,
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

    /// A restart must not spawn the replacement while the old process is still
    /// alive: on the AMD NPU the second process loses the race for the hardware
    /// context, and on any runtime it can be pushed off its own port.
    #[test]
    #[ignore = "spawns real processes; run with --ignored --test-threads=1"]
    fn restart_waits_for_the_old_process_before_respawning() {
        use std::thread::sleep;
        use std::time::Duration;

        let base = std::env::temp_dir().join(format!("llmctl-restart-{}", std::process::id()));
        let sess_dir = base.join("sessions");
        let log_dir = base.join("logs");
        std::fs::create_dir_all(&sess_dir).unwrap();
        std::fs::create_dir_all(&log_dir).unwrap();

        // `sleep` ignores SIGTERM's urgency no more than any other process, but
        // it takes a moment to die — which is exactly the window under test.
        let req = LaunchRequest {
            runtime: "FastFlowLM".into(),
            model: "qwen3:0.6b".into(),
            model_path: "300".into(),
            command: Command { argv: vec!["sleep".into(), "300".into()] },
            health_path: "/v1/models".into(),
            download: None,
            profile: "Default".into(),
            size_bytes: None,
            device: None,
            host: "127.0.0.1".into(),
            port: 18930,
        };

        let mut mgr = SessionManager::new(sess_dir.clone(), log_dir.clone());
        let idx = mgr.launch(req).expect("launch");
        sleep(Duration::from_millis(200)); // let exec happen
        let old_pid = mgr.sessions[idx].record.pid;
        let old_port = mgr.sessions[idx].record.port;

        mgr.restart(idx).expect("restart");
        assert_eq!(mgr.sessions[idx].status, SessionStatus::Restarting);
        assert_eq!(mgr.sessions[idx].record.pid, old_pid, "respawned before the old process died");
        // A second R while one is in flight must not stack another respawn.
        assert!(mgr.restart(idx).is_err(), "restart should refuse to overlap itself");
        // Refreshing mid-restart must not read the gap as a crash.
        mgr.refresh();
        assert_eq!(mgr.sessions[idx].status, SessionStatus::Restarting);

        let mut respawned = false;
        for _ in 0..50 {
            mgr.poll_restarts();
            if mgr.sessions[idx].status != SessionStatus::Restarting {
                respawned = true;
                break;
            }
            sleep(Duration::from_millis(100));
        }
        assert!(respawned, "restart never completed");
        assert!(!proc::is_alive(old_pid), "the old process outlived the restart");
        assert_ne!(mgr.sessions[idx].record.pid, old_pid, "a new process should be running");
        // The port was genuinely free by the time we spawned, so it is reused.
        assert_eq!(mgr.sessions[idx].record.port, old_port, "restart moved to a different port");

        let _ = mgr.kill(idx);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// With nothing left alive to wait for, the first poll respawns immediately.
    #[test]
    #[ignore = "spawns real processes; run with --ignored --test-threads=1"]
    fn restarting_a_dead_session_respawns_on_the_first_poll() {
        use std::thread::sleep;
        use std::time::Duration;

        let base = std::env::temp_dir().join(format!("llmctl-restart-dead-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let mut mgr = SessionManager::new(base.join("sessions"), base.join("logs"));
        mgr.sessions.push(Session::new(
            SessionRecord {
                id: "0-0".into(),
                name: "dead".into(),
                runtime: "llama.cpp".into(),
                model: "m".into(),
                model_path: "300".into(),
                profile: "Default".into(),
                size_bytes: None,
                device: None,
                pid: -1,
                host: "127.0.0.1".into(),
                port: 18931,
                command: vec!["sleep".into(), "300".into()],
                health_path: "/health".into(),
                log_file: base.join("dead.log"),
                download: None,
                started_unix: 0,
            },
            SessionStatus::Crashed,
        ));

        mgr.restart(0).expect("restart");
        assert!(mgr.poll_restarts().is_empty(), "no spawn errors expected");
        assert_eq!(mgr.sessions[0].status, SessionStatus::Starting);
        assert!(mgr.sessions[0].record.pid > 0);

        sleep(Duration::from_millis(100));
        let _ = mgr.kill(0);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// A replacement that cannot be spawned at all reports the failure instead
    /// of leaving the session stuck in `Restarting`.
    #[test]
    fn a_replacement_that_cannot_spawn_is_reported() {
        let base = std::env::temp_dir().join(format!("llmctl-restart-bad-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let mut mgr = SessionManager::new(base.join("sessions"), base.join("logs"));
        mgr.sessions.push(Session::new(
            SessionRecord {
                id: "0-0".into(),
                name: "broken".into(),
                runtime: "llama.cpp".into(),
                model: "m".into(),
                model_path: "nothing".into(),
                profile: "Default".into(),
                size_bytes: None,
                device: None,
                pid: -1,
                host: "127.0.0.1".into(),
                port: 18932,
                command: vec!["/nonexistent/llmctl-no-such-binary".into()],
                health_path: "/health".into(),
                log_file: base.join("broken.log"),
                download: None,
                started_unix: 0,
            },
            SessionStatus::Crashed,
        ));

        mgr.restart(0).expect("restart");
        let errors = mgr.poll_restarts();
        assert_eq!(errors.len(), 1, "the spawn failure should be reported");
        assert!(errors[0].starts_with("broken:"), "{errors:?}");
        assert_eq!(mgr.sessions[0].status, SessionStatus::Crashed);
        assert!(mgr.poll_restarts().is_empty(), "a failed restart should not retry forever");
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
