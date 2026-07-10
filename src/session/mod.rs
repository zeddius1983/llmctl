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

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};

use crate::domain::{OptionItem, RuntimeId};
use command::Command;
use health::Health;
use proc::CpuSample;
use record::SessionRecord;
use supervisor::{DetachedSupervisor, LaunchSpec, SessionSupervisor};

/// Observable lifecycle state of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
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
            SessionStatus::Running => "●",
            SessionStatus::Starting => "◐",
            SessionStatus::Crashed => "✖",
            SessionStatus::Stopped => "■",
            SessionStatus::Unknown => "?",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
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
    /// True once the user requested a stop/kill — distinguishes Stopped vs Crashed.
    requested_stop: bool,
    /// Previous CPU sample for delta-based percentage.
    last_cpu: Option<CpuSample>,
}

impl Session {
    fn new(record: SessionRecord, status: SessionStatus) -> Self {
        Self {
            record,
            status,
            cpu_percent: None,
            rss_bytes: None,
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
}

/// Everything the manager needs to launch a server. Built by the app from the
/// current runtime/model/profile selection and resolved options.
pub struct LaunchRequest {
    pub runtime_id: RuntimeId,
    pub runtime: String,
    pub binary: String,
    pub model: String,
    pub model_path: String,
    pub profile: String,
    pub host: String,
    pub port: u16,
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
            let direct =
                proc::is_alive(record.pid) && proc::cmdline_matches(record.pid, &record.model_path);
            let live_pid = direct.then_some(record.pid).or_else(|| {
                let binary = record.command.first()?;
                proc::find_server(binary, &record.model_path, record.port)
            });
            if let Some(pid) = live_pid {
                if pid != record.pid {
                    record.pid = pid;
                    record.save(&self.dir);
                }
                let status = match health::probe(&record.host, record.port) {
                    Health::Ready => SessionStatus::Running,
                    _ => SessionStatus::Starting,
                };
                self.sessions.push(Session::new(record, status));
            } else {
                record.delete(&self.dir);
            }
        }
    }

    /// Launch a server from `req`, resolving a free port if the preferred one is
    /// taken. Returns the index of the new session.
    pub fn launch(&mut self, req: LaunchRequest) -> Result<usize> {
        let port = self.resolve_port(req.port, None);

        // Reflect the resolved port in the options we render into the command.
        let mut options = req.options;
        if let Some(opt) = options.iter_mut().find(|o| o.key == "port") {
            opt.value = port.to_string();
        }
        let command = Command::build_for(req.runtime_id, &req.binary, &req.model_path, &options);

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
            command: command.argv,
            log_file,
            started_unix: now_unix(),
        };
        record.save(&self.dir);
        self.sessions.push(Session::new(record, SessionStatus::Starting));
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
        // Exact model identity is the PID-reuse guard. Do not require `comm` to
        // equal the configured executable: Python and container wrappers often
        // expose a different process name while retaining the launch argv.
        if proc::is_alive(pid) && proc::cmdline_matches(pid, &model_path) {
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
                s.last_cpu = None;
                continue;
            };

            let host = self.sessions[idx].record.host.clone();
            let port = self.sessions[idx].record.port;
            let prev = self.sessions[idx].last_cpu;
            let was_running = self.sessions[idx].status == SessionStatus::Running;

            let rss = proc::rss_bytes(pid);
            let sample = proc::cpu_sample(pid);
            let health = health::probe(&host, port);

            let s = &mut self.sessions[idx];
            s.rss_bytes = rss;
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
        session.status = SessionStatus::Starting;
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
            runtime_id: RuntimeId::LlamaCpp,
            runtime: "llama.cpp".into(),
            binary: server.display().to_string(),
            model: "fake.gguf".into(),
            model_path: "/models/fake.gguf".into(),
            profile: "Default".into(),
            host: "127.0.0.1".into(),
            port: 18900,
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
