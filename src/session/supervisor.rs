//! Process lifecycle behind a trait (ADR-005).
//!
//! The MVP `DetachedSupervisor` spawns `llama-server` in its own session via
//! `setsid()`, with stdio redirected to a per-session log file, so the server
//! survives the TUI exiting and isn't disturbed by terminal signals. Stop/kill
//! signal the whole process group. A daemon or `systemd-run` backend could
//! implement the same trait later.

use std::fs::File;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result};

/// Everything needed to launch one server process.
pub struct LaunchSpec {
    pub argv: Vec<String>,
    pub log_file: PathBuf,
}

/// A freshly spawned process. After `setsid` the pid equals the new session/
/// process-group id, so signals can target the whole group via `-pid`.
pub struct Spawned {
    pub pid: i32,
}

/// Abstract process lifecycle so the UI never spawns directly (ADR-005).
pub trait SessionSupervisor {
    fn spawn(&mut self, spec: &LaunchSpec) -> Result<Spawned>;
    /// Reap only children owned by this supervisor, without waiting.
    fn reap(&mut self) {}
    /// Graceful stop: SIGTERM to the process group.
    fn stop(&self, pid: i32) -> Result<()>;
    /// Forceful stop: SIGKILL to the process group.
    fn kill(&self, pid: i32) -> Result<()>;
}

/// Owns detached children until they exit; never changes signal dispositions.
#[derive(Default)]
pub struct DetachedSupervisor {
    children: Vec<Child>,
}

impl DetachedSupervisor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Drop for DetachedSupervisor {
    fn drop(&mut self) {
        // A manager can be dropped while the application stays alive. Transfer
        // wait ownership without terminating its servers or blocking shutdown.
        // At process exit the OS reparents any servers still running.
        for mut child in std::mem::take(&mut self.children) {
            std::thread::spawn(move || {
                let _ = child.wait();
            });
        }
    }
}

impl SessionSupervisor for DetachedSupervisor {
    fn spawn(&mut self, spec: &LaunchSpec) -> Result<Spawned> {
        let (program, args) = spec.argv.split_first().context("empty command")?;

        if let Some(parent) = spec.log_file.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let log = File::options()
            .create(true)
            .append(true)
            .open(&spec.log_file)
            .with_context(|| format!("opening log {}", spec.log_file.display()))?;
        let log_err = log.try_clone().context("cloning log handle")?;

        let mut cmd = Command::new(program);
        cmd.args(args).stdin(Stdio::null()).stdout(Stdio::from(log)).stderr(Stdio::from(log_err));

        // Detach into a new session (no controlling terminal, own process group)
        // before exec, so the server outlives the TUI and ignores tty signals.
        // SAFETY: `setsid` is async-signal-safe and the only call in the hook.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        self.reap();
        let child = cmd.spawn().with_context(|| format!("spawning {}", program))?;
        let pid = child.id() as i32;
        self.children.push(child);
        Ok(Spawned { pid })
    }

    fn reap(&mut self) {
        self.children.retain_mut(|child| match child.try_wait() {
            Ok(Some(_)) => false,
            Ok(None) => true,
            // ECHILD means this handle can no longer be waited on. Other
            // transient errors are retried on the next tick.
            Err(error) => error.raw_os_error() != Some(libc::ECHILD),
        });
    }

    fn stop(&self, pid: i32) -> Result<()> {
        signal_session(pid, libc::SIGTERM)
    }

    fn kill(&self, pid: i32) -> Result<()> {
        signal_session(pid, libc::SIGKILL)
    }
}

/// Signal an entire launched server with `sig`. We target the process group led
/// by `pid` (the `setsid` leader) *and* every descendant pid individually, so a
/// launcher wrapper that forks the real `llama-server` into a different process
/// group is still stopped. Descendants are collected before any signal is sent,
/// so a parent exiting (and its children reparenting) can't drop them.
fn signal_session(pid: i32, sig: i32) -> Result<()> {
    let mut targets = crate::session::proc::descendants(pid);
    targets.push(pid);

    // SAFETY: `kill` is a thin syscall wrapper with no memory preconditions.
    let mut delivered = unsafe { libc::kill(-pid, sig) } == 0;
    for target in targets {
        if unsafe { libc::kill(target, sig) } == 0 {
            delivered = true;
        }
    }
    if delivered { Ok(()) } else { Err(std::io::Error::last_os_error()).context("sending signal") }
}

/// Best-effort base64 (standard alphabet) for OSC 52 clipboard copy — avoids a
/// dependency just to yank a command string to the system clipboard.
pub fn base64(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | (b[2] as u32);
        out.push(ALPHABET[(n >> 18 & 63) as usize] as char);
        out.push(ALPHABET[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 { ALPHABET[(n >> 6 & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[(n & 63) as usize] as char } else { '=' });
    }
    out
}

/// Resolve a log file path for a session id under `log_dir`.
pub fn log_path(log_dir: &Path, id: &str) -> PathBuf {
    log_dir.join(format!("session-{id}.log"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"hello"), "aGVsbG8=");
    }

    fn test_log(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("llmctl-{name}-{}.log", std::process::id()))
    }

    #[test]
    fn output_capture_and_failed_exec_work_while_a_supervisor_exists() {
        let log_file = test_log("child-capture");
        let mut sup = DetachedSupervisor::new();
        let workers: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..8 {
                        let output =
                            Command::new("sh").args(["-c", "printf captured"]).output().unwrap();
                        assert!(output.status.success());
                        assert_eq!(output.stdout, b"captured");
                    }
                })
            })
            .collect();
        for _ in 0..8 {
            assert!(
                sup.spawn(&LaunchSpec {
                    argv: vec!["/nonexistent/llmctl-missing-server".into()],
                    log_file: log_file.clone(),
                })
                .is_err()
            );
            sup.reap();
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert!(sup.children.is_empty());
        std::fs::remove_file(log_file).unwrap();
    }

    #[test]
    fn dropping_a_supervisor_leaves_its_server_alive_and_eventually_reaps_it() {
        let log_file = test_log("child-drop");
        let mut sup = DetachedSupervisor::new();
        let spawned = sup
            .spawn(&LaunchSpec {
                argv: vec!["sleep".into(), "30".into()],
                log_file: log_file.clone(),
            })
            .unwrap();
        drop(sup);
        assert!(crate::session::proc::is_alive(spawned.pid));
        signal_session(spawned.pid, libc::SIGKILL).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let proc_path = PathBuf::from(format!("/proc/{}", spawned.pid));
        while proc_path.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!proc_path.exists(), "background owner must reap the exited server");
        std::fs::remove_file(log_file).unwrap();
    }

    /// End-to-end check of the real spawn → liveness → signal path.
    #[test]
    fn detached_process_is_alive_then_dies_on_kill() {
        use crate::session::proc;
        use std::time::Duration;

        let dir = std::env::temp_dir().join("llmctl-supervisor-test");
        std::fs::create_dir_all(&dir).unwrap();
        let mut sup = DetachedSupervisor::new();
        let spec =
            LaunchSpec { argv: vec!["sleep".into(), "30".into()], log_file: dir.join("s.log") };

        let spawned = sup.spawn(&spec).expect("spawn sleep");
        std::thread::sleep(Duration::from_millis(150)); // let exec happen
        assert!(proc::is_alive(spawned.pid), "process should be alive after spawn");
        assert!(proc::cmdline_matches(spawned.pid, "30"), "cmdline should include the arg");

        sup.kill(spawned.pid).expect("kill group");
        let mut dead = false;
        for _ in 0..40 {
            sup.reap();
            if !Path::new(&format!("/proc/{}", spawned.pid)).exists() {
                dead = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(dead, "process must be reaped after SIGKILL");
        assert!(sup.children.is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
