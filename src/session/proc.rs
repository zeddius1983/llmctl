//! Linux `/proc` helpers for inspecting detached server processes.
//!
//! Used to rediscover sessions after an llmctl restart (liveness + cmdline
//! match guards against PID reuse) and to show live resource usage.

use std::fs;

/// Is a process with `pid` currently alive (and not a zombie)?
pub fn is_alive(pid: i32) -> bool {
    match fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => {
            // Field 3 (after `comm`, which may contain spaces/parens) is state.
            // `Z` is a zombie — treat as dead.
            stat.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().next()) != Some("Z")
        }
        Err(_) => false,
    }
}

/// The process's command line as a vector of arguments, or empty if unreadable.
pub fn cmdline(pid: i32) -> Vec<String> {
    match fs::read(format!("/proc/{pid}/cmdline")) {
        Ok(bytes) => bytes
            .split(|b| *b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Does the live process at `pid` look like the server we launched? We require
/// the recorded model path to appear in its current command line.
pub fn cmdline_matches(pid: i32, needle: &str) -> bool {
    cmdline(pid).iter().any(|arg| arg == needle)
}

/// The process's `comm` (executable name, truncated to 15 chars by the kernel).
pub fn comm(pid: i32) -> Option<String> {
    fs::read_to_string(format!("/proc/{pid}/comm")).ok().map(|s| s.trim_end().to_string())
}

/// Re-acquire a running server by identity when its launcher re-exec'd or
/// daemonized it under a different pid (and possibly its own session). We match
/// the server executable's name plus the unique model path and port from the
/// launch command, ignoring shell/launcher wrappers that merely carry the same
/// args. Returns the lowest matching live pid.
pub fn find_server(binary: &str, model_path: &str, port: u16) -> Option<i32> {
    let exe = std::path::Path::new(binary).file_name()?.to_string_lossy().into_owned();
    let exe = &exe[..exe.len().min(15)]; // kernel truncates comm to TASK_COMM_LEN-1
    let port = port.to_string();
    let mut best: Option<i32> = None;
    for entry in fs::read_dir("/proc").ok()?.filter_map(|e| e.ok()) {
        let Ok(name) = entry.file_name().into_string() else { continue };
        let Ok(pid) = name.parse::<i32>() else { continue };
        if comm(pid).as_deref() != Some(exe) || !is_alive(pid) {
            continue;
        }
        let args = cmdline(pid);
        if args.iter().any(|a| a == model_path) && args.iter().any(|a| *a == port) {
            best = Some(best.map_or(pid, |b| b.min(pid)));
        }
    }
    best
}

/// Parent PID from `/proc/<pid>/status`, or `None` if unreadable.
pub fn ppid(pid: i32) -> Option<i32> {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    status.lines().find_map(|l| l.strip_prefix("PPid:")?.trim().parse().ok())
}

/// Every descendant PID of `pid` (children, grandchildren, …), found by walking
/// the parent links of all live processes. Used to signal an entire server
/// subtree: the recorded pid may be a launcher wrapper that forked the real
/// `llama-server`, so signalling only the recorded process group can miss it.
pub fn descendants(pid: i32) -> Vec<i32> {
    let mut children: std::collections::HashMap<i32, Vec<i32>> = std::collections::HashMap::new();
    let Ok(entries) = fs::read_dir("/proc") else {
        return Vec::new();
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let Ok(name) = entry.file_name().into_string() else { continue };
        let Ok(child) = name.parse::<i32>() else { continue };
        if let Some(parent) = ppid(child) {
            children.entry(parent).or_default().push(child);
        }
    }

    let mut out = Vec::new();
    let mut stack = vec![pid];
    while let Some(p) = stack.pop() {
        for &k in children.get(&p).map(Vec::as_slice).unwrap_or(&[]) {
            if !out.contains(&k) {
                out.push(k);
                stack.push(k);
            }
        }
    }
    out
}

/// Resident set size in bytes (physical memory), from `/proc/<pid>/statm`.
pub fn rss_bytes(pid: i32) -> Option<u64> {
    let statm = fs::read_to_string(format!("/proc/{pid}/statm")).ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * page_size())
}

/// A point-in-time CPU sample for one process: its busy jiffies and the
/// system-wide total jiffies, so a later sample yields a delta-based percentage.
#[derive(Debug, Clone, Copy)]
pub struct CpuSample {
    proc_jiffies: u64,
    total_jiffies: u64,
}

/// Take a CPU sample for `pid`, or `None` if the process can't be read.
pub fn cpu_sample(pid: i32) -> Option<CpuSample> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // utime (14) + stime (15), counting fields after the closing paren of comm.
    let rest = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // After the paren, index 0 = state, so utime is index 11 and stime 12.
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(CpuSample { proc_jiffies: utime + stime, total_jiffies: total_jiffies()? })
}

/// CPU usage between two samples, in "one core = 100%" units (can exceed 100%).
pub fn cpu_percent(prev: CpuSample, now: CpuSample) -> Option<f64> {
    let proc_delta = now.proc_jiffies.checked_sub(prev.proc_jiffies)?;
    let total_delta = now.total_jiffies.checked_sub(prev.total_jiffies)?;
    if total_delta == 0 {
        return None;
    }
    let ncpu = num_cpus().max(1) as f64;
    Some(proc_delta as f64 / total_delta as f64 * ncpu * 100.0)
}

/// Sum of all jiffies on the aggregate `cpu` line of `/proc/stat`.
fn total_jiffies() -> Option<u64> {
    let stat = fs::read_to_string("/proc/stat").ok()?;
    let line = stat.lines().next()?; // "cpu  u n s i w ..."
    Some(line.split_whitespace().skip(1).filter_map(|v| v.parse::<u64>().ok()).sum())
}

fn page_size() -> u64 {
    // SAFETY: sysconf with a constant name has no preconditions.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 { v as u64 } else { 4096 }
}

fn num_cpus() -> usize {
    // SAFETY: sysconf with a constant name has no preconditions.
    let v = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    if v > 0 { v as usize } else { 1 }
}
