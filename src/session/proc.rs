//! Linux `/proc` helpers for inspecting detached server processes.
//!
//! Used to rediscover sessions after an llmctl restart (liveness + cmdline
//! match guards against PID reuse) and to find the real server behind a
//! launcher wrapper.

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
        if args.iter().any(|a| a == model_path) && args.contains(&port) {
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
