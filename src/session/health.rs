//! Minimal, dependency-free HTTP readiness probe.
//!
//! Runtimes disagree on which path answers "am I ready?" — llama-server has a
//! dedicated `/health`, while FastFlowLM has none and its `/v1/models` serves
//! the same purpose — so the path is the caller's (that is, the backend's)
//! choice. What they share is the contract: `200` means loaded and ready,
//! anything else on an open port means still starting. Keeping this hand-rolled
//! avoids pulling an HTTP client into the tick loop.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// The outcome of a single health probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    /// The probe returned `200` — the server is ready.
    Ready,
    /// The port accepted a connection but the server isn't ready yet
    /// (e.g. `503` while the model loads).
    Loading,
    /// Could not connect (port closed / process not listening yet).
    Down,
}

/// Probe `http://{host}:{port}{path}` with a short timeout.
///
/// `host` may be a bind address like `0.0.0.0`; we probe `127.0.0.1` in that
/// case since the wildcard address isn't directly connectable.
pub fn probe(host: &str, port: u16, path: &str) -> Health {
    let connect_host = match host {
        "0.0.0.0" | "::" | "" => "127.0.0.1",
        other => other,
    };
    let timeout = Duration::from_millis(300);

    let Some(addr) = (connect_host, port).to_socket_addrs().ok().and_then(|mut a| a.next()) else {
        return Health::Down;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, timeout) else {
        return Health::Down;
    };
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));

    let req = format!("GET {path} HTTP/1.0\r\nHost: {connect_host}\r\nConnection: close\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return Health::Loading; // connected but couldn't speak; treat as not-ready
    }

    let mut buf = [0u8; 256];
    match stream.read(&mut buf) {
        Ok(n) if n > 0 => {
            let head = String::from_utf8_lossy(&buf[..n]);
            if head.starts_with("HTTP/") && head.contains(" 200") {
                Health::Ready
            } else {
                Health::Loading
            }
        }
        _ => Health::Loading,
    }
}

/// Is `port` free to bind on `127.0.0.1`? Used for auto port-conflict resolution.
pub fn port_is_free(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

/// Identifies a probe attempt across restart, PID reacquisition, and endpoint changes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProbeKey {
    pub session: String,
    pub pid: i32,
    pub host: String,
    pub port: u16,
    pub path: String,
}

/// At most four socket probes run concurrently; submitting/polling never waits.
pub struct HealthChecks {
    pending: std::collections::HashSet<ProbeKey>,
    queued: std::collections::VecDeque<ProbeKey>,
    active: usize,
    tx: std::sync::mpsc::Sender<(ProbeKey, Health)>,
    rx: std::sync::mpsc::Receiver<(ProbeKey, Health)>,
    probe: std::sync::Arc<dyn Fn(&ProbeKey) -> Health + Send + Sync>,
}

impl Default for HealthChecks {
    fn default() -> Self {
        Self::new(std::sync::Arc::new(|key| probe(&key.host, key.port, &key.path)))
    }
}

impl HealthChecks {
    fn new(probe: std::sync::Arc<dyn Fn(&ProbeKey) -> Health + Send + Sync>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self { pending: Default::default(), queued: Default::default(), active: 0, tx, rx, probe }
    }

    pub fn request(&mut self, key: ProbeKey) {
        if !self.pending.insert(key.clone()) {
            return;
        }
        self.queued.push_back(key);
        self.start_queued();
    }

    fn start_queued(&mut self) {
        while self.active < 4 {
            let Some(key) = self.queued.pop_front() else { break };
            self.active += 1;
            let tx = self.tx.clone();
            let probe = self.probe.clone();
            std::thread::spawn(move || {
                let result = probe(&key);
                let _ = tx.send((key, result));
            });
        }
    }

    pub fn poll(&mut self) -> Vec<(ProbeKey, Health)> {
        let results: Vec<_> = self.rx.try_iter().collect();
        for (key, _) in &results {
            self.pending.remove(key);
            self.active -= 1;
        }
        self.start_queued();
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Condvar, Mutex, mpsc};
    use std::time::Instant;

    #[test]
    fn workers_are_bounded_deduplicated_and_do_not_starve_queued_sessions() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let worker_gate = gate.clone();
        let (tx, rx) = mpsc::channel();
        let mut checks = HealthChecks::new(Arc::new(move |key| {
            tx.send(key.clone()).unwrap();
            let (lock, ready) = &*worker_gate;
            let guard = lock.lock().unwrap();
            let _guard =
                ready.wait_timeout_while(guard, Duration::from_secs(2), |open| !*open).unwrap();
            Health::Ready
        }));
        let keys: Vec<_> = (0..5)
            .map(|i| ProbeKey {
                session: i.to_string(),
                pid: i,
                host: "localhost".into(),
                port: 80,
                path: "/".into(),
            })
            .collect();
        for key in &keys {
            checks.request(key.clone());
            checks.request(key.clone());
        }
        assert_eq!(checks.active, 4);
        assert_eq!(checks.queued.len(), 1);
        assert_eq!(checks.pending.len(), 5);
        for _ in 0..4 {
            rx.recv_timeout(Duration::from_secs(1)).unwrap();
        }
        assert!(rx.try_recv().is_err());
        assert!(checks.poll().is_empty());
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut completed = Vec::new();
        while completed.len() < 5 && Instant::now() < deadline {
            completed.extend(checks.poll());
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(completed.len(), 5);
        assert_eq!(rx.recv_timeout(Duration::from_secs(1)).unwrap(), keys[4]);
        assert!(checks.pending.is_empty());
        assert_eq!(checks.active, 0);
    }
}
