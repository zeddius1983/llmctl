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
