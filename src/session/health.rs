//! Minimal, dependency-free HTTP `/health` probe for llama-server.
//!
//! llama-server returns `200` on `GET /health` once the model is loaded and
//! `503` while still loading, so this lets us move a session from `Starting`
//! to `Running` without pulling in an HTTP client crate.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// The outcome of a single health probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    /// `GET /health` returned `200` — the server is ready.
    Ready,
    /// The port accepted a connection but the server isn't ready yet
    /// (e.g. `503` while the model loads).
    Loading,
    /// Could not connect (port closed / process not listening yet).
    Down,
}

/// Probe `http://{host}:{port}/health` with a short timeout.
///
/// `host` may be a bind address like `0.0.0.0`; we probe `127.0.0.1` in that
/// case since the wildcard address isn't directly connectable.
pub fn probe(host: &str, port: u16) -> Health {
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

    let req = format!("GET /health HTTP/1.0\r\nHost: {connect_host}\r\nConnection: close\r\n\r\n");
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
