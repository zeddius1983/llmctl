//! Runtime discovery for llama.cpp: locate the server binary, capture its
//! version, and cache `--help` output for later use (option validation, etc.).

use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::{debug, warn};

use crate::config::LlamaCppConfig;
use crate::domain::Runtime;

/// Discover the llama.cpp runtime from configuration.
pub fn discover_llama_cpp(cfg: &LlamaCppConfig, cache_dir: &Path) -> Runtime {
    let binary_path = resolve_binary(&cfg.binary);
    let version = binary_path.as_deref().and_then(query_version);

    if let Some(path) = &binary_path {
        if let Err(err) = cache_help(path, cache_dir) {
            debug!(%err, "could not cache llama-server --help");
        }
    } else {
        warn!(binary = %cfg.binary, "llama-server binary not found");
    }

    Runtime {
        name: "llama.cpp".into(),
        description: "GGUF inference via llama-server".into(),
        version,
        binary_path,
        formats: vec!["GGUF".into()],
    }
}

/// Resolve a binary to an absolute path: honor an explicit path, else search
/// `$PATH`.
fn resolve_binary(binary: &str) -> Option<PathBuf> {
    let candidate = Path::new(binary);
    if candidate.is_absolute() || binary.contains('/') {
        return candidate.exists().then(|| candidate.to_path_buf());
    }
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var).map(|dir| dir.join(binary)).find(|p| p.is_file())
}

/// Run `--version` and return a short version string. llama.cpp prints version
/// info to stderr, so both streams are considered.
fn query_version(path: &Path) -> Option<String> {
    let output = Command::new(path).arg("--version").output().ok()?;
    let text = if output.stderr.is_empty() { &output.stdout } else { &output.stderr };
    let text = String::from_utf8_lossy(text);
    text.lines().map(str::trim).find(|l| !l.is_empty()).map(|l| l.to_string())
}

/// Capture `--help` to `<cache_dir>/llama-server.help.txt`.
fn cache_help(path: &Path, cache_dir: &Path) -> std::io::Result<()> {
    let output = Command::new(path).arg("--help").output()?;
    let body = if output.stdout.is_empty() { output.stderr } else { output.stdout };
    std::fs::write(cache_dir.join("llama-server.help.txt"), body)
}
