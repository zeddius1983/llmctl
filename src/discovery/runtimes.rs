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
    let devices = binary_path.as_deref().map(query_devices).unwrap_or_default();

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
        devices,
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

/// Run `--list-devices` and extract device identifiers from lines such as
/// `ROCm0: AMD Radeon ...`. Both streams are considered because llama.cpp's
/// informational output has moved between stdout and stderr across versions.
fn query_devices(path: &Path) -> Vec<String> {
    let Ok(output) = Command::new(path).arg("--list-devices").output() else {
        return Vec::new();
    };
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push('\n');
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    parse_devices(&text)
}

fn parse_devices(output: &str) -> Vec<String> {
    let mut devices = Vec::new();
    let mut in_device_list = false;
    for line in output.lines().map(str::trim) {
        if line.eq_ignore_ascii_case("Available devices:") {
            in_device_list = true;
            continue;
        }
        if !in_device_list {
            continue;
        }
        let Some((name, _description)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty()
            || name.chars().any(char::is_whitespace)
            || !name.ends_with(|c: char| c.is_ascii_digit())
        {
            continue;
        }
        if !devices.iter().any(|device| device == name) {
            devices.push(name.to_string());
        }
    }
    devices
}

/// Capture `--help` to `<cache_dir>/llama-server.help.txt`.
fn cache_help(path: &Path, cache_dir: &Path) -> std::io::Result<()> {
    let output = Command::new(path).arg("--help").output()?;
    let body = if output.stdout.is_empty() { output.stderr } else { output.stdout };
    std::fs::write(cache_dir.join("llama-server.help.txt"), body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_deduplicates_list_devices_output() {
        let output = "Available devices:\n  ROCm0: AMD Radeon RX 7900 XTX\n\
                      Vulkan0: AMD Radeon RX 7900 XTX\n  ROCm0: duplicate\n";
        assert_eq!(parse_devices(output), vec!["ROCm0", "Vulkan0"]);
    }

    #[test]
    fn ignores_headers_and_unrelated_diagnostics() {
        let output = "ggml: initialization message\nAvailable devices:\nno devices found\nwarning: ignored\n";
        assert!(parse_devices(output).is_empty());
    }
}
