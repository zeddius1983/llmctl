//! Runtime discovery: locate configured binaries, capture their versions, and
//! cache help output for later option compatibility checks.

use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::{debug, warn};

use crate::config::{LlamaCppConfig, VllmConfig};
use crate::domain::{Runtime, RuntimeId};

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
        id: RuntimeId::LlamaCpp,
        name: "llama.cpp".into(),
        description: "GGUF inference via llama-server".into(),
        version,
        binary_path,
        formats: vec!["GGUF".into()],
    }
}

/// Discover vLLM's top-level CLI without executing it on the startup path.
/// `vllm --version` can take several seconds for Python/container installs, so
/// version/help inspection is deferred until there is an explicit UI action.
pub fn discover_vllm(cfg: &VllmConfig) -> Runtime {
    let binary_path = resolve_binary(&cfg.binary);

    if binary_path.is_none() {
        warn!(binary = %cfg.binary, "vllm binary not found");
    }

    Runtime {
        id: RuntimeId::Vllm,
        name: "vLLM".into(),
        description: "High-throughput serving for Hugging Face models".into(),
        version: None,
        binary_path,
        formats: vec!["Safetensors".into(), "Hugging Face".into()],
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn discovers_vllm_without_executing_it() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        // Keep the cache under the workspace: some containerized test runners
        // expose a split /tmp namespace to spawned commands.
        let root = std::env::current_dir()
            .unwrap()
            .join("target/test-tmp")
            .join(format!("llmctl-vllm-runtime-{nonce}"));
        std::fs::create_dir_all(&root).unwrap();
        let binary = root.join("vllm");
        std::fs::write(
            &binary,
            format!("#!/bin/sh\necho called >> '{}'/calls\necho 'vllm 0.17.0'\n", root.display()),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&binary).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&binary, permissions).unwrap();

        let runtime = discover_vllm(&VllmConfig { binary: binary.display().to_string() });
        assert_eq!(runtime.id, RuntimeId::Vllm);
        assert_eq!(runtime.version, None);
        assert_eq!(runtime.binary_path.as_deref(), Some(binary.as_path()));
        assert!(!root.join("calls").exists(), "runtime discovery must not execute vllm");
        std::fs::remove_dir_all(root).unwrap();
    }
}
