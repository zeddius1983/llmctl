//! Configuration and XDG path resolution.
//!
//! llmctl follows the XDG specification:
//!   ~/.config/llmctl/config.toml   (config)
//!   ~/.local/state/llmctl/         (sessions, logs)
//!   ~/.cache/llmctl/               (model/runtime scan cache)

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::Deserialize;
use tracing::warn;

const DEFAULT_CONFIG: &str = r#"# llmctl configuration
#
# Model sources are scanned recursively for GGUF files. `directory` preserves
# relative folders; the two store-specific layouts normalize their cache paths.

[[models.sources]]
name = "llama-cache"
path = "~/.cache/llama.cpp"
layout = "directory"

[[models.sources]]
name = "huggingface"
path = "~/.cache/huggingface/hub"
layout = "hugging-face"

[[models.sources]]
name = "lmstudio"
path = "~/.lmstudio/models"
layout = "lm-studio"

[[models.sources]]
name = "models"
path = "~/models"
layout = "directory"

[runtime.llama_cpp]
binary = "llama-server"

[defaults]
host = "127.0.0.1"
port = 8000
"#;

/// Parsed `config.toml`. Missing sections/fields fall back to defaults so a
/// brand-new install runs with zero configuration.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub models: ModelsConfig,
    pub runtime: RuntimeConfig,
    pub defaults: Defaults,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct ModelsConfig {
    /// Directories scanned (recursively) for GGUF models. Never defaults to
    /// `$HOME` — recursive scanning only happens inside configured paths.
    pub paths: Vec<PathBuf>,
    /// Named model roots. Known layouts are parsed semantically; arbitrary
    /// directories retain their relative hierarchy below `name`.
    pub sources: Vec<ModelSourceConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSourceConfig {
    pub name: String,
    pub path: PathBuf,
    #[serde(default)]
    pub layout: ModelLayout,
}

#[derive(Debug, Clone, Copy, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelLayout {
    #[default]
    Auto,
    Directory,
    Flat,
    LmStudio,
    HuggingFace,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct RuntimeConfig {
    pub llama_cpp: LlamaCppConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LlamaCppConfig {
    /// Server binary name or absolute path. Resolved on `$PATH` if not absolute.
    pub binary: String,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self { binary: "llama-server".to_string() }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Defaults {
    pub host: String,
    pub port: u16,
}

impl Default for Defaults {
    fn default() -> Self {
        Self { host: "127.0.0.1".to_string(), port: 8000 }
    }
}

impl Config {
    /// Load configuration, creating a documented default on first run.
    pub fn load() -> Result<Self> {
        let paths = Paths::resolve()?;
        Self::load_from(&paths.config_file)
    }

    fn load_from(path: &Path) -> Result<Self> {
        ensure_default_config(path)?;
        let legacy = path.with_extension("yaml");
        if legacy.exists() {
            warn!(
                path = %legacy.display(),
                "legacy config.yaml is ignored; keep it as a backup until its presets are migrated"
            );
        }
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
    }
}

fn ensure_default_config(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    match std::fs::OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => file
            .write_all(DEFAULT_CONFIG.as_bytes())
            .with_context(|| format!("writing default {}", path.display())),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(err) => Err(err).with_context(|| format!("creating default {}", path.display())),
    }
}

/// Resolved on-disk locations for the current user.
#[derive(Debug, Clone)]
pub struct Paths {
    pub config_file: PathBuf,
    pub models_dir: PathBuf,
    pub state_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub log_dir: PathBuf,
    pub sessions_dir: PathBuf,
}

impl Paths {
    pub fn resolve() -> Result<Self> {
        let dirs = ProjectDirs::from("", "", "llmctl")
            .context("could not determine XDG base directories")?;
        let state_dir = dirs.state_dir().unwrap_or_else(|| dirs.data_dir()).to_path_buf();
        Ok(Self {
            config_file: dirs.config_dir().join("config.toml"),
            models_dir: dirs.config_dir().join("models"),
            log_dir: state_dir.join("logs"),
            sessions_dir: state_dir.join("sessions"),
            state_dir,
            cache_dir: dirs.cache_dir().to_path_buf(),
        })
    }

    /// Create the state/cache directory tree if it does not exist yet.
    pub fn ensure_dirs(&self) -> Result<()> {
        for dir in
            [&self.state_dir, &self.cache_dir, &self.log_dir, &self.sessions_dir, &self.models_dir]
        {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn first_load_creates_parseable_default_with_standard_sources() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-config-{nonce}"));
        let path = root.join("llmctl/config.toml");
        let config = Config::load_from(&path).unwrap();

        assert!(path.is_file());
        assert_eq!(config.models.sources.len(), 4);
        assert_eq!(config.models.sources[0].name, "llama-cache");
        assert_eq!(config.models.sources[1].layout, ModelLayout::HuggingFace);
        assert_eq!(config.models.sources[2].layout, ModelLayout::LmStudio);
        assert_eq!(config.models.sources[3].path, PathBuf::from("~/models"));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn existing_config_is_never_replaced() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("llmctl-config-existing-{nonce}"));
        let path = root.join("config.toml");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(&path, "[defaults]\nport = 9000\n").unwrap();

        let config = Config::load_from(&path).unwrap();
        assert_eq!(config.defaults.port, 9000);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "[defaults]\nport = 9000\n");

        std::fs::remove_dir_all(root).unwrap();
    }
}
