//! Configuration and XDG path resolution.
//!
//! llmctl follows the XDG specification:
//!   ~/.config/llmctl/config.toml   (config)
//!   ~/.local/state/llmctl/         (sessions, logs)
//!   ~/.cache/llmctl/               (model/runtime scan cache)

use std::path::PathBuf;

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::Deserialize;

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
    /// Load configuration, falling back to defaults when no file is present.
    pub fn load() -> Result<Self> {
        let paths = Paths::resolve()?;
        if paths.config_file.exists() {
            let raw = std::fs::read_to_string(&paths.config_file)
                .with_context(|| format!("reading {}", paths.config_file.display()))?;
            let cfg: Config = toml::from_str(&raw)
                .with_context(|| format!("parsing {}", paths.config_file.display()))?;
            Ok(cfg)
        } else {
            Ok(Config::default())
        }
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
