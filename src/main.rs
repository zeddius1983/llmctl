//! llmctl — keyboard-driven TUI for managing local llama.cpp servers.

mod app;
mod config;
mod discovery;
mod domain;
mod profiles;
mod ui;

use anyhow::Result;

use crate::app::App;
use crate::config::{Config, Paths};

fn main() -> Result<()> {
    let paths = Paths::resolve()?;
    paths.ensure_dirs()?;
    init_tracing(&paths);

    let config = Config::load()?;

    let mut terminal = ratatui::init();
    let result = App::new(config, paths).run(&mut terminal);
    ratatui::restore();

    result
}

/// Logs go to a file under the state dir; writing to stderr would corrupt the
/// alternate-screen TUI. Controlled by `RUST_LOG` (default: info).
fn init_tracing(paths: &Paths) {
    use tracing_subscriber::EnvFilter;

    let log_file = paths.log_dir.join("llmctl.log");
    if let Ok(file) = std::fs::OpenOptions::new().create(true).append(true).open(&log_file) {
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(file)
            .with_ansi(false)
            .try_init();
    }
}
