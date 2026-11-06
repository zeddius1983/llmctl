//! llmctl — keyboard-driven TUI for managing local LLM inference servers.

mod app;
mod config;
mod discovery;
mod domain;
mod persistence;
mod profiles;
mod runtime;
mod session;
mod ui;

use anyhow::Result;

use crate::app::App;
use crate::config::{Config, Paths};

fn main() -> Result<()> {
    let paths = Paths::resolve()?;
    paths.ensure_dirs()?;
    init_tracing(&paths);

    let config = Config::load()?;

    let mut terminal = app::init_terminal();
    // `ratatui::init` restores the screen on a panic but knows nothing of mouse
    // capture; without this a crash would leave the shell receiving mouse codes.
    let restore_screen = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);
        restore_screen(info);
    }));
    let result = App::new(config, paths).run(&mut terminal);
    app::restore_terminal();

    result
}

/// Logs go to a file under the state dir; writing to stderr would corrupt the
/// alternate-screen TUI. Controlled by `RUST_LOG` (default: info).
fn init_tracing(paths: &Paths) {
    use tracing_subscriber::EnvFilter;

    let log_file = paths.log_dir.join("llmctl.log");
    if let Ok(file) = std::fs::OpenOptions::new().create(true).append(true).open(&log_file) {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(file)
            .with_ansi(false)
            .try_init();
    }
}
