# CLAUDE.md

Stable, slow-changing context for working in this repo. For volatile details
(what's done, what's next) see [docs/roadmap.md](docs/roadmap.md). For the
"why" behind choices see [docs/decisions.md](docs/decisions.md).

## Project purpose

`llmctl` is a keyboard-driven terminal UI (TUI) for discovering, configuring,
launching, and managing local LLM inference servers — in the style of Yazi,
Lazygit, and systemctl. The goal: never hand-type a complex `llama-server`
command again. The MVP targets **llama.cpp + GGUF on Linux**; other runtimes
(vLLM, Ollama, …) are future work. Full spec: [docs/requirements.md](docs/requirements.md).

## Tech stack

- **Rust** (edition 2024) — single static binary, fast startup.
- **ratatui** + **crossterm** — TUI rendering and terminal/input handling.
- **serde** / **serde_json** / **toml** — config and cache/state persistence.
- **directories** — XDG base directories.
- **walkdir** + **regex** — model discovery.
- **anyhow** / **thiserror** — errors. **tracing** — file-based logging.
- Planned (Phase 3): **tokio** (async process mgmt), **nix** (`setsid`, signals).

## Architecture (summary)

Yazi-style sliding three-column view (Parent | Current | Preview) over the
hierarchy `root ▸ Runtime ▸ Model ▸ Profile ▸ Options`. Child lists are derived
from the parent selection. See [docs/architecture.md](docs/architecture.md) for
component structure and data flow.

## Directory layout

```
src/
  main.rs        entry: XDG paths, file tracing, launch TUI
  app/           App state, event loop, navigation, prompts, actions
  config/        Config (config.toml) + XDG Paths resolution
  domain/        pure types (Runtime, Model, Profile, OptionItem), helpers, vLLM stubs
  discovery/     gguf.rs (header parser), models.rs (scan+cache), runtimes.rs (llama.cpp)
  profiles/      registry.rs (option specs), templates.rs, store.rs (persistence), mod.rs (resolution)
  ui/            ratatui rendering (columns, 3-line footer, prompts, help)
docs/            requirements, architecture, decisions (ADRs), roadmap
```

XDG paths used at runtime:
`~/.config/llmctl/config.toml`, `~/.local/state/llmctl/` (profiles.json, logs,
sessions), `~/.cache/llmctl/` (models.json, llama-server.help.txt).

## Key design decisions (see decisions.md for full ADRs)

- Rust + ratatui (not Go/Bubble Tea or Python/Textual) — ADR-001.
- Profiles scoped per **runtime + model**; built-ins are global read-only
  templates that fork into instances on edit — ADR-002.
- GGUF / llama.cpp only in the MVP — ADR-003.
- Yazi sliding 3-column navigation, not fixed panes — ADR-004.
- Sessions: detached processes + rediscover, behind a `SessionSupervisor`
  trait (Phase 3) — ADR-005.

## Coding standards

- Match the style of surrounding code (naming, comment density, idioms).
- `cargo build` must be **warning-free**; run `cargo fmt`. Use `#[allow(dead_code)]`
  with a note (e.g. "used in Phase N") only for genuinely forward-looking fields.
- Unit-test pure logic (resolution, validation, parsing). The TUI is smoke-tested
  via a PTY driver (`$CLAUDE_JOB_DIR/tmp/drive.py`); per-key delays matter, and
  escape sequences (Home/End/arrows) get split by the driver — rely on unit tests
  for those.
- Logs go to a **file** under the state dir, never stderr (it corrupts the TUI).
- Keep `domain/` IO-free. Discovery/process/IO lives in `discovery/`, `profiles/`,
  (and future `session/`).

## Dev & branching guidelines

- Work proceeds in phases (see roadmap). Each phase lands on its own branch
  `phase-N-name`, branched from the previous phase's branch (stacked); not yet
  merged to `main`. Docs/meta work goes on a `docs`/topic branch.
- Commit only when asked. Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Don't commit the legacy Go `llmctl` binary or `/target` (see `.gitignore`).

## Context compaction companion

Before compacting the conversation, update these files so project state can be
reconstructed from the repo rather than chat history:

1. **CLAUDE.md** — if stable facts changed (stack, layout, standards).
2. **docs/decisions.md** — append/adjust ADRs for any decisions made.
3. **docs/roadmap.md** — move completed items to "Done", update "In progress"
   and "Next", note any new follow-ups.

After compaction, prefer these files (plus the code and git log) as the source
of truth over recalled conversation.
