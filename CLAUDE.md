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
- **serde** / **serde_json** / **serde_yaml** / **toml** — config, catalog,
  cache, and profile persistence.
- **directories** — XDG base directories.
- **walkdir** + **regex** — model discovery.
- **anyhow** / **thiserror** — errors. **tracing** — file-based logging.
- **libc** — `setsid`/signals for detached sessions, `/proc` sampling, `sysconf`.
  No async runtime: a poll-based tick (`crossterm::event::poll`) drives live
  session refresh instead of tokio (ADR-007).

## Architecture (summary)

Yazi-style sliding three-column view (Parent | Current | Preview) over the
hierarchy `root ▸ Runtime ▸ source ▸ provider/repository ▸ Model ▸ Profile ▸ Options`.
The catalog portion has variable depth. Child lists are derived
from the parent selection. See [docs/architecture.md](docs/architecture.md) for
component structure and data flow.

## Directory layout

```
src/
  main.rs        entry: XDG paths, file tracing, launch TUI
  app/           App state, event loop, navigation, prompts, actions
  config/        Config (first-run config.toml generation) + XDG Paths resolution
  domain/        pure types (Runtime, Model, Profile, OptionItem), helpers, vLLM stubs
  discovery/     catalog.rs (source parsing + managed tree), gguf.rs (header parser),
                 models.rs (scan+cache), runtimes.rs (llama.cpp)
  profiles/      registry.rs (option specs), templates.rs, store.rs (per-model YAML), mod.rs (resolution)
  session/       command.rs (builder), supervisor.rs (DetachedSupervisor: setsid/signals),
                 record.rs (session-<id>.json), proc.rs (/proc), health.rs (/health), mod.rs (SessionManager)
  ui/            ratatui rendering (browser columns, Session Manager, log view, footer, prompts, help)
docs/            requirements, architecture, decisions (ADRs), roadmap
```

XDG paths used at runtime:
`~/.config/llmctl/config.toml`, `~/.config/llmctl/models/` (managed model
catalog + per-model YAML profiles), `~/.local/state/llmctl/` (logs, sessions,
legacy profile migration), `~/.cache/llmctl/` (models.json, llama-server.help.txt).

## Key design decisions (see decisions.md for full ADRs)

- Rust + ratatui (not Go/Bubble Tea or Python/Textual) — ADR-001.
- Profiles scoped per **runtime + model**; built-ins are global read-only
  templates that fork into instances on edit — ADR-002.
- GGUF / llama.cpp only in the MVP — ADR-003.
- Yazi sliding 3-column navigation, not fixed panes — ADR-004.
- Sessions: detached processes (`setsid`) + rediscover, behind a
  `SessionSupervisor` trait — ADR-005 (implemented in Phase 3).
- Synchronous poll-tick refresh + `libc` for process control, not tokio/nix —
  ADR-007.
- Source-aware physical model catalog with per-model profiles — ADR-009.

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
  and `session/`.

## Dev & branching guidelines

- **Branch naming:** every branch is prefixed with `feature/` or `bugfix/`,
  followed by a short task name of **1–3 words** (kebab-case) that reflects the
  work. Examples: `feature/launch-sessions`, `feature/model-discovery`,
  `bugfix/shard-size`.
- A `feature/` branch may instead name a **target version** (e.g.
  `feature/v0.0.1`) to act as an umbrella that accumulates several features
  before a release.
- **Release plan:** each remaining phase is built on its own `feature/<task>`
  branch. Once all planned phases are complete, they are merged together into
  **`feature/v0.1.0`** (the release umbrella), which is then merged to `main`.
  (The early `phase-*` and `docs` branches predate this policy and are
  grandfathered.)
- Commit only when asked. Do not add AI co-author trailers or attribution to
  commit messages unless the user explicitly requests it.
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
