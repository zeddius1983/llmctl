# CLAUDE.md

Stable, slow-changing context for working in this repo. For volatile details
(what's done, what's next) see [docs/roadmap.md](docs/roadmap.md). For the
"why" behind choices see [docs/decisions.md](docs/decisions.md).

## Project purpose

`llmctl` is a keyboard-driven terminal UI (TUI) for discovering, configuring,
launching, and managing local LLM inference servers — in the style of Yazi,
Lazygit, and systemctl. The goal: never hand-type a complex inference-server
command again. Two runtimes ship today, both Linux: **llama.cpp + GGUF** (CPU/GPU)
and **FastFlowLM** (`flm`, AMD XDNA2 NPU). Others (vLLM, Ollama, …) are future
work and plug in behind the `RuntimeBackend` trait. Full spec:
[docs/requirements.md](docs/requirements.md).

## Tech stack

- **Rust** (edition 2024) — single static binary, fast startup.
- **ratatui** + **crossterm** — TUI rendering and terminal/input handling.
- **serde** / **serde_json** / **serde_yaml** / **toml** — config, catalog,
  cache, and profile persistence.
- **directories** — XDG base directories.
- **walkdir** + **regex** — model discovery and download progress sizing.
- **ureq** — blocking HTTPS used from background workers for Hugging Face browsing.
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

Everything runtime-specific sits behind the `RuntimeBackend` trait in
`src/runtime/` (ADR-011): binary discovery, the option schema and CLI dialect,
built-in templates, model enumeration, launch/chat/benchmark argv, the readiness
path, and the `/proc` identity token. The app, profile store, and session manager
dispatch through `&dyn RuntimeBackend` and hold no per-runtime knowledge. Adding
a runtime = a module under `src/runtime/` + one entry in `runtime::discover`.

## Directory layout

```
src/
  main.rs        entry: XDG paths, file tracing, launch TUI
  app/           App state, event loop, navigation, prompts, actions
  config/        Config (first-run config.toml generation) + XDG Paths resolution
  domain/        pure types (Runtime, Model, FlmModel, Profile, OptionItem), helpers
  discovery/     catalog.rs (source parsing + managed tree), gguf.rs (header parser),
                 models.rs (scan+cache), online.rs (lazy Hugging Face catalog),
                 hf.rs (shared Hub transfer: Range resume, tree API, URLs)
  runtime/       mod.rs (RuntimeBackend trait, CatalogCtx, LaunchContext, discover),
                 llama_cpp.rs (llama-server: discovery, option table, argv, /health),
                 flm.rs (FastFlowLM: flm discovery+validate, catalog, options,
                 resumable Hub downloads)
  profiles/      registry.rs (generic option model + OptionSchema), templates.rs
                 (Template type; tables live with the backends), store.rs
                 (per-model YAML), mod.rs (resolution layering)
  session/       command.rs (builder), supervisor.rs (DetachedSupervisor: setsid/signals),
                 record.rs (session-<id>.json), proc.rs (/proc), health.rs (/health), mod.rs (SessionManager)
  ui/            ratatui rendering (browser columns, Session Manager, log view, footer, prompts, help)
docs/            requirements, architecture, decisions (ADRs), roadmap
```

FastFlowLM keeps its own models under `~/.config/flm/models/` (or
`$FLM_MODEL_PATH`); llmctl reads that catalog via `flm list --json` and never
scans it.

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
- Lazy `online ▸ huggingface` catalog, with native llama.cpp launches and
  resumable llmctl-managed downloads — ADR-010.
- Runtimes behind a `RuntimeBackend` trait; the vLLM stub deleted — ADR-011.
- FastFlowLM as a curated, virtual, tag-addressed catalog on the AMD NPU — ADR-012.
- llmctl downloads FastFlowLM models from the Hub, not via `flm pull` — ADR-013.

## Coding standards

- Match the style of surrounding code (naming, comment density, idioms).
- `cargo build` must be **warning-free**; run `cargo fmt`. Use `#[allow(dead_code)]`
  with a note (e.g. "used in Phase N") only for genuinely forward-looking fields.
- Unit-test pure logic (resolution, validation, parsing). The TUI is smoke-tested
  via a PTY driver (`$CLAUDE_JOB_DIR/tmp/drive.py`); per-key delays matter, and
  escape sequences (Home/End/arrows) get split by the driver — rely on unit tests
  for those.
- Tests needing real binaries, hardware, or spawned processes are `#[ignore]`d
  with a reason and run via `cargo test -- --ignored --test-threads=1` (see the
  FastFlowLM discovery and launch-lifecycle tests in `src/runtime/flm.rs`).
  **Single-threaded is required:** building a `SessionManager` sets `SIGCHLD` to
  `SIG_IGN` process-wide, which makes any concurrent `Command::output()` fail to
  reap its child. The same ordering constraint is why `App::new` discovers
  runtimes before constructing the manager.
- Reading a subprocess's output at runtime must go through
  `session::supervisor::output`. The supervisor sets `SIGCHLD` to `SIG_IGN` so
  detached servers self-reap, which makes a plain `Command::output()` fail to
  `wait()` — silently, if the caller treats an error as "no output". The same
  trap bites `Command::spawn()`, which *panics* when exec fails because it reaps
  the failed child to read its errno; `DetachedSupervisor::spawn` wraps it in
  `supervisor::with_default_sigchld` for that reason. Anything in std that waits
  on a child belongs inside that helper. This applies to tests too — a
  single-threaded `--ignored` run inherits the disposition from whichever test
  built a `SessionManager` first.
- Logs go to a **file** under the state dir, never stderr (it corrupts the TUI).
- Keep `domain/` IO-free. Discovery/process/IO lives in `discovery/`, `runtime/`,
  `profiles/`, and `session/`.

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
