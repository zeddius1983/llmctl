# Architecture

Component structure and application design for `llmctl`. See
[decisions.md](decisions.md) for the rationale behind these choices and
[requirements.md](requirements.md) for the product spec.

## Overview

`llmctl` is a single-binary Rust TUI built on ratatui + crossterm. It runs a
synchronous draw/input loop; since Phase 3 a short `event::poll` timeout adds a
periodic tick that refreshes live session status without an async runtime
(ADR-007). State lives in one `App` struct; rendering is a pure function of that
state.

```
            ┌──────────────┐   draw(&App)   ┌──────────────┐
            │   ui (render)│◀───────────────│              │
            └──────────────┘                │     App      │
            ┌──────────────┐   on_key(ev)   │  (app/)      │
 crossterm ─▶│  event loop  │───────────────▶│              │
            └──────────────┘                └──────┬───────┘
                                                   │ reads
              ┌────────────────────────────────────┼────────────────────┐
              ▼                    ▼                ▼                     ▼
         config/             discovery/         profiles/            domain/
       Config, Paths     gguf, models,      registry, templates,   pure types,
       (config.toml,     runtimes           store, resolution      helpers,
        XDG dirs)        (scan + cache)      (per runtime+model)    vLLM stubs
```

## Modules

- **`main.rs`** — resolves XDG `Paths`, ensures dirs, initializes file-based
  tracing, loads `Config`, constructs `App`, runs the ratatui loop.
- **`config/`** — `Config` (deserialized from `config.toml`; a documented
  standard-source file is generated on first run) and `Paths`
  (config/state/cache/log/sessions locations).
- **`domain/`** — IO-free types: `Runtime`, `Model`, `Profile`, `OptionItem`;
  helpers (`human_size`, `format_unix_date`); and `stubs` for the demo vLLM
  runtime/models.
- **`discovery/`**
  - `catalog.rs` — normalize known/custom source layouts and reconcile the
    managed directory tree, identity manifests, and model symlinks.
  - `gguf.rs` — minimal GGUF header reader. Parses only the KV metadata section
    (token arrays skipped via buffered skipping, never loaded) to extract
    architecture, context length, `general.file_type`, and chat-template presence.
  - `models.rs` — recursive scan of configured/default dirs; multi-shard dedup
    (first shard only, suffix stripped, sizes summed); projector (`mmproj`)
    filtering; filename-first quant detection; cache to `models.json` keyed by
    size+mtime; `F5` rescan.
  - `online.rs` — lazy `online ▸ huggingface` virtual source. Background HTTPS
    requests fetch trending repositories and repository GGUF details; cached
    metadata is exposed as flat `provider/repository` rows followed by GGUF
    artifacts, and profile leaves are materialized below the managed catalog.
    Downloaded files are detected in the standard Hugging Face cache. Online
    search is Hub-wide and transient from the repository list, persisting only
    the result selected with Enter; it is artifact-local after a repository is
    entered. View state maps Trending/Most likes/Most downloads to
    `trendingScore`/`likes`/`downloads`; switching view or online `F5`
    invalidates in-flight responses and rebuilds generated online metadata
    while preserving profiles and downloaded files. Independently identified
    download workers stream `d`-selected artifacts into the standard Hub
    blob/snapshot layout and report aggregate shard progress to `App` over a
    shared channel. Per-job cancellation tokens preserve partial blobs for a
    later `R`/`d` resume. Minimal job records are atomically persisted beneath
    `online/huggingface/.downloads`; startup restores unfinished jobs as
    `Interrupted` and recomputes progress from cached blob sizes. Catalogue
    cleanup preserves those records. The left jobs column stacks Sessions over Downloads
    with a 70/30 split; both panes map into one continuous selection index and
    share the right-hand Detail pane.
  - `runtimes.rs` — locate `llama-server` (explicit path or `$PATH`), capture
    `--version`, cache `--help`.
- **`profiles/`**
  - `registry.rs` — static `REGISTRY` of `OptionSpec`s (kind, default, range,
    step, CLI flag, description) plus `OptionKind` validate/adjust/extreme logic.
  - `templates.rs` — built-in global templates (Default/Chat/Coding/Long
    Context/Server) as option overrides.
  - `store.rs` — `ProfileStore`: model-scoped instances persisted as YAML in
    each catalog leaf; create/rename/delete/favorite/set-value, auto-saved.
  - `mod.rs` — resolution: `list_profiles`, `resolve_options`,
    `current_values`, `effective_kind` (model-aware ctx-size bound).
- **`session/`**
  - `command.rs` — pure launch-command builder (argv + shell-quoted display;
    bool flags emitted only when on, local model via `-m`, remote model via
    `--hf-repo` and `--hf-file`).
  - `supervisor.rs` — `SessionSupervisor` trait + `DetachedSupervisor` (`setsid`
    pre-exec, stdio→log file, `SIGCHLD` auto-reap, `kill(-pgid, …)`); plus the
    OSC 52 base64 helper used for clipboard yank.
  - `record.rs` — `SessionRecord` persisted as `session-<id>.json`; load/prune.
  - `proc.rs` — `/proc` liveness, cmdline match (PID-reuse guard), RSS, CPU%.
  - `health.rs` — minimal `/health` TCP probe; bindable-port check.
  - `mod.rs` — `SessionManager`: launch, rediscover + prune, refresh
    (status/resources), stop/kill/restart, port-conflict resolution.
- **`ui/`** — all rendering: the browser's three columns + footer, the Session
  Manager (list + detail), the log-tail view, the modal prompt/message overlays,
  and the help overlay.

## Navigation model (Yazi sliding three-column)

The UI shows **Parent | Current | Preview** over `root ▸ Runtime ▸ model
catalog… ▸ Model ▸ Profile ▸ Options`. The catalog has variable depth and
normally reads `source ▸ provider ▸ repository ▸ artifact`; arbitrary configured
directories preserve their relative hierarchy. Drilling in (`l`/`→`) slides
columns left; `h`/`←` slides right. The Preview column shows the hovered item's children; at the Options
leaf it becomes the option detail/editor.

| Current | Parent (left) | Current (middle) | Preview (right)   |
|---------|---------------|------------------|-------------------|
| Runtime | root (virtual)| runtimes         | models            |
| Catalog | previous level| directories      | children          |
| Model   | repository    | model artifacts  | profiles          |
| Profile | models        | profiles         | options (values)  |
| Options | profiles      | options          | option detail     |

Child lists are **derived from the parent selection**: moving the cursor in the
current column rebuilds/resets all descendant levels (new parent → fresh
subtree). The app tracks the current level in `App.focus`; `rebuild_below`
recomputes descendant lists.

## Option resolution

For a `(runtime, model, profile)`, each option's value is resolved by layering:

```
instance override (store) → template override → config default → registry default
```

`ctx-size` is model-aware: `effective_kind` sets its max to the model's trained
context length (from the GGUF header), and the resolved value is clamped down so
a default never exceeds what the model supports.

## Persistence & paths (XDG)

- `~/.config/llmctl/config.toml` — generated first-run config (model sources,
  runtime binary, default host/port); user editable.
- `~/.config/llmctl/config.yaml` — ignored legacy config from the former
  implementation; retained for manual preset migration/backup.
- `~/.cache/llmctl/models.json` — model scan cache; `llama-server.help.txt`.
- `~/.config/llmctl/models/` — managed source-aware tree; each model leaf has
  `.llmctl.yml`, `model.gguf`, and YAML files below `profiles/`.
- `~/.local/state/llmctl/profiles.json.bak` — backup made when migrating the
  former flat profile store (offline models remain in JSON until seen).
- `~/.local/state/llmctl/logs/` — app log + (Phase 3) per-session server logs.
- `~/.local/state/llmctl/sessions/` — (Phase 3) session metadata for rediscovery.

## Process management (Phase 3, implemented)

Lifecycle is hidden behind a `SessionSupervisor` trait. The MVP
`DetachedSupervisor` spawns `llama-server` via `setsid()` in its own session/
process group (survives TUI exit, ignores tty signals), redirects stdio to a
per-session log file, and ignores `SIGCHLD` so detached children are auto-reaped.
Each launch writes `session-<id>.json` (id, name, pid, host, port, full argv,
model/profile, log path, optional Hub download blobs, start time). On startup `SessionManager::rediscover`
keeps sessions whose PID is alive *and* whose `/proc/<pid>/cmdline` still
contains the model path (PID-reuse guard), deleting the JSON for the rest.

`SessionManager::refresh` (called on the ≈1 s tick) derives status —
`Downloading (N%)` while known Hub blobs are incomplete, `Starting` while the
model loads, then `Running` after `GET /health` returns 200; `Stopped` if the
user asked it to stop and the process is gone, else `Crashed` — and samples
`/proc` for RSS and CPU%. Launch resolves a bindable port (skipping ports held by other live
sessions) before spawning. A future `DaemonSupervisor` or `systemd-run` backend
can implement the same trait. See ADR-005 and ADR-007.

## Testing strategy

- **Unit tests** for pure logic: option resolution, validation, adjust/clamp/
  cycle, extremes, model-aware ctx-size, command building, session
  naming/uptime, port resolution, OSC 52 base64.
- **`#[ignore]` integration tests** in `session/` that spawn real processes
  (a `sleep`, and a fake `/health` server) to exercise the actual spawn →
  liveness → signal path and the full launch → Running → rediscover → stop
  lifecycle. Run with `cargo test -- --ignored`.
- **PTY smoke tests** for the TUI via a Python driver (per-key delays; the pty
  must be given a window size via `TIOCSWINSZ` or crossterm renders blank).
  Multi-byte escape sequences (Home/End/arrows) are split by the driver, so
  those bindings are verified by unit tests instead.
