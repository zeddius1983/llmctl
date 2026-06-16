# Architecture

Component structure and application design for `llmctl`. See
[decisions.md](decisions.md) for the rationale behind these choices and
[requirements.md](requirements.md) for the product spec.

## Overview

`llmctl` is a single-binary Rust TUI built on ratatui + crossterm. It runs a
synchronous draw/input loop (Phase 0–2; async via tokio arrives with process
management in Phase 3). State lives in one `App` struct; rendering is a pure
function of that state.

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
- **`config/`** — `Config` (deserialized from `config.toml`, all-defaults when
  absent) and `Paths` (config/state/cache/log/sessions locations).
- **`domain/`** — IO-free types: `Runtime`, `Model`, `Profile`, `OptionItem`;
  helpers (`human_size`, `format_unix_date`); and `stubs` for the demo vLLM
  runtime/models.
- **`discovery/`**
  - `gguf.rs` — minimal GGUF header reader. Parses only the KV metadata section
    (token arrays skipped via buffered skipping, never loaded) to extract
    architecture, context length, `general.file_type`, and chat-template presence.
  - `models.rs` — recursive scan of configured/default dirs; multi-shard dedup
    (first shard only, suffix stripped, sizes summed); projector (`mmproj`)
    filtering; filename-first quant detection; cache to `models.json` keyed by
    size+mtime; `F5` rescan.
  - `runtimes.rs` — locate `llama-server` (explicit path or `$PATH`), capture
    `--version`, cache `--help`.
- **`profiles/`**
  - `registry.rs` — static `REGISTRY` of `OptionSpec`s (kind, default, range,
    step, CLI flag, description) plus `OptionKind` validate/adjust/extreme logic.
  - `templates.rs` — built-in global templates (Default/Chat/Coding/Long
    Context/Server) as option overrides.
  - `store.rs` — `ProfileStore`: model-scoped instances persisted to
    `profiles.json`; create/rename/delete/favorite/set-value, auto-saved.
  - `mod.rs` — resolution: `list_profiles`, `resolve_options`,
    `current_values`, `effective_kind` (model-aware ctx-size bound).
- **`ui/`** — all rendering: the three columns, header breadcrumb, three-line
  footer (path / metadata / context hotkeys), the modal prompt, and help overlay.

## Navigation model (Yazi sliding three-column)

The UI shows **Parent | Current | Preview** over `root ▸ Runtime ▸ Model ▸
Profile ▸ Options`. Drilling in (`l`/`→`) slides columns left; `h`/`←` slides
right. The Preview column shows the hovered item's children; at the Options
leaf it becomes the option detail/editor.

| Current | Parent (left) | Current (middle) | Preview (right)   |
|---------|---------------|------------------|-------------------|
| Runtime | root (virtual)| runtimes         | models            |
| Model   | runtimes      | models           | profiles          |
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

- `~/.config/llmctl/config.toml` — user config (model paths, runtime binary,
  default host/port). Optional.
- `~/.cache/llmctl/models.json` — model scan cache; `llama-server.help.txt`.
- `~/.local/state/llmctl/profiles.json` — model-scoped profile instances.
- `~/.local/state/llmctl/logs/` — app log + (Phase 3) per-session server logs.
- `~/.local/state/llmctl/sessions/` — (Phase 3) session metadata for rediscovery.

## Process management (Phase 3, planned)

Lifecycle hidden behind a `SessionSupervisor` trait. The MVP `DetachedSupervisor`
spawns `llama-server` via `setsid()` in its own process group (survives TUI
exit), writes `session-<id>.json` (pid, pgid, port, cmd, model, profile, log,
start_token), and on startup rediscovers live sessions by validating the PID and
`/proc/<pid>/cmdline`, pruning stale records. A future `DaemonSupervisor` or
`systemd-run` backend can implement the same trait. See ADR-005.

## Testing strategy

- **Unit tests** for pure logic: option resolution, validation, adjust/clamp/
  cycle, extremes, model-aware ctx-size.
- **PTY smoke tests** for the TUI via a Python driver (per-key delays); note
  multi-byte escape sequences (Home/End/arrows) are split by the driver, so
  those bindings are verified by unit tests instead.
