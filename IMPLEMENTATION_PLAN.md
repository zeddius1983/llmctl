# llmctl — Implementation Plan

A keyboard-driven TUI for discovering, configuring, launching, and managing
local llama.cpp servers. Yazi-style five-pane navigation. See
[functional-requirements.md](./functional-requirements.md) for the product spec.

## Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Language / TUI | **Rust + ratatui + crossterm + tokio** | Same ecosystem as Yazi; single static binary; fast startup; strong async process management. |
| Platforms (MVP) | **Linux only** | `/proc` stats, `setsid`/pgid signals, native XDG. macOS/Windows deferred. |
| Process model | **Detached + rediscover, behind a `SessionSupervisor` trait** | Meets the spec's "rediscover on restart" with minimal scope. Trait lets a daemon / `systemd-run` backend drop in later without TUI changes. |
| Profile scoping | **Global read-only built-in templates that fork into model-scoped instances on first edit** | Reconciles "built-in profiles" with "scoped to runtime+model" and "editing auto-saves". |
| Option metadata | **Curated static option registry** (`assets/option_registry.toml`) | `llama-server --help` is unstable to parse; `--help` used only for Runtime preview + flag-existence validation. |
| GGUF metadata | **Parse the GGUF header KV section directly** (first few KB–MB) | Authoritative source for architecture, ctx length, quant, chat template; no full-file read, no shelling out. |

## Process model detail

- Spawn each `llama-server` via `setsid()` in its own process group so it
  survives TUI exit and can be signalled as a group.
- Persist `~/.local/state/llmctl/sessions/session-<id>.json`:
  `{id, pid, pgid, port, cmd, model, profile, log_path, started_at, start_token}`.
- On startup: scan sessions dir, verify PID alive **and** `/proc/<pid>/cmdline`
  matches our injected `start_token`; prune stale records.
- Crash vs stop: exited via our SIGTERM/SIGKILL or exit 0 = Stopped; any other
  exit / signal = Crashed.
- Health: poll `GET /health` to flip `◐ Starting → ● Running` deterministically.

```rust
trait SessionSupervisor {
    fn start(&self, spec: &LaunchSpec) -> Result<Session>;
    fn stop(&self, id: SessionId, timeout: Duration) -> Result<()>;
    fn kill(&self, id: SessionId) -> Result<()>;
    fn restart(&self, id: SessionId) -> Result<Session>;
    fn list(&self) -> Vec<Session>; // rediscovers + prunes
}
```

## Navigation model (Yazi miller-columns)

The five panes are **not** all populated at once. Like Yazi's
parent / current / preview columns, llmctl reveals one level ahead of focus:

- A pane is rendered with content only when `pane_index <= focus_index + 1`.
  The pane immediately right of focus is the **preview** of the hovered parent
  item; panes beyond that are hidden until you drill in.
- So: Models are visible once a Runtime is hovered; Profiles once a Model is
  hovered; Options once a Profile is hovered — mirroring "you don't see a
  directory's contents until you select it".
- Child lists are **derived from the parent selection**. Moving the cursor in a
  parent pane rebuilds and resets every descendant pane to the top (new parent →
  fresh subtree), exactly like hovering a different directory in Yazi.
- `l`/`Enter` shifts focus right (only if the preview has items); `h` shifts
  left. The Info pane (far right) always previews the focused pane's selection.
- A breadcrumb in the footer shows the committed path (Runtime ▸ Model ▸ …).

## Module layout

```
src/
  main.rs            arg parse, init, run TUI or subcommand
  app/               app state, event loop, focus/navigation, keymap
  ui/panes/          runtime, model, profile, options, info
  ui/                sessions, logs, help, search
  domain/            pure types: runtime, model, profile, option_spec, launch
  discovery/         models scan+cache, gguf header parser, runtime discovery
  session/           supervisor trait + DetachedSupervisor, store, proc, health
  logs/              log file mgmt, tail, search, error rules
  config/            XDG load, config.toml, defaults
  profiles/          built-in templates + instance store
assets/option_registry.toml
```

Crates: ratatui, crossterm, tokio, serde/toml/serde_json, nix, directories,
notify, regex, anyhow/thiserror, tracing.

## Phases

- **Phase 0 — Skeleton.** Cargo project, XDG config load + defaults, domain
  types, ratatui 5-pane shell, focus/navigation (`hjkl`, `g/G`), `q`/`?`.
  Static stubs prove layout + key routing.
- **Phase 1 — Discovery.** GGUF header parser; recursive model scan + cache +
  `F5`; runtime discovery (path/version/cached `--help`). Runtime/Model panes
  show real data with Info previews.
- **Phase 2 — Profiles & Options.** Global templates + model-scoped instance
  store (auto-save); option registry; Options edit (`e`); profile mgmt
  (`a r D d` + favorite); Profile/Option previews.
- **Phase 3 — Launch & Sessions.** Command builder (+ `y` yank / dry-run);
  `DetachedSupervisor`; Session Manager (`t`) with status, metadata, `/proc`
  stats; rediscover+prune on startup; health poll. **MVP milestone.**
- **Phase 4 — Process control & logs.** `s/x/R/K`; session detail view; log
  view (`L`): tail/search/copy; error-rule classification + failure surfacing.
- **Phase 5 — Search/filter & polish.** Incremental `/` + `n/N`; structured
  filters; favorites/recents; help overlay; theming; startup doctor.

## Enhancements (folded into phases)

- `y` yank / `--print-command` non-TUI subcommand (Phase 3/5).
- `/health` polling for deterministic Starting→Running (Phase 3).
- VRAM/RAM pre-flight fit estimate in previews (Phase 2/3).
- Startup doctor: binary found, paths exist, GPU backend present (Phase 5).
- Log error classification rule table mapping the 8 spec failure modes (Phase 4).
- Auto port-conflict resolution (next free port) (Phase 3).
