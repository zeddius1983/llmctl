# Roadmap

Living status of the build. Update this when phases complete or scope shifts
(see the context-compaction companion in [CLAUDE.md](../CLAUDE.md)).

## Status at a glance

| Phase | Title | Status |
|-------|-------|--------|
| 0 | TUI skeleton + Yazi navigation | ✅ Done |
| 1 | Runtime & GGUF model discovery | ✅ Done |
| 2 | Profiles & options | ✅ Done |
| 3 | Launch & sessions (MVP milestone) | ✅ Done |
| 4 | Log search & startup-failure classification | ⏳ Next |
| 5 | Search/filter & polish | ◻ Planned |

Branching: each remaining phase is built on its own `feature/<task>` branch.
When all planned phases are done, they merge into the release umbrella
**`feature/v0.1.0`**, which then merges to `main`. (Early `phase-*`/`docs`
branches predate this policy and are grandfathered.)

## Done

### Phase 0 — Skeleton
Cargo project (Rust 2024), XDG config + `Paths`, domain types, ratatui shell,
Yazi sliding three-column navigation (`hjkl`, `g/G`, drill/back), per-level
nerd-font icons, breadcrumb, help overlay, file-based tracing, vLLM stub runtime.

### Phase 1 — Discovery
GGUF header parser (arch, ctx length, file_type, chat-template); recursive model
scan of configured + well-known dirs (LM Studio, llama.cpp cache, HF hub,
`~/models`) with size/mtime cache and `F5` rescan; multi-shard dedup + name
cleanup + summed sizes; `mmproj` projector filtering; filename-first quant
labels; runtime discovery (`llama-server` path/version, cached `--help`);
two-line→three-line status bar with left-truncated path.

### Phase 2 — Profiles & options
Static option registry (12 options + an enum example); built-in templates;
model-scoped instance store with auto-save; resolution layering; option editing
(`e` text prompt with live validation; bool/enum cycle in place); inline adjust
(`+`/`-`/`[`/`]` by per-option step, clamped) and `Home`/`End` min/max;
model-aware `ctx-size` (max = model context length); profile CRUD (`a` create,
`r` rename custom, `D` duplicate, `d` delete custom / reset built-in, `f`
favorite); context-aware footer hotkeys; 10 unit tests.

### Phase 3 — Launch & sessions (MVP success milestone)
Command builder from resolved options (`session/command.rs`, bool flags emitted
only when on); `y` yank with a launch-command preview modal + OSC 52 clipboard
copy; `SessionSupervisor` trait + `DetachedSupervisor` (`setsid`, stdio→log file,
`SIGCHLD` auto-reap, group signalling) per ADR-005/007; `s` launch with auto
port-conflict resolution; Session Manager screen (`t`) with status glyphs,
PID/port/uptime and `/proc` CPU+memory; `/health` TCP probe promoting
Starting→Running; rediscover + prune `session-<id>.json` on startup; `x`/`K`/`R`
stop/kill/restart; `c` copy endpoint; tailing `L` log view; periodic poll-tick
refresh. 21 tests (incl. ignored real-process integration tests).

## Next

### Phase 4 — Log search & startup-failure classification
- [ ] Log view search / filtering (`L` already tails + scrolls)
- [ ] Startup-failure classification (port in use, model missing, OOM, GPU/Vulkan/
      CUDA init, unsupported arg) via a regex rule table → drives the `Crashed`/
      `Unknown` distinction and a failure banner
- [ ] Configurable stop timeout (SIGTERM → escalate to SIGKILL)
- [ ] Optional `--print-command` subcommand (headless dry-run)

### Phase 5 — Search/filter & polish
- [ ] Incremental `/` search + `n`/`N` in every pane
- [ ] Structured filters (`name:`, `quant:`, `size:>10GB`, `favorite:`, `recent:`,
      session `running`/`port:`)
- [ ] Favorites/recents surfacing; theming; startup doctor (binary, paths, GPU
      backend); VRAM/RAM pre-flight fit estimate

## Deferred / out of MVP scope

- Additional runtimes (vLLM, Ollama, LM Studio, SGLang, ExLlamaV2) — currently
  vLLM is a navigation-only stub.
- macOS / Windows support.
- Supervisor daemon / auto-restart-on-crash (see ADR-005).
- Chat mode (server mode only for now).
