# Roadmap

Living status of the build. Update this when phases complete or scope shifts
(see the context-compaction companion in [CLAUDE.md](../CLAUDE.md)).

## Status at a glance

| Phase | Title | Status |
|-------|-------|--------|
| 0 | TUI skeleton + Yazi navigation | ✅ Done |
| 1 | Runtime & GGUF model discovery | ✅ Done |
| 2 | Profiles & options | ✅ Done |
| 3 | Launch & sessions (MVP milestone) | ⏳ Next |
| 4 | Process control & logs | ◻ Planned |
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

## Next

### Phase 3 — Launch & sessions (MVP success milestone)
- [ ] Command builder from resolved options → `llama-server -m … --ctx-size … --port …`
- [ ] `y` yank / dry-run command preview; optional `--print-command` subcommand
- [ ] `SessionSupervisor` trait + `DetachedSupervisor` (`setsid`, process group,
      log file, `session-<id>.json`) — see ADR-005
- [ ] Session Manager screen (`t`): status indicators, PID/port/uptime,
      `/proc` CPU+memory
- [ ] Rediscover + prune sessions on startup; `/health` poll for Starting→Running
- [ ] Auto port-conflict resolution (next free port)

## Planned

### Phase 4 — Process control & logs
- [ ] `s` start, `x` stop (SIGTERM + timeout), `R` restart (stored config),
      `K` kill (SIGKILL)
- [ ] Session detail view (resolved options, generated command, env, resources)
- [ ] Log view (`L`): tail, search, copy
- [ ] Startup-failure classification (port in use, model missing, OOM, GPU/Vulkan/
      CUDA init, unsupported arg) via a regex rule table

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
