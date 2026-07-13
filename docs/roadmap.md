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
| 4 | Log search & startup-failure classification | ◻ Post-v0.1.0 |
| 5 | Search/filter & polish | ◻ Post-v0.1.0 |
| 6 | Source-aware model catalog | ✅ Done |
| 7 | Hugging Face hub browser | ✅ Done (`feature/hf-browse`, unreleased) |

**v0.1.0 released** — Phases 0–3 (the MVP), plus extra launch options
(`--no-mmap`, `--cache-type-k`/`-v`, speculative decoding) and a README, were
merged via the `feature/v0.1.0` umbrella and tagged `v0.1.0` on `main`. Phases 4
and 5 are deferred to a future release; the roadmap will be revisited then.

**v0.1.1 released** — option defaults & template controls (see the Done section
below), tagged `v0.1.1` on `main`. The release workflow now creates the GitHub
Release itself on tag push and attaches the prebuilt Linux binaries.

**v0.2.0 — source-aware model catalog** — replaces the flat filename list with
a physical source/provider/repository/artifact tree, moves profiles beside each
model as YAML, adds global model search, and generates an explicit standard-source
configuration on first run. See [release notes](release-notes-v0.2.0.md).

**v0.2.1 — device selection and benchmarking** — adds profile-level llama.cpp
device selection populated by `llama-server --list-devices`, plus optional
`llama-bench` discovery and the `b` benchmark shortcut. See
[release notes](release-notes-v0.2.1.md).

Branching: each remaining phase is built on its own `feature/<task>` branch.
When a batch is ready to ship, the feature branches merge into a release umbrella
(e.g. **`feature/v0.1.0`**), which then merges to `main` and is tagged. (Early
`phase-*`/`docs` branches predate this policy and are grandfathered.)

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

### v0.1.0 release polish
Extra `llama-server` launch options: `mmap` (emits the bare `--no-mmap` flag when
off, for ROCm/AMD), KV `--cache-type-k`/`--cache-type-v` (enum with an in-band
`default` that omits the flag), and speculative decoding (`--spec-type`,
`--spec-draft-n-max`, `--spec-draft-n-min`, available for all models). Added a
top-level `README.md`.

### v0.1.1 — option defaults & template controls
The `default` omit sentinel extended to `ctx-size` and all sampling params
(`temperature`, `top-p`, `top-k`, `min-p`, `repeat-penalty`) — at `default` the
flag is dropped and llama.cpp's own default applies; new profiles start sampling
params there. `ctx-size` still starts at the ctx/8 heuristic (its `default` =
the model's full trained context); `host`/`port` stay always-emitted (llmctl
needs the concrete endpoint). New options: `reasoning-effort` (delivered as
`--chat-template-kwargs '{"reasoning_effort":…}'`), `chat-template` (enum of the
54 built-in template names), `jinja` (bare `--no-jinja` when off). Editing: `d`
resets an option to its resolved default; `Home`/`End` are pure min/max; `Enter`
edits in Options; enums with >8 variants open a filterable selector popup
instead of cycling. Bugfix: the base snapshot that seeds a profile instance on
first edit/favorite/create is now model-aware, so materializing no longer reset
unedited options (ctx-size silently fell from the ctx/8 default back to the
global 4096).

### Phase 6 — Source-aware model catalog
Managed `~/.config/llmctl/models` tree with ownership manifests and model
symlinks; LM Studio and Hugging Face parsing plus arbitrary configured-source
fallbacks; variable-depth Miller navigation; per-model YAML profiles with
legacy JSON migration and write-failure fallback; incremental global model
search with atomic jump-to-result. Prefix collisions, Hugging Face snapshot
selection, and catalog/profile write amplification are covered by regression
tests. First run creates an editable `config.toml` with the four standard model
sources while retaining any legacy `config.yaml` as an ignored backup.

### v0.2.1 — device selection and benchmarking
Profile-level `device` selection discovers accelerator identifiers such as
`ROCm0` and `Vulkan0`, persists the choice, emits `llama-server --device`, and
supports selector or inline hotkey cycling. When `llama-bench` is installed,
`b` benchmarks the selected model in the foreground and forwards concrete
profile device and GPU-layer settings.

### Phase 7 — Hugging Face hub browser (`feature/hf-browse`)
The hub as a virtual `online/huggingface` folder in the normal Miller
navigation (no separate screen): trending repos pre-filtered to
llama.cpp-runnable GGUF models (`pipeline_tag=text-generation&library=gguf&
apps=llama.cpp`), repo folders listing artifacts with shard grouping, quant
labels, sizes, arch/ctx, gated flag. `/` is folder-scoped everywhere
(recursive under the current prefix only): local filter in local folders, live
online search at the hub folder, file filter inside a repo. Enter on a file
downloads in the background (blocking `ureq` on worker threads + mpsc, per
ADR-010) with row/header progress, `x` cancel, `.part` staging and HTTP-Range
resume; `HF_TOKEN` for gated repos. Downloads land in `models.download_dir`
(default `~/models/huggingface`, implicit `downloads` source if uncovered),
rescan quietly on completion, and Enter on a downloaded file jumps to its
catalog leaf. 10 new unit tests; PTY-verified end-to-end (real 88 MB download,
byte-exact, catalog reconciled; online search, filter, cancel/resume, jump).

## Next (post-v0.2.1)

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
