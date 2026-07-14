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
| 7 | Online Hugging Face catalog | ✅ Done |

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

**v0.3.0 — online Hugging Face catalog** — adds a lazy virtual
`online ▸ huggingface` source, Trending/Most likes/Most downloads views,
scoped Hub search, remote model profiles and llama.cpp-native launch, plus
concurrent resumable downloads that survive restart. See
[release notes](release-notes-v0.3.0.md).

**v0.3.1 — MTP and multimodal companions** — detects integrated and sidecar
MTP models, launches them with model-aware speculative-decoding defaults, pairs
multimodal projectors with compatible base models, and preserves companion
relationships across Hugging Face discovery, downloads, and cached launches.
See [release notes](release-notes-v0.3.1.md).

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

### Phase 7 — Online Hugging Face catalog
Virtual `online ▸ huggingface` hierarchy with cached 30-model Trending,
Most likes, and Most downloads views across text and multimodal pipelines; lazy
repository file/metadata fetches; debounced `/` Hub search; split-shard
grouping; remote profile identity; `HF_TOKEN`-safe `--hf-repo`/`--hf-file`
launch; clean-layout `F5`; and automatic linking to the standard Hugging Face
cache after download. Sessions track known LFS blobs in that cache and display
`Downloading (N%)` before the model-loading `Starting` phase. Uncached GGUF
artifacts can also be downloaded directly with `d`, with resumable aggregate
shard progress displayed as concurrent jobs in a Downloads pane below Sessions;
selected downloads support cancellation and resume. Incomplete download jobs
survive restart as explicitly resumable `Interrupted` rows.

### v0.3.1 — Local MTP discovery and launch
Integrated MTP heads are detected from GGUF `nextn_predict_layers` metadata,
with an MTP filename-token fallback for older converters. Officially named
`mtp-*.gguf` sidecars are hidden as standalone models and paired with their
same-directory base GGUF, including sidecar names that omit the base artifact's
quantization suffix. Paired and integrated models default `spec-type` to
`draft-mtp`; local llama-server and llama-cli commands add
`--spec-draft-model` for the sidecar form. The managed manifest and model status
preserve and display the discovered relationship.

### v0.3.1 — GGUF companions and online discovery follow-up
Local and online `mmproj-*.gguf` files are hidden as auxiliary projector
artifacts and associated with compatible base models. Online `mtp-*` files are
likewise paired instead of exposed as standalone models; root publisher aliases
win over nested precision variants. Direct downloads include selected
companions, native Hub launches use llama.cpp auto-discovery, and cached/local
launches pass explicit companion paths. The default Hub repository page was
raised from 20 to 30 models; pagination remains deferred because Hub-wide search
already covers models outside the initial page.

## Next (post-v0.3.1)

### Online Hugging Face follow-ups
- [ ] Recent sorting and size/quantization filters
- [ ] Optional per-model MTP/projector precision selector; discovery currently
      follows the publisher's root default and deterministic precision fallback

### Diffusion model support
- [ ] Discover `llama-diffusion-cli` beside `llama-server` and on `$PATH`,
      including its version/help and supported launch flags.
- [ ] Detect diffusion GGUF architectures (initially DiffusionGemma) and expose
      them as launchable only when a compatible `llama-diffusion-cli` is found;
      keep them out of the regular `llama-server` launch path.
- [ ] Add a foreground diffusion chat workflow, suspending/restoring the TUI as
      for `llama-cli`, with profile options for output length, GPU offload,
      entropy-bounded sampling, prompt KV cache, and optional live canvas view.
- [ ] Defer detached sessions, health checks, and OpenAI-compatible endpoints
      until the upstream diffusion runtime provides a stable server interface.

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
