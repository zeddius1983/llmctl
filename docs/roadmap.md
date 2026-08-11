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
| 8 | Runtime backends + FastFlowLM (AMD NPU) | ✅ Done |

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

**v0.4.0 — runtime backends and FastFlowLM** — replaces the string-literal
runtime dispatch with a `RuntimeBackend` trait (ADR-011), deletes the vLLM
navigation stub, and adds FastFlowLM as a real second runtime running models on
an AMD XDNA2 NPU (ADR-012): curated `flm list` catalog grouped by capability
label, NPU-specific option set and templates, `flm serve` sessions with
`/v1/models` readiness, resumable llmctl-owned downloads on `d` (ADR-013),
`flm run` chat, and `flm bench` benchmarking. The NPU's single hardware context
is modelled as `RuntimeBackend::single_session`. See
[release notes](release-notes-v0.4.0.md).

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

### Phase 8 — Runtime backends and FastFlowLM
Runtime-specific behavior moved behind a `RuntimeBackend` trait in
`src/runtime/` (ADR-011): binary discovery, option schema, templates, model
enumeration, command/chat/benchmark argv, readiness path, `/proc` identity, and
per-launch capability checks. The option tables moved from `profiles/` to their
backends, leaving `profiles/registry.rs` as the generic option model plus the
`OptionSchema` that binds a table to its CLI dialect. `LaunchRequest` now
carries a finished command and health path, so `SessionManager` no longer knows
any runtime's flags. The vLLM navigation stub and every `"llama.cpp"`/`"vLLM"`
string branch are gone; `Model` carries its runtime, which also fixed profiles
being attributed to llama.cpp regardless of origin.

FastFlowLM (`flm`) is the first backend added through that seam (ADR-012),
running models on an AMD XDNA2 NPU. Its curated `flm list --json` catalog covers
installed and available models in one call and is grouped by capability label
(`reasoning`, `vision`, `tool-calling`, `audio`, `embeddings`) with `chat` and
`installed` groups; because labels overlap, identity is the tag rather than the
tree position. NPU-specific options (`--ctx-len`, `--pmode`, `--prefill-chunk-len`,
`--q-len`, `--socket`, `--preemption`, …) and templates including Low Power;
`flm serve` sessions with `/v1/models` readiness on port 52625; `flm run` chat
on `C`; `/` filtering the catalog in place and `s` switching between the
Categories and Flat arrangements. `flm
validate` gates launching and explains an unready NPU stack in the status line.
Sessions survive `flm` being a launcher wrapper (a distrobox entry point, for
instance) via the existing `/proc` re-acquisition. Because the XDNA driver
grants one hardware context at a time, `RuntimeBackend::single_session` lets a
runtime declare itself exclusive; a second FastFlowLM start (server or chat) is
refused with the name of the session holding the device, instead of spawning one
that dies during model load (ADR-012). For the same reason `R` (restart) is now
two-phase: it signals the old process, shows the session as `Restarting`, and
spawns the replacement from the poll loop once the old one has actually exited
(escalating to SIGKILL after five seconds) — so the replacement inherits a free
port and a free device rather than racing the process it replaced.

The `flm list --json` catalog is memoized on the backend, since it costs ~150 ms
through the launcher script. `CatalogCtx.reload` marks the callers that need
fresh data (`F5`, and a finished download); switching arrangement does not, and
went from ~155 ms to ~60 µs (ADR-012).

## In progress

### dFlash speculative decoding and multi-GPU options (`feature/dflash-speculation`)
`--spec-type` gains llama.cpp's `draft-dflash` and `draft-dspark` variants.
A `dflash-*.gguf` drafter is discovered as a companion of the model published
beside it — by directory locally (single model family only, as for a generic
projector) and repository-wide on Hugging Face, so every quantization of e.g.
`unsloth/Muse-Glimmer-30B-GGUF` pairs the one `dflash-kquant.gguf`. It is hidden
as a standalone artifact, downloaded with the model it drafts for, and passed
through `--spec-draft-model`; a cached drafter makes `draft-dflash` the
model-aware `spec-type` default. Because `--spec-draft-hf` accepts only a
`repo[:quant]` selector, an unqualified drafter must be downloaded before use —
`launch_blocker` says so instead of letting llama-server fail (ADR-014).
`spec-draft-n-max` defaults to the drafter's `dflash.block_size` (parsed from
the companion header) instead of llama.cpp's 3 — worth ~12 t/s on a 30B target
(see ADR-014 for the measurements).
New llama.cpp options: `split-mode` (`-sm`), `tensor-split` (`-ts`),
`parallel` (`-np`), `sleep-idle-seconds`, and `load-mode` (`-lm`) replacing the
deprecated `--no-mmap`, with `mmap: off` profiles migrated via the new
`RuntimeBackend::legacy_value` hook.

Measured on the maintainer's Radeon 8060S (llama.cpp b10353): dFlash **aborts on
the Vulkan backend** — `pre-allocated tensor (output.weight) in a buffer
(Vulkan0) that cannot run the operation (NONE)`, with or without
`--spec-draft-ngl 0` — but works on ROCm. Undrafted throughput is the same on
both backends, so the 2.5x is dFlash, not the backend. Not an llmctl bug;
noted here because the failure looks like one.

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

### FastFlowLM follow-ups
- [ ] `DownloadRecord::Directory` variant, replacing the sentinel `complete_file`
      used to track `flm serve`'s own native downloads (ADR-013)
- [ ] `flm remove` for installed models (the CLI supports it; no llmctl binding yet)
- [ ] Surface `think` / `think_toggleable` from the catalog as a profile option
- [x] `flm bench` — it was a hidden subcommand of v0.9.45 all along, not a
      missing one; `b` now benchmarks the selected FastFlowLM model (ADR-012)
- [ ] Per-model gating of `--asr` / `--embed` / `--img-pre-resize` by the
      catalog's `asr` / `vlm` flags, rather than offering them for every model

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

- Additional runtimes (vLLM, Ollama, LM Studio, SGLang, ExLlamaV2). The
  `RuntimeBackend` trait (ADR-011) is the extension point: add a module under
  `src/runtime/` and one entry in `runtime::discover`.
- macOS / Windows support.
- Supervisor daemon / auto-restart-on-crash (see ADR-005).
- Chat mode (server mode only for now).
