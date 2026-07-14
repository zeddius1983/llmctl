# Architecture Decision Records

Each ADR captures a decision, its context, and consequences. Status is one of
Proposed / Accepted / Superseded.

---

## ADR-001: Rust + ratatui for the TUI

**Status:** Accepted

**Context:** We needed a language/TUI stack for a keyboard-driven manager in the
style of Yazi. Candidates: Rust + ratatui, Go + Bubble Tea, Python + Textual.

**Decision:** Use **Rust + ratatui + crossterm** (with tokio planned for async
process management). Yazi itself is Rust; this gives a single static binary,
fast startup, strong TUI libraries, and good async process control.

> Note: an early template draft labelled this "Bubble Tea instead of Ratatui".
> That was a placeholder; the actual decision is the reverse — ratatui was
> chosen and Bubble Tea/Textual were rejected.

**Consequences:** More upfront code than Bubble Tea/Textual and slower iteration
than Python, but the best fit for performance, distribution (one binary), and
long-term process-management needs. GGUF header parsing is done by hand.

---

## ADR-002: Profiles scoped per runtime + model

**Status:** Accepted

**Context:** The spec lists built-in profiles (Default/Chat/Coding/…) *and* says
profiles are scoped to runtime+model *and* that editing options auto-saves. Taken
literally these conflict: shared built-ins can't be mutated per model.

**Decision:** Built-in profiles are **global, read-only templates** (defined in
`templates.rs`). Editing an option (or favoriting) for a given model **forks a
model-scoped instance** keyed by `(runtime, model, profile)`, persisted to
`profiles.json` and auto-saved. Resolution layers: instance override → template
override → config default → registry default.

**Consequences:** Clean separation between shared defaults and per-model tweaks.
Built-ins can be used, forked, favorited, and duplicated but not renamed or
deleted; `d` on a built-in *resets* (drops the model-scoped instance) rather than
deleting. Custom profiles (`a`/`D`) support full rename/delete.

---

## ADR-003: GGUF / llama.cpp only in the MVP

**Status:** Accepted

**Context:** Many runtimes and formats exist (vLLM, Ollama, LM Studio, SGLang,
ExLlamaV2; GGUF, safetensors, …). Supporting all upfront would balloon scope.

**Decision:** The MVP supports **only llama.cpp and GGUF models**, on **Linux**.
A vLLM runtime appears in the UI as a **stub** (with placeholder models) purely
to exercise multi-runtime navigation; it is not launchable.

**Consequences:** Discovery, the option registry, and templates are specialized
to llama-server. Abstractions (runtime list, `SessionSupervisor` trait) leave
room to add runtimes later without rewriting the UI. macOS/Windows deferred.

---

## ADR-004: Yazi sliding three-column navigation

**Status:** Accepted (supersedes the initial five-fixed-pane sketch)

**Context:** The requirements illustrate five panes (Runtime/Model/Profile/
Options/Info). An initial implementation rendered all five populated at once,
which did not feel like Yazi.

**Decision:** Render a **sliding three-column miller view** (Parent | Current |
Preview) over `root ▸ Runtime ▸ Model ▸ Profile ▸ Options`. The Preview column
shows the hovered item's children; at the Options leaf it becomes the option
detail/editor (absorbing the spec's "Info" pane). All status sits in a
three-line footer (path, metadata, hotkeys).

**Consequences:** Matches the file-manager metaphor exactly; you only ever see
one level ahead. Child lists are derived from and reset by parent selection.

---

## ADR-005: Detached processes + rediscovery (not a daemon)

**Status:** Accepted (implemented in Phase 3)

**Context:** Sessions must survive TUI restarts ("never lose visibility of a
running server"). Options: a long-lived supervisor daemon, or detached child
processes that are rediscovered.

**Decision:** Ship a **`DetachedSupervisor`** behind a `SessionSupervisor`
trait: spawn `llama-server` via `setsid()` in its own process group, persist
`session-<id>.json`, and rediscover live sessions on startup by validating the
PID and `/proc/<pid>/cmdline` (pruning stale records). A daemon or
`systemd-run --user --scope` backend can implement the same trait later.

**Consequences:** Far less scope than a daemon (no second binary, no IPC
protocol) while meeting the persistence requirement. Trade-off: no built-in
auto-restart-on-crash — deferred to a future supervisor backend.

---

## ADR-006: Static option registry; filename-first quantization

**Status:** Accepted

**Context:** The Options pane needs authoritative metadata (default, range, CLI
flag, description) and validation. `llama-server --help` is unstable to parse.
GGUF `general.file_type` is often coarse/wrong for modern quants (e.g. Unsloth
`Q4_K_XL`, `MXFP4`).

**Decision:** Maintain a **static option registry** in `registry.rs` as the
source of truth; use `--help` only to display the runtime and validate flag
existence (future). For quantization, **prefer the filename label**, falling
back to the header's `file_type`.

**Consequences:** Predictable option metadata and validation independent of
llama.cpp version. Quant labels match what users downloaded. New options are a
one-line registry addition.

---

## ADR-007: Synchronous poll-tick + `libc` for sessions (not tokio/nix)

**Status:** Accepted (supersedes the Phase 3 plan to add tokio + nix)

**Context:** Phase 3 (launch & sessions) needs to spawn detached servers, signal
them, sample `/proc`, poll `/health`, and refresh the UI periodically. The
original plan listed **tokio** (async) and **nix** (`setsid`/signals). The user
asked to keep things as simple as possible.

**Decision:** Keep the existing **synchronous** draw/input loop and add a
periodic tick driven by `crossterm::event::poll` with a short timeout (no async
runtime). Use **`libc`** directly for the few OS primitives needed:
`setsid()` (in a `pre_exec` hook), `kill(-pgid, …)`, `sysconf` (page size / CPU
count), and `signal(SIGCHLD, SIG_IGN)` so detached children are auto-reaped.
`/proc` and `/health` are read with std file/TCP I/O — no HTTP-client crate
(a tiny `GET /health` over `TcpStream`), and clipboard yank uses the OSC 52
terminal escape (with a hand-rolled base64) rather than a clipboard dependency.

**Consequences:** One small dependency (`libc`) instead of two large subsystems,
and no async rewrite of the event loop. The tick cadence (≈1 s) bounds how
quickly status/resource readings update — fine for a manager view. A future
runtime with genuinely concurrent needs could still adopt tokio behind the
`SessionSupervisor` trait without disturbing the UI.

---

## ADR-008: Process control folded into Phase 3 (not deferred to Phase 4)

**Status:** Accepted (adjusts the roadmap split)

**Context:** The roadmap originally put the `SessionSupervisor`/launch machinery
in Phase 3 but the `s`/`x`/`R`/`K` keybindings and the log view in Phase 4. In
practice a supervisor with no way to start, stop, or inspect a session is not
demonstrable, and the MVP success criteria require launching, monitoring logs,
and stopping/restarting in one flow.

**Decision:** Ship the full launch→manage→stop lifecycle in Phase 3: command
builder + `y` yank, `s` launch, the Session Manager screen (status/PID/port/
uptime/CPU/memory), rediscovery + prune, `/health` promotion, port-conflict
resolution, `x`/`K`/`R` stop/kill/restart, `c` copy endpoint, and a tailing
`L` log view. Phase 4 is narrowed to richer log search and startup-failure
classification.

**Consequences:** Phase 3 is the working MVP milestone. Startup-failure
classification (port-in-use, OOM, GPU/Vulkan/CUDA init, …) and log search remain
for Phase 4; the `Unknown` session state is reserved for that richer
classification.

---

## ADR-009: Managed model catalog with source-aware identity

**Status:** Accepted

**Context:** A flat list keyed and displayed by GGUF filename is ambiguous when
the same artifact name exists under multiple providers or model stores. Profiles
were also persisted in one `profiles.json`, keyed by an absolute model path,
which made the user-visible hierarchy and the persistence identity diverge.
Users need a Yazi-style source/provider/repository/artifact hierarchy, global
model search, and support for arbitrary configured model directories.

**Decision:** Discovery normalizes models into a managed physical catalog under
`~/.config/llmctl/models`. Known LM Studio and Hugging Face layouts receive
source-specific parsing; arbitrary configured sources preserve their relative
directory layout as a best-effort fallback. Each artifact leaf contains a
`model.gguf` symlink, a generated hidden `.llmctl.yml` identity/ownership
manifest, and a `profiles/` directory containing YAML profile instances. The
TUI mirrors this variable-depth catalog in its Miller columns. Search indexes
catalog leaves and jumps back to the regular hierarchy.

The catalog is derived from discovery but is the stable user-visible identity
layer. Launches continue to use the original path recorded in the manifest,
which avoids split-GGUF sibling lookup problems. Generated entries are only
reconciled when marked by an llmctl manifest; user profile data is never removed
merely because a source is temporarily unavailable.

Auxiliary GGUF sidecars are attributes of that identity rather than additional
catalog leaves. In particular, `mtp-<base filename>.gguf` is paired with its
same-directory base model and recorded in the base manifest. The sidecar stem
may omit a quantization suffix present on the base artifact. Integrated MTP is
identified from GGUF metadata (with a filename fallback). Both forms make
`draft-mtp` the model-aware profile default; only the sidecar form needs
llama.cpp's `--spec-draft-model` argument.

`mmproj-*.gguf` files follow the same companion rule. A generic projector is
attached locally only when its directory contains one unambiguous base-model
family, preventing a flat mixed-model directory from receiving an unrelated
projector. Local server and chat launches pass the selected projector through
`--mmproj`.

Catalog/profile writes are change-aware and profile mutations persist only the
affected YAML file. If a catalog leaf cannot be created or written, that
profile remains in the legacy JSON fallback until YAML persistence succeeds.
Hugging Face snapshot selection prefers `refs/main`, then uses a deterministic
mtime/revision/path ordering. Search results are cached per query, and selecting
a GGUF result atomically switches to the compatible llama.cpp runtime and tree
route.

On first run, llmctl creates a readable `config.toml` that explicitly lists the
four standard sources (llama.cpp cache, Hugging Face, LM Studio, and
`~/models`). A `config.yaml` from the former implementation is ignored but never
deleted automatically because it may contain model presets worth migrating.

**Consequences:** Models with identical filenames remain distinguishable by
source and provider, profiles live beside their model identity, and custom
folders work without requiring a prescribed layout. Discovery now needs source
descriptors, catalog reconciliation, collision-safe path normalization, legacy
profile migration, and variable-depth browser state. The catalog contains
absolute source paths and is therefore local machine state despite residing in
the XDG configuration directory.

---

## ADR-010: Hugging Face as a lazy virtual catalog; llama.cpp owns downloads

**Status:** Accepted

**Context:** Online GGUF models should participate in the existing Yazi-style
hierarchy and profile workflow. A separate screen would split that experience,
while an llmctl downloader would duplicate revisioned cache, shard, resume,
authentication, and projector behavior already handled by llama.cpp.

**Decision:** Add `online ▸ huggingface` as a virtual source below llama.cpp.
Selecting it fetches 30 trending compatible repositories; selecting a
repository lazily fetches GGUF files and metadata. Background threads perform
blocking HTTPS and return results to the synchronous event loop. Metadata,
stable remote identity, and profiles live under the managed catalog. Launch
uses `--hf-repo` plus `--hf-file`, inheriting `HF_TOKEN` only from the
environment. Once cached, the same leaf links to the downloaded file and
launches by local path. `F5` refreshes the current online scope.

**Consequences:** Online models reuse Model → Profile → Options and llama.cpp's
cache behavior. The domain model carries explicit remote identity because an
empty local path no longer necessarily means a directory. Online `/` searches
the Hub after a short debounce, keeps results transient, and promotes only the
repository selected with Enter into the cached catalog. Local `/` searches
recurse only below the current catalog directory, so remote and unrelated local
sources never leak into the results. Richer filters, structured progress, and
download-only remain follow-ups.

Repository IDs are presented as flat `provider/repository` rows, with likes and
download counts visible on each row. Online search is Hub-wide from the
repository list and artifact-local after entering a repository.

Compatibility filtering uses the Hub's `gguf` and `llama.cpp` facets without a
pipeline constraint. A `text-generation` constraint incorrectly excludes
llama.cpp-compatible multimodal repositories classified as `image-text-to-text`
or `any-to-any` (for example, current Gemma 4 GGUF releases).

The online repository pane exposes three views: Trending (`trendingScore`),
Most likes (`likes`), and Most downloads (`downloads`), cycled with `s`. The GGUF
files pane uses the same `Model` title as local repositories.
Switching views or pressing online `F5` cancels the logical generation, removes
generated online metadata and symlinks, and fetches a clean first page. Profile
YAML and actual Hugging Face cache files are user/model data and remain intact.

Online repository parsing classifies `mtp-*` and `mmproj-*` GGUFs as companion
artifacts and hides them as standalone model leaves. A root MTP publisher alias
is preferred over nested precision variants; projector selection prefers an
unqualified publisher default, then BF16/F16 and smaller quantizations. Direct
downloads materialize the base and selected companions in the standard Hub
cache. Native `-hf` launches use llama.cpp's automatic root-MTP/projector
discovery, with `--spec-draft-hf` reserved for repositories that expose only a
nested MTP quant. Once cached, launches use explicit `--spec-draft-model` and
`--mmproj` paths.

An uncached artifact can also be downloaded without launching a server by
pressing `d`. llmctl streams every GGUF shard into the standard Hugging Face
blob and snapshot cache. Multiple transfers can run concurrently as peers of
server processes. Sessions and Downloads occupy a 70/30 vertical split in the
left jobs column and use one continuous up/down selection. Each job owns a
cancellation token; cancelled partial files remain resumable with `R` or
another `d`. This keeps download-only files compatible with llama.cpp and
other Hub-cache consumers. A minimal per-job JSON record lives under the
managed catalogue's `online/huggingface/.downloads` directory. Refresh and sort
cleanup explicitly skip that directory. On restart, llmctl reconstructs byte
progress from the Hub blobs and presents the job as `Interrupted`; it does not
resume network activity until the user presses `R` or selects the model with
`d`. Completed or explicitly removed jobs delete their record.
