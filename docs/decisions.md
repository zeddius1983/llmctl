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

**Status:** Accepted (supersedes the Phase 3 plan to add tokio + nix).
*Amended 2026-07-13:* avoiding an async runtime is no longer a hard
constraint — tokio is acceptable when it genuinely simplifies a feature
(see ADR-010, which still chose blocking threads).

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

## ADR-010: Online Hugging Face hub browser with blocking worker threads

**Status:** Accepted

**Context:** Users want to discover and fetch models without leaving the TUI:
search Hugging Face online, pick a quantization, download it, and launch it.
The hub's browse page filter (`pipeline_tag=text-generation&library=gguf&
apps=llama.cpp&sort=trending`) maps directly onto the REST API
(`sort=trendingScore`), so results can be pre-filtered to models llama.cpp can
actually run. ADR-007 chose a synchronous poll-tick loop with no async runtime;
the user has since relaxed that constraint — an async runtime is acceptable
when it simplifies development.

**Decision:** The hub is browsed **as a folder, not a screen**: a virtual
`online ▸ huggingface` directory beside the local sources in the normal Miller
columns (initially built as a dedicated `H` screen, reworked per user feedback
before merging). Its children are synthesized from hub state: the repo list
(trending, or the last committed search) and, inside a repo, its GGUF
artifacts as remote file leaves (`Model::remote` marks virtual nodes; remote
files are leaves but not launchable). Enter on a remote file downloads it, or
jumps to the local catalog leaf once it is on disk.

`/` search became **folder-scoped everywhere**: it searches recursively under
the current catalog prefix only — never parents. Locally that filters the
scanned models under the prefix; at the hub folder it queries the Hugging Face
API live per keystroke (epoch-guarded); inside a repo it filters that repo's
files. Committing an online search makes its results the folder listing.

Networking uses blocking `ureq` (rustls, no new runtime): even with tokio
permitted, the TUI loop stays synchronous, so worker `std::thread`s reporting
over an `mpsc` channel — drained once per loop turn — remain the simpler
bridge. Download progress is shared via `Arc<AtomicU64>` and rendered every
frame (row markers plus a header activity indicator); cancellation via
`Arc<AtomicBool>` checked between chunks.

Repo file listings (`/api/models/<id>?blobs=true`) are collapsed into logical
artifacts: `mmproj` projectors dropped, `-000NN-of-000NN` shard sets grouped
and size-summed, quant labels reused from discovery. Downloads stream into
`<file>.part` and are renamed into place only when complete, so the scanner
never sees partial GGUFs; interrupted/cancelled transfers resume with HTTP
`Range`. Files land in `models.download_dir` (default `~/models/huggingface`,
under the standard `~/models` source) as `<owner>/<repo>/<file>`; when the
configured directory is outside every scanned source an implicit `downloads`
source covers it. A finished download triggers the normal rescan, so the model
appears in the catalog with profiles and is launched like any local model —
no special "remote model" state beyond the virtual browse nodes. `HF_TOKEN`
(env) is forwarded for gated repos.

**Consequences:** One new dependency (`ureq`). Search/listing/download logic is
unit-tested from JSON fixtures; the API's response shape is normalized in one
place (`hub/api.rs`). Because completed downloads are plain files in a scanned
source, removal, re-download detection ("✓ downloaded"), and launching all
reuse existing machinery. Parallel multi-file downloads work naturally (one
thread each); a future switch to async (per the relaxed ADR-007) would only
change the transport, not the screen or event flow.
