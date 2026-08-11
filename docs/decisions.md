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

## ADR-011: Runtimes behind a `RuntimeBackend` trait

**Status:** Accepted (2026-07-26).

**Context:** llmctl shipped as a single-runtime tool wearing a multi-runtime
interface. The Runtime column existed, but its second row was `domain::stubs`,
a fabricated non-launchable "vLLM" node whose only job was to make the
navigation exercisable. Everything below that column — the option registry, the
command builder, the health probe, model discovery, the profile store — was one
hardcoded llama.cpp implementation, and `app/mod.rs` dispatched by comparing
`runtime.name` against the string literals `"llama.cpp"` and `"vLLM"` in about
a dozen places. Adding a real second runtime by extending that pattern would
have multiplied the string branches with no compiler help for the sites missed.

**Decision:** Introduce `src/runtime/` with a `RuntimeBackend` trait and hold
backends as `PaneList<Box<dyn RuntimeBackend>>`. The trait owns everything that
differs: binary discovery, the option schema, built-in templates, model
enumeration, model-aware defaults and clamping, command/chat/benchmark argv, the
readiness path, the `/proc` identity token, and per-launch capability checks.

Two supporting types make this work without threading `&dyn RuntimeBackend`
into deep call sites:

* `OptionSchema` (in `profiles/registry.rs`) bundles a runtime's `&'static
  [OptionSpec]` table with its CLI-encoding rules (`omit_token`, `is_flag`,
  `cli_value`) as function pointers. It is `Copy`, so resolution and command
  building pass it around freely. The option *tables* moved out of
  `profiles/registry.rs` and `profiles/templates.rs` into the backends; what
  remains in `profiles/` is the generic option model.
* `LaunchRequest` now carries a finished `Command` plus a `health_path`, built
  by the backend before it reaches `SessionManager`. The manager keeps port
  resolution — it patches the resolved `--port` into the finished argv, which
  every backend emits explicitly — but no longer knows anyone's flags.

`domain::stubs` and every `"vLLM"` / `"llama.cpp"` string branch are deleted.
`Model` gains a `runtime` field, which also fixes a latent bug: the profile
store previously attributed *every* loaded profile to `"llama.cpp"` regardless
of origin.

**Alternatives considered:** A `RuntimeKind` enum with exhaustive `match`
dispatch would have been a smaller change and would still have let the compiler
find every site needing a new arm, but it keeps per-runtime knowledge spread
across the modules that match on it. A minimal string branch mirroring the vLLM
stub was rejected outright: it scales worst and the compiler cannot check it.

**Consequences:** Adding a runtime means adding a module under `src/runtime/`
and one entry in `runtime::discover`, not another branch in the app. One place
still resolves a runtime by *name* rather than through the trait —
`runtime::templates_for`, used by the profile store, which reads runtime names
off disk long before any backend has been probed. Session records gained a
`health_path` field defaulting to `/health`, so records written by older
versions still rediscover correctly.

## ADR-012: FastFlowLM as a curated, virtual, tag-addressed catalog

**Status:** Accepted (2026-07-26).

**Context:** FastFlowLM (`flm`) runs models on an AMD XDNA2 NPU — hardware
llama.cpp cannot target — making it complementary rather than redundant. It
differs from llama.cpp in ways that resist the existing model plumbing: its
catalog is curated rather than scanned, its models are tags rather than file
paths, and it exposes no `/health`. The published documentation also disagrees
with the shipping CLI in several places, so this design was derived by probing
`flm` v0.9.45 directly.

**Decision:**

*Catalog.* `flm list --json` returns the entire catalog — installed and not — in
one call, with context length, quantization, disk footprint, and capability
labels. There is no filesystem scan and no separate "online" subtree; a
not-yet-downloaded model is fully browsable and directly launchable, because
`flm serve` fetches it on demand. Every `flm` invocation prints a
`[FLM] Fetching models from: …` banner before its JSON, which is stripped
before parsing.

*Tree shape.* Models split into `local` and `online` at the top, mirroring
llama.cpp's catalog, then group by capability label (`reasoning`, `vision`,
`tool-calling`, `audio`, `embeddings`) with a `chat` fallback for the unlabeled.
Labels overlap, so a model is emitted once per group it belongs to. This makes
the FastFlowLM tree **virtual**, unlike the local GGUF catalog that
`discovery::catalog::reconcile` materializes as real directories: identity must
not derive from `catalog_path`. `Model::profile_key` therefore returns
`flm:<tag>`, and the managed catalog leaf is keyed by tag alone, so a model
rendered under three labels still has exactly one set of profiles.
`Model::is_catalog_dir` additionally consults the new `flm` field, because a
not-installed model has an empty path and would otherwise read as a folder.

*Sessions.* The `--port` flag is always emitted explicitly. `flm`'s own default
is the sentinel `-1`, and llmctl needs a concrete port both for health checks
and to re-acquire the server process by command line. Readiness is `GET
/v1/models` returning 200; FastFlowLM has no `/health`. The process token
recorded for `/proc` matching is the tag, since that is what appears in argv.

*Exclusivity.* The XDNA driver grants one hardware context at a time. A second
`flm serve` (or `flm run`) spawns happily, gets as far as loading the model, and
dies with `DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22)`, leaving a
crashed session behind. `RuntimeBackend::single_session` lets a runtime declare
that, and llmctl refuses the second start with the name of the session already
holding the device. This is deliberately narrow: how many servers to run is
normally the user's call — two llama.cpp servers that overcommit VRAM still
start, and may be exactly what was wanted — so the guard is only for hardware
that admits one client and where the second launch *cannot* work. It gates
starting a model, not building its command, so `y` still previews and copies the
launch line while a session is up.

Restart is the same collision with the user on the right side of it, so it is
handled by waiting rather than refusing. `SessionManager::restart` signals the
old process and records a `PendingRestart`; `poll_restarts`, driven from the
input loop, spawns the replacement only once that process is actually gone,
escalating to SIGKILL after five seconds. The wait is deferred rather than a
blocking loop because llmctl has no async runtime (ADR-007) and a synchronous
wait would freeze the TUI for the whole of a large model's teardown. This also
fixes a smaller pre-existing race for every runtime: respawning while the old
server still held its socket could push a restart onto a different port.

*Catalog caching.* `flm list --json` is memoized on the backend. Measured at
~150 ms on the development machine, because `flm` is frequently a launcher
script and every call pays a container hop — and cycling the arrangement with
`s` used to re-read it purely to regroup identical entries. `CatalogCtx.reload`
is how a caller says it has reason to believe the catalog changed: the `F5`
refresh, and a download finishing. Rearranging does not set it, and drops from
~155 ms to ~60 µs. Materializing the managed profile directories moved into the
same cache-fill path, so that write happens once per catalog read rather than
once per regroup. The trade is that a model installed by another process is
invisible until `F5` — which is what `F5` is for.

*Downloads.* llmctl fetches the model's files from Hugging Face itself — see
the amendment below.

*Browsing.* `local`/`online` reuses llama.cpp's group names but not its Hugging
Face browser, and `discovery::online::is_online_path` tests only the first path
segment — so the runtime, not the path, decides which surface is active.
`RuntimeBackend::supports_online_browse` gates that: `/` filters FastFlowLM's
catalog in place rather than querying the Hub, and `s` does not offer Hub sort
orders for a catalog that is a fixed list rather than a ranked feed. In their
place `RuntimeBackend::catalog_views` lets a runtime offer arrangements of its
own catalog; FastFlowLM offers Categories (grouped by capability label) and Flat
(one row per model). An arrangement is a view only — identity stays the tag, so
profiles are unaffected by switching.

**Consequences:** FastFlowLM is listed even when absent or unusable, with the
reason (missing binary, or an NPU stack that `flm validate` reports as not
ready) surfaced in the status line rather than at launch time.

`flm bench` was originally recorded here as absent from v0.9.45, on the strength
of it not appearing in `flm --help`. That was wrong: it is a *hidden* subcommand
that the same v0.9.45 parses and runs. `bench_argv` now returns
`flm bench <tag> [--pmode …]` and `bench_path` points at `flm` itself, so `b`
works for FastFlowLM. Only the power mode carries over from the profile — the
benchmark drives its own per-stage context lengths and opens no socket. The
lesson generalizes: probe a runtime's CLI by invoking a subcommand, not by
trusting its help text to be complete.

A practical note that shaped the session design: `flm` may legitimately be a
*wrapper* rather than the server binary — on the development machine it is a
distrobox entry point, so the process llmctl spawns has `comm` `podman` and the
real server lives inside a container. This works without special-casing because
`session::proc::find_server` already re-acquires the real process by `comm` plus
argv match, and the container shares the host PID and network namespaces. Any
runtime fronted by a launcher script benefits from the same mechanism.

## ADR-013: llmctl downloads FastFlowLM models, not `flm pull`

**Status:** Accepted (2026-07-27). Amends the *Downloads* section of ADR-012.

**Context:** ADR-012 built downloading on `flm pull`, tracking progress by
watching the model directory grow. In use that proved unreliable: `flm pull`
cannot resume, and `flm` does not correctly recognize a partially-downloaded
model — an interrupted pull leaves a directory that reads as installed but is
not. A multi-gigabyte transfer with no resume and a failure mode that
misrepresents itself as success is not something to build a download UX on.

llmctl already had the right machinery for llama.cpp's Hugging Face downloads:
`Range` resume, cancellation, per-file size verification, and progress into the
Session Manager. FastFlowLM's models *are* Hugging Face repositories under
<https://huggingface.co/FastFlowLM>, so it applies directly.

**Decision:** llmctl performs the download itself.

The transfer core was extracted from `discovery::online`'s blob downloader into
`discovery::hf` and is now shared by both runtimes. `runtime::flm::download`
resolves per-file sizes from `GET /api/models/<repo>/tree/<revision>`, then
fetches each file the catalog lists into `~/.config/flm/models/<Repo>/`, writing
to a `<file>.llmctl-part` scratch name and renaming only once the file is
byte-complete.

That gives resume at two levels — a half-written file continues via `Range`, and
a file already present at its expected size is skipped on a later attempt — and,
because nothing appears under its real name until it is whole, `flm` can never
mistake a partial download for an installed model. The failure mode that
prompted this ADR is structurally impossible.

Three facts made this viable, all verified against `flm` v0.9.45 rather than
taken from documentation:

* Placing the files is sufficient. A directory populated by llmctl makes
  `flm list` report `installed: true` and `flm check` pass.
* `flm list`'s `files[]` is exactly a model directory's contents — confirmed
  against an installed model. The repository also holds a README and `.xclbin`
  NPU kernels, which are **not** part of it; they ship with `flm`.
* Revisions matter. Several models are pinned to a tag
  (`v0.9.22-faster-q4-1`), so `FlmModel` carries a `revision` and the downloader
  honors it. Fetching `main` for those would produce weights the installed `flm`
  cannot load. This is also why the `online` catalog is driven by `flm list`
  rather than by browsing the Hub organization: only `flm list` knows the right
  revision, the right file set, and `flm_min_version`.

Note that the repository *id* (`FastFlowLM/Qwen3-0.6B-NPU2`) and the directory
name `flm` stores it under (`Qwen3-0.6B-NPU2`) differ; conflating them yields a
404 from the Hub.

**Consequences:** `flm pull` is no longer used anywhere. Downloading needs
neither the `flm` binary nor a ready NPU — it is a plain Hugging Face fetch — so
a model can be staged on a machine whose NPU stack is not yet configured. No
resume record is persisted, because none is needed: the partial files on disk
*are* the resume state, which also means resume survives an llmctl restart.

Launching a model that is not downloaded still falls through to `flm serve`,
which fetches it natively — deliberately the same shape as llama.cpp's `-hf`
launch. `FlmBackend::launch_download` reports progress for that path too. Since
`flm` writes straight to final filenames there is no rename to observe, so the
tracking record points at a sentinel `complete_file` that never exists, keeping
progress tied to byte growth; the `Downloading` state ends when the health probe
reports ready. A `DownloadRecord::Directory` variant would express this more
honestly and is noted as a follow-up.

---

## ADR-014: dFlash drafters are directory-scoped companions, downloaded first

**Status:** Accepted

**Context:** llama.cpp's `--spec-type` gained `draft-dflash`, a drafter loaded
from a separate GGUF through the same `--spec-draft-model` slot MTP uses.
Publishers ship it beside the model it drafts for under an unqualified name —
`unsloth/Muse-Glimmer-30B-GGUF` carries one `dflash-kquant.gguf` for all
fourteen quantizations of the model. That breaks the two assumptions the MTP
pairing rests on: the companion filename names its base model, and llama.cpp can
find or fetch the companion itself from an `-hf` launch.

**Decision:** A dFlash drafter is a companion attribute of a base model, like an
MTP sidecar or a projector, recorded as `Model::dflash_path` (and
`RemoteModel::dflash_file`) rather than as its own catalog leaf. Because its
name identifies no base model, it is paired the way a generic projector is: by
directory, and only when that directory publishes a single model family. In a
Hugging Face repository the pairing is repository-wide, so every quantization
gets the same drafter. Where several variants exist, the publisher's unqualified
file wins, then the most compact quantization — a drafter is run for speed.

`spec-type` becomes `draft-dflash` by model-aware default when a drafter is on
disk **and** the configured `llama-server` advertises that spec type — a
model-aware default must never turn a working undrafted launch into one the
binary rejects, so the capability is sniffed from the cached `--help` like
`--hf-repo` and `--mmproj-auto` already are, and an explicitly saved
`draft-dflash` on an older binary is refused by `launch_blocker`. It outranks
`draft-mtp` if a model somehow offers both; the resolved
`spec-type` (not the model) then decides which companion fills
`--spec-draft-model`, so the two drafters can coexist in one catalog.

A dFlash drafter must be downloaded before it can be used. `--spec-draft-hf`
takes only a `repo[:quant]` selector — there is no `--hf-file` equivalent for
the draft model — so an unqualified `dflash-*.gguf` cannot be addressed
remotely. llmctl includes the drafter in the blobs it downloads with the model,
and a launch that selects `draft-dflash` without a cached drafter is refused by
`launch_blocker` with that instruction rather than being started and failing
inside llama-server.

Reproducing a published multi-GPU configuration also needs options llmctl did
not expose, so `split-mode`, `tensor-split`, `parallel`, and
`sleep-idle-seconds` join the llama.cpp option table, all omitted at the
`default` sentinel.

`--spec-draft-n-max` defaults to the drafter's `dflash.block_size` (read from
the companion's GGUF header) rather than llama.cpp's 3. dFlash emits a whole
block per forward pass, so throughput keeps climbing to the block size even as
the acceptance *rate* falls — measured on a Radeon 8060S with
Muse-Glimmer-30B-Q4_K_XL: 11.2 t/s undrafted, then 16.3 / 27.4 / 28.5 t/s at
n-max 3 / 8 / 16, with acceptance 0.66 / 0.35 / 0.24 and mean accepted length
3.0 / 3.9 / 4.5. llama.cpp clamps above the block size anyway, so the default is
also the ceiling.

The deprecated `--no-mmap` switch is replaced by the `load-mode` option
(`-lm {none,mmap,mlock,mmap+mlock,dio}`), which supersedes `--mmap`, `--mlock`,
and `--direct-io` upstream. A profile saved as `mmap: off` is recovered as
`load-mode: none` through a new `RuntimeBackend::legacy_value` hook — a
key-level counterpart to `normalize_legacy`, consulted only when the profile
carries no value under the current key — so a renamed flag never silently drops
a user's setting. Binaries too old to advertise `--load-mode` are refused by
`launch_blocker` unless the option sits at its omitted default.

**Consequences:** A model whose directory mixes families gets no drafter rather
than a guessed one; that is deliberate, and a per-model companion selector
(already a noted follow-up for MTP/projector precision) would cover it. The
`Drafter` enum is the single place that maps a `spec-type` value to the
companion it needs, so `draft-eagle3`/`draft-dspark` can be paired later without
touching the command builder.
