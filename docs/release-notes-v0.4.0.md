# v0.4.0 — runtime backends and FastFlowLM (AMD NPU)

v0.4.0 makes llmctl multi-runtime. Until now "runtime" was a string llmctl
branched on, and every option table, launch command, and readiness check assumed
llama.cpp. This release replaces that with a real seam — the `RuntimeBackend`
trait — and ships **FastFlowLM** (`flm`) through it as a second runtime, running
models on the AMD XDNA2 NPU: hardware llama.cpp cannot target at all.

Most of the diff is the abstraction. FastFlowLM is the proof that it works.

## Highlights

- **`RuntimeBackend` trait (ADR-011)** — binary discovery, the option schema and
  CLI dialect, built-in templates, model enumeration, launch/chat/benchmark
  argv, the readiness path, the `/proc` identity token, and per-launch capability
  checks all moved behind one trait. The app, profile store, and session manager
  dispatch through `&dyn RuntimeBackend` and hold no per-runtime knowledge.
  Adding a runtime is now a module under `src/runtime/` plus one line in
  `runtime::discover`.
- **FastFlowLM runtime (ADR-012)** — `flm list --json` supplies a curated
  catalog, grouped by capability label (reasoning, vision, tool-calling, audio,
  embeddings, with a `chat` fallback) under the same `local`/`online` split as
  llama.cpp. Labels overlap, so a model's identity is its tag, not its position
  in the tree: a model shown under three labels has one set of profiles.
- **NPU-specific options and templates** — power mode, prefill chunk length, NPU
  queue length, socket limit, preemption, CORS, ASR/embedding companions, and
  vision pre-resize, clamped to each model's trained context and maximum
  prefill. Templates (Default, Chat, Long Context, Server, Low Power) mirror
  llama.cpp's intent in FastFlowLM's dialect.
- **Sessions, chat, and benchmarking** — `flm serve` with `/v1/models` as the
  readiness signal, `flm run` chat on `C`, and `flm bench` on `b`. Sessions work
  even when `flm` is a launcher wrapper rather than the server itself.
- **One model at a time on the NPU** — the XDNA driver grants a single hardware
  context. `RuntimeBackend::single_session` lets a runtime declare that, so a
  colliding start, chat, or benchmark is refused up front, naming the session
  holding the device, instead of dying at model load with an opaque
  `DRM_IOCTL_AMDXDNA_CREATE_HWCTX` error.
- **Resumable FastFlowLM downloads (ADR-013)** — llmctl fetches from the Hub
  itself rather than shelling out to `flm pull`, which cannot resume and reports
  an interrupted pull as installed. The Hugging Face transfer core (`Range`
  resume, cancellation, per-file size verification) is now shared with
  llama.cpp's downloader.
- **NPU readiness surfaced early** — `flm validate` gates launching, so a
  missing driver, unsupported device, or too-low memlock limit is explained in
  the status line instead of failing at load time.
- **vLLM stub removed** — it was navigable but non-functional. Real runtimes now
  arrive through the trait.

## Fixes

Several of these were pre-existing and affected llama.cpp too:

- **`Command::spawn()` could take down the TUI.** The supervisor sets `SIGCHLD`
  to `SIG_IGN` so detached servers self-reap; std reaps a failed child to read
  its errno, so launching a binary that had been moved or deleted panicked
  instead of reporting an error.
- **Restart raced the process it replaced** — it signalled and respawned
  immediately. Restart is now two-phase: a `Restarting` status, the replacement
  spawned from the poll loop once the old process has actually exited, and
  SIGKILL after five seconds. The replacement inherits a free port and a free
  device.
- **Searching a non-llama.cpp catalog panicked**, from ranking against one model
  list and resolving against another. Fixed structurally, so the mismatch is
  now unrepresentable.
- **Size column misalignment** for any sub-gigabyte model.
- **Log lines crossing the pane border**, from bare `\r`, `ESC[K`, and emoji
  whose variation selector makes terminals draw two columns where
  `unicode-width` reports one.
- **A `/health` probe every second** for already-running sessions, filling
  server logs with requests whose result was discarded.
- **Profiles attributed to llama.cpp regardless of origin** — `Model` now
  carries its runtime.

## Performance

Switching catalog arrangement (`s`) re-ran `flm list` just to regroup identical
data — 155 ms of visible hitch, because `flm` is often a wrapper script and
every call pays a container hop. The catalog is now memoized, with an explicit
`reload` flag marking the callers that genuinely need fresh data (`F5`, and
finishing a download): **155 ms → 58 µs**.

## Upgrade notes

- **No migration.** Existing models, profiles, and sessions are unaffected.
  Profiles were already scoped per runtime + model.
- **FastFlowLM is optional.** Without `flm` on your `$PATH` the runtime is still
  listed, with the reason it is unusable in the status line; llama.cpp behaves
  exactly as in v0.3.1.
- **`config.toml` gains a `[runtime.fastflowlm]` section** on new installs.
  Existing config files need no edit — the section defaults to `binary = "flm"`.
- **FastFlowLM models are not scanned.** They live under `~/.config/flm/models/`
  (or `$FLM_MODEL_PATH`), owned by `flm`; llmctl reads that catalog via
  `flm list --json`. Your configured model sources apply to llama.cpp only.

## Install

Download a prebuilt Linux binary from the GitHub release (the musl build is
fully static), or install from source with `cargo install --path .`.

## Known limitations

- **Linux only.** `setsid`, `/proc` sampling, and POSIX signals are load-bearing.
- **One FastFlowLM model at a time**, imposed by the XDNA driver's single
  hardware context — not by llmctl.
- **The FastFlowLM catalog is curated by `flm`**, not browsable Hub-wide. There
  is no equivalent of llama.cpp's `online ▸ huggingface` search for NPU models;
  you get the models FastFlowLM has converted.
- **`flm bench` is undocumented upstream** — it is absent from `flm --help`,
  though v0.9.45 runs it. If a future release removes it, `b` will fail rather
  than being hidden.
- **Downloads are tracked by byte growth**, since `flm serve` writes straight to
  final filenames with no rename to observe. A native `flm`-initiated download
  shows progress but cannot be resumed by llmctl; press `d` to let llmctl own
  the transfer instead.
- Phases 4 and 5 (log search, startup-failure classification) remain deferred.

---

**Full changelog:** https://github.com/zeddius1983/llmctl/compare/v0.3.1...v0.4.0
