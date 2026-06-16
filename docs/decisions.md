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

**Status:** Accepted (planned for Phase 3)

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
